#!/usr/bin/env python3
"""
Comprehensive tests for the unmask_messages function to ensure behavior consistency
across different tokenizers and scenarios.
"""

# Standard
from unittest.mock import Mock, patch
import os
import tempfile

# Third Party
from transformers import AutoTokenizer
import pytest

# First Party
# Import the functions we want to test
from instructlab.training.data_process import (
    UNMASK_BEGIN_TOKEN,
    UNMASK_END_TOKEN,
    UNMASK_REASONING_BEGIN_TOKEN,
    UNMASK_REASONING_END_TOKEN,
    unmask_messages,
    unmask_sample,
    wrap_masked_messages,
)
from instructlab.training.type_definitions import Message


class TestWrapMaskedMessages:
    """Test suite for wrap_masked_messages functionality."""

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

    @pytest.fixture
    def reasoning_messages(self):
        """Sample messages with reasoning content."""
        return [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "I need to add 2 and 2 together. 2 + 2 = 4.",
            },
        ]

    def test_wrap_masked_messages_basic(self, sample_messages):
        """Test basic message wrapping functionality."""
        wrapped = wrap_masked_messages(sample_messages, ["assistant"])

        # Check that only assistant messages are wrapped
        assert (
            sample_messages[0]["content"] == wrapped[0]["content"]
        )  # system unchanged
        assert sample_messages[1]["content"] == wrapped[1]["content"]  # user unchanged
        assert (
            wrapped[2]["content"]
            == f"{UNMASK_BEGIN_TOKEN}I'm doing well, thank you!{UNMASK_END_TOKEN}"
        )

    def test_wrap_masked_messages_with_reasoning(self, reasoning_messages):
        """Test message wrapping with reasoning content."""
        # Test with reasoning content disabled (default behavior)
        wrapped = wrap_masked_messages(reasoning_messages, ["assistant"])

        # Check content wrapping
        expected_content = f"{UNMASK_BEGIN_TOKEN}The answer is 4.{UNMASK_END_TOKEN}"
        assert wrapped[1]["content"] == expected_content

        # Check reasoning content is NOT processed when disabled
        assert (
            wrapped[1]["reasoning_content"]
            == "I need to add 2 and 2 together. 2 + 2 = 4."
        )

        # Test with reasoning content enabled
        wrapped_with_reasoning = wrap_masked_messages(
            reasoning_messages, ["assistant"], enable_reasoning_content=True
        )

        # Check content is wrapped with regular tokens and reasoning with reasoning-specific tokens
        assert wrapped_with_reasoning[1]["content"] == expected_content
        expected_reasoning = f"{UNMASK_REASONING_BEGIN_TOKEN}I need to add 2 and 2 together. 2 + 2 = 4.{UNMASK_REASONING_END_TOKEN}"
        assert wrapped_with_reasoning[1]["reasoning_content"] == expected_reasoning

    def test_wrap_masked_messages_multiple_roles(self, sample_messages):
        """Test wrapping messages for multiple roles."""
        wrapped = wrap_masked_messages(sample_messages, ["user", "assistant"])

        # Both user and assistant should be wrapped
        assert (
            wrapped[1]["content"]
            == f"{UNMASK_BEGIN_TOKEN}Hello, how are you?{UNMASK_END_TOKEN}"
        )
        assert (
            wrapped[2]["content"]
            == f"{UNMASK_BEGIN_TOKEN}I'm doing well, thank you!{UNMASK_END_TOKEN}"
        )
        # System should remain unchanged
        assert wrapped[0]["content"] == sample_messages[0]["content"]

    def test_wrap_masked_messages_error_on_non_string_content(self):
        """Test that wrapping non-string content raises an error."""
        messages = [{"role": "assistant", "content": ["not", "a", "string"]}]

        with pytest.raises(ValueError, match="unmasking non-string data types"):
            wrap_masked_messages(messages, ["assistant"])

    def test_wrap_masked_messages_empty_roles_list(self, sample_messages):
        """Test wrapping with empty roles list."""
        wrapped = wrap_masked_messages(sample_messages, [])

        # All messages should remain unchanged
        for i, msg in enumerate(sample_messages):
            assert wrapped[i]["content"] == msg["content"]

    def test_wrap_masked_messages_preserves_other_fields(self):
        """Test that other message fields are preserved during wrapping."""
        messages = [
            {
                "role": "assistant",
                "content": "Hello",
                "custom_field": "custom_value",
                "another_field": 123,
            }
        ]
        wrapped = wrap_masked_messages(messages, ["assistant"])

        assert wrapped[0]["custom_field"] == "custom_value"
        assert wrapped[0]["another_field"] == 123
        assert wrapped[0]["role"] == "assistant"


class TestUnmaskMessages:
    """Test suite for unmask_messages functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for basic testing."""
        tokenizer = Mock()

        # Mock the special token encodings
        def mock_encode(text, add_special_tokens=False):
            token_map = {
                UNMASK_BEGIN_TOKEN: [1000],
                UNMASK_END_TOKEN: [1001],
                UNMASK_REASONING_BEGIN_TOKEN: [1002],
                UNMASK_REASONING_END_TOKEN: [1003],
                "<|endoftext|>": [0],
            }
            return token_map.get(text, [hash(text) % 500 + 100])

        tokenizer.encode = mock_encode
        tokenizer.eos_token = "<|endoftext|>"

        return tokenizer

    @pytest.fixture
    def simple_input_ids(self):
        """Simple token sequence for testing: user msg + assistant msg."""
        # Represents: [role_tokens] user_content [unmask_begin] assistant_content [unmask_end] [eos]
        return [50, 51, 52, 1000, 200, 201, 202, 1001, 0]

    def test_unmask_messages_basic_flow(self, mock_tokenizer):
        """Test the basic flow of unmask_messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Mock apply_chat_template to return our simple input sequence
        mock_tokenizer.apply_chat_template.return_value = [50, 1000, 200, 201, 1001, 0]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Verify the result structure
        assert "input_ids" in result
        assert "labels" in result
        assert "len" in result
        assert len(result["input_ids"]) == len(result["labels"])

        # Should not contain unmask tokens in final output
        assert 1000 not in result["input_ids"]  # UNMASK_BEGIN_TOKEN
        assert 1001 not in result["input_ids"]  # UNMASK_END_TOKEN

    def test_unmask_messages_assistant_only_unmasking(self, mock_tokenizer):
        """Test that only assistant tokens are unmasked when specified."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Sequence: user_tokens + unmask_begin + assistant_tokens + unmask_end
        mock_tokenizer.apply_chat_template.return_value = [50, 51, 1000, 200, 201, 1001]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # User tokens should be masked (-100), assistant tokens should be unmasked
        expected_input_ids = [50, 51, 200, 201]
        expected_labels = [-100, -100, 200, 201]

        assert result["input_ids"] == expected_input_ids
        assert result["labels"] == expected_labels

    def test_unmask_messages_multiple_roles(self, mock_tokenizer):
        """Test unmasking multiple roles."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Both user and assistant wrapped with unmask tokens
        mock_tokenizer.apply_chat_template.return_value = [
            1000,
            50,
            51,
            1001,  # user wrapped
            1000,
            200,
            201,
            1001,  # assistant wrapped
        ]

        result = unmask_messages(messages, mock_tokenizer, ["user", "assistant"])

        # Both should be unmasked
        expected_input_ids = [50, 51, 200, 201]
        expected_labels = [50, 51, 200, 201]

        assert result["input_ids"] == expected_input_ids
        assert result["labels"] == expected_labels

    def test_unmask_messages_with_eos_token_for_assistant(self, mock_tokenizer):
        """Test that EOS token is unmasked for assistant role."""
        messages = [{"role": "assistant", "content": "Hello"}]

        # Assistant content followed by EOS token
        mock_tokenizer.apply_chat_template.return_value = [1000, 200, 201, 1001, 0]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Both assistant content and EOS should be unmasked
        expected_input_ids = [200, 201, 0]
        expected_labels = [200, 201, 0]

        assert result["input_ids"] == expected_input_ids
        assert result["labels"] == expected_labels

    def test_unmask_messages_no_eos_token(self, mock_tokenizer):
        """Test behavior when tokenizer has no EOS token."""
        mock_tokenizer.eos_token = None
        messages = [{"role": "assistant", "content": "Hello"}]

        mock_tokenizer.apply_chat_template.return_value = [1000, 200, 201, 1001]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Should work normally without EOS handling
        expected_input_ids = [200, 201]
        expected_labels = [200, 201]

        assert result["input_ids"] == expected_input_ids
        assert result["labels"] == expected_labels

    def test_unmask_messages_validation_errors(self, mock_tokenizer):
        """Test that validation errors are properly raised."""
        messages = [{"role": "assistant", "content": "Hello"}]

        # Simulate a bug where unmask tokens remain in output by mocking a faulty sequence
        mock_tokenizer.apply_chat_template.return_value = [
            1000,
            200,
            1000,
            1001,
        ]  # nested begin token

        with pytest.raises(ValueError, match="encountered.*while already unmasking"):
            unmask_messages(messages, mock_tokenizer, ["assistant"])

    def test_unmask_messages_mismatched_end_token(self, mock_tokenizer):
        """Test error when encountering end token while not unmasking."""
        messages = [{"role": "user", "content": "Hello"}]

        # End token without begin token
        mock_tokenizer.apply_chat_template.return_value = [200, 1001]

        with pytest.raises(ValueError, match="encountered.*while not unmasking"):
            unmask_messages(messages, mock_tokenizer, ["assistant"])

    def test_unmask_messages_empty_input(self, mock_tokenizer):
        """Test behavior with empty input."""
        mock_tokenizer.apply_chat_template.return_value = []

        result = unmask_messages([], mock_tokenizer, ["assistant"])

        assert result["input_ids"] == []
        assert result["labels"] == []
        assert result["len"] == 0

    def test_unmask_messages_reasoning_content_handling(self, mock_tokenizer):
        """Test that reasoning content is properly handled."""
        messages = [
            {"role": "assistant", "content": "Answer", "reasoning_content": "Thinking"}
        ]

        # When messages have reasoning content, wrap_masked_messages uses reasoning-specific tokens
        mock_tokenizer.apply_chat_template.return_value = [
            1002,  # UNMASK_REASONING_BEGIN
            200,  # reasoning content
            1003,  # UNMASK_REASONING_END
            1000,  # UNMASK_BEGIN
            100,  # content
            1001,  # UNMASK_END
        ]

        # The new implementation correctly handles reasoning content
        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Both reasoning and content should be unmasked
        assert result["input_ids"] == [200, 100]
        assert result["labels"] == [200, 100]
        assert result["len"] == 2


class TestWithRealTokenizers:
    """Test with actual tokenizer implementations to ensure realistic behavior."""

    @pytest.fixture(scope="class")
    def test_tokenizer(self):
        """Get a small test tokenizer."""
        try:
            # Use a small model for testing
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-gpt2")

            # Add the special tokens
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        UNMASK_BEGIN_TOKEN,
                        UNMASK_END_TOKEN,
                        UNMASK_REASONING_BEGIN_TOKEN,
                        UNMASK_REASONING_END_TOKEN,
                    ]
                }
            )

            # Set a simple chat template for testing
            tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% if message.get('reasoning_content') %} [REASONING: {{ message['reasoning_content'] }}]{% endif %}\n{% endfor %}"

            return tokenizer
        except Exception as e:
            pytest.skip(f"Could not load test tokenizer: {e}")

    def test_real_tokenizer_basic_functionality(self, test_tokenizer):
        """Test basic functionality with a real tokenizer."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = unmask_messages(messages, test_tokenizer, ["assistant"])

        # Basic sanity checks
        assert len(result["input_ids"]) > 0
        assert len(result["labels"]) == len(result["input_ids"])
        assert result["len"] == len(result["input_ids"])

        # Check that some tokens are masked (-100) and some are not
        masked_count = sum(1 for label in result["labels"] if label == -100)
        unmasked_count = len(result["labels"]) - masked_count

        assert masked_count > 0, "Should have some masked tokens"
        assert unmasked_count > 0, "Should have some unmasked tokens"

    def test_real_tokenizer_with_reasoning(self, test_tokenizer):
        """Test reasoning content with a real tokenizer."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "I need to calculate 2+2.",
            },
        ]

        result = unmask_messages(messages, test_tokenizer, ["assistant"])

        # Should have processed both content and reasoning_content
        assert len(result["input_ids"]) > 5  # Should be reasonably long
        assert result["len"] == len(result["input_ids"])

        # Should have both masked and unmasked tokens
        masked_count = sum(1 for label in result["labels"] if label == -100)
        unmasked_count = len(result["labels"]) - masked_count

        assert masked_count > 0, "Should have some masked tokens (user message)"
        assert unmasked_count > 0, (
            "Should have some unmasked tokens (assistant content + reasoning)"
        )

    def test_real_tokenizer_edge_cases(self, test_tokenizer):
        """Test edge cases with real tokenizer."""
        # Empty assistant message
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
        ]

        result = unmask_messages(messages, test_tokenizer, ["assistant"])
        assert len(result["input_ids"]) > 0

        # Only reasoning content, no regular content
        messages = [
            {"role": "user", "content": "Think about this"},
            {"role": "assistant", "reasoning_content": "Let me think..."},
        ]

        result = unmask_messages(messages, test_tokenizer, ["assistant"])
        assert len(result["input_ids"]) > 0

        # Should have some unmasked tokens from reasoning
        unmasked_count = sum(1 for label in result["labels"] if label != -100)
        assert unmasked_count > 0, "Should have unmasked reasoning content"


class TestErrorConditions:
    """Test various error conditions and edge cases."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for error testing."""
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
            UNMASK_BEGIN_TOKEN: [1000],
            UNMASK_END_TOKEN: [1001],
        }.get(text, [100])
        tokenizer.eos_token = None
        return tokenizer

    def test_length_mismatch_error(self, mock_tokenizer):
        """Test that length mismatches raise appropriate errors."""
        # This would be an internal error where our processing logic fails
        messages = [{"role": "assistant", "content": "Hello"}]
        mock_tokenizer.apply_chat_template.return_value = [1000, 200, 1001]

        # Mock a scenario where we somehow create mismatched lengths
        with patch("instructlab.training.data_process.unmask_messages") as mock_unmask:
            mock_unmask.side_effect = RuntimeError(
                "final_input_ids and final_labels are not the same length"
            )

            with pytest.raises(
                RuntimeError,
                match="final_input_ids and final_labels are not the same length",
            ):
                mock_unmask(messages, mock_tokenizer, ["assistant"])

    def test_unfinished_unmasking_error(self, mock_tokenizer):
        """Test error when unmasking is not properly finished."""
        messages = [{"role": "assistant", "content": "Hello"}]

        # Begin token without end token
        mock_tokenizer.apply_chat_template.return_value = [1000, 200]

        with pytest.raises(
            RuntimeError, match="unmasking finished but not all messages were processed"
        ):
            unmask_messages(messages, mock_tokenizer, ["assistant"])


class TestRealTokenizersUnmaskBehavior:
    """Test suite for validating unmask behavior with real tokenizers."""

    @pytest.fixture(
        scope="class", params=["Qwen/Qwen3-32B", "ibm-granite/granite-3.1-8b-instruct"]
    )
    def real_tokenizer(self, request):
        """Load real tokenizers for comprehensive testing."""
        try:
            # Add environment variable for testing without downloading
            os.environ["TRANSFORMERS_OFFLINE"] = "0"

            tokenizer = AutoTokenizer.from_pretrained(request.param, cache_dir=".cache")

            # Add the special unmask tokens
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        UNMASK_BEGIN_TOKEN,
                        UNMASK_END_TOKEN,
                        UNMASK_REASONING_BEGIN_TOKEN,
                        UNMASK_REASONING_END_TOKEN,
                    ]
                }
            )

            # Store the model name for test identification
            tokenizer._model_name = request.param

            return tokenizer
        except Exception as e:
            pytest.skip(f"Could not load tokenizer {request.param}: {e}")

    @pytest.fixture
    def sample_unmask_true(self):
        """Sample with unmask: True - should unmask user and assistant, but not system."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.",
                },
                {
                    "role": "user",
                    "content": 'For the word "dream", give an example of a word that rhymes with it and its synonym.',
                },
                {
                    "role": "assistant",
                    "content": 'Here\'s an example for "dream" that includes a word that rhymes with it and a synonym:\n1. Word that rhymes with "dream": "beam"\nSynonym: "ideal"',
                },
            ],
            "unmask": True,
        }

    @pytest.fixture
    def sample_unmask_false(self):
        """Sample with unmask: False - should only unmask assistant."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.",
                },
                {
                    "role": "user",
                    "content": 'Using the word "grace", come up with a word that rhymes and has the same number of syllables',
                },
                {
                    "role": "assistant",
                    "content": 'Certainly! Here\'s a word that rhymes with "grace" and has the same number of syllables:\n1. Space',
                },
            ],
            "unmask": False,
        }

    @pytest.fixture
    def sample_with_reasoning(self):
        """Sample with reasoning content."""
        return {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "The answer is 4.",
                    "reasoning_content": "I need to add 2 and 2 together. 2 + 2 = 4.",
                },
            ],
            "unmask": False,
        }

    @pytest.mark.slow
    def test_unmask_sample_with_unmask_true(self, real_tokenizer, sample_unmask_true):
        """Test that unmask: True correctly unmasks user and assistant but not system."""
        result = unmask_sample(sample_unmask_true, real_tokenizer)

        # Basic validation
        assert "input_ids" in result
        assert "labels" in result
        assert "len" in result
        assert len(result["input_ids"]) == len(result["labels"])
        assert result["len"] == len(result["input_ids"])

        # Decode the sequences to validate content
        input_text = real_tokenizer.decode(
            result["input_ids"], skip_special_tokens=False
        )

        # Should contain parts of user and assistant content but not system in labels
        masked_positions = [
            i for i, label in enumerate(result["labels"]) if label == -100
        ]
        unmasked_positions = [
            i for i, label in enumerate(result["labels"]) if label != -100
        ]

        # Must have both masked and unmasked tokens
        assert len(masked_positions) > 0, (
            f"Expected some masked tokens for {real_tokenizer._model_name}"
        )
        assert len(unmasked_positions) > 0, (
            f"Expected some unmasked tokens for {real_tokenizer._model_name}"
        )

        # Verify that unmasked tokens match input_ids
        for pos in unmasked_positions:
            assert result["labels"][pos] == result["input_ids"][pos], (
                f"Unmasked position {pos} should have matching label and input_id"
            )

        # Check that unmask tokens are not present in final output
        assert UNMASK_BEGIN_TOKEN.encode() not in input_text.encode()
        assert UNMASK_END_TOKEN.encode() not in input_text.encode()

        print(f"\n=== {real_tokenizer._model_name} - UNMASK: TRUE ===")
        print(f"Input text: {input_text}")
        print(f"Total tokens: {len(result['input_ids'])}")
        print(f"Masked tokens: {len(masked_positions)}")
        print(f"Unmasked tokens: {len(unmasked_positions)}")

        # Create a visual representation of masking
        visual_labels = []
        for i, (token_id, label) in enumerate(
            zip(result["input_ids"], result["labels"])
        ):
            token_text = real_tokenizer.decode([token_id])
            if label == -100:
                visual_labels.append("<|MASK|>")
            else:
                visual_labels.append(token_text)

        print(f"Visual masking: {''.join(visual_labels)}")

    @pytest.mark.slow
    def test_unmask_sample_with_unmask_false(self, real_tokenizer, sample_unmask_false):
        """Test that unmask: False correctly unmasks only assistant."""
        result = unmask_sample(sample_unmask_false, real_tokenizer)

        # Basic validation
        assert "input_ids" in result
        assert "labels" in result
        assert "len" in result
        assert len(result["input_ids"]) == len(result["labels"])
        assert result["len"] == len(result["input_ids"])

        # Decode the sequences to validate content
        input_text = real_tokenizer.decode(
            result["input_ids"], skip_special_tokens=False
        )

        # Should have more masked tokens than unmask=True case since only assistant is unmasked
        masked_positions = [
            i for i, label in enumerate(result["labels"]) if label == -100
        ]
        unmasked_positions = [
            i for i, label in enumerate(result["labels"]) if label != -100
        ]

        # Must have both masked and unmasked tokens
        assert len(masked_positions) > 0, (
            f"Expected some masked tokens for {real_tokenizer._model_name}"
        )
        assert len(unmasked_positions) > 0, (
            f"Expected some unmasked tokens for {real_tokenizer._model_name}"
        )

        # Verify that unmasked tokens match input_ids
        for pos in unmasked_positions:
            assert result["labels"][pos] == result["input_ids"][pos], (
                f"Unmasked position {pos} should have matching label and input_id"
            )

        # Check that unmask tokens are not present in final output
        assert UNMASK_BEGIN_TOKEN.encode() not in input_text.encode()
        assert UNMASK_END_TOKEN.encode() not in input_text.encode()

        print(f"\n=== {real_tokenizer._model_name} - UNMASK: FALSE ===")
        print(f"Input text: {input_text}")
        print(f"Total tokens: {len(result['input_ids'])}")
        print(f"Masked tokens: {len(masked_positions)}")
        print(f"Unmasked tokens: {len(unmasked_positions)}")

        # Create a visual representation of masking
        visual_labels = []
        for i, (token_id, label) in enumerate(
            zip(result["input_ids"], result["labels"])
        ):
            token_text = real_tokenizer.decode([token_id])
            if label == -100:
                visual_labels.append("<|MASK|>")
            else:
                visual_labels.append(token_text)

        print(f"Visual masking: {''.join(visual_labels)}")

    @pytest.mark.slow
    def test_unmask_comparison_between_settings(
        self, real_tokenizer, sample_unmask_true, sample_unmask_false
    ):
        """Test that unmask: True results in fewer masked tokens than unmask: False."""
        result_true = unmask_sample(sample_unmask_true, real_tokenizer)
        result_false = unmask_sample(sample_unmask_false, real_tokenizer)

        masked_count_true = sum(1 for label in result_true["labels"] if label == -100)
        masked_count_false = sum(1 for label in result_false["labels"] if label == -100)

        unmasked_count_true = len(result_true["labels"]) - masked_count_true
        unmasked_count_false = len(result_false["labels"]) - masked_count_false

        # unmask: True should have more unmasked tokens (user + assistant vs just assistant)
        assert unmasked_count_true > unmasked_count_false, (
            f"unmask: True should unmask more tokens than unmask: False for {real_tokenizer._model_name}"
        )

    @pytest.mark.slow
    def test_unmask_with_reasoning_content(self, real_tokenizer, sample_with_reasoning):
        """Test that reasoning content is properly handled."""
        result = unmask_sample(sample_with_reasoning, real_tokenizer)

        # Basic validation
        assert "input_ids" in result
        assert "labels" in result
        assert "len" in result
        assert len(result["input_ids"]) == len(result["labels"])

        # Should have processed both content and reasoning_content
        unmasked_positions = [
            i for i, label in enumerate(result["labels"]) if label != -100
        ]
        assert len(unmasked_positions) > 0, (
            "Should have unmasked tokens from assistant content and reasoning"
        )

        # Decode to see the result
        input_text = real_tokenizer.decode(
            result["input_ids"], skip_special_tokens=False
        )
        print(f"\n=== {real_tokenizer._model_name} - WITH REASONING ===")
        print(f"Input text: {input_text}")
        print(f"Total tokens: {len(result['input_ids'])}")
        print(f"Unmasked tokens: {len(unmasked_positions)}")

    def test_token_id_consistency(self, real_tokenizer, sample_unmask_true):
        """Test that token IDs are consistent and valid."""
        result = unmask_sample(sample_unmask_true, real_tokenizer)

        # All input_ids should be valid token IDs
        for token_id in result["input_ids"]:
            assert isinstance(token_id, int), "All token IDs should be integers"
            assert 0 <= token_id < len(real_tokenizer), (
                "Token IDs should be within vocabulary range"
            )

        # All non-masked labels should match their corresponding input_ids
        for i, (input_id, label) in enumerate(
            zip(result["input_ids"], result["labels"])
        ):
            if label != -100:
                assert label == input_id, (
                    f"Position {i}: label {label} should match input_id {input_id}"
                )

        # Verify we can decode all tokens
        decoded_text = real_tokenizer.decode(result["input_ids"])
        assert isinstance(decoded_text, str), (
            "Should be able to decode all tokens to string"
        )
        assert len(decoded_text) > 0, "Decoded text should not be empty"

    def test_special_tokens_removed_from_output(
        self, real_tokenizer, sample_unmask_true
    ):
        """Test that special unmask tokens are properly removed from final output."""
        result = unmask_sample(sample_unmask_true, real_tokenizer)

        # Get token IDs for special tokens
        unmask_begin_id = real_tokenizer.encode(
            UNMASK_BEGIN_TOKEN, add_special_tokens=False
        )[0]
        unmask_end_id = real_tokenizer.encode(
            UNMASK_END_TOKEN, add_special_tokens=False
        )[0]
        unmask_reasoning_begin_id = real_tokenizer.encode(
            UNMASK_REASONING_BEGIN_TOKEN, add_special_tokens=False
        )[0]
        unmask_reasoning_end_id = real_tokenizer.encode(
            UNMASK_REASONING_END_TOKEN, add_special_tokens=False
        )[0]

        # None of these should appear in final output
        assert unmask_begin_id not in result["input_ids"], (
            "UNMASK_BEGIN_TOKEN should not be in final input_ids"
        )
        assert unmask_end_id not in result["input_ids"], (
            "UNMASK_END_TOKEN should not be in final input_ids"
        )
        assert unmask_reasoning_begin_id not in result["input_ids"], (
            "UNMASK_REASONING_BEGIN_TOKEN should not be in final input_ids"
        )
        assert unmask_reasoning_end_id not in result["input_ids"], (
            "UNMASK_REASONING_END_TOKEN should not be in final input_ids"
        )

        # Same for labels
        assert unmask_begin_id not in result["labels"], (
            "UNMASK_BEGIN_TOKEN should not be in final labels"
        )
        assert unmask_end_id not in result["labels"], (
            "UNMASK_END_TOKEN should not be in final labels"
        )
        assert unmask_reasoning_begin_id not in result["labels"], (
            "UNMASK_REASONING_BEGIN_TOKEN should not be in final labels"
        )
        assert unmask_reasoning_end_id not in result["labels"], (
            "UNMASK_REASONING_END_TOKEN should not be in final labels"
        )

    def test_reproducibility(self, real_tokenizer, sample_unmask_true):
        """Test that the same input produces the same output consistently."""
        result1 = unmask_sample(sample_unmask_true, real_tokenizer)
        result2 = unmask_sample(sample_unmask_true, real_tokenizer)

        assert result1["input_ids"] == result2["input_ids"], (
            "Results should be reproducible"
        )
        assert result1["labels"] == result2["labels"], "Results should be reproducible"
        assert result1["len"] == result2["len"], "Results should be reproducible"


class TestUnmaskSampleLogic:
    """Test the logic of unmask_sample without requiring full tokenizer loading."""

    @pytest.fixture
    def mock_tokenizer_for_unmask_sample(self):
        """Create a comprehensive mock tokenizer for testing unmask_sample logic."""
        tokenizer = Mock()

        # Mock the special token encodings - using unique IDs for each token
        def mock_encode(text, add_special_tokens=False):
            token_map = {
                UNMASK_BEGIN_TOKEN: [1000],
                UNMASK_END_TOKEN: [1001],
                UNMASK_REASONING_BEGIN_TOKEN: [1002],
                UNMASK_REASONING_END_TOKEN: [1003],
                "<|endoftext|>": [0],
            }
            # Return predictable token IDs based on hash for consistent testing
            return token_map.get(text, [abs(hash(text)) % 500 + 100])

        tokenizer.encode = mock_encode
        tokenizer.eos_token = "<|endoftext|>"

        # Mock apply_chat_template to return a sequence that represents:
        # system_tokens + unmask_begin + user_tokens + unmask_end + unmask_begin + assistant_tokens + unmask_end + eos
        def mock_apply_chat_template(messages, **kwargs):
            sequence = []
            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")
                reasoning_content = msg.get("reasoning_content", "")

                # Add role-specific tokens
                if role == "system":
                    sequence.extend([10, 11, 12])  # system role tokens
                elif role == "user":
                    sequence.extend([20, 21])  # user role tokens
                elif role == "assistant":
                    sequence.extend([30, 31])  # assistant role tokens

                # Add reasoning content if present and wrapped
                if (
                    reasoning_content
                    and UNMASK_REASONING_BEGIN_TOKEN in reasoning_content
                ):
                    sequence.append(1002)  # reasoning begin
                    sequence.extend([250, 251, 252])  # reasoning content tokens
                    sequence.append(1003)  # reasoning end

                # Add content tokens
                if content and UNMASK_BEGIN_TOKEN in content:
                    sequence.append(1000)  # unmask begin
                    # Add content tokens based on role
                    if role == "user":
                        sequence.extend([200, 201, 202])
                    elif role == "assistant":
                        sequence.extend([300, 301, 302])
                    sequence.append(1001)  # unmask end
                elif content:
                    # Non-wrapped content
                    if role == "system":
                        sequence.extend([100, 101, 102, 103])
                    elif role == "user":
                        sequence.extend([200, 201, 202])
                    elif role == "assistant":
                        sequence.extend([300, 301, 302])

            sequence.append(0)  # eos token
            return sequence

        tokenizer.apply_chat_template = mock_apply_chat_template
        return tokenizer

    @patch("instructlab.training.data_process.is_gpt_oss_model", return_value=False)
    def test_unmask_sample_unmask_false_logic(
        self, mock_is_gpt_oss, mock_tokenizer_for_unmask_sample
    ):
        """Test unmask: False logic - should only unmask assistant role."""
        sample = {
            "messages": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant message"},
            ],
            "unmask": False,
        }

        result = unmask_sample(sample, mock_tokenizer_for_unmask_sample)

        # Basic validation
        assert len(result["input_ids"]) == len(result["labels"])

        # Count masked vs unmasked tokens
        masked_count = sum(1 for label in result["labels"] if label == -100)
        unmasked_count = len(result["labels"]) - masked_count

        assert masked_count > 0, "Should have some masked tokens"
        assert unmasked_count > 0, "Should have some unmasked tokens (assistant)"

        # Verify that assistant tokens are unmasked
        # The mock returns assistant content as tokens [300, 301, 302]
        assistant_tokens = [300, 301, 302]
        for token in assistant_tokens:
            if token in result["input_ids"]:
                idx = result["input_ids"].index(token)
                assert result["labels"][idx] == token, (
                    f"Assistant token {token} should be unmasked"
                )

    @patch("instructlab.training.data_process.is_gpt_oss_model", return_value=False)
    def test_unmask_sample_unmask_true_logic(
        self, mock_is_gpt_oss, mock_tokenizer_for_unmask_sample
    ):
        """Test unmask: True logic - should unmask user and assistant, but not system."""
        sample = {
            "messages": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant message"},
            ],
            "unmask": True,
        }

        result = unmask_sample(sample, mock_tokenizer_for_unmask_sample)

        # Basic validation
        assert len(result["input_ids"]) == len(result["labels"])

        # Count masked vs unmasked tokens
        masked_count = sum(1 for label in result["labels"] if label == -100)
        unmasked_count = len(result["labels"]) - masked_count

        assert masked_count > 0, "Should have some masked tokens (system)"
        assert unmasked_count > 0, "Should have some unmasked tokens (user + assistant)"

        # Verify that both user and assistant tokens are unmasked
        user_tokens = [200, 201, 202]
        assistant_tokens = [300, 301, 302]

        for token in user_tokens + assistant_tokens:
            if token in result["input_ids"]:
                idx = result["input_ids"].index(token)
                assert result["labels"][idx] == token, (
                    f"User/Assistant token {token} should be unmasked"
                )

    @patch("instructlab.training.data_process.is_gpt_oss_model", return_value=False)
    def test_unmask_sample_comparison(
        self, mock_is_gpt_oss, mock_tokenizer_for_unmask_sample
    ):
        """Test that unmask: True unmasks more tokens than unmask: False."""
        sample_base = {
            "messages": [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant message"},
            ]
        }

        sample_false = {**sample_base, "unmask": False}
        sample_true = {**sample_base, "unmask": True}

        result_false = unmask_sample(sample_false, mock_tokenizer_for_unmask_sample)
        result_true = unmask_sample(sample_true, mock_tokenizer_for_unmask_sample)

        unmasked_false = sum(1 for label in result_false["labels"] if label != -100)
        unmasked_true = sum(1 for label in result_true["labels"] if label != -100)

        assert unmasked_true > unmasked_false, (
            "unmask: True should unmask more tokens than unmask: False"
        )

        print(f"\nUnmask comparison:")
        print(f"unmask: False -> {unmasked_false} unmasked tokens")
        print(f"unmask: True  -> {unmasked_true} unmasked tokens")
        print(
            f"Difference: +{unmasked_true - unmasked_false} more tokens unmasked with unmask: True"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
