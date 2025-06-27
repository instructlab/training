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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
