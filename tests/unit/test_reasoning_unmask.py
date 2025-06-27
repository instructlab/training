#!/usr/bin/env python3
"""
Comprehensive tests for reasoning content unmasking functionality.
Tests the merging of reasoning and content unmask regions as described
in the InstructLab training documentation.
"""

# Standard
from unittest.mock import Mock

# Third Party
from transformers import AutoTokenizer
import pytest

# First Party
from instructlab.training.data_process import (
    UNMASK_BEGIN_TOKEN,
    UNMASK_END_TOKEN,
    UNMASK_REASONING_BEGIN_TOKEN,
    UNMASK_REASONING_END_TOKEN,
    unmask_messages,
    wrap_masked_messages,
)


class TestReasoningContentUnmasking:
    """Test suite for reasoning content unmasking with region merging."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer with all necessary tokens."""
        tokenizer = Mock()

        # Mock token encoding
        def mock_encode(text, add_special_tokens=False):
            token_map = {
                UNMASK_BEGIN_TOKEN: [1000],
                UNMASK_END_TOKEN: [1001],
                UNMASK_REASONING_BEGIN_TOKEN: [1002],
                UNMASK_REASONING_END_TOKEN: [1003],
                "<|endoftext|>": [50256],
                "<|im_end|>": [50257],
                "<|im_start|>": [50258],
                "<think>": [50259],
                "</think>": [50260],
                "\n\n": [50261],
            }
            # For unknown text, return a hash-based token
            return token_map.get(text, [hash(text) % 10000 + 100])

        tokenizer.encode = mock_encode
        tokenizer.eos_token = "<|im_end|>"

        return tokenizer

    def test_reasoning_content_merging_basic(self, mock_tokenizer):
        """Test basic merging of reasoning and content regions."""
        messages = [
            {"role": "user", "content": "Where is Paris?"},
            {
                "role": "assistant",
                "content": "Paris is in Europe",
                "reasoning_content": "Paris is the capital of France, France is in Europe",
            },
        ]

        # Simulate Qwen/DeepSeek style template output:
        # <|im_start|>user\nWhere is Paris?<|im_end|>
        # <|im_start|>assistant\n<think>\n[REASONING]\n</think>\n\n[CONTENT]<|im_end|>
        mock_tokenizer.apply_chat_template.return_value = [
            50258,
            100,
            101,
            50257,  # user message
            50258,
            200,  # <|im_start|>assistant
            50259,  # <think>
            1002,
            300,
            301,
            1003,  # wrapped reasoning content
            50260,  # </think>
            50261,  # \n\n
            1000,
            400,
            401,
            402,
            1001,  # wrapped content
            50257,  # <|im_end|>
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # The unmask regions should be merged, unmasking everything from reasoning to content
        # including the </think> and \n\n tokens
        expected_unmasked_tokens = [300, 301, 50260, 50261, 400, 401, 402, 50257]

        # Verify that all expected tokens are unmasked
        for tok in expected_unmasked_tokens:
            if tok in result["input_ids"]:
                # For tokens that appear multiple times, check if at least one is unmasked
                indices = [i for i, t in enumerate(result["input_ids"]) if t == tok]
                assert any(result["labels"][i] == tok for i in indices), (
                    f"Token {tok} not unmasked"
                )

    def test_reasoning_content_no_merging_when_different_messages(self, mock_tokenizer):
        """Test that regions are not merged when they belong to different messages."""
        messages = [
            {
                "role": "assistant",
                "content": "Answer1",
                "reasoning_content": "Thinking1",
            },
            {
                "role": "assistant",
                "content": "Answer2",
                "reasoning_content": "Thinking2",
            },
        ]

        # Simulate regions from two different assistant messages
        mock_tokenizer.apply_chat_template.return_value = [
            # First message
            1002,
            100,
            1003,  # reasoning region for message 1
            1000,
            200,
            1001,  # content region for message 1
            50257,  # EOS
            # Second message
            1002,
            300,
            1003,  # reasoning region for message 2
            1000,
            400,
            1001,  # content region for message 2
            50257,  # EOS
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # All regions should be unmasked, but first message's regions should be separate from second's
        assert 100 in result["input_ids"]  # reasoning 1
        assert 200 in result["input_ids"]  # content 1
        assert 300 in result["input_ids"]  # reasoning 2
        assert 400 in result["input_ids"]  # content 2

        # Both messages' content should be unmasked
        for tok in [100, 200, 300, 400]:
            idx = result["input_ids"].index(tok)
            assert result["labels"][idx] == tok

    def test_multiple_assistant_messages_with_reasoning(self, mock_tokenizer):
        """Test handling of multiple assistant messages with reasoning content."""
        messages = [
            {"role": "user", "content": "Question 1"},
            {
                "role": "assistant",
                "content": "Answer 1",
                "reasoning_content": "Thinking 1",
            },
            {"role": "user", "content": "Question 2"},
            {
                "role": "assistant",
                "content": "Answer 2",
                "reasoning_content": "Thinking 2",
            },
        ]

        # Simulate chat template output with two assistant responses
        mock_tokenizer.apply_chat_template.return_value = [
            # First exchange
            50258,
            100,
            50257,  # user 1
            50258,
            200,  # assistant start
            50259,  # <think>
            1002,
            300,
            1003,  # reasoning 1
            50260,
            50261,  # </think>\n\n
            1000,
            400,
            1001,  # content 1
            50257,  # <|im_end|>
            # Second exchange
            50258,
            500,
            50257,  # user 2
            50258,
            600,  # assistant start
            50259,  # <think>
            1002,
            700,
            1003,  # reasoning 2
            50260,
            50261,  # </think>\n\n
            1000,
            800,
            1001,  # content 2
            50257,  # <|im_end|>
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Both assistant messages should have their reasoning and content unmasked
        # Check first assistant response
        assert 300 in [
            t for i, t in enumerate(result["input_ids"]) if result["labels"][i] != -100
        ]
        assert 400 in [
            t for i, t in enumerate(result["input_ids"]) if result["labels"][i] != -100
        ]

        # Check second assistant response
        assert 700 in [
            t for i, t in enumerate(result["input_ids"]) if result["labels"][i] != -100
        ]
        assert 800 in [
            t for i, t in enumerate(result["input_ids"]) if result["labels"][i] != -100
        ]

    def test_reasoning_without_content(self, mock_tokenizer):
        """Test messages that only have reasoning_content without regular content."""
        messages = [
            {"role": "user", "content": "Think about this"},
            {"role": "assistant", "reasoning_content": "Let me think..."},
        ]

        mock_tokenizer.apply_chat_template.return_value = [
            50258,
            100,
            50257,  # user
            50258,
            200,  # assistant start
            50259,  # <think>
            1002,
            300,
            301,
            1003,  # reasoning
            50260,  # </think>
            50257,  # <|im_end|>
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Reasoning content and EOS should be unmasked
        assert 300 in result["input_ids"]
        assert 301 in result["input_ids"]
        assert 50257 in result["input_ids"]  # EOS token

        # Check labels
        idx_300 = result["input_ids"].index(300)
        idx_301 = result["input_ids"].index(301)
        # Find the last EOS token (assistant's)
        eos_indices = [i for i, t in enumerate(result["input_ids"]) if t == 50257]
        idx_eos = eos_indices[-1] if eos_indices else None

        assert result["labels"][idx_300] == 300
        assert result["labels"][idx_301] == 301
        if idx_eos is not None:
            assert result["labels"][idx_eos] == 50257

    def test_content_without_reasoning(self, mock_tokenizer):
        """Test messages that only have content without reasoning_content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        mock_tokenizer.apply_chat_template.return_value = [
            50258,
            100,
            50257,  # user
            50258,
            200,  # assistant start
            1000,
            300,
            301,
            302,
            1001,  # content
            50257,  # <|im_end|>
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Content and EOS should be unmasked
        assert all(tok in result["input_ids"] for tok in [300, 301, 302, 50257])

        # Verify unmasking
        for tok in [300, 301, 302]:
            idx = result["input_ids"].index(tok)
            assert result["labels"][idx] == tok

        # For EOS token, check the last occurrence (assistant's EOS)
        eos_indices = [i for i, t in enumerate(result["input_ids"]) if t == 50257]
        assert len(eos_indices) >= 1
        last_eos_idx = eos_indices[-1]
        assert result["labels"][last_eos_idx] == 50257

    def test_reasoning_content_order_variations(self, mock_tokenizer):
        """Test different orderings of reasoning and content regions."""
        messages = [
            {"role": "assistant", "content": "Answer", "reasoning_content": "Reasoning"}
        ]

        # Test content before reasoning (unusual but possible)
        mock_tokenizer.apply_chat_template.return_value = [
            1000,
            100,
            1001,  # content first
            50261,  # separator
            1002,
            200,
            1003,  # reasoning after
            50257,  # EOS
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Both regions should be merged and unmasked
        assert 100 in result["input_ids"]
        assert 200 in result["input_ids"]
        assert 50257 in result["input_ids"]

    def test_unmask_all_roles_with_reasoning(self, mock_tokenizer):
        """Test unmasking all roles when some have reasoning content."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {
                "role": "user",
                "content": "Question",
                "reasoning_content": "User thinking",
            },
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning_content": "Assistant thinking",
            },
        ]

        mock_tokenizer.apply_chat_template.return_value = [
            # System (no reasoning)
            1000,
            50,
            1001,
            # User with reasoning
            1002,
            100,
            1003,
            1000,
            150,
            1001,
            # Assistant with reasoning
            1002,
            200,
            1003,
            1000,
            250,
            1001,
            50257,
        ]

        result = unmask_messages(
            messages, mock_tokenizer, ["system", "user", "assistant"]
        )

        # All content should be unmasked
        unmasked_tokens = [50, 100, 150, 200, 250, 50257]
        for tok in unmasked_tokens:
            idx = result["input_ids"].index(tok)
            assert result["labels"][idx] == tok

    def test_edge_case_empty_reasoning_content(self, mock_tokenizer):
        """Test handling of empty reasoning_content field."""
        messages = [{"role": "assistant", "content": "Answer", "reasoning_content": ""}]

        # Empty reasoning should still create unmask tokens but with no content between
        mock_tokenizer.apply_chat_template.return_value = [
            1002,
            1003,  # empty reasoning
            1000,
            100,
            1001,  # content
            50257,
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Content and EOS should be unmasked
        assert 100 in result["input_ids"]
        assert 50257 in result["input_ids"]

    def test_complex_qwen_deepseek_scenario(self, mock_tokenizer):
        """Test a complex scenario mimicking Qwen/DeepSeek behavior."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "I need to add 2 and 2. 2 + 2 = 4.",
            },
            {
                "role": "assistant",
                "content": "Let me elaborate: it's basic arithmetic.",
                "reasoning_content": "",
            },
        ]

        # Simulate the complex template behavior described in the instructions
        mock_tokenizer.apply_chat_template.return_value = [
            50258,
            100,
            50257,  # user
            50258,
            200,  # first assistant
            50259,  # <think>
            1002,
            300,
            301,
            302,
            1003,  # reasoning
            50260,  # </think>
            50261,  # \n\n
            1000,
            400,
            401,
            1001,  # content
            50257,  # <|im_end|>
            50258,
            500,  # second assistant (continuation)
            50259,  # <think> (empty)
            1002,
            1003,  # empty reasoning
            50260,  # </think>
            50261,  # \n\n
            1000,
            600,
            601,
            602,
            1001,  # content
            50257,  # <|im_end|>
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Both assistant messages should be properly unmasked
        # First assistant: reasoning + content
        for tok in [300, 301, 302, 400, 401]:
            assert tok in result["input_ids"]
            idx = result["input_ids"].index(tok)
            assert result["labels"][idx] == tok

        # Second assistant: content only (empty reasoning)
        for tok in [600, 601, 602]:
            assert tok in result["input_ids"]
            idx = result["input_ids"].index(tok)
            assert result["labels"][idx] == tok

    def test_validation_nested_reasoning_tokens(self, mock_tokenizer):
        """Test that nested reasoning tokens raise appropriate errors."""
        messages = [{"role": "assistant", "content": "Test"}]

        # Nested reasoning begin tokens
        mock_tokenizer.apply_chat_template.return_value = [
            1002,
            100,
            1002,
            200,
            1003,
            1003,
        ]

        with pytest.raises(
            ValueError, match="encountered.*UNMASK_REASONING.*while already unmasking"
        ):
            unmask_messages(messages, mock_tokenizer, ["assistant"])

    def test_known_token_ids_preserved(self, mock_tokenizer):
        """Test that specific known token IDs are handled correctly."""
        messages = [
            {
                "role": "assistant",
                "content": "Final answer",
                "reasoning_content": "Thinking process",
            }
        ]

        # Use specific token IDs to verify preservation
        mock_tokenizer.apply_chat_template.return_value = [
            50258,  # <|im_start|>
            1002,
            12345,
            1003,  # reasoning with specific ID
            50260,  # </think>
            1000,
            67890,
            1001,  # content with specific ID
            50257,  # <|im_end|>
        ]

        result = unmask_messages(messages, mock_tokenizer, ["assistant"])

        # Verify specific tokens are preserved
        assert 12345 in result["input_ids"]
        assert 67890 in result["input_ids"]

        # Verify they're unmasked
        idx_12345 = result["input_ids"].index(12345)
        idx_67890 = result["input_ids"].index(67890)
        assert result["labels"][idx_12345] == 12345
        assert result["labels"][idx_67890] == 67890


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
