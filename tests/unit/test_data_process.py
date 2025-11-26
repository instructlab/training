# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import typing as t
import unittest

# Third Party
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase

# First Party
from instructlab.training.data_process import (
    MASK_TOKEN,
    UNMASK_BEGIN_TOKEN,
    UNMASK_END_TOKEN,
    UNMASK_REASONING_BEGIN_TOKEN,
    UNMASK_REASONING_END_TOKEN,
    unmask_messages,
    unmask_sample,
    wrap_masked_messages,
)
from instructlab.training.type_definitions import Message, ProcessedMessagesData


class TestComprehensiveUnmasking(unittest.TestCase):
    """Comprehensive test suite for unmasking behavior across various scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock tokenizer for basic tests
        self.mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        self.mock_tokenizer.name_or_path = "test-model"

        # Set up token IDs for unmask tokens
        self.unmask_begin_id = 1001
        self.unmask_end_id = 1002
        self.eos_id = 1003
        self.think_id = 1004
        self.end_think_id = 1005

        def mock_encode_special(text, add_special_tokens=False):
            if text == UNMASK_BEGIN_TOKEN:
                return [self.unmask_begin_id]
            elif text == UNMASK_END_TOKEN:
                return [self.unmask_end_id]
            elif text == "</s>":
                return [self.eos_id]
            elif text == "<think>":
                return [self.think_id]
            elif text == "</think>":
                return [self.end_think_id]
            else:
                # Simple hash-based encoding for text
                return [hash(text) % 1000 + 100 for _ in text.split()]

        self.mock_tokenizer.encode.side_effect = mock_encode_special
        self.mock_tokenizer.decode.side_effect = lambda tokens: " ".join(
            [f"token_{t}" for t in tokens]
        )
        self.mock_tokenizer.apply_chat_template.side_effect = (
            self._mock_apply_chat_template
        )
        self.mock_tokenizer.eos_token = "</s>"

    def _mock_apply_chat_template(
        self,
        messages: t.List[Message],
        tokenize: bool = True,
        add_special_tokens: bool = True,
        return_dict: bool = False,
        **kwargs,
    ) -> t.Union[str, t.List[int], t.Dict[str, t.Any]]:
        """Mock implementation of apply_chat_template."""
        template_tokens = []

        for msg in messages:
            # Add role tokens
            role_tokens = [hash(f"<|{msg['role']}|>") % 1000 + 2000]
            template_tokens.extend(role_tokens)

            # Add content tokens
            if "content" in msg and msg["content"]:
                content_tokens = [
                    hash(msg["content"]) % 1000 + 3000 for _ in msg["content"].split()
                ]
                template_tokens.extend(content_tokens)

            # Add reasoning content tokens
            if "reasoning_content" in msg and msg["reasoning_content"]:
                reasoning_tokens = [
                    hash(msg["reasoning_content"]) % 1000 + 4000
                    for _ in msg["reasoning_content"].split()
                ]
                template_tokens.extend(reasoning_tokens)

        result = (
            template_tokens
            if tokenize
            else " ".join([f"token_{t}" for t in template_tokens])
        )
        if return_dict:
            return {"input_ids": result}
        return result

    def test_single_turn_assistant_only_content(self):
        """Test basic single-turn conversation with assistant content only."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])

        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

        # Verify unmask tokens are not in final output
        self.assertNotIn(self.unmask_begin_id, result["input_ids"])
        self.assertNotIn(self.unmask_end_id, result["input_ids"])
        self.assertNotIn(self.unmask_begin_id, result["labels"])
        self.assertNotIn(self.unmask_end_id, result["labels"])

    def test_single_turn_assistant_only_reasoning(self):
        """Test single-turn with assistant reasoning_content only."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "I need to add 2 and 2 together.",
            },
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])

        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_single_turn_assistant_both_content_and_reasoning(self):
        """Test single-turn with both content and reasoning_content."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "I need to add 2 and 2 together.",
            },
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])

        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_multi_turn_conversation_basic(self):
        """Test basic multi-turn conversation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's the weather like?"},
            {
                "role": "assistant",
                "content": "I don't have access to current weather data.",
            },
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])

        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_multi_turn_with_reasoning_content(self):
        """Test multi-turn conversation with reasoning content in multiple turns."""
        messages = [
            {"role": "user", "content": "What is 5*7?"},
            {
                "role": "assistant",
                "content": "35",
                "reasoning_content": "5 times 7 equals 35",
            },
            {"role": "user", "content": "What about 6*8?"},
            {
                "role": "assistant",
                "content": "48",
                "reasoning_content": "6 times 8 equals 48",
            },
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])

        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_multi_turn_mixed_content_types(self):
        """Test multi-turn with mixed content types (some with reasoning, some without)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},  # No reasoning
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "I need to add 2 and 2.",
            },  # Both content and reasoning
            {"role": "user", "content": "Think about the meaning of life."},
            {
                "role": "assistant",
                "reasoning_content": "This is a deep philosophical question.",
            },  # Reasoning only
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])

        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_system_user_assistant_conversation(self):
        """Test conversation with system, user, and assistant roles."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is AI?"},
            {
                "role": "assistant",
                "content": "AI stands for Artificial Intelligence.",
                "reasoning_content": "This is a straightforward definition question.",
            },
        ]

        # Test unmasking only assistant
        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_multiple_unmask_roles(self):
        """Test unmasking multiple roles."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": "Question about math?",
                "reasoning_content": "I'm asking about mathematics.",
            },
            {
                "role": "assistant",
                "content": "Sure, I can help with math.",
                "reasoning_content": "Math questions are common.",
            },
        ]

        # Test unmasking both user and assistant
        result = unmask_messages(messages, self.mock_tokenizer, ["user", "assistant"])
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_reasoning_only_conversation(self):
        """Test conversation where all assistant messages have only reasoning_content."""
        messages = [
            {"role": "user", "content": "Think step by step."},
            {"role": "assistant", "reasoning_content": "Step 1: Consider the problem."},
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "reasoning_content": "Step 2: Analyze the solution."},
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_empty_content_edge_cases(self):
        """Test edge cases with empty content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Empty content case",
            },
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": "Response", "reasoning_content": ""},
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_consecutive_assistant_messages(self):
        """Test consecutive assistant messages (simulating the Qwen scenario)."""
        messages = [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": "First response A",
                "reasoning_content": "Reasoning A",
            },
            {"role": "user", "content": "Second question"},
            {
                "role": "assistant",
                "content": "Second response B",
                "reasoning_content": "Reasoning B",
            },
            {
                "role": "assistant",
                "content": "Third response C",
                "reasoning_content": "Reasoning C",
            },
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_long_multi_turn_conversation(self):
        """Test long multi-turn conversation with various content types."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"User message {i}"})

            if i % 3 == 0:
                # Content only
                messages.append(
                    {"role": "assistant", "content": f"Assistant response {i}"}
                )
            elif i % 3 == 1:
                # Reasoning only
                messages.append(
                    {"role": "assistant", "reasoning_content": f"Reasoning {i}"}
                )
            else:
                # Both content and reasoning
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Response {i}",
                        "reasoning_content": f"Reasoning {i}",
                    }
                )

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    @patch("instructlab.training.data_process.is_gpt_oss_model", return_value=False)
    def test_unmask_sample_function(self, mock_is_gpt_oss):
        """Test the unmask_sample function with various scenarios."""
        sample_scenarios = [
            # Basic conversation
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            },
            # With reasoning content
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {
                        "role": "assistant",
                        "content": "4",
                        "reasoning_content": "2 plus 2 equals 4",
                    },
                ]
            },
            # With unmask flag
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "unmask": True,
            },
            # Multi-turn with system
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            },
        ]

        for i, sample in enumerate(sample_scenarios):
            with self.subTest(scenario=i):
                result = unmask_sample(sample, self.mock_tokenizer)
                self.assertIsInstance(result, dict)
                self.assertIn("input_ids", result)
                self.assertIn("labels", result)
                self.assertIn("len", result)
                self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_wrap_masked_messages_comprehensive(self):
        """Test wrap_masked_messages with comprehensive scenarios."""
        test_cases = [
            # Single role, content only
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "unmask_roles": ["assistant"],
                "expected_wrapped_count": 1,
            },
            # Single role, reasoning only
            {
                "messages": [
                    {"role": "user", "content": "Think"},
                    {"role": "assistant", "reasoning_content": "Thinking..."},
                ],
                "unmask_roles": ["assistant"],
                "expected_wrapped_count": 1,
            },
            # Single role, both content types
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning_content": "Thinking",
                    },
                ],
                "unmask_roles": ["assistant"],
                "expected_wrapped_count": 2,  # Both content and reasoning_content wrapped
            },
            # Multiple roles
            {
                "messages": [
                    {"role": "system", "content": "System message"},
                    {
                        "role": "user",
                        "content": "User question",
                        "reasoning_content": "User thinking",
                    },
                    {
                        "role": "assistant",
                        "content": "Assistant answer",
                        "reasoning_content": "Assistant thinking",
                    },
                ],
                "unmask_roles": ["user", "assistant"],
                "expected_wrapped_count": 4,  # 2 messages × 2 fields each
            },
        ]

        for i, case in enumerate(test_cases):
            with self.subTest(case=i):
                result = wrap_masked_messages(
                    case["messages"], case["unmask_roles"], True
                )

                # Count wrapped fields
                wrapped_count = 0
                for msg in result:
                    if msg["role"] in case["unmask_roles"]:
                        if msg.get("content") and UNMASK_BEGIN_TOKEN in msg["content"]:
                            wrapped_count += 1
                        if (
                            msg.get("reasoning_content")
                            and UNMASK_REASONING_BEGIN_TOKEN in msg["reasoning_content"]
                        ):
                            wrapped_count += 1

                self.assertEqual(wrapped_count, case["expected_wrapped_count"])

    def test_error_conditions(self):
        """Test various error conditions."""
        # Test non-string content
        with self.assertRaises(ValueError):
            wrap_masked_messages(
                [{"role": "assistant", "content": ["not", "a", "string"]}],
                ["assistant"],
                True,
            )

        # Test non-string reasoning_content
        with self.assertRaises(ValueError):
            wrap_masked_messages(
                [{"role": "assistant", "reasoning_content": {"not": "a string"}}],
                ["assistant"],
                True,
            )

    def test_think_tag_handling(self):
        """Test that <think> and </think> tags are properly handled."""
        # This is a basic test since the mock tokenizer handles think tags
        messages = [
            {"role": "user", "content": "Question with <think>thinking</think>"},
            {
                "role": "assistant",
                "content": "Answer with <think>more thinking</think>",
            },
        ]

        result = unmask_messages(messages, self.mock_tokenizer, ["assistant"])
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))


class TestReasoningContentSupport(unittest.TestCase):
    """Test suite for reasoning_content field support in data processing."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock tokenizer for basic tests
        self.mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        self.mock_tokenizer.name_or_path = "test-model"
        self.mock_tokenizer.encode.side_effect = (
            lambda text, add_special_tokens=False: [
                hash(text) % 1000 for _ in text.split()
            ]
        )
        self.mock_tokenizer.decode.side_effect = lambda tokens: " ".join(
            [f"token_{t}" for t in tokens]
        )
        self.mock_tokenizer.apply_chat_template.side_effect = (
            self._mock_apply_chat_template
        )
        self.mock_tokenizer.eos_token = "</s>"

        # Set up token IDs for unmask tokens
        self.unmask_begin_id = 1001
        self.unmask_end_id = 1002
        self.unmask_reasoning_begin_id = 1004
        self.unmask_reasoning_end_id = 1005
        self.eos_id = 1003

        def mock_encode_special(text, add_special_tokens=False):
            if text == UNMASK_BEGIN_TOKEN:
                return [self.unmask_begin_id]
            elif text == UNMASK_END_TOKEN:
                return [self.unmask_end_id]
            elif text == UNMASK_REASONING_BEGIN_TOKEN:
                return [self.unmask_reasoning_begin_id]
            elif text == UNMASK_REASONING_END_TOKEN:
                return [self.unmask_reasoning_end_id]
            elif text == "</s>":
                return [self.eos_id]
            else:
                return [hash(text) % 1000]

        self.mock_tokenizer.encode.side_effect = mock_encode_special

    def _mock_apply_chat_template(
        self,
        messages: t.List[Message],
        tokenize: bool = True,
        add_special_tokens: bool = True,
        return_dict: bool = False,
        **kwargs,
    ) -> t.Union[str, t.List[int], t.Dict[str, t.Any]]:
        """Mock implementation of apply_chat_template."""
        template_str = ""
        for msg in messages:
            template_str += f"<|{msg['role']}|>\n"
            if "content" in msg:
                template_str += msg["content"]
            if "reasoning_content" in msg:
                template_str += msg["reasoning_content"]
            template_str += "\n"

        result = (
            [hash(template_str) % 1000 for _ in range(len(template_str.split()))]
            if tokenize
            else template_str
        )
        if return_dict:
            return {"input_ids": result}
        return result

    def test_wrap_masked_messages_with_reasoning_content(self):
        """Test that wrap_masked_messages correctly wraps both content and reasoning_content."""
        messages = [
            {
                "role": "user",
                "content": "What is 2+2?",
            },
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "I need to add 2 and 2 together. 2 + 2 = 4.",
            },
        ]

        unmask_roles = ["assistant"]
        result = wrap_masked_messages(messages, unmask_roles, True)

        # Check that user message is unchanged
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "What is 2+2?")
        self.assertNotIn("reasoning_content", result[0])

        # Check that assistant message has both fields wrapped
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(
            result[1]["content"],
            f"{UNMASK_BEGIN_TOKEN}The answer is 4.{UNMASK_END_TOKEN}",
        )
        self.assertEqual(
            result[1]["reasoning_content"],
            f"{UNMASK_REASONING_BEGIN_TOKEN}I need to add 2 and 2 together. 2 + 2 = 4.{UNMASK_REASONING_END_TOKEN}",
        )

    def test_wrap_masked_messages_content_only(self):
        """Test that wrap_masked_messages works with messages that only have content."""
        messages = [
            {
                "role": "user",
                "content": "Hello!",
            },
            {
                "role": "assistant",
                "content": "Hi there!",
            },
        ]

        unmask_roles = ["assistant"]
        result = wrap_masked_messages(messages, unmask_roles, True)

        # Check that user message is unchanged
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "Hello!")

        # Check that assistant message has content wrapped
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(
            result[1]["content"],
            f"{UNMASK_BEGIN_TOKEN}Hi there!{UNMASK_END_TOKEN}",
        )
        self.assertNotIn("reasoning_content", result[1])

    def test_wrap_masked_messages_reasoning_content_only(self):
        """Test that wrap_masked_messages works with messages that only have reasoning_content."""
        messages = [
            {
                "role": "user",
                "content": "Think step by step.",
            },
            {
                "role": "assistant",
                "reasoning_content": "Let me think about this step by step...",
            },
        ]

        unmask_roles = ["assistant"]
        result = wrap_masked_messages(messages, unmask_roles, True)

        # Check that user message is unchanged
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "Think step by step.")

        # Check that assistant message has reasoning_content wrapped
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(
            result[1]["reasoning_content"],
            f"{UNMASK_REASONING_BEGIN_TOKEN}Let me think about this step by step...{UNMASK_REASONING_END_TOKEN}",
        )
        self.assertNotIn("content", result[1])

    def test_wrap_masked_messages_multiple_unmask_roles(self):
        """Test that wrap_masked_messages works with multiple roles to unmask."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "What is the capital of France?",
                "reasoning_content": "This is a geography question about France.",
            },
            {
                "role": "assistant",
                "content": "The capital of France is Paris.",
                "reasoning_content": "This is a straightforward geography question.",
            },
        ]

        unmask_roles = ["user", "assistant"]
        result = wrap_masked_messages(messages, unmask_roles, True)

        # Check that system message is unchanged
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], "You are a helpful assistant.")

        # Check that user message has both fields wrapped
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(
            result[1]["content"],
            f"{UNMASK_BEGIN_TOKEN}What is the capital of France?{UNMASK_END_TOKEN}",
        )
        self.assertEqual(
            result[1]["reasoning_content"],
            f"{UNMASK_REASONING_BEGIN_TOKEN}This is a geography question about France.{UNMASK_REASONING_END_TOKEN}",
        )

        # Check that assistant message has both fields wrapped
        self.assertEqual(result[2]["role"], "assistant")
        self.assertEqual(
            result[2]["content"],
            f"{UNMASK_BEGIN_TOKEN}The capital of France is Paris.{UNMASK_END_TOKEN}",
        )
        self.assertEqual(
            result[2]["reasoning_content"],
            f"{UNMASK_REASONING_BEGIN_TOKEN}This is a straightforward geography question.{UNMASK_REASONING_END_TOKEN}",
        )

    def test_wrap_masked_messages_non_string_content_error(self):
        """Test that wrap_masked_messages raises error for non-string content."""
        messages = [
            {
                "role": "assistant",
                "content": ["This", "is", "not", "a", "string"],
            }
        ]

        unmask_roles = ["assistant"]

        with self.assertRaises(ValueError) as context:
            wrap_masked_messages(messages, unmask_roles, True)

        self.assertIn(
            "unmasking non-string data types is currently unsupported",
            str(context.exception),
        )

    def test_wrap_masked_messages_non_string_reasoning_content_error(self):
        """Test that wrap_masked_messages raises error for non-string reasoning_content."""
        messages = [
            {
                "role": "assistant",
                "content": "Valid content",
                "reasoning_content": {"thinking": "This is not a string"},
            }
        ]

        unmask_roles = ["assistant"]

        with self.assertRaises(ValueError) as context:
            wrap_masked_messages(messages, unmask_roles, True)

        self.assertIn(
            "received an entry for `reasoning_content` which was not a string",
            str(context.exception),
        )

    def test_unmask_messages_with_reasoning_content(self):
        """Test that unmask_messages correctly processes reasoning_content."""
        # This is a complex integration test, so we'll test it with a real tokenizer in the integration tests
        # For unit testing, we just verify that the wrap_masked_messages function properly handles reasoning_content
        messages = [
            {
                "role": "user",
                "content": "What is 5*7?",
            },
            {
                "role": "assistant",
                "content": "35",
                "reasoning_content": "5 times 7 equals 35",
            },
        ]

        unmask_roles = ["assistant"]

        # Test that wrap_masked_messages works correctly with reasoning_content
        wrapped = wrap_masked_messages(messages, unmask_roles, True)

        # Verify that both content and reasoning_content are wrapped
        self.assertIn(UNMASK_BEGIN_TOKEN, wrapped[1]["content"])
        self.assertIn(UNMASK_END_TOKEN, wrapped[1]["content"])
        self.assertIn(UNMASK_REASONING_BEGIN_TOKEN, wrapped[1]["reasoning_content"])
        self.assertIn(UNMASK_REASONING_END_TOKEN, wrapped[1]["reasoning_content"])

        # Verify the user message is unchanged
        self.assertEqual(wrapped[0]["content"], "What is 5*7?")
        self.assertNotIn("reasoning_content", wrapped[0])

    @patch("instructlab.training.data_process.is_gpt_oss_model", return_value=False)
    def test_unmask_sample_with_reasoning_content(self, mock_is_gpt_oss):
        """Test that unmask_sample correctly processes samples with reasoning_content."""
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain photosynthesis.",
                },
                {
                    "role": "assistant",
                    "content": "Photosynthesis is the process by which plants make food.",
                    "reasoning_content": "I need to explain photosynthesis in simple terms.",
                },
            ]
        }

        result = unmask_sample(sample, self.mock_tokenizer)

        # Check that result has the expected structure
        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)

    @patch("instructlab.training.data_process.is_gpt_oss_model", return_value=False)
    def test_unmask_sample_with_unmask_flag(self, mock_is_gpt_oss):
        """Test that unmask_sample correctly handles the unmask flag."""
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                },
                {
                    "role": "assistant",
                    "content": "Hi",
                    "reasoning_content": "Simple greeting",
                },
            ],
            "unmask": True,
        }

        result = unmask_sample(sample, self.mock_tokenizer)

        # Check that result has the expected structure
        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)


class TestReasoningContentWithRealTokenizers(unittest.TestCase):
    """Test reasoning_content functionality with real tokenizers."""

    def test_with_qwen_tokenizer(self):
        """Test reasoning_content functionality with Qwen3-32B tokenizer."""
        # Use a smaller Qwen model that's more readily available
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

        # Add the unmask tokens to the tokenizer
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    UNMASK_BEGIN_TOKEN,
                    UNMASK_END_TOKEN,
                    UNMASK_REASONING_BEGIN_TOKEN,
                    UNMASK_REASONING_END_TOKEN,
                    MASK_TOKEN,
                ]
            }
        )

        messages = [
            {
                "role": "user",
                "content": "What is 2+2?",
            },
            {
                "role": "assistant",
                "content": "4",
                "reasoning_content": "I need to add 2 and 2, which equals 4.",
            },
        ]

        # Test wrap_masked_messages
        wrapped = wrap_masked_messages(messages, ["assistant"], True)

        # Verify that both content and reasoning_content are wrapped
        self.assertIn(UNMASK_BEGIN_TOKEN, wrapped[1]["content"])
        self.assertIn(UNMASK_END_TOKEN, wrapped[1]["content"])
        self.assertIn(UNMASK_REASONING_BEGIN_TOKEN, wrapped[1]["reasoning_content"])
        self.assertIn(UNMASK_REASONING_END_TOKEN, wrapped[1]["reasoning_content"])

        # Test unmask_messages
        result = unmask_messages(messages, tokenizer, ["assistant"])

        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)

        # Verify that special tokens are not in the final output
        unmask_begin_id = tokenizer.encode(
            UNMASK_BEGIN_TOKEN, add_special_tokens=False
        )[0]
        unmask_end_id = tokenizer.encode(UNMASK_END_TOKEN, add_special_tokens=False)[0]

        self.assertNotIn(unmask_begin_id, result["input_ids"])
        self.assertNotIn(unmask_end_id, result["input_ids"])

    def test_with_phi_tokenizer(self):
        """Test reasoning_content functionality with Phi-4 tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

        # Add the unmask tokens to the tokenizer
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    UNMASK_BEGIN_TOKEN,
                    UNMASK_END_TOKEN,
                    UNMASK_REASONING_BEGIN_TOKEN,
                    UNMASK_REASONING_END_TOKEN,
                    MASK_TOKEN,
                ]
            }
        )

        messages = [
            {
                "role": "user",
                "content": "Calculate 5*6",
            },
            {
                "role": "assistant",
                "content": "30",
                "reasoning_content": "5 multiplied by 6 equals 30.",
            },
        ]

        # Test the full pipeline
        result = unmask_messages(messages, tokenizer, ["assistant"])

        # Verify the result structure and content
        self.assertIsInstance(result, dict)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertIn("len", result)
        self.assertGreater(len(result["input_ids"]), 0)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))

    def test_edge_cases_with_reasoning_content(self):
        """Test edge cases for reasoning_content functionality."""
        # Test empty reasoning_content
        messages = [
            {
                "role": "assistant",
                "content": "Response",
                "reasoning_content": "",
            }
        ]

        wrapped = wrap_masked_messages(messages, ["assistant"], True)
        self.assertEqual(
            wrapped[0]["reasoning_content"],
            f"{UNMASK_REASONING_BEGIN_TOKEN}{UNMASK_REASONING_END_TOKEN}",
        )

        # Test only reasoning_content without content
        messages = [
            {
                "role": "assistant",
                "reasoning_content": "Thinking process",
            }
        ]

        wrapped = wrap_masked_messages(messages, ["assistant"], True)
        self.assertNotIn("content", wrapped[0])
        self.assertEqual(
            wrapped[0]["reasoning_content"],
            f"{UNMASK_REASONING_BEGIN_TOKEN}Thinking process{UNMASK_REASONING_END_TOKEN}",
        )


if __name__ == "__main__":
    unittest.main()
