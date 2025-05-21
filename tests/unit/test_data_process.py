# Third Party
from datasets import Dataset
from transformers import LlamaTokenizerFast
import pytest

# First Party
from instructlab.training.data_process import process_samples


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = LlamaTokenizerFast.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")

    # Ensure UNMASK tokens are treated atomically
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|UNMASK_BEGIN|>", "<|UNMASK_END|>"]}
    )

    # Safety: add a pad token if it's missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "</s>"

    return tokenizer


def test_process_samples_outputs_input_ids_and_labels(tokenizer):
    # Create a dummy dataset of 100 samples
    messages = [
        [
            {"role": "user", "content": f"Hello {i}"},
            {"role": "assistant", "content": f"Hi there {i}!"},
            {"role": "pretraining", "content": f"Pretraining text {i}"},
        ]
        for i in range(100)
    ]

    unmask_flags = [True for _ in range(100)]

    dummy_data = Dataset.from_dict(
        {
            "messages": messages,
            "unmask": unmask_flags,
        }
    )

    # Use realistic batch size
    processed = process_samples(dummy_data, tokenizer, num_cpu_procs=1, batch_size=8)

    # Check the structure
    assert "input_ids" in processed.column_names
    assert "labels" in processed.column_names
    assert len(processed) == 100

    # Check that input_ids and labels exist and match length for a few random samples
    for i in [0, 25, 50, 99]:
        sample = processed[i]
        assert isinstance(sample["input_ids"], list)
        assert isinstance(sample["labels"], list)
        assert len(sample["input_ids"]) == len(sample["labels"])
        assert all(isinstance(x, int) for x in sample["input_ids"])
