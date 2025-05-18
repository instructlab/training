import pytest
from datasets import Dataset
from transformers import LlamaTokenizerFast
from instructlab.training.data_process import process_samples

@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = LlamaTokenizerFast.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")

    # Ensure UNMASK tokens are treated atomically
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|UNMASK_BEGIN|>", "<|UNMASK_END|>"]
    })

    # Safety: add a pad token if it's missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "</s>"

    return tokenizer

def test_process_samples_outputs_input_ids_and_labels(tokenizer):
    dummy_data = Dataset.from_dict({
        "messages": [[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "pretraining", "content": "Some pretraining text"}
        ]],
        "unmask": [True],
    })

    # Run the function
    processed = process_samples(dummy_data, tokenizer, num_cpu_procs=1, batch_size=1)

    # Check the structure of the output
    assert "input_ids" in processed.column_names
    assert "labels" in processed.column_names

    # Sanity check one sample
    sample = processed[0]
    assert isinstance(sample["input_ids"], list)
    assert isinstance(sample["labels"], list)
    assert len(sample["input_ids"]) == len(sample["labels"])
    assert all(isinstance(x, int) for x in sample["input_ids"])
