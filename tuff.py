import argparse
import tempfile
import os
from typing import List
from dataclasses import dataclass
import json
from datasets import Dataset
import random

random.seed(42)

# this script is so tuff 💔

from instructlab.training.data_process import new_main
from instructlab.training.config import DataProcessArgs


@dataclass
class Message:
    role: str
    content: str

@dataclass
class Sample:
    messages: List[Message]
    unmask: bool
    id: str

    def to_dict(self):
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "unmask": self.unmask,
            "id": self.id
        }

def create_sample_message():
    # Create a simple conversation with 2-3 turns
    messages = []
    messages.append(Message(role="user", content="What is the capital of France?"))
    messages.append(Message(role="assistant", content="The capital of France is Paris."))
    
    # Sometimes add a follow-up
    if random.random() > 0.5:
        messages.append(Message(role="user", content="What's the population of Paris?"))
        messages.append(Message(role="assistant", content="Paris has a population of approximately 2.2 million people in the city proper."))
    
    return messages

def create_datasets():
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create 5 samples for fully unmasked dataset
    full_unmask_samples = []
    for i in range(5):
        sample = Sample(
            messages=create_sample_message(),
            unmask=True,
            id=f"full_unmask_{i}"
        )
        full_unmask_samples.append(sample.to_dict())
    
    # Create 5 samples for mixed dataset
    mixed_unmask_samples = []
    for i in range(5):
        sample = Sample(
            messages=create_sample_message(),
            unmask=random.choice([True, False]),  # Randomly choose unmask status
            id=f"mixed_unmask_{i}"
        )
        mixed_unmask_samples.append(sample.to_dict())
    
    # Save datasets using datasets library
    full_unmask_path = os.path.join(temp_dir, "full_unmask.jsonl")
    mixed_unmask_path = os.path.join(temp_dir, "mixed_unmask.jsonl")
    
    Dataset.from_list(full_unmask_samples).to_json(full_unmask_path)
    Dataset.from_list(mixed_unmask_samples).to_json(mixed_unmask_path)
    
    return full_unmask_path, mixed_unmask_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat-tmpl-path', type=str, required=True, help='path to the chat template we should use')
    parser.add_argument('--max-seq-len', type=int, default=131072)

    args = parser.parse_args()

    # Create our test datasets
    full_unmask_path, mixed_unmask_path = create_datasets()
    print(f"Created datasets at:\nFull unmask: {full_unmask_path}\nMixed unmask: {mixed_unmask_path}")

    # the tokenizers we care about 
    tokenizers = {
        "meta-llama/Llama-3.1-8B-Instruct",
        "microsoft/phi-4",
        "ibm-granite/granite-3.1-8b-instruct",
        "ibm-granite/granite-7b-instruct"
    }
    completed = set()
    for tokenizer in tokenizers:
        # Process both datasets
        for dataset_path in [full_unmask_path, mixed_unmask_path]:
            output_name = os.path.basename(dataset_path).replace('.jsonl', '_processed.jsonl')
            args_dp = DataProcessArgs(
                chat_tmpl_path=args.chat_tmpl_path,
                data_path=dataset_path,
                data_output_path=output_name,
                max_seq_len=args.max_seq_len,
                num_cpu_procs=16,
                model_path=tokenizer
            )
            try:
                new_main(args_dp)
            except Exception as e:
                print(f'failed with {tokenizer=}, error: {e}')
            else:
                completed.add(tokenizer)
    
    failed = tokenizers - completed
    assert not failed, f"failed to tokenizer with the following models: {list(failed)}"
    print("Processing complete!")

    # next thing we need to do is buy a property in egypt by adding in logic to:
    # 1. perform the unmasking per our existing policies
    # 2. handle legacy pretraining samples

