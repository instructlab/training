# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional
import os

# Third Party
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

# First Party
# TokenDataset is now imported from sampler but not used directly here
from instructlab.training.sampler import (
    EpochSampler,
    MaxTokensPerRankCollator,
    get_data_loader,
)
from instructlab.training.utils import log_rank_0


class TokenDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_dataset("json", data_files=data_path, split="train")
        if "len" not in self.data.column_names:
            self.lengths = np.array(
                self.data.map(
                    lambda x: {"len": len(x["input_ids"])},
                    num_proc=8,
                )["len"]
            )
        else:
            self.lengths = np.array(self.data["len"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[int(idx)]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        labels = torch.tensor(item["labels"], dtype=torch.long)

        # Calculate num_loss_counted_tokens if not present
        if (loss_counted_tokens := item.get("num_loss_counted_tokens", None)) is None:
            loss_counted_tokens = sum(
                1 if label != -100 else 0 for label in item["labels"]
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "len": len(input_ids),
            "num_loss_counted_tokens": loss_counted_tokens,
        }

    def get_lengths(self):
        return self.lengths


class MockDataset(Dataset):
    def __init__(self, max_seq_len=4600):
        self.input_ids = np.random.randint(
            0, 10000, size=(92000, max_seq_len), dtype=np.int16
        )
        self.labels = np.random.randint(
            0, 10000, size=(92000, max_seq_len), dtype=np.int16
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)

        # For mock data, assume all tokens count toward loss
        loss_counted_tokens = len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "len": len(input_ids),
            "num_loss_counted_tokens": loss_counted_tokens,
        }

    def get_lengths(self):
        return np.array([len(self.input_ids[0])] * len(self.input_ids))


def setup_dataset(
    data_path: str,
    mock: bool = False,
    mock_len: int = 2600,
) -> Dataset:
    if mock:
        log_rank_0("Using a mock dataset.")
        dataset = MockDataset(max_seq_len=mock_len)
    else:
        dataset = TokenDataset(data_path)
    return dataset


def setup_dataloader(
    dataset: Dataset,
    num_workers: int = 8,
    packing_max_batch_len=60000,
    batch_size: int = 128,  # Mini_trainer style batch_size parameter
    seed=47,
    data_path: Optional[str] = None,  # Original data path for mini_trainer approach
) -> DataLoader:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Check if this is a MockDataset (for testing)
    if isinstance(dataset, MockDataset):
        # For mock datasets, use mini_trainer approach with EpochSampler and MaxTokensPerRankCollator
        epoch_sampler = EpochSampler(len(dataset), seed=seed)
        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=packing_max_batch_len,
            rank=rank,
            world_size=world_size,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,  # Use the passed batch_size parameter
            sampler=epoch_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

    # For real datasets, use the provided data_path with mini_trainer's get_data_loader
    if data_path is None:
        # Try to extract from dataset as fallback
        if hasattr(dataset, "data") and hasattr(dataset.data, "data_files"):
            data_path = (
                dataset.data.data_files[0]
                if isinstance(dataset.data.data_files, list)
                else dataset.data.data_files
            )
        else:
            raise ValueError(
                "No data_path provided and cannot extract from dataset. Pass data_path parameter."
            )

    # Use mini_trainer's exact get_data_loader approach
    return get_data_loader(
        data_path=data_path,
        batch_size=batch_size,  # Use the passed batch_size parameter
        max_tokens_per_gpu=packing_max_batch_len,
        seed=seed,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
    )
