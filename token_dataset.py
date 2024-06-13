import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from datasets import load_dataset

from utils import log_rank_0
from multipack_sampler import (
    MultipackDistributedBatchSampler,
    find_packing_max_batch_len_and_grad_accum,
)


class TokenDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_dataset("json", data_files=data_path, split="train")
        self.lengths = np.array(
            self.data.map(lambda x: {"len": len(x["input_ids"])}, num_proc=72)["len"]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[int(idx)]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        labels = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def get_lengths(self):
        return self.lengths


class MockDataset(Dataset):
    def __init__(self, data_path, max_seq_len=4600):
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
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def get_lengths(self):
        np.array([len(self.input_ids[0])] * len(self.input_ids))


def make_collate_fn(pad_token_id, is_granite=False, max_batch_len=60000):
    rank = int(os.environ["RANK"])
    if is_granite:

        def pad_collate_fn(batch):
            lens = np.array([len(item["input_ids"]) for item in batch])

            cumsum_lens = np.cumsum(lens)
            valid_up_to = int((cumsum_lens < max_batch_len).sum())
            total_len = cumsum_lens[valid_up_to - 1]

            batch = batch[:valid_up_to]
            input_ids = [x["input_ids"].tolist() for x in batch]
            labels = [x["labels"].tolist() for x in batch]
            num_loss_counted_tokens = sum(
                [(x["labels"] != -100).sum().item() for x in batch]
            )

            print(
                f"\033[96m total length: {total_len} dropped: {cumsum_lens[-1] - total_len} "
                f"num samples {len(batch)} - rank: {rank} "
                f"max len: {lens.max()} min len: {lens.min()} avg len: {lens.mean()} "
                f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
            )

            return {
                "input_ids": input_ids,
                "labels": labels,
                "num_loss_counted_tokens": num_loss_counted_tokens,
            }

    else:

        def pad_collate_fn(batch):
            lens = np.array([len(item["input_ids"]) for item in batch])
            max_len = max(lens)

            input_ids = torch.stack(
                [
                    F.pad(
                        item["input_ids"],
                        (max_len - len(item["input_ids"]), 0),
                        mode="constant",
                        value=pad_token_id,
                    )
                    for item in batch
                ]
            )
            labels = torch.stack(
                [
                    F.pad(
                        item["labels"],
                        (max_len - len(item["labels"]), 0),
                        mode="constant",
                        value=-100,
                    )
                    for item in batch
                ]
            )
            num_loss_counted_tokens = (labels != -100).sum()

            attention_mask = torch.stack(
                [
                    F.pad(
                        item["attention_mask"],
                        (max_len - len(item["attention_mask"]), 0),
                        mode="constant",
                        value=0,
                    )
                    for item in batch
                ]
            )
            print(
                f"\033[96m total tokens: {max_len * len(batch)} num samples: {len(batch)} num padding tokens: {max_len * len(batch) - lens.sum()} - rank: {rank} "
                f"max len: {max_len} min len: {min(lens)} avg len: {lens.mean()} "
                f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
            )

            return {
                "input_ids": input_ids,
                "labels": labels,
                "num_loss_counted_tokens": num_loss_counted_tokens,
                "attention_mask": attention_mask,
            }

    return pad_collate_fn


def setup_dataset(
    data_path: str,
    mock: bool = False,
    mock_len: int = 2600,
) -> Dataset:
    if mock:
        log_rank_0("Using a mock dataset.")
        dataset = MockDataset(data_path, max_seq_len=mock_len)
    else:
        dataset = TokenDataset(data_path)
    return dataset


def setup_dataloader(
    dataset: Dataset,
    pad_token_id: int,
    num_workers: int = 8,
    is_granite=False,
    max_batch_len=60000,
    packing_max_batch_len=60000,
    seed=47,
) -> DataLoader:
    collate_fn = make_collate_fn(
        pad_token_id, is_granite=is_granite, max_batch_len=max_batch_len
    )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    lengths = dataset.get_lengths()
    sampler = MultipackDistributedBatchSampler(
        batch_max_length=packing_max_batch_len,
        lengths=lengths,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        padding=not is_granite,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader
