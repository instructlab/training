# Standard
import os

# Third Party
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import torch

# First Party
from instructlab.training.multipack_sampler import MultipackDistributedBatchSampler
from instructlab.training.utils import log_rank_0, make_collate_fn


class TokenDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_dataset("json", data_files=data_path, split="train")
        self.lengths = np.array(
            [
                len(x["input_ids"])
                for x in tqdm(self.data, desc="Data length calculation", colour="cyan")
            ]
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
        return np.array([len(self.input_ids[0])] * len(self.input_ids))


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
    samples_per_gpu=None,
    sampler="multipack",
    seed=47,
) -> DataLoader:
    collate_fn = make_collate_fn(
        pad_token_id, is_granite=is_granite, max_batch_len=max_batch_len
    )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    lengths = dataset.get_lengths()
    if sampler == "multipack":
        sampler = MultipackDistributedBatchSampler(
            batch_max_length=packing_max_batch_len,
            lengths=lengths,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            padding=not is_granite,
        )
        sampler = {"batch_sampler": sampler}
    elif sampler == "distributed":
        # Third Party
        from torch.utils.data import DistributedSampler

        sampler = (
            DistributedSampler(dataset) if torch.distributed.is_initialized() else None
        )
        sampler = {
            "sampler": sampler,
            "batch_size": samples_per_gpu,
        }
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        dataset,
        **sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader
