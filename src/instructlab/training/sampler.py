# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional

# Third Party
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import torch
import torch.distributed as dist

# First Party
from instructlab.training.batch_packer import batch_lengths_to_minibatches_lpt


class EpochSampler(Sampler):
    """
    Epoch-based sampler that provides shuffled data indices for each epoch.
    Replaces the naive distributed sampler with reproducible epoch-based shuffling.
    """

    def __init__(self, len_data: int, seed: int = 67, epoch: int = 0):
        self.len_data = len_data
        self.seed = seed
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def generate_samples(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        samples = torch.randperm(self.len_data, generator=g).tolist()
        return samples

    def __iter__(self):
        samples = self.generate_samples()
        yield from samples

    def __len__(self):
        return self.len_data


def mb_collate_fn(minibatch, batch_num_loss_counted_tokens):
    """Collates a list of samples into a single packed batch for Flash Attention.

    This function takes a 'minibatch' (list of pre-fetched dataset samples)
    and concatenates their 'input_ids', 'labels', and generates corresponding
    'position_ids'. It does *not* add padding.

    The resulting batch format is 'packed' or 'unpadded', where multiple sequences
    are concatenated into single tensors. Sequence boundaries are implicitly defined
    by the 'position_ids', which restart from 0 for each concatenated sequence.

    **IMPORTANT**: This format requires the downstream model's attention mechanism
    (e.g., Flash Attention) to correctly handle packed sequences. Standard attention
    implementations may not work correctly as they expect padded inputs and explicit
    attention masks. Flash Attention typically uses mechanisms like `cu_seqlens`
    (cumulative sequence lengths), derived from position IDs or sequence lengths,
    to compute the correct block-diagonal attention implicitly.

    Args:
        minibatch: A list of dictionaries, where each dictionary represents a
                   sample and contains at least 'input_ids' and 'labels'.
        batch_num_loss_counted_tokens: Total number of loss-counted tokens in the batch.

    Returns:
        A dictionary containing the collated batch:
        - 'input_ids': Single tensor of concatenated input IDs.
        - 'labels': Single tensor of concatenated labels.
        - 'position_ids': Single tensor of position IDs, reset for each sequence.
        - 'num_loss_counted_tokens': Total number of non-ignored label tokens (-100).
        - 'num_samples': The number of sequences packed into this batch.
    """
    input_ids = []
    labels = []
    position_ids = []
    total_len = 0
    num_loss_counted_tokens = 0
    num_samples = 0

    for item in minibatch:
        item_len = len(item["input_ids"])

        input_ids.extend(item["input_ids"])
        labels.extend(item["labels"])
        position_ids.extend(range(item_len))

        total_len += item_len
        num_loss_counted_tokens += item["num_loss_counted_tokens"]

        # Dummy samples don't have labels != -100 and should not count
        num_samples += 1 if item["num_loss_counted_tokens"] > 0 else 0

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "position_ids": torch.tensor([position_ids], dtype=torch.long),
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "num_samples": num_samples,
        "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
        "total_length": total_len,  # Total tokens in the batch
    }


class MaxTokensPerRankCollator:
    """A collate function for PyTorch DataLoader for distributed training.

    This collator takes a batch of samples (obtained using indices from a sampler
    like EpochSampler) and performs two main tasks:
    1. Filters out samples longer than `max_tokens_per_rank`.
    2. Uses `batch_lengths_to_minibatches_lpt` to determine how to distribute the
       remaining samples across ranks into one or more 'minibatches', ensuring
       no rank exceeds `max_tokens_per_rank` per minibatch.
    3. For the current rank, it fetches the assigned samples (or dummy samples
       for padding) for each determined minibatch.
    4. Uses `mb_collate_fn` to collate the samples for each minibatch into the
       packed format required by Flash Attention.

    Args:
        max_tokens_per_rank (int): Maximum number of tokens allowed per rank
            in a single processed minibatch.
        rank (int, optional): The rank of the current process. If None, attempts
            to get it from `torch.distributed`.
        world_size (int, optional): Total number of ranks. If None, attempts
            to get it from `torch.distributed`.
        dummy_sample (dict, optional): A sample used for padding when a rank
            has no real samples assigned in a minibatch.
    """

    def __init__(
        self,
        max_tokens_per_rank: int,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        dummy_sample=None,
    ):
        self.max_tokens_per_rank = max_tokens_per_rank

        self.global_rank = rank if rank is not None else dist.get_rank()
        self.world_size = (
            world_size if world_size is not None else dist.get_world_size()
        )
        if dummy_sample is None:
            dummy_sample = {
                "input_ids": torch.tensor([15, 14, 13, 12, 11], dtype=torch.long),
                "labels": torch.tensor(
                    [-100, -100, -100, -100, -100], dtype=torch.long
                ),
                "len": 5,
                "num_loss_counted_tokens": 0,
            }
        self.dummy_sample = dummy_sample

    def __call__(self, batch: list[dict]):
        """Processes a batch of samples into a list of packed minibatches for the current rank.

        Args:
            batch: A list of sample dictionaries from the Dataset.

        Returns:
            A list where each element is a dictionary representing a collated minibatch
            (output of `mb_collate_fn`) ready for processing by the current rank.
        """
        batch_ = [b for b in batch if b["len"] <= self.max_tokens_per_rank]
        if len(batch_) < len(batch):
            print(
                f"\033[38;5;196mremoved {len(batch) - len(batch_)} samples from batch because they are longer than the max tokens per gpu\033[0m"
            )

        # Use filtered batch for lengths and loss counts
        batch_lengths = [sample["len"] for sample in batch_]
        batch_num_loss_counted_tokens = sum(
            [sample["num_loss_counted_tokens"] for sample in batch_]
        )
        all_minibatches_indices = batch_lengths_to_minibatches_lpt(
            batch_lengths, self.max_tokens_per_rank, self.world_size, self.global_rank
        )

        all_minibatches = []
        for mb_indices in all_minibatches_indices:
            mb = [batch_[i] if i != -1 else self.dummy_sample for i in mb_indices]
            all_minibatches.append(mb_collate_fn(mb, batch_num_loss_counted_tokens))

        return all_minibatches


class TokenDataset(Dataset):
    """Dataset for loading tokenized data from JSONL files.

    Handles both InstructLab format and mini_trainer format data.
    """

    def __init__(self, data_path: str):
        dataset = load_dataset("json", data_files=data_path, split="train")
        self.dataset = dataset

        # Compute lengths if not present
        if "len" not in self.dataset.column_names:
            self.lengths = np.array(
                self.dataset.map(
                    lambda x: {"len": len(x["input_ids"])},
                    num_proc=8,
                )["len"]
            )
        else:
            self.lengths = np.array(self.dataset["len"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[int(index)]

        # Calculate num_loss_counted_tokens if not present
        if (loss_counted_tokens := sample.get("num_loss_counted_tokens", None)) is None:
            loss_counted_tokens = sum(
                1 if label != -100 else 0 for label in sample["labels"]
            )

        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "len": sample["len"],
            "num_loss_counted_tokens": loss_counted_tokens,
        }

    def get_lengths(self):
        return self.lengths


def get_data_loader(
    data_path: str,
    batch_size: int,
    max_tokens_per_gpu: int,
    seed: int,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    dummy_sample: Optional[dict] = None,
    num_workers: int = 0,
):
    """Create a data loader with epoch-based sampling and LPT batch packing.

    Args:
        data_path: Path to the JSONL data file
        batch_size: Number of samples to fetch per batch (before packing)
        max_tokens_per_gpu: Maximum tokens allowed per GPU
        seed: Random seed for sampling
        rank: Current process rank
        world_size: Total number of processes
        dummy_sample: Sample used for padding
        num_workers: Number of data loading workers

    Returns:
        DataLoader configured with EpochSampler and MaxTokensPerRankCollator
    """
    dataset = TokenDataset(data_path)
    sampler = EpochSampler(len(dataset), seed=seed)

    return DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        collate_fn=MaxTokensPerRankCollator(
            max_tokens_per_gpu,
            rank=rank,
            world_size=world_size,
            dummy_sample=dummy_sample,
        ),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
