# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional
import logging

# Third Party
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import torch
import torch.distributed as dist

# First Party
from instructlab.training.batch_packer import batch_lengths_to_minibatches_lpt
from instructlab.training.config import PretrainingConfig
from instructlab.training.padded_batch_packer import (
    batch_lengths_to_minibatches_padded,
)
from instructlab.training.type_definitions import CollatedItem

logger = logging.getLogger(__name__)


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


def mb_collate_fn(minibatch, batch_num_loss_counted_tokens) -> CollatedItem:
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


def padded_mb_collate_fn(
    minibatch, batch_num_loss_counted_tokens, pad_token_id=0
) -> CollatedItem:
    """Collates a list of samples into a padded batch for standard attention.

    This function takes a minibatch (list of dataset samples) and creates padded
    tensors suitable for standard attention mechanisms. Unlike the flash attention
    version, this pads all sequences to the same length and creates attention masks.

    Args:
        minibatch: A list of dictionaries, where each dictionary represents a
                   sample and contains 'input_ids' and 'labels'.
        batch_num_loss_counted_tokens: Total number of loss-counted tokens in the batch.

    Returns:
        A dictionary containing the collated batch:
        - 'input_ids': 2D tensor of padded input IDs [batch_size, max_len]
        - 'labels': 2D tensor of padded labels [batch_size, max_len]
        - 'attention_mask': 2D tensor indicating real vs padding tokens
        - 'num_loss_counted_tokens': Total number of non-ignored label tokens
        - 'num_samples': The number of real sequences in this batch
    """
    if not minibatch:
        # Return empty batch
        return {
            "input_ids": torch.tensor([[]], dtype=torch.long),
            "labels": torch.tensor([[]], dtype=torch.long),
            "attention_mask": torch.tensor([[]], dtype=torch.long),
            "num_loss_counted_tokens": 0,
            "num_samples": 0,
            "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
            "total_length": 0,
        }

    # Find max length in this batch
    max_len = max(len(item["input_ids"]) for item in minibatch)

    # Prepare lists for batched tensors
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    num_loss_counted_tokens = 0
    num_samples = 0

    for item in minibatch:
        item_len = len(item["input_ids"])

        # Pad input_ids with the provided pad_token_id
        pad_length = max_len - item_len
        input_ids = item["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        padded_input = input_ids + [pad_token_id] * pad_length
        padded_input_ids.append(padded_input)

        # Pad labels with -100 (ignore index)
        labels = item["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        padded_label = labels + [-100] * pad_length
        padded_labels.append(padded_label)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * item_len + [0] * pad_length
        attention_masks.append(attention_mask)

        # Count loss tokens and samples
        num_loss_counted_tokens += item["num_loss_counted_tokens"]
        # Only count as a sample if it has loss-counted tokens
        if item["num_loss_counted_tokens"] > 0:
            num_samples += 1

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "num_samples": num_samples,
        "batch_num_loss_counted_tokens": batch_num_loss_counted_tokens,
        "total_length": max_len * len(minibatch),  # Total padded tokens
    }


class MaxTokensPerRankCollator:
    """A unified collate function for PyTorch DataLoader for distributed training.

    This collator supports both flash attention (unpadded) and standard attention (padded) modes.
    It takes a batch of samples and:
    1. Filters out samples longer than `max_tokens_per_rank`.
    2. Uses the appropriate batch packing algorithm to distribute samples across ranks.
    3. Collates samples into the format required by the model.

    Args:
        max_tokens_per_rank (int): Maximum number of tokens allowed per rank in a minibatch.
        rank (int, optional): The rank of the current process. If None, uses torch.distributed.
        world_size (int, optional): Total number of ranks. If None, uses torch.distributed.
        dummy_sample (dict, optional): A sample used for padding when a rank has no real samples.
        flash_enabled (bool): Whether to use flash attention mode (default: True).
        pad_token_id (int): Token ID to use for padding in non-flash mode (default: 0).
    """

    def __init__(
        self,
        max_tokens_per_rank: int,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        dummy_sample=None,
        flash_enabled: bool = True,
        pad_token_id: int = 0,
    ):
        self.max_tokens_per_rank = max_tokens_per_rank
        self.flash_enabled = flash_enabled
        self.pad_token_id = pad_token_id

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

        # Select the appropriate batch packer and collate function
        if flash_enabled:
            self.batch_packer = batch_lengths_to_minibatches_lpt
            self.collate_fn = mb_collate_fn
        else:
            self.batch_packer = batch_lengths_to_minibatches_padded
            # Create a wrapper for padded collate that includes pad_token_id
            self.collate_fn = (
                lambda minibatch, batch_num_loss_counted_tokens: padded_mb_collate_fn(
                    minibatch, batch_num_loss_counted_tokens, pad_token_id
                )
            )

    def __call__(self, batch: list[dict]):
        """Processes a batch of samples into minibatches for the current rank.

        Args:
            batch: A list of sample dictionaries from the Dataset.

        Returns:
            A list where each element is a collated minibatch ready for processing.
        """
        # Filter out samples longer than max_tokens_per_rank
        batch_ = [b for b in batch if b["len"] <= self.max_tokens_per_rank]
        if len(batch_) < len(batch):
            print(
                f"\033[38;5;196mremoved {len(batch) - len(batch_)} samples from batch because they are longer than the max tokens per gpu\033[0m"
            )

        # Extract lengths and count loss tokens
        batch_lengths = [sample["len"] for sample in batch_]
        batch_num_loss_counted_tokens = sum(
            [sample["num_loss_counted_tokens"] for sample in batch_]
        )

        # Use the appropriate batch packer
        all_minibatches_indices = self.batch_packer(
            batch_lengths, self.max_tokens_per_rank, self.world_size, self.global_rank
        )

        # Collate minibatches
        all_minibatches = []
        for mb_indices in all_minibatches_indices:
            mb = [batch_[i] if i != -1 else self.dummy_sample for i in mb_indices]
            all_minibatches.append(self.collate_fn(mb, batch_num_loss_counted_tokens))

        return all_minibatches


class PretrainingBlockDataset(Dataset):
    """Dataset that concatenates documents and exposes fixed-size blocks."""

    def __init__(self, dataset: HFDataset, block_size: int, pad_token_id: int):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if "input_ids" not in dataset.column_names:
            raise ValueError("Pretraining data must provide an 'input_ids' column.")
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer.")

        self.block_size = block_size
        self.pad_token_id = pad_token_id

        all_input_ids: list[int] = []
        for sample in dataset:
            ids = sample["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            all_input_ids.extend(ids)

        total_tokens = len(all_input_ids)
        if total_tokens == 0:
            raise ValueError("Pretraining dataset is empty after concatenation.")

        num_blocks, remainder = divmod(total_tokens, block_size)
        if remainder:
            num_blocks += 1

        self.all_input_ids = all_input_ids
        self.num_blocks = num_blocks
        self.last_block_len = remainder if remainder else block_size
        self.total_tokens = total_tokens

        logger.info(
            "Pretraining dataset: %s tokens → %s block(s) (block_size=%s, remainder=%s)",
            f"{total_tokens:,}",
            f"{self.num_blocks:,}",
            block_size,
            remainder,
        )

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int):
        if index < 0 or index >= self.num_blocks:
            raise IndexError(
                f"Index {index} out of range for {self.num_blocks} blocks."
            )

        start = index * self.block_size
        end = start + self.block_size
        is_last_block = index == self.num_blocks - 1
        is_partial = is_last_block and self.last_block_len != self.block_size

        if is_partial:
            actual_tokens = self.all_input_ids[start:]
            actual_len = len(actual_tokens)
            pad_len = self.block_size - actual_len

            input_ids = actual_tokens + [self.pad_token_id] * pad_len
            labels = actual_tokens + [-100] * pad_len
            loss_tokens = max(actual_len - 1, 0)
        else:
            input_ids = self.all_input_ids[start:end]
            labels = list(input_ids)
            loss_tokens = self.block_size - 1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "len": self.block_size,
            "num_loss_counted_tokens": loss_tokens,
        }

    @classmethod
    def from_jsonl_file(
        cls,
        data_path: str,
        block_size: int,
        pad_token_id: int,
    ) -> "PretrainingBlockDataset":
        dataset = load_dataset("json", data_files=data_path, split="train")
        return cls(dataset, block_size, pad_token_id)

    def get_lengths(self) -> np.ndarray:
        lengths = np.full(self.num_blocks, self.block_size, dtype=np.int64)
        if self.num_blocks and self.last_block_len != self.block_size:
            lengths[-1] = self.last_block_len
        return lengths


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
    flash_enabled: bool = True,
    pad_token_id: int = 0,
    pretraining_config: Optional[PretrainingConfig] = None,
):
    """Create a data loader with epoch-based sampling and batch packing.

    Args:
        data_path: Path to the JSONL data file
        batch_size: Number of samples to fetch per batch (before packing)
        max_tokens_per_gpu: Maximum tokens allowed per GPU
        seed: Random seed for sampling
        rank: Current process rank
        world_size: Total number of processes
        dummy_sample: Sample used for padding
        num_workers: Number of data loading workers
        flash_enabled: Whether flash attention is enabled (affects collation strategy)
        pad_token_id: Token ID to use for padding (only used when flash_enabled=False)
        pretraining_config: When provided, enables block-based pretraining dataset loading

    Returns:
        DataLoader configured with appropriate collator based on flash_enabled
    """
    if pretraining_config is not None:
        dataset = PretrainingBlockDataset.from_jsonl_file(
            data_path, pretraining_config.block_size, pad_token_id
        )
        logger.info(
            "Using pretraining dataset with block_size=%s and %s block(s)",
            pretraining_config.block_size,
            f"{len(dataset):,}",
        )
    else:
        dataset = TokenDataset(data_path)

    sampler = EpochSampler(len(dataset), seed=seed)

    # Create unified collator with appropriate mode
    collate_fn = MaxTokensPerRankCollator(
        max_tokens_per_gpu,
        rank=rank,
        world_size=world_size,
        dummy_sample=dummy_sample,
        flash_enabled=flash_enabled,
        pad_token_id=pad_token_id,
    )

    return DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
