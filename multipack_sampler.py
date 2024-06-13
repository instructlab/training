"""
MIT License

Copyright (c) 2023 One

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
taken from https://github.com/imoneoi/multipack_sampler
"""

from typing import Optional, List

import torch.distributed as dist
from torch.utils.data import Sampler

import numpy as np
import numba


def find_packing_max_batch_len_and_grad_accum(
    num_gpus, avg_sample_len, effective_batch_size, max_batch_len_per_gpu
):
    """
    Calculate the minimum gradient accumulation steps required and the corresponding maximum batch length.

    This function determines the minimum number of gradient accumulation steps needed to process the
    effective batch size within the constraints of the maximum batch length per GPU. It starts with
    the assumption of a single step (no accumulation) and increases the number of steps until the
    calculated batch length does not exceed the maximum allowed per GPU. The goal is to find the
    lowest gradient accumulation that allows fitting the batch within GPU limits, ensuring efficient
    utilization of computational resources.

    Parameters:
    - num_gpus (int): The number of GPUs over which the batch is distributed.
    - avg_sample_len (int): The average length of samples in the dataset, used to estimate batch length.
    - effective_batch_size (int): The total batch size intended to be processed across all GPUs and
      accumulation steps.
    - max_batch_len_per_gpu (int): The maximum permissible number of tokens on each GPU to avoid memory overflow.

    Returns:
    - Tuple[int, int]: A tuple where the first element is the maximum batch length that can be achieved
      without exceeding the per-GPU limit, and the second element is the minimum number of gradient
      accumulation steps required to maintain the effective batch size.
    """
    packing_max_batch_len = max_batch_len_per_gpu + 1
    grad_accum = 0
    while packing_max_batch_len > max_batch_len_per_gpu:
        grad_accum += 1
        total_micro_batch = effective_batch_size / grad_accum
        packing_max_batch_len = int(avg_sample_len * total_micro_batch / num_gpus)
    return packing_max_batch_len, grad_accum


@numba.njit
def ffd_check(a: np.ndarray, c: int, n: int):
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins = np.full((n,), c, dtype=a.dtype)
    for size in a:
        not_found = True
        for idx in range(n):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        if not_found:
            return False

    return True


@numba.njit
def ffd_check_padding(a: np.ndarray, c: int, n: int):
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins_max_lengths = np.zeros(
        (n,), dtype=a.dtype
    )  # Track the maximum length in each bin
    bins_num_samples = np.zeros(
        (n,), dtype=np.int_
    )  # Track the number of samples in each bin

    for size in a:
        not_found = True
        for idx in range(n):
            # Calculate the new capacity if size is added to the bin
            new_capacity = max(bins_max_lengths[idx], size) * (
                bins_num_samples[idx] + 1
            )
            if new_capacity <= c:
                bins_max_lengths[idx] = max(bins_max_lengths[idx], size)
                bins_num_samples[idx] += 1
                not_found = False
                break

        if not_found:
            return False

    return True


@numba.njit
def ffd_with_result(a: np.ndarray, c: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)

    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins = []
    bins_result = []
    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break

        if add_new:
            bins.append(c - size)
            bins_result.append([indices[a_id] + start_index])

    return bins_result


@numba.njit
def ffd_with_result_padding(a: np.ndarray, c: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)

    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins_max_lengths = []  # Track the maximum length in each bin
    bins_num_samples = []  # Track the number of samples in each bin
    bins_result = []  # Track the indices of the samples in each bin

    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins_max_lengths)):
            # Calculate the new capacity if size is added to the bin
            new_capacity = max(bins_max_lengths[idx], size) * (
                bins_num_samples[idx] + 1
            )
            if new_capacity <= c:
                bins_max_lengths[idx] = max(bins_max_lengths[idx], size)
                bins_num_samples[idx] += 1
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break

        if add_new:
            bins_max_lengths.append(size)
            bins_num_samples.append(1)
            bins_result.append([indices[a_id] + start_index])

    return bins_result


@numba.njit
def allocate(
    lengths: np.ndarray,
    lengths_cumsum: np.ndarray,
    rank: int,
    c: int,
    n: int,
    padding: bool = True,
):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while r - l > 1:
            m = (l + r) // 2
            if padding:
                check = ffd_check_padding(lengths[start_index : start_index + m], c, n)
            else:
                check = ffd_check(lengths[start_index : start_index + m], c, n)
            if check:
                l = m
            else:
                r = m

        # use length l
        if padding:
            batch = ffd_with_result_padding(
                lengths[start_index : start_index + l], c, start_index
            )
        else:
            batch = ffd_with_result(
                lengths[start_index : start_index + l], c, start_index
            )
        assert len(batch) <= n
        if len(batch) < n:
            break

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])

    return result, s, len(result) * c * n


class MultipackDistributedBatchSampler(Sampler):
    """Unpadded length sampling using Multipack.
    Approximate (at most ~1.22x) the optimal solution of the identical-machines scheduling problem, which is NP-hard.
    """

    def __init__(
        self,
        batch_max_length: int,
        lengths: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        padding: bool = True,
    ):
        # Get rank
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.batch_max_length = batch_max_length
        self.lengths = lengths
        assert isinstance(self.lengths, np.ndarray)

        self.epoch = 0

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0
        self.padding = padding

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = np.random.default_rng(seed=self.seed + self.epoch).permutation(
            len(self.lengths)
        )

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, total_used, total_slots = allocate(
            lengths=lengths,
            lengths_cumsum=lengths_cumsum,
            rank=self.rank,
            c=self.batch_max_length,
            n=self.num_replicas,
            padding=self.padding,
        )

        batches = [indices[batch] for batch in batches]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots

        return batches

    def __iter__(self):
        batches = self.generate_batches(set_stats=True)
        return iter(batches)

    def __len__(self):
        return self.num_batches()

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots
