from typing import Optional, List

import torch.distributed as dist
from torch.utils.data import Sampler
import torch

import numpy as np
import numba


def find_packing_max_batch_len_and_grad_accum(
    num_gpus,
    avg_sample_len,
    effective_batch_size,
    max_batch_len_per_gpu,
    dataset,
    seed,
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
        samples_per_minibatch = effective_batch_size / grad_accum
        samples_per_gpu = samples_per_minibatch / num_gpus
        if int(avg_sample_len * samples_per_gpu) < dataset.get_lengths().max():
            raise RuntimeError(
                f"Effective batch size is too low for multipack sampling, max sample length={dataset.get_lengths().max()} and min packing length={int(avg_sample_len * samples_per_gpu)}. "
                "Switching to naive distributed sampling."
            )

        packing_max_batch_len = int((avg_sample_len) * samples_per_gpu)

    return packing_max_batch_len, grad_accum


@numba.njit
def lpt_check(heap: np.ndarray, A: np.ndarray, c: int, n: int):
    # LPT (Longest processing time first scheduling)
    # Time: O(|A| log |A| + |A| log n)

    A = np.sort(A)[::-1]
    heap.fill(0)
    for size in A:
        # Put into smallest element
        heap[1] += size
        if heap[1] > c:
            return False

        # Heapify (Sink)
        # https://stackoverflow.com/questions/20397674/replacing-element-in-min-heap
        u = 1
        while (u << 1) <= n:
            v = u << 1  # lch
            rch = (u << 1) | 1
            if rch <= n and heap[rch] < heap[v]:
                v = rch

            if heap[u] <= heap[v]:
                break

            heap[u], heap[v] = heap[v], heap[u]
            u = v

    return True


@numba.njit
def lpt_with_result(
    heap: np.ndarray, A: np.ndarray, n: int, start_index: int, rank: int
):
    # LPT (Longest processing time first scheduling)
    # Time: O(|A| log |A| + |A| log n)

    result = []

    indices = np.argsort(A)[::-1]
    A = A[indices]

    heap.fill(0)
    heap_id = np.arange(-1, n, dtype=A.dtype)
    for idx, size in enumerate(A):
        # Put into smallest element
        heap[1] += size
        if heap_id[1] == rank:
            result.append(start_index + indices[idx])

        # Heapify (Sink)
        # https://stackoverflow.com/questions/20397674/replacing-element-in-min-heap
        u = 1
        while (u << 1) <= n:
            v = u << 1  # lch
            rch = (u << 1) | 1
            if rch <= n and heap[rch] < heap[v]:
                v = rch

            if heap[u] <= heap[v]:
                break

            heap[u], heap[v] = heap[v], heap[u]
            heap_id[u], heap_id[v] = heap_id[v], heap_id[u]
            u = v

    return result


@numba.njit
def allocate(
    lengths: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int
):
    # Dynamic batch allocator, binary search + LPT
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    heap = np.zeros(n + 1, dtype=lengths.dtype)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while r - l > 1:
            m = (l + r) // 2
            if lpt_check(heap, lengths[start_index : start_index + m], c, n):
                l = m
            else:
                r = m

        # use length l
        if l < n:
            break  # Can't allocate each sequence to a single machine

        batch = lpt_with_result(
            heap, lengths[start_index : start_index + l], n, start_index, rank
        )

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch)

    return result, s, len(result) * c * n


class MultipackDistributedBatchSamplerV2(Sampler):
    """Unpadded length sampling using Multipack V2, for models with quadratic attention complexity.
    It also tries to evenly distribute the sequences using LPT, so that quadratic load is more balanced.

    Approximate (at most 1.33x ?) the optimal solution of the identical-machines scheduling problem, which is NP-hard.

    Time Complexity: O(n log n log k)
    n = maximum number of sequences per batch, k = number of nodes
    """

    def __init__(
        self,
        batch_max_length: int,
        lengths: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
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

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = np.random.Generator(
            np.random.Philox(seed=self.seed + self.epoch)
        ).permutation(len(self.lengths))

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, total_used, total_slots = allocate(
            lengths=lengths,
            lengths_cumsum=lengths_cumsum,
            rank=self.rank,
            c=self.batch_max_length,
            n=self.num_replicas,
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
        # use the latest cached value or create one if needed.
        return self.num_batches()

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots
