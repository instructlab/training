"""
Numba-optimized batch packing for padded training (non-flash attention).

This module provides high-performance batch packing that minimizes padding
while maintaining good load balance across distributed training ranks.
"""

# Third Party
from numba import int64, njit
import numpy as np


@njit
def _compute_padded_tokens(lengths: np.ndarray, start: int64, end: int64) -> int64:
    """Compute total tokens including padding for a batch."""
    if start >= end:
        return 0
    max_len = lengths[start]  # Since sorted descending, first is max
    return max_len * (end - start)


@njit
def _compute_wasted_tokens(lengths: np.ndarray, start: int64, end: int64) -> int64:
    """Compute wasted tokens (padding) for a batch."""
    if start >= end:
        return 0
    max_len = lengths[start]
    total_actual = 0
    for i in range(start, end):
        total_actual += lengths[i]
    return max_len * (end - start) - total_actual


@njit
def _find_optimal_batch_size(
    lengths: np.ndarray, start_idx: int64, max_tokens: int64, max_batch_size: int64
) -> int64:
    """Find optimal batch size that minimizes padding ratio while fitting constraints."""
    n = len(lengths)
    if start_idx >= n:
        return 0

    # Maximum possible batch size given token constraint
    max_len = lengths[start_idx]
    max_sequences = (
        min(max_tokens // max_len, max_batch_size) if max_len > 0 else max_batch_size
    )
    max_sequences = min(max_sequences, n - start_idx)

    if max_sequences <= 1:
        return 1

    # Find batch size that minimizes padding ratio
    best_size = 1
    best_ratio = 1.0

    for size in range(2, max_sequences + 1):
        # Check if this batch size fits within token limit
        padded_tokens = _compute_padded_tokens(lengths, start_idx, start_idx + size)
        if padded_tokens > max_tokens:  
            break

        # Compute padding ratio
        actual_tokens = 0
        for i in range(start_idx, start_idx + size):
            actual_tokens += lengths[i]

        if padded_tokens > 0:
            padding_ratio = float(padded_tokens - actual_tokens) / float(padded_tokens)

            # Prefer larger batches if padding ratio is similar (within 5%)
            if padding_ratio < best_ratio - 0.05 or (
                abs(padding_ratio - best_ratio) < 0.05 and size > best_size
            ):
                best_ratio = padding_ratio
                best_size = size

    return best_size


@njit
def _distribute_batches_balanced(
    batch_tokens: np.ndarray, num_ranks: int64, rank: int64
) -> np.ndarray:
    """Distribute batches across ranks to balance total padded tokens."""
    n_batches = len(batch_tokens)
    if n_batches == 0:
        return np.empty(0, dtype=np.int64)

    # Compute cumulative load for each rank using round-robin with load balancing
    rank_loads = np.zeros(num_ranks, dtype=np.int64)
    batch_assignments = np.empty(n_batches, dtype=np.int64)

    # Sort batches by size (largest first) for better load balancing
    sorted_indices = np.argsort(batch_tokens)[::-1]

    for i in range(n_batches):
        batch_idx = sorted_indices[i]
        # Assign to least loaded rank
        min_rank = 0
        min_load = rank_loads[0]
        for r in range(1, num_ranks):
            if rank_loads[r] < min_load:
                min_load = rank_loads[r]
                min_rank = r

        batch_assignments[batch_idx] = min_rank
        rank_loads[min_rank] += batch_tokens[batch_idx]

    # Return indices for this rank
    my_batches = []
    for i in range(n_batches):
        if batch_assignments[i] == rank:
            my_batches.append(i)

    return np.array(my_batches, dtype=np.int64)


@njit
def _padded_batch_packing_core(
    lengths: np.ndarray,
    max_tokens: int64,
    num_ranks: int64,
    rank: int64,
    max_batch_size: int64,
) -> tuple:
    """Core numba-optimized batch packing for padded training.

    Returns:
        batch_indices: 2D array where each row is a batch of sequence indices
        batch_sizes: number of sequences in each batch
    """
    n_sequences = len(lengths)
    if n_sequences == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)

    # Sort by length descending for better packing
    sorted_indices = np.argsort(lengths)[::-1]
    sorted_lengths = lengths[sorted_indices]

    # First pass: create all batches
    max_batches = n_sequences  # Worst case: one sequence per batch
    temp_batch_starts = np.zeros(max_batches, dtype=np.int64)
    temp_batch_sizes = np.zeros(max_batches, dtype=np.int64)
    temp_batch_tokens = np.zeros(max_batches, dtype=np.int64)
    n_batches = 0

    start_idx = 0
    while start_idx < n_sequences:
        # Find optimal batch size for current position
        batch_size = _find_optimal_batch_size(
            sorted_lengths, start_idx, max_tokens, max_batch_size
        )

        if batch_size == 0:
            break

        temp_batch_starts[n_batches] = start_idx
        temp_batch_sizes[n_batches] = batch_size
        temp_batch_tokens[n_batches] = _compute_padded_tokens(
            sorted_lengths, start_idx, start_idx + batch_size
        )

        n_batches += 1
        start_idx += batch_size

    # Distribute batches across ranks
    batch_tokens = temp_batch_tokens[:n_batches]
    my_batch_indices = _distribute_batches_balanced(batch_tokens, num_ranks, rank)

    if len(my_batch_indices) == 0:
        # Return single batch with padding indicator
        result_indices = np.full((1, 1), -1, dtype=np.int64)
        result_sizes = np.ones(1, dtype=np.int64)
        return result_indices, result_sizes

    # Build result for this rank
    n_my_batches = len(my_batch_indices)
    max_batch_len = 0
    for i in range(n_my_batches):
        batch_idx = my_batch_indices[i]
        size = temp_batch_sizes[batch_idx]
        max_batch_len = max(max_batch_len, size)

    result_indices = np.full((n_my_batches, max_batch_len), -1, dtype=np.int64)
    result_sizes = np.zeros(n_my_batches, dtype=np.int64)

    for i in range(n_my_batches):
        batch_idx = my_batch_indices[i]
        start = temp_batch_starts[batch_idx]
        size = temp_batch_sizes[batch_idx]

        for j in range(size):
            result_indices[i, j] = sorted_indices[start + j]
        result_sizes[i] = size

    return result_indices, result_sizes


def batch_lengths_to_minibatches_padded(
    batch_lengths: list[int],
    max_tokens_per_rank: int,
    num_ranks: int,
    rank: int,
    max_batch_size: int = 64,
) -> list[list[int]]:
    """Batch packing optimized for padded training (non-flash attention).

    Groups sequences to minimize total padding while maintaining good load
    balance across ranks. Sequences within each batch are padded to the
    length of the longest sequence in that batch.

    Args:
        batch_lengths: List of sequence lengths (in tokens)
        max_tokens_per_rank: Maximum tokens allowed per rank per batch (including padding)
        num_ranks: Total number of distributed training ranks (GPUs)
        rank: The specific rank to retrieve assigned indices for
        max_batch_size: Maximum sequences per batch (default 64)

    Returns:
        List of lists, where each inner list contains indices assigned to this rank
        for one batch. Index -1 indicates padding/placeholder.
    """
    if not batch_lengths:
        return []

    # Convert to numpy
    lengths = np.array(batch_lengths, dtype=np.int64)

    # Call numba-optimized core
    batch_indices, batch_sizes = _padded_batch_packing_core(
        lengths, max_tokens_per_rank, num_ranks, rank, max_batch_size
    )

    # Convert to list format
    result = []
    for i in range(len(batch_sizes)):
        size = batch_sizes[i]
        if batch_indices[i, 0] == -1:
            result.append([-1])
        else:
            result.append(batch_indices[i, :size].tolist())

    return result


def compute_padding_stats(batch_lengths: list[int], batches: list[list[int]]) -> dict:
    """Compute padding statistics for given batches.

    Args:
        batch_lengths: Original sequence lengths
        batches: List of batches (each batch is a list of indices)

    Returns:
        Dictionary with padding statistics
    """
    total_actual_tokens = 0
    total_padded_tokens = 0
    total_sequences = 0

    for batch in batches:
        if not batch or batch[0] == -1:
            continue

        # Find max length in this batch
        max_len = max(batch_lengths[idx] for idx in batch)

        # Compute tokens
        for idx in batch:
            actual_len = batch_lengths[idx]
            total_actual_tokens += actual_len
            total_padded_tokens += max_len
            total_sequences += 1

    padding_ratio = 0.0
    if total_padded_tokens > 0:
        padding_ratio = (
            total_padded_tokens - total_actual_tokens
        ) / total_padded_tokens

    return {
        "total_sequences": total_sequences,
        "total_actual_tokens": total_actual_tokens,
        "total_padded_tokens": total_padded_tokens,
        "total_padding_tokens": total_padded_tokens - total_actual_tokens,
        "padding_ratio": padding_ratio,
        "num_batches": len([b for b in batches if b and b[0] != -1]),
    }


import math
def compute_bucket_size(length: int) -> int:
    """
    Bucketing algorithm based on the most significant bit of the sample length.
    Finds the most significant bit position 'S', then divides the range 
    [2^S, 2^(S+1)] into 16 equal buckets of size 2^(S-4).
    Limits overhead to at most 1/16 while reducing number of graph recompilations.
    """
    msb_pos = length.bit_length()
    alignment = (1 << (msb_pos - 4)) if msb_pos >= 4 else 1
    return math.ceil(length / alignment) * alignment


def _batch_cost(lengths: list[int]) -> float:
    if not (isinstance(lengths, list) and len(lengths) > 0):
        raise TypeError(f"wrong input type")
    return lengths[0] * len(lengths)


def _batch_size_in_tokens(lengths: list[int]) -> float:
    if not (isinstance(lengths, list) and len(lengths) > 0):
        raise TypeError(f"wrong input type")
    return lengths[0] * len(lengths)


def _check_batch_cost(
        sorted_lengths:list[int],
        num_ranks:int64,
        max_cost:float,
    ) -> list[int]:
    if not (isinstance(sorted_lengths, list) and len(sorted_lengths) > 0):
        raise TypeError(f"wrong input type")
    
    bins = [[] for _ in range(num_ranks)]
    current_bin = 0
    
    for sample_length in sorted_lengths:
        while True:
            # try to add to current bin
            bins[current_bin].append(sample_length)
            cost = _batch_cost(bins[current_bin])
            
            # go to next sample if current fits
            if cost < max_cost:
                break

            # bin overflow, move last sample to next bin if possible
            if len(bins[current_bin]) == 1:
                break 
            
            if current_bin >= num_ranks - 1:
                return None
            
            bins[current_bin].pop()
            current_bin += 1

    bin_sizes = [len(bin) for bin in bins]
    return bin_sizes


def _distribute_samples_across_ranks(
        sorted_lengths: list[int],
        max_tokens: int64,
        num_ranks: int64,
        rank: int64,
    ) -> list[int]:

    # compute cost range, from 0 to max possible cost when we put all samples in one bin
    lower_bound = 0
    upper_bound = _batch_cost(sorted_lengths)

    # find optimal distribution based on batch cost
    prev_bin_sizes = None
    epsilon = 1. # cost has same OOM as batch token count
    while upper_bound - lower_bound > epsilon:
        mid = (lower_bound + upper_bound) / 2
        cur_bin_sizes = _check_batch_cost(sorted_lengths, num_ranks, mid)

        if cur_bin_sizes is None:
            lower_bound = mid + epsilon
        else:
            upper_bound = mid
            prev_bin_sizes = cur_bin_sizes

    # sanity check
    if prev_bin_sizes is not None:
        if (len(prev_bin_sizes) != num_ranks or sum(prev_bin_sizes) != len(sorted_lengths)):
            raise ValueError("Something went wrong, we lost samples during distribution across ranks")

        if any(size == 0 for size in prev_bin_sizes):
            if any(size > 1 for size in prev_bin_sizes):
                raise ValueError("Something went wrong, we put more than one sample per rank for small batch")
        
    return prev_bin_sizes


def _check_batch_size_in_tokens(
        sorted_lengths:list[int],
        max_tokens: int64,
        minibatch_sizes:list[int],
    ) -> bool:

    first_sample_idx = 0
    for bs in minibatch_sizes:
        if bs > 0:
            minibatch = sorted_lengths[first_sample_idx:first_sample_idx + bs]
            if _batch_size_in_tokens(minibatch) >= max_tokens:
                return False
            first_sample_idx += bs
    return True
    

def _compute_sample_indeces_for_current_rank(
        lengths: list[int],
        minibatches: list[list[int]],
        rank: int64,
    ) -> list[list[int]]:

    sorted_indices = np.argsort(lengths)[::-1]
    minibatch_indices = []
    first_sample_idx = 0
    for minibatch in minibatches:
        first_sample_idx += sum(minibatch[:rank])
        minibatch_indices.append(sorted_indices[first_sample_idx:first_sample_idx + minibatch[rank]].tolist())
        first_sample_idx += sum(minibatch[rank:])
    return minibatch_indices


def _compute_num_samples_in_grad_accum_step(
        num_samples: int, 
        grad_accum : int
    ) -> list[int]:

    if grad_accum <= 0:
        return []
    if grad_accum == 1:
        return [num_samples]
    
    step_size = num_samples // grad_accum
    remainder = num_samples % grad_accum
    result = [step_size] * grad_accum
    result[-1] += remainder
    return result


def _batch_packing_core_hpu(
    lengths: list[int],
    max_tokens: int64,
    num_ranks: int64,
    rank: int64,
) -> list[list[int]]:

    # try different gradient accumulation values
    for grad_accum in [1, 2, 4]:

        # break input batch to several gradient accumulation steps
        grad_accum_step_sizes = _compute_num_samples_in_grad_accum_step(len(lengths), grad_accum)

        first_sample_idx = 0
        minibatches = []
        for step_size in grad_accum_step_sizes:
            step_lengths = lengths[first_sample_idx:first_sample_idx + step_size]
            first_sample_idx += step_size
            sorted_lengths = sorted(step_lengths, reverse=True)

            # find optimal sample distribution for single step based on computation cost
            minibatch_sizes = _distribute_samples_across_ranks(sorted_lengths, max_tokens, num_ranks, rank)
            if minibatch_sizes is None:
                raise ValueError("Something went wrong")

            # check if found distribution fits in token limit
            if not _check_batch_size_in_tokens(sorted_lengths, max_tokens, minibatch_sizes):
                # does not fit, increase number of gradient accumulation steps
                break
            minibatches.append(minibatch_sizes)

        #check if we found suitable sample distribution
        if len(minibatches) == grad_accum:
            break

    # sanity check
    if not (
        len(minibatches) == grad_accum and
        all(len(minibatch) == num_ranks for minibatch in minibatches) and
        sum(sum(minibatch) for minibatch in minibatches) == len(lengths)
        ):
        raise ValueError("Could not distribute samples across ranks")


    # compute indices for current rank
    minibatch_indices = _compute_sample_indeces_for_current_rank(lengths, minibatches, rank)

    # sanity check
    from itertools import chain
    all_indices = list(chain.from_iterable(minibatch_indices))
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Something went wrong, duplicated indices in the list")

    # add one dummy sample to each empty minibatch
    for minibatch in minibatch_indices:
        if len(minibatch) == 0:
            minibatch.append(-1)

    return minibatch_indices


def batch_lengths_to_minibatches_hpu(
    batch_lengths: list[int],
    max_tokens_per_rank: int,
    num_ranks: int,
    rank: int,
) -> list[list[int]]:

    if not batch_lengths:
        return []
    result = _batch_packing_core_hpu( batch_lengths, max_tokens_per_rank, num_ranks, rank)
    return result
