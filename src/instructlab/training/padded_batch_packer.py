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
