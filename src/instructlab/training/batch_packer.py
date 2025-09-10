"""
Numba-optimized batch packing using LPT (Longest Processing Time) algorithm.

This module provides high-performance batch packing for distributed training,
using JIT compilation for optimal speed while maintaining superior load balancing.
"""

# Third Party
from numba import int64, njit
import numpy as np


@njit
def _lpt_check_heap(
    heap: np.ndarray, lengths: np.ndarray, max_tokens: int64, n: int64
) -> bool:
    """Check if sequences can fit using min-heap for O(log n) insertions.

    Uses a binary min-heap where heap[1] is the root (heap[0] unused).
    """
    # Sort lengths in descending order (longest first)
    sorted_lengths = np.sort(lengths)[::-1]
    heap[:] = 0  # Reset heap

    for size in sorted_lengths:
        # Add to smallest element (root of min-heap)
        heap[1] += size
        if heap[1] > max_tokens:
            return False

        # Heapify down (sink operation)
        u = 1
        while (u << 1) <= n:  # While node has at least one child
            v = u << 1  # Left child
            rch = (u << 1) | 1  # Right child

            # Find smallest child
            if rch <= n and heap[rch] < heap[v]:
                v = rch

            # If parent is smaller than smallest child, we're done
            if heap[u] <= heap[v]:
                break

            # Swap with smallest child
            heap[u], heap[v] = heap[v], heap[u]
            u = v

    return True


@njit
def _lpt_distribute_heap(
    heap: np.ndarray,
    heap_id: np.ndarray,
    lengths: np.ndarray,
    indices: np.ndarray,
    n: int64,
    rank: int64,
) -> np.ndarray:
    """Distribute sequences using min-heap for efficient LPT scheduling.

    Returns indices assigned to the specified rank.
    """
    # Sort by length descending, keeping track of original indices
    sort_idx = np.argsort(lengths)[::-1]
    sorted_lengths = lengths[sort_idx]
    sorted_indices = indices[sort_idx]

    # Initialize heap and rank assignments
    heap[:] = 0
    heap_id[:] = np.arange(-1, n, dtype=np.int64)

    # Pre-allocate result array (worst case: all sequences to one rank)
    result = np.empty(len(lengths), dtype=np.int64)
    result_count = 0

    for i in range(len(sorted_lengths)):
        # Add to smallest load (root of min-heap)
        heap[1] += sorted_lengths[i]

        # If this is our rank, add the index
        if heap_id[1] == rank:
            result[result_count] = sorted_indices[i]
            result_count += 1

        # Heapify down
        u = 1
        while (u << 1) <= n:
            v = u << 1  # Left child
            rch = (u << 1) | 1  # Right child

            # Find smallest child
            if rch <= n and heap[rch] < heap[v]:
                v = rch

            # If parent is smaller, we're done
            if heap[u] <= heap[v]:
                break

            # Swap values and IDs
            heap[u], heap[v] = heap[v], heap[u]
            heap_id[u], heap_id[v] = heap_id[v], heap_id[u]
            u = v

    # Return only the filled portion
    return result[:result_count]


@njit
def _batch_to_minibatches_lpt_core(
    lengths: np.ndarray, max_tokens: int64, num_ranks: int64, rank: int64
) -> tuple:
    """Core numba-optimized LPT batching algorithm.

    Returns:
        minibatch_indices: 2D array of indices for this rank
        minibatch_sizes: number of sequences in each minibatch
    """
    n_sequences = len(lengths)
    if n_sequences == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)

    # Get sorted indices (longest first)
    sorted_indices = np.argsort(lengths)[::-1]
    sorted_lengths = lengths[sorted_indices]

    # Calculate cumulative sum for efficient range queries
    lengths_cumsum = np.cumsum(sorted_lengths)

    # Pre-allocate heap (1-indexed, so size n+1)
    heap = np.zeros(num_ranks + 1, dtype=np.int64)
    heap_id = np.zeros(num_ranks + 1, dtype=np.int64)

    # Pre-allocate output arrays
    max_minibatches = n_sequences  # Worst case
    minibatch_indices = np.full((max_minibatches, n_sequences), -1, dtype=np.int64)
    minibatch_sizes = np.zeros(max_minibatches, dtype=np.int64)
    n_minibatches = 0

    start_idx = 0
    s = 0  # Current cumulative sum

    while start_idx < n_sequences:
        # Binary search for maximum sequences that fit
        # Search up to the point where total tokens would exceed capacity
        remaining = n_sequences - start_idx

        # Find upper bound: sequences whose cumsum doesn't exceed current + capacity
        if start_idx > 0:
            s = lengths_cumsum[start_idx - 1]
        else:
            s = 0

        # Binary search in the remaining sequences
        left = 1
        right = min(
            remaining + 1,
            1
            + np.searchsorted(
                lengths_cumsum[start_idx:], s + max_tokens * num_ranks, "right"
            ),
        )

        while right - left > 1:
            mid = (left + right) // 2
            if _lpt_check_heap(
                heap, sorted_lengths[start_idx : start_idx + mid], max_tokens, num_ranks
            ):
                left = mid
            else:
                right = mid

        end_idx = start_idx + left

        # Distribute this batch using LPT
        batch_indices = sorted_indices[start_idx:end_idx]
        batch_lengths = sorted_lengths[start_idx:end_idx]

        my_indices = _lpt_distribute_heap(
            heap, heap_id, batch_lengths, batch_indices, num_ranks, rank
        )

        # Store result
        if len(my_indices) > 0:
            minibatch_indices[n_minibatches, : len(my_indices)] = my_indices
            minibatch_sizes[n_minibatches] = len(my_indices)
        else:
            # Empty minibatch for this rank - use padding
            minibatch_indices[n_minibatches, 0] = -1
            minibatch_sizes[n_minibatches] = 1

        n_minibatches += 1
        start_idx = end_idx

    # Return trimmed arrays
    return (minibatch_indices[:n_minibatches], minibatch_sizes[:n_minibatches])


def batch_lengths_to_minibatches_lpt(
    batch_lengths: list[int], max_tokens_per_rank: int, num_ranks: int, rank: int
):
    """High-performance LPT batch packing using numba optimization.

    Distributes sequences across ranks using the Longest Processing Time (LPT)
    algorithm with min-heap optimization for O(n log n log k) complexity.

    This provides optimal load balancing while being significantly faster than
    naive implementations through JIT compilation.

    Args:
        batch_lengths: List of sequence lengths (in tokens)
        max_tokens_per_rank: Maximum tokens allowed per rank per minibatch
        num_ranks: Total number of distributed training ranks (GPUs)
        rank: The specific rank to retrieve assigned indices for

    Returns:
        List of lists, where each inner list contains indices assigned to this rank
        for one minibatch. Index -1 indicates padding.
    """
    if not batch_lengths:
        return []

    # Convert to numpy
    lengths = np.array(batch_lengths, dtype=np.int64)

    # Call numba-optimized core
    minibatch_indices, minibatch_sizes = _batch_to_minibatches_lpt_core(
        lengths, max_tokens_per_rank, num_ranks, rank
    )

    # Convert back to list format
    result = []
    for i in range(len(minibatch_sizes)):
        size = minibatch_sizes[i]
        if minibatch_indices[i, 0] == -1:
            result.append([-1])
        else:
            result.append(minibatch_indices[i, :size].tolist())

    return result
