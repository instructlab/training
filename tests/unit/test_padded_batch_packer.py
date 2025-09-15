# Standard
from unittest.mock import patch
import unittest

# Third Party
import numpy as np

# First Party
from instructlab.training.padded_batch_packer import (
    batch_lengths_to_minibatches_padded,
    compute_padding_stats,
)


class TestPaddedBatchPacker(unittest.TestCase):
    """Unit tests for the padded batch packer algorithm."""

    def test_empty_input(self):
        """Test that empty input returns empty list."""
        result = batch_lengths_to_minibatches_padded([], 100, 2, 0)
        self.assertEqual(result, [])

    def test_single_sequence(self):
        """Test single sequence with single rank."""
        lengths = [50]
        result = batch_lengths_to_minibatches_padded(lengths, 100, 1, 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0])

    def test_all_sequences_assigned(self):
        """Test that all sequences are assigned exactly once across ranks."""
        lengths = [30, 40, 50, 20, 60, 70, 80, 90]
        num_ranks = 4
        max_tokens = 200

        # Collect all assigned indices across ranks
        all_indices = set()
        for rank in range(num_ranks):
            batches = batch_lengths_to_minibatches_padded(
                lengths, max_tokens, num_ranks, rank
            )
            for batch in batches:
                for idx in batch:
                    if idx != -1:
                        all_indices.add(idx)

        # Check all sequences are assigned
        self.assertEqual(sorted(all_indices), list(range(len(lengths))))

    def test_max_tokens_constraint(self):
        """Test that batches respect max tokens constraint."""
        lengths = [80, 60, 40, 90, 50, 70]
        max_tokens = 100
        num_ranks = 2

        for rank in range(num_ranks):
            batches = batch_lengths_to_minibatches_padded(
                lengths, max_tokens, num_ranks, rank
            )

            for batch in batches:
                if not batch or batch[0] == -1:
                    continue

                # Calculate padded tokens for this batch
                max_len = max(lengths[idx] for idx in batch)
                padded_tokens = max_len * len(batch)

                self.assertLessEqual(
                    padded_tokens,
                    max_tokens,
                    f"Batch exceeds max tokens: {padded_tokens} > {max_tokens}",
                )

    def test_padding_efficiency_similar_lengths(self):
        """Test that similar-length sequences have minimal padding."""
        # Sequences with similar lengths
        lengths = [98, 95, 97, 96, 94, 99, 93, 96]
        batches = batch_lengths_to_minibatches_padded(lengths, 400, 1, 0)
        stats = compute_padding_stats(lengths, batches)

        # Should have very low padding ratio
        self.assertLess(
            stats["padding_ratio"],
            0.1,
            f"Similar lengths should have low padding, got {stats['padding_ratio']:.2%}",
        )

    def test_same_length_zero_padding(self):
        """Test that same-length sequences have zero padding."""
        lengths = [50] * 10
        batches = batch_lengths_to_minibatches_padded(lengths, 200, 2, 0)
        stats = compute_padding_stats(lengths, batches)

        self.assertEqual(stats["padding_ratio"], 0.0)

    def test_load_balancing(self):
        """Test that load is reasonably balanced across ranks."""
        lengths = list(np.random.randint(20, 200, size=50))
        num_ranks = 4
        max_tokens = 500

        rank_tokens = {}
        for rank in range(num_ranks):
            batches = batch_lengths_to_minibatches_padded(
                lengths, max_tokens, num_ranks, rank
            )
            stats = compute_padding_stats(lengths, batches)
            rank_tokens[rank] = stats["total_padded_tokens"]

        # Check load balance
        if max(rank_tokens.values()) > 0:
            max_load = max(rank_tokens.values())
            min_load = min(rank_tokens.values())
            imbalance = (max_load - min_load) / max_load

            # Allow up to 50% imbalance for small test cases
            self.assertLess(
                imbalance,
                0.5,
                f"Load imbalance too high: {imbalance:.2%}",
            )

    def test_accumulation_steps(self):
        """Test batch packing with gradient accumulation steps."""
        # Simulate dataset split across accumulation steps
        total_lengths = list(np.random.randint(20, 100, size=100))
        num_ranks = 4
        accumulation_steps = 4
        max_tokens = 200

        samples_per_step = len(total_lengths) // accumulation_steps
        all_processed = set()

        for step in range(accumulation_steps):
            start = step * samples_per_step
            end = min((step + 1) * samples_per_step, len(total_lengths))
            step_lengths = total_lengths[start:end]

            for rank in range(num_ranks):
                batches = batch_lengths_to_minibatches_padded(
                    step_lengths, max_tokens, num_ranks, rank
                )

                for batch in batches:
                    for idx in batch:
                        if idx != -1:
                            global_idx = start + idx
                            all_processed.add(global_idx)

        # Verify all samples in the steps were processed
        expected = set(range(accumulation_steps * samples_per_step))
        self.assertEqual(all_processed, expected)

    def test_deterministic_output(self):
        """Test that the algorithm produces deterministic results."""
        lengths = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        max_tokens = 200
        num_ranks = 2

        # Run multiple times and verify same output
        results = []
        for _ in range(3):
            rank_results = {}
            for rank in range(num_ranks):
                batches = batch_lengths_to_minibatches_padded(
                    lengths, max_tokens, num_ranks, rank
                )
                rank_results[rank] = batches
            results.append(rank_results)

        # Check all runs produce same results
        for i in range(1, len(results)):
            for rank in range(num_ranks):
                self.assertEqual(
                    results[0][rank],
                    results[i][rank],
                    f"Non-deterministic output for rank {rank}",
                )

    def test_edge_case_single_long_sequence(self):
        """Test handling of sequences longer than max_tokens."""
        lengths = [1000]  # Much longer than typical max_tokens
        max_tokens = 100

        batches = batch_lengths_to_minibatches_padded(lengths, max_tokens, 1, 0)

        # Should still process the sequence even if it exceeds max_tokens
        all_indices = []
        for batch in batches:
            all_indices.extend([idx for idx in batch if idx != -1])

        # The long sequence should be included
        self.assertIn(0, all_indices)

    def test_max_batch_size_constraint(self):
        """Test that max_batch_size parameter is respected."""
        lengths = [10] * 20  # Many small sequences
        max_tokens = 1000  # Large enough to fit many sequences
        max_batch_size = 5

        batches = batch_lengths_to_minibatches_padded(
            lengths, max_tokens, 1, 0, max_batch_size=max_batch_size
        )

        for batch in batches:
            if batch and batch[0] != -1:
                self.assertLessEqual(
                    len(batch),
                    max_batch_size,
                    f"Batch size {len(batch)} exceeds max {max_batch_size}",
                )

    def test_padding_stats_computation(self):
        """Test padding statistics computation."""
        lengths = [100, 90, 80, 70]
        # Manually create batches for testing
        batches = [[0, 1], [2, 3]]  # Two batches of 2 sequences each

        stats = compute_padding_stats(lengths, batches)

        # Batch 1: max=100, sequences=100+90=190, padded=200
        # Batch 2: max=80, sequences=80+70=150, padded=160
        # Total: actual=340, padded=360, padding=20

        self.assertEqual(stats["total_sequences"], 4)
        self.assertEqual(stats["total_actual_tokens"], 340)
        self.assertEqual(stats["total_padded_tokens"], 360)
        self.assertEqual(stats["total_padding_tokens"], 20)
        self.assertAlmostEqual(stats["padding_ratio"], 20 / 360, places=4)
        self.assertEqual(stats["num_batches"], 2)


if __name__ == "__main__":
    unittest.main()
