# Standard
from unittest.mock import MagicMock, patch
import unittest

# Third Party
import numpy as np
import torch

# First Party
from instructlab.training.batch_packer import batch_lengths_to_minibatches_lpt
from instructlab.training.padded_batch_packer import batch_lengths_to_minibatches_padded
from instructlab.training.sampler import MaxTokensPerRankCollator


class TestCollators(unittest.TestCase):
    """Comprehensive tests for the unified MaxTokensPerRankCollator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock for distributed operations when not in distributed mode
        self.dist_patcher = patch("instructlab.training.sampler.dist")
        self.mock_dist = self.dist_patcher.start()
        self.mock_dist.get_rank.return_value = 0
        self.mock_dist.get_world_size.return_value = 1

    def tearDown(self):
        """Clean up patches."""
        self.dist_patcher.stop()

    def create_test_batch(self, num_samples=10, min_len=50, max_len=200):
        """Create a test batch with varying sequence lengths."""
        batch = []
        np.random.seed(42)  # For reproducibility
        for i in range(num_samples):
            seq_len = np.random.randint(min_len, max_len)
            input_ids = torch.randint(1, 1000, (seq_len,))
            labels = torch.where(
                torch.rand(seq_len) > 0.3, input_ids, torch.tensor(-100)
            )
            num_loss_counted = (labels != -100).sum().item()

            batch.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "len": seq_len,
                    "num_loss_counted_tokens": num_loss_counted,
                }
            )

        return batch

    def test_flash_mode_output_format(self):
        """Test that flash mode produces the expected output format."""
        batch = self.create_test_batch(20)

        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=1024,
            rank=0,
            world_size=1,
            flash_enabled=True,
        )

        minibatches = collator(batch)

        self.assertGreater(len(minibatches), 0, "Should produce at least one minibatch")

        for mb in minibatches:
            # Check required keys for flash mode
            self.assertIn("input_ids", mb)
            self.assertIn("labels", mb)
            self.assertIn("position_ids", mb)
            self.assertIn("num_loss_counted_tokens", mb)
            self.assertIn("num_samples", mb)
            self.assertIn("total_length", mb)

            # Should NOT have attention_mask in flash mode
            self.assertNotIn("attention_mask", mb)

            # Check tensor shapes (flash mode uses concatenated format)
            self.assertEqual(mb["input_ids"].dim(), 2)  # [1, total_length]
            self.assertEqual(mb["input_ids"].shape[0], 1)
            self.assertEqual(mb["labels"].shape, mb["input_ids"].shape)
            self.assertEqual(mb["position_ids"].shape, mb["input_ids"].shape)

    def test_padded_mode_output_format(self):
        """Test that padded mode produces the expected output format."""
        batch = self.create_test_batch(20)

        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=1024,
            rank=0,
            world_size=1,
            flash_enabled=False,
            pad_token_id=0,
        )

        minibatches = collator(batch)

        self.assertGreater(len(minibatches), 0, "Should produce at least one minibatch")

        for mb in minibatches:
            # Check required keys for padded mode
            self.assertIn("input_ids", mb)
            self.assertIn("labels", mb)
            self.assertIn("attention_mask", mb)
            self.assertIn("num_loss_counted_tokens", mb)
            self.assertIn("num_samples", mb)
            self.assertIn("total_length", mb)

            # Should NOT have position_ids in padded mode
            self.assertNotIn("position_ids", mb)

            # Check tensor shapes (padded mode uses [batch_size, max_len])
            self.assertEqual(mb["input_ids"].dim(), 2)
            self.assertEqual(mb["labels"].shape, mb["input_ids"].shape)
            self.assertEqual(mb["attention_mask"].shape, mb["input_ids"].shape)

            # Verify attention mask correctness
            # Attention mask should be 1 for non-padding, 0 for padding
            batch_size = mb["input_ids"].shape[0]
            for i in range(batch_size):
                # Find non-padding positions (assuming pad_token_id=0)
                non_pad_mask = mb["input_ids"][i] != 0
                # Attention mask should match non-padding positions
                self.assertTrue(
                    torch.equal(mb["attention_mask"][i], non_pad_mask.long()),
                    "Attention mask should be 1 for non-padding tokens",
                )

    def test_multi_rank_distribution(self):
        """Test that samples are correctly distributed across multiple ranks."""
        # Create batch with known content for tracking
        batch_size = 20
        batch = []
        for i in range(batch_size):
            seq_len = 100 + i * 5  # Varying lengths
            # Use a unique pattern for each sample to track it
            input_ids = torch.full((seq_len,), i + 1000, dtype=torch.long)
            labels = torch.full((seq_len,), i + 2000, dtype=torch.long)

            batch.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "len": seq_len,
                    "num_loss_counted_tokens": seq_len,
                }
            )

        world_size = 4
        self.mock_dist.get_world_size.return_value = world_size

        # Track which samples each rank receives
        rank_sample_ids = {rank: set() for rank in range(world_size)}
        total_samples_per_rank = {rank: 0 for rank in range(world_size)}

        for rank in range(world_size):
            self.mock_dist.get_rank.return_value = rank

            # We need to use the actual batch packer to track indices
            batch_lengths = [b["len"] for b in batch]
            indices = batch_lengths_to_minibatches_lpt(
                batch_lengths, 1024, world_size, rank
            )

            # Track which sample indices this rank got
            for minibatch_indices in indices:
                for idx in minibatch_indices:
                    if idx != -1:  # Not a dummy sample
                        rank_sample_ids[rank].add(idx)
                        total_samples_per_rank[rank] += 1

        # Verify no sample appears on multiple ranks
        all_assigned_samples = set()
        for rank, samples in rank_sample_ids.items():
            for sample_id in samples:
                self.assertNotIn(
                    sample_id,
                    all_assigned_samples,
                    f"Sample {sample_id} assigned to multiple ranks!",
                )
                all_assigned_samples.add(sample_id)

        # Verify all samples are assigned exactly once
        self.assertEqual(
            len(all_assigned_samples),
            len(batch),
            "All samples should be assigned exactly once across all ranks",
        )

        # Verify reasonable load balance
        samples_counts = list(total_samples_per_rank.values())
        max_samples = max(samples_counts)
        min_samples = min(samples_counts)

        # Allow some imbalance but not too much
        self.assertLessEqual(
            max_samples - min_samples,
            len(batch) // 2,
            f"Load imbalance too high: max={max_samples}, min={min_samples}",
        )

        print(f"\nSample distribution across ranks: {total_samples_per_rank}")

    def test_padded_multi_rank_distribution(self):
        """Test padded mode also distributes samples correctly across ranks."""
        # Similar test but for padded mode
        batch_size = 20
        batch = []
        for i in range(batch_size):
            seq_len = 100 + i * 5
            input_ids = torch.full((seq_len,), i + 1000, dtype=torch.long)
            labels = torch.full((seq_len,), i + 2000, dtype=torch.long)

            batch.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "len": seq_len,
                    "num_loss_counted_tokens": seq_len,
                }
            )

        world_size = 4
        self.mock_dist.get_world_size.return_value = world_size

        rank_sample_ids = {rank: set() for rank in range(world_size)}

        for rank in range(world_size):
            self.mock_dist.get_rank.return_value = rank

            batch_lengths = [b["len"] for b in batch]
            indices = batch_lengths_to_minibatches_padded(
                batch_lengths, 1024, world_size, rank
            )

            for minibatch_indices in indices:
                for idx in minibatch_indices:
                    if idx != -1:
                        rank_sample_ids[rank].add(idx)

        # Verify no duplicates
        all_assigned = set()
        for rank, samples in rank_sample_ids.items():
            for sample_id in samples:
                self.assertNotIn(
                    sample_id,
                    all_assigned,
                    f"Sample {sample_id} assigned to multiple ranks in padded mode!",
                )
                all_assigned.add(sample_id)

        # All samples should be assigned
        self.assertEqual(
            len(all_assigned),
            len(batch),
            "All samples should be assigned in padded mode",
        )

    def test_accumulation_behavior(self):
        """Test that batches accumulate properly across multiple calls."""
        # Create multiple batches
        batch1 = self.create_test_batch(10, min_len=50, max_len=100)
        batch2 = self.create_test_batch(10, min_len=100, max_len=150)
        batch3 = self.create_test_batch(10, min_len=150, max_len=200)

        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=512,  # Smaller to force multiple minibatches
            rank=0,
            world_size=1,
            flash_enabled=True,
        )

        # Process batches
        minibatches1 = collator(batch1)
        minibatches2 = collator(batch2)
        minibatches3 = collator(batch3)

        # Each batch should be processed independently
        self.assertGreater(len(minibatches1), 0)
        self.assertGreater(len(minibatches2), 0)
        self.assertGreater(len(minibatches3), 0)

        # Verify that longer sequences produce more minibatches
        # (due to token limit constraints)
        total_tokens1 = sum(mb["total_length"] for mb in minibatches1)
        total_tokens3 = sum(mb["total_length"] for mb in minibatches3)

        # Batch 3 has longer sequences, so should have more total tokens
        self.assertGreater(
            total_tokens3,
            total_tokens1,
            "Batches with longer sequences should have more total tokens",
        )

    def test_max_tokens_constraint(self):
        """Test that max_tokens_per_rank constraint is respected."""
        batch = self.create_test_batch(20, min_len=100, max_len=200)
        max_tokens = 256  # Small limit to test constraint

        for flash_enabled in [True, False]:
            with self.subTest(flash_enabled=flash_enabled):
                collator = MaxTokensPerRankCollator(
                    max_tokens_per_rank=max_tokens,
                    rank=0,
                    world_size=1,
                    flash_enabled=flash_enabled,
                )

                minibatches = collator(batch)

                for mb in minibatches:
                    if flash_enabled:
                        # In flash mode, check total concatenated length
                        self.assertLessEqual(
                            mb["total_length"],
                            max_tokens,
                            f"Minibatch exceeds max tokens: {mb['total_length']} > {max_tokens}",
                        )
                    else:
                        # In padded mode, check max_len * batch_size
                        batch_size, max_len = mb["input_ids"].shape
                        padded_tokens = batch_size * max_len
                        # Note: padded mode might slightly exceed due to padding
                        # but should be close
                        self.assertLessEqual(
                            padded_tokens,
                            max_tokens * 1.5,  # Allow some padding overhead
                            f"Padded batch significantly exceeds max tokens",
                        )

    def test_filtering_long_sequences(self):
        """Test that sequences longer than max_tokens are filtered."""
        # Create batch with some very long sequences
        batch = []
        for i in range(10):
            if i < 5:
                seq_len = 100  # Normal length
            else:
                seq_len = 2000  # Very long

            batch.append(
                {
                    "input_ids": torch.randint(1, 1000, (seq_len,)),
                    "labels": torch.randint(0, 1000, (seq_len,)),
                    "len": seq_len,
                    "num_loss_counted_tokens": seq_len // 2,
                }
            )

        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=1024,
            rank=0,
            world_size=1,
            flash_enabled=True,
        )

        # Capture print output to verify filtering message
        # Standard
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        minibatches = collator(batch)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Should have printed a warning about filtered samples
        self.assertIn("removed", output)
        self.assertIn("5 samples", output)  # Should filter 5 long sequences

    def test_dummy_sample_handling(self):
        """Test that dummy samples are used correctly for padding."""
        batch = self.create_test_batch(5)
        world_size = 4  # More ranks than samples to force dummy usage

        self.mock_dist.get_world_size.return_value = world_size

        # Check a rank that might not get real samples
        self.mock_dist.get_rank.return_value = 3

        custom_dummy = {
            "input_ids": torch.tensor([999, 998, 997], dtype=torch.long),
            "labels": torch.tensor([-100, -100, -100], dtype=torch.long),
            "len": 3,
            "num_loss_counted_tokens": 0,
        }

        collator = MaxTokensPerRankCollator(
            max_tokens_per_rank=1024,
            rank=3,
            world_size=world_size,
            dummy_sample=custom_dummy,
            flash_enabled=True,
        )

        minibatches = collator(batch)

        # Should have at least one minibatch (possibly with dummy)
        self.assertGreater(len(minibatches), 0)

        # Check if dummy values appear in output
        for mb in minibatches:
            # If this rank got a dummy sample, it should have 0 loss tokens
            if mb["num_samples"] == 0:
                self.assertEqual(
                    mb["num_loss_counted_tokens"],
                    0,
                    "Dummy samples should not contribute to loss",
                )

    def test_mode_switching(self):
        """Test that the same collator can switch between modes correctly."""
        batch = self.create_test_batch(10)

        # Test flash mode
        collator_flash = MaxTokensPerRankCollator(
            max_tokens_per_rank=1024,
            rank=0,
            world_size=1,
            flash_enabled=True,
        )

        # Test padded mode
        collator_padded = MaxTokensPerRankCollator(
            max_tokens_per_rank=1024,
            rank=0,
            world_size=1,
            flash_enabled=False,
            pad_token_id=0,
        )

        mb_flash = collator_flash(batch)
        mb_padded = collator_padded(batch)

        # Verify different output formats
        self.assertIn("position_ids", mb_flash[0])
        self.assertNotIn("attention_mask", mb_flash[0])

        self.assertNotIn("position_ids", mb_padded[0])
        self.assertIn("attention_mask", mb_padded[0])


if __name__ == "__main__":
    unittest.main()
