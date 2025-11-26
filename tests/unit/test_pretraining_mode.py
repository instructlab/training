# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from unittest.mock import patch
import json
import os
import tempfile
import unittest

# Third Party
from datasets import Dataset as HFDataset
import torch

# First Party
from instructlab.training.config import PretrainingConfig
from instructlab.training.data_process import process_documents_for_pretraining
from instructlab.training.sampler import (
    PretrainingBlockDataset,
    get_data_loader,
)


class TestPretrainingBlockDataset(unittest.TestCase):
    """Tests for the PretrainingBlockDataset behavior."""

    def test_blocks_are_padded_and_loss_counts_tracked(self):
        dataset = HFDataset.from_dict({"input_ids": [[1, 2, 3], [4, 5, 6, 7]]})
        block_ds = PretrainingBlockDataset(dataset, block_size=4, pad_token_id=0)

        self.assertEqual(len(block_ds), 2)

        first = block_ds[0]
        self.assertTrue(
            torch.equal(
                first["input_ids"], torch.tensor([1, 2, 3, 4], dtype=torch.long)
            )
        )
        self.assertTrue(
            torch.equal(first["labels"], torch.tensor([1, 2, 3, 4], dtype=torch.long))
        )
        self.assertEqual(first["num_loss_counted_tokens"], 3)
        self.assertEqual(first["len"], 4)

        second = block_ds[1]
        self.assertTrue(
            torch.equal(
                second["input_ids"], torch.tensor([5, 6, 7, 0], dtype=torch.long)
            )
        )
        self.assertTrue(
            torch.equal(
                second["labels"], torch.tensor([5, 6, 7, -100], dtype=torch.long)
            )
        )
        self.assertEqual(second["num_loss_counted_tokens"], 2)
        self.assertEqual(second["len"], 4)

        lengths = block_ds.get_lengths()
        self.assertEqual(lengths.tolist(), [4, 3])


class TestPretrainingDataLoader(unittest.TestCase):
    """Tests for the pretraining-specific data loader integration."""

    def test_pretraining_loader_returns_packed_batches(self):
        cfg = PretrainingConfig(block_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            records = [
                {"input_ids": [1, 2, 3, 4]},
                {"input_ids": [5, 6, 7, 8]},
            ]
            with data_path.open("w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")

            loader = get_data_loader(
                data_path=str(data_path),
                batch_size=2,
                max_tokens_per_gpu=8,
                seed=42,
                rank=0,
                world_size=1,
                pretraining_config=cfg,
                pad_token_id=0,
            )

            self.assertIsInstance(loader.dataset, PretrainingBlockDataset)
            self.assertEqual(len(loader.dataset), 2)

            step = next(iter(loader))
            self.assertIsInstance(step, list)
            self.assertEqual(len(step), 1)

            microbatch = step[0]
            self.assertIn("input_ids", microbatch)
            self.assertTrue(torch.is_tensor(microbatch["input_ids"]))
            self.assertEqual(microbatch["input_ids"].shape, (1, 8))
            self.assertEqual(microbatch["num_samples"], 2)
            self.assertEqual(microbatch["num_loss_counted_tokens"], 6)
            self.assertEqual(microbatch["batch_num_loss_counted_tokens"], 6)


class TestPretrainingDataProcessing(unittest.TestCase):
    """Tests for the pretraining data processing pipeline."""

    def test_process_documents_for_pretraining_outputs_expected_fields(self):
        class StubTokenizer:
            eos_token_id = 999

            def encode(self, text, add_special_tokens=True):
                base = [ord(ch) % 50 + 1 for ch in text]
                return base if add_special_tokens else base[1:]

        documents = [
            {"document": "alpha"},
            {"document": "beta"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "raw.jsonl"
            with data_path.open("w", encoding="utf-8") as fh:
                for record in documents:
                    fh.write(json.dumps(record) + "\n")

            output_dir = Path(tmpdir) / "processed"

            with patch(
                "instructlab.training.data_process.AutoTokenizer.from_pretrained",
                return_value=StubTokenizer(),
            ) as mock_auto:
                process_documents_for_pretraining(
                    data_path=str(data_path),
                    data_output_path=str(output_dir),
                    model_path="stub-model",
                    num_cpu_procs=1,
                )

                mock_auto.assert_called_once_with("stub-model")

            output_file = output_dir / "data.jsonl"
            self.assertTrue(output_file.exists())

            with output_file.open("r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]

            self.assertEqual(len(rows), len(documents))
            for row in rows:
                self.assertIn("input_ids", row)
                self.assertIn("len", row)
                self.assertIsInstance(row["input_ids"], list)
                self.assertIsInstance(row["len"], int)
                self.assertEqual(len(row["input_ids"]), row["len"])
                self.assertEqual(row["input_ids"][-1], StubTokenizer.eos_token_id)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
