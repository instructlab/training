# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pretraining sampler functionality."""

# Standard
from unittest.mock import MagicMock, patch
import json

# Third Party
import pytest
import torch

# First Party
from instructlab.training.config import PretrainingConfig
from instructlab.training.sampler import PretrainingBlockDataset, get_data_loader


class TestPretrainingBlockDataset:
    """Test suite for PretrainingBlockDataset class."""

    @pytest.fixture
    def sample_pretraining_data(self):
        """Sample tokenized data (14 total tokens)."""
        return [
            {"input_ids": [1, 2, 3, 4, 5], "len": 5},
            {"input_ids": [6, 7, 8, 9, 10, 11], "len": 6},
            {"input_ids": [12, 13, 14], "len": 3},
        ]

    @pytest.fixture
    def mock_hf_dataset(self, sample_pretraining_data):
        """Mock HuggingFace dataset."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: len(sample_pretraining_data)
        mock_ds.__iter__ = lambda self: iter(sample_pretraining_data)
        return mock_ds

    def test_dataset_initialization(self, mock_hf_dataset):
        """Test basic initialization of PretrainingBlockDataset."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        # Verify basic attributes
        assert dataset.block_size == 5
        assert dataset.pad_token_id == 0
        assert dataset.num_blocks == 3  # 14 tokens / 5 = 2 complete + 1 partial
        assert dataset.last_block_len == 4  # 14 % 5 = 4
        assert len(dataset.all_input_ids) == 14  # All tokens concatenated

    def test_concatenation_of_documents(self, mock_hf_dataset):
        """Verify documents are concatenated in the correct order."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        # Check concatenation order
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        assert dataset.all_input_ids == expected

    def test_num_blocks_calculation_with_partial(self, mock_hf_dataset):
        """Test num_blocks calculation with partial block."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        # 14 tokens / 5 = 2 complete + 1 partial
        assert dataset.num_blocks == 3
        assert dataset.last_block_len == 4

    def test_num_blocks_calculation_exact_multiple(self, sample_pretraining_data):
        """Test num_blocks calculation when tokens exactly divide by block_size."""
        # Add one more token to make it 15 (exact multiple of 5)
        data = sample_pretraining_data + [{"input_ids": [15], "len": 1}]

        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: len(data)
        mock_ds.__iter__ = lambda self: iter(data)

        dataset = PretrainingBlockDataset(dataset=mock_ds, block_size=5, pad_token_id=0)

        # 15 tokens / 5 = 3 complete blocks
        assert dataset.num_blocks == 3
        assert dataset.last_block_len == 5  # Last block is complete

    def test_getitem_complete_block(self, mock_hf_dataset):
        """Test __getitem__ for a complete block."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        # Get first block (indices 0-4)
        block = dataset[0]

        assert block["input_ids"].shape == (5,)
        assert block["labels"].shape == (5,)
        assert block["len"] == 5
        assert block["num_loss_counted_tokens"] == 4  # block_size - 1 (causal shift)

        # Check actual token values
        assert torch.equal(
            block["input_ids"], torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        )
        assert torch.equal(
            block["labels"], torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        )

    def test_getitem_partial_block_with_padding(self, mock_hf_dataset):
        """Test __getitem__ for partial last block with padding."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        # Get last block (index 2) - should have 4 real tokens + 1 padding
        block = dataset[2]

        assert block["input_ids"].shape == (5,)
        assert block["labels"].shape == (5,)
        assert block["len"] == 5

        # 4 real tokens - 1 for causal shift = 3
        assert block["num_loss_counted_tokens"] == 3

        # Last token should be pad_token_id (0)
        assert block["input_ids"][-1].item() == 0

        # Last label should be masked (-100)
        assert block["labels"][-1].item() == -100

        # First 4 tokens should be real data [11, 12, 13, 14] from position 10-13
        assert block["input_ids"][0].item() == 11
        assert block["input_ids"][1].item() == 12
        assert block["input_ids"][2].item() == 13
        assert block["input_ids"][3].item() == 14

    def test_labels_are_copy_not_reference(self, mock_hf_dataset):
        """Test that labels are a copy, not a reference to input_ids."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        block = dataset[0]

        # Tensors should not be the same object
        assert block["input_ids"] is not block["labels"]

        # But values should be equal for complete blocks
        assert torch.equal(block["input_ids"], block["labels"])

        # Modify labels to verify they're independent
        original_labels = block["labels"].clone()
        block["labels"][0] = 999

        # input_ids should remain unchanged
        assert block["input_ids"][0].item() != 999
        assert block["input_ids"][0].item() == 1

    def test_num_loss_counted_tokens_complete_block(self):
        """Test num_loss_counted_tokens for complete blocks with various block sizes."""
        for block_size in [5, 10, 20]:
            # Create data with at least 2 complete blocks
            num_tokens = block_size * 2
            data = [{"input_ids": list(range(num_tokens)), "len": num_tokens}]

            mock_ds = MagicMock()
            mock_ds.column_names = ["input_ids", "len"]
            mock_ds.__len__ = lambda self: len(data)
            mock_ds.__iter__ = lambda self: iter(data)

            dataset = PretrainingBlockDataset(
                dataset=mock_ds, block_size=block_size, pad_token_id=0
            )

            # Check first complete block
            block = dataset[0]
            assert block["num_loss_counted_tokens"] == block_size - 1

    def test_num_loss_counted_tokens_partial_block(self, mock_hf_dataset):
        """Test num_loss_counted_tokens for partial blocks."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        # Last block has 4 real tokens
        block = dataset[2]

        # Should be actual_length - 1 = 4 - 1 = 3
        assert block["num_loss_counted_tokens"] == 3

    def test_index_out_of_range(self, mock_hf_dataset):
        """Test that accessing beyond num_blocks raises IndexError."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        # Try to access block beyond num_blocks (which is 3)
        with pytest.raises(IndexError) as exc_info:
            _ = dataset[3]

        assert "out of range" in str(exc_info.value).lower()

    def test_missing_input_ids_field_raises_error(self):
        """Test that missing input_ids field raises ValueError."""
        # Create dataset without input_ids field
        mock_ds = MagicMock()
        mock_ds.column_names = ["len"]  # Missing input_ids

        with pytest.raises(ValueError) as exc_info:
            _ = PretrainingBlockDataset(dataset=mock_ds, block_size=5, pad_token_id=0)

        assert "input_ids" in str(exc_info.value)

    def test_tensor_dtype_correct(self, mock_hf_dataset):
        """Test that all tensors use torch.long dtype."""
        dataset = PretrainingBlockDataset(
            dataset=mock_hf_dataset, block_size=5, pad_token_id=0
        )

        block = dataset[0]

        assert block["input_ids"].dtype == torch.long
        assert block["labels"].dtype == torch.long


class TestGetDataLoaderPretraining:
    """Test suite for get_data_loader with pretraining mode."""

    @pytest.fixture
    def temp_pretraining_file(self, tmp_path):
        """Create temp JSONL with pretraining data."""
        data_file = tmp_path / "pretraining_data.jsonl"
        samples = [
            {"input_ids": list(range(100, 150)), "len": 50},
            {"input_ids": list(range(200, 280)), "len": 80},
            {"input_ids": list(range(300, 370)), "len": 70},
        ]

        with open(data_file, "w") as f:
            for sample in samples:
                json.dump(sample, f)
                f.write("\n")

        return str(data_file)

    @patch("instructlab.training.sampler.load_dataset")
    def test_pretraining_mode_creates_block_dataset(
        self, mock_load_dataset, temp_pretraining_file
    ):
        """Test that is_pretraining=True creates PretrainingBlockDataset."""
        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: 3
        mock_ds.__iter__ = lambda self: iter(
            [
                {"input_ids": [1, 2, 3], "len": 3},
                {"input_ids": [4, 5, 6], "len": 3},
                {"input_ids": [7, 8, 9], "len": 3},
            ]
        )
        mock_load_dataset.return_value = mock_ds

        # Call with pretraining mode
        loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=2,
            max_tokens_per_gpu=100,
            seed=42,
            rank=0,
            world_size=1,
            pretraining_config=PretrainingConfig(block_size=128),
        )

        # Verify load_dataset was called
        mock_load_dataset.assert_called_once()

        # Verify dataset is PretrainingBlockDataset
        assert isinstance(loader.dataset, PretrainingBlockDataset)

    def test_instruction_tuning_mode_creates_token_dataset(self, temp_pretraining_file):
        """Test that is_pretraining=False uses TokenDataset."""
        # Create a valid instruction tuning JSONL file
        # Standard
        from pathlib import Path

        inst_file = Path(temp_pretraining_file).parent / "inst_data.jsonl"
        samples = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "len": 3},
            {"input_ids": [4, 5, 6], "labels": [4, 5, 6], "len": 3},
        ]
        with open(inst_file, "w") as f:
            for sample in samples:
                json.dump(sample, f)
                f.write("\n")

        # Call with instruction tuning mode (default)
        loader = get_data_loader(
            data_path=str(inst_file),
            batch_size=2,
            max_tokens_per_gpu=100,
            seed=42,
            rank=0,
            world_size=1,
            pretraining_config=None,
        )

        # Verify dataset is TokenDataset (not PretrainingBlockDataset)
        # First Party
        from instructlab.training.sampler import TokenDataset

        assert isinstance(loader.dataset, TokenDataset)
        assert not isinstance(loader.dataset, PretrainingBlockDataset)

    @patch("instructlab.training.sampler.load_dataset")
    def test_pretraining_block_size_parameter(
        self, mock_load_dataset, temp_pretraining_file
    ):
        """Test that block_size parameter is correctly passed."""
        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: 1
        mock_ds.__iter__ = lambda self: iter(
            [{"input_ids": list(range(100)), "len": 100}]
        )
        mock_load_dataset.return_value = mock_ds

        # Call with specific block_size
        block_size = 256
        loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=2,
            max_tokens_per_gpu=1000,
            seed=42,
            rank=0,
            world_size=1,
            pretraining_config=PretrainingConfig(block_size=block_size),
        )

        # Verify dataset has correct block_size
        assert loader.dataset.block_size == block_size

    @patch("instructlab.training.sampler.load_dataset")
    def test_pretraining_pad_token_id_used(
        self, mock_load_dataset, temp_pretraining_file
    ):
        """Test that pad_token_id is correctly passed to PretrainingBlockDataset."""
        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: 1
        mock_ds.__iter__ = lambda self: iter(
            [{"input_ids": list(range(10)), "len": 10}]
        )
        mock_load_dataset.return_value = mock_ds

        # Call with specific pad_token_id
        pad_token_id = 99
        loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=2,
            max_tokens_per_gpu=100,
            seed=42,
            rank=0,
            world_size=1,
            pretraining_config=PretrainingConfig(
                block_size=7
            ),  # Will create partial block
            pad_token_id=pad_token_id,
        )

        # Verify dataset has correct pad_token_id
        assert loader.dataset.pad_token_id == pad_token_id

    @patch("instructlab.training.sampler.load_dataset")
    def test_data_loader_returns_correct_structure(
        self, mock_load_dataset, temp_pretraining_file
    ):
        """Test that get_data_loader returns a properly configured DataLoader."""
        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: 2
        mock_ds.__iter__ = lambda self: iter(
            [
                {"input_ids": list(range(50)), "len": 50},
                {"input_ids": list(range(50, 100)), "len": 50},
            ]
        )
        mock_load_dataset.return_value = mock_ds

        # Call get_data_loader
        loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=2,
            max_tokens_per_gpu=100,
            seed=42,
            rank=0,
            world_size=1,
            pretraining_config=PretrainingConfig(block_size=25),
        )

        # Verify it's a DataLoader
        # Third Party
        from torch.utils.data import DataLoader

        assert isinstance(loader, DataLoader)

        # Verify batch_size
        assert loader.batch_size == 2

    @patch("instructlab.training.sampler.load_dataset")
    def test_epoch_sampler_created(self, mock_load_dataset, temp_pretraining_file):
        """Test that EpochSampler is created with correct parameters."""
        # Create mock dataset with known length
        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: 1
        mock_ds.__iter__ = lambda self: iter(
            [{"input_ids": list(range(100)), "len": 100}]
        )
        mock_load_dataset.return_value = mock_ds

        seed = 123
        block_size = 25

        loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=2,
            max_tokens_per_gpu=100,
            seed=seed,
            rank=0,
            world_size=1,
            pretraining_config=PretrainingConfig(block_size=block_size),
        )

        # Verify sampler is EpochSampler
        # First Party
        from instructlab.training.sampler import EpochSampler

        assert isinstance(loader.sampler, EpochSampler)

        # Verify seed is set correctly
        assert loader.sampler.seed == seed

    @patch("instructlab.training.sampler.load_dataset")
    def test_collator_configuration(self, mock_load_dataset, temp_pretraining_file):
        """Test that MaxTokensPerRankCollator is configured correctly."""
        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: 1
        mock_ds.__iter__ = lambda self: iter(
            [{"input_ids": list(range(50)), "len": 50}]
        )
        mock_load_dataset.return_value = mock_ds

        flash_enabled = False
        pad_token_id = 42
        max_tokens = 200

        loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=2,
            max_tokens_per_gpu=max_tokens,
            seed=42,
            rank=0,
            world_size=1,
            pretraining_config=PretrainingConfig(block_size=25),
            flash_enabled=flash_enabled,
            pad_token_id=pad_token_id,
        )

        # Verify collate_fn is MaxTokensPerRankCollator
        # First Party
        from instructlab.training.sampler import MaxTokensPerRankCollator

        assert isinstance(loader.collate_fn, MaxTokensPerRankCollator)

        # Verify collator configuration
        assert loader.collate_fn.max_tokens_per_rank == max_tokens
        assert loader.collate_fn.flash_enabled == flash_enabled
        assert loader.collate_fn.pad_token_id == pad_token_id

    @patch("instructlab.training.sampler.load_dataset")
    def test_num_workers_parameter(self, mock_load_dataset, temp_pretraining_file):
        """Test that num_workers parameter is correctly applied."""
        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.column_names = ["input_ids", "len"]
        mock_ds.__len__ = lambda self: 1
        mock_ds.__iter__ = lambda self: iter(
            [{"input_ids": list(range(50)), "len": 50}]
        )
        mock_load_dataset.return_value = mock_ds

        num_workers = 4

        loader = get_data_loader(
            data_path=temp_pretraining_file,
            batch_size=2,
            max_tokens_per_gpu=100,
            seed=42,
            rank=0,
            world_size=1,
            pretraining_config=PretrainingConfig(block_size=25),
            num_workers=num_workers,
        )

        # Verify num_workers is set
        assert loader.num_workers == num_workers

        # When num_workers > 0, persistent_workers should be True
        assert loader.persistent_workers == True
