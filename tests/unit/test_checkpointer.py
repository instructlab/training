# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch
import torch.distributed as dist

# First Party
from instructlab.training.accelerator import Accelerator
from instructlab.training.checkpointer import Checkpointer
from instructlab.training.config import DistributedBackend


@pytest.fixture(autouse=True)
def mock_distributed():
    """Mock PyTorch distributed functionality for all tests."""
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.barrier") as mock_barrier,
        patch("torch.distributed.get_rank", return_value=0),
    ):
        yield mock_barrier


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.lora_config = None
    model.model_type = "llama"
    model.module = MagicMock()
    model.module.config = MagicMock()
    model.tokenizer = MagicMock()
    return model


@pytest.fixture
def mock_optimizer():
    return MagicMock()


@pytest.fixture
def mock_accelerator():
    accelerator = MagicMock(spec=Accelerator)
    accelerator.is_main_process = True
    accelerator.distributed_type = DistributedBackend.FSDP
    accelerator.distributed_framework = "fsdp"
    accelerator.get_state_dict = MagicMock()
    # Add missing methods that are used in the checkpointer
    accelerator.save_state = MagicMock()
    accelerator.save_model = MagicMock()
    return accelerator


def test_checkpointer_initialization(mock_model, mock_optimizer, mock_accelerator):
    checkpointer = Checkpointer(
        model=mock_model,
        optimizer=mock_optimizer,
        accelerator=mock_accelerator,
        strategy="all",
    )

    assert checkpointer.model == mock_model
    assert checkpointer.optimizer == mock_optimizer
    assert checkpointer.accelerator == mock_accelerator
    assert checkpointer.strategy == "all"


def test_checkpointer_no_checkpoint(mock_model, mock_optimizer, mock_accelerator):
    checkpointer = Checkpointer(
        model=mock_model,
        optimizer=mock_optimizer,
        accelerator=mock_accelerator,
        strategy="none",
    )

    # Test that no checkpointing occurs
    checkpointer.checkpoint(output_dir="test_dir", epoch=1, samples_seen=100)
    mock_accelerator.save_state.assert_not_called()


def test_checkpointer_full_state(mock_model, mock_optimizer, mock_accelerator):
    checkpointer = Checkpointer(
        model=mock_model,
        optimizer=mock_optimizer,
        accelerator=mock_accelerator,
        strategy="full_state",
    )

    output_dir = Path("test_dir")
    full_state_dir = output_dir / "full_state" / "epoch_1"
    full_state_dir.mkdir(parents=True, exist_ok=True)

    checkpointer.checkpoint(output_dir=output_dir, epoch=1, samples_seen=100)

    # Verify accelerator save_state was called
    mock_accelerator.save_state.assert_called_once()
    # Verify metadata was saved
    assert (full_state_dir / "training_metadata.json").exists()


def test_checkpointer_hf_format(mock_model, mock_optimizer, mock_accelerator):
    checkpointer = Checkpointer(
        model=mock_model,
        optimizer=mock_optimizer,
        accelerator=mock_accelerator,
        strategy="hf_format",
    )

    output_dir = Path("test_dir")
    hf_format_dir = output_dir / "hf_format" / "samples_100"
    hf_format_dir.mkdir(parents=True, exist_ok=True)

    checkpointer.checkpoint(output_dir=output_dir, epoch=1, samples_seen=100)

    # Verify model config and tokenizer were saved
    mock_model.module.config.to_json_file.assert_called_once()
    mock_model.tokenizer.save_pretrained.assert_called_once()
    # Verify accelerator save_model was called
    mock_accelerator.save_model.assert_called_once()


def test_checkpointer_all_strategies(mock_model, mock_optimizer, mock_accelerator):
    checkpointer = Checkpointer(
        model=mock_model,
        optimizer=mock_optimizer,
        accelerator=mock_accelerator,
        strategy="all",
    )

    output_dir = Path("test_dir")
    full_state_dir = output_dir / "full_state" / "epoch_1"
    hf_format_dir = output_dir / "hf_format" / "samples_100"
    full_state_dir.mkdir(parents=True, exist_ok=True)
    hf_format_dir.mkdir(parents=True, exist_ok=True)

    checkpointer.checkpoint(output_dir=output_dir, epoch=1, samples_seen=100)

    # Verify both full state and HF format were saved
    mock_accelerator.save_state.assert_called_once()
    mock_model.module.config.to_json_file.assert_called_once()
    mock_model.tokenizer.save_pretrained.assert_called_once()
    mock_accelerator.save_model.assert_called_once()


def test_checkpointer_lora_not_supported(mock_model, mock_optimizer, mock_accelerator):
    mock_model.lora_config = MagicMock()  # Set lora_config to non-None

    checkpointer = Checkpointer(
        model=mock_model,
        optimizer=mock_optimizer,
        accelerator=mock_accelerator,
        strategy="full_state",
    )

    with pytest.raises(NotImplementedError):
        checkpointer.checkpoint(output_dir="test_dir", epoch=1, samples_seen=100)


def test_checkpointer_load_latest_full_state(
    mock_model, mock_optimizer, mock_accelerator
):
    checkpointer = Checkpointer(
        model=mock_model,
        optimizer=mock_optimizer,
        accelerator=mock_accelerator,
        strategy="all",
    )

    # Mock the output directory structure
    output_dir = Path("test_dir")
    checkpoint_dir = output_dir / "full_state" / "epoch_1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Mock the accelerator's load_state method
    mock_accelerator.load_state = MagicMock()

    checkpointer.load_latest_full_state(output_dir)

    # Verify accelerator load_state was called
    mock_accelerator.load_state.assert_called_once()


def test_checkpointer_save_last_epoch(mock_model, mock_optimizer, mock_accelerator):
    checkpointer = Checkpointer(
        model=mock_model,
        optimizer=mock_optimizer,
        accelerator=mock_accelerator,
        strategy="hf_format",
    )

    output_dir = Path("test_dir")
    last_epoch_dir = output_dir / "hf_format" / "last_epoch"
    last_epoch_dir.mkdir(parents=True, exist_ok=True)

    checkpointer.checkpoint(
        output_dir=output_dir,
        epoch=1,
        samples_seen=100,
        last_epoch=True,
    )

    # Verify model was saved in last_epoch directory
    mock_model.module.config.to_json_file.assert_called_once()
    mock_model.tokenizer.save_pretrained.assert_called_once()
    mock_accelerator.save_model.assert_called_once()
