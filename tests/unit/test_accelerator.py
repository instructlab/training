# Standard
from unittest.mock import MagicMock, patch
import os

# Third Party
from torch.utils.data import DataLoader
import pytest
import torch

# First Party
from instructlab.training.accelerator import Accelerator
from instructlab.training.config import DeepSpeedOptions, DistributedBackend
from instructlab.training.model import Model


@pytest.fixture
def mock_model():
    model = MagicMock(spec=Model)
    model.model = MagicMock()
    model.lora_config = None
    model._no_split_modules = ["LlamaDecoderLayer"]
    # Add children method to model
    model.children = MagicMock(return_value=[])
    model.model.children = MagicMock(return_value=[])
    # Add get_module_class_from_name method
    model.get_module_class_from_name = MagicMock(return_value=torch.nn.Module)
    return model


@pytest.fixture
def mock_train_loader():
    loader = MagicMock(spec=DataLoader)
    loader.dataset = MagicMock()
    return loader


@pytest.fixture
def mock_optimizer():
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    # Add param_groups attribute with required keys
    optimizer.param_groups = [{"params": [], "lr": 1e-4}]
    return optimizer


@pytest.fixture
def mock_transformers_accel():
    with patch("instructlab.training.accelerator.TransformersAccel") as mock:
        yield mock


def test_accelerator_init_deepspeed(
    mock_model, mock_train_loader, mock_transformers_accel
):
    with patch("torch.distributed.get_world_size", return_value=2):
        accelerator = Accelerator(
            model=mock_model,
            samples_per_gpu=8,
            grad_accum=2,
            train_loader=mock_train_loader,
            save_samples=1000,
            distributed_framework=DistributedBackend.DEEPSPEED,
            deepspeed_cpu_offload_optimizer_ratio=1.0,  # Add default value
        )

        assert accelerator.samples_per_gpu == 8
        assert accelerator.grad_accum == 2
        assert accelerator.model == mock_model
        assert accelerator.distributed_framework == DistributedBackend.DEEPSPEED
        assert accelerator.train_loader == mock_train_loader
        assert accelerator.save_samples == 1000


def test_accelerator_init_fsdp(mock_model, mock_train_loader, mock_transformers_accel):
    with patch("torch.distributed.get_world_size", return_value=2):
        accelerator = Accelerator(
            model=mock_model,
            samples_per_gpu=8,
            grad_accum=2,
            train_loader=mock_train_loader,
            save_samples=1000,
            distributed_framework=DistributedBackend.FSDP,
            fsdp_sharding_strategy="HYBRID_SHARD",
        )

        assert accelerator.samples_per_gpu == 8
        assert accelerator.grad_accum == 2
        assert accelerator.model == mock_model
        assert accelerator.distributed_framework == DistributedBackend.FSDP
        assert accelerator.fsdp_sharding_strategy == "HYBRID_SHARD"


def test_accelerator_prepare_with_optimizer(
    mock_model, mock_train_loader, mock_optimizer, mock_transformers_accel
):
    with patch("torch.distributed.get_world_size", return_value=2):
        accelerator = Accelerator(
            model=mock_model,
            samples_per_gpu=8,
            grad_accum=2,
            train_loader=mock_train_loader,
            save_samples=1000,
            distributed_framework=DistributedBackend.DEEPSPEED,
            deepspeed_cpu_offload_optimizer_ratio=1.0,  # Add default value
        )

        # Mock the accelerator's prepare method
        accelerator.accelerator = MagicMock()
        accelerator.accelerator.prepare.return_value = (
            mock_model.model,
            mock_optimizer,
            mock_train_loader,
            MagicMock(),  # lr_scheduler
        )

        accelerator.prepare_with_optimizer(
            optimizer=mock_optimizer,
            lr_scheduler="cosine",
            num_epochs=3,
            num_warmup_steps=100,
        )

        # Verify that prepare was called with the correct arguments
        accelerator.accelerator.prepare.assert_called_once()
        assert accelerator.optimizer == mock_optimizer


def test_accelerator_deepspeed_cpu_offload(
    mock_model, mock_train_loader, mock_transformers_accel
):
    with patch("torch.distributed.get_world_size", return_value=2):
        accelerator = Accelerator(
            model=mock_model,
            samples_per_gpu=8,
            grad_accum=2,
            train_loader=mock_train_loader,
            save_samples=1000,
            distributed_framework=DistributedBackend.DEEPSPEED,
            deepspeed_cpu_offload_optimizer=True,
            deepspeed_cpu_offload_optimizer_pin_memory=True,
            deepspeed_cpu_offload_optimizer_ratio=0.5,
        )

        assert accelerator.deepspeed_cpu_offload_optimizer is True
        assert accelerator.deepspeed_cpu_offload_optimizer_pin_memory is True
        assert accelerator.deepspeed_cpu_offload_optimizer_ratio == 0.5


def test_accelerator_fsdp_cpu_offload(
    mock_model, mock_train_loader, mock_transformers_accel
):
    with patch("torch.distributed.get_world_size", return_value=2):
        accelerator = Accelerator(
            model=mock_model,
            samples_per_gpu=8,
            grad_accum=2,
            train_loader=mock_train_loader,
            save_samples=1000,
            distributed_framework=DistributedBackend.FSDP,
            fsdp_sharding_strategy="HYBRID_SHARD",
            fsdp_cpu_offload_params=True,
        )

        assert accelerator.fsdp_cpu_offload_params is True


def test_accelerator_getattr(mock_model, mock_train_loader, mock_transformers_accel):
    with patch("torch.distributed.get_world_size", return_value=2):
        accelerator = Accelerator(
            model=mock_model,
            samples_per_gpu=8,
            grad_accum=2,
            train_loader=mock_train_loader,
            save_samples=1000,
            distributed_framework=DistributedBackend.DEEPSPEED,
            deepspeed_cpu_offload_optimizer_ratio=1.0,  # Add default value
        )

        # Mock a method on the underlying accelerator
        mock_method = MagicMock()
        accelerator.accelerator = MagicMock()
        accelerator.accelerator.some_method = mock_method

        # Test that __getattr__ forwards to the underlying accelerator
        result = accelerator.some_method()
        assert result == mock_method.return_value


def test_accelerator_setup_deepspeed_classmethod(
    mock_model, mock_train_loader, mock_transformers_accel
):
    with patch("torch.distributed.get_world_size", return_value=2):
        accelerator = Accelerator.setup_deepspeed(
            model=mock_model,
            samples_per_gpu=8,
            grad_accum=2,
            train_loader=mock_train_loader,
            deepspeed_cpu_offload_optimizer=True,
            deepspeed_cpu_offload_optimizer_pin_memory=True,
            deepspeed_cpu_offload_optimizer_ratio=0.5,
            save_samples=1000,
        )

        assert isinstance(accelerator, Accelerator)
        assert accelerator.distributed_framework == DistributedBackend.DEEPSPEED
        assert accelerator.deepspeed_cpu_offload_optimizer is True


def test_accelerator_setup_fsdp_classmethod(
    mock_model, mock_train_loader, mock_transformers_accel
):
    with patch("torch.distributed.get_world_size", return_value=2):
        accelerator = Accelerator.setup_fsdp(
            model=mock_model,
            samples_per_gpu=8,
            grad_accum=2,
            train_loader=mock_train_loader,
            fsdp_sharding_strategy="HYBRID_SHARD",
            fsdp_cpu_offload_params=True,
            save_samples=1000,
        )

        assert isinstance(accelerator, Accelerator)
        assert accelerator.distributed_framework == DistributedBackend.FSDP
        assert accelerator.fsdp_sharding_strategy == "HYBRID_SHARD"
        assert accelerator.fsdp_cpu_offload_params is True


def test_accelerator_with_lora(mock_model, mock_train_loader, mock_transformers_accel):
    # Set up a mock LoRA config
    mock_model.lora_config = MagicMock()
    mock_model.lora_config.target_modules = ["q_proj", "v_proj"]

    # Mock the fsdp_auto_wrap_policy function
    mock_wrap_policy = MagicMock()
    with patch("peft.utils.other.fsdp_auto_wrap_policy", return_value=mock_wrap_policy):
        with patch("torch.distributed.get_world_size", return_value=2):
            accelerator = Accelerator(
                model=mock_model,
                samples_per_gpu=8,
                grad_accum=2,
                train_loader=mock_train_loader,
                save_samples=1000,
                distributed_framework=DistributedBackend.FSDP,
                fsdp_sharding_strategy="HYBRID_SHARD",
            )

            # Verify that the accelerator was initialized with LoRA config
            assert accelerator.model.lora_config is not None
            assert accelerator.model.lora_config.target_modules == ["q_proj", "v_proj"]
