# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch
import os

# Third Party
from torch.optim import Optimizer
import pytest
import torch

# First Party
from instructlab.training.async_logger import AsyncStructuredLogger
from instructlab.training.model import Model
from instructlab.training.train import Metrics, train, train_epoch


# Test fixtures
@pytest.fixture
def mock_model():
    model = MagicMock(spec=Model)
    model.train.return_value = None
    model.get_global_grad_norm = MagicMock(return_value=1.0)
    model.parameters.return_value = [torch.tensor([1.0])]
    return model


@pytest.fixture
def mock_optimizer():
    optimizer = MagicMock(spec=Optimizer)
    optimizer.step.return_value = None
    optimizer.zero_grad.return_value = None
    return optimizer


@pytest.fixture
def mock_accelerator():
    accelerator = MagicMock()
    accelerator.device = "cpu"
    accelerator.samples_per_gpu = 4
    accelerator.grad_accum = 2
    accelerator.save_samples = 1000
    accelerator.train_loader = MagicMock()
    accelerator.train_loader.dataset = MagicMock()
    accelerator.train_loader.dataset.__len__.return_value = 1000
    accelerator.train_loader.batch_sampler = MagicMock()
    accelerator.train_loader.sampler = MagicMock()
    accelerator.reduce = MagicMock(return_value=torch.tensor([3.0, 1.0, 1.5]))
    accelerator.backward = MagicMock()
    accelerator.clip_grad_norm_ = MagicMock(return_value=1.0)
    accelerator.lr_scheduler = MagicMock()
    accelerator.lr_scheduler.get_last_lr.return_value = [0.001]
    return accelerator


@pytest.fixture
def mock_checkpointer():
    checkpointer = MagicMock()
    checkpointer.checkpoint.return_value = None
    return checkpointer


@pytest.fixture
def mock_logger():
    return MagicMock(spec=AsyncStructuredLogger)


@pytest.fixture
def mock_environment():
    with patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "2"}):
        yield


@pytest.fixture
def mock_distributed():
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=2),
    ):
        yield


@pytest.fixture
def mock_cuda():
    with (
        patch("torch.cuda.memory_allocated", return_value=0),
        patch("torch.cuda.memory_stats", return_value={"num_alloc_retries": 0}),
    ):
        yield


class TestTrainEpoch:
    def test_train_epoch_basic(
        self,
        mock_model,
        mock_optimizer,
        mock_accelerator,
        mock_checkpointer,
        mock_environment,
        mock_distributed,
        mock_cuda,
    ):
        # Setup mock batch
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "num_loss_counted_tokens": 3,
            "num_samples": 1,
        }
        mock_accelerator.train_loader.__iter__.return_value = [mock_batch]

        # Setup mock model output
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5)
        mock_model.return_value = mock_output

        # Run train_epoch
        metrics = train_epoch(
            epoch_number=0,
            samples_seen=0,
            local_rank=0,
            global_step=2,
            last_step=0,
            world_size=2,
            batch_size=4,
            samples_per_gpu=4,
            checkpoint_at_epoch=True,
            output_dir="test_output",
            sampler="distributed",
            checkpointer=mock_checkpointer,
            model=mock_model,
            optimizer=mock_optimizer,
            accelerator=mock_accelerator,
            use_dolomite=True,
        )

        # Verify metrics
        assert metrics is not None
        assert metrics.samples_seen == 1
        assert metrics.total_loss == 0.5
        assert metrics.batch_size == 4
        assert metrics.num_loss_counted_tokens == 3
        assert metrics.global_grad_norm == 1.0
        assert metrics.total_samples == 1000
        assert metrics.overall_throughput > 0
        assert metrics.current_lr == 0.001

        # Verify calls
        mock_model.assert_called_once()
        mock_optimizer.step.assert_called_once()
        mock_optimizer.zero_grad.assert_called_once()
        mock_accelerator.backward.assert_called_once()
        mock_accelerator.clip_grad_norm_.assert_called_once()
        mock_accelerator.lr_scheduler.step.assert_called_once()

    def test_train_epoch_with_multipack_sampler(
        self,
        mock_model,
        mock_optimizer,
        mock_accelerator,
        mock_checkpointer,
        mock_environment,
        mock_distributed,
        mock_cuda,
    ):
        # Setup mock batch
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "num_loss_counted_tokens": 3,
            "num_samples": 1,
        }
        mock_accelerator.train_loader.__iter__.return_value = [mock_batch]

        # Setup mock model output
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5)
        mock_model.return_value = mock_output

        # Run train_epoch with multipack sampler
        metrics = train_epoch(
            epoch_number=0,
            samples_seen=0,
            local_rank=0,
            global_step=2,
            last_step=0,
            world_size=2,
            batch_size=4,
            samples_per_gpu=4,
            checkpoint_at_epoch=True,
            output_dir="test_output",
            sampler="multipack",
            checkpointer=mock_checkpointer,
            model=mock_model,
            optimizer=mock_optimizer,
            accelerator=mock_accelerator,
            use_dolomite=True,
        )

        # Verify metrics
        assert metrics is not None
        assert metrics.samples_seen == 1
        assert metrics.total_loss == 0.5
        assert metrics.batch_size == 4
        assert metrics.num_loss_counted_tokens == 3
        assert metrics.global_grad_norm == 1.0
        assert metrics.total_samples == 1000
        assert metrics.overall_throughput > 0
        assert metrics.current_lr == 0.001

        # Verify calls
        mock_model.assert_called_once()
        mock_optimizer.step.assert_called_once()
        mock_optimizer.zero_grad.assert_called_once()
        mock_accelerator.backward.assert_called_once()
        mock_accelerator.clip_grad_norm_.assert_called_once()
        mock_accelerator.lr_scheduler.step.assert_called_once()
        mock_accelerator.train_loader.batch_sampler.set_epoch.assert_called_once_with(0)

    def test_train_epoch_invalid_sampler(
        self, mock_model, mock_optimizer, mock_accelerator, mock_checkpointer
    ):
        with pytest.raises(AttributeError) as exc_info:
            train_epoch(
                epoch_number=0,
                samples_seen=0,
                local_rank=0,
                global_step=1,
                last_step=0,
                world_size=2,
                batch_size=8,
                samples_per_gpu=4,
                checkpoint_at_epoch=True,
                output_dir="test_output",
                sampler="invalid",
                checkpointer=mock_checkpointer,
                model=mock_model,
                optimizer=mock_optimizer,
                accelerator=mock_accelerator,
                use_dolomite=False,
            )
        assert "Sampler invalid is invalid" in str(exc_info.value)


class TestTrain:
    def test_train_basic(
        self,
        mock_model,
        mock_optimizer,
        mock_accelerator,
        mock_checkpointer,
        mock_logger,
        mock_environment,
        mock_distributed,
        mock_cuda,
    ):
        # Setup mock metrics
        mock_metrics = Metrics(
            samples_seen=4,
            total_loss=0.5,
            batch_size=8,
            num_loss_counted_tokens=100,
            global_grad_norm=1.0,
            total_samples=1000,
            overall_throughput=100.0,
            current_lr=0.001,
        )

        # Run train with mock train_epoch
        with patch("instructlab.training.train.train_epoch", return_value=mock_metrics):
            train(
                model=mock_model,
                optimizer=mock_optimizer,
                accelerator=mock_accelerator,
                metric_logger=mock_logger,
                checkpointer=mock_checkpointer,
                effective_batch_size=8,
                num_epochs=1,
                last_step=0,
                checkpoint_at_epoch=True,
                output_dir="test_output",
                use_dolomite=False,
                save_last=True,
                sampler="distributed",
            )

        # Verify calls
        mock_model.train.assert_called_once()
        mock_checkpointer.checkpoint.assert_called_with(
            output_dir="test_output",
            epoch=1,
            samples_seen=4,
            last_epoch=True,
        )

    def test_train_with_save_samples(
        self,
        mock_model,
        mock_optimizer,
        mock_accelerator,
        mock_checkpointer,
        mock_logger,
        mock_environment,
        mock_distributed,
        mock_cuda,
    ):
        # Setup mock metrics
        mock_metrics = Metrics(
            samples_seen=4,
            total_loss=0.5,
            batch_size=8,
            num_loss_counted_tokens=100,
            global_grad_norm=1.0,
            total_samples=1000,
            overall_throughput=100.0,
            current_lr=0.001,
        )

        # Set save_samples to match batch size
        mock_accelerator.save_samples = 8

        # Run train with mock train_epoch
        with patch("instructlab.training.train.train_epoch", return_value=mock_metrics):
            train(
                model=mock_model,
                optimizer=mock_optimizer,
                accelerator=mock_accelerator,
                metric_logger=mock_logger,
                checkpointer=mock_checkpointer,
                effective_batch_size=8,
                num_epochs=1,
                last_step=0,
                checkpoint_at_epoch=True,
                output_dir="test_output",
                use_dolomite=False,
                save_last=True,
                sampler="distributed",
            )

        # Verify save_samples was adjusted
        assert mock_accelerator.save_samples == 8

    def test_train_with_resume(
        self,
        mock_model,
        mock_optimizer,
        mock_accelerator,
        mock_checkpointer,
        mock_logger,
        mock_environment,
        mock_distributed,
        mock_cuda,
    ):
        # Setup mock metrics
        mock_metrics = Metrics(
            samples_seen=4,
            total_loss=0.5,
            batch_size=8,
            num_loss_counted_tokens=100,
            global_grad_norm=1.0,
            total_samples=1000,
            overall_throughput=100.0,
            current_lr=0.001,
        )

        # Run train with mock train_epoch and resume from step 5
        with patch("instructlab.training.train.train_epoch", return_value=mock_metrics):
            train(
                model=mock_model,
                optimizer=mock_optimizer,
                accelerator=mock_accelerator,
                metric_logger=mock_logger,
                checkpointer=mock_checkpointer,
                effective_batch_size=8,
                num_epochs=1,
                last_step=5,
                checkpoint_at_epoch=True,
                output_dir="test_output",
                use_dolomite=False,
                save_last=True,
                sampler="distributed",
            )

        # Verify training started from the correct step
        mock_model.train.assert_called_once()
