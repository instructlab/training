# SPDX-License-Identifier: Apache-2.0
"""Tests for on-demand checkpointing."""

# Standard
from unittest.mock import MagicMock, patch
import os
import signal

# Third Party
import pytest
import torch

# First Party
from instructlab.training.on_demand_checkpoint import (
    _CATCHABLE_SIGNALS,
    ParentSignalHandler,
    _get_trigger_path,
    check_checkpoint_requested,
    remove_trigger_file,
    trigger_file_exists,
    write_trigger_file,
)

# ---------------------------------------------------------------------------
# Trigger file helpers
# ---------------------------------------------------------------------------


class TestGetTriggerPath:
    def test_returns_correct_name(self):
        path = _get_trigger_path()
        assert path.name == "instructlab_checkpoint_requested"
        assert str(path.parent) == "/dev/shm"


class TestWriteTriggerFile:
    def test_creates_file(self, tmp_path):
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            path = write_trigger_file()
            assert path.exists()
            assert path.read_text() == "1"

    def test_returns_correct_path(self, tmp_path):
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            path = write_trigger_file()
            assert path == tmp_path / "instructlab_checkpoint_requested"


class TestTriggerFileExists:
    def test_returns_false_when_absent(self, tmp_path):
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            assert trigger_file_exists() is False

    def test_returns_true_when_present(self, tmp_path):
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            write_trigger_file()
            assert trigger_file_exists() is True


class TestRemoveTriggerFile:
    def test_removes_existing_file(self, tmp_path):
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            write_trigger_file()
            assert trigger_file_exists() is True
            remove_trigger_file()
            assert trigger_file_exists() is False

    def test_noop_on_missing_file(self, tmp_path):
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            # Should not raise
            remove_trigger_file()


# ---------------------------------------------------------------------------
# ParentSignalHandler
# ---------------------------------------------------------------------------


class TestParentSignalHandler:
    def test_install_registers_handlers(self):
        handler = ParentSignalHandler()
        original_handlers = {sig: signal.getsignal(sig) for sig in _CATCHABLE_SIGNALS}
        try:
            handler.install()
            for sig in _CATCHABLE_SIGNALS:
                current = signal.getsignal(sig)
                assert current == handler._handle, (
                    f"Expected handler._handle for {sig.name}, got {current}"
                )
        finally:
            for sig, orig in original_handlers.items():
                signal.signal(sig, orig)

    def test_handle_writes_trigger_and_records_signal(self, tmp_path):
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            handler = ParentSignalHandler()
            assert handler.signal_received is None
            assert handler._trigger_written is False

            handler._handle(signal.SIGUSR1, None)

            assert handler.signal_received == signal.SIGUSR1
            assert handler._trigger_written is True
            assert trigger_file_exists() is True

    def test_handle_is_idempotent(self, tmp_path):
        """Multiple signals should only write the trigger file once."""
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            handler = ParentSignalHandler()

            with patch(
                "instructlab.training.on_demand_checkpoint.write_trigger_file"
            ) as mock_write:
                mock_write.return_value = tmp_path / "dummy"
                handler._handle(signal.SIGUSR1, None)
                handler._handle(signal.SIGTERM, None)
                handler._handle(signal.SIGINT, None)

                mock_write.assert_called_once()

            # signal_received should be the LAST signal
            assert handler.signal_received == signal.SIGINT

    def test_uninstall_restores_original_handlers(self):
        handler = ParentSignalHandler()
        originals = {sig: signal.getsignal(sig) for sig in _CATCHABLE_SIGNALS}

        handler.install()
        for sig in _CATCHABLE_SIGNALS:
            assert signal.getsignal(sig) == handler._handle

        handler.uninstall()
        for sig in _CATCHABLE_SIGNALS:
            assert signal.getsignal(sig) == originals[sig], f"{sig.name} not restored"

    def test_install_via_real_signal(self, tmp_path):
        """End-to-end: install handler, send SIGUSR1, verify trigger written."""
        with patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path):
            handler = ParentSignalHandler()
            handler.install()
            try:
                os.kill(os.getpid(), signal.SIGUSR1)
                assert handler.signal_received == signal.SIGUSR1
                assert trigger_file_exists() is True
            finally:
                handler.uninstall()
                remove_trigger_file()


# ---------------------------------------------------------------------------
# check_checkpoint_requested (worker-side, mocked dist)
# ---------------------------------------------------------------------------


class TestCheckCheckpointRequested:
    def _mock_all_reduce_propagate(self, tensor, op=None):
        """Mock all_reduce that just keeps the local value."""
        pass

    def test_returns_false_when_no_trigger(self, tmp_path):
        with (
            patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path),
            patch("instructlab.training.on_demand_checkpoint.dist") as mock_dist,
            patch("torch.cuda.current_device", return_value=0),
        ):
            mock_dist.all_reduce = self._mock_all_reduce_propagate
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 0

            assert check_checkpoint_requested() is False

    def test_returns_true_when_trigger_exists(self, tmp_path):
        with (
            patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path),
            patch("instructlab.training.on_demand_checkpoint.dist") as mock_dist,
            patch("torch.cuda.current_device", return_value=0),
        ):
            mock_dist.all_reduce = self._mock_all_reduce_propagate
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 0

            write_trigger_file()
            assert check_checkpoint_requested() is True

    def test_cleans_up_trigger_after_detection(self, tmp_path):
        with (
            patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path),
            patch("instructlab.training.on_demand_checkpoint.dist") as mock_dist,
            patch("torch.cuda.current_device", return_value=0),
        ):
            mock_dist.all_reduce = self._mock_all_reduce_propagate
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 0

            write_trigger_file()
            check_checkpoint_requested()
            assert trigger_file_exists() is False

    def test_all_reduce_is_called(self, tmp_path):
        with (
            patch("instructlab.training.on_demand_checkpoint._TRIGGER_DIR", tmp_path),
            patch("instructlab.training.on_demand_checkpoint.dist") as mock_dist,
            patch("torch.cuda.current_device", return_value=0),
        ):
            mock_dist.all_reduce = MagicMock()
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 0
            mock_dist.ReduceOp.MAX = torch.distributed.ReduceOp.MAX

            check_checkpoint_requested()
            mock_dist.all_reduce.assert_called_once()
            _, kwargs = mock_dist.all_reduce.call_args
            assert kwargs.get("op") == torch.distributed.ReduceOp.MAX


# ---------------------------------------------------------------------------
# BatchLossManager.process_batch interrupt handling
# ---------------------------------------------------------------------------


class TestBatchLossManagerInterrupt:
    """Test that interrupt_check callbacks stop processing correctly."""

    @pytest.fixture
    def manager(self):
        model = MagicMock()
        model.compute_loss.return_value = (
            torch.tensor(1.0, requires_grad=True),
            MagicMock(main_loss=torch.tensor(0.5), aux_loss=None),
        )
        accelerator = MagicMock()
        accelerator.device = torch.device("cpu")
        accelerator.reduce.side_effect = lambda t, **kw: t
        accelerator.backward = MagicMock()

        # First Party
        from instructlab.training.batch_loss_manager import BatchLossManager

        mgr = BatchLossManager(
            model=model,
            accelerator=accelerator,
            world_size=1,
            local_rank=0,
        )
        return mgr

    def _make_batch(self, n_minibatches=3):
        """Create a fake batch with n minibatches."""
        return [
            {
                "input_ids": torch.randint(0, 100, (2, 32)),
                "labels": torch.randint(0, 100, (2, 32)),
                "num_samples": 2,
                "total_length": 32,
                "batch_num_loss_counted_tokens": 64,
            }
            for _ in range(n_minibatches)
        ]

    def test_no_interrupt_processes_all_minibatches(self, manager):
        batch = self._make_batch(3)
        metrics, _ = manager.process_batch(batch, interrupt_check=None)
        assert metrics.interrupted is False
        assert metrics.grad_accum_steps == 3

    def test_interrupt_before_first_forward(self, manager):
        """Interrupt fires immediately — no forward/backward should run."""
        batch = self._make_batch(3)
        metrics, _ = manager.process_batch(batch, interrupt_check=lambda: True)
        assert metrics.interrupted is True
        assert metrics.grad_accum_steps == 0
        manager.model.compute_loss.assert_not_called()
        manager.accelerator.backward.assert_not_called()

    def test_interrupt_before_backward(self, manager):
        """Interrupt fires after forward but before backward."""
        call_count = 0

        def interrupt_on_second_call():
            nonlocal call_count
            call_count += 1
            return call_count == 2

        batch = self._make_batch(3)
        metrics, _ = manager.process_batch(
            batch, interrupt_check=interrupt_on_second_call
        )
        assert metrics.interrupted is True
        assert manager.model.compute_loss.call_count == 1
        manager.accelerator.backward.assert_not_called()
        assert metrics.grad_accum_steps == 0

    def test_interrupt_after_backward(self, manager):
        """Interrupt fires after first backward — one grad accum step done."""
        call_count = 0

        def interrupt_on_third_call():
            nonlocal call_count
            call_count += 1
            return call_count == 3

        batch = self._make_batch(3)
        metrics, _ = manager.process_batch(
            batch, interrupt_check=interrupt_on_third_call
        )
        assert metrics.interrupted is True
        assert metrics.grad_accum_steps == 1
        manager.model.compute_loss.assert_called_once()
        manager.accelerator.backward.assert_called_once()

    def test_interrupt_never_fires(self, manager):
        """interrupt_check always returns False — full batch processed."""
        batch = self._make_batch(3)
        metrics, _ = manager.process_batch(batch, interrupt_check=lambda: False)
        assert metrics.interrupted is False
        assert metrics.grad_accum_steps == 3

    def test_compute_average_loss_handles_float_when_interrupted(self, manager):
        """When interrupted before any forward, accumulated_loss is 0.0 (float)."""
        result = manager._compute_average_loss(
            accumulated_loss=0.0,
            accumulated_aux_loss=None,
            batch_num_loss_counted_tokens=64,
        )
        assert isinstance(result, float)
