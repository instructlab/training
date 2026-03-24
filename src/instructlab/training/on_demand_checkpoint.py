# SPDX-License-Identifier: Apache-2.0

"""
On-demand checkpointing for distributed training.

This module enables graceful checkpoint-and-exit when termination signals are
received. It is designed for environments like OpenShift AI / KubeFlow where
training jobs can be preempted at any time and the platform sends Unix signals
before killing the pod.

Architecture
------------
There are two sides to this feature:

**Parent process** (``run_training`` in ``main_ds.py``):
    Installs signal handlers that catch every signal OpenShift / Kubernetes can
    send before a SIGKILL. When a signal arrives the handler writes a small
    *trigger file* to ``/dev/shm`` (a tmpfs shared between containers in the
    same pod). Because ``/dev/shm`` is node-local, every worker on the **same
    node** can see the file instantly with zero network I/O.

**Worker processes** (torchrun children):
    The training loop calls ``check_checkpoint_requested()`` at five
    synchronization points per training step, allowing the system to
    react as quickly as possible to termination signals:

    1. **Before each minibatch forward pass** — no partial computation;
       the current state is saved as-is.
    2. **Before each minibatch backward pass** — the forward result is
       discarded; the pre-step state is saved.
    3. **After each minibatch backward pass** — gradients are computed but
       not yet applied; the pre-step state is saved (gradients will be
       recomputed on resume).
    4. **Before the optimizer step** — all minibatches are done and
       gradients are ready, but the step is skipped; the pre-step state
       is saved.
    5. **After the optimizer step** — the step has been applied;
       ``samples_seen`` is updated and the post-step state is saved.

    Each rank checks its local ``/dev/shm`` for the trigger file, converts
    the boolean to a tensor, and does an ``all_reduce(MAX)`` so that if
    *any* rank on *any* node detected the trigger, *every* rank agrees to
    save a checkpoint. This works correctly in multi-node training because
    all_reduce is a global collective.

Signals handled
---------------
We intercept every signal that Kubernetes / OpenShift can deliver before the
hard SIGKILL (which cannot be caught):

* **SIGTERM** – the standard graceful-shutdown signal. Kubernetes sends this
  first (configurable via ``terminationGracePeriodSeconds``).
* **SIGINT** – sent on Ctrl-C or by some job controllers.
* **SIGUSR1 / SIGUSR2** – commonly used by batch schedulers and custom
  preemption controllers to signal upcoming eviction.
* **SIGXCPU** – sent when CPU time limits are exceeded (relevant for jobs
  with resource quotas).
* **SIGHUP** – sent when the controlling terminal disconnects; some
  container runtimes forward this on pod eviction.
"""

# Standard
from pathlib import Path
from typing import Callable, Optional, Union
import logging
import os
import signal
import tempfile
import types

# Third Party
import torch
import torch.distributed as dist

# Type alias matching the return type of signal.getsignal().
_SignalHandler = Union[
    Callable[[int, Optional[types.FrameType]], None], int, signal.Handlers, None
]

logger = logging.getLogger("instructlab.training")

# ---------------------------------------------------------------------------
# Trigger file helpers
# ---------------------------------------------------------------------------

# The trigger file lives in /dev/shm which is a tmpfs (RAM-backed filesystem).
# It is:
#   1. Extremely fast (no disk I/O).
#   2. Shared between all containers in the same Kubernetes pod.
#   3. Automatically cleaned up when the pod is destroyed.
_TRIGGER_DIR = Path("/dev/shm")
_TRIGGER_FILENAME = "instructlab_checkpoint_requested"


def _get_trigger_path(job_id: Optional[str] = None) -> Path:
    """Return the path to the checkpoint trigger file.

    An optional *job_id* can be supplied to avoid collisions if multiple
    training jobs share the same ``/dev/shm`` (unlikely but possible).
    """
    name = f"{_TRIGGER_FILENAME}_{job_id}" if job_id else _TRIGGER_FILENAME
    return _TRIGGER_DIR / name


def write_trigger_file(job_id: Optional[str] = None) -> Path:
    """Create the trigger file that tells workers to checkpoint.

    This is called from the *parent* process signal handler.
    Returns the path that was written.
    """
    path = _get_trigger_path(job_id)
    # Use a atomic write via tempfile + rename to avoid partial reads.
    fd, tmp = tempfile.mkstemp(dir=_TRIGGER_DIR, prefix=".ckpt_trigger_")
    try:
        os.write(fd, b"1")
    finally:
        os.close(fd)
    os.rename(tmp, path)
    logger.info(
        "On-demand checkpoint trigger file written: %s",
        path,
    )
    return path


def trigger_file_exists(job_id: Optional[str] = None) -> bool:
    """Check whether the trigger file exists (worker-side)."""
    return _get_trigger_path(job_id).exists()


def remove_trigger_file(job_id: Optional[str] = None) -> None:
    """Remove the trigger file after the checkpoint has been saved."""
    path = _get_trigger_path(job_id)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Parent-side signal handling
# ---------------------------------------------------------------------------

# Signals that OpenShift / Kubernetes / batch schedulers may send before
# the hard SIGKILL. SIGKILL (9) and SIGSTOP (19) cannot be caught.
_CATCHABLE_SIGNALS = (
    signal.SIGTERM,  # Kubernetes default graceful shutdown signal
    signal.SIGINT,  # Ctrl-C / some job controllers
    signal.SIGUSR1,  # Custom preemption controllers
    signal.SIGUSR2,  # Custom preemption controllers
    signal.SIGXCPU,  # CPU time limit exceeded (resource quotas)
    signal.SIGHUP,  # Terminal disconnect / some eviction paths
)


class ParentSignalHandler:
    """Installs signal handlers in the parent (launcher) process.

    When any of the catchable signals fire, the handler:
    1. Writes the trigger file to ``/dev/shm``.
    2. Records that a signal was received (so the caller can decide to
       wait for the child process to finish checkpointing).

    The handler is idempotent – multiple signals will not create multiple
    trigger files.

    Parameters
    ----------
    job_id : str, optional
        Unique identifier for this training job. Used to namespace the
        trigger file.
    """

    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id
        self.signal_received: Optional[signal.Signals] = None
        self._original_handlers: dict[signal.Signals, _SignalHandler] = {}
        self._trigger_written = False

    def install(self) -> None:
        """Register signal handlers for all catchable signals."""
        # Clear any stale trigger file from a previous run. If the file
        # exists before we've even installed signal handlers, it cannot
        # be from this job — it's left over from a prior run that was
        # killed before the workers could clean it up.
        if trigger_file_exists(self.job_id):
            logger.info(
                "On-demand checkpoint: clearing stale trigger file from "
                "a previous run (job_id=%s).",
                self.job_id,
            )
            try:
                remove_trigger_file(self.job_id)
            except Exception:
                logger.warning(
                    "On-demand checkpoint: failed to remove stale trigger file, "
                    "but continuing anyway.",
                    exc_info=True,
                )

        for sig in _CATCHABLE_SIGNALS:
            try:
                self._original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except (OSError, ValueError):
                # Some signals may not be available on all platforms
                logger.debug("Could not install handler for %s", sig.name)

        logger.info(
            "On-demand checkpoint signal handlers installed for: %s",
            ", ".join(s.name for s in self._original_handlers),
        )

    def uninstall(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)  # type: ignore[arg-type]
            except (OSError, ValueError):
                pass
        self._original_handlers.clear()

    def _handle(self, signum: int, _frame) -> None:
        """Signal handler callback."""
        sig = signal.Signals(signum)
        logger.info(
            "On-demand checkpoint: received signal %s (%d). "
            "Writing trigger file for workers to checkpoint before exit.",
            sig.name,
            signum,
        )
        self.signal_received = sig

        if not self._trigger_written:
            write_trigger_file(self.job_id)
            self._trigger_written = True


# ---------------------------------------------------------------------------
# Worker-side synchronization
# ---------------------------------------------------------------------------


def check_checkpoint_requested(job_id: Optional[str] = None) -> bool:
    """Check across all ranks whether an on-demand checkpoint was requested.

    This function must be called by **all ranks** at the same point in the
    training loop (it contains a collective all_reduce).

    Returns ``True`` if any rank detected the trigger file, meaning all
    ranks should save a checkpoint.
    """
    local_trigger = trigger_file_exists(job_id)

    # Convert to a tensor and all-reduce (MAX) so that if ANY rank on ANY
    # node saw the trigger, every rank gets True.
    trigger_tensor = torch.tensor(
        [1 if local_trigger else 0],
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(trigger_tensor, op=dist.ReduceOp.MAX)

    requested = trigger_tensor.item() > 0

    if requested:
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info(
                "On-demand checkpoint: global consensus reached – "
                "all ranks will save a checkpoint."
            )
        # Clean up the trigger file so that if the process somehow
        # continues, we don't save again immediately.
        remove_trigger_file(job_id)

    return requested


def save_on_demand_checkpoint(
    args,
    accelerator,
    model,
    tokenizer,
    samples_seen: int,
    epoch: int,
    global_step: int,
    is_lora: bool,
) -> None:
    """Save a full-state distributed checkpoint for on-demand resume.

    This is a thin wrapper that calls the existing ``save_checkpoint``
    utility with ``full_state=True`` so that optimizer + LR scheduler
    state are also persisted, enabling exact training resumption.

    The ``global_step`` is saved to the checkpoint metadata so that
    on resume the training loop can fast-forward to the exact step
    within the epoch where training was interrupted.
    """
    # First Party
    from instructlab.training.utils import save_checkpoint

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        logger.info(
            "On-demand checkpoint: saving full-state checkpoint at "
            "epoch=%d, global_step=%d, samples_seen=%d",
            epoch,
            global_step,
            samples_seen,
        )

    save_checkpoint(
        args=args,
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        samples_seen=samples_seen,
        is_lora=is_lora,
        full_state=True,
        hf_format=True,
        epoch=epoch,
        global_step=global_step,
    )

    if local_rank == 0:
        logger.info("On-demand checkpoint: checkpoint saved successfully.")
