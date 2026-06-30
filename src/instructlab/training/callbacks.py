# SPDX-License-Identifier: Apache-2.0

"""
Callback system for training lifecycle hooks.

Provides async, fire-and-forget callbacks that observe training events
without blocking the training loop or propagating exceptions.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any
import asyncio
import base64
import copy
import dataclasses
import inspect
import json
import logging
import textwrap
import threading

logger = logging.getLogger("instructlab.training")

HOOK_NAMES = [
    "on_train_begin",
    "on_epoch_begin",
    "on_step_begin",
    "on_before_forward",
    "on_after_backward",
    "on_pre_optimizer_step",
    "on_optimizer_step",
    "on_log",
    "on_evaluate",
    "on_save",
    "on_step_end",
    "on_epoch_end",
    "on_train_end",
]


@dataclass
class TrainingContext:
    """Mutable training state maintained by the training loop.

    The CallbackManager snapshots this before dispatching to callbacks,
    so callback authors receive an effectively read-only view.
    """

    hook_name: str = ""

    step: int = 0
    epoch: int = 0
    total_samples: int = 0
    total_tokens: int = 0

    loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    elapsed_time: float | None = None
    overall_throughput: float | None = None
    cuda_mem_allocated: float | None = None

    batch_metrics: dict[str, Any] = field(default_factory=dict)
    val_metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None

    output_dir: str = ""
    model_name_or_path: str = ""
    max_epochs: int = 0
    world_size: int = 1
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True


_CONTEXT_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(TrainingContext))


class TrainerCallback:
    """Base class for training callbacks. Subclass and override hooks you need.

    All methods are no-ops by default. Callbacks receive a TrainingContext
    snapshot and are purely observational (they cannot affect training flow).
    Callbacks fire on all ranks; use context.is_world_process_zero or
    context.is_local_process_zero to gate rank-specific behavior.

    Note: on_before_forward and on_after_backward fire once per microbatch
    inside the gradient accumulation loop, not once per training step.

    Callbacks must be self-contained for serialization across the torchrun
    subprocess boundary: all imports must be inside method bodies, and
    constructors must work with no arguments (or all-default arguments).
    """

    def on_train_begin(self, context: TrainingContext) -> None:
        pass

    def on_epoch_begin(self, context: TrainingContext) -> None:
        pass

    def on_step_begin(self, context: TrainingContext) -> None:
        pass

    def on_before_forward(self, context: TrainingContext) -> None:
        pass

    def on_after_backward(self, context: TrainingContext) -> None:
        pass

    def on_pre_optimizer_step(self, context: TrainingContext) -> None:
        pass

    def on_optimizer_step(self, context: TrainingContext) -> None:
        pass

    def on_log(self, context: TrainingContext) -> None:
        pass

    def on_evaluate(self, context: TrainingContext) -> None:
        pass

    def on_save(self, context: TrainingContext) -> None:
        pass

    def on_step_end(self, context: TrainingContext) -> None:
        pass

    def on_epoch_end(self, context: TrainingContext) -> None:
        pass

    def on_train_end(self, context: TrainingContext) -> None:
        pass


class CallbackManager:
    """Dispatches lifecycle hooks to registered TrainerCallback instances."""

    def __init__(self):
        self._callbacks: list[TrainerCallback] = []
        self.context = TrainingContext()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def add_callback(self, callback: TrainerCallback) -> None:
        if not isinstance(callback, TrainerCallback):
            raise TypeError(
                f"Expected a TrainerCallback instance, got "
                f"{type(callback).__name__}. "
                f"Pass an instance, not a class: callbacks=[MyCallback()]"
            )
        self._callbacks.append(callback)

    def remove_callback(self, callback_or_type) -> None:
        if isinstance(callback_or_type, type):
            self._callbacks = [
                cb for cb in self._callbacks if not isinstance(cb, callback_or_type)
            ]
        else:
            self._callbacks = [
                cb for cb in self._callbacks if cb is not callback_or_type
            ]

    def fire(self, hook_name: str, **kwargs) -> None:
        if hook_name not in HOOK_NAMES:
            raise ValueError(f"Unknown hook: '{hook_name}'. Valid hooks: {HOOK_NAMES}")
        if self._loop.is_closed():
            return
        if not self.has_callbacks(hook_name):
            return

        snapshot = copy.copy(self.context)
        snapshot.hook_name = hook_name
        snapshot.batch_metrics = dict(snapshot.batch_metrics)
        snapshot.val_metrics = dict(snapshot.val_metrics)
        for key, value in kwargs.items():
            if key not in _CONTEXT_FIELD_NAMES:
                raise ValueError(
                    f"Unknown TrainingContext field: '{key}'. Valid fields: {sorted(_CONTEXT_FIELD_NAMES)}"
                )
            setattr(snapshot, key, value)

        for callback in self._callbacks:
            method = getattr(callback, hook_name)
            if getattr(type(callback), hook_name) is getattr(
                TrainerCallback, hook_name
            ):
                continue
            cb_snapshot = copy.copy(snapshot)
            cb_snapshot.batch_metrics = dict(snapshot.batch_metrics)
            cb_snapshot.val_metrics = dict(snapshot.val_metrics)
            future = asyncio.run_coroutine_threadsafe(
                self._safe_invoke(method, cb_snapshot), self._loop
            )
            if hook_name == "on_train_end":
                try:
                    future.result(timeout=10)
                except TimeoutError:
                    logger.warning(
                        "Callback %s.%s timed out during on_train_end (10s limit).",
                        type(callback).__name__,
                        hook_name,
                    )
                except Exception:
                    logger.warning(
                        "Callback %s.%s failed during on_train_end.",
                        type(callback).__name__,
                        hook_name,
                        exc_info=True,
                    )

    async def _safe_invoke(self, method, context: TrainingContext) -> None:
        try:
            result = method(context)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception(
                "Callback %s.%s raised an exception (hook=%s, step=%d). "
                "This exception is suppressed and will not affect training.",
                type(method.__self__).__name__
                if hasattr(method, "__self__")
                else repr(method),
                method.__name__,
                context.hook_name,
                context.step,
            )

    def has_callbacks(self, hook_name: str) -> bool:
        base_method = getattr(TrainerCallback, hook_name)
        return any(
            getattr(type(cb), hook_name) is not base_method for cb in self._callbacks
        )

    def close(self) -> None:
        """Shut down the background event loop and thread."""
        if self._loop.is_closed():
            return
        try:
            pending = asyncio.all_tasks(self._loop)
        except RuntimeError:
            pending = set()
        if pending:

            async def _drain():
                await asyncio.gather(*pending, return_exceptions=True)

            future = asyncio.run_coroutine_threadsafe(_drain(), self._loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._thread.is_alive() and not self._loop.is_closed():
            self._loop.close()


def serialize_callback(callback: TrainerCallback) -> str:
    """Serialize a TrainerCallback subclass to a base64 string.

    The class must be self-contained: all imports must be inside method
    bodies. The constructor must work with no arguments (or all defaults).
    """
    source = inspect.getsource(type(callback))
    source = textwrap.dedent(source)
    return base64.b64encode(source.encode("utf-8")).decode("ascii")


def deserialize_callback(encoded: str) -> TrainerCallback:
    """Reconstruct a TrainerCallback instance from a base64-encoded class source."""
    source = base64.b64decode(encoded).decode("utf-8")
    namespace: dict[str, Any] = {
        "TrainerCallback": TrainerCallback,
        "TrainingContext": TrainingContext,
    }
    # Only called with source from run_training() serialization, never untrusted input
    exec(source, namespace)  # noqa: S102  # pylint: disable=exec-used
    classes = [
        v
        for v in namespace.values()
        if isinstance(v, type)
        and issubclass(v, TrainerCallback)
        and v is not TrainerCallback
    ]
    if len(classes) != 1:
        raise ValueError(
            f"Expected exactly one TrainerCallback subclass, got {len(classes)}."
        )
    return classes[0]()


def serialize_callbacks_for_cli(
    callbacks: list[TrainerCallback],
) -> str:
    """Serialize a list of callbacks to a base64 string for CLI transport."""
    serialized = [serialize_callback(cb) for cb in callbacks]
    return base64.b64encode(json.dumps(serialized).encode("utf-8")).decode("ascii")


def deserialize_callbacks_from_cli(
    encoded: str,
) -> list[TrainerCallback]:
    """Reconstruct TrainerCallback instances from a CLI-transported base64 string."""
    decoded = json.loads(base64.b64decode(encoded).decode("utf-8"))
    return [deserialize_callback(s) for s in decoded]
