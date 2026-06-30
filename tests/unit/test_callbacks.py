# SPDX-License-Identifier: Apache-2.0
"""Tests for the callback system."""

# Standard
import asyncio
import time

# Third Party
import pytest

# First Party
from instructlab.training.callbacks import (
    HOOK_NAMES,
    CallbackManager,
    TrainerCallback,
    TrainingContext,
    deserialize_callback,
    deserialize_callbacks_from_cli,
    serialize_callbacks_for_cli,
)


@pytest.fixture
def mgr():
    """Create a CallbackManager and close it after the test."""
    m = CallbackManager()
    yield m
    m.close()


class TestTrainingContext:
    def test_defaults(self):
        ctx = TrainingContext()
        assert ctx.step == 0
        assert ctx.epoch == 0
        assert ctx.loss is None
        assert ctx.batch_metrics == {}
        assert ctx.val_metrics == {}
        assert ctx.is_world_process_zero is True

    def test_field_assignment(self):
        ctx = TrainingContext(step=10, epoch=2, loss=0.5)
        assert ctx.step == 10
        assert ctx.epoch == 2
        assert ctx.loss == 0.5


class TestTrainerCallback:
    def test_all_hooks_are_noop(self):
        cb = TrainerCallback()
        ctx = TrainingContext()
        for hook in HOOK_NAMES:
            getattr(cb, hook)(ctx)

    def test_subclass_override(self):
        class MyCallback(TrainerCallback):
            def __init__(self):
                self.called = False

            def on_train_begin(self, context):
                self.called = True

        cb = MyCallback()
        cb.on_train_begin(TrainingContext())
        assert cb.called


class TestCallbackManager:
    def test_fire_dispatches(self, mgr):
        results = []

        class Recorder(TrainerCallback):
            def on_step_begin(self, context):
                results.append(("on_step_begin", context.step))

        mgr.add_callback(Recorder())
        mgr.context.step = 5
        mgr.fire("on_step_begin")
        time.sleep(0.1)
        assert results == [("on_step_begin", 5)]

    def test_fire_skips_non_overridden(self, mgr):
        results = []

        class Partial(TrainerCallback):
            def on_log(self, context):
                results.append("on_log")

        mgr.add_callback(Partial())
        mgr.fire("on_step_begin")
        mgr.fire("on_log")
        time.sleep(0.1)
        assert results == ["on_log"]

    def test_has_callbacks(self, mgr):
        class OnlyLog(TrainerCallback):
            def on_log(self, context):
                pass

        mgr.add_callback(OnlyLog())
        assert mgr.has_callbacks("on_log") is True
        assert mgr.has_callbacks("on_save") is False

    def test_snapshot_isolation(self, mgr):
        captured = []

        class Capture(TrainerCallback):
            def on_step_begin(self, context):
                captured.append(context.step)

        mgr.add_callback(Capture())
        mgr.context.step = 1
        mgr.fire("on_step_begin")
        mgr.context.step = 999
        time.sleep(0.1)
        assert captured == [1]

    def test_exception_isolation(self, mgr):
        class Broken(TrainerCallback):
            def on_train_begin(self, context):
                raise RuntimeError("boom")

        mgr.add_callback(Broken())
        mgr.fire("on_train_begin")
        time.sleep(0.1)

    def test_multiple_callbacks(self, mgr):
        results = []

        class A(TrainerCallback):
            def on_save(self, context):
                results.append("A")

        class B(TrainerCallback):
            def on_save(self, context):
                results.append("B")

        mgr.add_callback(A())
        mgr.add_callback(B())
        mgr.fire("on_save")
        time.sleep(0.1)
        assert sorted(results) == ["A", "B"]

    def test_kwargs_set_on_snapshot(self, mgr):
        captured = []

        class SaveCb(TrainerCallback):
            def on_save(self, context):
                captured.append(context.checkpoint_path)

        mgr.add_callback(SaveCb())
        mgr.fire("on_save", checkpoint_path="/tmp/ckpt")
        time.sleep(0.1)
        assert captured == ["/tmp/ckpt"]

    def test_add_callback_type_error(self, mgr):
        with pytest.raises(TypeError, match="TrainerCallback instance"):
            mgr.add_callback("not a callback")

    def test_remove_callback_by_instance(self, mgr):
        class Dummy(TrainerCallback):
            def on_log(self, context):
                pass

        cb = Dummy()
        mgr.add_callback(cb)
        assert mgr.has_callbacks("on_log")
        mgr.remove_callback(cb)
        assert not mgr.has_callbacks("on_log")

    def test_remove_callback_by_type(self, mgr):
        class Dummy(TrainerCallback):
            def on_log(self, context):
                pass

        mgr.add_callback(Dummy())
        mgr.remove_callback(Dummy)
        assert not mgr.has_callbacks("on_log")

    def test_fire_all_ranks(self, mgr):
        results = []

        class RankCb(TrainerCallback):
            def on_log(self, context):
                results.append(context.is_world_process_zero)

        mgr.add_callback(RankCb())
        mgr.context.is_world_process_zero = False
        mgr.fire("on_log")
        time.sleep(0.1)
        assert results == [False]

    def test_on_train_end_blocks(self, mgr):
        called = []

        class SlowCb(TrainerCallback):
            def on_train_end(self, context):
                called.append(True)

        mgr.add_callback(SlowCb())
        mgr.fire("on_train_end")
        assert called == [True]

    def test_fire_invalid_kwarg_raises(self, mgr):
        class Dummy(TrainerCallback):
            def on_save(self, context):
                pass

        mgr.add_callback(Dummy())
        with pytest.raises(ValueError, match="Unknown TrainingContext field"):
            mgr.fire("on_save", nonexistent_field="bad")

    def test_close(self):
        m = CallbackManager()
        try:
            assert m._thread.is_alive()
        finally:
            m.close()
        assert not m._thread.is_alive()

    def test_close_idempotent(self):
        m = CallbackManager()
        m.close()
        m.close()

    def test_fire_after_close_is_noop(self):
        m = CallbackManager()
        results = []

        class Recorder(TrainerCallback):
            def on_log(self, context):
                results.append("fired")

        m.add_callback(Recorder())
        m.close()
        m.fire("on_log")
        assert results == []

    def test_async_callback(self, mgr):
        results = []

        class AsyncCb(TrainerCallback):
            async def on_log(self, context):
                await asyncio.sleep(0.01)
                results.append("async_done")

        mgr.add_callback(AsyncCb())
        mgr.fire("on_log")
        time.sleep(0.2)
        assert results == ["async_done"]

    def test_remove_callback_others_still_fire(self, mgr):
        results = []

        class A(TrainerCallback):
            def on_log(self, context):
                results.append("A")

        class B(TrainerCallback):
            def on_log(self, context):
                results.append("B")

        mgr.add_callback(A())
        mgr.add_callback(B())
        mgr.remove_callback(A)
        mgr.fire("on_log")
        time.sleep(0.1)
        assert results == ["B"]

    def test_empty_manager_no_callbacks(self, mgr):
        assert mgr.has_callbacks("on_log") is False
        mgr.fire("on_log")

    def test_hook_name_set_on_snapshot(self, mgr):
        captured = []

        class HookNameCb(TrainerCallback):
            def on_step_begin(self, context):
                captured.append(context.hook_name)

        mgr.add_callback(HookNameCb())
        mgr.fire("on_step_begin")
        time.sleep(0.1)
        assert captured == ["on_step_begin"]

    def test_per_callback_snapshot_isolation(self, mgr):
        captured_a = []
        captured_b = []

        class MutatingCb(TrainerCallback):
            def on_log(self, context):
                context.batch_metrics["injected"] = "from_a"
                captured_a.append(dict(context.batch_metrics))

        class ObserverCb(TrainerCallback):
            def on_log(self, context):
                captured_b.append(dict(context.batch_metrics))

        mgr.add_callback(MutatingCb())
        mgr.add_callback(ObserverCb())
        mgr.context.batch_metrics = {"loss": 1.0}
        mgr.fire("on_log")
        time.sleep(0.2)
        assert captured_a == [{"loss": 1.0, "injected": "from_a"}]
        assert captured_b == [{"loss": 1.0}]

    def test_dict_fields_snapshot_isolation(self, mgr):
        captured = []

        class MetricsCb(TrainerCallback):
            def on_log(self, context):
                captured.append(context.batch_metrics)

        mgr.add_callback(MetricsCb())
        mgr.context.batch_metrics = {"loss": 1.0}
        mgr.fire("on_log")
        mgr.context.batch_metrics["loss"] = 999.0
        time.sleep(0.1)
        assert captured == [{"loss": 1.0}]


class TestAllRanksAndUserAPI:
    """Tests that callbacks fire on all ranks and users can configure rank behavior."""

    def test_fires_on_non_zero_rank(self, mgr):
        results = []

        class AllRankCb(TrainerCallback):
            def on_step_begin(self, context):
                results.append(context.is_world_process_zero)

        mgr.add_callback(AllRankCb())
        mgr.context.is_world_process_zero = False
        mgr.context.is_local_process_zero = False
        mgr.fire("on_step_begin")
        time.sleep(0.1)
        assert results == [False]

    def test_fires_on_rank_zero(self, mgr):
        results = []

        class AllRankCb(TrainerCallback):
            def on_step_begin(self, context):
                results.append(context.is_world_process_zero)

        mgr.add_callback(AllRankCb())
        mgr.context.is_world_process_zero = True
        mgr.fire("on_step_begin")
        time.sleep(0.1)
        assert results == [True]

    def test_user_can_gate_on_rank_zero(self, mgr):
        results = []

        class RankGatedCb(TrainerCallback):
            def on_log(self, context):
                if context.is_world_process_zero:
                    results.append("logged")

        mgr.add_callback(RankGatedCb())

        mgr.context.is_world_process_zero = False
        mgr.fire("on_log")
        time.sleep(0.1)
        assert results == []

        mgr.context.is_world_process_zero = True
        mgr.fire("on_log")
        time.sleep(0.1)
        assert results == ["logged"]

    def test_user_can_gate_on_local_rank_zero(self, mgr):
        results = []

        class LocalRankCb(TrainerCallback):
            def on_save(self, context):
                if context.is_local_process_zero:
                    results.append("local_rank_0")

        mgr.add_callback(LocalRankCb())

        mgr.context.is_local_process_zero = False
        mgr.fire("on_save")
        time.sleep(0.1)
        assert results == []

        mgr.context.is_local_process_zero = True
        mgr.fire("on_save")
        time.sleep(0.1)
        assert results == ["local_rank_0"]

    def test_both_rank_flags_exposed(self, mgr):
        captured = {}

        class RankInfoCb(TrainerCallback):
            def on_train_begin(self, context):
                captured["world"] = context.is_world_process_zero
                captured["local"] = context.is_local_process_zero

        mgr.add_callback(RankInfoCb())
        mgr.context.is_world_process_zero = False
        mgr.context.is_local_process_zero = True
        mgr.fire("on_train_begin")
        time.sleep(0.1)
        assert captured == {"world": False, "local": True}

    def test_all_13_hooks_fire(self, mgr):
        fired = []

        class AllHooksCb(TrainerCallback):
            def on_train_begin(self, context):
                fired.append("on_train_begin")

            def on_epoch_begin(self, context):
                fired.append("on_epoch_begin")

            def on_step_begin(self, context):
                fired.append("on_step_begin")

            def on_before_forward(self, context):
                fired.append("on_before_forward")

            def on_after_backward(self, context):
                fired.append("on_after_backward")

            def on_pre_optimizer_step(self, context):
                fired.append("on_pre_optimizer_step")

            def on_optimizer_step(self, context):
                fired.append("on_optimizer_step")

            def on_log(self, context):
                fired.append("on_log")

            def on_evaluate(self, context):
                fired.append("on_evaluate")

            def on_save(self, context):
                fired.append("on_save")

            def on_step_end(self, context):
                fired.append("on_step_end")

            def on_epoch_end(self, context):
                fired.append("on_epoch_end")

            def on_train_end(self, context):
                fired.append("on_train_end")

        mgr.add_callback(AllHooksCb())
        for hook in HOOK_NAMES:
            mgr.fire(hook)
            if hook != "on_train_end":
                time.sleep(0.05)
        assert sorted(fired) == sorted(HOOK_NAMES)

    def test_public_import_from_package(self):
        # First Party
        from instructlab.training import TrainerCallback, TrainingContext

        assert TrainerCallback is not None
        assert TrainingContext is not None

        cb = TrainerCallback()
        ctx = TrainingContext()
        cb.on_log(ctx)

    def test_training_args_accepts_callbacks(self):
        # First Party
        from instructlab.training import TrainingArgs

        assert "callbacks" in TrainingArgs.model_fields

    def test_context_has_training_config_fields(self):
        ctx = TrainingContext(
            output_dir="/tmp/output",
            model_name_or_path="my-model",
            max_epochs=3,
            world_size=4,
        )
        assert ctx.output_dir == "/tmp/output"
        assert ctx.model_name_or_path == "my-model"
        assert ctx.max_epochs == 3
        assert ctx.world_size == 4


class TestSerialization:
    def test_round_trip(self):
        class TestCallback(TrainerCallback):
            def on_log(self, context):
                pass

        callbacks = [TestCallback()]
        encoded = serialize_callbacks_for_cli(callbacks)
        restored = deserialize_callbacks_from_cli(encoded)
        assert len(restored) == 1
        assert isinstance(restored[0], TrainerCallback)
        assert type(restored[0]).__name__ == "TestCallback"

    def test_round_trip_preserves_behavior(self):
        class Adder(TrainerCallback):
            def on_log(self, context):
                context.loss = 42.0

        encoded = serialize_callbacks_for_cli([Adder()])
        restored = deserialize_callbacks_from_cli(encoded)
        ctx = TrainingContext()
        restored[0].on_log(ctx)
        assert ctx.loss == 42.0

    def test_multiple_callbacks_round_trip(self):
        class First(TrainerCallback):
            def on_save(self, context):
                pass

        class Second(TrainerCallback):
            def on_log(self, context):
                pass

        encoded = serialize_callbacks_for_cli([First(), Second()])
        restored = deserialize_callbacks_from_cli(encoded)
        assert len(restored) == 2
        assert type(restored[0]).__name__ == "First"
        assert type(restored[1]).__name__ == "Second"

    def test_non_zero_arg_constructor_raises_on_deserialize(self):
        class BadCallback(TrainerCallback):
            def __init__(self, url):
                self.url = url

            def on_log(self, context):
                pass

        encoded = serialize_callbacks_for_cli([BadCallback("http://example.com")])
        with pytest.raises(TypeError):
            deserialize_callbacks_from_cli(encoded)

    def test_malformed_base64_raises(self):
        with pytest.raises(Exception):
            deserialize_callback("not-valid-base64!!!")

    def test_inline_imports_survive_round_trip(self):
        class InlineImportCb(TrainerCallback):
            def on_log(self, context):
                # Standard
                import json

                return json.dumps({"step": context.step})

        encoded = serialize_callbacks_for_cli([InlineImportCb()])
        restored = deserialize_callbacks_from_cli(encoded)
        ctx = TrainingContext(step=42)
        result = restored[0].on_log(ctx)
        assert result == '{"step": 42}'
