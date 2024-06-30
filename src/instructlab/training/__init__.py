__all__ = (
    "DataProcessArgs",
    "DeepSpeedOffloadStrategy",
    "DeepSpeedOptions",
    "LoraOptions",
    "QuantizeDataType",
    "TorchrunArgs",
    "TrainingArgs",
    "run_training",  # pylint: disable=undefined-all-variable
)

# Local
from .config import (
    DataProcessArgs,
    DeepSpeedOffloadStrategy,
    DeepSpeedOptions,
    LoraOptions,
    QuantizeDataType,
    TorchrunArgs,
    TrainingArgs,
)


def __dir__():
    return globals().keys() | {"run_training"}


def __getattr__(name):
    # lazy import run_training
    if name == "run_training":
        # pylint: disable=global-statement,import-outside-toplevel
        global run_training
        # Local
        from .main_ds import run_training

        return run_training

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
