# SPDX-License-Identifier: Apache-2.0

# Standard
import os
from logging.config import dictConfig
from collections.abc import Mapping
import logging
from pathlib import Path
from datetime import datetime
from collections.abc import Mapping
from instructlab.training import async_logger
import warnings
from typing import Any

# Third Party
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None

import torch
from rich.logging import RichHandler

# Disable package logging by default
package_logger = logging.getLogger("instructlab.training")
package_logger.addHandler(logging.NullHandler())
package_logger.propagate = False

### Helper functions


def _substitute_placeholders(
    run_name: str | None, default_template: str = "{time}_rank{rank}"
) -> str:
    """Replace placeholders in the run name with actual values.

    Supported placeholders:
        - {time}: Current timestamp in ISO format
        - {rank}: Process rank from RANK environment variable
        - {local_rank}: Local process rank from LOCAL_RANK environment variable

    Args:
        run_name: String containing placeholders to be replaced. If None, uses default_template
        default_template: Default template to use if run_name is None

    Returns:
        String with all placeholders replaced by their values
    """
    if run_name is None:
        run_name = default_template

    substitutions = {
        "{time}": datetime.now().isoformat(),
        "{rank}": os.environ.get("RANK", 0),
        "{local_rank}": os.environ.get("LOCAL_RANK", 0),
    }
    for placeholder_pat, value in substitutions.items():
        run_name = run_name.replace(placeholder_pat, str(value))

    return run_name


def _flatten_dict(d: Mapping, sep: str = "/", prefix: str = "") -> dict:
    """Flatten a nested dictionary into a single-level dictionary.

    This function recursively traverses a nested dictionary and creates a new
    dictionary with keys that represent the path to each value in the original
    dictionary.

    Args:
        d: The dictionary to flatten
        sep: Separator to use between nested keys
        prefix: Prefix to add to all keys

    Returns:
        A flattened dictionary with keys joined by the separator
    """
    flattened = {}

    for k, v in d.items():
        if isinstance(v, Mapping):
            flattened |= _flatten_dict(v, sep=sep, prefix=f"{prefix}{k}{sep}")
        else:
            flattened[prefix + k] = v

    return flattened


### Filters
class IsMappingFilter(logging.Filter):
    def filter(self, record):
        return isinstance(record.msg, Mapping)


class IsRank0Filter(logging.Filter):
    def __init__(self, rank_val: int | None = None, local_rank: bool = False):
        self.rank_val = rank_val
        if local_rank:
            self.rank_attr = "local_rank"
        else:
            self.rank_attr = "rank"

    def _get_rank(self, record):
        rank = (
            self.rank_val
            or getattr(record, self.rank_attr, None)
            or (isinstance(record.msg, Mapping) and record.msg.get(self.rank_attr))
            or os.environ.get(self.rank_attr.upper(), None)
            or (
                self.rank_attr == "rank"
                and torch.distributed.is_initialized()
                and torch.distributed.get_rank()
            )
            or 0
        )

        return int(rank)

    def filter(self, record):
        return self._get_rank(record) == 0


### Handlers
class TensorBoardHandler(logging.Handler):
    """Logger that writes metrics to TensorBoard.

    This handler expects a (nested) dictionary of metrics or text to be logged with string keys.
    A step can be specified by passing `extra={"step": <step>}` to the logging method (e.g. `logger.info(..., extra={"step": 10)}`).
    To log hyperparameters, pass a (nested) mapping of hyperparameters to the logging method and set `extra={"hparams": True}` (e.g. `logger.info({"lr": 0.001, "batch_size": 128}, extra={"hparams": True})`).
    """

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | os.PathLike = "logs",
        **tboard_init_kwargs: Any,
    ):
        """Initialize the TensorBoard logger and check for required dependencies.

        Args:
            level: The logging level for this handler
            run_name: Name of the run, can contain placeholders
            log_dir: Directory where TensorBoard logs should be stored
        """
        super().__init__(level)

        self.tboard_init_kwargs = tboard_init_kwargs.copy()
        self.tboard_init_kwargs.setdefault(
            "log_dir", Path(log_dir) / _substitute_placeholders(run_name)
        )

        self._tboard_writer = None

    def _setup(self):
        """Create the TensorBoard log directory and initialize the writer.

        Raises:
            RuntimeError: If tensorboard package is not installed
        """
        if SummaryWriter is None:
            msg = (
                "Could not initialize TensorBoardHandler because package tensorboard could not be imported.\n"
                "Please ensure it is installed by running 'pip install tensorboard'"
            )
            raise RuntimeError(msg)
        os.makedirs(self.tboard_init_kwargs["log_dir"], exist_ok=True)
        self._tboard_writer = SummaryWriter(**self.tboard_init_kwargs)

    def emit(self, record: logging.LogRecord):
        """Emit a log record to TensorBoard.

        This method handles both scalar metrics and text logs, automatically
        detecting the type of data being logged.

        Args:
            record: The log record to emit
        """
        if self._tboard_writer is None:
            self._setup()

        if not isinstance(record.msg, Mapping):
            warnings.warn(
                f"TensorBoardHandler expected a mapping, got {type(record.msg)}. Skipping log. Please ensure the handler is configured correctly to filter out non-mapping objects."
            )
            return

        flat_dict = _flatten_dict(record.msg)
        step = getattr(record, "step", None)
        if getattr(record, "hparams", None):
            self._tboard_writer.add_hparams(
                flat_dict, {}, run_name=".", global_step=step
            )
            return

        for k, v in flat_dict.items():
            try:
                # Check that `v` can be converted to float
                float(v)
            except ValueError:
                # Occurs for strings that cannot be converted to floats (e.g. "3.2.3") and aren't "inf" or "nan"
                self._tboard_writer.add_text(k, v, global_step=step)
            except TypeError:
                warnings.warn(
                    f"TensorBoardHandler expected a scalar or text, got {type(v)}. Skipping log. Please ensure metric logger is only called with mappings containing scalar values or text."
                )
            else:
                self._tboard_writer.add_scalar(k, v, global_step=step)

    def flush(self):
        """Flush the TensorBoard writer."""
        if self._tboard_writer is not None:
            self._tboard_writer.flush()

    def close(self):
        """Close the TensorBoard writer and cleanup resources."""
        if self._tboard_writer is not None:
            self._tboard_writer.close()
            self._tboard_writer = None
        super().close()


class WandbHandler(logging.Handler):
    """Logger that sends metrics to Weights & Biases (wandb).

    This handler expects a (nested) dictionary of metrics or text to be logged with string keys.
    A step can be specified by passing `extra={"step": <step>}` to the logging method (e.g. `logger.info(..., extra={"step": 10)}`).
    To log hyperparameters, pass a (nested) mapping of hyperparameters to the logging method and set `extra={"hparams": True}` (e.g. `logger.info({"lr": 0.001, "batch_size": 128}, extra={"hparams": True})`).
    """

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | os.PathLike = "logs",
        **wandb_init_kwargs: Any,
    ):
        """Initialize the wandb logger and check for required dependencies.

        Args:
            level: The logging level for this handler
            run_name: Name of the run, can contain placeholders
            log_dir: Directory where wandb logs should be stored

        Raises:
            RuntimeError: If wandb package is not installed
        """
        super().__init__(level)

        self.wandb_init_kwargs = wandb_init_kwargs.copy()
        self.wandb_init_kwargs.setdefault("dir", Path(log_dir))
        self.wandb_init_kwargs.setdefault("name", _substitute_placeholders(run_name))
        self.wandb_init_kwargs.setdefault("config", {})

        self._wandb_run = None

    def _setup(self):
        """Initialize the wandb run with the configured settings."""
        if wandb is None:
            msg = (
                "Could not initialize WandbLogger because package wandb could not be imported.\n"
                "Please ensure it is installed by running 'pip install wandb'"
            )
            raise RuntimeError(msg)
        self._wandb_run = wandb.init(**self.wandb_init_kwargs)

    def emit(self, record: logging.LogRecord):
        """Emit a log record to wandb.

        Args:
            record: The log record to emit
        """
        if self._wandb_run is None:
            self._setup()

        if not isinstance(record.msg, Mapping):
            warnings.warn(
                f"WandbHandler expected a mapping, got {type(record.msg)}. Skipping log. Please ensure the handler is configured correctly to filter out non-mapping objects."
            )
            return

        flat_dict = _flatten_dict(record.msg)
        step = getattr(record, "step", None)
        if getattr(record, "hparams", None):
            for k, v in flat_dict.items():
                self._wandb_run.config[k] = v
            return

        self._wandb_run.log(flat_dict, step=step)


class AsyncStructuredHandler(logging.Handler):
    """Logger that asynchronously writes data to a JSONL file.

    This handler expects a (nested) dictionary of metrics or text to be logged with string keys.
    A step can be specified by passing `extra={"step": <step>}` to the logging method (e.g. `logger.info(..., extra={"step": 10)}`).
    """

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | os.PathLike = "logs",
        **struct_init_kwargs: Any,
    ):
        """Initialize the async logger.

        Args:
            level: The logging level for this handler
            run_name: Name of the run, can contain placeholders
            log_dir: Directory where the logs should be stored
        """
        super().__init__(level)

        self.struct_init_kwargs = struct_init_kwargs.copy()
        maybe_file_name = Path(log_dir) / (
            _substitute_placeholders(
                run_name, default_template="training_params_and_metrics_global{rank}"
            )
            + ".jsonl"
        )
        self.struct_init_kwargs.setdefault("file_name", maybe_file_name)
        self._struct_logger = None

    def _setup(self):
        """Initialize the async logger and create the log file."""
        os.makedirs(Path(self.struct_init_kwargs["file_name"]).parent, exist_ok=True)
        self._struct_logger = async_logger.AsyncStructuredLogger(
            **self.struct_init_kwargs
        )

    def emit(self, record: logging.LogRecord):
        """Log a dictionary synchronously using the async logger.

        Args:
            record: The log record containing the dictionary to log
        """
        if self._struct_logger is None:
            self._setup()

        if not isinstance(record.msg, Mapping):
            warnings.warn(
                f"AsyncStructuredHandler expected a mapping, got {type(record.msg)}. Skipping log. Please ensure the handler is configured correctly to filter out non-mapping objects."
            )
            return

        flat_dict = _flatten_dict(record.msg)
        step = getattr(record, "step", None)
        if step:
            flat_dict.setdefault("step", step)

        self._struct_logger.log_sync(flat_dict)


### Main functions


def propagate_package_logs():
    """Enable instructlab.training package logs to be propagated to the root logger."""
    package_logger.propagate = True


def setup_root_logger(level="DEBUG"):
    # Enable package logging
    propagate_package_logs()

    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )


def setup_metric_logger(loggers, run_name, output_dir):
    if not loggers:
        return

    # Enable package logging
    propagate_package_logs()

    if isinstance(loggers, str):
        loggers = loggers.split(",")
    loggers = [logger.strip() for logger in loggers]

    async_filters = ["is_mapping"]
    if run_name is not None and "{rank}" not in run_name:
        async_filters.append("is_rank0")

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "is_mapping": {
                "()": IsMappingFilter,
            },
            "is_rank0": {
                "()": IsRank0Filter,
            },
        },
        "handlers": {
            "async": {
                "()": AsyncStructuredHandler,
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": async_filters,
            },
            "tensorboard": {
                "()": TensorBoardHandler,
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping", "is_rank0"],
            },
            "wandb": {
                "()": WandbHandler,
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping", "is_rank0"],
            },
        },
        "loggers": {
            "instructlab.training.metrics": {
                "handlers": loggers,
                "filters": ["is_mapping"],
                "level": "INFO",
                "propagate": True,
            },
            "instructlab.training": {
                "filters": ["is_rank0"],
                "level": "INFO",
                "propagate": True,
            },
        },
    }
    dictConfig(logging_config)
