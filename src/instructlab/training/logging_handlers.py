import logging
import os
from pathlib import Path
from datetime import datetime
from collections.abc import Mapping
from instructlab.training import async_logger

try:
    # Third Party
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    # Third Party
    import wandb
except ImportError:
    wandb = None


def substitute_placeholders(run_name: str) -> str:
    """Replace placeholders in the run name with actual values.

    Supported placeholders:
        - {time}: Current timestamp in ISO format
        - {rank}: Process rank from RANK environment variable
        - {local_rank}: Local process rank from LOCAL_RANK environment variable

    Args:
        run_name: String containing placeholders to be replaced

    Returns:
        String with all placeholders replaced by their values
    """
    substitutions = {
        "{time}": datetime.now().isoformat(),
        "{rank}": os.environ.get("RANK", 0),
        "{local_rank}": os.environ.get("LOCAL_RANK", 0),
    }
    for placeholder_pat, value in substitutions.items():
        run_name = run_name.replace(placeholder_pat, str(value))

    return run_name


def flatten_dict(
    d: Mapping,
    sep: str = ".",
    prefix: str = "",
    _flattened: dict | None = None,
) -> dict:
    """Flatten a nested dictionary into a single-level dictionary.

    Args:
        d: The dictionary to flatten
        sep: Separator to use between nested keys
        prefix: Prefix to add to all keys
        _flattened: Internal parameter for recursion, should not be set by caller

    Returns:
        A flattened dictionary with keys joined by the separator
    """
    if _flattened is None:
        _flattened = {}

    for k, v in d.items():
        if isinstance(v, Mapping):
            flatten_dict(v, sep=sep, prefix=f"{prefix}{k}{sep}", _flattened=_flattened)
        else:
            _flattened[prefix + k] = v

    return _flattened


class TensorBoardHandler(logging.Handler):
    """Logger that writes metrics to TensorBoard."""

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | Path = "logs",
    ):
        """Initialize the TensorBoard logger and check for required dependencies.

        Args:
            run_name: Name of the run
            log_dir: Directory where TensorBoard logs should be stored

        Raises:
            RuntimeError: If tensorboard package is not installed
        """
        super().__init__(level)
        self.run_name = run_name
        self.log_dir = log_dir

        if SummaryWriter is None:
            msg = (
                "Could not initialize TensorBoardLogger because package tensorboard could not be imported.\n"
                "Please ensure it is installed by running 'pip install tensorboard'"
            )
            raise RuntimeError(msg)

        self._tboard_writer = None

    def setup(self):
        """Create the TensorBoard log directory and initialize the writer."""
        log_dir = Path(self.log_dir) / self.run_name
        os.makedirs(log_dir, exist_ok=True)

        self._tboard_writer = SummaryWriter(log_dir=str(log_dir))

    def emit(self, record: logging.LogRecord):
        """Emit a log record to TensorBoard.

        Args:
            record: The log record to emit
        """
        if self._tboard_writer is None:
            self.setup()

        if record.levelno < self.level:
            return

        flat_dict = flatten_dict(record.msg, sep="/")
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
                # Occurs for strings that can do not represent floats (e.g. "3.2.3") and aren't "inf" or "nan"
                self._tboard_writer.add_text(k, v, global_step=step)
            else:
                self._tboard_writer.add_scalar(k, v, global_step=step)

    def flush(self):
        """Flush the TensorBoard writer."""
        if self._tboard_writer is not None:
            self._tboard_writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        if self._tboard_writer is not None:
            self._tboard_writer.close()
            self._tboard_writer = None
        super().close()


class WandbHandler(logging.Handler):
    """Logger that sends metrics to Weights & Biases (wandb)."""

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | Path = "logs",
    ):
        """Initialize the wandb logger and check for required dependencies.

        Args:
            run_name: Name of the run
            log_dir: Directory where wandb logs should be stored

        Raises:
            RuntimeError: If wandb package is not installed
        """
        super().__init__(level)
        self.run_name = run_name
        self.log_dir = log_dir

        if wandb is None:
            msg = (
                "Could not initialize WandbLogger because package wandb could not be imported.\n"
                "Please ensure it is installed by running 'pip install wandb'"
            )
            raise RuntimeError(msg)

        self._wandb_run = None

    def setup(self):
        """Initialize the wandb run."""
        self._wandb_run = wandb.init(name=self.run_name, dir=self.log_dir, config={})

    def emit(self, record: logging.LogRecord):
        """Emit a log record to wandb.

        Args:
            record: The log record to emit
        """
        if self._wandb_run is None:
            self.setup()

        flat_dict = flatten_dict(record.msg, sep="/")
        step = getattr(record, "step", None)
        if getattr(record, "hparams", None):
            for k, v in flat_dict.items():
                self._wandb_run.config[k] = v
            return

        self._wandb_run.log(flat_dict, step=step)


class AsyncStructuredHandler(logging.Handler):
    """Logger that asynchronously writes data to a JSONL file."""

    def __init__(
        self,
        level: int = logging.INFO,
        run_name: str | None = None,
        log_dir: str | Path = "logs",
    ):
        """Initialize the async logger.

        Args:
            level: The logging level
            run_name: The name of the run
            log_dir: The directory where the logs should be stored
        """
        super().__init__(level)
        if run_name is None:
            # Async custom template for backwards compat
            run_name = "training_params_and_metrics_global{rank}"
        self.run_name = substitute_placeholders(run_name)
        self.log_dir = log_dir
        self._struct_logger = None

    def setup(self):
        """Initialize the async logger and create the log file."""
        filename = Path(self.log_dir) / f"{self.run_name}.jsonl"
        os.makedirs(filename.parent, exist_ok=True)
        self._struct_logger = async_logger.AsyncStructuredLogger(filename)

    def emit(self, record: logging.LogRecord):
        """Log a dictionary synchronously using the async logger.

        Args:
            d: Dictionary to log
            step: Optional step number
        """
        if self._struct_logger is None:
            self.setup()
        flat_dict = flatten_dict(record.msg, sep="/")
        self._struct_logger.log_sync(flat_dict)
