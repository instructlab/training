# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import json
import os

try:
    # Third Party
    import wandb
except ImportError:
    wandb = None

try:
    # Third Party
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# First Party
from instructlab.training import async_logger


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


class BaseLogger:
    """Base class for all loggers in the system.

    This class defines the interface that all logger implementations must follow.
    It handles common functionality like run name substitution and provides abstract
    methods that must be implemented by concrete logger classes.
    """

    def __init__(self, run_name: str | None, log_dir: str | Path):
        """Initialize the base logger.

        Args:
            run_name: Name of the run, can include placeholders like {time}, {local_rank}, and {rank}
            log_dir: Directory where logs should be stored
        """
        if run_name is None:
            # default template
            run_name = "{time}_rank{rank}"

        self.run_name = self.substitute_placeholders(run_name)
        self.log_dir = log_dir

    @staticmethod
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

    def setup(self):
        """Initialize the logger. Must be implemented by subclasses."""
        raise NotImplementedError

    def log_dict(self, d: dict, step: int | None = None):
        """Log a dictionary of metrics/parameters.

        Args:
            d: Dictionary of values to log
            step: Optional step number for the log entry
        """
        raise NotImplementedError

    def log_hparams(self, hparam_dict: dict, step: int | None = None):
        """Log hyperparameters.

        Note: calling this method multiple times may overwrite the previous values with matching keys.

        Args:
            hparam_dict: Dictionary of hyperparameters to log
            step: Optional step number for the log entry
        """
        return self.log_dict(hparam_dict, step)

    def teardown(self, **_kwargs):
        """Clean up resources used by the logger. Must be implemented by subclasses."""
        raise NotImplementedError


class FileLogger(BaseLogger):
    """Logger that writes metrics to a JSONL file."""

    def setup(self):
        """Setup the file logger."""
        filename = Path(self.log_dir) / f"{self.run_name}.jsonl"
        os.makedirs(filename.parent, exist_ok=True)

        self.file = open(filename, "a")  # pylint: disable=R1732

    def log_dict(self, d: dict, step: int | None = None):
        """Write a dictionary to the JSONL file.

        Args:
            d: Dictionary to log
            step: Optional step number to include in the log entry
        """
        if step is not None and not "step" in d:
            d["step"] = step

        self.file.write(json.dumps(d, indent=None) + "\n")

    def teardown(self, **_kwargs):
        """Close the log file."""
        self.file.close()


class AsyncStructuredLogger(BaseLogger):
    """Logger that asynchronously writes data to a JSONL file."""

    def __init__(self, run_name: str | None, log_dir: str | Path):
        """Initialize the async logger.

        Args:
            run_name: Name of the run
            log_dir: Directory where logs should be stored
        """
        if run_name is None:
            # Async custom template for backwards compat
            run_name = "training_params_and_metrics_global{rank}"

        super().__init__(run_name, log_dir)

    def setup(self):
        """Initialize the async logger and create the log file."""
        filename = Path(self.log_dir) / f"{self.run_name}.jsonl"
        os.makedirs(filename.parent, exist_ok=True)
        self.struct_logger = async_logger.AsyncStructuredLogger(filename)

    def log_dict(self, d: dict, step: int | None = None):
        """Log a dictionary synchronously using the async logger.

        Args:
            d: Dictionary to log
            step: Optional step number 
        """
        self.struct_logger.log_sync(d)

    def teardown(self, **_kwargs):
        """Clean up the async logger."""
        pass


class WandbLogger(BaseLogger):
    """Logger that sends metrics to Weights & Biases (wandb)."""

    def __init__(self, run_name: str | None, log_dir: str | Path):
        """Initialize the wandb logger and check for required dependencies.

        Args:
            run_name: Name of the run
            log_dir: Directory where wandb logs should be stored

        Raises:
            RuntimeError: If wandb package is not installed
        """
        super().__init__(run_name, log_dir)

        if wandb is None:
            msg = (
                "Could not initialize WandbLogger because package wandb could not be imported.\n"
                "Please ensure it is installed by running 'pip install wandb'"
            )
            raise RuntimeError(msg)

        self._run = None

    def setup(self):
        """Initialize the wandb run."""
        self._run = wandb.init(name=self.run_name, dir=self.log_dir, config={})

    def log_dict(self, d: dict, step: int | None = None):
        """Log metrics to wandb.

        Args:
            d: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        flat_dict = flatten_dict(d, sep="/")
        self._run.log(flat_dict, step=step)

    def log_hparams(self, hparam_dict, step=None):
        """Log hyperparameters to wandb config.

        Args:
            hparam_dict: Dictionary of hyperparameters
            step: Optional step number
        """
        for k, v in flatten_dict(hparam_dict, sep="/").items():
            self._run.config[k] = v

    def teardown(self, **kwargs):
        """Finish the wandb run.

        Args:
            **kwargs: Additional arguments, including `exit_code`
        """
        self._run.finish(exit_code=kwargs.get("exit_code", 0))


class TensorBoardLogger(BaseLogger):
    """Logger that writes metrics to TensorBoard."""

    def __init__(self, run_name: str | None, log_dir: str | Path):
        """Initialize the TensorBoard logger and check for required dependencies.

        Args:
            run_name: Name of the run
            log_dir: Directory where TensorBoard logs should be stored

        Raises:
            RuntimeError: If tensorboard package is not installed
        """
        super().__init__(run_name, log_dir)

        if SummaryWriter is None:
            msg = (
                "Could not initialize TensorBoardLogger because package tensorboard could not be imported.\n"
                "Please ensure it is installed by running 'pip install tensorboard'"
            )
            raise RuntimeError(msg)

    def setup(self):
        """Create the TensorBoard log directory and initialize the writer."""
        log_dir = Path(self.log_dir) / self.run_name
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_dict(self, d: dict, step: int | None = None):
        """Log metrics to TensorBoard.

        Args:
            d: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        flat_dict = flatten_dict(d, sep="/")

        for k, v in flat_dict.items():
            try:
                # Check that `v` can be converted to float
                float(v)
            except ValueError:
                # Occurs for strings that can do not represent floats (e.g. "3.2.3") and aren't "inf" or "nan"
                self.writer.add_text(k, v, global_step=step)
            else:
                self.writer.add_scalar(k, v, global_step=step)

    def log_hparams(self, hparam_dict, step=None):
        """Log hyperparameters to TensorBoard.

        Args:
            hparam_dict: Dictionary of hyperparameters
            step: Optional step number for the hyperparameters
        """
        flat_dict = flatten_dict(hparam_dict, sep="/")
        self.writer.add_hparams(flat_dict, {}, run_name=".", global_step=step)

    def teardown(self, **_kwargs):
        """Close the TensorBoard writer."""
        self.writer.close()


LOGGERS: dict[str, type[BaseLogger]] = {
    "file": FileLogger,
    "wandb": WandbLogger,
    "tensorboard": TensorBoardLogger,
    "async": AsyncStructuredLogger,
}


def setup_metric_logger(
    logger_type: str, run_name: str, log_dir: str | Path
) -> BaseLogger:
    """Create and initialize a logger of the specified type.

    Args:
        logger_type: Type of logger to create (must be one of ["file", "wandb", "tensorboard", "async"])
        run_name: Name of the run
        log_dir: Directory where logs should be stored

    Returns:
        An initialized logger instance with `setup()` called

    Raises:
        ValueError: If logger_type is not one of the supported types
    """
    if logger_type not in LOGGERS:
        msg = (
            f"Invalid logger type {logger_type}. Must be one of [{', '.join(LOGGERS)}]"
        )
        raise ValueError(msg)

    logger = LOGGERS[logger_type](run_name, log_dir)
    logger.setup()
    return logger
