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
):
    if _flattened is None:
        _flattened = {}

    for k, v in d.items():
        if isinstance(v, Mapping):
            flatten_dict(v, sep=sep, prefix=f"{prefix}{k}{sep}", _flattened=_flattened)
        else:
            _flattened[prefix + k] = v

    return _flattened


class BaseLogger:
    def __init__(self, run_name: str | None, log_dir: str | Path):
        if run_name is None:
            # default template
            run_name = "{time}_rank{rank}"

        self.run_name = self.substitute_placeholders(run_name)
        self.log_dir = log_dir

    @staticmethod
    def substitute_placeholders(run_name: str):
        substitutions = {
            "{time}": datetime.now().isoformat(),
            "{rank}": os.environ.get("RANK", 0),
            "{local_rank}": os.environ.get("LOCAL_RANK", 0),
        }
        for placeholder_pat, value in substitutions.items():
            run_name = run_name.replace(placeholder_pat, str(value))

        return run_name

    def setup(self):
        raise NotImplementedError

    def log_dict(self, d: dict, step: int | None = None):
        raise NotImplementedError

    def log_hparams(self, hparam_dict: dict, step: int | None = None):
        return self.log_dict(hparam_dict, step)

    def teardown(self, **_kwargs):
        raise NotImplementedError


class FileLogger(BaseLogger):
    def setup(self):
        filename = Path(self.log_dir) / f"{self.run_name}.jsonl"
        os.makedirs(filename.parent, exist_ok=True)

        self.file = open(filename, "a")  # pylint: disable=R1732

    def log_dict(self, d: dict, step: int | None = None):
        if step is not None and not "step" in d:
            d["step"] = step

        self.file.write(json.dumps(d, indent=None) + "\n")

    def teardown(self, **_kwargs):
        self.file.close()


class AsyncStructuredLogger(BaseLogger):
    def __init__(self, run_name: str | None, log_dir: str | Path):
        if run_name is None:
            # Async custom template for backwards compat
            run_name = "training_params_and_metrics_global{rank}"

        super().__init__(run_name, log_dir)

    def setup(self):
        filename = Path(self.log_dir) / f"{self.run_name}.jsonl"
        os.makedirs(filename.parent, exist_ok=True)
        self.struct_logger = async_logger.AsyncStructuredLogger(filename)

    def log_dict(self, d: dict, step: int | None = None):
        self.struct_logger.log_sync(d)

    def teardown(self, **_kwargs):
        pass


class WandbLogger(BaseLogger):
    def __init__(self, run_name: str | None, log_dir: str | Path):
        super().__init__(run_name, log_dir)

        if wandb is None:
            msg = (
                "Could not initialize WandbLogger because package wandb could not be imported.\n"
                "Please ensure it is installed by running 'pip install wandb'"
            )
            raise RuntimeError(msg)

        self._run = None

    def setup(self):
        self._run = wandb.init(name=self.run_name, dir=self.log_dir, config={})

    def log_dict(self, d: dict, step: int | None = None):
        flat_dict = flatten_dict(d, sep="/")
        self._run.log(flat_dict, step=step)

    def log_hparams(self, hparam_dict, step=None):
        for k, v in flatten_dict(hparam_dict, sep="/").items():
            self._run.config[k] = v

    def teardown(self, **kwargs):
        self._run.finish(exit_code=kwargs.get("exit_code", 0))


class TensorBoardLogger(BaseLogger):
    def __init__(self, run_name: str | None, log_dir: str | Path):
        super().__init__(run_name, log_dir)

        if SummaryWriter is None:
            msg = (
                "Could not initialize TensorBoardLogger because package tensorboard could not be imported.\n"
                "Please ensure it is installed by running 'pip install tensorboard'"
            )
            raise RuntimeError(msg)

    def setup(self):
        log_dir = Path(self.log_dir) / self.run_name
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_dict(self, d: dict, step: int | None = None):
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
        flat_dict = flatten_dict(hparam_dict, sep="/")
        self.writer.add_hparams(flat_dict, {}, run_name=".", global_step=step)

    def teardown(self, **_kwargs):
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
    if logger_type not in LOGGERS:
        msg = (
            f"Invalid logger type {logger_type}. Must be one of [{', '.join(LOGGERS)}]"
        )
        raise ValueError(msg)

    logger = LOGGERS[logger_type](run_name, log_dir)
    logger.setup()
    return logger
