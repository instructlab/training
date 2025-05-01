# SPDX-License-Identifier: Apache-2.0

# Standard
import os
from logging.config import dictConfig
from collections.abc import Mapping
import logging

# Third Party
import torch
from rich.logging import RichHandler

# Disable package logging by default
package_logger = logging.getLogger("instructlab.training")
package_logger.addHandler(logging.NullHandler())
package_logger.propagate = False


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
            or (isinstance(record.msg, Mapping) and record.msg.get("rank", None))
            or os.environ.get(self.rank_attr.upper(), None)
            or (self.rank_attr == "rank" and torch.distributed.get_rank())
        )

        if rank is None:
            raise RuntimeError(
                f"Could not determine rank for logger. Please set the {self.rank_attr} environment variable or pass a rank to the logger."
            )
        return int(rank)

    def filter(self, record):
        return self._get_rank(record) == 0


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
                "class": "instructlab.training.logging_handlers.AsyncStructuredHandler",
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": async_filters,
            },
            "tensorboard": {
                "class": "instructlab.training.logging_handlers.TensorBoardHandler",
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping", "is_rank0"],
            },
            "wandb": {
                "class": "instructlab.training.logging_handlers.WandbHandler",
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
