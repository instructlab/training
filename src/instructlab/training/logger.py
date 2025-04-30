# SPDX-License-Identifier: Apache-2.0

# Standard
from logging.config import dictConfig
from collections.abc import Mapping
from rich.logging import RichHandler
import logging

# Disable package logging by default
package_logger = logging.getLogger("instructlab.training")
package_logger.addHandler(logging.NullHandler())
package_logger.propagate = False


class IsMappingFilter(logging.Filter):
    def filter(self, record):
        return isinstance(record.msg, Mapping)


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
    # Enable package logging
    propagate_package_logs()

    if isinstance(loggers, str):
        loggers = [loggers]

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "is_mapping": {
                "()": IsMappingFilter,
            },
        },
        "handlers": {
            "async": {
                "class": "instructlab.training.logging_handlers.AsyncStructuredHandler",
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping"],
            },
            "tensorboard": {
                "class": "instructlab.training.logging_handlers.TensorBoardHandler",
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping"],
            },
            "wandb": {
                "class": "instructlab.training.logging_handlers.WandbHandler",
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping"],
            },
        },
        "loggers": {
            "instructlab.training.metrics": {
                "handlers": loggers,
                "filters": ["is_mapping"],
                "level": "INFO",
                "propagate": True,
            },
        },
    }
    dictConfig(logging_config)
