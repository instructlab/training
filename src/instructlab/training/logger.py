# SPDX-License-Identifier: Apache-2.0

# Standard
from logging.config import dictConfig
from collections.abc import Mapping
import logging


class IsMappingFilter(logging.Filter):
    def filter(self, record):
        return isinstance(record.msg, Mapping)


def setup_metric_logger(loggers, run_name, output_dir):
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
                "level": "INFO",
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping"],
            },
            "tensorboard": {
                "class": "instructlab.training.logging_handlers.TensorBoardHandler",
                "level": "INFO",
                "log_dir": output_dir,
                "run_name": run_name,
                "filters": ["is_mapping"],
            },
            "wandb": {
                "class": "instructlab.training.logging_handlers.WandbHandler",
                "level": "INFO",
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
