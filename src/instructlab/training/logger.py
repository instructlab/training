# SPDX-License-Identifier: Apache-2.0

# Standard
from logging.config import dictConfig


def setup_metric_logger(loggers, run_name, output_dir):
    if isinstance(loggers, str):
        loggers = [loggers]

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "async": {
                "class": "instructlab.training.logging_handlers.AsyncStructuredHandler",
                "level": "INFO",
                "log_dir": output_dir,
                "run_name": run_name,
            },
            "tensorboard": {
                "class": "instructlab.training.logging_handlers.TensorBoardHandler",
                "level": "INFO",
                "log_dir": output_dir,
                "run_name": run_name,
            },
            "wandb": {
                "class": "instructlab.training.logging_handlers.WandbHandler",
                "level": "INFO",
                "log_dir": output_dir,
                "run_name": run_name,
            },
        },
        "loggers": {
            "instructlab.training.metrics": {
                "handlers": loggers,
                "level": "INFO",
                "propagate": True,
            },
        },
    }
    dictConfig(logging_config)
