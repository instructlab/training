# Logging

## Brief overview of logging in python stdlib

The python stdlib logging module consists of 5 components:

1. Logger
2. Handler
3. Filter
4. Formatter
5. LogRecord

### Logger

The logger is the root of the logging system. A logger object can be created by calling the `logging.getLogger("logger_name")` function (calling this function with the same name will return the same logger object).

Loggers have a heirarchical structure such that a logger with name `"a"` is the parent of a logger with name `"a.b"`. It is common practice to give loggers a name that is the module name of the file that the logger is defined in. (e.g. `"instructlab.training.logger"`). All loggers are children of the root logger (specified by the empty string `""`).

Once a logger is created, it can be used to log messages by calling its `debug`/`info`/`warning`/`error`/`critical` methods, which create log records with the appropriate log level. Logs that pass the logger's log level and filters are then passed to the logger's handlers. If the logger's `propagate` attribute is `True`, the log record is also passed to the logger's parent (and so on up the hierarchy).

### Handler

A handler is an object that can process log messages. Each logger can have multiple handlers. Handlers can have their own log level and filters. Handlers accept `LogRecord` objects and can do things like write to a file, send an email, or push to an external API.

### Filter

A filter is an object that can be used to filter log messages. They can be set on loggers or handlers. Another use of filters is to modify the log message or its context before it is sent to a handler.

### Formatter

A formatter is an object that can be used to format log messages. It can be set on a handler. There can be only one formatter per handler, which is why filters are sometimes used to preprocess the log message before it is formatted.

### LogRecord

A LogRecord is the object that is created when a log message is made. It contains all the information about the log message, including the message, file, line number, log level, logger name, thread name, process name, and timestamp. The log message is the value that is passed to the logger's `debug`/`info`/`warning`/`error`/`critical` methods.

```python
import logging

logger = logging.getLogger("logger_name")

logger.debug("This is a debug message")
# We can also pass arbitrary objects to the logger and they will be set as the `LogRecord.msg` attribute.
logger.info({"value": 1})
logger.info({"lr": 0.001, "epochs": 10}, extra={"hparams": True})
```

Most handlers will stringify the `LogRecord.msg` value in order to print it, however, we can create handlers that use the value directly. Values set in the `extra` dictionary will become attributes of the `LogRecord` object, so it is important that the keys do not conflict with existing attributes.

## Logging in InstructLab

In InstructLab, we've added support for logging with the stdlib logging module.

### Logging to metric backends

We've created special handlers `AsyncStructuredHandler`, `TensorBoardHandler`, and `WandbHandler` that accept `LogRecord` objects with dictionaries in the `msg` attribute. They can then log metrics to their respective systems.

They also will check for a `step` attribute in the `extra` argument of the logger methods, and if found, pass it to the system. Alternatively, a `hparams=True` value can be passed through the `extra` dictionary to indicate that the log message should be treated as hyperparameters.

Throughout the codebase, metric values are logged to the `"instructlab.training.metrics"` logger. By configuring this logger with `AsyncStructuredHandler`, `TensorBoardHandler`, and `WandbHandler` handlers, logs can be sent to any of the backend systems (or multiple systems at once).

### Filtering log messages

We also provide the `IsMappingFilter` and `IsRank0Filter` which can be used to filter logs for the above handlers.

Since the above handlers expect dictionary messages, we recommend using the `IsMappingFilter` to ensure that only log messages with dictionaries are processed. In addition, the `IsRank0Filter` can be used to ensure that only the main process logs to the backend systems, which may be necessary for some of the handlers to prevent duplicate logs.

Additionally, we provide the `FormatDictFilter` which is not a true filter, but rather formats a dictionary message into a string for prettier logging. This filter can be used to format dictionary messages before they are passed to stdlib or third party library handlers that expect string messages.

### Setup calling InstructLab as a library

If you are using `instructlab.training` as a library, you may wish to enable logging in your application, and propagate logs from the `instructlab.training` package to your logging system.

This can be done by calling `instructlab.training.logger.propagate_package_logs(True)` in your application, and configuring the stdlib root logger to output to your logging system.

### Setup for training with InstructLab directly

If you are training with `instructlab.training` directly, you likely don't need to do anything beyond passing in the `run_name`, `logger_type`, and `output_dir` arguments when calling the script. Internally, the training script will call `setup_root_logger()` and `setup_metric_logger()` with a reasonable default configuration.

```bash
python src/instructlab/training/main_ds.py \
    ... \
    --run_name "my_run" \
    --logger_type "async,tensorboard,wandb" \
    --output_dir "out/"
```
