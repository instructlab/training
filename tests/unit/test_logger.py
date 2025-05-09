# Standard
from datetime import datetime, timezone
from unittest.mock import patch
import asyncio
import logging
import os
import time

# Third Party
import pytest

# First Party
from instructlab.training.logger import (
    AsyncStructuredHandler,
    FormatDictFilter,
    IsMappingFilter,
    IsRank0Filter,
    TensorBoardHandler,
    WandbHandler,
    _flatten_dict,
    _substitute_placeholders,
)


def test_flatten_dict():
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}

    flat = _flatten_dict(nested)
    assert flat == {"a": 1, "b/c": 2, "b/d/e": 3}

    # Test with custom separator
    flat = _flatten_dict(nested, sep=".")
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}

    # Test with prefix
    flat = _flatten_dict(nested, prefix="test_")
    assert flat == {"test_a": 1, "test_b/c": 2, "test_b/d/e": 3}


def test_substitute_placeholders():
    with patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "2"}):
        # Test with default template
        before_time = datetime.now()
        name = _substitute_placeholders(None)
        after_time = datetime.now()

        time_str = name.split("_rank")[0]
        parsed_time = datetime.fromisoformat(time_str)
        assert before_time <= parsed_time <= after_time
        assert name.endswith("_rank1")

        # Test with custom template containing all placeholders
        before_time = datetime.now()
        name = _substitute_placeholders("{time}_{utc_time}_{rank}_{local_rank}")
        after_time = datetime.now()

        time_str, utc_time_str, rank, local_rank = name.split("_")
        parsed_time = datetime.fromisoformat(time_str)
        parsed_utc = datetime.fromisoformat(utc_time_str)
        assert before_time <= parsed_time <= after_time
        assert parsed_utc.tzinfo == timezone.utc
        assert rank == "1"
        assert local_rank == "2"

        # Test with no placeholders
        name = _substitute_placeholders("test_run")
        assert name == "test_run"


def test_is_mapping_filter():
    filter_obj = IsMappingFilter()
    record = logging.LogRecord(
        "test", logging.INFO, "", 0, {"key": "value"}, None, None
    )
    assert filter_obj.filter(record) is True

    record.msg = "not a mapping"
    assert filter_obj.filter(record) is False


def test_is_rank0_filter():
    filter_obj = IsRank0Filter()

    # Test with rank in record
    record = logging.LogRecord("test", logging.INFO, "", 0, {"rank": 0}, None, None)
    assert filter_obj.filter(record) is True

    record = logging.LogRecord("test", logging.INFO, "", 0, {"rank": 1}, None, None)
    assert filter_obj.filter(record) is False

    # Test with local rank
    filter_obj = IsRank0Filter(local_rank=True)
    record = logging.LogRecord(
        "test", logging.INFO, "", 0, {"local_rank": 0}, None, None
    )
    assert filter_obj.filter(record) is True


def test_format_dict_filter():
    filter_obj = FormatDictFilter()

    # Test number formatting
    record = logging.LogRecord(
        "test",
        logging.INFO,
        "",
        0,
        {"small": 0.0001, "large": 1000.0, "normal": 0.123},
        None,
        None,
    )
    filter_obj.filter(record)
    assert "small=1.00e-04" in record.msg
    assert "large=1.00e+03" in record.msg
    assert "normal=0.123" in record.msg

    # Test nested dictionary
    record = logging.LogRecord("test", logging.INFO, "", 0, {"a": {"b": 1}}, None, None)
    filter_obj.filter(record)
    assert record.msg == "a/b=1"

    # Test non-dictionary message
    record = logging.LogRecord("test", logging.INFO, "", 0, "not a dict", None, None)
    filter_obj.filter(record)
    assert record.msg == "not a dict"


@pytest.fixture(params=[TensorBoardHandler, WandbHandler, AsyncStructuredHandler])
def logger_with_handler(request, tmp_path):
    logger_name = "instructlab.training.test_logger"
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler_cls = request.param
    handler = handler_cls(log_dir=tmp_path, run_name="test_run")
    handler.addFilter(IsMappingFilter())
    logger.addHandler(handler)

    yield logger

    # Remove all handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@patch.dict(os.environ, {"WANDB_MODE": "offline"})
def test_handlers(logger_with_handler, tmp_path):
    def get_log_files():
        if isinstance(logger_with_handler.handlers[0], AsyncStructuredHandler):
            time.sleep(0.1)
        return [
            p
            for p in tmp_path.iterdir()
            if p.name.startswith("test_run") or p.name.startswith("wandb")
        ]

    # Test non-mapping content gets filtered and file/directory isn't created
    logger_with_handler.info("test non-mapping content")
    assert len(get_log_files()) == 0, "Expected no log files in tmp_path"

    # Test mapping content creates a file/directory
    logger_with_handler.info({"test": 3, "test2": 3.7})
    assert (
        len(get_log_files()) == 1
    ), "Expected test_run file/directory found in tmp_path"

    # Test call with step
    for i in range(10):
        logger_with_handler.info({"test": 3, "test2": 3.7}, extra={"step": i})

    # Test call with hparams
    logger_with_handler.info({"epoch": 2, "lr": 0.001}, extra={"hparams": True})
    assert (
        len(get_log_files()) == 1
    ), "Expected test_run file/directory found in tmp_path"
