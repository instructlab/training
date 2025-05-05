# Standard
from datetime import datetime, timezone
from unittest.mock import patch
import logging
import os

# Third Party
import pytest
import torch

# First Party
from instructlab.training.logger import (
    FormatDictFilter,
    IsMappingFilter,
    IsRank0Filter,
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
    # Note: time testing could be done better with freezegun library if added to dev dependencies

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

        # Test with partial placeholders
        before_time = datetime.now()
        name = _substitute_placeholders("run_{time}")
        after_time = datetime.now()

        time_str = name.split("run_")[1]
        parsed_time = datetime.fromisoformat(time_str)
        assert before_time <= parsed_time <= after_time

        name = _substitute_placeholders("rank{rank}")
        assert name == "rank1"

        name = _substitute_placeholders("local{local_rank}")
        assert name == "local2"


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
