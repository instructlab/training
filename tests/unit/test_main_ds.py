# Standard
from unittest import mock
import datetime

# Third Party
import pytest

# First Party
from instructlab.training import main_ds


def test__get_collective_timeout():
    # Test with default timeout
    assert main_ds._get_collective_timeout() is None

    # Test with custom timeout
    timeout = 1234
    with mock.patch.dict(
        main_ds.os.environ, {"INSTRUCTLAB_NCCL_TIMEOUT_MS": str(timeout)}
    ):
        assert main_ds._get_collective_timeout() == datetime.timedelta(
            milliseconds=timeout
        )

    # Test with invalid timeout (negative)
    invalid_timeout = "-100"
    with mock.patch.dict(
        main_ds.os.environ, {"INSTRUCTLAB_NCCL_TIMEOUT_MS": invalid_timeout}
    ):
        with pytest.raises(ValueError):
            main_ds._get_collective_timeout()

    # Test with invalid timeout (string)
    invalid_timeout = "invalid"
    with mock.patch.dict(
        main_ds.os.environ, {"INSTRUCTLAB_NCCL_TIMEOUT_MS": invalid_timeout}
    ):
        with pytest.raises(ValueError):
            main_ds._get_collective_timeout()
