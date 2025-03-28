# SPDX-License-Identifier: Apache-2.0
"""
Types for the training library.

TODO (osilkin):
    Move other classes acting as types into this file.
    Namely everything in `config.py` that's not actually a config.
"""

# Standard
import typing as t


class Message(t.TypedDict):
    """
    Format of a single message sample.
    """

    content: str
    role: str


class ProcessedMessagesData(t.TypedDict):
    """
    This class represents the data generated when a single sample is
    consumed and processed.
    """

    input_ids: t.List[int]
    labels: t.List[int]
    len: int
