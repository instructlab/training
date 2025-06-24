# SPDX-License-Identifier: Apache-2.0
"""
Types for the training library.

TODO (osilkin):
    Move other classes acting as types into this file.
    Namely everything in `config.py` that's not actually a config.
"""

# Standard
import typing as t

# For Python 3.8+ compatibility
try:
    # Standard
    from typing import NotRequired, Required
except ImportError:
    try:
        # Third Party
        from typing_extensions import NotRequired, Required
    except ImportError:
        # Fallback for older Python versions
        Required = t.Annotated
        NotRequired = t.Annotated


class Message(t.TypedDict):
    """
    Format of a single message sample.

    Fields:
        content: The main content of the message.
        role: The role of the message sender (e.g., "user", "assistant", "system").
        reasoning_content: Optional reasoning trace or thinking process associated with the message.
                          This field is particularly useful for training reasoning-capable models
                          that can separate their thinking process from their final output.
    """

    content: Required[str]
    role: Required[str]
    reasoning_content: NotRequired[str]


class ProcessedMessagesData(t.TypedDict):
    """
    This class represents the data generated when a single sample is
    consumed and processed.
    """

    input_ids: t.List[int]
    labels: t.List[int]
    len: int
