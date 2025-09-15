# SPDX-License-Identifier: Apache-2.0
"""
Types for the training library.

TODO (osilkin):
    Move other classes acting as types into this file.
    Namely everything in `config.py` that's not actually a config.
"""

# Standard
from dataclasses import dataclass
import typing as t

# Third Party
# Third-party
import torch

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


class CollatedItem(t.TypedDict):
    """
    Items being returned by the collator function.
    """

    input_ids: Required[torch.Tensor]
    labels: Required[torch.Tensor]
    position_ids: NotRequired[torch.Tensor]  # Only required for flash attention
    attention_mask: NotRequired[torch.Tensor]  # Required for non-flash attention
    num_samples: Required[int]
    batch_num_loss_counted_tokens: Required[int]
    total_length: Required[int]
    num_loss_counted_tokens: Required[int]


class ModelInputs(t.TypedDict):
    """
    These are the inputs that models will be passed
    """

    input_ids: Required[torch.Tensor]
    labels: Required[torch.Tensor]
    position_ids: NotRequired[torch.Tensor]
    attention_mask: NotRequired[torch.Tensor]  # used when not training in padding free


class ProcessedMessagesData(t.TypedDict):
    """
    This class represents the data generated when a single sample is
    consumed and processed.
    """

    input_ids: t.List[int]
    labels: t.List[int]
    len: int


@dataclass
class ModelLosses:
    main_loss: torch.Tensor
    aux_loss: torch.Tensor | None
