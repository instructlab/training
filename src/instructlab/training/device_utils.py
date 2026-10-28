# SPDX-License-Identifier: Apache-2.0
"""
Accelerator abstraction so training runs on CUDA or Intel XPU.

Device-specific calls (device type, torch device module, memory stats) are
resolved here so the rest of the library stays accelerator-neutral.

CUDA remains the default: XPU is selected only when an XPU is present and CUDA
is not, so behavior on CUDA and CPU-only hosts is unchanged.
"""

# Standard
import functools
import logging
import os

# Third Party
import torch

logger = logging.getLogger("instructlab.training")

# Hugging Face hub flash-attention kernel with Intel XPU builds: the XPU
# counterpart of flash_attention_2 (the flash_attn package is CUDA/ROCm only).
XPU_FLASH_ATTN_KERNEL = "kernels-community/flash-attn2"


def get_device_type() -> str:
    """Return the accelerator type, ``"cuda"`` or ``"xpu"``.

    Set ``INSTRUCTLAB_TRAINING_DEVICE`` to force a specific type.
    """
    forced = os.environ.get("INSTRUCTLAB_TRAINING_DEVICE")
    if forced:
        return forced.lower()
    if (
        not torch.cuda.is_available()
        and hasattr(torch, "xpu")
        and torch.xpu.is_available()
    ):
        return "xpu"
    return "cuda"


def is_xpu() -> bool:
    return get_device_type() == "xpu"


def get_device_module():
    """Return ``torch.cuda`` or ``torch.xpu`` for the active accelerator."""
    return getattr(torch, get_device_type())


def get_device(local_rank: int | None = None) -> torch.device:
    """Return the accelerator ``torch.device`` for this rank (``LOCAL_RANK`` when not given)."""
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.device(get_device_type(), local_rank)


def num_alloc_retries() -> int:
    """Allocator retry count; XPU's memory_stats() has no such counter, so 0 there."""
    return get_device_module().memory_stats().get("num_alloc_retries", 0)


@functools.cache
def xpu_flash_attn_implementation() -> str:
    """Attention implementation for padding-free (packed) training on XPU.

    Returns the hub flash-attention kernel when it can be loaded (needs the
    ``kernels`` package, see the ``xpu`` extra), else ``"sdpa"``. Both keep
    packed sequences isolated through ``position_ids``, so the padding-free
    batching path is used either way.
    """
    try:
        # Third Party
        from transformers.integrations.hub_kernels import (
            load_and_register_attn_kernel,
        )

        load_and_register_attn_kernel(XPU_FLASH_ATTN_KERNEL)
        logger.info("XPU: using flash attention kernel %s", XPU_FLASH_ATTN_KERNEL)
        return XPU_FLASH_ATTN_KERNEL
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Flash attention kernel %s unavailable on XPU (%s); using SDPA with "
            "padding-free batches instead. Install it with "
            "`pip install instructlab-training[xpu]`.",
            XPU_FLASH_ATTN_KERNEL,
            exc,
        )
        return "sdpa"
