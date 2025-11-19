# SPDX-License-Identifier: Apache-2.0

"""
Correct GPT-OSS MXFP4 quantization implementation that matches OpenAI's format exactly.
Based on the official OSS specification.
"""

# Standard
from typing import Dict
import logging
import re

# Third Party
from transformers import AutoConfig, PretrainedConfig
import torch

logger = logging.getLogger("instructlab.training")

GROUP_SIZE = 32  # MXFP4 block size (last-dim groups)


# ---- E2M1 codebook (FP4: 1 sign, 2 exp, 1 mant, bias=1), 16 values ----
# Exact values from PyTorch AO MXFP4 implementation
def _e2m1_decode_table(device=torch.device("cpu"), dtype=torch.float32):
    # Exact FP4 E2M1 values from PyTorch AO - force float32 for consistency
    fp4_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,  # Positive values
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,  # Negative values
        ],
        device=device,
        dtype=dtype,
    )  # Always use float32 for consistency
    return fp4_values  # shape [16]


@torch.no_grad()
def _find_closest_with_first_tie_breaking(values, table):
    """
    Find closest FP4 values with OpenAI's exact tie-breaking rules.
    Key insight: OpenAI respects IEEE 754 sign bits for zero values.
    When input is negative zero (-0.0), prefer index 8 over index 0.
    """
    # Ensure consistent precision for all calculations
    values = values.to(torch.float32)
    table = table.to(torch.float32)

    # Calculate squared distances with high precision
    distances = (values.unsqueeze(-1) - table) ** 2  # [..., 16]

    # Start with argmin (which handles most cases correctly)
    result_indices = torch.argmin(distances, dim=-1)

    # Special case: handle negative zero
    # When input is negative zero and distance to both +0.0 and -0.0 is equal,
    # prefer -0.0 (index 8) over +0.0 (index 0)

    min_distances = distances.min(dim=-1, keepdim=True)[0]
    epsilon = 1e-10

    # Find positions where both index 0 and 8 are tied (zero case)
    zero_tie_mask = (
        (distances[..., 0] <= min_distances[..., 0] + epsilon)  # Index 0 is tied
        & (distances[..., 8] <= min_distances[..., 0] + epsilon)  # Index 8 is tied
    )

    # Check which values are actually negative zero (sign bit = 1)
    # Use torch.signbit to detect negative zero
    is_negative_zero = torch.zeros_like(values, dtype=torch.bool)

    # Only check values that are actually zero
    zero_mask = torch.abs(values) < 1e-10
    if zero_mask.any():
        # For zero values, check sign bit
        with torch.no_grad():
            is_negative_zero = zero_mask & torch.signbit(values)

    # Apply the rule: if it's a zero tie and input is negative zero, choose index 8
    negative_zero_correction = zero_tie_mask & is_negative_zero
    result_indices = torch.where(negative_zero_correction, 8, result_indices)

    return result_indices


# Quantize floats (normalized by the block scale) to nearest E2M1 code 0..15
@torch.no_grad()
def _e2m1_encode(normalized: torch.Tensor) -> torch.Tensor:
    # Force float32 for all calculations to ensure consistency
    normalized = normalized.to(torch.float32)
    table = _e2m1_decode_table(device=normalized.device, dtype=torch.float32)  # [16]

    # Clamp to valid range first
    normalized_clamped = torch.clamp(normalized, min=-6.0, max=6.0)

    # OPTIMIZED: Increased batch sizes and smarter memory management
    # Process larger chunks since we're now batching more efficiently
    if (
        normalized_clamped.dim() >= 3 and normalized_clamped.shape[0] > 32
    ):  # Very large tensors
        # Process in larger batches - modern GPUs can handle much more
        batch_size = 32  # Increased from 4 to 32 for better GPU utilization
        expert_results = []
        for start_idx in range(0, normalized_clamped.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, normalized_clamped.shape[0])
            expert_batch = normalized_clamped[start_idx:end_idx]
            # Use proper tie-breaking that matches OpenAI's implementation
            batch_indices = _find_closest_with_first_tie_breaking(expert_batch, table)
            expert_results.append(batch_indices.to(torch.uint8))
        return torch.cat(expert_results, dim=0)
    else:
        # Process normally for smaller tensors
        # Use proper tie-breaking that matches OpenAI's implementation
        idx = _find_closest_with_first_tie_breaking(normalized_clamped, table)
        return idx.to(torch.uint8)


@torch.no_grad()
def _pack_nibbles(low_nib: torch.Tensor, high_nib: torch.Tensor) -> torch.Tensor:
    # both uint8 in [0,15]
    return (low_nib | (high_nib << 4)).to(torch.uint8)


@torch.no_grad()
def _power2_scales_from_maxabs(blocks: torch.Tensor) -> torch.Tensor:
    # blocks: [..., nblocks, G]
    # Use exact PyTorch AO scale calculation with bit manipulation
    maxabs = (
        blocks.abs().amax(dim=-1, keepdim=True).clamp_min(2 ** (-126))
    )  # [..., nblocks, 1]

    # Extract power-of-2 component from float32 representation (PyTorch AO method)
    maxabs_int32 = maxabs.view(torch.int32)
    extracted_pow2 = ((maxabs_int32 >> 23) & 0xFF) - 127  # Extract FP32 exponent

    # Calculate scale with target maximum power (4.0 = 2^2, so target_pow2 = 2)
    target_max_pow2 = 2  # For FP4 E2M1 max value 4.0
    scale_unbiased = extracted_pow2 - target_max_pow2

    # Clamp to valid range and remove keepdim
    scale_unbiased = scale_unbiased.squeeze(-1).clamp(-127, 128)  # [..., nblocks]

    # Return signed int8 exponent (transformers will handle +127 offset)
    return scale_unbiased.to(torch.int8)  # [..., nblocks]


@torch.no_grad()
def _quantize_tensor_to_mxfp4_param(weight: torch.Tensor, group_size: int = GROUP_SIZE):
    """
    Returns (blocks_u8, scales_i8, meta) for a single 2D+ tensor quantized along the last dim.

    This function now uses OpenAI's exact algorithm with:
    1. Perfect signed zero handling in tie-breaking
    2. Interleaved nibble packing (even positions as low, odd as high)
    3. Correct dimensional mapping for expert parameters
    """
    assert weight.ndim >= 2, "Quantize only 2D+ tensors"
    x = weight.to(torch.float32)

    # Pad last dim to multiple of group_size
    last = x.shape[-1]
    pad = (group_size - (last % group_size)) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))

    # [..., nblocks, G]
    new_shape = (*x.shape[:-1], x.shape[-1] // group_size, group_size)
    xb = x.view(new_shape)

    # per-block signed exponent e (int8); scale = 2**e
    e_i8 = _power2_scales_from_maxabs(
        xb.to(torch.float32)
    )  # [..., nblocks] - ensure float32
    scale = torch.pow(
        torch.tensor(2.0, device=x.device, dtype=torch.float32), e_i8.to(torch.float32)
    ).unsqueeze(-1)  # [..., nblocks, 1]

    y = xb * (1.0 / scale)  # normalized (use reciprocal like Triton)

    # encode each element to E2M1 code [0..15] using OpenAI's exact tie-breaking
    codes = _e2m1_encode(y)  # uint8 [..., nblocks, G]

    # Pack using OpenAI's INTERLEAVED method:
    # - Even positions (0, 2, 4, ...) become low nibbles
    # - Odd positions (1, 3, 5, ...) become high nibbles
    G = codes.shape[-1]
    assert G % 2 == 0

    # Split into even and odd positions (interleaved packing)
    low_nibbles = codes[..., ::2]  # Even positions: [0, 2, 4, 6, ...]
    high_nibbles = codes[..., 1::2]  # Odd positions: [1, 3, 5, 7, ...]

    # Pack nibbles: each byte = low_nibble | (high_nibble << 4)
    packed = _pack_nibbles(low_nibbles, high_nibbles)  # [..., nblocks, G/2]

    # Keep the 4D structure: [..., nblocks, 16] for blocks
    # packed shape is [..., nblocks, G/2] where G=32, so G/2=16
    blocks_u8 = packed.contiguous()  # Keep as [..., nblocks, 16]

    meta = {
        "orig_shape": tuple(weight.shape),
        "padded_last": int(pad),
        "group_size": int(group_size),
        "layout": "blocks_scales_lastdim",
        "dtype": "mxfp4_e2m1",
    }
    return blocks_u8.to(torch.uint8), e_i8.contiguous(), meta


def convert_dequantized_to_quantized_format_correct(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convert dequantized GPT-OSS parameters to quantized format using the correct OSS-compatible algorithm.

    This function converts:
    - experts.down_proj -> experts.down_proj_blocks + experts.down_proj_scales
    - experts.gate_up_proj -> experts.gate_up_proj_blocks + experts.gate_up_proj_scales

    Using the exact MXFP4 algorithm that matches OpenAI's format.

    Args:
        state_dict: Model state dict with dequantized parameters

    Returns:
        State dict with quantized format parameter names and correct MXFP4 quantization
    """
    converted_state_dict = {}
    conversion_count = 0

    logger.info("🔧 Starting CORRECT GPT-OSS parameter conversion...")
    logger.info(f"📥 Input state dict has {len(state_dict)} parameters")

    # Pattern to match MoE expert weight parameters (not biases)
    moe_param_pattern = re.compile(r"experts\.(gate_up_proj|down_proj)$")

    # First, copy all non-expert parameters to save memory
    expert_params_to_convert = []

    for param_name, param_tensor in state_dict.items():
        if moe_param_pattern.search(param_name):
            # Store expert params for later conversion
            expert_params_to_convert.append((param_name, param_tensor))
        else:
            # Keep non-expert parameters - move to CPU and convert to bf16 for memory efficiency
            if param_tensor.dtype == torch.float32:
                converted_param = param_tensor.to(torch.bfloat16).cpu()
                logger.debug(
                    f"💾 {param_name}: converted float32 → bf16 and moved to CPU"
                )
            else:
                converted_param = param_tensor.cpu()
                logger.debug(
                    f"💾 {param_name}: moved to CPU, kept {param_tensor.dtype}"
                )

            converted_state_dict[param_name] = converted_param

    # Now convert expert parameters one at a time to manage GPU memory
    for param_name, param_tensor in expert_params_to_convert:
        logger.info(
            f"🔄 Converting {param_name}: {param_tensor.shape} {param_tensor.dtype}"
        )

        try:
            # Use OpenAI's exact dimensional mapping discovered through reverse engineering
            # OpenAI's format: dequant[expert, row, col] -> blocks[expert, col, block_idx, byte_idx]
            # This means we quantize along the row dimension (dim=1), not the column dimension

            logger.info(
                f"🔄 Processing {param_name} with OpenAI's exact dimensional mapping"
            )
            logger.info(f"   Input shape: {param_tensor.shape}")

            if "gate_up_proj" in param_name:
                # gate_up_proj: dequantized is [experts, rows, cols] = [32, 2880, 5760]
                # OpenAI quantizes each column separately: [32, 2880] -> [32, 90, 16] per column
                # Result: [experts, cols, blocks_per_col, bytes_per_block] = [32, 5760, 90, 16]
                experts, rows, cols = param_tensor.shape
                blocks_per_col = rows // GROUP_SIZE

                logger.info(
                    f"   Processing {cols} columns, each with {blocks_per_col} blocks"
                )

                # OPTIMIZED: Process ALL columns at once using vectorized operations
                # Reshape to process all columns simultaneously: [experts, cols, rows] = [32, 5760, 2880]
                # Transpose to put columns first for efficient memory access
                tensor_transposed = param_tensor.transpose(1, 2)  # [32, 5760, 2880]

                # Reshape for batch quantization: [32*5760, 1, 2880]
                # This allows us to quantize all expert-column pairs at once
                total_columns = experts * cols
                reshaped_for_quant = tensor_transposed.reshape(total_columns, 1, rows)

                logger.info(
                    f"   VECTORIZED: Quantizing {total_columns} columns simultaneously"
                )

                # Single quantization call for all columns - MASSIVE speedup!
                all_blocks_flat, all_scales_flat, _ = _quantize_tensor_to_mxfp4_param(
                    reshaped_for_quant, GROUP_SIZE
                )

                # Reshape back to the correct format
                # all_blocks_flat: [32*5760, 1, 90, 16] -> [32, 5760, 90, 16]
                blocks_u8 = all_blocks_flat.squeeze(1).reshape(
                    experts, cols, blocks_per_col, 16
                )
                # all_scales_flat: [32*5760, 1, 90] -> [32, 5760, 90]
                scales_i8 = all_scales_flat.squeeze(1).reshape(
                    experts, cols, blocks_per_col
                )

            else:  # down_proj
                # down_proj: dequantized is [experts, rows, cols] = [32, 2880, 2880]
                # Same vectorized logic as gate_up_proj
                experts, rows, cols = param_tensor.shape
                blocks_per_col = rows // GROUP_SIZE

                logger.info(
                    f"   Processing {cols} columns, each with {blocks_per_col} blocks"
                )

                # OPTIMIZED: Process ALL columns at once using vectorized operations
                # Transpose to put columns first: [32, 2880, 2880] -> [32, 2880, 2880]
                tensor_transposed = param_tensor.transpose(1, 2)  # [32, 2880, 2880]

                # Reshape for batch quantization: [32*2880, 1, 2880]
                total_columns = experts * cols
                reshaped_for_quant = tensor_transposed.reshape(total_columns, 1, rows)

                logger.info(
                    f"   VECTORIZED: Quantizing {total_columns} columns simultaneously"
                )

                # Single quantization call for all columns - MASSIVE speedup!
                all_blocks_flat, all_scales_flat, _ = _quantize_tensor_to_mxfp4_param(
                    reshaped_for_quant, GROUP_SIZE
                )

                # Reshape back to the correct format
                # all_blocks_flat: [32*2880, 1, 90, 16] -> [32, 2880, 90, 16]
                blocks_u8 = all_blocks_flat.squeeze(1).reshape(
                    experts, cols, blocks_per_col, 16
                )
                # all_scales_flat: [32*2880, 1, 90] -> [32, 2880, 90]
                scales_i8 = all_scales_flat.squeeze(1).reshape(
                    experts, cols, blocks_per_col
                )

            # Create new parameter names with _blocks and _scales
            blocks_name = param_name + "_blocks"
            scales_name = param_name + "_scales"

            # Add +127 offset to scales for uint8 storage (HF format)
            scales_u8 = (scales_i8.to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)

            # Store quantized parameters (move to CPU to save GPU memory)
            converted_state_dict[blocks_name] = blocks_u8.cpu()
            converted_state_dict[scales_name] = scales_u8.cpu()

            logger.info(f"✅ {blocks_name}: {blocks_u8.shape} {blocks_u8.dtype}")
            logger.info(f"✅ {scales_name}: {scales_u8.shape} {scales_u8.dtype}")

            conversion_count += 1

            # Clear GPU memory after each conversion
            del blocks_u8, scales_i8, scales_u8
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ Failed to convert {param_name}: {e}")
            raise e

    logger.info(
        f"🎯 Converted {conversion_count} expert parameters using correct MXFP4 algorithm"
    )
    logger.info(f"📊 Output state dict has {len(converted_state_dict)} parameters")

    return converted_state_dict


def is_gpt_oss(model_path_or_config: str | PretrainedConfig) -> bool:
    """
    Determine if we should convert GPT-OSS format during saving.
    """
    return is_known_model(model_path_or_config, "gpt_oss")


def is_known_model(
    model_path_or_config: str | PretrainedConfig, known_model_type: str | list[str]
) -> bool:
    """
    Determine if the model is a known model.
    """
    if not isinstance(model_path_or_config, (PretrainedConfig, str)):
        raise ValueError(
            f"cannot detect model: received invalid argument of type {type(model_path_or_config)}"
        )

    # convert to config
    model_config = model_path_or_config
    if isinstance(model_path_or_config, str):
        model_config = AutoConfig.from_pretrained(model_path_or_config)

    known_model_types = (
        [known_model_type] if isinstance(known_model_type, str) else known_model_type
    )
    return getattr(model_config, "model_type", None) in known_model_types


def add_gpt_oss_quantization_config(config):
    """
    Add GPT-OSS quantization configuration to a model config object.

    Args:
        config: A transformers PretrainedConfig object

    Returns:
        The config object with quantization settings added
    """
    # add the quantization config if not present
    if not hasattr(config, "quantization_config") or config.quantization_config is None:
        config.quantization_config = {
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
            "quant_method": "mxfp4",
        }
        logger.info("Added GPT-OSS quantization config to model config")

    return config
