# SPDX-License-Identifier: Apache-2.0

"""
Correct GPT-OSS MXFP4 quantization implementation that matches OpenAI's format exactly.
Based on the official OSS specification.
"""

import math
import re
import torch
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("instructlab.training")

GROUP_SIZE = 32  # MXFP4 block size (last-dim groups)

# ---- E2M1 codebook (FP4: 1 sign, 2 exp, 1 mant, bias=1), 16 values ----
# Exact values from PyTorch AO MXFP4 implementation
def _e2m1_decode_table(device=torch.device("cpu"), dtype=torch.float32):
    # Exact FP4 E2M1 values from PyTorch AO - force float32 for consistency
    fp4_values = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,  # Positive values
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0  # Negative values
    ], device=device, dtype=torch.float32)  # Always use float32 for consistency
    return fp4_values  # shape [16]

@torch.no_grad()
def _find_closest_with_last_tie_breaking(values, table):
    """
    Find closest FP4 values with custom tie-breaking that picks the last index.
    """
    # Calculate squared distances
    distances = (values.unsqueeze(-1) - table) ** 2  # [..., 16]
    
    # Find minimum distance for each value
    min_distances = torch.min(distances, dim=-1, keepdim=True)[0]  # [..., 1]
    
    # Create mask for tied values (equal to minimum distance)
    is_tied = (distances == min_distances)  # [..., 16]
    
    # Use a more efficient approach: create index tensor and use where with is_tied
    # Start with -1 to detect uninitialized positions
    result_indices = torch.full(values.shape, -1, dtype=torch.long, device=values.device)
    
    # Iterate through indices in forward order (0, 1, ..., 15)
    # Each iteration will overwrite previous values, so the last tied index wins
    for idx in range(table.shape[0]):
        mask = is_tied[..., idx]
        result_indices[mask] = idx
    
    return result_indices

# Quantize floats (normalized by the block scale) to nearest E2M1 code 0..15
@torch.no_grad()
def _e2m1_encode(normalized: torch.Tensor) -> torch.Tensor:
    # Force float32 for all calculations to ensure consistency
    normalized = normalized.to(torch.float32)
    table = _e2m1_decode_table(device=normalized.device, dtype=torch.float32)  # [16]
    
    # Clamp to valid range first
    normalized_clamped = torch.clamp(normalized, min=-6.0, max=6.0)
    
    # Process in chunks along the expert dimension to avoid memory issues
    # Keep block structure intact: [..., nblocks, GROUP_SIZE]
    if normalized_clamped.dim() >= 3 and normalized_clamped.shape[0] > 8:  # Multiple experts
        # Process experts in small batches
        batch_size = 4  # Process 4 experts at a time
        expert_results = []
        for start_idx in range(0, normalized_clamped.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, normalized_clamped.shape[0])
            expert_batch = normalized_clamped[start_idx:end_idx]
            # Use squared error for better precision
            batch_indices = torch.argmin((expert_batch.unsqueeze(-1) - table)**2, dim=-1)
            expert_results.append(batch_indices.to(torch.uint8))
        return torch.cat(expert_results, dim=0)
    else:
        # Small tensor, process normally
        # Use squared error for better precision
        idx = torch.argmin((normalized_clamped.unsqueeze(-1) - table)**2, dim=-1)
        return idx.to(torch.uint8)

@torch.no_grad()
def _pack_nibbles(low_nib: torch.Tensor, high_nib: torch.Tensor) -> torch.Tensor:
    # both uint8 in [0,15]
    return (low_nib | (high_nib << 4)).to(torch.uint8)

@torch.no_grad()
def _power2_scales_from_maxabs(blocks: torch.Tensor) -> torch.Tensor:
    # blocks: [..., nblocks, G]
    # Use exact PyTorch AO scale calculation with bit manipulation
    maxabs = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(2**(-126))  # [..., nblocks, 1]
    
    # Extract power-of-2 component from float32 representation (PyTorch AO method)
    maxabs_int32 = maxabs.view(torch.int32)
    extracted_pow2 = ((maxabs_int32 >> 23) & 0xFF) - 127  # Extract FP32 exponent
    
    # Calculate scale with target maximum power (6.0 = 2^2.58, so target_pow2 = 2.58)
    # But use exact Triton approach: maxabs / 6.0 instead of maxabs / 4.0
    target_max_pow2 = torch.log2(torch.tensor(6.0)).item()  # ~2.585 for FP4 E2M1 max value 6.0
    scale_unbiased = extracted_pow2 - target_max_pow2
    
    # Clamp to valid range and remove keepdim
    scale_unbiased = scale_unbiased.squeeze(-1).clamp(-127, 128)  # [..., nblocks]
    
    # Return signed int8 exponent (transformers will handle +127 offset)
    return scale_unbiased.to(torch.int8)  # [..., nblocks]

@torch.no_grad()
def _quantize_tensor_to_mxfp4_param(weight: torch.Tensor, group_size: int = GROUP_SIZE):
    """
    Returns (blocks_u8, scales_i8, meta) for a single 2D+ tensor quantized along the last dim.
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
    e_i8 = _power2_scales_from_maxabs(xb.to(torch.float32))         # [..., nblocks] - ensure float32
    scale = torch.pow(torch.tensor(2.0, device=x.device, dtype=torch.float32), e_i8.to(torch.float32)).unsqueeze(-1)  # [..., nblocks, 1]

    y = xb / scale                                                   # normalized

    # encode each element to E2M1 code [0..15]
    codes = _e2m1_encode(y)                                          # uint8 [..., nblocks, G]

    # pack 2 nibbles -> 1 byte along G
    G = codes.shape[-1]
    assert G % 2 == 0
    codes2 = codes.view(*codes.shape[:-1], G // 2, 2)                # [..., nblocks, G/2, 2]
    low  = codes2[..., 0]                                            # even index -> low nibble
    high = codes2[..., 1]                                            # odd  index -> high nibble
    packed = _pack_nibbles(low, high)                                # [..., nblocks, G/2]

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


def convert_dequantized_to_quantized_format_correct(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
                logger.debug(f"💾 {param_name}: converted float32 → bf16 and moved to CPU")
            else:
                converted_param = param_tensor.cpu()
                logger.debug(f"💾 {param_name}: moved to CPU, kept {param_tensor.dtype}")
            
            converted_state_dict[param_name] = converted_param
    
    # Now convert expert parameters one at a time to manage GPU memory
    for param_name, param_tensor in expert_params_to_convert:
        logger.info(f"🔄 Converting {param_name}: {param_tensor.shape} {param_tensor.dtype}")
        
        try:
            # Check if this is gate_up_proj - it needs special dimension handling
            if "gate_up_proj" in param_name:
                logger.info(f"🔄 Processing gate_up_proj with special dimension handling")
                # For gate_up_proj: dequantized is [experts, hidden_size, 2*intermediate_size]
                # HF expects storage as [experts, 2*intermediate_size, hidden_size//32, 16]
                # So we need to transpose before quantization
                param_to_quantize = param_tensor.transpose(1, 2).contiguous()
                logger.info(f"   Transposed from {param_tensor.shape} to {param_to_quantize.shape}")
            else:
                # For down_proj: dequantized is [experts, intermediate_size, hidden_size]
                # HF expects storage as [experts, hidden_size, intermediate_size//32, 16] 
                # So we need to transpose before quantization
                param_to_quantize = param_tensor.transpose(1, 2).contiguous()
                logger.info(f"   Transposed from {param_tensor.shape} to {param_to_quantize.shape}")
            
            # Apply correct MXFP4 quantization (keep on GPU)
            blocks_u8, scales_i8, meta = _quantize_tensor_to_mxfp4_param(param_to_quantize, GROUP_SIZE)
            
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
            del param_to_quantize, blocks_u8, scales_i8, scales_u8
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {param_name}: {e}")
            raise e
    
    logger.info(f"🎯 Converted {conversion_count} expert parameters using correct MXFP4 algorithm")
    logger.info(f"📊 Output state dict has {len(converted_state_dict)} parameters")
    
    return converted_state_dict


def should_convert_gpt_oss_format(model_config) -> bool:
    """
    Determine if we should convert GPT-OSS format during saving.
    """
    return getattr(model_config, 'model_type', None) == 'gpt_oss'


def update_config_for_quantized_format(config_path):
    """
    Update config.json to include proper GPT-OSS quantization configuration.
    """
    import json
    from pathlib import Path
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add the actual GPT-OSS quantization config if not present
    if 'quantization_config' not in config:
        config['quantization_config'] = {
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router", 
                "model.embed_tokens",
                "lm_head"
            ],
            "quant_method": "mxfp4"
        }
        
        # Create backup
        backup_path = config_path.with_suffix('.json.backup')
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Added GPT-OSS quantization config to {config_path}")
        logger.info(f"Backup saved as {backup_path}")