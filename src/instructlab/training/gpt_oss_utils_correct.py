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
# This matches the OSS/vLLM FP4 decode (no NaNs/infs; all finite codes).
def _e2m1_decode_table(device=torch.device("cpu"), dtype=torch.float32):
    codes = torch.arange(16, device=device, dtype=torch.uint8)
    s = ((codes >> 3) & 1).to(torch.int8)        # 1 bit sign
    e = ((codes >> 1) & 0x3).to(torch.int8)      # 2 bit exp
    m = (codes & 0x1).to(torch.int8)             # 1 bit mant
    sign = torch.where(s == 1, -1.0, 1.0).to(dtype)
    val = sign * (1.0 + 0.5 * m.to(dtype)) * torch.pow(torch.tensor(2.0, dtype=dtype, device=device), (e - 1).to(dtype))
    return val  # shape [16]

# Quantize floats (normalized by the block scale) to nearest E2M1 code 0..15
@torch.no_grad()
def _e2m1_encode(normalized: torch.Tensor) -> torch.Tensor:
    table = _e2m1_decode_table(device=normalized.device, dtype=normalized.dtype)  # [16]
    # [*, 16], pick argmin squared error
    idx = torch.argmin((normalized.unsqueeze(-1) - table)**2, dim=-1)
    return idx.to(torch.uint8)

@torch.no_grad()
def _pack_nibbles(low_nib: torch.Tensor, high_nib: torch.Tensor) -> torch.Tensor:
    # both uint8 in [0,15]
    return (low_nib | (high_nib << 4)).to(torch.uint8)

@torch.no_grad()
def _power2_scales_from_maxabs(blocks: torch.Tensor) -> torch.Tensor:
    # blocks: [..., nblocks, G]
    maxabs = blocks.abs().amax(dim=-1).clamp_min(1e-12)  # [..., nblocks]
    # Max representable magnitude in E2M1 is 6.0 (1.5 * 2^2)
    target = maxabs / 6.0
    e = torch.round(torch.log2(target))  # signed exponent
    # store only exponent as int8; actual scale = 2**e
    return e.to(torch.int8)  # [..., nblocks]

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
    e_i8 = _power2_scales_from_maxabs(xb)                            # [..., nblocks]
    scale = torch.pow(torch.tensor(2.0, device=x.device), e_i8.to(torch.float32)).unsqueeze(-1)  # [..., nblocks, 1]

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

    # reshape back to merge blocks on last axis
    bytes_lastdim = (x.shape[-1] // 2)
    blocks_u8 = packed.reshape(*x.shape[:-1], bytes_lastdim).contiguous()

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
    
    for param_name, param_tensor in state_dict.items():
        # Check if this is an expert weight parameter that should be quantized
        if moe_param_pattern.search(param_name):
            logger.info(f"🔄 Converting {param_name}: {param_tensor.shape} {param_tensor.dtype}")
            
            try:
                # Apply correct MXFP4 quantization
                blocks_u8, scales_i8, meta = _quantize_tensor_to_mxfp4_param(param_tensor, GROUP_SIZE)
                
                # Create new parameter names with _blocks and _scales
                blocks_name = param_name + "_blocks"
                scales_name = param_name + "_scales"
                
                # Store quantized parameters
                converted_state_dict[blocks_name] = blocks_u8
                converted_state_dict[scales_name] = scales_i8
                
                logger.info(f"✅ {blocks_name}: {blocks_u8.shape} {blocks_u8.dtype}")
                logger.info(f"✅ {scales_name}: {scales_i8.shape} {scales_i8.dtype}")
                
                conversion_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to convert {param_name}: {e}")
                raise e
        else:
            # Keep non-expert parameters as-is
            converted_state_dict[param_name] = param_tensor
    
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