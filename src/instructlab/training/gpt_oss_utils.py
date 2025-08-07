# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific utilities for handling quantization format conversions.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger("instructlab.training")


def convert_dequantized_to_quantized_format(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert dequantized GPT-OSS parameters back to quantized format for compatibility.
    
    This function converts:
    - experts.down_proj -> experts.down_proj_blocks
    - experts.gate_up_proj -> experts.gate_up_proj_blocks
    
    And generates placeholder quantization metadata (scales/zeros).
    
    Args:
        state_dict: Model state dict with dequantized parameters
        
    Returns:
        State dict with quantized format parameter names and metadata
    """
    converted_state_dict = {}
    expert_params_converted = []
    
    logger.info("Starting GPT-OSS parameter conversion...")
    logger.info(f"Input state dict has {len(state_dict)} parameters")
    
    conversion_count = 0
    for param_name, param_tensor in state_dict.items():
        new_name = param_name
        
        # Convert expert parameter names to quantized block format
        if ".mlp.experts.down_proj" in param_name and not param_name.endswith("_bias"):
            new_name = param_name.replace(".mlp.experts.down_proj", ".mlp.experts.down_proj_blocks")
            expert_params_converted.append((param_name, new_name, param_tensor))
            conversion_count += 1
            logger.info(f"Converting {param_name} -> {new_name}")
            
        elif ".mlp.experts.gate_up_proj" in param_name and not param_name.endswith("_bias"):
            new_name = param_name.replace(".mlp.experts.gate_up_proj", ".mlp.experts.gate_up_proj_blocks")
            expert_params_converted.append((param_name, new_name, param_tensor))
            conversion_count += 1
            logger.info(f"Converting {param_name} -> {new_name}")
        
        converted_state_dict[new_name] = param_tensor
    
    logger.info(f"Converted {conversion_count} parameter names")
    
    # Generate real quantization for converted expert parameters
    if expert_params_converted:
        logger.info(f"Generating real MXFP4 quantization for {len(expert_params_converted)} expert parameters")
        metadata = _generate_real_quantization_metadata(expert_params_converted)
        
        # Remove original dequantized parameters and add quantized versions
        for original_name, new_name, param_tensor in expert_params_converted:
            if original_name in converted_state_dict:
                del converted_state_dict[original_name]  # Remove dequantized version
        
        converted_state_dict.update(metadata)
        logger.info(f"Added {len(metadata)} quantized parameters (blocks + scales)")
    
    logger.info(f"Conversion complete: {len(converted_state_dict)} total parameters")
    return converted_state_dict


def _generate_real_quantization_metadata(expert_params_converted):
    """
    Generate real MXFP4 quantization for dequantized expert parameters.
    
    This function implements the inverse of transformers' convert_moe_packed_tensors
    to convert trained full-precision weights back to the packed uint8 MXFP4 format
    that transformers expects.
    
    Args:
        expert_params_converted: List of (original_name, new_name, tensor) tuples
        
    Returns:
        Dict of quantization metadata parameters
    """
    # Define FP4 lookup table values (from transformers implementation)
    FP4_VALUES = torch.tensor([
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,  # Positive values
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0   # Negative values  
    ])
    
    def quantize_weights_to_mxfp4_format(full_precision_weights):
        """
        Convert full precision weights to MXFP4 packed uint8 format.
        
        This implements the inverse of transformers' convert_moe_packed_tensors function.
        """
        logger.info(f"🔧 Quantizing full precision weights to MXFP4 format")
        logger.info(f"📥 Input shape: {full_precision_weights.shape}, dtype: {full_precision_weights.dtype}")
        
        # Move FP4_VALUES to same device as weights
        fp4_values = FP4_VALUES.to(full_precision_weights.device, full_precision_weights.dtype)
        
        # Reshape weights to process in groups of 32 (matches original format)
        # Input shape: [32, 2880, 2880] or [32, 2880, 5760]
        # We need to group into chunks of 32 values for quantization
        experts, dim1, dim2 = full_precision_weights.shape
        
        # Reshape to [experts, dim1, groups_of_32, 32] then [experts, dim1, groups_of_32, 16, 2]
        # This groups values into sets of 32, then pairs for nibble packing
        groups_of_32 = dim2 // 32
        if dim2 % 32 != 0:
            raise ValueError(f"Weight dimension {dim2} is not divisible by 32")
        
        # Reshape to group values
        weights_grouped = full_precision_weights.view(experts, dim1, groups_of_32, 32)
        weights_pairs = weights_grouped.view(experts, dim1, groups_of_32, 16, 2)
        
        logger.info(f"📦 Grouped shape: {weights_pairs.shape}")
        
        # For each group of 32 values, find the best scale (exponent)
        # and quantize the values to FP4 indices
        blocks = torch.zeros(experts, dim1, groups_of_32, 16, dtype=torch.uint8, device=full_precision_weights.device)
        scales = torch.zeros(experts, dim1, groups_of_32, dtype=torch.uint8, device=full_precision_weights.device)
        
        # Process each group of 32 values
        for e in range(experts):
            for d1 in range(dim1):
                for g in range(groups_of_32):
                    group_values = weights_grouped[e, d1, g, :]  # 32 values
                    
                    # Find the best scale for this group
                    # We need to find the exponent that minimizes quantization error
                    max_abs_val = torch.max(torch.abs(group_values))
                    
                    if max_abs_val == 0:
                        # All zeros - use scale 0 (which becomes 127 in uint8)
                        exponent = 0
                        quantized_indices = torch.zeros(32, dtype=torch.long, device=full_precision_weights.device)
                    else:
                        # Find best exponent (this is a simplified approach)
                        # In practice, this should try different exponents and pick the best one
                        log_val = torch.log2(max_abs_val / 6.0).item()
                        exponent = max(-127, min(127, int(log_val)))
                        
                        # Scale the values and quantize to FP4
                        scale_factor = 2.0 ** (-exponent)
                        scaled_values = group_values * scale_factor
                        
                        # Find closest FP4 values
                        quantized_indices = torch.zeros(32, dtype=torch.long, device=full_precision_weights.device)
                        for i in range(32):
                            val = scaled_values[i].item()
                            # Find closest FP4 value
                            distances = torch.abs(fp4_values - val)
                            closest_idx = torch.argmin(distances).item()
                            quantized_indices[i] = closest_idx
                    
                    # Pack pairs of indices into uint8 blocks (nibble packing)
                    for i in range(16):
                        idx_lo = quantized_indices[i * 2].item()      # Lower nibble (even index)
                        idx_hi = quantized_indices[i * 2 + 1].item() # Upper nibble (odd index)
                        
                        # Pack into uint8: upper nibble << 4 | lower nibble
                        packed_byte = (idx_hi << 4) | (idx_lo & 0x0F)
                        blocks[e, d1, g, i] = packed_byte
                    
                    # Store scale with +127 offset for uint8 storage
                    scales[e, d1, g] = max(0, min(255, exponent + 127))
        
        logger.info(f"✅ Quantization complete:")
        logger.info(f"   Blocks: {blocks.shape} {blocks.dtype}")
        logger.info(f"   Scales: {scales.shape} {scales.dtype}")
        
        return blocks, scales

    metadata = {}
    
    for original_name, new_name, param_tensor in expert_params_converted:
        # Generate base name for scales
        base_name = new_name.replace("_blocks", "")
        
        try:
            logger.info(f"🔧 Converting {original_name} to MXFP4 packed format")
            logger.info(f"📥 INPUT: shape={param_tensor.shape}, dtype={param_tensor.dtype}")
            
            # Quantize the full precision weights to MXFP4 packed format
            blocks, scales = quantize_weights_to_mxfp4_format(param_tensor)
            
            # Add to metadata
            metadata[new_name] = blocks
            scales_name = f"{base_name}_scales"
            metadata[scales_name] = scales
            
            logger.info(f"✅ Added {new_name}: {blocks.shape} {blocks.dtype}")
            logger.info(f"✅ Added {scales_name}: {scales.shape} {scales.dtype}")
            
        except Exception as e:
            logger.error(f"❌ Failed to quantize {original_name}: {e}")
            import traceback
            traceback.print_exc()
            logger.info("🔄 Falling back to placeholder quantization")
            return _generate_placeholder_quantization_metadata(expert_params_converted)
    
    return metadata


def _generate_placeholder_quantization_metadata(expert_params_converted):
    """
    Fallback placeholder quantization (for debugging/compatibility).
    """
    metadata = {}
    
    for original_name, new_name, param_tensor in expert_params_converted:
        # Generate base name for scales
        base_name = new_name.replace("_blocks", "")
        
        # For MXFP4, we need scales and zeros matching tensor shape[:-1]
        if param_tensor.dim() >= 2:
            # Scales should match all dims except the last one
            scales_shape = param_tensor.shape[:-1]
            
            # Generate scales (small positive values around 1.0)
            scales_name = f"{base_name}_scales"
            scales = torch.ones(scales_shape, dtype=torch.float32, device=param_tensor.device) * 0.1
            metadata[scales_name] = scales
            
            logger.debug(f"Generated placeholder metadata: {scales_name} ({scales.shape})")
    
    return metadata


def should_convert_gpt_oss_format(model_config) -> bool:
    """
    Determine if we should convert GPT-OSS format during saving.
    
    Args:
        model_config: Model configuration
        
    Returns:
        True if conversion is needed
    """
    return getattr(model_config, 'model_type', None) == 'gpt_oss'


def update_config_for_quantized_format(config_path: Path):
    """
    Update config.json to include proper GPT-OSS quantization configuration.
    
    Args:
        config_path: Path to config.json file
    """
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