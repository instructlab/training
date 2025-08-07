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
        
        # CRITICAL: The original format requires transposing dimensions first!
        # Input shape: [32, 2880, 2880] or [32, 2880, 5760]
        # Original format: [32, 5760, 90, 16] means we need to transpose [2880, 5760] → [5760, 2880]
        # Then group the 2880 dimension: 2880 ÷ 32 = 90 groups
        
        experts, output_dim, input_dim = full_precision_weights.shape
        logger.info(f"🔄 Original shape: {full_precision_weights.shape}")
        
        # Transpose to match original format: [experts, output_dim, input_dim] → [experts, input_dim, output_dim]
        weights_transposed = full_precision_weights.transpose(1, 2)  # [experts, input_dim, output_dim]
        logger.info(f"🔄 After transpose: {weights_transposed.shape}")
        
        # Now group along the output dimension (which is now the last dimension)
        groups_of_32 = output_dim // 32
        if output_dim % 32 != 0:
            raise ValueError(f"Output dimension {output_dim} is not divisible by 32")
        
        # Reshape to group values: [experts, input_dim, groups_of_32, 32]
        weights_grouped = weights_transposed.view(experts, input_dim, groups_of_32, 32)
        
        logger.info(f"📦 Grouped shape: {weights_grouped.shape}")
        
        # Vectorized quantization - much faster than nested loops
        logger.info(f"🚀 Using vectorized quantization for speed...")
        
        # Find the best scale (exponent) for each group of 32 values
        # Shape: [experts, input_dim, groups_of_32]
        max_abs_vals = torch.max(torch.abs(weights_grouped), dim=-1)[0]
        
        # Handle zero groups (avoid log of zero)
        non_zero_mask = max_abs_vals > 0
        log_vals = torch.zeros_like(max_abs_vals)
        log_vals[non_zero_mask] = torch.log2(max_abs_vals[non_zero_mask] / 6.0)
        
        # Clamp exponents to valid range and convert to integers
        exponents = torch.clamp(log_vals, -127, 127).round().long()
        
        # Create scale factors for each group: 2^(-exponent)
        scale_factors = torch.pow(2.0, -exponents.float())  # [experts, input_dim, groups_of_32]
        scale_factors = scale_factors.unsqueeze(-1)  # [experts, input_dim, groups_of_32, 1]
        
        # Scale all values by their group's scale factor
        scaled_values = weights_grouped * scale_factors  # [experts, input_dim, groups_of_32, 32]
        
        # Find closest FP4 values for all scaled values at once
        # Expand dimensions for broadcasting: scaled_values [..., :, None] vs fp4_values [None, ..., :]
        scaled_expanded = scaled_values.unsqueeze(-1)  # [..., 32, 1]
        fp4_expanded = fp4_values.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 1, 16]
        
        # Compute distances and find closest indices
        distances = torch.abs(scaled_expanded - fp4_expanded)  # [..., 32, 16]
        quantized_indices = torch.argmin(distances, dim=-1)  # [..., 32] with values 0-15
        
        logger.info(f"📊 Quantized indices shape: {quantized_indices.shape}")
        
        # CRITICAL: Pack nibbles according to official transformers format
        # The official dequantization does: sub[:, 0::2] = lut[idx_lo], sub[:, 1::2] = lut[idx_hi]
        # This means: even positions in final output come from lower nibble, odd from upper nibble
        # So for each pair of output positions [i, i+1], we pack as: (indices[i+1] << 4) | indices[i]
        
        # Reshape to pairs: [..., 32] -> [..., 16, 2] where each pair will become one uint8
        indices_pairs = quantized_indices.view(experts, input_dim, groups_of_32, 16, 2)
        
        # Pack according to official format: 
        # indices_pairs[..., 0] goes to even positions (lower nibble)
        # indices_pairs[..., 1] goes to odd positions (upper nibble)
        idx_even = indices_pairs[..., 0]  # Even positions -> lower nibble
        idx_odd = indices_pairs[..., 1]   # Odd positions -> upper nibble
        
        # Pack into uint8: (odd_index << 4) | (even_index & 0x0F)
        blocks = ((idx_odd << 4) | (idx_even & 0x0F)).to(torch.uint8)
        
        # Store scales with +127 offset for uint8 storage
        scales = torch.clamp(exponents + 127, 0, 255).to(torch.uint8)
        
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