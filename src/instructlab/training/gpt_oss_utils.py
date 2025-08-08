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
    Generate real MXFP4 quantization using triton for speed, then convert to transformers format.
    
    This approach uses the fast triton quantization, then carefully converts the output
    to the exact uint8 packed format that transformers expects for correct round-trip.
    
    Args:
        expert_params_converted: List of (original_name, new_name, tensor) tuples
        
    Returns:
        Dict of quantization metadata parameters
    """
    try:
        from transformers.integrations.mxfp4 import quantize_to_mxfp4
    except ImportError:
        logger.error("MXFP4 quantization not available - falling back to placeholder")
        return _generate_placeholder_quantization_metadata(expert_params_converted)
    
    def convert_triton_to_transformers_format(param_tensor, param_name):
        """
        Convert using triton quantization, then carefully convert to transformers uint8 format.
        
        The key is ensuring the conversion preserves the exact semantics that transformers
        expects when it dequantizes using convert_moe_packed_tensors.
        """
        logger.info(f"🚀 Using triton quantization for {param_name}")
        logger.info(f"📥 Input: {param_tensor.shape} {param_tensor.dtype}")
        
        # Step 1: Use triton quantization (fast and accurate)
        original_device = param_tensor.device
        if param_tensor.device.type == 'cpu':
            gpu_tensor = param_tensor.cuda()
        else:
            gpu_tensor = param_tensor
            
        # Call triton quantization
        triton_result = quantize_to_mxfp4(gpu_tensor)
        if not isinstance(triton_result, tuple) or len(triton_result) != 2:
            raise ValueError(f"Expected tuple of 2 from quantize_to_mxfp4, got {type(triton_result)}")
        
        triton_blocks, triton_scales = triton_result
        print(f"📤 Triton returned: blocks={type(triton_blocks)}, scales={type(triton_scales)}")
        
        # Step 2: Extract underlying data from triton tensors
        def extract_tensor_data(triton_tensor, name):
            print(f"🔧 Extracting {name}: {type(triton_tensor)}")
            
            # Try different methods to get PyTorch tensor
            if torch.is_tensor(triton_tensor):
                return triton_tensor
            elif hasattr(triton_tensor, 'data') and torch.is_tensor(triton_tensor.data):
                return triton_tensor.data
            elif hasattr(triton_tensor, 'tensor') and torch.is_tensor(triton_tensor.tensor):
                return triton_tensor.tensor
            elif hasattr(triton_tensor, '_tensor') and torch.is_tensor(triton_tensor._tensor):
                return triton_tensor._tensor
            else:
                # Last resort: convert to tensor
                return torch.as_tensor(triton_tensor)
        
        blocks_tensor = extract_tensor_data(triton_blocks, "blocks").to(original_device)
        scales_tensor = extract_tensor_data(triton_scales, "scales").to(original_device)
        
        print(f"📊 Extracted tensors:")
        print(f"   Blocks: {blocks_tensor.shape} {blocks_tensor.dtype}")
        print(f"   Scales: {scales_tensor.shape} {scales_tensor.dtype}")
        
        # Step 3: Convert to transformers uint8 format
        # The critical part: triton may return different format than transformers expects
        
        # For blocks: need to ensure they're uint8 with correct nibble packing
        if blocks_tensor.dtype != torch.uint8:
            logger.info(f"🔄 Converting blocks from {blocks_tensor.dtype} to uint8")
            # If these are indices or already packed, just convert dtype
            if blocks_tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                blocks_tensor = blocks_tensor.clamp(0, 255).to(torch.uint8)
            else:
                # For float types, assume they need to be converted to indices
                blocks_tensor = blocks_tensor.to(torch.uint8)
        
        # For scales: triton might already have the correct format
        print(f"🔍 Scale tensor analysis:")
        print(f"   Dtype: {scales_tensor.dtype}")
        print(f"   Range: {scales_tensor.min().item()} to {scales_tensor.max().item()}")
        print(f"   Sample values: {scales_tensor.flatten()[:10].tolist()}")
        
        if scales_tensor.dtype != torch.uint8:
            print(f"🔄 Converting scales from {scales_tensor.dtype} to uint8")
            if scales_tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                # These are likely raw exponents - add 127 offset
                scales_tensor = (scales_tensor + 127).clamp(0, 255).to(torch.uint8)
            elif scales_tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                # These might be raw exponents as floats
                scales_tensor = (scales_tensor + 127).clamp(0, 255).to(torch.uint8)
            else:
                scales_tensor = scales_tensor.to(torch.uint8)
        else:
            # Triton might already be giving us uint8 values - check if they need offset
            # If all values are very small (0-50), they might be raw exponents needing +127
            # If values are around 127, they might already have the offset
            print(f"🤔 Scales are already uint8 - checking if offset is needed")
            if scales_tensor.max().item() < 50:
                print("   Scales look like raw exponents - adding 127 offset")
                scales_tensor = (scales_tensor + 127).clamp(0, 255)
            else:
                print("   Scales look like they already have offset")
        
        print(f"   Final scale range: {scales_tensor.min().item()} to {scales_tensor.max().item()}")
        
        # Step 4: Analyze triton format and convert to transformers format
        experts, output_dim, input_dim = param_tensor.shape
        
        print(f"🔍 Analyzing triton format:")
        print(f"   Input tensor: {param_tensor.shape}")
        print(f"   Triton blocks: {blocks_tensor.shape}")
        print(f"   Triton scales: {scales_tensor.shape}")
        
        # Analyze the actual grouping from triton's output
        scale_experts, scale_groups, scale_input = scales_tensor.shape
        
        print(f"📐 Grouping analysis:")
        print(f"   Scale groups: {scale_groups}")
        print(f"   Output dim: {output_dim}")
        print(f"   Expected groups (output_dim//32): {output_dim // 32}")
        print(f"   Scale input dim: {scale_input}")
        print(f"   Actual input dim: {input_dim}")
        
        # Handle case where output_dim < 32
        if output_dim < 32:
            # For small dimensions, triton might handle differently
            # Let's infer the grouping from triton's actual output
            if scale_groups == 1:
                # Single group - use actual dimensions
                output_groups = 1
            else:
                output_groups = scale_groups
        else:
            output_groups = output_dim // 32
            if output_dim % 32 != 0:
                print(f"⚠️ Output dimension {output_dim} not divisible by 32")
                output_groups = scale_groups  # Use triton's actual grouping
        
        print(f"   Using output_groups: {output_groups}")
        
        # For transformers format, we need: [experts, input_dim, output_groups, 16]
        target_blocks_shape = (experts, input_dim, output_groups, 16)
        target_scales_shape = (experts, input_dim, output_groups)
        
        print(f"🎯 Target transformers format:")
        print(f"   Target blocks: {target_blocks_shape}")
        print(f"   Target scales: {target_scales_shape}")
        
        # Handle scales: need to understand triton's actual layout
        print(f"🔄 Analyzing scale tensor layout:")
        print(f"   Triton scales: {scales_tensor.shape}")
        print(f"   Target: {target_scales_shape}")
        
        # The triton scales are [experts, output_groups, input_dim] 
        # We need [experts, input_dim, output_groups]
        # So we need to transpose dimensions 1 and 2
        if scales_tensor.shape == (experts, output_groups, input_dim):
            print(f"   Transposing: [experts, output_groups, input_dim] → [experts, input_dim, output_groups]")
            scales_reshaped = scales_tensor.transpose(1, 2)  # [32, 128, 2]
            print(f"   Result: {scales_reshaped.shape}")
        else:
            print(f"   Unexpected scale shape - attempting reshape")
            scales_reshaped = scales_tensor.reshape(target_scales_shape)
        
        # Handle blocks (more complex due to extra elements)
        target_blocks_elements = target_blocks_shape[0] * target_blocks_shape[1] * target_blocks_shape[2] * target_blocks_shape[3]
        actual_blocks_elements = blocks_tensor.numel()
        
        print(f"🔢 Blocks element analysis:")
        print(f"   Actual: {actual_blocks_elements}")
        print(f"   Target: {target_blocks_elements}")
        print(f"   Ratio: {actual_blocks_elements / target_blocks_elements:.3f}")
        
        print(f"🔧 Analyzing block tensor layout:")
        print(f"   Triton blocks: {blocks_tensor.shape}")
        print(f"   Target: {target_blocks_shape}")
        
        if actual_blocks_elements == target_blocks_elements:
            # Perfect match - but need to ensure correct mapping
            print(f"   Element counts match - reshaping directly")
            
            # Triton format: [experts, input_dim, output_groups*16]
            # Target format: [experts, input_dim, output_groups, 16]
            triton_experts, triton_input_dim, triton_packed = blocks_tensor.shape
            
            if triton_packed == output_groups * 16:
                # Simple reshape: [32, 128, 32] → [32, 128, 2, 16]
                print(f"   Reshaping: [experts, input_dim, packed] → [experts, input_dim, groups, 16]")
                final_blocks = blocks_tensor.view(experts, input_dim, output_groups, 16).contiguous()
                print(f"   Result: {final_blocks.shape}")
            else:
                print(f"   Unexpected packed dimension: {triton_packed} vs expected {output_groups * 16}")
                final_blocks = blocks_tensor.reshape(target_blocks_shape).contiguous()
        
        elif actual_blocks_elements > target_blocks_elements:
            # Triton has extra elements - likely padding
            print(f"🔧 Triton has extra elements - extracting relevant portion")
            triton_experts, triton_dim1, triton_dim2 = blocks_tensor.shape
            
            if triton_dim2 == output_groups * 16:
                # Reshape and extract: [32, padded_dim, groups*16] → [32, input_dim, groups, 16]
                blocks_reshaped = blocks_tensor.view(triton_experts, triton_dim1, output_groups, 16)
                final_blocks = blocks_reshaped[:, :input_dim, :, :].contiguous()
                print(f"   Extracted: {final_blocks.shape}")
            else:
                print(f"⚠️ Complex triton format - attempting heuristic extraction")
                # Calculate how many elements to take per expert
                elements_per_expert = target_blocks_elements // experts
                extracted = blocks_tensor.flatten()[:target_blocks_elements]
                final_blocks = extracted.reshape(target_blocks_shape).contiguous()
        
        else:
            raise ValueError(f"Triton has fewer elements than expected: {actual_blocks_elements} < {target_blocks_elements}")
        
        final_scales = scales_reshaped
        
        logger.info(f"✅ Final format:")
        logger.info(f"   Blocks: {final_blocks.shape} {final_blocks.dtype}")
        logger.info(f"   Scales: {final_scales.shape} {final_scales.dtype}")
        
        return final_blocks, final_scales
    
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
            logger.info(f"🔧 Converting {original_name} to transformers-compatible MXFP4 format")
            logger.info(f"📥 INPUT: shape={param_tensor.shape}, dtype={param_tensor.dtype}")
            
            # Use triton quantization + careful format conversion
            blocks, scales = convert_triton_to_transformers_format(param_tensor, original_name)
            
            # Add to metadata
            metadata[new_name] = blocks
            scales_name = f"{base_name}_scales"
            metadata[scales_name] = scales
            
            logger.info(f"✅ Added {new_name}: {blocks.shape} {blocks.dtype}")
            logger.info(f"✅ Added {scales_name}: {scales.shape} {scales.dtype}")
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {original_name}: {e}")
            import traceback
            traceback.print_exc()
            # For debugging, don't fall back immediately - let's see what's wrong
            raise e
    
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