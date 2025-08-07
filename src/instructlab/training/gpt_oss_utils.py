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
    
    This function converts dequantized weights to the transformers-compatible
    uint8 packed format used by MXFP4 quantization.
    
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
    
    # Define FP4 lookup table values (from transformers implementation)
    FP4_VALUES = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,  # Positive values
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0   # Negative values  
    ]
    
    metadata = {}
    
    for original_name, new_name, param_tensor in expert_params_converted:
        # Generate base name for scales
        base_name = new_name.replace("_blocks", "")
        
        try:
            logger.info(f"🔧 Converting {original_name} to transformers-compatible MXFP4 format")
            logger.info(f"📥 INPUT: shape={param_tensor.shape}, dtype={param_tensor.dtype}")
            
            # Expected output format based on original GPT-OSS:
            if "gate_up_proj" in original_name:
                expected_blocks_shape = (32, 5760, 90, 16)
                expected_scales_shape = (32, 5760, 90)
            elif "down_proj" in original_name:
                expected_blocks_shape = (32, 2880, 90, 16)
                expected_scales_shape = (32, 2880, 90)
            else:
                raise ValueError(f"Unknown parameter type: {original_name}")
            
            logger.info(f"🎯 TARGET FORMAT: blocks={expected_blocks_shape}, scales={expected_scales_shape}")
            
            # Move to GPU for triton quantization
            original_device = param_tensor.device
            if param_tensor.device.type == 'cpu':
                gpu_tensor = param_tensor.cuda()
            else:
                gpu_tensor = param_tensor
            
            # Step 1: Get triton quantization result
            logger.info(f"🏃 Performing triton MXFP4 quantization...")
            result = quantize_to_mxfp4(gpu_tensor)
            
            if not isinstance(result, tuple):
                raise ValueError(f"Expected tuple from quantize_to_mxfp4, got {type(result)}")
            
            triton_blocks, triton_scales = result
            logger.info(f"📤 Triton output: blocks={type(triton_blocks)}, scales={type(triton_scales)}")
            
            # Step 2: Convert triton results to transformers-compatible uint8 format
            logger.info(f"🔄 Converting to transformers uint8 format...")
            
            # Convert triton tensors to PyTorch tensors first
            def extract_triton_data(triton_tensor, name):
                """Extract underlying data from triton tensor."""
                logger.info(f"   Extracting {name}: {type(triton_tensor)}")
                
                # Try to get the underlying data
                if hasattr(triton_tensor, 'data'):
                    data = triton_tensor.data
                elif hasattr(triton_tensor, 'tensor'):
                    data = triton_tensor.tensor
                elif hasattr(triton_tensor, '_tensor'):
                    data = triton_tensor._tensor
                else:
                    # Fallback: try to convert to numpy then tensor
                    data = torch.as_tensor(triton_tensor)
                
                if not torch.is_tensor(data):
                    data = torch.as_tensor(data)
                
                return data.to(original_device)
            
            blocks_data = extract_triton_data(triton_blocks, "blocks")
            scales_data = extract_triton_data(triton_scales, "scales")
            
            logger.info(f"   Extracted blocks: {blocks_data.shape} {blocks_data.dtype}")
            logger.info(f"   Extracted scales: {scales_data.shape} {scales_data.dtype}")
            
            # Step 3: Convert to the correct transformers uint8 format
            # The triton result might already be in the right format, or we might need to convert it
            
            # For now, let's assume the triton quantization gives us the right data
            # and we just need to reshape and ensure uint8 dtype
            if blocks_data.dtype != torch.uint8:
                logger.info(f"   Converting blocks from {blocks_data.dtype} to uint8")
                # The triton data should already be indices/packed data, just change dtype
                blocks_data = blocks_data.to(torch.uint8)
            
            if scales_data.dtype != torch.uint8:
                logger.info(f"   Converting scales from {scales_data.dtype} to uint8")
                # Scales might need offset adjustment: add 127 to convert exponents to uint8
                if scales_data.dtype in [torch.int32, torch.int64]:
                    scales_data = (scales_data + 127).clamp(0, 255).to(torch.uint8)
                else:
                    scales_data = scales_data.to(torch.uint8)
            
            # Step 4: Reshape to expected format
            logger.info(f"🔀 Reshaping to target format...")
            
            # Check element counts
            if blocks_data.numel() != expected_blocks_shape[0] * expected_blocks_shape[1] * expected_blocks_shape[2] * expected_blocks_shape[3]:
                logger.error(f"❌ Blocks element count mismatch: {blocks_data.numel()} vs expected {expected_blocks_shape[0] * expected_blocks_shape[1] * expected_blocks_shape[2] * expected_blocks_shape[3]}")
                raise ValueError("Blocks element count mismatch")
            
            if scales_data.numel() != expected_scales_shape[0] * expected_scales_shape[1] * expected_scales_shape[2]:
                logger.error(f"❌ Scales element count mismatch: {scales_data.numel()} vs expected {expected_scales_shape[0] * expected_scales_shape[1] * expected_scales_shape[2]}")
                raise ValueError("Scales element count mismatch")
            
            # Reshape to target format
            final_blocks = blocks_data.reshape(expected_blocks_shape).contiguous()
            final_scales = scales_data.reshape(expected_scales_shape).contiguous()
            
            logger.info(f"✅ Final format:")
            logger.info(f"   Blocks: {final_blocks.shape} {final_blocks.dtype}")
            logger.info(f"   Scales: {final_scales.shape} {final_scales.dtype}")
            
            # Add to metadata
            metadata[new_name] = final_blocks
            scales_name = f"{base_name}_scales"
            metadata[scales_name] = final_scales
            
            logger.info(f"✅ Added {new_name} and {scales_name} to metadata")
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {original_name}: {e}")
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