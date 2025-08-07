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
    
    metadata = {}
    
    for original_name, new_name, param_tensor in expert_params_converted:
        # Generate base name for scales
        base_name = new_name.replace("_blocks", "")
        
        try:
            # Perform actual MXFP4 quantization (requires GPU for Triton kernels)
            logger.info(f"🔧 Performing real MXFP4 quantization for {original_name}")
            logger.info(f"📥 INPUT: shape={param_tensor.shape}, dtype={param_tensor.dtype}, device={param_tensor.device}")
            
            # Expected output format based on original GPT-OSS:
            if "gate_up_proj" in original_name:
                expected_blocks_shape = (32, 5760, 90, 16)
                expected_scales_shape = (32, 5760, 90)
                logger.info(f"🎯 EXPECTED OUTPUT: blocks={expected_blocks_shape}, scales={expected_scales_shape}")
            elif "down_proj" in original_name:
                expected_blocks_shape = (32, 2880, 90, 16)
                expected_scales_shape = (32, 2880, 90)
                logger.info(f"🎯 EXPECTED OUTPUT: blocks={expected_blocks_shape}, scales={expected_scales_shape}")
            
            # Move to GPU if not already there
            original_device = param_tensor.device
            if param_tensor.device.type == 'cpu':
                gpu_tensor = param_tensor.cuda()
                logger.info(f"📱 Moved to GPU: {gpu_tensor.device}")
            else:
                gpu_tensor = param_tensor
                logger.info(f"📱 Already on GPU: {gpu_tensor.device}")
            
            # Perform quantization on GPU
            logger.info(f"🏃 Calling quantize_to_mxfp4...")
            result = quantize_to_mxfp4(gpu_tensor)
            logger.info(f"📤 quantize_to_mxfp4 returned: {type(result)}")
            
            if isinstance(result, tuple):
                quantized_blocks, scales = result
                logger.info(f"📦 Unpacked tuple: blocks={type(quantized_blocks)}, scales={type(scales)}")
            else:
                logger.error(f"❌ Expected tuple, got {type(result)}")
                raise ValueError(f"Unexpected return type from quantize_to_mxfp4: {type(result)}")
            
            # Check what we actually got
            logger.info(f"📊 ACTUAL OUTPUT:")
            logger.info(f"   Blocks: type={type(quantized_blocks)}, shape={getattr(quantized_blocks, 'shape', 'no shape')}, device={getattr(quantized_blocks, 'device', 'no device attr')}")
            logger.info(f"   Scales: type={type(scales)}, shape={getattr(scales, 'shape', 'no shape')}, device={getattr(scales, 'device', 'no device attr')}")
            
            # Convert triton tensors to PyTorch tensors to match original format
            # Original GPT-OSS uses: blocks=torch.uint8, scales=torch.uint8
            logger.info(f"🔄 Converting triton tensors to PyTorch tensors...")
            
            # Try multiple methods to extract PyTorch tensor from triton tensor
            def convert_triton_to_torch(triton_tensor, target_name):
                logger.info(f"🔧 Converting {target_name}: {type(triton_tensor)}")
                methods = ['data', 'tensor', '_tensor', 'value', '__array__']
                
                for method in methods:
                    if hasattr(triton_tensor, method):
                        try:
                            logger.info(f"   Trying method: {method}")
                            attr = getattr(triton_tensor, method)
                            if callable(attr):
                                result = attr()
                            else:
                                result = attr
                            
                            logger.info(f"   Method {method} returned: {type(result)}")
                            
                            # Convert to PyTorch tensor if needed
                            if not torch.is_tensor(result):
                                result = torch.as_tensor(result)
                            
                            logger.info(f"   Converted to tensor: shape={result.shape}, dtype={result.dtype}")
                            return result.to(original_device)
                        except Exception as e:
                            logger.debug(f"   Method {method} failed for {target_name}: {e}")
                            continue
                
                # Last resort: try to get underlying data
                available_attrs = [attr for attr in dir(triton_tensor) if not attr.startswith('__')]
                logger.error(f"❌ Failed to convert {target_name}. Available attributes: {available_attrs}")
                raise RuntimeError(f"Could not convert triton tensor {target_name} to PyTorch tensor")
            
            quantized_blocks = convert_triton_to_torch(quantized_blocks, "blocks")
            scales = convert_triton_to_torch(scales, "scales")
            
            logger.info(f"✅ Converted tensors:")
            logger.info(f"   Blocks: {quantized_blocks.shape} {quantized_blocks.dtype} {quantized_blocks.device}")
            logger.info(f"   Scales: {scales.shape} {scales.dtype} {scales.device}")
            
            # Ensure correct dtypes to match original format
            if quantized_blocks.dtype != torch.uint8:
                logger.info(f"Converting blocks from {quantized_blocks.dtype} to torch.uint8")
                quantized_blocks = quantized_blocks.to(torch.uint8)
                
            if scales.dtype != torch.uint8:
                logger.info(f"Converting scales from {scales.dtype} to torch.uint8") 
                scales = scales.to(torch.uint8)
            
            # Reshape tensors to match original GPT-OSS format
            # Original format: blocks=[experts, dim, 90, 16], scales=[experts, dim, 90]
            logger.info(f"🔀 Reshaping tensors to match original GPT-OSS format...")
            
            # Check if reshaping is even possible (total elements must match)
            blocks_total_elements = quantized_blocks.numel()
            scales_total_elements = scales.numel()
            
            expected_blocks_elements = 32 * expected_blocks_shape[1] * 90 * 16
            expected_scales_elements = 32 * expected_scales_shape[1] * 90
            
            logger.info(f"📊 Element count check:")
            logger.info(f"   Blocks: actual={blocks_total_elements}, expected={expected_blocks_elements}")
            logger.info(f"   Scales: actual={scales_total_elements}, expected={expected_scales_elements}")
            
            if blocks_total_elements != expected_blocks_elements:
                logger.error(f"❌ Blocks element count mismatch! Cannot reshape {quantized_blocks.shape} -> {expected_blocks_shape}")
                logger.error(f"   Actual elements: {blocks_total_elements}, Expected: {expected_blocks_elements}")
                raise ValueError(f"Cannot reshape blocks tensor: element count mismatch")
            
            if scales_total_elements != expected_scales_elements:
                logger.error(f"❌ Scales element count mismatch! Cannot reshape {scales.shape} -> {expected_scales_shape}")
                logger.error(f"   Actual elements: {scales_total_elements}, Expected: {expected_scales_elements}")
                raise ValueError(f"Cannot reshape scales tensor: element count mismatch")
            
            # Reshape blocks tensor (use reshape instead of view for non-contiguous tensors)
            if quantized_blocks.shape != expected_blocks_shape:
                logger.info(f"🔀 Reshaping blocks from {quantized_blocks.shape} to {expected_blocks_shape}")
                quantized_blocks = quantized_blocks.reshape(expected_blocks_shape)
                logger.info(f"✅ Blocks reshaped successfully")
            else:
                logger.info(f"✅ Blocks already correct shape: {quantized_blocks.shape}")
            
            # Reshape scales tensor (use reshape instead of view for non-contiguous tensors)
            if scales.shape != expected_scales_shape:
                logger.info(f"🔀 Reshaping scales from {scales.shape} to {expected_scales_shape}")
                scales = scales.reshape(expected_scales_shape)
                logger.info(f"✅ Scales reshaped successfully")
            else:
                logger.info(f"✅ Scales already correct shape: {scales.shape}")
            
            # Ensure tensors are contiguous for safetensors compatibility
            quantized_blocks = quantized_blocks.contiguous()
            scales = scales.contiguous()
            
            # Verify we now have PyTorch tensors
            logger.info(f"After conversion: blocks={type(quantized_blocks)}, scales={type(scales)}")
            
            # Add quantized blocks (this replaces the original parameter)
            metadata[new_name] = quantized_blocks
            
            # Add scales
            scales_name = f"{base_name}_scales"
            metadata[scales_name] = scales
            
            logger.info(f"Real quantization: {new_name} → blocks: {quantized_blocks.shape} ({quantized_blocks.dtype}), scales: {scales.shape} ({scales.dtype})")
            
        except Exception as e:
            logger.error(f"Failed to quantize {original_name}: {e}")
            logger.info("Falling back to placeholder quantization")
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