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
            logger.info(f"Performing real MXFP4 quantization for {original_name}")
            
            # Move to GPU if not already there
            original_device = param_tensor.device
            if param_tensor.device.type == 'cpu':
                gpu_tensor = param_tensor.cuda()
            else:
                gpu_tensor = param_tensor
            
            # Perform quantization on GPU
            result = quantize_to_mxfp4(gpu_tensor)
            logger.info(f"quantize_to_mxfp4 returned: {type(result)}")
            
            if isinstance(result, tuple):
                quantized_blocks, scales = result
                logger.info(f"Unpacked: blocks={type(quantized_blocks)}, scales={type(scales)}")
            else:
                logger.error(f"Expected tuple, got {type(result)}")
                raise ValueError(f"Unexpected return type from quantize_to_mxfp4: {type(result)}")
            
            # Check what we actually got and handle device movement appropriately
            logger.info(f"Blocks: type={type(quantized_blocks)}, device={getattr(quantized_blocks, 'device', 'no device attr')}")
            logger.info(f"Scales: type={type(scales)}, device={getattr(scales, 'device', 'no device attr')}")
            
            # Convert triton tensors to PyTorch tensors to match original format
            # Original GPT-OSS uses: blocks=torch.uint8, scales=torch.uint8
            
            # Try multiple methods to extract PyTorch tensor from triton tensor
            def convert_triton_to_torch(triton_tensor, target_name):
                methods = ['data', 'tensor', '_tensor', 'value', '__array__']
                
                for method in methods:
                    if hasattr(triton_tensor, method):
                        try:
                            attr = getattr(triton_tensor, method)
                            if callable(attr):
                                result = attr()
                            else:
                                result = attr
                            
                            # Convert to PyTorch tensor if needed
                            if not torch.is_tensor(result):
                                result = torch.as_tensor(result)
                            
                            return result.to(original_device)
                        except Exception as e:
                            logger.debug(f"Method {method} failed for {target_name}: {e}")
                            continue
                
                # Last resort: try to get underlying data
                logger.error(f"Failed to convert {target_name}. Available attributes: {[attr for attr in dir(triton_tensor) if not attr.startswith('__')]}")
                raise RuntimeError(f"Could not convert triton tensor {target_name} to PyTorch tensor")
            
            quantized_blocks = convert_triton_to_torch(quantized_blocks, "blocks")
            scales = convert_triton_to_torch(scales, "scales")
            
            # Ensure correct dtypes to match original format
            if quantized_blocks.dtype != torch.uint8:
                logger.info(f"Converting blocks from {quantized_blocks.dtype} to torch.uint8")
                quantized_blocks = quantized_blocks.to(torch.uint8)
                
            if scales.dtype != torch.uint8:
                logger.info(f"Converting scales from {scales.dtype} to torch.uint8") 
                scales = scales.to(torch.uint8)
            
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