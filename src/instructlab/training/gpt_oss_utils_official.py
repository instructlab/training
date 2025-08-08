# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific utilities using OFFICIAL transformers quantization.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger("instructlab.training")


def convert_dequantized_to_quantized_format(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert dequantized GPT-OSS parameters back to quantized format using official transformers approach.
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
    
    # Generate official quantization for converted expert parameters
    if expert_params_converted:
        logger.info(f"Generating official MXFP4 quantization for {len(expert_params_converted)} expert parameters")
        metadata = _generate_official_quantization_metadata(expert_params_converted)
        
        # Remove original dequantized parameters and add quantized versions
        for original_name, new_name, param_tensor in expert_params_converted:
            if original_name in converted_state_dict:
                del converted_state_dict[original_name]  # Remove dequantized version
        
        converted_state_dict.update(metadata)
        logger.info(f"Added {len(metadata)} quantized parameters (blocks + scales)")
    
    logger.info(f"Conversion complete: {len(converted_state_dict)} total parameters")
    return converted_state_dict


def _generate_official_quantization_metadata(expert_params_converted):
    """
    Generate MXFP4 quantization using the exact same approach as transformers.
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
            logger.info(f"🔧 Converting {original_name} using official transformers quantization")
            logger.info(f"📥 INPUT: shape={param_tensor.shape}, dtype={param_tensor.dtype}")
            
            # CRITICAL: Handle gate_up_proj orientation (from transformers conversion script)
            if "gate_up_proj" in original_name:
                logger.info(f"🔄 Transposing gate_up_proj for transformers format")
                # From transformers: gate_up_proj should be [experts, 2*intermediate_size, hidden_size]
                param_tensor = param_tensor.transpose(1, 2)
                logger.info(f"📥 After transpose: {param_tensor.shape}")
            
            # Move to GPU for triton if needed
            original_device = param_tensor.device
            if param_tensor.device.type == 'cpu' and torch.cuda.is_available():
                gpu_tensor = param_tensor.cuda()
            else:
                gpu_tensor = param_tensor
            
            # Use exact same quantization as transformers (with bfloat16 like they do)
            logger.info(f"⚡ Running official transformers quantization...")
            gpu_tensor = gpu_tensor.to(torch.bfloat16)
            triton_blocks, triton_scales = quantize_to_mxfp4(gpu_tensor)
            
            # Extract PyTorch tensors (same as transformers conversion script)
            def extract_tensor(triton_tensor):
                if torch.is_tensor(triton_tensor):
                    return triton_tensor
                elif hasattr(triton_tensor, 'data'):
                    return triton_tensor.data
                elif hasattr(triton_tensor, 'tensor'):
                    return triton_tensor.tensor
                elif hasattr(triton_tensor, '_tensor'):
                    return triton_tensor._tensor
                else:
                    return torch.as_tensor(triton_tensor)
            
            blocks = extract_tensor(triton_blocks).to(original_device)
            scales = extract_tensor(triton_scales).to(original_device)
            
            logger.info(f"📦 Triton output: blocks={blocks.shape}, scales={scales.shape}")
            
            # Convert to uint8 format (same as transformers)
            if blocks.dtype != torch.uint8:
                blocks = blocks.to(torch.uint8)
            
            if scales.dtype != torch.uint8:
                if scales.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                    scales = (scales + 127).clamp(0, 255).to(torch.uint8)
                elif scales.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    scales = (scales + 127).clamp(0, 255).to(torch.uint8)
                else:
                    scales = scales.to(torch.uint8)
            elif scales.max().item() < 50:
                scales = (scales + 127).clamp(0, 255)
            
            # Add to metadata (use triton's output format directly)
            metadata[new_name] = blocks
            scales_name = f"{base_name}_scales"
            metadata[scales_name] = scales
            
            logger.info(f"✅ Added {new_name}: {blocks.shape} {blocks.dtype}")
            logger.info(f"✅ Added {scales_name}: {scales.shape} {scales.dtype}")
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {original_name}: {e}")
            import traceback
            traceback.print_exc()
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
    """
    return getattr(model_config, 'model_type', None) == 'gpt_oss'


def update_config_for_quantized_format(config_path: Path):
    """
    Update config.json to include proper GPT-OSS quantization configuration.
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