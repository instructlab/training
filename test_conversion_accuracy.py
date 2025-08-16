#!/usr/bin/env python3
"""
Test our quantization conversion accuracy by comparing:
1. Original quantized model (ground truth)
2. Dequantized model (training format) 
3. Our re-quantized model (after conversion)

This will help isolate if the conversion process is introducing corruption.
"""

import torch
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_conversion_accuracy():
    """Test conversion accuracy by comparing original vs converted quantized models."""
    print("🔍 Testing Quantization Conversion Accuracy")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
        from instructlab.training.gpt_oss_utils_correct import convert_dequantized_to_quantized_format_correct
        
        model_id = "openai/gpt-oss-20b"
        
        # Step 1: Load original quantized model (ground truth)
        print(f"📥 Step 1: Loading ORIGINAL quantized model (ground truth)...")
        original_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",  # Use auto dtype
            device_map="auto",   # Use GPU
            trust_remote_code=True
            # Use default quantization config from model config.json
        )
        
        # Get original quantized state dict
        original_state = original_model.state_dict()
        print(f"   Original model loaded with {len(original_state)} parameters")
        
        # Check original parameter format
        print("\\n🔍 Original quantized parameter format:")
        expert_params_orig = {}
        
        # First, show ALL expert parameters to understand the format
        print("   All expert parameters in original model:")
        for name, param in original_state.items():
            if "experts." in name:
                print(f"     {name}: {param.shape} {param.dtype}")
        
        # Look for quantized format (_blocks, _scales)
        print("\\n   Quantized format parameters:")
        for name, param in original_state.items():
            if "experts." in name and ("_blocks" in name or "_scales" in name):
                expert_params_orig[name] = param
                print(f"     {name}: {param.shape} {param.dtype}")
                if len(expert_params_orig) >= 10:  # Show more
                    break
        
        if len(expert_params_orig) == 0:
            print("   ⚠️ No _blocks/_scales found - original model might be in dequantized format too!")
            # Fall back to regular weight parameters
            for name, param in original_state.items():
                if "experts." in name and ("gate_up_proj" in name or "down_proj" in name) and not name.endswith("_bias"):
                    expert_params_orig[name] = param
                    print(f"     {name}: {param.shape} {param.dtype}")
                    if len(expert_params_orig) >= 6:
                        break
        
        # Keep original_model for now, will delete before conversion
        
        # Step 2: Load dequantized model (training format)
        print(f"\\n📥 Step 2: Loading DEQUANTIZED model (training format)...")
        dequant_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=Mxfp4Config(dequantize=True),  # Dequantized for training
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Use GPU
            trust_remote_code=True
        )
        
        # Get dequantized state dict
        dequant_state = dequant_model.state_dict()
        print(f"   Dequantized model loaded with {len(dequant_state)} parameters")
        
        # Check dequantized parameter format
        print("\\n🔍 Dequantized parameter format:")
        expert_params_dequant = {}
        for name, param in dequant_state.items():
            if "experts." in name and ("down_proj" in name or "gate_up_proj" in name) and not name.endswith("_bias"):
                expert_params_dequant[name] = param
                print(f"   {name}: {param.shape} {param.dtype}")
                if len(expert_params_dequant) >= 4:  # Show first few
                    break
        
        # Analyze expected dimensions for quantization
        print("\\n🔍 Expected quantized dimensions analysis:")
        for name, param in expert_params_dequant.items():
            if "gate_up_proj" in name:
                # gate_up_proj: [32, 2880, 5760] -> should become [32, 5760, 90, 16]
                experts, hidden_size, two_intermediate = param.shape
                expected_blocks_shape = (experts, two_intermediate, hidden_size // 32, 16)
                expected_scales_shape = (experts, two_intermediate, hidden_size // 32)
                print(f"   {name} -> blocks: {expected_blocks_shape}, scales: {expected_scales_shape}")
            elif "down_proj" in name:
                # down_proj: [32, 5760, 2880] -> should become [32, 2880, 180, 16]  
                experts, intermediate, hidden_size = param.shape
                expected_blocks_shape = (experts, hidden_size, intermediate // 32, 16)
                expected_scales_shape = (experts, hidden_size, intermediate // 32)
                print(f"   {name} -> blocks: {expected_blocks_shape}, scales: {expected_scales_shape}")
            break  # Just show first one of each type
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Step 3: Test inference on dequantized model
        print(f"\\n🧪 Step 3: Testing inference on DEQUANTIZED model...")
        test_inference(dequant_model, tokenizer, "DEQUANTIZED")
        
        # Free GPU memory before conversion
        del dequant_model
        del original_model  # Also delete original model if not already deleted
        torch.cuda.empty_cache()
        print("   🧹 Cleared GPU memory before conversion")
        
        # Step 4: Apply our CORRECT conversion
        print(f"\\n🔄 Step 4: Applying our CORRECT quantization conversion...")
        converted_state = convert_dequantized_to_quantized_format_correct(dequant_state)
        print(f"   Converted state dict has {len(converted_state)} parameters")
        
        # Check converted parameter format
        print("\\n🔍 Converted parameter format:")
        expert_params_converted = {}
        for name, param in converted_state.items():
            if "experts." in name and ("_blocks" in name or "_scales" in name):
                expert_params_converted[name] = param
                print(f"   {name}: {param.shape} {param.dtype}")
                if len(expert_params_converted) >= 4:  # Show first few
                    break
        
        # Step 5: Save and reload converted model for fair comparison
        print(f"\\n💾 Step 5: Saving converted model...")
        output_dir = Path("test_converted_model")
        output_dir.mkdir(exist_ok=True)
        
        import safetensors.torch as safe_torch
        safe_torch.save_file(converted_state, output_dir / "model.safetensors")
        
        # Save config (use original model's config with updated quantization settings)
        import json
        from transformers import AutoConfig
        
        config_path = output_dir / "config.json"
        
        # Get original config
        try:
            original_config = AutoConfig.from_pretrained(model_id)
            config_dict = original_config.to_dict()
            
            # Ensure quantization config is set correctly
            config_dict["quantization_config"] = {
                "modules_to_not_convert": [
                    "model.layers.*.self_attn",
                    "model.layers.*.mlp.router", 
                    "model.embed_tokens",
                    "lm_head"
                ],
                "quant_method": "mxfp4"
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            print(f"   ✅ Saved config with quantization settings")
            
        except Exception as e:
            print(f"   ⚠️ Failed to copy original config: {e}")
            # Fallback to basic config
            config = {
                "model_type": "gpt_oss",
                "quantization_config": {
                    "modules_to_not_convert": [
                        "model.layers.*.self_attn",
                        "model.layers.*.mlp.router", 
                        "model.embed_tokens",
                        "lm_head"
                    ],
                    "quant_method": "mxfp4"
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        tokenizer.save_pretrained(output_dir)
        
        # Step 6: Compare safetensors files directly
        print(f"\\n📊 Step 6: Comparing safetensors files directly...")
        
        # Get original model safetensors files
        from huggingface_hub import snapshot_download
        import safetensors.torch as safe_torch
        
        try:
            # Download original model files
            print("   📥 Downloading original model files...")
            original_model_dir = snapshot_download(model_id, allow_patterns="*.safetensors")
            original_safetensor_files = list(Path(original_model_dir).glob("*.safetensors"))
            
            print(f"   📂 Original safetensor files: {[f.name for f in original_safetensor_files]}")
            
            # Load all original safetensor files
            original_tensors = {}
            for file_path in original_safetensor_files:
                file_tensors = safe_torch.load_file(file_path)
                original_tensors.update(file_tensors)
            
            print(f"   📊 Original model has {len(original_tensors)} tensors total")
            
            # Show ALL original expert parameters in safetensors
            print("\\n   🔍 ALL Original expert parameters in safetensors:")
            orig_expert_params = {}
            orig_layers_with_blocks = set()
            for name, tensor in original_tensors.items():
                if "experts." in name and ("_blocks" in name or "_scales" in name):
                    orig_expert_params[name] = tensor
                    # Extract layer number
                    layer_num = name.split('.')[2]  # model.layers.X.mlp.experts...
                    orig_layers_with_blocks.add(layer_num)
                    print(f"     {name}: {tensor.shape} {tensor.dtype}")
            
            print(f"\\n   📊 Original model has quantized expert params in layers: {sorted(orig_layers_with_blocks)}")
            print(f"   📊 Total original expert parameters: {len(orig_expert_params)}")
            
            # Load our converted safetensor file
            converted_file = output_dir / "model.safetensors"
            converted_tensors = safe_torch.load_file(converted_file)
            
            print(f"\\n   📊 Converted model has {len(converted_tensors)} tensors total")
            
            # Show ALL converted expert parameters
            print("\\n   🔍 ALL Converted expert parameters in safetensors:")
            conv_expert_params = {}
            conv_layers_with_blocks = set()
            for name, tensor in converted_tensors.items():
                if "experts." in name and ("_blocks" in name or "_scales" in name):
                    conv_expert_params[name] = tensor
                    # Extract layer number
                    layer_num = name.split('.')[2]  # model.layers.X.mlp.experts...
                    conv_layers_with_blocks.add(layer_num)
                    print(f"     {name}: {tensor.shape} {tensor.dtype}")
            
            print(f"\\n   📊 Converted model has quantized expert params in layers: {sorted(conv_layers_with_blocks)}")
            print(f"   📊 Total converted expert parameters: {len(conv_expert_params)}")
            
            # Compare layer sets
            orig_layers = set(orig_layers_with_blocks)
            conv_layers = set(conv_layers_with_blocks)
            
            print(f"\\n   🔍 Layer comparison:")
            print(f"     Original layers: {sorted(orig_layers)}")
            print(f"     Converted layers: {sorted(conv_layers)}")
            print(f"     Missing in converted: {sorted(orig_layers - conv_layers)}")
            print(f"     Extra in converted: {sorted(conv_layers - orig_layers)}")
            
            if orig_layers == conv_layers:
                print(f"     ✅ Layer sets match perfectly!")
            else:
                print(f"     ❌ Layer sets don't match - this explains the parameter mismatch!")
            
            # Compare matching parameters
            print(f"\\n   📊 Comparing matching expert parameters...")
            matches = 0
            mismatches = 0
            
            for orig_name, orig_tensor in orig_expert_params.items():
                if orig_name in conv_expert_params:
                    conv_tensor = conv_expert_params[orig_name]
                    
                    # Check shapes and dtypes
                    if orig_tensor.shape != conv_tensor.shape:
                        print(f"     ❌ Shape mismatch {orig_name}: {orig_tensor.shape} vs {conv_tensor.shape}")
                        mismatches += 1
                        continue
                        
                    if orig_tensor.dtype != conv_tensor.dtype:
                        print(f"     ❌ Dtype mismatch {orig_name}: {orig_tensor.dtype} vs {conv_tensor.dtype}")
                        mismatches += 1
                        continue
                    
                    # Compare values (sample)
                    if torch.equal(orig_tensor, conv_tensor):
                        print(f"     ✅ Perfect match: {orig_name}")
                        matches += 1
                    else:
                        # Count differences
                        diff_count = (orig_tensor != conv_tensor).sum().item()
                        total_elements = orig_tensor.numel()
                        diff_percent = (diff_count / total_elements) * 100
                        
                        print(f"     ⚠️  Partial match {orig_name}: {diff_count}/{total_elements} different ({diff_percent:.1f}%)")
                        
                        # Show some differing values
                        if diff_count > 0 and diff_count < 20:  # Only if manageable number
                            diff_positions = torch.nonzero(orig_tensor != conv_tensor)
                            for i, pos in enumerate(diff_positions[:3]):  # Show first 3
                                pos_tuple = tuple(pos.tolist())
                                orig_val = orig_tensor[pos_tuple].item()
                                conv_val = conv_tensor[pos_tuple].item()
                                print(f"         Position {pos_tuple}: {orig_val} vs {conv_val}")
                        
                        mismatches += 1
                else:
                    print(f"     ❌ Missing in converted: {orig_name}")
                    mismatches += 1
            
            # Check for extra parameters in converted
            for conv_name in conv_expert_params:
                if conv_name not in orig_expert_params:
                    print(f"     ❌ Extra in converted: {conv_name}")
                    mismatches += 1
            
            total_comparisons = len(orig_expert_params) + len([n for n in conv_expert_params if n not in orig_expert_params])
            
            print(f"\\n   📈 Safetensors Comparison Summary:")
            print(f"     ✅ Perfect matches: {matches}")
            print(f"     ❌ Mismatches: {mismatches}")
            if total_comparisons > 0:
                print(f"     📊 Match rate: {matches/max(1, matches+mismatches)*100:.1f}%")
            
            # Try to load converted model for inference test
            print(f"\\n🧪 Step 7: Testing converted model inference...")
            try:
                converted_model = AutoModelForCausalLM.from_pretrained(
                    output_dir,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True
                )
                print("   ✅ Converted model loads successfully")
                test_inference(converted_model, tokenizer, "CONVERTED")
                del converted_model
                
            except Exception as load_e:
                print(f"   ❌ Failed to load converted model for inference: {load_e}")
            
        except Exception as e:
            print(f"   ❌ Safetensors comparison failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\\n🧹 Test completed. Check results above for conversion accuracy.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_inference(model, tokenizer, stage_name):
    """Inference test using Chinese/English prompt."""
    try:
        # Use your original Chinese/English test
        test_prompt = "Please translate the following Chinese to English: 你好，世界！"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   🔤 {stage_name} OUTPUT: '{generated_text}'")
        
        # Check if output contains reasonable translation
        output_lower = generated_text.lower()
        if "hello" in output_lower and "world" in output_lower:
            print(f"   ✅ {stage_name}: Output contains correct translation!")
        elif len(generated_text.strip()) > len(test_prompt) + 10:
            print(f"   ✅ {stage_name}: Output looks reasonable (generated new text)")
        else:
            print(f"   ⚠️  {stage_name}: Output looks corrupted or repetitive!")
            
    except Exception as e:
        print(f"   ❌ {stage_name} inference failed: {e}")

if __name__ == "__main__":
    test_conversion_accuracy()