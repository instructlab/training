# SPDX-License-Identifier: Apache-2.0

# Standard
import math
from copy import deepcopy
from typing import Any, Tuple

# Third Party
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_scheduler

# First Party
from instructlab.training.config import DistributedBackend
from instructlab.training.setup_optimizer import setup_optimizer
from instructlab.training.utils import (
    ensure_loadable_granite_checkpoint,
    convert_loss_to_reduce_sum,
    add_noisy_embeddings,
    get_projection_layer_names,
    prepare_peft_model,
    setup_accelerator,
    apply_gradient_checkpointing,  
)
from instructlab.dolomite.hf_models import GPTDolomiteForCausalLM 

def setup_model(
    args: Any,
    tokenizer: Any,
    train_loader: Any,
    grad_accum: int
) -> Tuple[torch.nn.Module, Any, torch.optim.Optimizer, Any]:
    bnb_config = None
    if args.lora_r > 0 and args.lora_quant_bits == 4:
        # Third Party
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # if not set will throw a warning about slow speeds when training
        )

    base_model_args = {
        "pretrained_model_name_or_path": args.model_name_or_path,
        "torch_dtype": torch.bfloat16,
        "quantization_config": bnb_config,
    }
    if not args.disable_flash_attn:
        base_model_args["attn_implementation"] = "flash_attention_2"
    elif args.is_granite:
        raise RuntimeError(
            "ERROR: Trying to use padding-free transformer without flash attention is not supported"
        )

    if args.is_granite:
        with ensure_loadable_granite_checkpoint(
            args.model_name_or_path, args.output_dir
        ) as path:
            base_model_args["pretrained_model_name_or_path"] = path
            model = GPTDolomiteForCausalLM.from_pretrained(
                **base_model_args,
                use_padding_free_transformer=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(**base_model_args)

    if len(tokenizer) > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    # Fix any discrepancy between model and tokenizer
    if (
        model.config.pad_token_id is not None
        and tokenizer.pad_token_id is not None
        and model.config.pad_token_id != tokenizer.pad_token_id
    ):
        print(
            f"WARNING: There is a mismatch between pad token id of model ({model.config.pad_token_id}) and tokenizer({tokenizer.pad_token_id}). Fixing model pad token id to be same as tokenizer's pad token id"
        )
        model.config.pad_token_id = tokenizer.pad_token_id
    if (
        model.config.bos_token_id is not None
        and tokenizer.bos_token_id is not None
        and model.config.bos_token_id != tokenizer.bos_token_id
    ):
        print(
            f"WARNING: There is a mismatch between bos token id of model({model.config.bos_token_id}) and tokenizer({tokenizer.bos_token_id}). Fixing model bos token id to be same as tokenizer's bos token id"
        )
        model.config.bos_token_id = tokenizer.bos_token_id
    if (
        model.config.eos_token_id is not None
        and tokenizer.eos_token_id
        and model.config.eos_token_id != tokenizer.eos_token_id
    ):
        print(
            f"WARNING: There is a mismatch between eos token id of model({model.config.eos_token_id}) and tokenizer({tokenizer.eos_token_id}). Fixing model eos token id to be same as tokenizer's eos token id"
        )
        model.config.eos_token_id = tokenizer.eos_token_id

    assert model.__class__.__name__ in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM",
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "MixtralForCausalLM",
    ], f"Model class name: {model.__class__.__name__} is not supported."

    model = convert_loss_to_reduce_sum(model, is_granite=args.is_granite)
    model = add_noisy_embeddings(model, noise_alpha=args.NEFTune_alpha)

    # handling of gradient checkpointing
    # it is handled differently for lora and full
    # - with the exception of granite, which handles it
    #   in the later stanza
    if args.lora_r > 0:
        # if lora
        # Third Party
        from peft import LoraConfig

        # ensure we select only the modules that exist in the model
        proj_layers = get_projection_layer_names(model)
        if not args.lora_target_modules:
            print(
                f"WARNING: lora_target_modules was not specified, defaulting to all of the model's projection modules"
            )
            if not proj_layers:
                raise RuntimeError("could not find any projection layers in the model")
            args.__dict__["lora_target_modules"] = proj_layers
        else:
            # when the user specifies the module, we should verify that they align with what's in the model
            lora_target_modules_set = set(args.lora_target_modules)
            diff = lora_target_modules_set - set(proj_layers)
            layers_to_target = lora_target_modules_set - diff
            if len(diff) == len(args.lora_target_modules):
                raise ValueError(
                    f"None of the modules you requested exist in the model.\nRequested modules: {args.lora_target_modules}; Available modules: {proj_layers}.\nThis is usually a misconfiuration error. Consider omitting your `lora_target_modules` list to have these discovered automatically."
                )
            if diff:
                print(
                    f"\033[33mWARNING: the following modules were targeted for LoRA but are not present in the model: {list(diff)}. Applying LoRA only to {list(layers_to_target)} modules.\033[0m"
                )
            args.__dict__["lora_target_modules"] = list(layers_to_target)

        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )
        model = prepare_peft_model(
            model, peft_config, gradient_checkpointing=not args.is_granite
        )

    elif not args.is_granite:
        model.gradient_checkpointing_enable()

    # granite gradient checkpointing is handled uniformly
    # for both lora and full here
    if args.is_granite:
        block_name = model._no_split_modules[0]
        apply_gradient_checkpointing(
            model,
            block_name=block_name,
            use_reentrant=True,  # this should be the HF default mode
        )

        if args.lora_r > 0:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    accelerator = setup_accelerator(args, model, grad_accum)
    if args.distributed_training_framework == DistributedBackend.FSDP.value:
        model = accelerator.prepare(model)
    optimizer = setup_optimizer(args, model)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_epochs * len(train_loader) // grad_accum,
    )
    model, optimizer, _, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        deepcopy(train_loader),
        lr_scheduler,
    )
    return model, lr_scheduler, optimizer, accelerator
