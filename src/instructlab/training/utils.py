# SPDX-License-Identifier: Apache-2.0

# Standard
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Tuple
import importlib
import inspect
import logging
import os
import random
import subprocess
import sys
import time
import traceback
import warnings

# Third Party
# pylint: disable=no-name-in-module
from accelerate import Accelerator, DistributedType
from peft import LoraConfig, LoraModel
from torch import distributed as dist
from torch import nn
from torch.distributed import get_rank, is_initialized
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import numpy as np
import torch
import torch.nn.functional as F

# First Party
from instructlab.training.config import (
    DistributedBackend,
    QuantizeDataType,
    TrainingArgs,
)
from instructlab.training.gpt_oss_utils_correct import (
    add_gpt_oss_quantization_config,
    convert_dequantized_to_quantized_format_correct,
    is_gpt_oss,
)
from instructlab.training.model import Model

logger = logging.getLogger("instructlab.training")


def check_valid_train_args(train_args: TrainingArgs):
    # early validation logic here
    if train_args.max_batch_len < train_args.max_seq_len:
        raise ValueError(
            f"the `max_batch_len` cannot be less than `max_seq_len`: {train_args.max_batch_len=} < {train_args.max_seq_len=}"
        )

    if os.path.exists(train_args.model_path):
        if not os.path.isdir(train_args.model_path):
            raise FileNotFoundError(
                "Model path does not appear to be a directory. Please make sure that you're passing a Hugging Face Transformers compatible directory checkpoint."
            )
    elif not len(train_args.model_path.split("/")) == 2:
        raise FileNotFoundError(
            f"Provided path does not exist locally and is not an HF format name. Please make sure that you've passed a valid model path and that it has appropriate permissions, or a Huggingface model name (org/repo): {train_args.model_path}"
        )

    if train_args.use_dolomite:
        warnings.warn(
            "use_dolomite is ignored; Dolomite support was removed.", DeprecationWarning
        )

    if train_args.is_padding_free:
        warnings.warn(
            "is_padding_free is being deprecated due to adoption of the default padding-free support in Hugging Face Transformers. As such, this flag is non-functional in 0.6.0 and beyond.",
            DeprecationWarning,
        )

    if (
        train_args.accelerate_full_state_at_epoch
        and train_args.lora
        and train_args.lora.rank > 0
    ):
        raise ValueError(
            "`accelerate_full_state_at_epoch` is not currently supported when training LoRA models."
        )

    if (
        train_args.lora
        and train_args.lora.rank > 0
        and train_args.lora.quantize_data_type != QuantizeDataType.NONE
        and train_args.distributed_backend == DistributedBackend.FSDP.value
    ):
        raise ValueError(
            "Quantization is not supported when training LoRA models with FSDP. For quantized LoRA training, please switch to DeepSpeed."
        )

    if Model.check_flash_attn_enabled(train_args.disable_flash_attn):
        # verify that the flash_attn package is actually installed
        try:
            # pylint: disable=unused-import
            # Third Party
            import flash_attn
        except ImportError as exc:
            raise ImportError(
                "Flash attention is enabled but flash_attn is not installed. You can resolve this in the following ways:\n"
                "1. Ensure the CUDA/ROCM version of the training library is installed via: `pip install instructlab-training[cuda]` or `pip install instructlab-training[rocm]`\n"
                "2. Install flash_attn manually via: `pip install flash-attn --no-build-isolation`\n"
                "3. Disable flash attention by setting `disable_flash_attn=True` in your training arguments\n"
            ) from exc

    # liger checks
    if train_args.lora and train_args.lora.rank > 0 and train_args.use_liger:
        raise ValueError(
            "Using LoRA and Liger kernels is not supported. Please use either LoRA or Liger kernels, but not both."
        )

    if train_args.use_liger:
        try:
            # Third Party
            # pylint: disable-next=W0611
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
        except ImportError as e:
            raise ValueError(
                "Liger kernels are not installed. Please install Liger kernels using the following command: pip install liger-kernel"
            ) from e


def retrieve_chat_template(chat_tmpl_path):
    try:
        spec = importlib.util.spec_from_file_location("spcl_chat_tmpl", chat_tmpl_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["spcl_chat_tmpl"] = module
        spec.loader.exec_module(module)
        SPECIAL_TOKENS = module.SPECIAL_TOKENS
        CHAT_TEMPLATE = module.CHAT_TEMPLATE
    except Exception:
        sys.exit(f"Invalid chat template path: {chat_tmpl_path}")
    return CHAT_TEMPLATE, SPECIAL_TOKENS


def add_noisy_embeddings(model, noise_alpha=None):
    if not noise_alpha:
        return model

    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(torch.numel(x))
                mag_norm = noise_alpha / torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(
                    -mag_norm, mag_norm
                )
            else:
                return orig_embed(x)

        return new_func

    model_class_name = model.__class__.__name__
    match model_class_name:
        case "GPTMegatronForCausalLM":
            orig_forward = model.get_input_embeddings().forward
            model.get_input_embeddings().forward = noised_embed(
                orig_forward, noise_alpha
            )
        case "MistralForCausalLM" | "LlamaForCausalLM":
            orig_forward = model.base_model.embed_tokens.forward
            model.base_model.embed_tokens.forward = noised_embed(
                orig_forward, noise_alpha
            )
        case _:
            raise ValueError(f"Unsupported model class: {model_class_name}")
    return model


class StreamablePopen(subprocess.Popen):
    """
    Provides a way of reading stdout and stderr line by line.
    """

    def __init__(self, output_file, *args, **kwargs):
        # remove the stderr and stdout from kwargs
        kwargs.pop("stderr", None)
        kwargs.pop("stdout", None)
        self.output_file = output_file

        super().__init__(
            *args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
        )

    def listen(self):
        with open(self.output_file, "wb") as full_log_file:
            while True:
                byte = self.stdout.read(1)
                if byte:
                    if buffer := getattr(sys.stdout, "buffer", None):
                        buffer.write(byte)
                    else:
                        sys.stdout.write(byte.decode("utf-8", "ignore"))
                    sys.stdout.flush()
                    full_log_file.write(byte)
                else:
                    break


def make_collate_fn(pad_token_id, flash_enabled=True, max_batch_len=60000):
    if flash_enabled:

        def pad_collate_fn(batch):
            input_ids = []
            labels = []
            position_ids = []
            total_len = 0
            num_loss_counted_tokens = 0

            for num_samples, item in enumerate(batch):
                item_len = len(item["input_ids"])
                if total_len + item_len > max_batch_len:
                    break

                input_ids.extend(item["input_ids"].tolist())
                labels.extend(item["labels"].tolist())
                position_ids.extend(range(item_len))

                total_len += item_len
                num_loss_counted_tokens += (item["labels"] != -100).sum().item()

            return {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
                "position_ids": torch.tensor([position_ids], dtype=torch.long),
                "num_loss_counted_tokens": num_loss_counted_tokens,
                "total_length": total_len,
                "num_samples": num_samples + 1,  # pylint: disable=W0631
            }

    else:

        def pad_collate_fn(batch):
            lens = np.array([len(item["input_ids"]) for item in batch])
            max_len = max(lens)

            input_ids = torch.stack(
                [
                    F.pad(
                        item["input_ids"],
                        (max_len - len(item["input_ids"]), 0),
                        mode="constant",
                        value=pad_token_id,
                    )
                    for item in batch
                ]
            )
            labels = torch.stack(
                [
                    F.pad(
                        item["labels"],
                        (max_len - len(item["labels"]), 0),
                        mode="constant",
                        value=-100,
                    )
                    for item in batch
                ]
            )
            num_loss_counted_tokens = (labels != -100).sum()

            attention_mask = torch.stack(
                [
                    F.pad(
                        item["attention_mask"],
                        (max_len - len(item["attention_mask"]), 0),
                        mode="constant",
                        value=0,
                    )
                    for item in batch
                ]
            )

            return {
                "input_ids": input_ids,
                "labels": labels,
                "total_length": max_len * len(batch),
                "num_loss_counted_tokens": num_loss_counted_tokens,
                "attention_mask": attention_mask,
                "num_samples": len(batch),
            }

    return pad_collate_fn


def convert_loss_to_reduce_sum(model):
    """
    this is necessary because multipack changes the samples per gpu, which biases the gradients to be larger for batches with less samples but longer lengths.
    """

    def reduce_sum_forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **_deprecated_arguments,
    ):
        output = model.__original_forward__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return_dict = isinstance(output, dict)
        logits = output.logits if return_dict else output[0]
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        output.loss = loss
        return output

    model.__original_forward__ = model.forward
    model.forward = reduce_sum_forward
    return model


# taken from https://github.com/foundation-model-stack/fms-acceleration/blob/main/plugins/accelerated-peft/src/fms_acceleration_peft/autogptq_utils.py
def patch_target_module(
    to_patch: str,
    replace_with: Any,
):
    to_patch = to_patch.split(".")
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    setattr(source, obj_name_to_patch, replace_with)


def wraps(module: nn.Module, wrapped_classes: Tuple[Any]) -> bool:
    """Checks if a module or its children are an instance of one of the provided classes.

    Args:
        module (nn.Module): A PyTorch module.
        wrapped_classes(Tuple): A tuple of potential classes the module could be.

    Returns:
        bool: True if the module or any of its children are instances of one of `wrapped_classes`, False otherwise.
    """
    if isinstance(module, wrapped_classes):
        return True

    for m in module.children():
        if wraps(m, wrapped_classes):
            return True

    return False


def save_fsdp_lora_model(
    args: Namespace,
    model: FSDP,
    tokenizer: PreTrainedTokenizer,
    accelerator: Accelerator,
    output_dir: Path,
):
    """Given a LoRA model wrapped by FSDP and Accelerate, save a full copy of the original
    model with the trained LoRA adapters merged into the copy.

    This function creates a full copy of the model being trained and stores it in CPU memory.
    If encountering OOM errors on CPU, this is likely a culprit.

    Args:
        args (Namespace): Args received by the ArgumentParser.
        model (FSDP): FSDP model as prepared by `accelerate.Accelerator`
        accelerator (Accelerator): The given accelerator object.
    """

    if accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError(
            "`save_fsdp_lora_model` was called when FSDP was not being used."
        )
    if not wraps(model, FSDP):
        raise RuntimeError(
            "`save_fsdp_lora_model` was called but provided model is not an FSDP model."
        )
    if not wraps(model, LoraModel):
        raise RuntimeError(
            "`save_fsdp_lora_model` was called but provided model is not a LoRA model."
        )

    # okay now that validation is out of the way, we are free to implement saving
    lora_conf: LoraConfig = args.lora_config
    sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_config):
        state = model.state_dict()

    # When training a LoRA with FSDP and Accelerate, you cannot directly merge the adapters into
    # the model wrapped by FSDP. To get around this limitation, we get a copy of the state dict
    # create an identical model on CPU, load the state dict into the CPU model, merge the adapters
    # and save the model to disk.
    if accelerator.is_main_process:
        # remove device_map from args list so we can load the model on CPU
        old_device_map = args.base_model_args.pop("device_map", None)
        model_copy = AutoModelForCausalLM.from_pretrained(
            **args.base_model_args, device_map="cpu"
        )
        model_copy = LoraModel(model_copy, lora_conf, "default")
        model_copy.load_state_dict(state)
        model_copy.merge_and_unload(progressbar=True)
        model_copy.save_pretrained(output_dir, safe_serialization=True)
        model.config.to_json_file(f"{output_dir}/config.json")
        tokenizer.save_pretrained(output_dir)
        del model_copy
        if old_device_map:
            # return the previous device_map so it can be used later on if needed
            args.base_model_args["device_map"] = old_device_map

    dist.barrier()


def save_fsdp_gpt_oss_model(
    model: FSDP,
    tokenizer: PreTrainedTokenizer,
    accelerator: Accelerator,
    output_dir: Path,
):
    """Save GPT-OSS model with parameter conversion, following FSDP LoRA pattern."""
    # Local

    if accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError(
            "`save_fsdp_gpt_oss_model` was called when FSDP was not being used."
        )
    if not wraps(model, FSDP):
        raise RuntimeError(
            "`save_fsdp_gpt_oss_model` was called but provided model is not an FSDP model."
        )

    logger.info("Converting GPT-OSS parameters to quantized format for compatibility")

    # Extract state dict with FSDP configuration (same as LoRA)
    sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_config):
        state = model.state_dict()

    # Convert parameters on main process only (same pattern as LoRA)
    if accelerator.is_main_process:
        clean_state = OrderedDict()
        expert_params_to_process = []

        for name, param in state.items():
            if (
                "experts." in name
                and ("down_proj" in name or "gate_up_proj" in name)
                and not name.endswith("_bias")
            ):
                expert_params_to_process.append((name, param))
            else:
                clean_state[name] = deepcopy(param).cpu()

        # Process expert parameters one by one on GPU to avoid OOM

        for clean_name, param in expert_params_to_process:
            # Create mini state dict with just this parameter on GPU
            mini_state = {
                clean_name: param.cuda() if param.device.type == "cpu" else param
            }

            # Convert this parameter
            mini_converted = convert_dequantized_to_quantized_format_correct(mini_state)

            # Move all results back to CPU and add to final state
            for conv_name, conv_param in mini_converted.items():
                tensor_cpu = (
                    conv_param.cpu() if conv_param.device.type != "cpu" else conv_param
                )
                clean_state[conv_name] = deepcopy(tensor_cpu)

            # Clean up GPU memory
            del mini_state, mini_converted
            torch.cuda.empty_cache()

        # Save state dict using accelerator.save
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use accelerator.save directly on our state dict
        accelerator.save(
            clean_state, output_dir / "model.safetensors", safe_serialization=True
        )

        # Save config and tokenizer
        # Add quantization config before saving to avoid double-write
        add_gpt_oss_quantization_config(model.config)
        model.config.to_json_file(f"{output_dir}/config.json")

        tokenizer.save_pretrained(output_dir)

    dist.barrier()


def get_module_class_from_name(
    model: torch.nn.Module, name: str
) -> torch.nn.Module | None:
    modules_children = list(model.children())

    if model.__class__.__name__ == name:
        return model.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


def get_caller(num_frames=1):
    frame = inspect.currentframe().f_back
    for _ in range(num_frames - 1):
        frame = frame.f_back
    file_name = frame.f_code.co_filename
    line_number = frame.f_lineno
    return f"In {file_name}, line {line_number}"


def log_rank_0(msg, include_caller=False, rank=None, to_print=False):
    if rank is None:
        rank = get_rank() if is_initialized() else 0
    if rank <= 0:
        if include_caller:
            msg = f"{get_caller(num_frames=2)}: {msg}"
        if to_print:
            print(msg)
        else:
            logger.info(msg)


def _copy_no_lora_dict(state_dict):
    cleaned_state_dict = OrderedDict()
    for param_tensor in state_dict:
        if not "lora" in param_tensor:
            cleaned_state_dict[
                param_tensor.replace(".base_layer", "").replace("base_model.model.", "")
            ] = deepcopy(state_dict[param_tensor]).cpu()
    return cleaned_state_dict


def _copy_gpt_oss_converted_dict(state_dict):
    """Copy and convert GPT-OSS state dict to quantized format."""
    # Local

    # First apply standard cleaning like LoRA does
    cleaned_state_dict = OrderedDict()
    for param_tensor in state_dict:
        cleaned_state_dict[
            param_tensor.replace(".base_layer", "").replace("base_model.model.", "")
        ] = deepcopy(state_dict[param_tensor]).cpu()

    # Then apply GPT-OSS parameter name conversion
    converted_state_dict = convert_dequantized_to_quantized_format_correct(
        cleaned_state_dict
    )

    return converted_state_dict


def save_dict_accelerate(
    accelerator: Accelerator,
    state_to_save,
    save_directory,
    max_shard_size="5GB",
    safe_serialization=True,
):
    old_get_state = accelerator.get_state_dict
    accelerator.get_state_dict = _copy_no_lora_dict

    def skip_precheck_loops():
        return []

    # The save model does a loop over modules and params in order to determine how to get state dict. Since we already have the state dict directly, we want to bypass those checks.
    state_to_save.modules = skip_precheck_loops
    state_to_save.parameters = skip_precheck_loops

    accelerator.save_model(
        state_to_save,
        save_directory=save_directory,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )

    accelerator.get_state_dict = old_get_state


def save_dict_accelerate_gpt_oss(
    accelerator: Accelerator,
    state_to_save,
    save_directory,
    max_shard_size="5GB",
    safe_serialization=True,
):
    """Save state dict with GPT-OSS parameter conversion (same pattern as LoRA)."""
    old_get_state = accelerator.get_state_dict
    accelerator.get_state_dict = _copy_gpt_oss_converted_dict

    def skip_precheck_loops():
        return []

    # The save model does a loop over modules and params in order to determine how to get state dict. Since we already have the state dict directly, we want to bypass those checks.
    state_to_save.modules = skip_precheck_loops
    state_to_save.parameters = skip_precheck_loops

    accelerator.save_model(
        state_to_save,
        save_directory=save_directory,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )

    accelerator.get_state_dict = old_get_state


def save_hf_format_accelerate(
    args,
    model,
    tokenizer,
    accelerator: Accelerator,
    samples_seen,
    is_lora=False,
):
    # Build the subdirectory name
    subdir = (
        "last_epoch" if args.keep_last_checkpoint_only else f"samples_{samples_seen}"
    )

    log_rank_0(
        f"\033[93mSaving model in huggingface format at: {subdir}\033[0m",
        to_print=True,
    )
    start = time.time()

    # Build the final output directory path
    final_output_dir = Path(args.output_dir) / "hf_format" / subdir

    output_dir = final_output_dir

    CONFIG_NAME = "config.json"
    output_config_file = output_dir / CONFIG_NAME

    # XXX(osilkin): LoRA + FSDP requires a different saving path than the others
    #               so we set this variable and use it to avoid those paths further down.
    is_fsdp_lora = is_lora and accelerator.distributed_type == DistributedType.FSDP
    if is_fsdp_lora:
        save_fsdp_lora_model(
            args=args,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            output_dir=output_dir,
        )

    get_state_dict_unpatched = accelerator.get_state_dict

    def _get_state_dict_patched(model, unwrap=False):
        return get_state_dict_unpatched(model, unwrap=unwrap)

    accelerator.get_state_dict = _get_state_dict_patched

    if not is_fsdp_lora and accelerator.is_main_process:
        if is_lora:
            model.module.merge_adapter()
            model_state = model.module.state_dict()

        output_dir.mkdir(parents=True, exist_ok=True)
        if not model.module.config.architectures:
            arch_added = False
            if args.model_type == "llama":
                model.module.config.architectures = ["LlamaForCausalLM"]
                arch_added = True
            elif args.model_type == "granite":
                model.module.config.architectures = ["GraniteForCausalLM"]
                arch_added = True
            if arch_added:
                warnings.warn(
                    f"Adding architectures to ckpt: {model.module.config.architectures}",
                )

        # For GPT-OSS models, ensure the config has proper quantization settings
        # before writing to file (avoids double-write)
        if is_gpt_oss(model.module.config):
            add_gpt_oss_quantization_config(model.module.config)

        model.module.config.to_json_file(output_config_file)

        tokenizer.save_pretrained(output_dir)

        if is_lora:
            save_dict_accelerate(
                accelerator,
                model_state,
                save_directory=output_dir,
                max_shard_size="5GB",
                safe_serialization=True,
            )
            model.module.unmerge_adapter()

    if not is_lora:
        # Check if this is a GPT-OSS model that needs format conversion

        if is_gpt_oss(model.module.config):
            # For GPT-OSS models, check if we need FSDP handling like LoRA does
            is_fsdp_gpt_oss = accelerator.distributed_type == DistributedType.FSDP

            if is_fsdp_gpt_oss:
                # Use FSDP GPT-OSS saving (same pattern as LoRA FSDP)
                log_rank_0(
                    "Converting GPT-OSS parameters to quantized format for compatibility (FSDP)"
                )
                save_fsdp_gpt_oss_model(
                    model=model,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    output_dir=output_dir,
                )
            elif accelerator.is_main_process:
                # Non-FSDP path
                log_rank_0(
                    "Converting GPT-OSS parameters to quantized format for compatibility"
                )
                model_state = model.module.state_dict()

                save_dict_accelerate_gpt_oss(
                    accelerator,
                    model_state,
                    save_directory=output_dir,
                    max_shard_size="5GB",
                    safe_serialization=True,
                )
        elif not is_gpt_oss(model.module.config):
            # Standard model saving
            accelerator.save_model(
                model,
                save_directory=output_dir,
                max_shard_size="5GB",
                safe_serialization=True,
            )

    log_rank_0(f"\033[93mModel saved in {final_output_dir}\033[0m", to_print=True)
    log_rank_0(f"saving took {time.time() - start} seconds")
    dist.barrier()

    accelerator.get_state_dict = get_state_dict_unpatched


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# TODO: move this to also live in the `Model` object
def save_checkpoint(
    args,
    accelerator: Accelerator,
    model,
    tokenizer,
    samples_seen,
    is_lora: bool,
    epoch: int = None,
    hf_format: bool = True,
    full_state: bool = False,
) -> None:
    if hf_format:
        save_hf_format_accelerate(
            args=args,
            model=model,
            accelerator=accelerator,
            tokenizer=tokenizer,
            samples_seen=samples_seen,
            is_lora=is_lora,
        )

    if full_state:
        save_full_state(
            args=args,
            accelerator=accelerator,
            is_lora=is_lora,
            epoch=epoch,
            samples_seen=samples_seen,
        )


def save_full_state(args, accelerator, is_lora: bool, epoch: int, samples_seen: int):
    """
    Saves model, optimizer, and lr_scheduler state.
    TODO: save model config - decided not to do this.
    TODO: save tokenizer - decided not to do this.
    TODO: handle LoRA
    TODO: handle granite
    """
    if is_lora:
        raise NotImplementedError("Can't save full state for LoRA at the moment.")

    # if args.is_granite:
    #     raise NotImplementedError("Can't save full state for Granite models yet.")

    output_dir = Path(args.output_dir) / "full_state" / f"epoch_{epoch}"
    log_rank_0(f"\033[93mSaving full model state in {output_dir}\033[0m", to_print=True)

    # patch FSDP state dict method so it works correctly.
    def _get_state_dict_patched(model, unwrap=False):
        return get_state_dict_unpatched(model, unwrap=unwrap)

    if args.distributed_training_framework == "fsdp":
        get_state_dict_unpatched = accelerator.get_state_dict
        accelerator.get_state_dict = _get_state_dict_patched

    accelerator.save_state(
        output_dir=output_dir,
        # max_shard_size="5GB",
        # safe_serialization=True,
    )

    # save metadata file for current training status
    if accelerator.is_main_process:
        # TODO: should we set the global_step here rather than calculating global_step
        #   based on samples_seen?
        metadata = {"current_epoch": epoch, "samples_seen": samples_seen}
        torch.save(metadata, output_dir / "training_metadata.json")
        log_rank_0(f"\033[93mSaving training state: {metadata}\033[0m", to_print=True)

    log_rank_0(f"\033[93mModel state saved in: {output_dir}\033[0m", to_print=True)

    # cleanup
    if args.distributed_training_framework == "fsdp":
        accelerator.get_state_dict = get_state_dict_unpatched


def load_latest_full_state(args, accelerator) -> None:
    """
    Loads accelerator state from most recently saved checkpoint
    in `output_dir/full_state`.
    """
    output_dir = Path(args.output_dir) / "full_state"

    if not output_dir.is_dir():
        return

    # picks checkpoint with the largest number of samples by splitting the "samples_NNNN" string on _
    # and comparing the number at the end of the string
    checkpoint_list = sorted(
        list(output_dir.iterdir()),
        reverse=True,
        key=lambda x: int(str(x).rsplit("_", maxsplit=1)[-1]),
    )

    if len(checkpoint_list) == 0:
        log_rank_0(
            f"\033[93mNo checkpoints to load from: {output_dir}\033[0m", to_print=True
        )
        return

    latest = checkpoint_list[0]

    log_rank_0(f"\033[93mLoading state from: {latest}\033[0m", to_print=True)
    accelerator.load_state(latest)

    training_metadata = torch.load(latest / "training_metadata.json")
    log_rank_0(
        f"\033[93mTraining metadata loaded: {training_metadata}\033[0m", to_print=True
    )

    # previous epoch is basis for current epoch.
    args.__dict__["current_epoch"] = training_metadata["current_epoch"] + 1
    args.__dict__["samples_seen"] = training_metadata["samples_seen"]


def freeze_router_params(model: Model):
    """
    Freeze router parameters for MoE models before FSDP setup.

    Args:
        model: The model to check and potentially freeze parameters

    Returns:
        bool: True if this is an MoE model and parameters were frozen
    """

    # Freeze router parameters BEFORE accelerator setup
    frozen_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "router" in name:
            param.requires_grad = False
            frozen_count += 1
            logger.info(f"❄️ Frozen router parameter: {name}")

    if frozen_count > 0:
        logger.info(f"✅ Frozen {frozen_count} router parameters for an MoE model")
        return True
    else:
        return False


def test_model_inference_quick(model, tokenizer, stage_name):
    """Quick inference test to check if model outputs are coherent."""
    try:
        logger.info(f"🧪 Running quick inference test at stage: {stage_name}")

        # Simple test prompt
        test_prompt = "The quick brown fox"
        inputs = tokenizer(test_prompt, return_tensors="pt")

        # Move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate a few tokens
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and log result
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"🔤 {stage_name} OUTPUT: '{generated_text}'")

        # Check if output looks reasonable (not just repeated tokens or gibberish)
        output_tokens = generated_text.split()
        if len(set(output_tokens)) < 3:
            logger.warning(f"⚠️ {stage_name}: Output looks repetitive/corrupted!")
        else:
            logger.info(f"✅ {stage_name}: Output looks reasonable")

    except Exception as e:
        logger.error(f"❌ {stage_name} inference test failed: {e}")

        traceback.print_exc()
