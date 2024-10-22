# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional
import importlib
import inspect
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import warnings

# Third Party
# pylint: disable=no-name-in-module
from accelerate import Accelerator
from instructlab.dolomite.hf_models import (
    GPTDolomiteConfig,
    export_to_huggingface,
    import_from_huggingface,
)
from rich.logging import RichHandler
from torch import distributed as dist
from torch.distributed import get_rank, is_initialized
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers import PreTrainedModel
import numpy as np
import torch
import torch.nn.functional as F

# First Party
from instructlab.training.config import TrainingArgs


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
    else:
        raise FileNotFoundError(
            f"Provided path to model does not exist. Please make sure that you've passed a valid model and that it has appropriate permissions: {train_args.model_path}"
        )

    if train_args.use_dolomite:
        with open(Path(train_args.model_path) / "config.json") as conf_json:
            model_conf = json.load(conf_json)
        if model_conf["model_type"] == "granite":
            raise RuntimeError(
                "Converting Granite models to Dolomite format is currently unsupported."
            )
        if train_args.disable_flash_attn:
            raise RuntimeError(
                "ERROR: Trying to use dolomite padding-free transformer without flash attention is not supported"
            )

    if train_args.is_padding_free:
        print(
            "\033[33m WARNING: is_padding_free is being deprecated due to adoption of the default padding-free support in Hugging Face Transformers. As such, this flag is non-functional in 0.6.0 and beyond. If you would like to use the older Dolomite padding-free implementation, please set use_dolomite moving forward.\033[0m"
        )


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
    if model_class_name in ["GPTMegatronForCausalLM", "GPTDolomiteForCausalLM"]:
        orig_forward = model.get_input_embeddings().forward
        model.get_input_embeddings().forward = noised_embed(orig_forward, noise_alpha)
    elif model_class_name in ["MistralForCausalLM", "LlamaForCausalLM"]:
        orig_forward = model.base_model.embed_tokens.forward
        model.base_model.embed_tokens.forward = noised_embed(orig_forward, noise_alpha)
    else:
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
                    sys.stdout.buffer.write(byte)
                    sys.stdout.flush()
                    full_log_file.write(byte)
                else:
                    break


def supports_flash_attention(device_id=0):
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability(device_id)
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0
    dev_name = torch.cuda.get_device_properties(device_id).gcnArchName.split(":")[0]
    is_compat_amd = dev_name in ("gfx90a", "gfx940", "gfx941", "gfx942")
    return is_sm8x or is_sm90 or is_compat_amd


def check_flash_attn_enabled(disable_flash_attn: bool, use_dolomite: bool) -> bool:
    if not disable_flash_attn:
        if supports_flash_attention():
            flash_enabled = True
        else:
            raise RuntimeError(
                "ERROR: Trying to use Flash Attention on unsupported hardware. Please set disable_flash_attn to True."
            )
    elif use_dolomite:
        raise RuntimeError(
            "ERROR: Trying to use dolomite padding-free transformer without flash attention is not supported"
        )
    else:
        flash_enabled = False
    return flash_enabled


def make_collate_fn(
    pad_token_id, use_dolomite=False, flash_enabled=True, max_batch_len=60000
):
    rank = int(os.environ["RANK"])
    if use_dolomite:

        def pad_collate_fn(batch):
            lens = np.array([len(item["input_ids"]) for item in batch])

            cumsum_lens = np.cumsum(lens)
            valid_up_to = int((cumsum_lens < max_batch_len).sum())
            total_len = cumsum_lens[valid_up_to - 1]

            batch = batch[:valid_up_to]
            input_ids = [x["input_ids"].tolist() for x in batch]
            labels = [x["labels"].tolist() for x in batch]
            num_loss_counted_tokens = sum(
                [(x["labels"] != -100).sum().item() for x in batch]
            )

            print(
                f"\033[96m total length: {total_len} dropped: {cumsum_lens[-1] - total_len} "
                f"num samples {len(batch)} - rank: {rank} "
                f"max len: {lens.max()} min len: {lens.min()} avg len: {lens.mean()} "
                f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
            )

            return {
                "input_ids": input_ids,
                "labels": labels,
                "num_loss_counted_tokens": num_loss_counted_tokens,
                "num_samples": len(batch),
            }

    else:
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
                    position_ids.extend(range(total_len, total_len + item_len))

                    total_len += item_len
                    num_loss_counted_tokens += (item["labels"] != -100).sum().item()

                print(
                    f"\033[96m total length: {total_len} "
                    f"num samples {len(batch)} - rank: {rank} "
                    f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
                )

                return {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long),
                    "labels": torch.tensor([labels], dtype=torch.long),
                    "position_ids": torch.tensor([position_ids], dtype=torch.long),
                    "num_loss_counted_tokens": num_loss_counted_tokens,
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
                print(
                    f"\033[96m total tokens: {max_len * len(batch)} num samples: {len(batch)} num padding tokens: {max_len * len(batch) - lens.sum()} - rank: {rank} "
                    f"max len: {max_len} min len: {min(lens)} avg len: {lens.mean()} "
                    f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
                )

                return {
                    "input_ids": input_ids,
                    "labels": labels,
                    "num_loss_counted_tokens": num_loss_counted_tokens,
                    "attention_mask": attention_mask,
                    "num_samples": len(batch),
                }

    return pad_collate_fn


def convert_loss_to_reduce_sum(model, use_dolomite=False):
    """
    this is necessary because multipack changes the samples per gpu, which biases the gradients to be larger for batches with less samples but longer lengths.
    """
    if use_dolomite:

        def get_autoregressive_language_modeling_loss(
            lm_logits: torch.Tensor,
            labels: torch.Tensor,
            cu_seqlens: torch.Tensor,
        ) -> torch.Tensor:
            loss = None
            # Shift so that tokens < n predict n
            if labels is not None:
                if model._use_padding_free_transformer:
                    shift_logits = lm_logits[:-1, :]
                    shift_labels = labels[1:].to(shift_logits.device)

                    # this is needed so that the last token of current example doesn't predict first token of next example
                    drop_loss_positions = cu_seqlens[1:-1] - 1
                    shift_labels[drop_loss_positions] = -100
                else:
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

            return loss

        model.get_autoregressive_language_modeling_loss = (
            get_autoregressive_language_modeling_loss
        )
        return model
    else:

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
            **deprecated_arguments,
        ):
            output = model.__original_forward__(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
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


def prepare_peft_model(
    model,
    peft_config,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    mixed_precision="bf16",
):
    # will guard this
    # Third Party
    from peft import (
        PeftConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from trl.trainer.utils import peft_module_casting_to_bf16

    if not isinstance(peft_config, PeftConfig):
        raise ValueError(
            "If you want to use the PeftModel, you need to pass a PeftConfig object, "
            f"and you passed a {type(peft_config)}."
        )

    if not isinstance(model, PeftModel):
        if getattr(model, "is_loaded_in_8bit", False) or getattr(
            model, "is_loaded_in_4bit", False
        ):
            preprare_model_kwargs = {
                "use_gradient_checkpointing": gradient_checkpointing
            }

            # if _support_gc_kwargs:
            preprare_model_kwargs["gradient_checkpointing_kwargs"] = (
                gradient_checkpointing_kwargs
            )

            model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)

        elif gradient_checkpointing:
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        model = get_peft_model(model, peft_config)
        if mixed_precision == "bf16" and getattr(model, "is_loaded_in_4bit", False):
            peft_module_casting_to_bf16(model)

    return model


def prepare_universal_checkpoint_from_latest(output_dir):
    """Populate the universal checkpoint in output_dir/step_folder
    - 1. read output_dir/latest to get step_folder
    - 2. populate tmp dir in output_dir/step_folder/tmp
    - 3. populate zero checkpoints in output_dir/step_folder/zero
    - 4. create output_dir/latest_universal

    Items 1, 2, 3, 4 are idempotent. There is atomicity in the sense that
    only after 4 is completed, then the output_dir/latest_universal
    checkpoint is created in which then the universal checkpoint
    can be loaded.

    Be aware that this creates an extra dir `zero/` in the checkpoint dir,
    which doubles the DS checkpoint storage requirement.
    - DS checkpoints store 3X model parameters in 32bit.
    - e.g., will be 6X a model parameter-only checkpoint in 16bit.

    Note that this requires a latest version of deepspeed. It kind of works if
    the model is not saving universal checkpoint info, but only in the
    the case where advanced features like tensor parallel (TP) and
    pipeline parallel (PP) are turned off.
    """

    log_rank_0(
        f"\033[93mPreparing universal checkpoint in {output_dir}\033[0m", to_print=True
    )
    # Third Party
    from transformers.utils.import_utils import _is_package_available

    _, ds_version = _is_package_available("deepspeed", return_version=True)
    if ds_version < "0.14.3":
        raise ValueError("universal checkpoint only supported on deepspeed >= 0.14.3")

    start = time.time()
    if torch.distributed.get_rank() == 0:
        # Third Party
        from deepspeed.checkpoint import DeepSpeedCheckpoint
        from deepspeed.checkpoint.ds_to_universal import (
            PARAM_SHAPES,
            UNIVERSAL_CHECKPOINT_INFO,
            _check_for_required_state,
            _extract_zero_shard_files,
            _merge_tp_slice_files,
            _save_optimizer_state,
        )

        # read the latest file to get the step folder
        latest_file = output_dir / "latest"
        with open(latest_file) as f:
            step_folder = f.read()

        # will process the checkpoint in the latest step folder
        input_folder = os.path.join(output_dir, step_folder)

        # create args for the scripts below
        class UniversalCheckpointArgs:
            num_extract_workers: int = 1
            num_merge_workers: int = 1
            output_folder: str = input_folder  # just put in same place
            strict: bool = True  # strict checkpoint

        args = UniversalCheckpointArgs()

        # get the checkpoint
        ds_checkpoint = DeepSpeedCheckpoint(input_folder)

        # hack, force this to null if we did not properly save
        # any universal checkpoint information
        # - this will not support any pipeline replication and other
        #   replication such as TP, row parallelism, vocab, sub_params
        if UNIVERSAL_CHECKPOINT_INFO not in ds_checkpoint.global_state:
            warnings.warn(
                "Universal checkpoint information not found, setting it to "
                "an empty dictionary."
            )
            ds_checkpoint.global_state[UNIVERSAL_CHECKPOINT_INFO] = {}
            assert (
                ds_checkpoint.tp_degree == 1
            ), "if universal checkpointing info is missing, TP must be absent"
            assert (
                ds_checkpoint.pp_degree == 1
            ), "if universal checkpointing info is missing, PP must be absent"
        _check_for_required_state(ds_checkpoint)

        slice_shapes = []
        for mp_rank_file in ds_checkpoint.mp_rank_files:
            mp_sd = torch.load(mp_rank_file, map_location=torch.device("cpu"))
            slice_shapes += mp_sd[PARAM_SHAPES]

        # fix back to normal flat dict, merge duplicates for tp>1
        slice_shapes = dict((k, v) for d in slice_shapes for k, v in d.items())
        temp_dir = os.path.join(args.output_folder, "tmp")

        log_rank_0(
            f"\033[93m1. Extracting ZeRO fragments into {temp_dir}\033[0m",
            to_print=True,
        )
        _extract_zero_shard_files(args, ds_checkpoint, temp_dir)

        zero_output_folder = os.path.join(args.output_folder, "zero")

        log_rank_0(
            f"\033[93m2. Merging slices into {zero_output_folder}\033[0m", to_print=True
        )
        _merge_tp_slice_files(args, ds_checkpoint, slice_shapes, temp_dir)

        log_rank_0(
            f"\033[93m3. Saving common optimizer states into {zero_output_folder}\033[0m",
            to_print=True,
        )
        _save_optimizer_state(args, ds_checkpoint)

        log_rank_0(
            f"\033[93m4. Removing temp directory {temp_dir}\033[0m", to_print=True
        )
        shutil.rmtree(temp_dir, ignore_errors=True)

        latest_file = os.path.join(output_dir, "latest_universal")
        log_rank_0(f"\033[93m5. Creating {latest_file}\033[0m", to_print=True)
        with open(latest_file, "w") as f:
            f.write(step_folder)

    dist.barrier()
    log_rank_0(f"Preparing universal checkpoint took {time.time() - start} seconds")


@contextmanager
def ensure_loadable_dolomite_checkpoint(
    model_name_or_path: str,
    tmpdir: str,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    group_rank = int(os.environ["GROUP_RANK"])

    try:
        GPTDolomiteConfig.from_pretrained(model_name_or_path)
        yield model_name_or_path
    except:  # pylint: disable=bare-except
        log_rank_0(
            f"\033[93mModel saved in {model_name_or_path} requires conversion \033[0m",
            to_print=True,
        )
        # if the load failed then it must not be a granite
        # for now just assume its a llama
        # make a temp directory name, but do not create it
        # previously we used mktemp, but it caused problems in multi node settings
        # so now we use a provided tmpdir
        # Assumption: tmpdir should be accessible by all ranks, even those
        # in different nodes
        tmpdir = Path(tmpdir) / f"tmp.{group_rank}"
        if os.path.exists(tmpdir) and (not dist.is_initialized() or local_rank == 0):
            # need to delete if it exists because import doesn't like it to
            shutil.rmtree(tmpdir, ignore_errors=True)

        if not dist.is_initialized() or local_rank == 0:
            import_from_huggingface(model_name_or_path, tmpdir)

        if dist.is_initialized():
            # the first barrier is to wait for local rank 0 to finish converting the model
            # and place into tmpdir
            dist.barrier()

        # return tmpdir out for loading
        yield tmpdir

        if dist.is_initialized():
            # the second barrier is to wait for all the models to finish loading
            dist.barrier()

        if not dist.is_initialized() or local_rank == 0:
            # at this point, we can be confident that the tmpdir is no longer needed
            shutil.rmtree(tmpdir, ignore_errors=True)


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


# this function is for supporting gradient checkpointing for padding free
# dolomite
def apply_gradient_checkpointing(
    model: torch.nn.Module,
    **kwargs,
) -> None:
    def block_checkpointing(
        model: torch.nn.Module,
        block_name: str,
        checkpoint_every: int = 1,
        use_reentrant: bool = False,
    ) -> None:
        block_class = get_module_class_from_name(model, block_name)
        block_idx = 0

        def _whether_to_checkpoint(submodule: torch.nn.Module) -> bool:
            nonlocal block_idx

            if isinstance(submodule, block_class):
                block_idx += 1
                if (block_idx - 1) % checkpoint_every == 0:
                    return True
            return False

        checkpoint_wrapper_function = checkpoint_wrapper
        if use_reentrant:
            checkpoint_wrapper_function = partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT
            )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper_function,
            check_fn=_whether_to_checkpoint,
        )

    block_checkpointing(model, **kwargs)


def setup_logger(level="DEBUG"):
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )


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
            logging.info(msg)
        # print(msg)


def _copy_no_lora_dict(state_dict):
    cleaned_state_dict = OrderedDict()
    for param_tensor in state_dict:
        if not "lora" in param_tensor:
            cleaned_state_dict[
                param_tensor.replace(".base_layer", "").replace("base_model.model.", "")
            ] = deepcopy(state_dict[param_tensor]).cpu()
    return cleaned_state_dict


def save_dict_accelerate(
    accelerator,
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


def save_hf_format_accelerate(
    args,
    model,
    tokenizer,
    accelerator: Accelerator,
    samples_seen,
    convert_dolomite=True,
    is_lora=False,
):
    log_rank_0(
        f"\033[93mSaving model in huggingface format at samples_seen: {samples_seen}\033[0m",
        to_print=True,
    )
    start = time.time()

    final_output_dir = Path(args.output_dir) / "hf_format" / f"samples_{samples_seen}"
    if args.use_dolomite and convert_dolomite:
        tmpdir = TemporaryDirectory("w")  # pylint: disable=consider-using-with
        output_dir = Path(tmpdir.name)
    else:
        output_dir = final_output_dir

    CONFIG_NAME = "config.json"
    output_config_file = output_dir / CONFIG_NAME

    get_state_dict_unpatched = accelerator.get_state_dict

    def _get_state_dict_patched(model, unwrap=False):
        return get_state_dict_unpatched(model, unwrap=unwrap)

    accelerator.get_state_dict = _get_state_dict_patched

    if accelerator.is_main_process:
        if is_lora:
            model.module.merge_adapter()
            model_state = model.module.state_dict()

        output_dir.mkdir(parents=True, exist_ok=True)
        if not model.module.config.architectures and convert_dolomite:
            model.module.config.architectures = ["LlamaForCausalLM"]
            warnings.warn(
                f"Adding architectures to ckpt: {model.module.config.architectures}",
            )
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
        accelerator.save_model(
            model,
            save_directory=output_dir,
            max_shard_size="5GB",
            safe_serialization=True,
        )

    if args.use_dolomite and convert_dolomite and accelerator.is_main_process:
        # export doesnt like the directory to exist
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir)
        export_to_huggingface(
            pretrained_model_name_or_path=tmpdir.name,
            save_path=final_output_dir,
            model_type="llama",
        )
        tmpdir.cleanup()

    log_rank_0(f"\033[93mModel saved in {final_output_dir}\033[0m", to_print=True)
    log_rank_0(f"saving took {time.time() - start} seconds")
    dist.barrier()

    accelerator.get_state_dict = get_state_dict_unpatched


# this is native deepspeed saving with optimizer, scheduler
def save_model_ds_native(
    args,
    model,
    tokenizer,
    samples_seen,
):
    # to get a statedict from a zero checkpoint, all you need to do is
    # - from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    # - sd = get_fp32_state_dict_from_zero_checkpoint('ckpt')
    # - sum([math.prod(x.shape) for x in sd.values()]) # check the size (should be correct)

    log_rank_0(
        f"\033[93mSaving model+optimizer+scheduler in format at samples_seen: {samples_seen}\033[0m",
        to_print=True,
    )
    start = time.time()
    # used to save huggingface format, so we can use it for hf.from_pretrained
    output_dir = Path(args.output_dir) / "ds_native"
    tag = f"samples_{samples_seen}"
    use_lora = args.lora_r > 0

    # NOTE: this is a distributed save
    # if its lora, we only save the adapters
    # - so we exclude frozen if use_lora==True
    model.save_checkpoint(
        output_dir,
        exclude_frozen_parameters=use_lora,
        tag=tag,  # this will create the subdirectory with the correct name
    )

    # for now we are not saving tokenizer, config, eg..
    # so it is not totally "HF compatible"

    log_rank_0(f"\033[93mModel saved in {output_dir}\033[0m", to_print=True)
    log_rank_0(f"saving took {time.time() - start} seconds")


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def get_projection_layer_names(model: PreTrainedModel) -> List[str]:
    """
    Given a pretrained model, returns all of the projection layers (matching '_proj')
    """
    proj_layers = set(
        name.split(".")[-1]
        for name, _ in model.named_modules()
        if name.endswith("_proj")
    )
    return list(proj_layers)
