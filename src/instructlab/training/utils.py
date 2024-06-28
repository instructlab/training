# Standard
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional
import importlib
import inspect
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
from instructlab.dolomite.hf_models import (
    GPTDolomiteConfig,
    export_to_huggingface,
    import_from_huggingface,
)
from rich.logging import RichHandler
from safetensors.torch import save_file
from torch import distributed as dist
from torch.distributed import get_rank, is_initialized
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
import numpy as np
import torch
import torch.nn.functional as F


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

    def __init__(self, *args, **kwargs):
        # remove the stderr and stdout from kwargs
        kwargs.pop("stderr", None)
        kwargs.pop("stdout", None)

        super().__init__(*args, **kwargs)
        while True:
            if self.stdout:
                output = self.stdout.readline().strip()
                print(output)
            if self.stderr:
                error = self.stderr.readline().strip()
                print(error, file=sys.stderr)
            if self.poll() is not None:
                break


def make_collate_fn(pad_token_id, is_granite=False, max_batch_len=60000):
    rank = int(os.environ["RANK"])
    if is_granite:

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
            }

    return pad_collate_fn


def convert_loss_to_reduce_sum(model, is_granite=False):
    """
    this is necessary because multipack changes the samples per gpu, which biases the gradients to be larger for batches with less samples but longer lengths.
    """
    if is_granite:

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
    - e.g., will be 6X a model paramter-only checkpoint in 16bit.

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
def ensure_loadable_granite_checkpoint(
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
            # need to delete if it exists because import doesnt like it to
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


# this function is for supporting gradient checkpointing for padding free
# dolomite
def apply_gradient_checkpointing(
    model: torch.nn.Module,
    **kwargs,
) -> None:
    def get_module_class_from_name(
        model: torch.nn.Module, name: str
    ) -> List[torch.nn.Module]:
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


def save_hf_format(args, model, tokenizer, samples_seen):
    torch.cuda.empty_cache()
    log_rank_0(
        f"\033[93mSaving model in huggingface format at samples_seen: {samples_seen}\033[0m",
        to_print=True,
    )
    start = time.time()
    # used to save huggingface format, so we can use it for hf.from_pretrained
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state = model.state_dict()
    output_dir = Path(args.output_dir) / "hf_format" / f"samples_{samples_seen}"
    if torch.distributed.get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_model_file = output_dir / WEIGHTS_NAME
        output_config_file = output_dir / CONFIG_NAME
        torch.save(model_state, str(output_model_file))
        model.module.config.to_json_file(str(output_config_file))
        tokenizer.save_pretrained(str(output_dir))
    dist.barrier()
    log_rank_0(f"\033[93mModel saved in {output_dir}\033[0m", to_print=True)
    log_rank_0(f"saving took {time.time() - start} seconds")


def save_hf_format_ds(args, model, tokenizer, samples_seen, convert_granite=True):
    model_to_save = model.module
    log_rank_0(
        f"\033[93mSaving model in huggingface format at samples_seen: {samples_seen}\033[0m",
        to_print=True,
    )
    start = time.time()
    # used to save huggingface format, so we can use it for hf.from_pretrained
    CONFIG_NAME = "config.json"
    if args.is_granite:
        # save if in a temp directory first then convert it
        WEIGHTS_NAME = "model.safetensors"
        MODEL_TYPE = "llama"
    else:
        WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = Path(args.output_dir) / "hf_format" / f"samples_{samples_seen}"
    if torch.distributed.get_rank() == 0:
        model_state = model_to_save.state_dict()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_model_file = output_dir / WEIGHTS_NAME
        output_config_file = output_dir / CONFIG_NAME

        if args.is_granite and convert_granite:
            with TemporaryDirectory("w") as tmpdir:
                save_file(model_state, Path(tmpdir) / WEIGHTS_NAME)
                model_to_save.config.to_json_file(Path(tmpdir) / CONFIG_NAME)
                tokenizer.save_pretrained(tmpdir)
                # export doesnt like the directory to exist
                shutil.rmtree(output_dir)

                export_to_huggingface(
                    pretrained_model_name_or_path=tmpdir,
                    save_path=output_dir,
                    model_type=MODEL_TYPE,
                )
        else:
            torch.save(model_state, str(output_model_file))
            model_to_save.config.to_json_file(str(output_config_file))
            tokenizer.save_pretrained(str(output_dir))

    dist.barrier()
    log_rank_0(f"\033[93mModel saved in {output_dir}\033[0m", to_print=True)
    log_rank_0(f"saving took {time.time() - start} seconds")


# this is native deepspeed saving with optimizer, schediuler
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
