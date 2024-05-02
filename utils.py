# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import List, Optional
import inspect
import logging
import random
import time

# Third Party
from rich.logging import RichHandler
from torch import distributed as dist
from torch.distributed import get_rank, is_initialized
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
import numpy as np
import torch


def convert_loss_to_reduce_sum(model, is_granite=False):
    """
    this is necessary because multipack changes the samples per gpu, which biases the gradients to be larger for batches with less samples but longer lengths.
    """
    if is_granite:

        def get_autoregressive_language_modeling_loss(
            self,
            lm_logits: torch.Tensor,
            labels: torch.Tensor,
            cu_seqlens: torch.Tensor,
        ) -> torch.Tensor:
            loss = None
            # Shift so that tokens < n predict n
            if labels is not None:
                if self._use_padding_free_transformer:
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


def save_hf_format_ds(args, model, tokenizer, samples_seen):
    model_to_save = model.module
    log_rank_0(
        f"\033[93mSaving model in huggingface format at samples_seen: {samples_seen}\033[0m",
        to_print=True,
    )
    start = time.time()
    # used to save huggingface format, so we can use it for hf.from_pretrained
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = Path(args.output_dir) / "hf_format" / f"samples_{samples_seen}"
    if torch.distributed.get_rank() == 0:
        model_state = model_to_save.state_dict()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_model_file = output_dir / WEIGHTS_NAME
        output_config_file = output_dir / CONFIG_NAME
        torch.save(model_state, str(output_model_file))
        model_to_save.config.to_json_file(str(output_config_file))
        tokenizer.save_pretrained(str(output_dir))
    dist.barrier()
    log_rank_0(f"\033[93mModel saved in {output_dir}\033[0m", to_print=True)
    log_rank_0(f"saving took {time.time() - start} seconds")


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
