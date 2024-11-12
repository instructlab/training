# Forcing Intel PyTorch bridge Eager mode.

# Standard
import contextlib
import functools
import math

# Standard Library
import os
import time


# Third Party
os.environ["PT_HPU_LAZY_MODE"] = "0"
import habana_frameworks.torch as htorch
import habana_frameworks.torch.distributed.hccl

from torch.distributed import ReduceOp, all_reduce
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm
import tokenizers
import torch
import transformers

# First Party
from instructlab.training import constants, utils
from instructlab.training.config import DistributedBackend
from instructlab.training.utils import add_noisy_embeddings, convert_loss_to_reduce_sum

# Constants

# Will emit a key error if these aren't available.
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE_HPU = torch.device("hpu")


def _setup_hpu_torch_distributed():
    """
    Initialized distributed process group.

    Raises: RuntimeError if initialization fails.
    """

    torch.distributed.init_process_group(
        backend="hccl", rank=LOCAL_RANK, world_size=WORLD_SIZE
    )

    if not torch.distributed.is_initialized():
        raise RuntimeError(
            f"Attempted to initialize torch distributed process group for HPU but failed."
        )


def setup_fsdp(model: torch.nn.Module, sharding_strategy: str, cpu_param_offload: bool):
    """Wraps model in FSDP class."""

    block_name = model._no_split_modules[0]
    transformer_attention_block_class: torch.nn.Module | None = (
        utils.get_module_class_from_name(model, block_name)
    )

    if transformer_attention_block_class is None:
        raise RuntimeError(
            f"Transformer block class cannot be derived from transformer module. Cannot correctly wrap block: ({transformer_attention_block_class})"
        )

    model = FSDP(
        module=model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={type(transformer_attention_block_class)},
        ),
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy[sharding_strategy],
        device_id=torch.device("hpu", torch.hpu.current_device()),
        cpu_offload=CPUOffload(offload_params=cpu_param_offload),
    )

    return model


def setup_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    return optimizer


def try_load_checkpoint(*args, **kwargs):
    raise NotImplementedError()


def save_checkpoint(*args, **kwargs):
    # save_checkpoint(model, optimizer, lr_scheduler, other_state: dict)

    raise NotImplementedError()


def _set_sampler_epoch(sampler_type: str, data_loader, epoch: int):
    if sampler_type == "multipack":
        data_loader.batch_sampler.set_epoch(epoch)
    elif sampler_type == "distributed":
        data_loader.sampler.set_epoch(epoch)
    else:
        raise RuntimeError(f"Sampler type ({sampler_type}) is not supported.")


def print_status(loss, num_loss_counted_tokens, global_step, epoch):
    print(
        f"\033[93mPer-token loss scaled by world size: {(loss/num_loss_counted_tokens) * WORLD_SIZE}\033[0m"
    )
    print(f"Epoch: {epoch}, Step: {global_step}, Rank: {RANK}, loss = {loss}")


def batch_metric_log(
    args,
    metric_logger,
    epoch,
    global_step,
    loss,
    reduced_loss,
    num_loss_counted_tokens,
    current_lr,
    grad_norm,
    samples_seen,
    start_time,
    last_batch_size,
):
    if LOCAL_RANK != 0:
        return

    elapsed_time = time.time() - start_time
    overall_throughput = args.samples_per_gpu * WORLD_SIZE / elapsed_time
    # vmem_allocated = htorch.memory_allocated() / (1024**3)
    # vmalloc_retries = htorch.memory_stats()["num_alloc_retries"]
    # global_grad_norm = model.get_global_grad_norm()
    metric_logger.log_sync(
        {
            "epoch": epoch,
            "step": global_step,
            "rank": LOCAL_RANK,
            "loss": loss.item(),
            "overall_throughput": overall_throughput,
            "lr": current_lr,
            # "vmem_allocated": vmem_allocated,
            # "vmalloc_retries": vmalloc_retries,
            # "num_loss_counted_tokens": int(num_loss_counted_tokens),
            "batch_size": last_batch_size,
            "total_loss": float(reduced_loss / num_loss_counted_tokens),
            "gradnorm": grad_norm,
            "weight_norm": 0.0,
        }
    )


def train(
    args,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    data_loader: torch.utils.data.DataLoader,
    lr_scheduler,
    grad_accum_steps: int,
    num_epochs: int,
    metric_logger,
):
    model.train()
    optimizer.zero_grad()
    global_step = 1
    global_grad_norm = None
    samples_seen = 0
    batch_size = args.effective_batch_size // grad_accum_steps
    args.save_samples = (args.save_samples // batch_size) * batch_size

    if LOCAL_RANK == 0:
        print(f"\033[93mNumber of samples per save: {args.save_samples}\033[0m")

    # (jkunstle) TODO: implement current_epoch
    for epoch in range(num_epochs):
        _set_sampler_epoch(
            sampler_type=args.sampler, data_loader=data_loader, epoch=epoch
        )

        if LOCAL_RANK == 0:
            progress_bar = tqdm(total=len(data_loader), desc=f"Epoch {epoch}")
            if args.last_step:
                progress_bar.update(args.last_step)

        dist_shared_buffer = torch.zeros(3, dtype=torch.float32).to(DEVICE_HPU)

        for batch in data_loader:
            start_time = time.time()
            dist_shared_buffer[0] = batch.pop("num_loss_counted_tokens")
            dist_shared_buffer[1] = len(batch["input_ids"])

            # batch = {input_ids: ..., labels: ..., attention_mask: ...},
            # each is a torch.Tensor.
            for k in batch:
                batch[k] = batch[k].to(DEVICE_HPU)

            no_sync = contextlib.nullcontext
            if global_step % grad_accum_steps != 0:
                no_sync = model.no_sync

            with no_sync():
                output = model(**batch, use_cache=False)
                loss = output.loss

            dist_shared_buffer[2] = loss.item()

            all_reduce(tensor=dist_shared_buffer, op=ReduceOp.SUM)

            # These have been summed over all participating cards.
            num_loss_counted_tokens = dist_shared_buffer[0]
            samples_seen += int(dist_shared_buffer[1])

            # (jkunstle) TODO: make sure this is correct for FSDP, was originally for DeepSpeed
            # dividing by the total number of non-padding tokens and multiplying by the number of GPUs so when FSDP averages by world_size, it will be the correct loss.
            loss = loss / num_loss_counted_tokens * WORLD_SIZE

            print_status(
                loss=loss,
                num_loss_counted_tokens=num_loss_counted_tokens,
                global_step=global_step,
                epoch=epoch,
            )

            loss.backward()

            if global_step % grad_accum_steps == 0:
                global_grad_norm = model.clip_grad_norm_(1.0)
                # global_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_grad_norm = (
                float(global_grad_norm) if global_grad_norm is not None else None
            )
            batch_metric_log(
                args=args,
                metric_logger=metric_logger,
                epoch=epoch,
                global_step=global_step,
                loss=loss,
                reduced_loss=dist_shared_buffer[2],
                num_loss_counted_tokens=num_loss_counted_tokens,
                current_lr=lr_scheduler.get_last_lr()[0],
                grad_norm=global_grad_norm,
                samples_seen=samples_seen,
                start_time=start_time,
                last_batch_size=int(
                    dist_shared_buffer[1]
                ),  # sum(len(input_ids) for all cards)
            )

            global_step += 1
            if LOCAL_RANK == 0:
                progress_bar.update(1)

        # (jkunstle) TODO: save checkpoint for save_samples, epochs, and final.


def _match_model_and_tokenizer_special_tokens(
    model: torch.nn.Module, tokenizer: tokenizers.Tokenizer, token_list: list[str]
) -> torch.nn.Module:
    """
    Model might have different representations for special tokens, like eos_token and bos_token.
    This function matches a model's tokens to that of the tokenizer.
    """

    for tok in token_list:
        model_tok = getattr(model.config, tok, None)
        tokenizer_tok = getattr(tokenizer, tok, None)

        if (
            model_tok is not None
            and tokenizer_tok is not None
            and model_tok != tokenizer_tok
        ):
            print(
                f"WARNING: There is a mismatch between {tok} of model ({model_tok}) and tokenizer({tokenizer_tok}). Fixing model {tok} to be same as tokenizer's {tok}"
            )

            setattr(model.config, tok, tokenizer_tok)

    return model


def _match_model_and_tokenizer_vocab_lengths(
    model: torch.nn.Module, tokenizer: tokenizers.Tokenizer
) -> torch.nn.Module:
    tokenizer_len = len(tokenizer)
    if tokenizer_len > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {tokenizer_len} tokens but model has {model.config.vocab_size} vocab size. Resizing token embeddings."
        )

        model.resize_token_embeddings(
            int(8 * math.ceil(tokenizer_len / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    return model


def prepare_model(
    model: torch.nn.Module, tokenizer: tokenizers.Tokenizer, noise_alpha: float
) -> torch.nn.Module:
    """
    Modifies model so that it works correctly with tokenizer vocab and special tokens, multipack sampler,
    and has gradient checkpointing enabled.
    """

    model = _match_model_and_tokenizer_vocab_lengths(model=model, tokenizer=tokenizer)
    model = _match_model_and_tokenizer_special_tokens(
        model=model,
        tokenizer=tokenizer,
        token_list=["bos_token_id", "eos_token_id", "pad_tok_id"],
    )

    model = convert_loss_to_reduce_sum(model, use_dolomite=False)
    model = add_noisy_embeddings(model, noise_alpha=noise_alpha)

    model.gradient_checkpointing_enable()

    return model


def load_model(model_name_or_path: str) -> torch.nn.Module:
    """Load Transformer model and validate that it's among supported models."""

    # (jkunstle) TODO: could load model config on its own and check for the class type before
    #   downloading / loading the entire model into memory.

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, torch_dtype=torch.bfloat16
    )

    if model.__class__.__name__ not in constants.SUPPORTED_MODEL_ARCHITECTURES:
        raise RuntimeError(
            f"Model class name: {model.__class__.__name__} is not supported."
        )

    return model


def _raise_exception_for_unsupported_args(args) -> None:
    """
    Make sure that user isn't expecting training to be configured for:
    1) LoRA PEFT
    2) Quantization
    3) Distributed backend that's not FSDP
    """

    if args.lora_r > 0:
        raise RuntimeError(
            f"LoRA rank was set (lora_r={args.lora_r}) but not supported when training with (--hpu)."
        )

    if args.lora_quant_bits is not None:
        raise RuntimeError(
            f"QLoRA was set (lora_quant_bits={args.lora_quant_bits}) but not supported when training with (--hpu)."
        )

    chosen_backend = DistributedBackend(args.distributed_training_framework)
    if chosen_backend != DistributedBackend.FSDP:
        raise RuntimeError(
            f"Distributed backend was set as (distributed_training_framework={chosen_backend.value}) but only ({DistributedBackend.FSDP.value}) is suppported with (--hpu)."
        )


def main(
    args,
    model_name_or_path: str,
    tokenizer: tokenizers.Tokenizer,
    data_loader: torch.utils.data.DataLoader,
    grad_accum_steps: int,
    metric_logger,
):
    # (jkunstle) TODO: setup logger for file

    _raise_exception_for_unsupported_args(args)
    _setup_hpu_torch_distributed()

    # (jkunstle) TODO: try to load checkpoint
    model = load_model(model_name_or_path=model_name_or_path)
    model = prepare_model(
        model=model, tokenizer=tokenizer, noise_alpha=args.NEFTune_alpha
    )

    model = setup_fsdp(
        model=model,
        sharding_strategy=args.fsdp_sharding_strategy,
        cpu_param_offload=args.cpu_offload_params_fsdp,
    )

    optimizer = setup_optimizer(model=model, learning_rate=args.lr)

    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_epochs * len(data_loader) // grad_accum_steps,
    )

    train(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metric_logger=metric_logger,
        data_loader=data_loader,
        grad_accum_steps=grad_accum_steps,
        num_epochs=args.num_epochs,
    )
