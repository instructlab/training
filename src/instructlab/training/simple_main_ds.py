from datetime import timedelta
import os
from pathlib import Path
import time
import torch
import math
from copy import deepcopy
import argparse
import shutil

# Third Party
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler
import torch
import torch.distributed
import yaml
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from accelerate import Accelerator

# First Party
from instructlab.training.async_logger import AsyncStructuredLogger
from instructlab.training.multipack_sampler import find_packing_max_batch_len_and_grad_accum
from instructlab.training.setup_accelerator import setup_accelerator
from instructlab.training.token_dataset import setup_dataloader, setup_dataset
from instructlab.training.tokenizer_utils import setup_tokenizer
from instructlab.training.utils import (
    add_noisy_embeddings,
    apply_gradient_checkpointing,
    convert_loss_to_reduce_sum,
    log_rank_0,
    retrieve_chat_template,
    save_checkpoint,
    save_hf_format_accelerate,
    setup_logger,
    supports_flash_attention,
)


def setup_optimizer(args, model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    return optimizer

def align_model_and_tokenizer(model, tokenizer):
    """
    Aligns the model's vocabulary and special tokens with the tokenizer.
    """
    if len(tokenizer) > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    # Fix any discrepancy between model and tokenizer
    special_tokens = {
        'pad': ('pad_token_id', 'Fixing model pad token id'),
        'bos': ('bos_token_id', 'Fixing model bos token id'),
        'eos': ('eos_token_id', 'Fixing model eos token id')
    }

    for token_type, (token_attr, message) in special_tokens.items():
        model_token = getattr(model.config, token_attr)
        tokenizer_token = getattr(tokenizer, token_attr)
        
        if (model_token is not None and tokenizer_token is not None 
            and model_token != tokenizer_token):
            print(
                f"WARNING: There is a mismatch between {token_type} token id of "
                f"model({model_token}) and tokenizer({tokenizer_token}). "
                f"{message} to be same as tokenizer's {token_type} token id"
            )
            setattr(model.config, token_attr, tokenizer_token)

    return model

def save_full_state(args, accelerator, global_step, samples_seen):
    """Save full training state, safely replacing the previous checkpoint"""
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", "last_checkpoint")
    temp_checkpoint_dir = os.path.join(args.output_dir, "checkpoints", "temp_checkpoint")
    
    # Create temp directory for new checkpoint
    os.makedirs(temp_checkpoint_dir, exist_ok=True)

    def _get_state_dict_patched(model, unwrap=False):
        return get_state_dict_unpatched(model, unwrap=unwrap)

    get_state_dict_unpatched = accelerator.get_state_dict
    accelerator.get_state_dict = _get_state_dict_patched
    
    accelerator.save_state(temp_checkpoint_dir)
    
    if accelerator.is_local_main_process:
        training_state = {
            "global_step": global_step,
            "samples_seen": samples_seen,
        }
        torch.save(training_state, os.path.join(temp_checkpoint_dir, "training_state.pt"))
    
    accelerator.get_state_dict = get_state_dict_unpatched

    torch.distributed.barrier()
    if accelerator.is_local_main_process:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.rename(temp_checkpoint_dir, checkpoint_dir)

def load_full_state_if_exists(args, accelerator):
    """Attempt to load the last checkpoint if it exists"""
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", "last_checkpoint")
    resume_dict = {
        "global_step": 1,
        "samples_seen": 0,
    }
    
    if os.path.exists(checkpoint_dir) and not args.disable_resume:
        try:
            log_rank_0("Found existing checkpoint. Loading...", to_print=True)
            accelerator.load_state(checkpoint_dir)
            # Load training state
            training_state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"))
            resume_dict.update(training_state)
            log_rank_0(f"Resuming from global step {resume_dict['global_step']}, samples seen {resume_dict['samples_seen']}", to_print=True)
        except Exception as e:
            log_rank_0(f"Error loading checkpoint: {str(e)}. Starting from scratch.", to_print=True)
    elif not args.disable_resume:
        log_rank_0("No checkpoint found. Starting from scratch.", to_print=True)
    
    return resume_dict

def setup_model(args, tokenizer, train_loader, grad_accum, flash_enabled):

    base_model_args = {
        "pretrained_model_name_or_path": args.model_name_or_path,
        "torch_dtype": torch.bfloat16,
    }
    if flash_enabled:
        base_model_args["attn_implementation"] = "flash_attention_2"

    # this fails when the embedding layer has not been resized
    # only necessary for models that can not be fit into cpu memory by all processes at the same time
    # this is NOT the case for Llama-3.1-8B

    # from contextlib import nullcontext
    # from accelerate import init_empty_weights
    # context = nullcontext if (os.environ["RANK"] == "0") else init_empty_weights
    # with context():
    model = AutoModelForCausalLM.from_pretrained(**base_model_args)

    # store the base model args so we can recall them later if saving a LoRA model
    args.base_model_args = base_model_args

    model = align_model_and_tokenizer(model, tokenizer)

    assert model.__class__.__name__ in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM",
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "MixtralForCausalLM",
        "GraniteForCausalLM",
    ], f"Model class name: {model.__class__.__name__} is not supported."

    model = convert_loss_to_reduce_sum(model, use_dolomite=False)
    model = add_noisy_embeddings(model, noise_alpha=args.NEFTune_alpha)


    block_name = model._no_split_modules[0]
    log_rank_0(f"\033[38;5;214mGradient checkpointing will be done at the {block_name} level\033[0m", to_print=True)
    apply_gradient_checkpointing(
        model,
        block_name=block_name,
        use_reentrant=True,
    )

    torch.compile(model)
    accelerator = setup_accelerator(args, model)
    #assuming fsdp is used
    model = accelerator.prepare(model)
    optimizer = setup_optimizer(args, model)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_epochs * len(train_loader) // grad_accum,
    )
    
    accelerator.register_for_checkpointing(lr_scheduler)
    
    model, optimizer, _, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        deepcopy(train_loader),
        lr_scheduler,
    )
    # Necessary so that Accelerate does not step once per GPU
    # see https://github.com/huggingface/accelerate/blob/127818fc27ebe5cb236357fff59ff1748326d643/src/accelerate/scheduler.py#L69
    lr_scheduler.split_batches = True
    return model, lr_scheduler, optimizer, accelerator

def init_distributed_environment(args):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=180))
    args.global_rank = torch.distributed.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

def train(
    args,
    model,
    optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    tokenizer,
    train_loader: DataLoader,
    grad_accum,
    metric_logger,
    resume_dict,
):
    model.train()

    global_step = 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    batch_size = args.effective_batch_size // grad_accum
    samples_seen = 0
    global_grad_norm = None


    if args.save_samples > 0:
        args.save_samples = (args.save_samples // batch_size) * batch_size
        (
            print(f"\033[93mNumber of samples per save: {args.save_samples}\033[0m")
            if local_rank == 0
            else None
        )
    samples_seen = resume_dict["samples_seen"]

    for epoch in range(args.num_epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        
        if local_rank == 0:
            inner_pb = tqdm(range(len(train_loader)), desc=f"Epoch {epoch}")

        for batch in train_loader:
            # Skip until we reach the saved step
            if global_step < resume_dict["global_step"]:
                global_step += 1
                if local_rank == 0:
                    inner_pb.update(1)
                continue

            start = time.time()
            num_loss_counted_tokens = float(
                torch.tensor([batch.pop("num_loss_counted_tokens")])
            )
            micro_batch_size = float(torch.tensor([batch.pop("num_samples")]))
            for k in batch:
                batch[k] = batch[k].to(local_rank)
            output = model(
                **batch,
                use_cache=False,
            )
            loss = output.loss
            log_loss = loss.detach().item()

            num_loss_counted_tokens, micro_batch_size, log_loss = map(
                float,
                accelerator.reduce(
                    torch.tensor(
                        [num_loss_counted_tokens, micro_batch_size, log_loss],
                        dtype=torch.float32,
                        device=accelerator.device,
                    ),
                    reduction="sum",
                ),
            )
            samples_seen += int(micro_batch_size)

            # num_loss_counted_tokens = aggregated_values[0]
            loss = (
                loss / num_loss_counted_tokens * world_size
            )  # dividing by the total number of non-padding tokens and multiplying by the number of GPUs so when accelerate averages by world_size, it will be the correct loss.
            print(
                f"Epoch: {epoch}, Step: {global_step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
            )
            accelerator.backward(loss)

            if global_step % grad_accum == 0:
                global_grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if local_rank == 0:
                elapsed_time = time.time() - start
                overall_throughput = args.samples_per_gpu * world_size / elapsed_time
                current_lr = lr_scheduler.get_last_lr()[0]
                cuda_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                cuda_malloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]
                global_grad_norm = (
                    model.get_global_grad_norm()
                    if hasattr(model, "get_global_grad_norm")
                    else global_grad_norm
                )
                global_grad_norm = (
                    float(global_grad_norm) if global_grad_norm is not None else None
                )
                # TODO - Bring back weight_norm gather
                # weight_norm = float(
                #     model.optimizer.single_partition_of_fp32_groups[0].norm()
                # )

                # TODO - Bring back consistent gradnorm and weight_norm logging
                metric_logger.log_sync(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "rank": torch.distributed.get_rank(),
                        "overall_throughput": overall_throughput,
                        "lr": current_lr,
                        "cuda_mem_allocated": cuda_mem_allocated,
                        "cuda_malloc_retries": cuda_malloc_retries,
                        "num_loss_counted_tokens": int(num_loss_counted_tokens),
                        "batch_size": int(micro_batch_size),
                        "total_loss": float(log_loss / num_loss_counted_tokens),
                        "samples_seen": samples_seen,
                        "gradnorm": global_grad_norm,
                        "total_samples": len(train_loader.dataset),
                        # "weight_norm": weight_norm,
                    }
                )

            if args.save_samples > 0 and (
                global_step * batch_size % args.save_samples == 0
            ):
                save_hf_format_accelerate(
                    args=args,
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    samples_seen=samples_seen,
                )
                # save_full_state(args, accelerator, global_step, samples_seen)

            global_step += 1
            if local_rank == 0:
                inner_pb.update(1)
            torch.cuda.empty_cache()

        save_full_state(args, accelerator, global_step, samples_seen)

def main(args):

    metric_logger = AsyncStructuredLogger(
        args.output_dir
        + f"/training_params_and_metrics_global{os.environ['RANK']}.jsonl"
    )
    if os.environ["LOCAL_RANK"] == "0":
        print(f"\033[38;5;120m{yaml.dump(vars(args), sort_keys=False)}\033[0m")
        metric_logger.log_sync({"script_params": vars(args)})

    setup_logger(args.log_level)
    CHAT_TEMPLATE, SPECIAL_TOKENS = retrieve_chat_template(args.chat_tmpl_path)
    tokenizer = setup_tokenizer(args.model_name_or_path, SPECIAL_TOKENS, CHAT_TEMPLATE)

    init_distributed_environment(args)

    flash_enabled = supports_flash_attention()

    dataset = setup_dataset(
        args.data_path,
        mock=args.mock_data,
        mock_len=args.mock_len,
    )


    packing_max_batch_len, grad_accum = find_packing_max_batch_len_and_grad_accum(
        num_gpus=torch.distributed.get_world_size(),
        avg_sample_len=dataset.get_lengths().mean(),
        effective_batch_size=args.effective_batch_size,
        max_batch_len_per_gpu=args.max_batch_len,
        is_padding=not flash_enabled,
        dataset=dataset,
        seed=args.seed,
    )
    args.sampler = "multipack"

    args.samples_per_gpu = (
        args.effective_batch_size // grad_accum // torch.distributed.get_world_size()
    )

    train_loader = setup_dataloader(
        dataset,
        tokenizer.pad_token_id,
        num_workers=8,
        flash_enabled=flash_enabled,
        max_batch_len=args.max_batch_len,
        packing_max_batch_len=packing_max_batch_len,
        samples_per_gpu=args.samples_per_gpu,
        sampler=args.sampler,
        seed=args.seed,
    )

    if args.local_rank == 0:
        metric_logger.log_sync(
            {
                "num_gpus": torch.distributed.get_world_size(),
                "avg_sample_len": dataset.get_lengths().mean(),
                "effective_batch_size": args.effective_batch_size,
                "max_batch_len_per_gpu": args.max_batch_len,
                "packing_max_batch_len": packing_max_batch_len,
                "grad_accum": grad_accum,
                "num_batches": len(train_loader),
                "avg_samples_per_batch": len(dataset) / len(train_loader),
                "samples_per_gpu": args.samples_per_gpu,
                "total_samples": len(dataset),  # emit the total number of samples
            }
        )

    model, lr_scheduler, optimizer, accelerator = setup_model(
        args, tokenizer, train_loader, grad_accum, flash_enabled
    )
    
    # Load checkpoint if it exists
    resume_dict = load_full_state_if_exists(args, accelerator)

    train(
        args,
        model,
        optimizer,
        lr_scheduler,
        accelerator,
        tokenizer,
        train_loader,
        grad_accum,
        metric_logger,
        resume_dict,
    )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    # General Settings
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level."
    )
    parser.add_argument(
        "--chat_tmpl_path",
        type=str,
        required=True,
        help="Path to the chat template file."
    )

    # Model and Tokenizer
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pre-trained model or model identifier from huggingface.co/models."
    )

    # Data Handling
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training data file."
    )
    parser.add_argument(
        "--mock_data",
        action='store_true',
        help="Use a mock dataset for training."
    )
    parser.add_argument(
        "--mock_len",
        type=int,
        default=2600,
        help="Maximum sequence length for mock samples."
    )

    # Training Parameters
    parser.add_argument(
        "--effective_batch_size",
        type=int,
        required=True,
        help="Effective batch size for training."
    )
    parser.add_argument(
        "--max_batch_len",
        type=int,
        required=True,
        help="Maximum number of tokens per batch."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--NEFTune_alpha",
        type=float,
        default=0.0,
        help="Noise scaling factor for embeddings."
    )

    # Scheduler and Optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="Learning rate scheduler type."
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for the scheduler."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Total number of training epochs."
    )
    parser.add_argument(
        "--save_samples",
        type=int,
        default=1000,
        help="Frequency of saving model checkpoints based on samples seen."
    )

    # FSDP Configuration
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"],
        help="Sharding strategy for Fully Sharded Data Parallel."
    )
    parser.add_argument(
        "--cpu_offload_params_fsdp",
        action='store_true',
        help="Enable CPU offloading for FSDP parameters."
    )

    parser.add_argument(
        "--disable_resume",
        action="store_true",
        help="Disable automatic resuming from last checkpoint"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
'''
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 \
--rdzv_endpoint="localhost:29500" simple_main_ds.py \
--model_name_or_path=meta-llama/Llama-3.1-8B \
--data_path="/dev/shm/data.jsonl" \
--output_dir="/home/ubuntu/experiments/ap-llama-3.1-rhel3.0_28-01-25" \
--effective_batch_size=3840 \
--max_batch_len=120000 \
--num_epochs=10 \
--lr_scheduler="constant_with_warmup" \
--learning_rate=6e-06 \
--num_warmup_steps=25 \
--save_samples=8000 \
--log_level="INFO" \
--fsdp_sharding_strategy="SHARD_GRAD_OP" \
--seed=42 \
--chat_tmpl_path="chat_templates/ibm_generic_tmpl.py" | tee /home/ubuntu/experiments/ap-llama-3.1-rhel3.0_28-01-25/train_rank_0.log
'''
