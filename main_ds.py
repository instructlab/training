import argparse
from datetime import timedelta
import math
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler, MistralForCausalLM
from dolomite_engine.hf_models.models import GPTDolomiteForCausalLM
from torch.distributed import (
    ReduceOp,
    all_reduce,
)

import deepspeed
from deepspeed.ops.adam import FusedAdam
from multipack_sampler import find_packing_max_batch_len_and_grad_accum
from token_dataset import setup_dataloader, setup_dataset
from tokenizer_utils import setup_tokenizer
from utils import (
    save_hf_format_ds,
    set_random_seed,
    setup_logger,
    convert_loss_to_reduce_sum,
)


def get_ds_config(world_size, samples_per_gpu, grad_accum):
    ds_config = {
        "train_batch_size": samples_per_gpu * world_size * grad_accum,
        "gradient_accumulation_steps": grad_accum,
        "train_micro_batch_size_per_gpu": samples_per_gpu,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": 2,
            "offload_param": {"device": "none"},
            "offload_optimizer": {"device": "none"},
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    return ds_config


def setup_model(args, tokenizer, train_loader, grad_accum):
    if args.is_granite:
        model = GPTDolomiteForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_padding_free_transformer=True,
        )
    else:
        bnb_config = None
        if args.lora_r > 0 and args.lora_quant_bits == 4:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,  # if not set will throw a warning about slow speeds when training
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )

    if len(tokenizer) > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    assert model.__class__.__name__ in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM",
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
    ], f"Model class name: {model.__class__.__name__} is not supported."

    model = convert_loss_to_reduce_sum(model, is_granite=args.is_granite)
    if args.is_granite:
        from dolomite_engine.gradient_checkpointing import apply_gradient_checkpointing
        from dolomite_engine.enums import GradientCheckpointingMethod

        block_name = model._no_split_modules[0]
        apply_gradient_checkpointing(
            model,
            GradientCheckpointingMethod.block,
            block_name=block_name,
            use_reentrant=True # this should be the HF default mode
        )
    elif args.lora_r > 0:
        # if lora
        from peft import LoraConfig
        from utils import prepare_peft_model, patch_target_module

        if args.lora_target_modules is None:
            args.__dict__['target_modules'] = ["q_proj", "k_proj", "v_proj", "o_proj"]

        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        )
        prepare_peft_model(model, peft_config)

        # patch DS to work with quantized models
        from deepspeed import DeepSpeedEngine
        from functools import partial

        if args.lora_quant_bits is not None:
            patch_target_module(
                'deepspeed.DeepSpeedEngine',
                partial(DeepSpeedEngine, dont_change_device=True)
            )
    else:
        model.gradient_checkpointing_enable()

    optimizer = FusedAdam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_epochs * len(train_loader),
    )

    model, _, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=get_ds_config(
            world_size=torch.distributed.get_world_size(),
            samples_per_gpu=args.samples_per_gpu,
            grad_accum=grad_accum,
        ),
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )
    # model = torch.compile(model)
    return model


def train(args, model, tokenizer, train_loader, grad_accum):
    model.train()

    global_step = 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    batch_size = args.effective_batch_size // grad_accum
    args.save_samples = (args.save_samples // batch_size) * batch_size
    (
        print(f"\033[93mNumber of samples per save: {args.save_samples}\033[0m")
        if local_rank == 0
        else None
    )

    for epoch in range(args.num_epochs):
        torch.distributed.barrier()
        train_loader.batch_sampler.set_epoch(epoch)

        if local_rank == 0:
            inner_pb = tqdm(range(len(train_loader)), desc=f"Epoch {epoch}")

        aggregated_values = torch.zeros(3, dtype=torch.float32).to(local_rank)
        for batch in train_loader:
            start = time.time()
            aggregated_values[0] = batch.pop("num_loss_counted_tokens")
            aggregated_values[1] = len(batch["input_ids"])
            if not args.is_granite:
                for k in batch:
                    batch[k] = batch[k].to(local_rank)

            output = model(
                **batch,
                use_cache=False,
            )
            loss = output.loss

            aggregated_values[2] = loss.item()

            all_reduce(aggregated_values, op=ReduceOp.SUM)

            num_loss_counted_tokens = aggregated_values[0]
            loss = (
                loss / num_loss_counted_tokens * world_size
            )  # dividing by the total number of non-padding tokens and multiplying by the number of GPUs so when deepspeed averages by world_size, it will be the correct loss.

            print(
                f"\033[93mPer-token loss scaled by world size: {(loss/num_loss_counted_tokens) * world_size}\033[0m"
            )
            print(
                f"Epoch: {epoch}, Step: {global_step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
            )

            model.backward(loss)
            model.step()

            if local_rank == 0:
                elapsed_time = time.time() - start
                overall_throughput = args.samples_per_gpu * world_size / elapsed_time
                current_lr = model.lr_scheduler.get_last_lr()[0]
                cuda_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                cuda_malloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]

                print(
                    f"throughput: {overall_throughput} "
                    f"samples/s, lr: {current_lr}, "
                    f"loss: {loss.item()} "
                    f"cuda_mem_allocated: {cuda_mem_allocated} GB "
                    f"cuda_malloc_retries: {cuda_malloc_retries} "
                    f"num_loss_counted_tokens: {num_loss_counted_tokens} "
                    f"batch_size: {aggregated_values[1]} "
                    f"total loss: {aggregated_values[2]/num_loss_counted_tokens}"
                )

            if global_step * batch_size % args.save_samples == 0:
                save_hf_format_ds(
                    args,
                    model,
                    tokenizer,
                    global_step * args.samples_per_gpu * world_size,
                )

            global_step += 1
            if local_rank == 0:
                inner_pb.update(1)
            torch.cuda.empty_cache()


def main(args):
    import yaml

    if os.environ["LOCAL_RANK"] == "0":
        print(f"\033[38;5;120m{yaml.dump(vars(args), sort_keys=False)}\033[0m")

    setup_logger(args.log_level)
    tokenizer = setup_tokenizer(args.model_name_or_path)
    # device = torch.device("cuda", args.local_rank)

    #### distributed init #####
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])
    deepspeed.init_distributed(timeout=timedelta(minutes=360))
    args.global_rank = torch.distributed.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

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
    )
    args.samples_per_gpu = (
        args.effective_batch_size // grad_accum // torch.distributed.get_world_size()
    )

    train_loader = setup_dataloader(
        dataset,
        tokenizer.pad_token_id,
        num_workers=8,
        is_granite=args.is_granite,
        max_batch_len=args.max_batch_len,
        packing_max_batch_len=packing_max_batch_len,
        seed=args.seed,
    )

    if args.local_rank == 0:
        print(
            f"\033[96mnum_gpus: {torch.distributed.get_world_size()}\n"
            f"avg_sample_len: {dataset.get_lengths().mean()}\n"
            f"effective_batch_size: {args.effective_batch_size}\n"
            f"max_batch_len_per_gpu: {args.max_batch_len}\n"
            f"packing_max_batch_len: {packing_max_batch_len}\n"
            f"grad_accum: {grad_accum}\n"
            f"num batches: {len(train_loader)}\n"
            f"avg_samples_per_batch: {len(dataset)/len(train_loader)}\n"
            f"samples_per_gpu: {args.samples_per_gpu}\033[0m"
        )

    model = setup_model(args, tokenizer, train_loader, grad_accum)

    train(args, model, tokenizer, train_loader, grad_accum)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_epochs", type=int, default=1)
    # parser.add_argument("--samples_per_gpu", type=int, default=8)
    parser.add_argument("--effective_batch_size", type=int, default=3840)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_samples", type=int)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock_data", action="store_true")
    parser.add_argument("--mock_len", type=int, default=2600)
    parser.add_argument(
        "--sharding_strategy",
        type=str,
        # choices=[e.name for e in ShardingStrategy],
        default="FULL_SHARD",
        help="Sharding strategy to be used for distributed training.",
    )
    parser.add_argument("--is_granite", action="store_true")
    parser.add_argument("--lora_r", type=int, default=0) # set to > 0 to activate lora
    parser.add_argument("--lora_alpha", type=int, default=32) 
    parser.add_argument("--lora_dropout", type=float, default=0.1) 
    parser.add_argument("--lora_quant_bits", type=int, default=None) 
    parser.add_argument("--lora_target_modules", type=int, nargs='+', default=None) 
    parser.add_argument("--max_batch_len", type=int, default=60000)
    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)

"""
pkill python
git reset --hard
git pull
export WORLD_SIZE=1
sleep 3
mkdir -p /new_data/experiments/ap-fsdp-p00-old-m-ds-2t
cd /app/fsdp
export WORLD_SIZE=1
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK \
--nproc_per_node=8 --rdzv_id=101 \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" main_ds.py \
--model_name_or_path=mistralai/Mistral-7B-v0.1 \
--data_path="/dev/shm/data.jsonl" \
--output_dir="/new_data/experiments/ap-fsdp-p00-old-m-ds-2t" \
--num_epochs=100 \
--samples_per_gpu=24 \
--learning_rate=1e-06 \
--num_warmup_steps=800 \
--gradient_accumulation_steps=2 \
--save_samples=12000 \
--log_level="INFO" \
--mock_data \
--mock_len=2048 \
--seed=42 | tee /new_data/experiments/ap-fsdp-p00-old-m-ds-2t/$RANK.log
export WORLD_SIZE=1
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK \
--nproc_per_node=8 --rdzv_id=101 \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" main_ds.py \
--model_name_or_path=/new_data/models/granite7b/ibm_models_version/ \
--data_path="/dev/shm/data.jsonl" \
--output_dir="/new_data/experiments/ap-granite-4t" \
--num_epochs=100 \
--samples_per_gpu=240 \
--learning_rate=2e-05 \
--num_warmup_steps=385 \
--gradient_accumulation_steps=2 \
--save_samples=250000 \
--log_level="INFO" \
--sharding_strategy="HYBRID_SHARD" \
--is_granite \
--max_batch_len 70000 \
--seed=42
"""
