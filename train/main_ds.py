import argparse
from pathlib import Path
from datetime import timedelta
import math
import os
import re
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler, MistralForCausalLM
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
    save_model_ds_native,
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
    bnb_config = None
    if args.lora_r > 0 and args.lora_quant_bits == 4:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # if not set will throw a warning about slow speeds when training
        )

    if args.is_granite:
        from dolomite_engine.hf_models.models import GPTDolomiteForCausalLM

        model = GPTDolomiteForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_padding_free_transformer=True,
            quantization_config=bnb_config,
        )
    else:
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

    # Fix any discrepancy between model and tokenizer
    if model.config.pad_token_id is not None and tokenizer.pad_token_id is not None and model.config.pad_token_id != tokenizer.pad_token_id:
        print(
            f"WARNING: There is a mismatch between pad token id of model ({model.config.pad_token_id}) and tokenizer({tokenizer.pad_token_id}). Fixing model pad token id to be same as tokenizer's pad token id"
        )
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.bos_token_id is not None and tokenizer.bos_token_id is not None and model.config.bos_token_id != tokenizer.bos_token_id:
        print(
            f"WARNING: There is a mismatch between bos token id of model({model.config.bos_token_id}) and tokenizer({tokenizer.bos_token_id}). Fixing model bos token id to be same as tokenizer's bos token id"
        )
        model.config.bos_token_id = tokenizer.bos_token_id
    if model.config.eos_token_id is not None and tokenizer.eos_token_id and model.config.eos_token_id != tokenizer.eos_token_id:
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
    ], f"Model class name: {model.__class__.__name__} is not supported."

    model = convert_loss_to_reduce_sum(model, is_granite=args.is_granite)

    # handling of gradient checkpointing
    # it is handled differently for lora and full
    # - with the exception of granite, which handles it
    #   in the later stanza
    if args.lora_r > 0:
        # if lora
        from peft import LoraConfig
        from utils import prepare_peft_model, patch_target_module

        if args.lora_target_modules is None:
            args.__dict__["lora_target_modules"] = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]

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

        # patch DS to work with quantized models
        from deepspeed import DeepSpeedEngine
        from functools import partial

        if args.lora_quant_bits is not None:
            patch_target_module(
                "deepspeed.DeepSpeedEngine",
                partial(DeepSpeedEngine, dont_change_device=True),
            )
    elif not args.is_granite:
        model.gradient_checkpointing_enable()

    # granite gradient checkpointing is handled uniformly
    # for both lora and full here
    if args.is_granite:
        from dolomite_engine.gradient_checkpointing import apply_gradient_checkpointing
        from dolomite_engine.enums import GradientCheckpointingMethod

        block_name = model._no_split_modules[0]
        apply_gradient_checkpointing(
            model,
            GradientCheckpointingMethod.block,
            block_name=block_name,
            use_reentrant=True,  # this should be the HF default mode
        )

        if args.lora_r > 0:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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


# this function is to check if the checkpoint provided can be resumed
def maybe_resume_training(args, model):
    local_rank = int(os.environ["LOCAL_RANK"])

    # DS's loading function will not raise if fails to reload a checkpoint
    # - if lora is used, then the checkpoints will only be for the adapters
    #   so we need to disable load_module_strict
    # - load checkpoint will find the latest checkpoint
    # - it will also load the optimizer and scheduler states by default
    load_module_strict = args.lora_r == 0  # can only be true if lora is not used
    output_dir = Path(args.output_dir) / "ds_native"
    model.load_checkpoint(output_dir, load_module_strict=load_module_strict)

    output_dir = Path(args.output_dir) / "ds_native"
    # need to figure out the resumed start step
    latest_file = output_dir / "latest"
    try:
        with open(latest_file) as f:
            # there is some assumption here that the ds_native
            # checkpoints are tagged as <something>_(samples_seen)
            samples_seen = f.read()
            (samples_seen,) = re.match("\w+_(\d+)", samples_seen).groups()
            samples_seen = int(samples_seen)

            last_step = samples_seen // args.effective_batch_size
            args.__dict__["last_step"] = last_step
        (
            print(f"\033[93mStarting from: {last_step}\033[0m")
            if local_rank == 0
            else None
        )
    except FileNotFoundError:
        pass

    # we will update the start step here
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
    if args.save_samples_ds is not None:
        args.save_samples_ds = (args.save_samples_ds // batch_size) * batch_size
        (
            print(
                f"\033[93mNumber of samples per DS save: {args.save_samples_ds}\033[0m"
            )
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
            if global_step <= args.last_step:
                # in the case of resuming, last_step > 0
                global_step += 1
                if local_rank == 0:
                    inner_pb.update(1)
                continue

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

            if (
                args.save_samples_ds is not None
                and global_step * batch_size % args.save_samples_ds == 0
            ):
                save_model_ds_native(
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
    model = maybe_resume_training(args, model)

    train(args, model, tokenizer, train_loader, grad_accum)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def run_training(torch_args: TorchrunTrainArgs, train_args: FullTrainArgs):
    """
    Wrapper around the main training job that calls torchrun.
    """
    try:
        command = [
            "torchrun",
            f"--nnodes={torch_args.nnodes}",
            f"--node_rank={torch_args.node_rank}",
            f"--nproc_per_node={torch_args.nproc_per_node}",
            f"--rdzv_id={torch_args.rdzv_id}",
            f"--rdzv_endpoint={torch_args.rdzv_endpoint}",
            __file__,
            f"--model_name_or_path={train_args.model_name_or_path}",
            f"--data_path={train_args.data_path}",
            f"--output_dir={train_args.output_dir}",
            f"--num_epochs={train_args.num_epochs}",
            f"--effective_batch_size={train_args.effective_batch_size}",
            f"--learning_rate={train_args.learning_rate}",
            f"--num_warmup_steps={train_args.num_warmup_steps}",
            f"--save_samples={train_args.save_samples}",
            f"--log_level={train_args.log_level}",
            f"--max_batch_len={train_args.max_batch_len}",
            f"--seed={train_args.seed}",
        ]

        if train_args.mock_data:
            command.append("--mock_data")
            if train_args.mock_len:
                command.append(f"--mock_len={train_args.mock_len}")

        if train_args.is_granite:
            command.append("--is_granite")

        print(f"\033[92mRunning command: {' '.join(command)}\033[0m")

        with open("logfile.out", "w", encoding="utf-8") as logfile:
            subprocess.run(
                command,
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )

        # stream the stdout and stderr output
        # process = subprocess.Popen(
        #     command,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT,
        #     bufsize=1,
        #     universal_newlines=True
        # )
        # while True:
        #     output = process.stdout.readline()
        #     if output == "" and process.poll() is not None:
        #         break
        #     if output:
        #         print(output.strip())

        #     rc = process.poll()

        #     if rc != 0:
        #         if process.stderr:
        #             print(process.stderr)
        #         if process.stdout:
        #             print(process.stdout)
        #         break

        # for line in iter(process.stdout.readline, b''):
        #     print(line.decode('utf-8'), end='')
        # process.stdout.close()
        # process.wait()
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if "process" not in locals() or process is None:
            return

        print("\033[91mTerminating process 🤖\033[0m")
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("\033[91mProcess did not terminate in time, killing it.\033[0m")
            process.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--last_step",
        type=int,
        default=0,
        help="understand this as the last completed step. "
        "The default is 0, since global_step starts from 1 by default.",
    )
    # parser.add_argument("--samples_per_gpu", type=int, default=8)
    parser.add_argument("--effective_batch_size", type=int, default=3840)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_samples", type=int)
    parser.add_argument(
        "--save_samples_ds",
        type=int,
        help="for saving in ds native format",
        default=None,
    )
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
    parser.add_argument("--lora_r", type=int, default=0)  # set to > 0 to activate lora
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_quant_bits", type=int, default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
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
