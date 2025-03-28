# SPDX-License-Identifier: Apache-2.0

# Standard
from copy import deepcopy
from pathlib import Path
import argparse
import math
import os
import re
import subprocess
import time

# Third Party
from accelerate import Accelerator

try:
    # Third Party
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except ImportError:
    DeepSpeedCPUAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        print(
            "DeepSpeed CPU Optimizer is not available. Some features may be unavailable."
        )

try:
    # Third Party
    from deepspeed.ops.adam import FusedAdam
    from deepspeed.runtime.zero.utils import ZeRORuntimeException
except ImportError:
    FusedAdam = None
    ZeRORuntimeException = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        print("DeepSpeed is not available. Some features may be unavailable.")

# Third Party
from instructlab.dolomite.hf_models import GPTDolomiteForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    get_scheduler,
)
import torch
import torch.distributed

# First Party
from instructlab.training import config
from instructlab.training.async_logger import AsyncStructuredLogger

# pylint: disable=no-name-in-module
from instructlab.training.config import DistributedBackend, TorchrunArgs, TrainingArgs
from instructlab.training.multipack_sampler import (
    find_packing_max_batch_len_and_grad_accum,
)
from instructlab.training.setup_accelerator import setup_accelerator
from instructlab.training.token_dataset import setup_dataloader, setup_dataset
from instructlab.training.tokenizer_utils import setup_tokenizer
from instructlab.training.utils import (
    StreamablePopen,
    add_noisy_embeddings,
    apply_gradient_checkpointing,
    check_flash_attn_enabled,
    check_valid_train_args,
    convert_loss_to_reduce_sum,
    create_lora_config,
    ensure_loadable_dolomite_checkpoint,
    load_latest_full_state,
    prepare_peft_model,
    prepare_universal_checkpoint_from_latest,
    save_checkpoint,
    save_hf_format_accelerate,
    set_random_seed,
    setup_logger,
)
import instructlab.training.data_process as dp


def setup_optimizer(args, model):
    if args.distributed_training_framework == DistributedBackend.FSDP.value:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
    elif args.distributed_training_framework == DistributedBackend.DEEPSPEED.value:
        # need to use this only when the CPU offload optimizer is enabled
        if args.cpu_offload_optimizer:
            print(
                "\033[33m!!! CPU offload optimizer enabled, using DeepSpeedCPUAdam !!!\033[0m"
            )
            optimizer = DeepSpeedCPUAdam(
                model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95)
            )
        else:
            optimizer = FusedAdam(
                model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95)
            )
    else:
        raise ValueError(
            f"Sharding framework {args.distributed_training_framework} is not supported."
        )
    return optimizer


def setup_model(
    args, tokenizer: PreTrainedTokenizer, train_loader, grad_accum, flash_enabled
):
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
    if flash_enabled:
        base_model_args["attn_implementation"] = "flash_attention_2"

    if args.use_dolomite:
        with ensure_loadable_dolomite_checkpoint(
            args.model_name_or_path, args.output_dir
        ) as path:
            base_model_args["pretrained_model_name_or_path"] = path
            base_model_args["use_padding_free_transformer"] = True
            model = GPTDolomiteForCausalLM.from_pretrained(
                **base_model_args,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(**base_model_args)

    # store the base model args so we can recall them later if saving a LoRA model
    args.base_model_args = base_model_args

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

    if "ForCausalLM" not in model.__class__.__name__:
        raise ValueError(
            f"Model class name: {model.__class__.__name__} is not supported."
        )

    # ensure the model has any tokens which were added to the tokenizer
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None and model.config.bos_token_id is None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None and model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id

    model = convert_loss_to_reduce_sum(model, use_dolomite=args.use_dolomite)
    model = add_noisy_embeddings(model, noise_alpha=args.NEFTune_alpha)

    # handling of gradient checkpointing
    # it is handled differently for lora and full
    # - with the exception of granite, which handles it
    #   in the later stanza
    if args.lora_r > 0:
        lora_config = create_lora_config(model, args)
        model = prepare_peft_model(
            model,
            lora_config,
            args.distributed_training_framework,
            gradient_checkpointing=not args.use_dolomite,
        )
        args.lora_config = lora_config
    elif not args.use_dolomite:
        model.gradient_checkpointing_enable()

    # granite gradient checkpointing is handled uniformly
    # for both lora and full here
    if args.use_dolomite:
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
    # Necessary so that Accelerate does not step once per GPU
    # see https://github.com/huggingface/accelerate/blob/127818fc27ebe5cb236357fff59ff1748326d643/src/accelerate/scheduler.py#L69
    lr_scheduler.split_batches = True
    return model, lr_scheduler, optimizer, accelerator


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

    try:
        # attempt to load a regular checkpoint first
        model.load_checkpoint(output_dir, load_module_strict=load_module_strict)
    except ZeRORuntimeException as e:
        if str(e).startswith("The checkpoint being loaded used a DP world size of"):
            # if it fails with the above exception, then a universal
            # checkpoint is required

            # prepare the universal checkpoint
            # - by reading 'latest' to get the resumable checkpoint
            prepare_universal_checkpoint_from_latest(output_dir)

            # need to do this to trigger the universal checkpoint
            # loading
            model._config.load_universal_checkpoint = True

            # then attempt to load again
            model.load_checkpoint(output_dir, load_module_strict=load_module_strict)

            # reset to regular checkpoint loading
            model._config.load_universal_checkpoint = False
        else:
            raise e  # reraise

    # do this to figure out the last_step
    latest_file = output_dir / "latest"
    try:
        with open(latest_file) as f:
            # there is some assumption here that the ds_native
            # checkpoints are tagged as <something>_(samples_seen)
            step_folder = f.read()
            (samples_seen,) = re.match("\w+_(\d+)", step_folder).groups()
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


def train(
    args,
    model,
    optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    tokenizer: PreTrainedTokenizer,
    train_loader: DataLoader,
    grad_accum,
    metric_logger,
):
    model.train()

    global_step = 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    batch_size = args.effective_batch_size // grad_accum
    samples_seen = 0

    if hasattr(args, "samples_seen"):
        print(f"\033[93mUpdating 'samples_seen' {args.samples_seen}\033[0m")
        samples_seen = args.samples_seen

    if args.save_samples > 0:
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

    global_grad_norm = None
    for epoch in range(args.current_epoch, args.num_epochs):
        if args.sampler in ("multipack"):
            train_loader.batch_sampler.set_epoch(epoch)
        elif args.sampler in ("distributed"):
            train_loader.sampler.set_epoch(epoch)
        else:
            raise NotADirectoryError

        if local_rank == 0:
            inner_pb = tqdm(range(len(train_loader)), desc=f"Epoch {epoch}")

        # blast through the batches in the train loader up to the last step within the epoch.
        for batch in train_loader:
            if global_step <= args.last_step:
                # in the case of resuming, last_step > 0
                global_step += 1
                if local_rank == 0:
                    inner_pb.update(1)
                continue
            start = time.time()
            num_loss_counted_tokens = float(
                torch.tensor([batch.pop("num_loss_counted_tokens")])
            )
            micro_batch_size = float(torch.tensor([batch.pop("num_samples")]))
            if not args.use_dolomite:
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
                save_checkpoint(
                    args=args,
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    samples_seen=samples_seen,
                    is_lora=bool(args.lora_r),
                    hf_format=True,
                )

            # if (
            #     args.save_samples_ds is not None
            #     and global_step * batch_size % args.save_samples_ds == 0
            # ):
            #     save_model_ds_native(
            #         args,
            #         model,
            #         tokenizer,
            #         global_step * args.samples_per_gpu * world_size,
            #     )
            global_step += 1
            if local_rank == 0:
                inner_pb.update(1)
            torch.cuda.empty_cache()
        if args.checkpoint_at_epoch:
            save_checkpoint(
                args=args,
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                samples_seen=samples_seen,
                is_lora=bool(args.lora_r),
                full_state=args.accelerate_full_state_at_epoch,
                hf_format=True,
                epoch=epoch,
            )

    if args.save_last:
        save_hf_format_accelerate(
            args,
            model,
            tokenizer,
            accelerator,
            samples_seen,
            is_lora=bool(args.lora_r),
        )


def main(args):
    # Third Party
    import yaml

    if args.distributed_training_framework == "deepspeed" and not FusedAdam:
        raise ImportError(
            "DeepSpeed was selected but we cannot import the `FusedAdam` optimizer"
        )

    if (
        args.distributed_training_framework == "deepspeed"
        and args.cpu_offload_optimizer
        and not DeepSpeedCPUAdam
    ):
        raise ImportError(
            "DeepSpeed was selected and CPU offloading was requested, but DeepSpeedCPUAdam could not be imported. This likely means you need to build DeepSpeed with the CPU adam flags."
        )

    metric_logger = AsyncStructuredLogger(
        args.output_dir
        + f"/training_params_and_metrics_global{os.environ['RANK']}.jsonl"
    )
    if os.environ["LOCAL_RANK"] == "0":
        print(f"\033[38;5;120m{yaml.dump(vars(args), sort_keys=False)}\033[0m")
        metric_logger.log_sync({"script_params": vars(args)})

    setup_logger(args.log_level)
    tokenizer = setup_tokenizer(args.model_name_or_path, args.chat_tmpl_path)
    # device = torch.device("cuda", args.local_rank)

    model_conf = AutoConfig.from_pretrained(args.model_name_or_path)
    args.model_type = model_conf.model_type

    #### distributed init #####
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl")
    args.global_rank = torch.distributed.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

    flash_enabled = check_flash_attn_enabled(args.disable_flash_attn, args.use_dolomite)

    dataset = setup_dataset(
        args.data_path,
        mock=args.mock_data,
        mock_len=args.mock_len,
    )

    try:
        packing_max_batch_len, grad_accum = find_packing_max_batch_len_and_grad_accum(
            num_gpus=torch.distributed.get_world_size(),
            avg_sample_len=dataset.get_lengths().mean(),
            effective_batch_size=args.effective_batch_size,
            max_batch_len_per_gpu=args.max_batch_len,
            is_padding=not (args.use_dolomite or flash_enabled),
            dataset=dataset,
            seed=args.seed,
        )
        args.sampler = "multipack"
    except RuntimeError as e:
        if os.environ["LOCAL_RANK"] == "0":
            print(f"\033[38;5;120m{e}\033[0m")

        # fallback to grad accum = 1
        # NOTE: packing max batch len will not be used
        packing_max_batch_len = None
        grad_accum = 1
        args.sampler = "distributed"

    args.samples_per_gpu = (
        args.effective_batch_size // grad_accum // torch.distributed.get_world_size()
    )

    train_loader = setup_dataloader(
        dataset,
        tokenizer.pad_token_id,
        num_workers=8,
        use_dolomite=args.use_dolomite,
        flash_enabled=flash_enabled,
        max_batch_len=args.max_batch_len,
        packing_max_batch_len=packing_max_batch_len,
        samples_per_gpu=args.samples_per_gpu,
        sampler=args.sampler,
        seed=args.seed,
    )
    if len(train_loader) == 0:
        # this happens sometimes when we have more GPUs than data to process. In this case
        # we should either alert the user to switch samplers, or do it automatically and
        # warn them about it happening
        print(
            "\033[93mThe dataset is too small for multipack to distribute all of the samples across GPUs. Falling back to the distributed sampler!\033[0m"
        )
        args.sampler = "distributed"
        train_loader = setup_dataloader(
            dataset,
            tokenizer.pad_token_id,
            num_workers=8,
            use_dolomite=args.use_dolomite,
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

    load_latest_full_state(args=args, accelerator=accelerator)

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
    )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


# public API
def run_training(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    """
    Wrapper around the main training job that calls torchrun.
    """
    check_valid_train_args(train_args)

    # switch out generic tmpl for legacy tmpl if requested
    if train_args.use_legacy_tmpl:
        train_args.chat_tmpl_path = os.path.join(
            os.path.dirname(__file__), "chat_templates/ibm_legacy_tmpl.py"
        )

    if train_args.process_data:
        # TODO(osilkin):
        #   Decouple the data processing logic from training.
        #   Now that we've decided that repos will be less tethered to the
        #   design choices of the `ilab` CLI, we can make this change.
        dp.process_data(
            data_output_path=train_args.data_output_dir,
            model_path=train_args.model_path,
            data_path=train_args.data_path,
            max_seq_len=train_args.max_seq_len,
            chat_tmpl_path=train_args.chat_tmpl_path,
            num_cpu_procs=train_args.data_process_num_cpu_procs,
        )

    if not os.path.exists(train_args.ckpt_output_dir):
        os.makedirs(train_args.ckpt_output_dir, exist_ok=True)

    command = [
        "torchrun",
        f"--nnodes={torch_args.nnodes}",
        f"--node_rank={torch_args.node_rank}",
        f"--nproc_per_node={torch_args.nproc_per_node}",
        f"--rdzv_id={torch_args.rdzv_id}",
        f"--rdzv_endpoint={torch_args.rdzv_endpoint}",
        __file__,
        f"--model_name_or_path={train_args.model_path}",
        f"--data_path={train_args.data_output_dir}/data.jsonl",
        f"--output_dir={train_args.ckpt_output_dir}",
        f"--num_epochs={train_args.num_epochs}",
        f"--effective_batch_size={train_args.effective_batch_size}",
        f"--learning_rate={train_args.learning_rate}",
        f"--num_warmup_steps={train_args.warmup_steps}",
        f"--save_samples={train_args.save_samples}",
        f"--log_level=INFO",
        f"--max_batch_len={train_args.max_batch_len}",
        f"--seed={train_args.random_seed}",
    ]

    if train_args.chat_tmpl_path is not None:
        command.append(f"--chat-tmpl-path={train_args.chat_tmpl_path}")

    if train_args.keep_last_checkpoint_only:
        command.append("--keep_last_checkpoint_only")

    if train_args.checkpoint_at_epoch:
        command.append("--checkpoint_at_epoch")

    if train_args.accelerate_full_state_at_epoch:
        command.append("--accelerate_full_state_at_epoch")

    if train_args.mock_data:
        command.append("--mock_data")
        if train_args.mock_len:
            command.append(f"--mock_len={train_args.mock_len}")

    if train_args.use_dolomite:
        command.append("--use_dolomite")

    if train_args.disable_flash_attn:
        command.append("--disable_flash_attn")

    if train_args.lora:
        command.extend(
            [
                f"--lora_r={train_args.lora.rank}",
                f"--lora_alpha={train_args.lora.alpha}",
                f"--lora_dropout={train_args.lora.dropout}",
                "--lora_target_modules",
            ]
        )
        if train_args.lora.target_modules:
            command.extend(train_args.lora.target_modules)
        # hard-code 4-bit quantization for now, change this when we add more
        quant_dtype = train_args.lora.quantize_data_type
        quantization_is_enabled = quant_dtype in (
            config.QuantizeDataType.NF4,
            config.QuantizeDataType.NF4.value,
        )
        if quantization_is_enabled:
            command.append("--lora_quant_bits=4")

    # specify which distributed training backend we use
    command.append(
        f"--distributed_training_framework={train_args.distributed_backend.value}"
    )

    # deepspeed options
    if train_args.distributed_backend == DistributedBackend.DEEPSPEED:
        if not FusedAdam:
            raise ImportError(
                "DeepSpeed was selected as the distributed backend, but FusedAdam could not be imported. Please double-check that DeepSpeed is installed correctly"
            )

        if train_args.deepspeed_options.cpu_offload_optimizer and not DeepSpeedCPUAdam:
            raise ImportError(
                "DeepSpeed CPU offloading was enabled, but DeepSpeedCPUAdam could not be imported. This is most likely because DeepSpeed was not built with CPU Adam. Please rebuild DeepSpeed to have CPU Adam, or disable CPU offloading."
            )
    if train_args.deepspeed_options.save_samples:
        command.append(f"--save_samples_ds={train_args.deepspeed_options.save_samples}")
    if train_args.deepspeed_options.cpu_offload_optimizer:
        command.extend(
            [
                "--cpu_offload_optimizer",
                f"--cpu_offload_optimizer_ratio={train_args.deepspeed_options.cpu_offload_optimizer_ratio}",
            ]
        )
        if train_args.deepspeed_options.cpu_offload_optimizer_pin_memory:
            command.append("--cpu_offload_optimizer_pin_memory")

    # FSDP Options
    if train_args.fsdp_options.cpu_offload_params:
        command.extend(
            [
                "--cpu_offload_params_fsdp",
            ]
        )

    # specify the sharding strategy
    command.append(
        f"--fsdp_sharding_strategy={train_args.fsdp_options.sharding_strategy.value}"
    )

    if train_args.keep_last_checkpoint_only:
        command.append("--keep_last_checkpoint_only")

    print(f"\033[92mRunning training command as subprocess: {' '.join(command)}\033[0m")
    process = None
    interrupt: KeyboardInterrupt | Exception | None = None
    failure = False
    try:
        process = StreamablePopen(
            f"{train_args.ckpt_output_dir}/full_logs_global{torch_args.node_rank}.log",
            command,
        )
        process.listen()
    except KeyboardInterrupt as e:
        print("Training subprocess interrupted by user.")
        interrupt = e
    except Exception as e:
        print("Unexpected exception received during distributed training")
        interrupt = e
    finally:
        if "process" not in locals() or process is None:
            return

        failure = process.poll() != 0
        if not failure:
            print("\033[92mOperation completed successfully! 🎉\033[0m")
        else:
            print(
                "\033[91mTraining subprocess has not exited yet. Sending SIGTERM.\033[0m"
            )

        process.terminate()
        try:
            print("Waiting for process to exit, 60s...")
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print(
                "\033[91mTraining subprocess did not terminate before timeout, sending SIGKILL.\033[0m"
            )
            process.kill()

        if interrupt:
            raise interrupt
        if failure:
            raise RuntimeError(
                "Suffered a failure during distributed training. Please see the training logs for more context."
            )


if __name__ == "__main__":
    # TODO(osilkin): Configure a type that these args must adhere to for the sake of type checking
    #               Maybe switch out from argparse to something smarter
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--current_epoch",
        type=int,
        default=0,
        help="Helpful flag for resuming on a later epoch. Sets dataloader correctly.",
    )
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
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--save_samples",
        type=int,
        help="The number of samples seen between each checkpoint save. If --save_samples<=0, this feature is disabled.",
    )
    parser.add_argument(
        "--save_samples_ds",
        type=int,
        help="for saving in ds native format",
        default=None,
    )
    parser.add_argument(
        "--save_last", action="store_true", help="save after finishing training"
    )
    parser.add_argument(
        "--checkpoint_at_epoch",
        action="store_true",
        help="Save a model checkpoint after finishing an epoch.",
    )
    parser.add_argument(
        "--accelerate_full_state_at_epoch",
        action="store_true",
        help="Save full model state using Accelerate after finishing an epoch.",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock_data", action="store_true")
    parser.add_argument("--mock_len", type=int, default=2600)
    parser.add_argument(
        "--distributed_training_framework",
        type=str,
        choices=[
            DistributedBackend.DEEPSPEED.value,
            DistributedBackend.FSDP.value,
        ],
        default=DistributedBackend.DEEPSPEED.value,
    )
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        # choices=[e.name for e in ShardingStrategy],
        default="SHARD_GRAD_OP",
        help="Sharding strategy to be used for FSDP distributed training.",
    )
    parser.add_argument("--use_dolomite", action="store_true")
    parser.add_argument("--lora_r", type=int, default=0)  # set to > 0 to activate lora
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_quant_bits", type=int, default=None)
    parser.add_argument(
        "--lora_target_modules",
        nargs="*",
        default=None,
        help="Which modules we should target for injecting LoRA layers. Defaults to selecting all projection layers when no values are provided.",
    )
    parser.add_argument("--max_batch_len", type=int, default=60000)
    parser.add_argument(
        "--cpu_offload_optimizer",
        action="store_true",
        default=False,
        help="Offload optimizer to CPU when using DeepSpeed. This configures it to use ZeRO stage 2.",
    )
    parser.add_argument(
        "--cpu_offload_params_fsdp",
        action="store_true",
        default=False,
        help="Offload to CPU when using FSDP.",
    )
    parser.add_argument(
        "--cpu_offload_optimizer_pin_memory",
        action="store_true",
        default=False,
        help="Pin memory when offloading optimizer to CPU. This allows for faster transfers between CPU and GPU. Comes at the cost of higher memory usage and CPU overhead.",
    )
    parser.add_argument(
        "--cpu_offload_optimizer_ratio",
        type=float,
        default=1.0,
        help="Ratio of the optimizer to be offloaded to CPU. The rest will be on GPU(s).",
    )
    parser.add_argument("--NEFTune_alpha", type=float, default=None)
    parser.add_argument(
        # TODO(osilkin): rename to chat_tmpl_path
        "--chat-tmpl-path",
        type=str,
        default=None,
        help="Path to the chat template to set on the model for training. If none is provided, the chat template used in the model will be used.",
    )
    parser.add_argument("--disable_flash_attn", action="store_true")
    parser.add_argument(
        "--keep_last_checkpoint_only",
        action="store_true",
        help=(
            "Keep only the last checkpoint directory - overwrite the previous ones. Useful for saving disk space."
            "The last checkpoint will be saved as 'last_epoch'."
        ),
    )
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
--fsdp_sharding_strategy="SHARD_GRAD_OP" \
--use_dolomite \
--max_batch_len 70000 \
--seed=42
"""
