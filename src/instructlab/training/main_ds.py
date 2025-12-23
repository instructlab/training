# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import datetime
import logging
import os
import subprocess
import time
import warnings

try:
    # Third Party
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except ImportError:
    DeepSpeedCPUAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        warnings.warn(
            "DeepSpeed CPU Optimizer is not available. Some features may be unavailable.",
            UserWarning,
        )

try:
    # Third Party
    from deepspeed.ops.adam import FusedAdam
except ImportError:
    FusedAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        warnings.warn(
            "DeepSpeed is not available. Some features may be unavailable.",
            UserWarning,
        )

# Third Party
from tqdm import tqdm
from transformers import AutoConfig
import torch
import torch.distributed as dist

# First Party
from instructlab.training import config
from instructlab.training.accelerator import Accelerator
from instructlab.training.batch_loss_manager import BatchLossManager
from instructlab.training.config import (
    DistributedBackend,
    ModelTypes,
    PretrainingConfig,
    TorchrunArgs,
    TrainingArgs,
)

# pylint: disable=no-name-in-module
from instructlab.training.logger import (
    propagate_package_logs,
    setup_metric_logger,
    setup_root_logger,
)
from instructlab.training.model import (
    CausalLMModel,
    LigerModel,
    Model,
    setup_optimizer,
)
from instructlab.training.sampler import get_data_loader

# Removed old multipack_sampler import - using mini_trainer approach
from instructlab.training.tokenizer_utils import setup_tokenizer
from instructlab.training.utils import (
    StreamablePopen,
    check_valid_train_args,
    freeze_router_params,
    load_latest_full_state,
    save_checkpoint,
    save_hf_format_accelerate,
    set_random_seed,
)
import instructlab.training.data_process as dp

logger = logging.getLogger(__name__)


def train(
    args,
    model: Model,
    accelerator: Accelerator,
):
    model.train()

    global_step = 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    metric_logger = logging.getLogger("instructlab.training.metrics")
    base_logger = logging.getLogger("instructlab.training")

    # Mini_trainer approach: batch_size will be determined dynamically by data loader
    # For save logic, use effective_batch_size since that's the target
    samples_seen = 0

    if hasattr(args, "samples_seen"):
        logger.info("Updating 'samples_seen' %d", args.samples_seen)
        samples_seen = args.samples_seen

    if accelerator.save_samples > 0:
        logger.info("Number of samples per save: %d", args.save_samples)

    if args.save_samples_ds is not None:
        logger.info("Number of samples per DS save: %d", args.save_samples_ds)

    global_grad_norm = None

    # Initialize the batch loss manager
    batch_loss_manager = BatchLossManager(model, accelerator, world_size, local_rank)

    # Blast through batches
    for epoch in range(args.current_epoch, args.num_epochs):
        # set the epoch for correct sampling
        accelerator.train_loader.sampler.set_epoch(epoch)
        num_epoch_steps = len(accelerator.train_loader)
        if local_rank == 0:
            inner_pb = tqdm(range(num_epoch_steps), desc=f"Epoch {epoch}")

        # blast through the batches in the train loader up to the last step within the epoch.
        for batch in accelerator.train_loader:
            if global_step <= args.last_step:
                # in the case of resuming, last_step > 0
                global_step += 1
                if local_rank == 0:
                    inner_pb.update(1)
                continue
            start = time.time()

            # Process the batch using the BatchLossManager
            batch_metrics, avg_loss_across_ranks = batch_loss_manager.process_batch(
                batch
            )

            # Update samples seen
            samples_seen += batch_metrics.total_samples

            base_logger.info(
                f"Epoch: {epoch}, Step: {global_step}, Rank: {dist.get_rank()}, loss = {avg_loss_across_ranks:.6f}, grad_accum_steps = {batch_metrics.grad_accum_steps}"
            )

            # Take optimizer step after all minibatches
            accelerator.take_optimizer_step()

            if local_rank == 0:
                elapsed_time = time.time() - start
                overall_throughput = batch_metrics.total_samples / elapsed_time
                current_lr = accelerator.lr_scheduler.get_last_lr()[0]
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
                metric_logger.info(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "rank": dist.get_rank(),
                        "overall_throughput": overall_throughput,
                        "lr": current_lr,
                        "cuda_mem_allocated": cuda_mem_allocated,
                        "cuda_malloc_retries": cuda_malloc_retries,
                        "num_loss_counted_tokens": batch_metrics.num_loss_counted_tokens,
                        "num_tokens_rank0": batch_metrics.total_length,
                        "batch_size": batch_metrics.total_samples,
                        "num_minibatches": batch_metrics.num_minibatches,
                        "avg_loss": float(avg_loss_across_ranks),
                        "samples_seen": samples_seen,
                        "gradnorm": global_grad_norm,
                        "total_samples": len(accelerator.train_loader.dataset),
                        "num_epoch_steps": num_epoch_steps,
                        # "weight_norm": weight_norm,
                    },
                    extra={"step": global_step},
                )

            if args.save_samples > 0 and (samples_seen % args.save_samples == 0):
                base_logger.debug(f"Saving checkpoint at step {global_step}")
                save_checkpoint(
                    args=args,
                    accelerator=accelerator,
                    model=model,
                    tokenizer=model.tokenizer,
                    samples_seen=samples_seen,
                    is_lora=bool(args.lora_r),
                    hf_format=True,
                )
                base_logger.debug("RANK (%d) waiting at post-save barrier.", local_rank)
                dist.barrier()

            global_step += 1
            if local_rank == 0:
                inner_pb.update(1)
            torch.cuda.empty_cache()
        if args.checkpoint_at_epoch:
            base_logger.debug(f"Saving checkpoint at epoch {epoch}")
            save_checkpoint(
                args=args,
                accelerator=accelerator,
                model=model,
                tokenizer=model.tokenizer,
                samples_seen=samples_seen,
                is_lora=bool(args.lora_r),
                full_state=args.accelerate_full_state_at_epoch,
                hf_format=True,
                epoch=epoch,
            )
            base_logger.debug("RANK (%d) waiting at post-save barrier.", local_rank)
            dist.barrier()

    if args.save_last:
        save_hf_format_accelerate(
            args,
            model,
            model.tokenizer,
            accelerator,
            samples_seen,
            is_lora=bool(args.lora_r),
        )


# This function makes an effort to stick to a default value from torch library,
# whatever it may be. That's why we don't just set to the current (as of the
# time of writing) default: to cover the unlikely event torch decides to tweak
# the default.
def _get_collective_timeout() -> datetime.timedelta | None:
    timeout_var = os.getenv("INSTRUCTLAB_NCCL_TIMEOUT_MS")
    if timeout_var is None:
        return None

    try:
        timeout = int(timeout_var)
    except ValueError:
        timeout = -1

    if timeout <= 0:
        raise ValueError(
            f"Invalid value for INSTRUCTLAB_NCCL_TIMEOUT_MS: {timeout_var}. Must be a positive integer."
        )

    return datetime.timedelta(milliseconds=timeout)


def main(args):
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

    setup_metric_logger(args.logger_type, args.run_name, args.output_dir)
    metric_logger = logging.getLogger("instructlab.training.metrics")
    if os.environ["LOCAL_RANK"] == "0":
        metric_logger.info(vars(args), extra={"hparams": True})

    setup_root_logger(args.log_level)
    tokenizer = setup_tokenizer(args.model_name_or_path, args.chat_tmpl_path)
    # device = torch.device("cuda", args.local_rank)

    model_conf = AutoConfig.from_pretrained(args.model_name_or_path)
    args.model_type = model_conf.model_type

    #### distributed init #####
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])

    timeout = _get_collective_timeout()
    if timeout is not None:
        dist.init_process_group(timeout=timeout)
    else:
        dist.init_process_group()

    args.global_rank = dist.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    dist.all_reduce(tensor)
    dist.barrier()

    flash_enabled = Model.check_flash_attn_enabled(args.disable_flash_attn)

    # This model class wraps the various AutoModel classes we support
    # based on model_type, and model_path -> choose auto_model
    lora_config = None

    if args.lora_r > 0:
        lora_config = Model.create_lora_config(
            lora_target_modules=args.lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_r=args.lora_r,
        )

    # Create model based on type
    model_class_map = {
        ModelTypes.LIGER: LigerModel,
        ModelTypes.CAUSALLM: CausalLMModel,
    }

    # Convert string to ModelTypes enum with fallback
    try:
        model_type = ModelTypes(args.model_class)
    except (ValueError, AttributeError):
        model_type = ModelTypes.CAUSALLM

    # Get the model class with default fallback
    model_class = model_class_map.get(model_type, CausalLMModel)
    m: Model = model_class(
        model_path=args.model_name_or_path,
        output_dir=args.output_dir,
        lora_config=lora_config,
        distributed_framework=DistributedBackend(args.distributed_training_framework),
        tokenizer=tokenizer,
        flash_enabled=flash_enabled,
        noise_alpha=args.NEFTune_alpha,
        lora_quant_bits=args.lora_quant_bits,
    )

    args.base_model_args = m.base_model_args

    # IMPORTANT: Freeze GPT-OSS router parameters BEFORE accelerator setup
    # This must happen before FSDP preparation to avoid requires_grad uniformity issues
    #
    # NOTE: in theory this can work for any MoE model with router parameters, but we handle
    #       GPT-OSS specifically
    # We don't want to use use_orig_params for GPT-OSS models
    fsdp_should_use_orig_params = False
    if m.is_gpt_oss or m.is_granitemoehybrid:
        frozen_router_params = freeze_router_params(m)
        if frozen_router_params:
            logger.info("🎯 Detected an MoE model - frozen router parameters")
            # For an MoE model, we need to use the original parameters so we can properly
            # freeze the router parameters.
            fsdp_should_use_orig_params = True

    # Mini_trainer approach: simplified setup
    # No complex calculations needed - the data loader handles everything
    packing_max_batch_len = args.max_batch_len

    # Mini_trainer approach: use effective_batch_size as the data loader batch_size
    # Let the collator handle distribution across GPUs and dynamic minibatching
    batch_size = args.effective_batch_size

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_loader = get_data_loader(
        data_path=args.data_path,
        batch_size=batch_size,
        max_tokens_per_gpu=packing_max_batch_len,
        seed=args.seed,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=8,  # I don't like this but am setting it for consistency
        flash_enabled=flash_enabled,
        pad_token_id=pad_token_id,
        pretraining_config=getattr(args, "pretraining_config", None),
    )

    if args.local_rank == 0:
        metric_logger.info(
            {
                "num_gpus": dist.get_world_size(),
                "avg_sample_len": train_loader.dataset.get_lengths().mean(),
                "effective_batch_size": args.effective_batch_size,
                "max_batch_len_per_gpu": args.max_batch_len,
                "packing_max_batch_len": packing_max_batch_len,
                # grad_accum will be determined dynamically per batch
                "num_batches": len(train_loader),
                "avg_samples_per_batch": len(train_loader.dataset) / len(train_loader),
                "total_samples": len(
                    train_loader.dataset
                ),  # emit the total number of samples
            },
            extra={"hparams": True},
        )
    # accelerator does not need optimizer to init, in fact, the optimizer needs to be initialized AFTER the Accelerator
    # Required for DeepSpeed/FSDP configuration even though mini_trainer uses dynamic approach
    # These set up the distributed framework but actual batching is handled dynamically
    samples_per_gpu = 1  # Base unit - actual samples determined by data loader
    grad_accum = 1  # Base unit - actual accumulation handled by minibatch loop

    accelerator = Accelerator(
        model=m,
        samples_per_gpu=samples_per_gpu,
        grad_accum=grad_accum,
        train_loader=train_loader,
        distributed_framework=DistributedBackend(args.distributed_training_framework),
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
        deepspeed_cpu_offload_optimizer=args.cpu_offload_optimizer,
        deepspeed_cpu_offload_optimizer_pin_memory=args.cpu_offload_optimizer_pin_memory,
        deepspeed_cpu_offload_optimizer_ratio=args.cpu_offload_optimizer_ratio,
        fsdp_cpu_offload_params=args.cpu_offload_params_fsdp,
        save_samples=args.save_samples,
        fsdp_use_orig_params=fsdp_should_use_orig_params,
    )
    # optimizer needs model that has been prepared by accelerator
    # and then accelerator needs to be prepared AGAIN once optimizer is initialized
    optimizer = setup_optimizer(
        model=m,
        cpu_offload=args.cpu_offload_optimizer,
        name=None,  # choose based on backend
        learning_rate=args.learning_rate,
        betas=(args.adamw_beta1, args.adamw_beta2),
        weight_decay=args.adamw_weight_decay,
        eps=args.adamw_eps,
    )
    accelerator.prepare_with_optimizer(
        optimizer=optimizer,
        lr_scheduler=args.lr_scheduler,
        num_epochs=args.num_epochs,
        num_warmup_steps=args.num_warmup_steps,
    )
    # TODO: make this work more seamlessly
    optimizer = accelerator.optimizer
    m = accelerator.model

    load_latest_full_state(args=args, accelerator=accelerator)

    train(
        args,
        model=m,
        accelerator=accelerator,
    )

    dist.barrier()
    dist.destroy_process_group()


# public API
def run_training(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    """
    Wrapper around the main training job that calls torchrun.
    """
    # Set up logging first before any processing
    # Enable package logging propagation before setting up loggers
    propagate_package_logs(True)
    setup_root_logger(train_args.log_level)
    setup_metric_logger("async", None, train_args.ckpt_output_dir)

    logger = logging.getLogger("instructlab.training")
    logger.info("Starting training setup...")

    check_valid_train_args(train_args)

    # switch out generic tmpl for legacy tmpl if requested
    if train_args.use_legacy_tmpl:
        train_args.chat_tmpl_path = os.path.join(
            os.path.dirname(__file__), "chat_templates/ibm_legacy_tmpl.py"
        )

    if train_args.process_data:
        if train_args.pretraining_config is not None:
            dp.process_documents_for_pretraining(
                data_path=train_args.data_path,
                data_output_path=train_args.data_output_dir,
                model_path=train_args.model_path,
                num_cpu_procs=train_args.data_process_num_cpu_procs,
                document_column_name=train_args.pretraining_config.document_column_name,
            )
        else:
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

    # build distributed training command
    command = [
        "torchrun",
        f"--nproc-per-node={torch_args.nproc_per_node}",
        f"--nnodes={torch_args.nnodes}",
        f"--node-rank={torch_args.node_rank}",
        f"--rdzv-id={torch_args.rdzv_id}",
    ]

    # validation should have already caught the mutually exclusive case earlier, but here we check
    # anyway just to be extra sure since Python is not type-safe and validation can bypassed (e.g. during testing)
    if torch_args.master_addr and torch_args.rdzv_endpoint:
        raise ValueError(
            "`torch_args.master_addr` and `torch_args.rdzv_endpoint` cannot be passed at the same time; please pass only one"
        )

    if torch_args.master_addr:
        command += [
            f"--master-addr={torch_args.master_addr}",
            "--rdzv-backend=static",
        ]
        command += (
            [f"--master-port={torch_args.master_port}"]
            if torch_args.master_port
            else []
        )
    elif torch_args.rdzv_endpoint:
        command += [f"--rdzv-endpoint={torch_args.rdzv_endpoint}"]
    else:
        command += ["--standalone"]

    command.extend(
        [
            __file__,
            f"--model_name_or_path={train_args.model_path}",
            f"--data_path={train_args.data_output_dir}/data.jsonl",
            f"--output_dir={train_args.ckpt_output_dir}",
            f"--num_epochs={train_args.num_epochs}",
            f"--effective_batch_size={train_args.effective_batch_size}",
            f"--learning_rate={train_args.learning_rate}",
            f"--num_warmup_steps={train_args.warmup_steps}",
            f"--save_samples={train_args.save_samples}",
            f"--log_level={train_args.log_level}",
            f"--max_batch_len={train_args.max_batch_len}",
            f"--seed={train_args.random_seed}",
            f"--adamw_weight_decay={train_args.adamw_weight_decay}",
            f"--adamw_beta1={train_args.adamw_betas[0]}",
            f"--adamw_beta2={train_args.adamw_betas[1]}",
            f"--adamw_eps={train_args.adamw_eps}",
        ]
    )

    if train_args.pretraining_config is not None:
        command.append(f"--block-size={train_args.pretraining_config.block_size}")
        command.append(
            f"--document-column-name={train_args.pretraining_config.document_column_name}"
        )

    if train_args.chat_tmpl_path is not None:
        command.append(f"--chat-tmpl-path={train_args.chat_tmpl_path}")

    if train_args.use_liger:
        command.append("--use_liger")

    if train_args.keep_last_checkpoint_only:
        command.append("--keep_last_checkpoint_only")

    if train_args.checkpoint_at_epoch:
        command.append("--checkpoint_at_epoch")

    if train_args.accelerate_full_state_at_epoch:
        command.append("--accelerate_full_state_at_epoch")

    if train_args.mock_data:
        command.append("--mock_data")
        if train_args.mock_data_len:
            command.append(f"--mock_len={train_args.mock_data_len}")

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

    logger.info("Running training command as subprocess: %s", " ".join(command))
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
        logger.info("Training subprocess interrupted by user.")
        interrupt = e
    except Exception as e:
        logger.error(
            "Unexpected exception received during distributed training", exc_info=e
        )
        interrupt = e
    finally:
        if "process" not in locals() or process is None:
            return

        # wait for the process to exit so we can properly read the exit code
        process.wait(timeout=60)
        process_code = process.poll()
        failure = process_code != 0

        if not failure:
            logger.info("Operation completed successfully! 🎉")
        else:
            logger.error(
                f"Training subprocess has not exited yet. Sending SIGTERM. Process code: {process_code}"
            )

        process.terminate()
        try:
            logger.info("Waiting for process to exit, 60s...")
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.error(
                "Training subprocess did not terminate before timeout, sending SIGKILL."
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
    parser.add_argument(
        "--model-class",
        type=str,
        default=ModelTypes.CAUSALLM.value,
        help=f"valid model classes are {[x.value for x in ModelTypes]}.",
    )
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
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--logger_type", type=str, default="async")
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
        default="HYBRID_SHARD",
        help="Sharding strategy to be used for FSDP distributed training.",
    )
    parser.add_argument(
        "--use_dolomite",
        action="store_true",
        help="(Deprecated, NoOp) Attempts to use GPTDolomite architecture",
    )
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
        "--block-size",
        type=int,
        default=None,
        help="When provided, enables pretraining mode with the given token block size.",
    )
    parser.add_argument(
        "--document-column-name",
        type=str,
        default=None,
        help="Column name containing raw documents for continual pretraining data.",
    )
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

    parser.add_argument(
        "--use_liger",
        action="store_true",
        help="Use Liger kernels for training.",
    )
    parser.add_argument(
        "--adamw_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay coefficient for AdamW optimizer.",
    )
    parser.add_argument(
        "--adamw_beta1",
        type=float,
        default=0.9,
        help="Beta1 coefficient for AdamW optimizer.",
    )
    parser.add_argument(
        "--adamw_beta2",
        type=float,
        default=0.95,
        help="Beta2 coefficient for AdamW optimizer.",
    )
    parser.add_argument(
        "--adamw_eps",
        type=float,
        default=1e-8,
        help="Epsilon for numerical stability in AdamW optimizer.",
    )
    args = parser.parse_args()
    if args.document_column_name is not None and args.block_size is None:
        parser.error("--document-column-name requires --block-size to be specified.")

    if args.block_size is not None:
        pretraining_kwargs = {}
        if args.document_column_name is not None:
            pretraining_kwargs["document_column_name"] = args.document_column_name
        args.pretraining_config = PretrainingConfig(
            block_size=args.block_size, **pretraining_kwargs
        )
    else:
        args.pretraining_config = None
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
--max_batch_len 70000 \
--seed=42
"""
