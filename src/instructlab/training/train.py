# Standard
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import logging
import os
import shutil
import time
import warnings

# Third Party
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

# First Party
from instructlab.training.async_logger import AsyncStructuredLogger

# Local
from instructlab.training.model import Accelerator, Checkpointer, Model, Optimizer


def train(
    model: Model,
    optimizer: Optimizer,
    accelerator: Accelerator,
    metric_logger: AsyncStructuredLogger,
    checkpointer: Checkpointer,
    effective_batch_size: int,
    num_epochs: int,
    last_step: int,
    checkpoint_at_epoch: bool,
    # accelerate_full_state_at_epoch: bool, # maybe a checkpointer class?
    output_dir,
    use_dolomite: bool,
    lr_scheduler: str | callable,
    save_last: bool,
):
    model.train()
    global_step = 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    batch_size = effective_batch_size // accelerator.grad_accum
    samples_seen = 0

    if save_samples > 0:
        save_samples = (save_samples // batch_size) * batch_size
        (
            print(f"\033[93mNumber of samples per save: {save_samples}\033[0m")
            if local_rank == 0
            else None
        )
    for epoch in range(num_epochs):
        train_epoch(
            epoch_number=epoch,
            local_rank=local_rank,
            global_step=global_step,
            last_step=last_step,
            world_size=world_size,
            save_samples=save_samples,
            batch_size=batch_size,
            samples_per_gpu=accelerator.samples_per_gpu,
            checkpoint=checkpoint_at_epoch,
            save_last=save_last,
            output_dir=output_dir,
            use_dolomite=use_dolomite,
            checkpointer=checkpointer,
            accelerator=accelerator,
            optimizer=optimizer,
            metric_logger=metric_logger,
            model=model,
            lr_scheduler=lr_scheduler,
        )


def train_epoch(
    epoch_number: int,
    local_rank: int,
    global_step: int,
    last_step: int,
    world_size: int,
    save_samples: int,
    batch_size: int,
    samples_per_gpu: int,
    checkpoint: bool,
    full_state: bool,
    output_dir: str,
    sampler: str,
    checkpointer: Checkpointer,
    model: Model,
    optimizer: Optimizer,
    accelerator: Accelerator,
    lr_scheduler: str | callable,
    metric_logger: AsyncStructuredLogger,
    use_dolomite: bool,
    save_last: bool,
):
    global_grad_norm = None
    if sampler in ("multipack"):
        accelerator.train_loader.batch_sampler.set_epoch(epoch_number)
    elif sampler in ("distributed"):
        accelerator.train_loader.sampler.set_epoch(epoch_number)
    else:
        raise Exception
    if local_rank == 0:
        inner_pb = tqdm(
            range(len(accelerator.train_loader)), desc=f"Epoch {epoch_number}"
        )

    # blast through the batches in the train loader up to the last step within the epoch.
    for batch in accelerator.train_loader:
        if global_step <= last_step:
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
        if not use_dolomite:
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
        loss = (
            loss / num_loss_counted_tokens * world_size
        )  # dividing by the total number of non-padding tokens and multiplying by the number of GPUs so when accelerate averages by world_size, it will be the correct loss.
        accelerator.backward(loss)

        if global_step % accelerator.grad_accum == 0:
            global_grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if local_rank == 0:
            elapsed_time = time.time() - start
            overall_throughput = samples_per_gpu * world_size / elapsed_time
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
                    "epoch": epoch_number,
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
                    "total_samples": len(accelerator.train_loader.dataset),
                    # "weight_norm": weight_norm,
                }
            )

        if save_samples > 0 and (global_step * batch_size % save_samples == 0):
            checkpointer.checkpoint(
                samples_seen=samples_seen,
                is_lora=False,
                hf_format=True,
            )

        global_step += 1
        if local_rank == 0:
            inner_pb.update(1)
        torch.cuda.empty_cache()
    if save_last:
        checkpointer.checkpoint(
            samples_seen=samples_seen,
            is_lora=False,
            full_state=full_state,
            hf_format=True,
            epoch=epoch_number,
            output_dir=output_dir,
        )
