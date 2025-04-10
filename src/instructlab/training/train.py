# Standard
import os
import time

# Third Party
from pydantic import BaseModel
from tqdm import tqdm
import torch

# First Party
from instructlab.training.async_logger import AsyncStructuredLogger
from instructlab.training.model import Accelerator, Checkpointer, Model


class Metrics(BaseModel):
    samples_seen: int
    total_loss: float
    batch_size: int
    num_loss_counted_tokens: int
    global_grad_norm: float
    total_samples: int
    overall_throughput: float
    current_lr: float


def train(
    model: Model,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    metric_logger: AsyncStructuredLogger,
    checkpointer: Checkpointer,
    effective_batch_size: int,
    num_epochs: int,
    last_step: int,
    checkpoint_at_epoch: bool,
    output_dir,
    use_dolomite: bool,
    save_last: bool,
    sampler: str,
):
    model.train()
    global_step = 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    batch_size = effective_batch_size // accelerator.grad_accum
    samples_seen = 0

    if accelerator.save_samples > 0:
        accelerator.save_samples = (accelerator.save_samples // batch_size) * batch_size
        (
            print(
                f"\033[93mNumber of samples per save: {accelerator.save_samples}\033[0m"
            )
            if local_rank == 0
            else None
        )
    for epoch in range(num_epochs):
        metrics = train_epoch(
            epoch_number=epoch,
            samples_seen=samples_seen,
            local_rank=local_rank,
            global_step=global_step,
            last_step=last_step,
            world_size=world_size,
            batch_size=batch_size,
            samples_per_gpu=accelerator.samples_per_gpu,
            checkpoint_at_epoch=checkpoint_at_epoch,
            output_dir=output_dir,
            use_dolomite=use_dolomite,
            checkpointer=checkpointer,
            accelerator=accelerator,
            optimizer=optimizer,
            model=model,
            sampler=sampler,
        )
        if local_rank == 0 and isinstance(metrics, Metrics):
            # TODO - Bring back consistent gradnorm and weight_norm logging
            metric_logger.log_sync(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "rank": torch.distributed.get_rank(),
                    "overall_throughput": metrics.overall_throughput,
                    "lr": metrics.current_lr,
                    "cuda_mem_allocated": torch.cuda.memory_allocated() / (1024**3),
                    "cuda_malloc_retries": torch.cuda.memory_stats()[
                        "num_alloc_retries"
                    ],
                    "num_loss_counted_tokens": metrics.num_loss_counted_tokens,
                    "batch_size": batch_size,
                    "total_loss": metrics.total_loss,
                    "samples_seen": samples_seen,
                    "gradnorm": metrics.global_grad_norm,
                    "total_samples": len(accelerator.train_loader.dataset),
                    # "weight_norm": weight_norm,
                }
            )

    if save_last:
        checkpointer.checkpoint(
            output_dir=output_dir,
            epoch=num_epochs,
            samples_seen=samples_seen,
            last_epoch=True,
        )


def train_epoch(
    epoch_number: int,
    samples_seen: int,
    local_rank: int,
    global_step: int,
    last_step: int,
    world_size: int,
    batch_size: int,
    samples_per_gpu: int,
    checkpoint_at_epoch: bool,
    output_dir: str,
    sampler: str,
    checkpointer: Checkpointer,
    model: Model,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    use_dolomite: bool,
) -> Metrics | None:
    metrics = None
    global_grad_norm = None
    if sampler in ("multipack"):
        accelerator.train_loader.batch_sampler.set_epoch(epoch_number)
    elif sampler in ("distributed"):
        accelerator.train_loader.sampler.set_epoch(epoch_number)
    else:
        raise AttributeError(
            f"Sampler {sampler} is invalid. Valid samplers are multipack and distributed."
        )
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
            accelerator.lr_scheduler.step()
            optimizer.zero_grad()

        if local_rank == 0:
            elapsed_time = time.time() - start
            overall_throughput = samples_per_gpu * world_size / elapsed_time
            current_lr = accelerator.lr_scheduler.get_last_lr()[0]
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

            metrics = Metrics(
                samples_seen=samples_seen,
                total_loss=float(log_loss / num_loss_counted_tokens),
                batch_size=batch_size,
                num_loss_counted_tokens=int(num_loss_counted_tokens),
                global_grad_norm=global_grad_norm,
                total_samples=len(accelerator.train_loader.dataset),
                overall_throughput=overall_throughput,
                current_lr=current_lr,
            )

        if (
            accelerator.save_samples > 0
            and (global_step * batch_size % accelerator.save_samples == 0)
            and checkpoint_at_epoch
        ):
            checkpointer.checkpoint(
                output_dir=output_dir,
                epoch=epoch_number,
                samples_seen=samples_seen,
            )

        global_step += 1
        if local_rank == 0:
            inner_pb.update(1)
        torch.cuda.empty_cache()
    return metrics
