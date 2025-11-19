# SPDX-License-Identifier: Apache-2.0
"""
Batch loss management for distributed training.

This module provides utilities for managing loss computation, accumulation,
and reduction across distributed training environments.
"""

# Standard
from dataclasses import dataclass
import logging

# Third Party
import torch
import torch.distributed

# First Party
from instructlab.training.accelerator import Accelerator
from instructlab.training.model import Model
from instructlab.training.type_definitions import CollatedItem, ModelInputs

logger = logging.getLogger("instructlab.training")


@dataclass
class BatchMetrics:
    """Metrics collected during batch processing."""

    total_samples: int
    total_length: int
    num_loss_counted_tokens: int
    accumulated_loss: torch.Tensor
    accumulated_aux_loss: torch.Tensor | None
    grad_accum_steps: int
    num_minibatches: int


class BatchLossManager:
    """
    Manages loss computation and metrics collection for batches in distributed training.

    This class handles:
    - Processing minibatches within a batch
    - Accumulating losses across minibatches
    - Reducing metrics across distributed ranks
    - Computing average losses for logging
    """

    def __init__(self, model, accelerator, world_size: int, local_rank: int):
        """
        Initialize the BatchLossManager.

        Args:
            model: The model used for training
            accelerator: The accelerator instance for distributed training
            world_size: Number of distributed processes
            local_rank: Local rank of the current process
        """
        self.model: Model = model
        self.accelerator: Accelerator = accelerator
        self.world_size: int = world_size
        self.local_rank: int = local_rank
        self.torch_device = torch.device("cuda", local_rank)

    def process_batch(self, batch: list[CollatedItem]) -> tuple[BatchMetrics, float]:
        """
        Process a batch of minibatches, computing losses and accumulating gradients.

        Args:
            batch: List of minibatches to process

        Returns:
            tuple: (BatchMetrics, average_loss_across_ranks)
        """
        # extract batch-level info (same across all minibatches)
        batch_num_loss_counted_tokens = batch[0]["batch_num_loss_counted_tokens"]
        num_minibatches = len(batch)

        # initialize accumulation variables
        batch_total_samples = 0
        batch_total_length = 0
        accumulated_loss = 0.0
        accumulated_aux_loss = 0.0
        grad_accum_steps = 0

        # process each minibatch
        for mb in batch:
            # extract minibatch-specific info
            micro_batch_size = mb["num_samples"]
            total_length = mb["total_length"]

            # accumulate minibatch metrics
            batch_total_samples += micro_batch_size
            batch_total_length += total_length

            # prepare model inputs
            model_inputs = self._prepare_model_inputs(mb)

            # compute loss and backward pass
            scaled_loss, raw_losses = self.model.compute_loss(
                model_inputs, self.world_size, batch_num_loss_counted_tokens
            )
            self.accelerator.backward(scaled_loss)

            # accumulate losses
            grad_accum_steps += 1
            accumulated_loss += raw_losses.main_loss
            if raw_losses.aux_loss is not None:
                accumulated_aux_loss += raw_losses.aux_loss

        # reduce metrics across ranks
        batch_total_samples, batch_total_length = self._reduce_metrics(
            batch_total_samples, batch_total_length
        )

        # calculate average loss across all ranks
        avg_loss_across_ranks = self._compute_average_loss(
            accumulated_loss, accumulated_aux_loss, batch_num_loss_counted_tokens
        )

        # create metrics object
        metrics = BatchMetrics(
            total_samples=int(batch_total_samples),
            total_length=int(batch_total_length),
            num_loss_counted_tokens=int(batch_num_loss_counted_tokens),
            accumulated_loss=accumulated_loss,
            accumulated_aux_loss=accumulated_aux_loss,
            grad_accum_steps=grad_accum_steps,
            num_minibatches=num_minibatches,
        )

        return metrics, avg_loss_across_ranks

    def _prepare_model_inputs(self, mb: CollatedItem) -> ModelInputs:
        """Prepare and move model inputs to GPU."""
        model_inputs = ModelInputs(
            input_ids=mb["input_ids"].to(device=self.torch_device),
            labels=mb["labels"].to(device=self.torch_device),
        )

        # add optional fields onto `model_inputs` object
        if "attention_mask" in mb:
            model_inputs["attention_mask"] = mb["attention_mask"].to(
                device=self.torch_device
            )
        if "position_ids" in mb:
            model_inputs["position_ids"] = mb["position_ids"].to(
                device=self.torch_device
            )

        return model_inputs

    def _reduce_metrics(
        self, batch_total_samples: int, batch_total_length: int
    ) -> tuple[int, int]:
        """Reduce rank-specific metrics across devices."""
        inputs_to_reduce = torch.tensor(
            [batch_total_samples, batch_total_length],
            dtype=torch.int32,
            device=self.accelerator.device,
        )

        reduced_outputs = self.accelerator.reduce(inputs_to_reduce, reduction="sum")
        return reduced_outputs[0].item(), reduced_outputs[1].item()

    def _compute_average_loss(
        self,
        accumulated_loss: torch.Tensor,
        accumulated_aux_loss: torch.Tensor | None,
        batch_num_loss_counted_tokens: int,
    ) -> float:
        """Compute average loss across all ranks for metrics logging."""
        # calculate total batch loss
        total_batch_loss = (
            accumulated_loss * self.world_size / batch_num_loss_counted_tokens
        )
        if accumulated_aux_loss is not None:
            total_batch_loss += accumulated_aux_loss

        # reduce across ranks
        avg_loss_across_ranks = self.accelerator.reduce(
            torch.tensor(
                total_batch_loss.detach().item(), device=self.accelerator.device
            ),
            reduction="mean",
        ).item()

        return avg_loss_across_ranks
