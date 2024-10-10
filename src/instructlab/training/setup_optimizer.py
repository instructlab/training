# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch
from torch.optim import Optimizer  
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam  
from typing import Any

# First Party
from instructlab.training.config import DistributedBackend 

def setup_optimizer(args: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
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