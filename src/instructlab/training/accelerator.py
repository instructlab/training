# Standard
from copy import deepcopy
from typing import Callable, Optional

# Third Party
from accelerate import Accelerator as TransformersAccel
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, MixedPrecision

# First Party
from instructlab.training.config import (  # Adjust this import if needed
    DeepSpeedOptions,
    DistributedBackend,
)

# Local
from .model import Model


class Accelerator:
    def __init__(
        self,
        model: Model,
        samples_per_gpu: int,
        grad_accum: int,
        train_loader: DataLoader,
        save_samples: int,
        distributed_framework: DistributedBackend,  # dist framework is assoc with Accelerator primarily.
        fsdp_sharding_strategy: Optional[str] = None,
        deepspeed_cpu_offload_optimizer: Optional[bool] = False,
        deepspeed_cpu_offload_optimizer_pin_memory: Optional[bool] = False,
        deepspeed_cpu_offload_optimizer_ratio: Optional[float] = None,
        fsdp_cpu_offload_params: Optional[bool] = False,
    ):
        self.samples_per_gpu = samples_per_gpu
        self.save_samples = save_samples
        self.grad_accum = grad_accum
        self.model = model
        self.distributed_framework = distributed_framework
        self.fsdp_sharding_strategy = fsdp_sharding_strategy
        self.deepspeed_cpu_offload_optimizer = deepspeed_cpu_offload_optimizer
        self.deepspeed_cpu_offload_optimizer_pin_memory = (
            deepspeed_cpu_offload_optimizer_pin_memory
        )
        self.train_loader = train_loader
        self.deepspeed_cpu_offload_optimizer_ratio = (
            deepspeed_cpu_offload_optimizer_ratio
        )
        self.fsdp_cpu_offload_params = fsdp_cpu_offload_params

        if self.distributed_framework == DistributedBackend.DEEPSPEED:
            # Standard
            accel_args = {
                "deepspeed_plugin": self.get_ds_plugin(
                    world_size=torch.distributed.get_world_size(),
                    samples_per_gpu=samples_per_gpu,
                    grad_accum=grad_accum,
                    opts=DeepSpeedOptions(
                        cpu_offload_optimizer=deepspeed_cpu_offload_optimizer,
                        cpu_offload_optimizer_ratio=self.deepspeed_cpu_offload_optimizer_ratio,
                        cpu_offload_optimizer_pin_memory=self.deepspeed_cpu_offload_optimizer_pin_memory,
                        save_samples=save_samples,
                    ),
                ),
            }
        elif self.distributed_framework in [DistributedBackend.FSDP, DistributedBackend.FSDP2]:
            accel_args = {
                "fsdp_plugin": self.get_fsdp_config(),
                "mixed_precision": "bf16",
            }
        self.accelerator = TransformersAccel(
            **accel_args,
        )
        self.accelerator.even_batches = False

        # For FSDP2, we need to defer model preparation until we have the optimizer
        # For FSDP1 and DeepSpeed, we can prepare the model immediately
        if self.distributed_framework != DistributedBackend.FSDP2:
            new_m = self.accelerator.prepare(model.model)
            self.model.update_model(new_m)

    def prepare_with_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: str,
        num_epochs: int,
        num_warmup_steps: int,
    ):
        self.lr_scheduler: Callable
        self.setup_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_epochs=num_epochs,
            num_warmup_steps=num_warmup_steps,
        )
        
        # For FSDP2, we prepare everything together (model wasn't prepared in __init__)
        # For others, the model was already prepared, so we prepare it again with optimizer
        new_m, new_opt, _, self.lr_scheduler = self.accelerator.prepare(
            self.model.model,
            optimizer,
            deepcopy(self.train_loader),
            self.lr_scheduler,
        )
        self.lr_scheduler.split_batches = True
        self.model.update_model(new_m)
        self.optimizer = new_opt

    def setup_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: str,
        num_epochs: int,
        num_warmup_steps: int,
    ):
        self.lr_scheduler = get_scheduler(
            name=lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_epochs * len(self.train_loader) // self.grad_accum,
        )

    def __getattr__(self, name):
        # Forward anything not found to the underlying optimizer
        return getattr(self.accelerator, name)

    def get_fsdp_config(self):
        # Standard
        from functools import partial

        # Third Party
        from accelerate.utils import FullyShardedDataParallelPlugin
        from peft.utils.other import fsdp_auto_wrap_policy
        from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy, CPUOffloadPolicy
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # First Party
        from instructlab.training.utils import get_module_class_from_name

        is_lora = self.model.lora_config is not None
        block_name = self.model._no_split_modules[0]

        wrap_policy = None
        if is_lora > 0:
            wrap_policy = fsdp_auto_wrap_policy(self.model)
        else:
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    get_module_class_from_name(self.model, block_name),
                },
            )

        # TODO(osilkin): BACKWARD_POST trades memory utilization for processing time, which is important for systems utilizing LoRA
        #                We should have this be configurable in the future.
        prefetch_policy = (
            BackwardPrefetch.BACKWARD_POST if is_lora else BackwardPrefetch.BACKWARD_PRE
        )
        fsdp_version = 2 if self.distributed_framework == DistributedBackend.FSDP2 else 1
        fsdp2_kwargs = {}
        if fsdp_version == 2:
            reshard_after_forward = self.fsdp_sharding_strategy is not None and self.fsdp_sharding_strategy in [ShardingStrategy.FULL_SHARD, ShardingStrategy.HYBRID_SHARD] 
            fsdp2_kwargs = {
                # 'reshard_after_forward': reshard_after_forward,
                # 'mixed_precision_policy': MixedPrecisionPolicy(
                #     param_dtype=torch.float16,
                #     reduce_dtype=torch.bfloat16,
                #     cast_forward_inputs=False,
                # ),
                'reshard_after_forward': False,
                # 'backward_prefetch': prefetch_policy,
                # 'sharding_strategy': ShardingStrategy[self.fsdp_sharding_strategy],
            }
            # todo: add cpu offload policy here if user wants it
        else:
            fsdp2_kwargs = {
                'cpu_offload': CPUOffload(self.fsdp_cpu_offload_params),
                'auto_wrap_policy': wrap_policy,
                'limit_all_gathers': True,
                'backward_prefetch': prefetch_policy,
                'sharding_strategy': ShardingStrategy[self.fsdp_sharding_strategy],
            }
            
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=fsdp_version,
            auto_wrap_policy=wrap_policy,
            limit_all_gathers=True,
            backward_prefetch=prefetch_policy,
            sharding_strategy=ShardingStrategy[self.fsdp_sharding_strategy],
            **fsdp2_kwargs
        )

        # old version
        # fsdp_plugin = FullyShardedDataParallelPlugin(
        #     auto_wrap_policy=wrap_policy,
        #     limit_all_gathers=True,
        #     backward_prefetch=prefetch_policy,
        #     sharding_strategy=ShardingStrategy[self.fsdp_sharding_strategy],
        #     cpu_offload=CPUOffload(self.fsdp_cpu_offload_params),
        # )

        # `use_orig_params` must be disabled when using LoRA and FSDP together
        # Source: https://huggingface.co/docs/peft/en/accelerate/fsdp#the-important-parts
        if self.model.lora_config is not None:
            fsdp_plugin.use_orig_params = False

        return fsdp_plugin

    def get_ds_plugin(
        self, world_size, samples_per_gpu, grad_accum, opts: DeepSpeedOptions
    ):
        # Third Party
        from accelerate.utils import DeepSpeedPlugin

        ds_config = {
            "train_batch_size": samples_per_gpu * world_size * grad_accum,
            "gradient_accumulation_steps": grad_accum,
            "train_micro_batch_size_per_gpu": samples_per_gpu,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": 2,
                # this option is only supported with DeepSpeed ZeRO stage 3
                "offload_param": {"device": "none"},
                "offload_optimizer": {"device": "none"},
            },
            "bf16": {"enabled": True},
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }

        if opts.cpu_offload_optimizer:
            # this only works when the cpu offload optimizer is enabled
            ds_config["zero_optimization"]["offload_optimizer"] = {
                # CPU offloading is the only option available in ZeRO stage 2
                "device": "cpu",
                "pin_memory": opts.cpu_offload_optimizer_pin_memory,
                "ratio": opts.cpu_offload_optimizer_ratio,
            }
        ds_plugin = DeepSpeedPlugin(
            hf_ds_config=ds_config,
        )
        return ds_plugin

    @classmethod
    def setup_deepspeed(
        cls,
        model: Model,
        samples_per_gpu: int,
        grad_accum: int,
        train_loader: DataLoader,
        deepspeed_cpu_offload_optimizer: Optional[bool],
        deepspeed_cpu_offload_optimizer_pin_memory: Optional[bool],
        deepspeed_cpu_offload_optimizer_ratio: float,
        save_samples: int,
    ):
        return cls(
            model=model,
            grad_accum=grad_accum,
            train_loader=train_loader,
            distributed_framework=DistributedBackend.DEEPSPEED,
            samples_per_gpu=samples_per_gpu,
            deepspeed_cpu_offload_optimizer=deepspeed_cpu_offload_optimizer,
            deepspeed_cpu_offload_optimizer_pin_memory=deepspeed_cpu_offload_optimizer_pin_memory,
            deepspeed_cpu_offload_optimizer_ratio=deepspeed_cpu_offload_optimizer_ratio,
            save_samples=save_samples,
        )

    @classmethod
    def setup_fsdp(
        cls,
        model: Model,
        samples_per_gpu: int,
        grad_accum: int,
        train_loader: DataLoader,
        fsdp_sharding_strategy: Optional[str],
        fsdp_cpu_offload_params: bool,
        save_samples: int,
    ):
        return cls(
            model=model,
            grad_accum=grad_accum,
            train_loader=train_loader,
            distributed_framework=DistributedBackend.FSDP,
            samples_per_gpu=samples_per_gpu,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            fsdp_cpu_offload_params=fsdp_cpu_offload_params,
            save_samples=save_samples,
        )
