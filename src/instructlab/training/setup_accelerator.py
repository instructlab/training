# Standard
from functools import partial

# Third Party
from accelerate import Accelerator
from peft.utils.other import fsdp_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedModel
import torch

# First Party
from instructlab.training.config import DeepSpeedOptions
from instructlab.training.utils import get_module_class_from_name, patch_target_module


def get_ds_plugin(world_size, samples_per_gpu, grad_accum, opts: DeepSpeedOptions):
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


def get_fsdp_config(args, model: PreTrainedModel):
    # Third Party
    from accelerate.utils import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

    is_lora = args.lora_r > 0
    block_name = model._no_split_modules[0]

    wrap_policy = None
    if is_lora > 0:
        wrap_policy = fsdp_auto_wrap_policy(model)
    else:
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                get_module_class_from_name(model, block_name),
            },
        )

    # TODO(osilkin): BACKWARD_POST trades memory utilization for processing time, which is important for systems utilizing LoRA
    #                We should have this be configurable in the future.
    prefetch_policy = (
        BackwardPrefetch.BACKWARD_POST if is_lora else BackwardPrefetch.BACKWARD_PRE
    )
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy=wrap_policy,
        limit_all_gathers=True,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=prefetch_policy,
        sharding_strategy=ShardingStrategy[args.fsdp_sharding_strategy],
        cpu_offload=CPUOffload(args.cpu_offload_params_fsdp),
    )

    # `use_orig_params` must be disabled when using LoRA and FSDP together
    # Source: https://huggingface.co/docs/peft/en/accelerate/fsdp#the-important-parts
    if args.lora_r > 0:
        fsdp_plugin.use_orig_params = False

    return fsdp_plugin


def setup_accelerator(args, model: PreTrainedModel, grad_accum):
    if args.distributed_training_framework == "deepspeed":
        try:
            # Third Party
            from deepspeed import DeepSpeedEngine
        except ImportError as exc:
            raise ImportError(
                "DeepSpeed selected as distributed framework, but not installed"
            ) from exc

        # patch deepspeed to work with quantized models.
        if args.lora_quant_bits is not None:
            patch_target_module(
                "deepspeed.DeepSpeedEngine",
                partial(DeepSpeedEngine, dont_change_device=True),
            )

        accel_args = {
            "deepspeed_plugin": get_ds_plugin(
                world_size=torch.distributed.get_world_size(),
                samples_per_gpu=args.samples_per_gpu,
                grad_accum=grad_accum,
                opts=DeepSpeedOptions(
                    cpu_offload_optimizer=args.cpu_offload_optimizer,
                    cpu_offload_optimizer_ratio=args.cpu_offload_optimizer_ratio,
                    cpu_offload_optimizer_pin_memory=args.cpu_offload_optimizer_pin_memory,
                    save_samples=args.save_samples_ds,
                ),
            ),
        }
    elif args.distributed_training_framework == "fsdp":
        accel_args = {
            "fsdp_plugin": get_fsdp_config(args, model),
        }
    else:
        raise ValueError(
            f"Unknown sharding framework: {args.distributed_training_framework}"
        )
    accelerator = Accelerator(
        **accel_args,
    )
    accelerator.even_batches = False
    return accelerator
