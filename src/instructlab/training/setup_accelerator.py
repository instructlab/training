# Standard
from functools import partial

# Third Party
from accelerate import Accelerator
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedModel
import torch

# First Party
from instructlab.training.config import DeepSpeedOptions
from instructlab.training.utils import get_module_class_from_name, patch_target_module


def get_fsdp_config(args, model: PreTrainedModel):
    # Third Party
    from accelerate.utils import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload



    # TODO(osilkin): BACKWARD_POST trades memory utilization for processing time, which is important for systems utilizing LoRA
    #                We should have this be configurable in the future.

    block_name = model._no_split_modules[0]
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                get_module_class_from_name(model, block_name),
            },
        ),
        limit_all_gathers=True,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy[args.fsdp_sharding_strategy],
        cpu_offload=CPUOffload(args.cpu_offload_params_fsdp),
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if torch.distributed.get_rank()!=0 else None,
        cpu_ram_efficient_loading=True,
        use_orig_params=True,
        state_dict_type="sharded_state_dict",
    )

    return fsdp_plugin


def setup_accelerator(args, model: PreTrainedModel):
    accelerator = Accelerator(
        fsdp_plugin=get_fsdp_config(args, model),
    )
    accelerator.even_batches = False
    return accelerator
