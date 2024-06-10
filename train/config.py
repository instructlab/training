from dataclasses import dataclass
import yaml
from enum import Enum

@dataclass
class TorchrunTrainArgs:
    """
    Representation of the arguments being used by torchrun.
    The full list of arguments can be found here:
    https://pytorch.org/docs/stable/elastic/run.html#definitions
    """
    nproc_per_node: int
    nnodes: int
    node_rank: int
    rdzv_id: int
    rdzv_endpoint: str

    def __str__(self):
        return yaml.dump(vars(self), sort_keys=False)


@dataclass
class FullTrainArgs:
    """
    This class represents the arguments being used by the training script.
    """
    model_path: str
    data_path: str
    ckpt_output_path: str

    num_gpus: int
    max_seq_len: int
    max_batch_len: int
    num_epochs: int
    effective_batch_size: int
    save_samples: int
    learning_rate: float
    warmup_steps: int

    ds_offload_strat: Enum["cpu", "nvme", None]
    cpu_offload_optimizer: bool
    cpu_offload_params: bool

    quantize_dtype: Enum["nf4", "fp8", None] #fp8 requires transformer engine or microsoft emp (not robust libraries though).
    lora: bool
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: list


    def __str__(self):
        return yaml.dump(vars(self), sort_keys=False)

