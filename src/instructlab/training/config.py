# SPDX-License-Identifier: Apache-2.0

"""
Collection of config objects used in the InstructLab training library.
"""

# Standard
from enum import Enum
from typing import List, Literal, Optional, Tuple

# Third Party
from pydantic import BaseModel, ConfigDict, Field, model_validator


# public API
class DeepSpeedOffloadStrategy(Enum):
    """
    Defines the offload strategy for DeepSpeed.

    To learn more, read about it here: https://www.deepspeed.ai/tutorials/zero-offload/
    """

    CPU = "cpu"
    # TODO: update this when we support ZeRO stage-3
    # https://github.com/instructlab/training/issues/26
    # NVME = "nvme"
    NONE = None


# public API
class DistributedBackend(Enum):
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"


# public API
class QuantizeDataType(Enum):
    """
    Defines what datatype we use during quantization.
    """

    NF4 = "nf4"
    # FP8 = "fp8" TODO: test and evaluate fp8
    NONE = None


# public API
class DataProcessArgs(BaseModel):
    """
    All the arguments consumed by the training data pre-process script.
    """

    data_path: str
    data_output_path: str
    max_seq_len: int  # defines the max sequence length of a sample
    model_path: str  # either a HF model name or path to HF model
    chat_tmpl_path: str | None = Field(
        default=None,
        description="this is the path to the chat template file in the instructlab/training library format",
    )
    num_cpu_procs: int = Field(
        default=16,
        description="this is the number of CPU procs we use for data processing parallelization",
    )

    # disable the protected namespace for the model_config field
    model_config = ConfigDict(protected_namespaces=())


class PretrainingConfig(BaseModel):
    """
    Configuration for pretraining mode.
    """

    block_size: int = Field(
        description="Size of each block in tokens for pretraining datasets."
    )
    document_column_name: str = Field(
        default="document",
        description="Name of the column containing raw documents for pretraining.",
    )


# public API
class TorchrunArgs(BaseModel):
    """
    Arguments for torchrun (https://pytorch.org/docs/stable/elastic/run.html#definitions)

    Precedence order: arg > env > defaults
    Ensures that either `rdzv_endpoint` OR both `master_addr` and `master_port`
    are provided, but not both.
    """

    # Core distributed training arguments
    nproc_per_node: Literal["gpu"] | int
    nnodes: int
    node_rank: int
    rdzv_id: str | int

    # Rendezvous / master configuration
    rdzv_endpoint: Optional[str] = None
    master_addr: Optional[str] = None
    master_port: Optional[int] = None

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def validate_endpoint_config(self):
        if self.rdzv_endpoint and self.master_addr:
            raise ValueError(
                "Provide either `rdzv_endpoint` OR both `master_addr` and `master_port`, not both."
            )
        return self


# public API
class LoraOptions(BaseModel):
    """
    Options to specify when training using a LoRA.
    """

    rank: int = 4
    alpha: int = 32
    dropout: float = 0.1

    """
    Selects the linear layers that we target for LoRA training.
    When set to `None`, it selects all projection layers in the model (matching `_proj`).
    `None` by default.
    """
    target_modules: Optional[List[str]] = None

    quantize_data_type: QuantizeDataType = QuantizeDataType.NONE

    class Config:
        use_enum_values = True


# public API
class DeepSpeedOptions(BaseModel):
    """
    Represents the available options we support when training with the DeepSpeed optimizer.
    For more information, please read:
    https://www.deepspeed.ai/docs/config-json/

    Defaults are all taken from the above docs.
    """

    cpu_offload_optimizer: Optional[bool] = False
    cpu_offload_optimizer_ratio: float = 1
    cpu_offload_optimizer_pin_memory: Optional[bool] = False

    # don't save in deepspeed format as a default
    save_samples: int | None = None


# public API
class ShardingStrategies(Enum):
    FULL_SHARD = "FULL_SHARD"
    SHARD_GRAD_OP = "SHARD_GRAD_OP"
    NO_SHARD = "NO_SHARD"
    HYBRID_SHARD = "HYBRID_SHARD"


# public API
class FSDPOptions(BaseModel):
    """
    Represents the options for configuring FSDP which are exposed by the Training Library
    """

    cpu_offload_params: Optional[bool] = False
    sharding_strategy: ShardingStrategies = ShardingStrategies.HYBRID_SHARD


class Optimizer(Enum):
    ADAMW = "Adamw"
    CPUAdam = "CPUAdam"
    FusedAdam = "FusedAdam"


# public API
class ModelTypes(Enum):
    LIGER = "Liger"
    CAUSALLM = "CausalLM"


# public API
class TrainingArgs(BaseModel):
    """
    This class represents the arguments being used by the training script.
    """

    # disable the protected namespace for the model_config field
    model_config = ConfigDict(protected_namespaces=())

    # Either the name of a HuggingFace model or a path to a model saved in HuggingFace format.
    model_path: str

    # Specify the chat template / special tokens for training (default is None)
    chat_tmpl_path: str | None = Field(
        default=None,
        description="this is the path to the chat template file in the instructlab/training library format",
    )

    # this field determines if ibm_legacy_tmpl should be used instead
    use_legacy_tmpl: bool = False

    # this field specifies the filepath to the training dataset before processing
    data_path: str
    ckpt_output_dir: str

    # this field defines where we should be saving the processed version of the training dataset
    # after we have tokenized it
    data_output_dir: str

    max_seq_len: int
    max_batch_len: int
    num_epochs: int = Field(
        default=1, description="Number of epochs to run through before stopping."
    )
    effective_batch_size: int
    save_samples: int = Field(
        default=0,
        description="Number of samples the model should see before saving a checkpoint. Consider this to be the checkpoint save frequency. If --save_samples<=0, this feature is disabled.",
    )
    learning_rate: float
    adamw_weight_decay: float = Field(
        default=0.0,
        description="Weight decay coefficient for AdamW optimizer.",
    )
    adamw_betas: Tuple[float, float] = Field(
        default=(0.9, 0.95),
        description="Beta coefficients (beta1, beta2) for AdamW optimizer.",
    )
    adamw_eps: float = Field(
        default=1e-8,
        description="Epsilon for numerical stability in AdamW optimizer.",
    )
    warmup_steps: int = Field(
        default=0,
        description="Number of warmup steps to run before starting the main training loop.",
    )
    random_seed: int = 42

    # (jkunstle) left here for compatibility, but Dolomite is removed.
    use_dolomite: bool = False
    is_padding_free: bool = False  # TODO: deprecate
    checkpoint_at_epoch: bool = True
    accelerate_full_state_at_epoch: bool = True

    mock_data: Optional[bool] = False
    mock_data_len: int = 0

    deepspeed_options: DeepSpeedOptions = Field(
        default_factory=lambda: DeepSpeedOptions(
            cpu_offload_optimizer=False,
            cpu_offload_optimizer_ratio=1,
            cpu_offload_optimizer_pin_memory=False,
        )
    )
    fsdp_options: FSDPOptions = Field(
        default_factory=lambda: FSDPOptions(
            cpu_offload_params=False,
        )
    )
    distributed_backend: DistributedBackend = DistributedBackend.FSDP

    disable_flash_attn: Optional[bool] = False

    # TODO(osilkin): support quantized full fine-tuning:
    # https://github.com/instructlab/training/issues/28
    # quantize_dtype: QuantizeDataType = QuantizeDataType.NONE
    lora: LoraOptions | None = None

    # This field defines whether or not data processing will occur inside of `run_training()`
    process_data: Optional[bool] = True

    # This field specifies whether only the last checkpoint should be retained. When set to true, it
    # will overwrite the previous checkpoint directory, keeping only one directory called
    # "last_epoch". This works alongside the '--checkpoint_at_epoch' flag.
    keep_last_checkpoint_only: Optional[bool] = False

    pretraining_config: Optional[PretrainingConfig] = Field(
        default=None,
        description=(
            "Pretraining configuration. When provided, enables block-based sampling "
            "for raw document pretraining datasets."
        ),
    )

    # TODO(osilkin):
    #   we are only exposing this here because `run_training` today is implicitly coupled
    #   with `process_data`. Since we don't have a specific field for data processing arguments,
    #   we are forced to expose this. We should uncouple training from data processing and remove this.
    data_process_num_cpu_procs: int = Field(
        default=16,
        description="This is the number of processes used for multiprocessing when processing the data",
    )

    use_liger: bool = Field(
        default=False,
        description="Whether to use Liger kernels for training.",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
