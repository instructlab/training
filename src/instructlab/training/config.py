# SPDX-License-Identifier: Apache-2.0

"""
Collection of config objects used in the InstructLab training library.
"""
# Standard
import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

# Third Party
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# public API
class TorchrunArgs(BaseModel):
    """
    Representation of the arguments being used by torchrun.
    The full list of arguments can be found here:
    https://pytorch.org/docs/stable/elastic/run.html#definitions

    This model implements the precedence order: arg > env > defaults
    For each argument, it checks both legacy and new environment variables.
    """

    # Core distributed training arguments
    nproc_per_node: Optional[int] = Field(
        default=None,
        description="Number of processes per node"
    )
    nnodes: Optional[int] = Field(
        default=None,
        description="Number of nodes"
    )
    node_rank: Optional[int] = Field(
        default=None,
        description="Rank of the node"
    )

    # Master address and port (legacy distributed training)
    master_addr: Optional[str] = Field(
        default=None,
        description="Master node address"
    )
    master_port: Optional[int] = Field(
        default=None,
        description="Master node port"
    )

    # Rendezvous configuration
    rdzv_backend: Optional[str] = Field(
        default=None,
        description="Rendezvous backend"
    )
    rdzv_endpoint: Optional[str] = Field(
        default=None,
        description="Rendezvous endpoint"
    )
    rdzv_id: Optional[str] = Field(
        default=None,
        description="Rendezvous ID"
    )

    # Process management
    max_restarts: Optional[int] = Field(
        default=None,
        description="Maximum number of restarts"
    )
    monitor_interval: Optional[int] = Field(
        default=None,
        description="Monitor interval in seconds"
    )
    start_method: Optional[str] = Field(
        default=None,
        description="Process start method"
    )

    # Role and module execution
    role: Optional[str] = Field(
        default=None,
        description="Role of the node"
    )
    module: Optional[bool] = Field(
        default=None,
        description="Run as module"
    )
    no_python: Optional[bool] = Field(
        default=None,
        description="Don't use Python"
    )
    run_path: Optional[bool] = Field(
        default=None,
        description="Run path"
    )

    # Logging and output
    log_dir: Optional[str] = Field(
        default=None,
        description="Log directory"
    )
    redirect_stdout: Optional[bool] = Field(
        default=None,
        description="Redirect stdout"
    )
    redirect_stderr: Optional[bool] = Field(
        default=None,
        description="Redirect stderr"
    )
    tee: Optional[bool] = Field(
        default=None,
        description="Tee output"
    )

    @classmethod
    def _get_env_var_mappings(cls) -> Dict[str, List[List[str]]]:
        """Get environment variable mappings for each argument."""
        return {
            'nproc_per_node': [
                ['PET_NPROC_PER_NODE', 'NPROC_PER_NODE']
            ],
            'nnodes': [
                ['PET_NNODES', 'NNODES']
            ],
            'node_rank': [
                ['PET_NODE_RANK', 'NODE_RANK'],
                ['RANK']  # Legacy: RANK can be used to infer node_rank
            ],
            'master_addr': [
                ['PET_MASTER_ADDR', 'MASTER_ADDR'],
                ['MASTER_ADDRESS']  # Legacy variant
            ],
            'master_port': [
                ['PET_MASTER_PORT', 'MASTER_PORT'],
                ['MASTER_PORT_NUM']  # Legacy variant
            ],
            'rdzv_backend': [
                ['PET_RDZV_BACKEND', 'RDZV_BACKEND'],
                ['BACKEND']  # Legacy variant
            ],
            'rdzv_endpoint': [
                ['PET_RDZV_ENDPOINT', 'RDZV_ENDPOINT'],
                ['MASTER_ADDR', 'MASTER_PORT']  # Legacy: can construct from master_addr:master_port
            ],
            'rdzv_id': [
                ['PET_RDZV_ID', 'RDZV_ID'],
                ['JOB_ID', 'GROUP_NAME']  # Legacy variants
            ],
            'max_restarts': [
                ['PET_MAX_RESTARTS', 'MAX_RESTARTS'],
                ['MAX_RESTART']  # Legacy variant
            ],
            'monitor_interval': [
                ['PET_MONITOR_INTERVAL', 'MONITOR_INTERVAL'],
                ['MONITOR_INTERVAL_SEC']  # Legacy variant
            ],
            'start_method': [
                ['PET_START_METHOD', 'START_METHOD'],
                ['MP_START_METHOD']  # Legacy variant
            ],
            'role': [
                ['PET_ROLE', 'ROLE'],
                ['NODE_ROLE']  # Legacy variant
            ],
            'module': [
                ['PET_MODULE', 'MODULE'],
                ['RUN_AS_MODULE']  # Legacy variant
            ],
            'no_python': [
                ['PET_NO_PYTHON', 'NO_PYTHON'],
                ['SKIP_PYTHON']  # Legacy variant
            ],
            'run_path': [
                ['PET_RUN_PATH', 'RUN_PATH'],
                ['USE_RUN_PATH']  # Legacy variant
            ],
            'log_dir': [
                ['PET_LOG_DIR', 'LOG_DIR'],
                ['LOG_DIRECTORY']  # Legacy variant
            ],
            'redirect_stdout': [
                ['PET_REDIRECT_STDOUT', 'REDIRECT_STDOUT'],
                ['STDOUT_REDIRECT']  # Legacy variant
            ],
            'redirect_stderr': [
                ['PET_REDIRECT_STDERR', 'REDIRECT_STDERR'],
                ['STDERR_REDIRECT']  # Legacy variant
            ],
            'tee': [
                ['PET_TEE', 'TEE'],
                ['TEE_OUTPUT']  # Legacy variant
            ]
        }

    @classmethod
    def from_env_and_args(cls, **kwargs) -> 'TorchrunArgs':
        """
        Create TorchrunArgs instance with proper precedence: arg > env > defaults

        Args:
            **kwargs: Command line arguments that take precedence over environment variables

        Returns:
            TorchrunArgs instance with values resolved in correct precedence order
        """
        resolved_values = {}

        for field_name, env_var_groups in cls._get_env_var_mappings().items():
            # Start with command line argument if provided
            if field_name in kwargs and kwargs[field_name] is not None:
                resolved_values[field_name] = kwargs[field_name]
                continue

            # Check environment variables in order (new vars first, then legacy)
            value = None
            for env_var_group in env_var_groups:
                for env_var in env_var_group:
                    env_value = os.getenv(env_var)
                    if env_value is not None:
                        value = env_value
                        break
                if value is not None:
                    break

            # Handle special cases for environment variable processing
            if value is not None:
                resolved_values[field_name] = cls._process_env_value(field_name, value, env_var_groups)
            else:
                # Handle WORLD_SIZE inference for nproc_per_node and nnodes
                if field_name in ['nproc_per_node', 'nnodes']:
                    world_size = os.getenv('WORLD_SIZE')
                    if world_size:
                        try:
                            world_size_int = int(world_size)
                            if field_name == 'nproc_per_node':
                                # If we have nnodes, calculate nproc_per_node
                                nnodes = os.getenv('NNODES') or os.getenv('PET_NNODES')
                                if nnodes:
                                    resolved_values[field_name] = world_size_int // int(nnodes)
                            elif field_name == 'nnodes':
                                # If we have nproc_per_node, calculate nnodes
                                nproc = os.getenv('NPROC_PER_NODE') or os.getenv('PET_NPROC_PER_NODE')
                                if nproc:
                                    resolved_values[field_name] = world_size_int // int(nproc)
                        except (ValueError, ZeroDivisionError):
                            pass

        return cls(**resolved_values)

    @classmethod
    def _process_env_value(cls, field_name: str, value: str, env_var_groups: List[List[str]]) -> Any:
        """
        Process environment variable value with appropriate type conversion and special handling

        Args:
            field_name: Name of the field
            value: Raw environment variable value
            env_var_groups: List of environment variable groups for this field

        Returns:
            Processed value with correct type
        """
        # Handle boolean fields
        if field_name in ['module', 'no_python', 'run_path', 'redirect_stdout', 'redirect_stderr', 'tee']:
            return value.lower() in ('true', '1', 'yes', 'on')

        # Handle integer fields
        if field_name in ['nproc_per_node', 'nnodes', 'node_rank', 'master_port', 'max_restarts', 'monitor_interval']:
            try:
                return int(value)
            except ValueError:
                return None

        # Handle special case for rdzv_endpoint construction from master_addr:master_port
        if field_name == 'rdzv_endpoint' and ':' not in value:
            master_addr = os.getenv('MASTER_ADDR') or os.getenv('MASTER_ADDRESS')
            master_port = os.getenv('MASTER_PORT') or os.getenv('MASTER_PORT_NUM')
            if master_addr and master_port:
                return f"{master_addr}:{master_port}"
            elif master_addr:
                return f"{master_addr}:29500"  # Default port

        # Handle special case for WORLD_SIZE inference
        if field_name in ['nproc_per_node', 'nnodes'] and 'WORLD_SIZE' in str(env_var_groups):
            world_size = os.getenv('WORLD_SIZE')
            if world_size:
                try:
                    world_size_int = int(world_size)
                    if field_name == 'nproc_per_node':
                        # If we have nnodes, calculate nproc_per_node
                        nnodes = os.getenv('NNODES') or os.getenv('TORCHRUN_NNODES')
                        if nnodes:
                            return world_size_int // int(nnodes)
                    elif field_name == 'nnodes':
                        # If we have nproc_per_node, calculate nnodes
                        nproc = os.getenv('NPROC_PER_NODE') or os.getenv('TORCHRUN_NPROC_PER_NODE')
                        if nproc:
                            return world_size_int // int(nproc)
                except (ValueError, ZeroDivisionError):
                    pass

        # Return string value as-is for other fields
        return value

    # this will tell the model construct to ignore
    # extra arguments that aren't part of this model
    class Config:
        extra = "ignore"


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
