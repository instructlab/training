"""
Collection of config objects used in the InstructLab training library.
"""

# Standard
from enum import Enum
from typing import Optional
import os

# Third Party
from pydantic import BaseModel, ConfigDict, Field


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
    chat_tmpl_path: str

    # disable the protected namespace for the model_config field
    model_config = ConfigDict(protected_namespaces=())


# public API
class TorchrunArgs(BaseModel):
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


# public API
class LoraOptions(BaseModel):
    """
    Options to specify when training using a LoRA.
    """

    rank: int = 4
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

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
class TrainingArgs(BaseModel):
    """
    This class represents the arguments being used by the training script.
    """

    # disable the protected namespace for the model_config field
    model_config = ConfigDict(protected_namespaces=())

    # Either the name of a HuggingFace model or a path to a model saved in HuggingFace format.
    model_path: str

    # Specify the chat template / special tokens for training (default is ibm-generic template/tokens)
    chat_tmpl_path: str = os.path.join(
        os.path.dirname(__file__), "chat_templates/ibm_generic_tmpl.py"
    )

    # this field specifies the filepath to the training dataset before processing
    data_path: str
    ckpt_output_dir: str

    # this field defines where we should be saving the processed version of the training dataset
    # after we have tokenized it
    data_output_dir: str

    max_seq_len: int
    max_batch_len: int
    num_epochs: int
    effective_batch_size: int
    save_samples: int
    learning_rate: float
    warmup_steps: int
    is_padding_free: bool
    random_seed: int = 42

    mock_data: Optional[bool] = False
    mock_data_len: int = 0

    deepspeed_options: DeepSpeedOptions = Field(
        default_factory=lambda: DeepSpeedOptions(
            cpu_offload_optimizer=False,
            cpu_offload_optimizer_ratio=1,
            cpu_offload_optimizer_pin_memory=False,
        )
    )

    disable_flash_attn: Optional[bool] = False

    # TODO(osilkin): support quantized full fine-tuning:
    # https://github.com/instructlab/training/issues/28
    # quantize_dtype: QuantizeDataType = QuantizeDataType.NONE
    lora: LoraOptions | None = None
