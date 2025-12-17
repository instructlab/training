# Standard
from typing import List, Optional, Tuple
import functools
import logging
import math
import os

logger = logging.getLogger("instructlab.training")

try:
    # Third Party
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except ImportError:
    DeepSpeedCPUAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        logger.warning(
            "DeepSpeed CPU Optimizer is not available. Some features may be unavailable."
        )

try:
    # Third Party
    from deepspeed.ops.adam import FusedAdam
except ImportError:
    FusedAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        logger.warning("DeepSpeed is not available. Some features may be unavailable.")

# Third Party
from peft import LoraConfig
from torch.optim import AdamW
from transformers import Mxfp4Config  # pylint: disable=no-name-in-module
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
import torch

# First Party
from instructlab.training.config import (  # Adjust this import if needed
    DistributedBackend,
    Optimizer,
)
from instructlab.training.gpt_oss_utils_correct import is_gpt_oss, is_known_model
from instructlab.training.type_definitions import ModelInputs, ModelLosses


class Model:
    def __init__(
        self,
        model_path: str,
        distributed_framework: DistributedBackend,
        noise_alpha: Optional[float],
        tokenizer: PreTrainedTokenizer,
        flash_enabled: bool = False,
        lora_config: Optional[LoraConfig] = None,
        lora_quant_bits: int = 0,
    ):
        self.lora_config = lora_config
        self.noise_alpha = noise_alpha
        self.tokenizer = tokenizer
        self.distributed_framework = distributed_framework
        quant_config = None

        # check model type & set on the mclasss
        self.is_granitemoehybrid = is_known_model(model_path, "granitemoehybrid")
        self.is_gpt_oss = is_gpt_oss(model_path)
        if self.is_gpt_oss:
            # Third Party
            quant_config = Mxfp4Config(dequantize=True)

        # TODO: Add support for 8bit quantization
        elif lora_config and lora_config.r > 0 and lora_quant_bits == 4:
            # Third Party

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,  # if not set will throw a warning about slow speeds when training
            )

        self.base_model_args = {
            "pretrained_model_name_or_path": model_path,
            "quantization_config": quant_config,
        }

        # load GPT-OSS in bfloat16 because it's a massive model, but otherwise
        # we do not specify the default dtype so mixed precision training works
        # correctly
        if self.is_gpt_oss:
            self.base_model_args["torch_dtype"] = torch.bfloat16

        # set flash attention accordingly
        if flash_enabled:
            self.base_model_args["attn_implementation"] = "flash_attention_2"
            if self.is_gpt_oss:
                self.base_model_args["attn_implementation"] = (
                    "kernels-community/vllm-flash-attn3"
                )

    def _post_model_init(self):
        """Common initialization steps that should happen after model initialization."""
        self.reconcile_tokenizer()
        if self.lora_config:
            self.model = self.prepare_peft_model()

    def train(self, mode=True):
        """Set the model in training mode.

        Args:
            mode (bool): Whether to set training mode (True) or evaluation mode (False).
        """
        return self.model.train(mode)

    @property
    def module(self):
        """Required to match PreTrainedModel interface.

        PreTrainedModel uses this property to access the underlying model when wrapped in FSDP.
        We need to provide the same interface to ensure compatibility with HuggingFace's
        training utilities and to maintain consistent behavior with PreTrainedModel.
        """
        return getattr(self.model, "module", self.model)

    def parameters(self):
        """Required to match PreTrainedModel interface.

        PreTrainedModel uses this to expose model parameters for optimization.
        We need to provide the same interface to ensure our model works with
        HuggingFace's optimizers and training loops.
        """
        return self.model.parameters()

    def update_model(self, new_model):
        """Required to handle model updates from HuggingFace's training utilities.

        When using HuggingFace's training utilities (like accelerator.prepare),
        they may replace our underlying model. This method ensures we can update
        the model while maintaining our wrapper's state and behavior.
        """
        if isinstance(new_model, Model):
            raise AttributeError("This will cause recursion")
        self.model = new_model

    def __getattr__(self, name):
        """Required to match PreTrainedModel interface.

        PreTrainedModel uses this to delegate attribute access to the underlying model.
        We need to provide the same behavior to ensure our model can be used
        interchangeably with PreTrainedModel in HuggingFace's training utilities.
        """
        if name == "model":
            return super().__getattribute__("model")
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        """Required to match PreTrainedModel interface.

        PreTrainedModel uses this to delegate forward passes to the underlying model.
        We need to provide the same behavior to ensure our model can be used
        in HuggingFace's training loops and generation utilities.
        """
        return self.model(*args, **kwargs)

    @property
    def projection_layer_names(self) -> List[str]:
        """
        Given a pretrained model, returns all of the projection layers (matching '_proj')
        """
        proj_layers = set(
            name.split(".")[-1]
            for name, _ in self.model.named_modules()
            if name.endswith("_proj")
        )
        return list(proj_layers)

    def prepare_peft_model(
        self,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        mixed_precision="bf16",
    ):
        # Third Party
        from peft import (
            LoraModel,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from trl.trainer.utils import peft_module_casting_to_bf16

        proj_layers = self.projection_layer_names
        if self.lora_config.target_modules:
            requested = set(self.lora_config.target_modules)
            available = set(proj_layers)
            missing = requested - available
            valid = requested & available

            if not valid:
                raise ValueError(
                    f"None of the requested LoRA target modules exist in the model.\n"
                    f"Requested: {self.lora_config.target_modules}\nAvailable: {proj_layers}"
                )
            if missing:
                logger.warning(
                    f"\033[33mWARNING: The following modules were not found in the model: {list(missing)}. "
                    f"Applying LoRA only to: {list(valid)}.\033[0m"
                )
            self.lora_config.target_modules = list(valid)
        else:
            logger.warning(
                "WARNING: lora_target_modules not specified. Using all projection layers."
            )
            if not proj_layers:
                raise RuntimeError("No projection layers found in the model.")
            self.lora_config.target_modules = proj_layers

        if isinstance(self.model, PeftModel):
            return self.model
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(
            self.model, "is_loaded_in_4bit", False
        ):
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )
        else:

            def make_inputs_require_grad(module, input, output):  # pylint: disable=unused-argument
                output.requires_grad_(True)

            self.model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

        if self.distributed_framework == DistributedBackend.FSDP.value:
            # FSDP doesn't like `get_peft_model` as it leads to dtype mismatches
            self.model = LoraModel(self.model, self.lora_config, "default")
        else:
            self.model = get_peft_model(self.model, self.lora_config)
        if mixed_precision == "bf16" and getattr(
            self.model, "is_loaded_in_4bit", False
        ):
            peft_module_casting_to_bf16(self.model)
        return self.model

    @staticmethod
    def create_lora_config(
        lora_target_modules: List[str],
        lora_alpha: Optional[int],
        lora_dropout: Optional[float],
        lora_r: int,
    ):
        # Local
        return LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )

    def _is_causal_lm_model(self) -> bool:
        """Check if the underlying model is a causal language model.

        Returns:
            bool: True if the model is a causal language model, False otherwise.
        """
        # Third Party
        return "ForCausalLM" in self.model.__class__.__name__

    def reconcile_tokenizer(self):
        if len(self.tokenizer) > self.model.config.vocab_size:
            logger.warning(
                f"WARNING: tokenizer has {len(self.tokenizer)} tokens but model has {self.model.config.vocab_size} vocab size"
            )
            self.model.resize_token_embeddings(
                int(8 * math.ceil(len(self.tokenizer) / 8.0))
            )  # make the vocab size multiple of 8 for sharding the embedding layer.

        # Fix any discrepancy between model and tokenizer
        if (
            self.model.config.pad_token_id is not None
            and self.tokenizer.pad_token_id is not None
            and self.model.config.pad_token_id != self.tokenizer.pad_token_id
        ):
            logger.warning(
                f"WARNING: There is a mismatch between pad token id of model ({self.model.config.pad_token_id}) and tokenizer({self.tokenizer.pad_token_id}). Fixing model pad token id to be same as tokenizer's pad token id"
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if (
            self.model.config.bos_token_id is not None
            and self.tokenizer.bos_token_id is not None
            and self.model.config.bos_token_id != self.tokenizer.bos_token_id
        ):
            logging.warning(
                f"WARNING: There is a mismatch between bos token id of model({self.model.config.bos_token_id}) and tokenizer({self.tokenizer.bos_token_id}). Fixing model bos token id to be same as tokenizer's bos token id"
            )
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
        if (
            self.model.config.eos_token_id is not None
            and self.tokenizer.eos_token_id
            and self.model.config.eos_token_id != self.tokenizer.eos_token_id
        ):
            logger.warning(
                f"WARNING: There is a mismatch between eos token id of model({self.model.config.eos_token_id}) and tokenizer({self.tokenizer.eos_token_id}). Fixing model eos token id to be same as tokenizer's eos token id"
            )
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

        if (
            self.tokenizer.pad_token_id is not None
            and self.model.config.pad_token_id is None
        ):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if (
            self.tokenizer.bos_token_id is not None
            and self.model.config.bos_token_id is None
        ):
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
        if (
            self.tokenizer.eos_token_id is not None
            and self.model.config.eos_token_id is None
        ):
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

        if not self._is_causal_lm_model():
            raise ValueError(
                f"Model must be a causal language model, got {type(self.model)}"
            )

        # Local
        from .utils import add_noisy_embeddings

        self.model = add_noisy_embeddings(self.model, noise_alpha=self.noise_alpha)

    @staticmethod
    def supports_flash_attention(device_id=0):
        """Check if a GPU supports FlashAttention."""
        major, minor = torch.cuda.get_device_capability(device_id)
        # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
        is_sm8x = major == 8 and minor >= 0
        is_sm90 = major == 9 and minor == 0
        dev_name = torch.cuda.get_device_properties(device_id).gcnArchName.split(":")[0]
        is_compat_amd = dev_name in ("gfx90a", "gfx940", "gfx941", "gfx942")
        return is_sm8x or is_sm90 or is_compat_amd

    @staticmethod
    def check_flash_attn_enabled(disable_flash_attn: bool) -> bool:
        """Check if flash attention should be enabled based on configuration.

        Args:
            disable_flash_attn: Whether flash attention is explicitly disabled

        Returns:
            bool: Whether flash attention should be enabled

        Raises:
            RuntimeError: If trying to use flash attention on unsupported hardware
        """
        if disable_flash_attn:
            return False
        if Model.supports_flash_attention():
            return True
        raise RuntimeError(
            "ERROR: Trying to use Flash Attention on unsupported hardware. Please set disable_flash_attn to True."
        )

    def compute_loss(
        self, inputs: ModelInputs, world_size: int, samples_in_batch: int
    ) -> tuple[torch.Tensor, ModelLosses]:
        """
        Computes the cross-entropy los for the given model and returns a final loss which can be backproped on, as well as
        a dataclass with the raw losses.

        Args:
            inputs (ModelInputs): dict containing the items we would need in order to make the forward computation
            world_size (int): number of total nodes
            samples_in_batch (int): samples in the entire mini-batch

        Returns:
            tuple[torch.Tensor, ModelLosses]:
                - The total loss which can be used to backprop
                - Dataclass containing the raw pre-scaled losses
        """
        # Forward pass to get logits
        output = self(
            **inputs,
            use_cache=False,
        )

        # Manual loss computation with reduction="none" following mini_trainer's exact approach

        # Manually compute cross-entropy loss with reduction="none"
        logits = output.logits
        labels = inputs["labels"]

        # Shift logits and labels for causal LM (standard approach)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute loss with reduction="none" to get per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits, shift_labels)

        # Only sum losses for non-padding tokens (labels != -100)
        valid_tokens = shift_labels != -100

        # compute losses
        aux_loss = None
        primary_loss = token_losses[valid_tokens].sum()

        # add the MoE auxiliary loss (currently we only support this for GPT-OSS)
        if (
            (self.is_gpt_oss or self.is_granitemoehybrid)
            and hasattr(output, "aux_loss")
            and output.aux_loss is not None
        ):
            # For GPT-OSS: separate main loss and aux loss
            aux_loss = output.aux_loss.float()  # Auxiliary loss stays as scalar

        # Scale main loss following mini_trainer approach: loss * world_size / batch_num_loss_counted_tokens
        scaled_main_loss = primary_loss * world_size / samples_in_batch

        # For GPT-OSS: add unscaled auxiliary loss after scaling main loss
        if aux_loss is not None:
            scaled_main_loss += aux_loss

        raw_losses = ModelLosses(main_loss=primary_loss, aux_loss=aux_loss)
        return scaled_main_loss, raw_losses


class LigerModel(Model):
    # pylint: disable=unused-argument
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        distributed_framework: DistributedBackend,
        noise_alpha: Optional[float],
        tokenizer: PreTrainedTokenizer,
        flash_enabled: bool = False,
        lora_config: Optional[LoraConfig] = None,
        lora_quant_bits: int = 0,
    ):
        super().__init__(
            model_path=model_path,
            distributed_framework=distributed_framework,
            noise_alpha=noise_alpha,
            tokenizer=tokenizer,
            flash_enabled=flash_enabled,
            lora_config=lora_config,
            lora_quant_bits=lora_quant_bits,
        )
        try:
            # Third Party
            # pylint: disable-next=W0611
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
        except ImportError as e:
            raise ValueError(
                "Liger kernels are not installed. Please install Liger kernels using the following command: pip install liger-kernel"
            ) from e
        # NOTE: (jkunstle) we disable fused_linear_cross_entropy, even though it's a default for most of the models with LK support,
        #   because reduce_sum_loss requires the logits, and fused_linear_cross_entropy explicitly skips materializing them for
        #   performance.
        self.model = AutoLigerKernelForCausalLM.from_pretrained(
            **self.base_model_args,
            cross_entropy=True,
            fused_linear_cross_entropy=False,
        )
        self.model.gradient_checkpointing_enable()
        self._post_model_init()


class CausalLMModel(Model):
    # pylint: disable=unused-argument
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        distributed_framework: DistributedBackend,
        noise_alpha: Optional[float],
        tokenizer: PreTrainedTokenizer,
        flash_enabled: bool = False,
        lora_config: Optional[LoraConfig] = None,
        lora_quant_bits: int = 0,
    ):
        super().__init__(
            model_path=model_path,
            distributed_framework=distributed_framework,
            noise_alpha=noise_alpha,
            tokenizer=tokenizer,
            flash_enabled=flash_enabled,
            lora_config=lora_config,
            lora_quant_bits=lora_quant_bits,
        )
        self.model = AutoModelForCausalLM.from_pretrained(**self.base_model_args)
        self._post_model_init()
        self.model.gradient_checkpointing_enable()


def setup_optimizer(
    model: Model,
    cpu_offload: bool,
    name: Optimizer | None,
    learning_rate: int,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Setup and return an optimizer based on the given parameters.

    Args:
        model: The model to optimize
        cpu_offload: Whether to offload optimizer to CPU (for DeepSpeed)
        name: Optional optimizer name to use
        learning_rate: Learning rate for the optimizer
        betas: Beta parameters for Adam optimizers
        weight_decay: Weight decay coefficient for AdamW
        eps: Epsilon for numerical stability

    Returns:
        A PyTorch optimizer instance
    """
    optimizer_cls = None
    if name is not None:
        if name == Optimizer.ADAMW:
            optimizer_cls = AdamW
        elif name == Optimizer.CPUAdam:
            optimizer_cls = DeepSpeedCPUAdam
        elif name == Optimizer.FusedAdam:
            optimizer_cls = FusedAdam
        else:
            raise ValueError(f"Unknown optimizer type: {name}")
    else:
        if model.distributed_framework == DistributedBackend.FSDP:
            optimizer_cls = AdamW
        elif model.distributed_framework == DistributedBackend.DEEPSPEED:
            if cpu_offload:
                optimizer_cls = DeepSpeedCPUAdam
            else:
                optimizer_cls = FusedAdam

    # Filter parameters to only include those that require gradients
    # This handles cases where some parameters (e.g., frozen router params) have requires_grad=False
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Count trainable parameters for logging
    total_params = sum(1 for _ in model.parameters())
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    if total_params != trainable_count:
        logger.info(
            f"📊 Using {trainable_count}/{total_params} trainable parameters in optimizer"
        )

    factory = functools.partial(
        optimizer_cls,
        trainable_params,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    return factory()
