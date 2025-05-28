# Standard
from copy import deepcopy
from typing import List, Optional, Tuple
import math
import os

# Third Party
from accelerate import Accelerator as TransformersAccel

try:
    # Third Party
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except ImportError:
    DeepSpeedCPUAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        print(
            "DeepSpeed CPU Optimizer is not available. Some features may be unavailable."
        )

try:
    # Third Party
    from deepspeed.ops.adam import FusedAdam
except ImportError:
    FusedAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        print("DeepSpeed is not available. Some features may be unavailable.")

# Third Party
from peft import LoraConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, get_scheduler
import torch

# First Party
from instructlab.training.config import (  # Adjust this import if needed
    DeepSpeedOptions,
    DistributedBackend,
    ModelTypes,
    Optimizers,
)

# mypy: disable_error_code="has-type"


class Model:
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        distributed_framework: DistributedBackend,
        model_type: ModelTypes,
        noise_alpha: Optional[float],
        tokenizer: PreTrainedTokenizer,
        flash_enabled: bool = False,
        lora_config: Optional[LoraConfig] = None,
    ):
        self.lora_config = lora_config
        self.noise_alpha = noise_alpha
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.distributed_framework = distributed_framework
        self.base_model_args = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch.bfloat16,
        }

        if flash_enabled:
            self.base_model_args["attn_implementation"] = "flash_attention_2"

        # Pick model loader based on type
        if model_type == ModelTypes.LIGER:
            try:
                # Third Party
                # pylint: disable-next=W0611
                from liger_kernel.transformers import AutoLigerKernelForCausalLM
            except ImportError as e:
                raise ValueError(
                    "Liger kernels are not installed. Please install Liger kernels using the following command: pip install liger-kernel"
                ) from e
            self.model = AutoLigerKernelForCausalLM.from_pretrained(
                **self.base_model_args,
                cross_entropy=True,
                fused_linear_cross_entropy=False,
            )
            self.model.gradient_checkpointing_enable()
        elif model_type == ModelTypes.DOLOMITE:
            # Third Party
            from instructlab.dolomite.hf_models import GPTDolomiteForCausalLM

            # First Party
            from instructlab.training.utils import (
                apply_gradient_checkpointing,
                ensure_loadable_dolomite_checkpoint,
            )

            with ensure_loadable_dolomite_checkpoint(model_path, output_dir) as path:
                self.base_model_args["pretrainedmodel_name_or_path"] = path
                self.base_model_args["use_padding_free_transformer"] = True
                self.model = GPTDolomiteForCausalLM.from_pretrained(
                    **self.base_model_args
                )
            apply_gradient_checkpointing(
                model=self.model,
                block_name=self.model._no_split_modules[0],
                use_reentrant=True,
            )
        elif model_type == ModelTypes.CAUSALLM:
            # Third Party
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(**self.base_model_args)
            self.model.gradient_checkpointing_enable()
        else:
            raise AttributeError(
                f"Invalid Model Type {model_type} valid types are {ModelTypes.LIGER.value}, {ModelTypes.DOLOMITE.value}, and {ModelTypes.CAUSALLM.value}."
            )

        self.reconcile_tokenizer()
        if self.lora_config:
            # First Party
            self.model = self.prepare_peft_model(
                gradient_checkpointing=not (model_type == "dolomite"),
            )
            if model_type == "dolomite":
                # pylint: disable=unused-argument
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

    def train(self, mode=True):
        """Set the model in training mode.

        Args:
            mode (bool): Whether to set training mode (True) or evaluation mode (False).
        """
        return self.model.train(mode)

    @property
    def module(self):
        return getattr(self.model, "module", self.model)

    def parameters(self):
        return self.model.parameters()

    def update_model(self, new_model):
        if isinstance(new_model, Model):
            raise AttributeError("This will cause recursion")
        self.model = new_model

    def __getattr__(self, name):
        if name == "model":
            return super().__getattribute__("model")
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_projection_layer_names(self) -> List[str]:
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
        gradient_checkpointing=True,
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

        proj_layers = self.get_projection_layer_names()
        if not self.lora_config.target_modules:
            print(
                "WARNING: lora_target_modules not specified. Using all projection layers."
            )
            if not proj_layers:
                raise RuntimeError("No projection layers found in the model.")
            self.lora_config.target_modules = proj_layers
        else:
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
                print(
                    f"\033[33mWARNING: The following modules were not found in the model: {list(missing)}. "
                    f"Applying LoRA only to: {list(valid)}.\033[0m"
                )
            self.lora_config.target_modules = list(valid)

        # if not isinstance(peft_config, PeftConfig):
        #     raise ValueError(
        #         "If you want to use the PeftModel, you need to pass a PeftConfig object, "
        #         f"and you passed a {type(peft_config)}."
        #     )

        if not isinstance(self.model, PeftModel):
            if getattr(self.model, "is_loaded_in_8bit", False) or getattr(
                self.model, "is_loaded_in_4bit", False
            ):
                preprare_model_kwargs = {
                    "use_gradient_checkpointing": gradient_checkpointing
                }

                # if _support_gc_kwargs:
                preprare_model_kwargs["gradient_checkpointing_kwargs"] = (
                    gradient_checkpointing_kwargs
                )

                self.model = prepare_model_for_kbit_training(
                    self.model, **preprare_model_kwargs
                )

            elif gradient_checkpointing:
                # For backward compatibility with older versions of transformers
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
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

    @classmethod
    def setup_liger(
        cls,
        model_path: str,
        output_dir: str,
        lora_config: LoraConfig,
        distributed_framework: DistributedBackend,
        noise_alpha: Optional[float],
        tokenizer: PreTrainedTokenizer,
        flash_enabled: bool = False,
    ):
        return cls(
            model_path=model_path,
            output_dir=output_dir,
            lora_config=lora_config,
            distributed_framework=distributed_framework,
            model_type=ModelTypes.LIGER,
            noise_alpha=noise_alpha,
            tokenizer=tokenizer,
            flash_enabled=flash_enabled,
        )

    @classmethod
    def setup_dolomite(
        cls,
        model_path: str,
        output_dir: str,
        lora_config: LoraConfig,
        distributed_framework: DistributedBackend,
        noise_alpha: Optional[float],
        tokenizer: PreTrainedTokenizer,
        flash_enabled: bool = False,
    ):
        return cls(
            model_path=model_path,
            output_dir=output_dir,
            lora_config=lora_config,
            distributed_framework=distributed_framework,
            model_type=ModelTypes.DOLOMITE,
            noise_alpha=noise_alpha,
            tokenizer=tokenizer,
            flash_enabled=flash_enabled,
        )

    def reconcile_tokenizer(self):
        if len(self.tokenizer) > self.model.config.vocab_size:
            print(
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
            print(
                f"WARNING: There is a mismatch between pad token id of model ({self.model.config.pad_token_id}) and tokenizer({self.tokenizer.pad_token_id}). Fixing model pad token id to be same as tokenizer's pad token id"
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if (
            self.model.config.bos_token_id is not None
            and self.tokenizer.bos_token_id is not None
            and self.model.config.bos_token_id != self.tokenizer.bos_token_id
        ):
            print(
                f"WARNING: There is a mismatch between bos token id of model({self.model.config.bos_token_id}) and tokenizer({self.tokenizer.bos_token_id}). Fixing model bos token id to be same as tokenizer's bos token id"
            )
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
        if (
            self.model.config.eos_token_id is not None
            and self.tokenizer.eos_token_id
            and self.model.config.eos_token_id != self.tokenizer.eos_token_id
        ):
            print(
                f"WARNING: There is a mismatch between eos token id of model({self.model.config.eos_token_id}) and tokenizer({self.tokenizer.eos_token_id}). Fixing model eos token id to be same as tokenizer's eos token id"
            )
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

        if "ForCausalLM" not in self.model.__class__.__name__:
            raise ValueError(
                f"Model class name: {self.model.__class__.__name__} is not supported."
            )

        # ensure the model has any tokens which were added to the tokenizer
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

        # Local
        from .utils import add_noisy_embeddings, convert_loss_to_reduce_sum

        self.model = convert_loss_to_reduce_sum(
            self.model, use_dolomite=(self.model_type == "dolomite")
        )
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
    def check_flash_attn_enabled(disable_flash_attn: bool, use_dolomite: bool) -> bool:
        """Check if flash attention should be enabled based on configuration.

        Args:
            disable_flash_attn: Whether flash attention is explicitly disabled
            use_dolomite: Whether dolomite padding-free transformer is being used

        Returns:
            bool: Whether flash attention should be enabled

        Raises:
            RuntimeError: If trying to use flash attention on unsupported hardware
                         or trying to use dolomite without flash attention
        """
        if not disable_flash_attn:
            if Model.supports_flash_attention():
                return True
            else:
                raise RuntimeError(
                    "ERROR: Trying to use Flash Attention on unsupported hardware. Please set disable_flash_attn to True."
                )
        elif use_dolomite:
            raise RuntimeError(
                "ERROR: Trying to use dolomite padding-free transformer without flash attention is not supported"
            )
        return False


def setup_optimizer(
    model: Model,
    cpu_offload: bool,
    name: Optimizers | None,
    learning_rate: int,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """Setup and return an optimizer based on the given parameters.

    Args:
        model: The model to optimize
        cpu_offload: Whether to offload optimizer to CPU (for DeepSpeed)
        name: Optional optimizer name to use
        learning_rate: Learning rate for the optimizer
        betas: Beta parameters for Adam optimizers

    Returns:
        A PyTorch optimizer instance
    """
    if name is not None:
        if name == Optimizers.ADAMW:
            return AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=betas,
                weight_decay=0.0,
            )
        elif name == Optimizers.CPUAdam:
            return DeepSpeedCPUAdam(model.parameters(), lr=learning_rate, betas=betas)
        elif name == Optimizers.FusedAdam:
            return FusedAdam(model.parameters(), lr=learning_rate, betas=betas)
        else:
            raise ValueError(f"Unknown optimizer type: {name}")
    else:
        if model.distributed_framework == DistributedBackend.FSDP:
            return AdamW(model.parameters(), lr=learning_rate, betas=betas)
        elif model.distributed_framework == DistributedBackend.DEEPSPEED:
            if cpu_offload:
                return DeepSpeedCPUAdam(
                    model.parameters(), lr=learning_rate, betas=betas
                )
            else:
                return FusedAdam(model.parameters(), lr=learning_rate, betas=betas)


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
        elif self.distributed_framework == DistributedBackend.FSDP:
            accel_args = {
                "fsdp_plugin": self.get_fsdp_config(),
                "mixed_precision": "bf16",
            }
        self.accelerator = TransformersAccel(
            **accel_args,
        )
        self.accelerator.even_batches = False

        new_m = self.accelerator.prepare(model.model)
        self.model.update_model(new_m)

    def prepare_with_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: str,
        num_epochs: int,
        num_warmup_steps: int,
    ):
        self.setup_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_epochs=num_epochs,
            num_warmup_steps=num_warmup_steps,
        )
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
        from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy
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
        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy=wrap_policy,
            limit_all_gathers=True,
            backward_prefetch=prefetch_policy,
            sharding_strategy=ShardingStrategy[self.fsdp_sharding_strategy],
            cpu_offload=CPUOffload(self.fsdp_cpu_offload_params),
        )

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
