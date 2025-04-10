# Standard
from pathlib import Path
from typing import List, Optional, Tuple
import math
import shutil
import time
import warnings

# Third Party
from config import DeepSpeedOptions, DistributedBackend, Optimizers
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from instructlab.dolomite.hf_models import (  # GPTDolomiteConfig,; import_from_huggingface,
    export_to_huggingface,
)
from peft import LoraConfig
from torch import distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, get_scheduler
import torch

# Local
from .utils import log_rank_0, wraps


class Model:
    def __init__(
        self,
        model_path,
        lora_target_modules,
        lora_alpha,
        lora_dropout,
        lora_r,
        distributed_framework: DistributedBackend,
        model_type: str,
        noise_alpha: float,
        tokenizer: PreTrainedTokenizer,
        flash_enabled: bool = False,
    ):
        self.noise_alpha = noise_alpha
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.lora_target_modules = lora_target_modules
        self.lora_alpha = (lora_alpha,)
        self.lora_dropout = (lora_dropout,)
        self.lora_r = (lora_r,)
        self.distributed_framework = distributed_framework
        self.base_model_args = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch.bfloat16,
        }

        if flash_enabled:
            self.base_model_args["attn_implementation"] = "flash_attention_2"

        # Pick model loader based on type
        if model_type == "liger":
            try:
                # Third Party
                # pylint: disable-next=W0611
                from liger_kernel.transformers import AutoLigerKernelForCausalLM
            except ImportError as e:
                raise ValueError(
                    "Liger kernels are not installed. Please install Liger kernels using the following command: pip install liger-kernel"
                ) from e
            self.model = AutoLigerKernelForCausalLM.from_pretrained(
                **self.base_model_args
            )
            self.model.gradient_checkpointing_enable()
        elif model_type == "dolomite":
            # Third Party
            from instructlab.dolomite.hf_models import GPTDolomiteForCausalLM

            # First Party
            from instructlab.training.utils import (
                apply_gradient_checkpointing,
                ensure_loadable_dolomite_checkpoint,
            )

            with ensure_loadable_dolomite_checkpoint(model_path, None) as path:
                self.base_model_args["pretrained_model_name_or_path"] = path
                self.base_model_args["use_padding_free_transformer"] = True
                self.model = GPTDolomiteForCausalLM.from_pretrained(
                    **self.base_model_args
                )
            apply_gradient_checkpointing(
                model=self.model,
                block_name=self.model._no_split_modules[0],
                use_reentrant=True,
            )
        else:
            # Third Party
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(**self.base_model_args)
            self.model.gradient_checkpointing_enable()
        if self.lora_r > 0:
            # Local
            from .utils import prepare_peft_model

            self.create_lora_config()
            self.model = prepare_peft_model(
                self.model,
                self.lora_config,
                self.distributed_framework,
                gradient_checkpointing=not (model_type == "dolomite"),
            )
            if model_type == "dolomite":

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

    def create_lora_config(self):
        # if lora
        # Third Party
        from peft import LoraConfig

        # Local
        from .utils import get_projection_layer_names

        # ensure we select only the modules that exist in the model
        proj_layers = get_projection_layer_names(self.model)
        if not self.lora_target_modules:
            print(
                f"WARNING: lora_target_modules was not specified, defaulting to all of the model's projection modules"
            )
            if not proj_layers:
                raise RuntimeError("could not find any projection layers in the model")
            self.lora_target_modules = proj_layers
        else:
            # when the user specifies the module, we should verify that they align with what's in the model
            lora_target_modules_set = set(self.lora_target_modules)
            diff = lora_target_modules_set - set(proj_layers)
            layers_to_target = lora_target_modules_set - diff
            if len(diff) == len(self.lora_target_modules):
                raise ValueError(
                    f"None of the modules you requested exist in the model.\nRequested modules: {self.lora_target_modules}; Available modules: {proj_layers}.\nThis is usually a misconfiuration error. Consider omitting your `lora_target_modules` list to have these discovered automatically."
                )
            if diff:
                print(
                    f"\033[33mWARNING: the following modules were targeted for LoRA but are not present in the model: {list(diff)}. Applying LoRA only to {list(layers_to_target)} modules.\033[0m"
                )
            self.lora_target_modules = list(layers_to_target)

        self.lora_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_target_modules,
        )

    @property
    def module(self):
        """Mimics .module behavior for DataParallel/DDP-wrapped models"""
        return getattr(self.model, "module", self.model)

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def half(self):
        self.model.half()

    def bfloat16(self):
        self.model.bfloat16()

    def float(self):
        self.model.float()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def save_pretrained(self, path, **kwargs):
        self.model.save_pretrained(path, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        # Forward anything not found to the underlying model
        return getattr(self.model, name)

    @classmethod
    def setup_liger(cls, model_path, tokenizer=None, grad_accum=1, flash_enabled=False):
        return cls(
            model_path=model_path,
            model_type="liger",
            tokenizer=tokenizer,
            grad_accum=grad_accum,
            flash_enabled=flash_enabled,
            use_liger=True,
        )

    @classmethod
    def setup_dolomite(
        cls, model_path, tokenizer=None, grad_accum=1, flash_enabled=False
    ):
        return cls(
            model_path=model_path,
            model_type="dolomite",
            tokenizer=tokenizer,
            grad_accum=grad_accum,
            flash_enabled=flash_enabled,
            use_liger=False,
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

    def update_model(self, new_model):
        """Update the internal model with a new prepared model.

        Args:
            new_model: The prepared model from accelerator.prepare()
        """
        self.model = new_model


class Optimizer:
    def __init__(
        self,
        model: Model,
        cpu_offload: bool,
        name: Optimizers | None,
        learning_rate: int,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        self.distributed_training_framework = model.distributed_training_framework
        if name is not None:
            if name == Optimizers.ADAMW.value:
                self.optimizer = AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    betas=betas,
                    weight_decay=0.0,
                )
            elif name == Optimizers.CPUAdam.value:
                self.optimizer = DeepSpeedCPUAdam(
                    model.parameters(), lr=learning_rate, betas=betas
                )
            elif name == Optimizers.FusedAdam.value:
                self.optimizer = FusedAdam(
                    model.parameters(), lr=learning_rate, betas=betas
                )
            else:
                raise ValueError(f"Unknown optimizer type: {name}")
        else:
            if model.distributed_training_framework == DistributedBackend.FSDP.value:
                self.optimizer = AdamW(
                    model.parameters(), lr=learning_rate, betas=betas
                )
            elif (
                model.distributed_training_framework
                == DistributedBackend.DEEPSPEED.value
            ):
                if cpu_offload:
                    self.optimizer = DeepSpeedCPUAdam(
                        model.parameters(), lr=learning_rate, betas=betas
                    )
                else:
                    self.optimizer = FusedAdam(
                        model.parameters(), lr=learning_rate, betas=betas
                    )

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_optimizer(self, new_optimizer):
        """Update the internal optimizer with a new prepared optimizer.

        Args:
            new_optimizer: The prepared optimizer from accelerator.prepare()
        """
        self.optimizer = new_optimizer


class Accelerator:
    def __init__(
        self,
        optimizer: Optimizer,
        model: Model,  # can we use the one from the optimizer?
        samples_per_gpu: int,
        grad_accum: int,
        num_epochs: int,
        num_warmup_steps: int,
        train_loader: DataLoader,
        distributed_framework: DistributedBackend,  # dist framework is assoc with Accelerator primarily.
        fsdp_sharding_strategy: Optional[str],
        cpu_offload_optimizer: Optional[bool],
        cpu_offload_optimizer_pin_memory: Optional[bool],
        save_samples,
        lr_scheduler: str,
    ):
        self.lr_scheduler = get_scheduler(
            name=lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_epochs * len(train_loader) // grad_accum,
        )
        self.optimizer = optimizer
        self.model = model
        self.distributed_framework = distributed_framework
        self.fsdp_sharding_strategy = fsdp_sharding_strategy
        self.cpu_offload_optimizer = cpu_offload_optimizer
        self.cpu_offload_optimizer_pin_memory = cpu_offload_optimizer_pin_memory
        self.train_loader = train_loader

        if self.distributed_framework == DistributedBackend.DEEPSPEED.value:
            # Standard
            from copy import deepcopy

            accel_args = {
                "deepspeed_plugin": self.get_ds_plugin(
                    world_size=torch.distributed.get_world_size(),
                    samples_per_gpu=samples_per_gpu,
                    grad_accum=grad_accum,
                    opts=DeepSpeedOptions(
                        cpu_offload_optimizer=cpu_offload_optimizer,
                        cpu_offload_optimizer_ratio=self.cpu_offload_optimizer_ratio,
                        cpu_offload_optimizer_pin_memory=self.cpu_offload_optimizer_pin_memory,
                        save_samples=save_samples,
                    ),
                ),
            }
        elif self.distributed_framework == DistributedBackend.FSDP.value:
            accel_args = {
                "fsdp_plugin": self.get_fsdp_config(),
            }
        accelerator = Accelerator(
            **accel_args,
        )
        accelerator.even_batches = False

        # ugh
        self.model, self.optimizer, _, self.lr_scheduler = accelerator.prepare(
            self.model,
            self.optimizer,
            deepcopy(train_loader),
            self.lr_scheduler,
        )
        self.lr_scheduler.split_batches = True

    def get_fsdp_config(self):
        # Standard
        from functools import partial

        # Third Party
        from accelerate.utils import FullyShardedDataParallelPlugin
        from peft.utils.other import fsdp_auto_wrap_policy
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # First Party
        from instructlab.training.utils import (
            get_module_class_from_name,
            patch_target_module,
        )

        is_lora = self.model.lora_r > 0
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
            mixed_precision_policy=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            backward_prefetch=prefetch_policy,
            sharding_strategy=ShardingStrategy[self.fsdp_sharding_strategy],
            cpu_offload=CPUOffload(self.cpu_offload_params_fsdp),
        )

        # `use_orig_params` must be disabled when using LoRA and FSDP together
        # Source: https://huggingface.co/docs/peft/en/accelerate/fsdp#the-important-parts
        if self.model.lora_r > 0:
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
        optimizer: Optimizer,
        train_loader,
        lr_scheduler,
        grad_accum,
    ):
        return cls(
            model=model,
            optimizer=optimizer,
            grad_accum=grad_accum,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            distributed_backend=DistributedBackend.DEEPSPEED.value,
        )

    @classmethod
    def setup_fsdp(
        cls, model: Model, optimizer: Optimizer, train_loader, lr_scheduler, grad_accum
    ):
        return cls(
            model=model,
            optimizer=optimizer,
            grad_accum=grad_accum,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            distributed_backend=DistributedBackend.FSDP.value,
        )


class Checkpointer:
    def __init__(self, strategy="full_state"):
        self.strategy = strategy.lower()

        # Map strategies to internal methods
        self._checkpoint_fn = {
            "full_state": self.save_hf_format_accelerate,
            "hf_format": self.save_full_state,
            "none": self._no_checkpoint,
        }.get(self.strategy, self._default_checkpoint)

    def checkpoint(self, model: Model, *args, **kwargs):
        # Calls the method chosen at init
        return self._checkpoint_fn(*args, **kwargs)

    def _no_checkpoint(self, *args, **kwargs):
        print("[None] Skipping checkpointing.")

    def save_fsdp_lora_model(
        self,
        model: Model,
        accelerator: Accelerator,
        output_dir: Path,
    ):
        """Given a LoRA model wrapped by FSDP and Accelerate, save a full copy of the original
        model with the trained LoRA adapters merged into the copy.

        This function creates a full copy of the model being trained and stores it in CPU memory.
        If encountering OOM errors on CPU, this is likely a culprit.

        Args:
            args (Namespace): Args received by the ArgumentParser.
            model (FSDP): FSDP model as prepared by `accelerate.Accelerator`
            accelerator (Accelerator): The given accelerator object.
        """
        # Third Party
        from peft import LoraModel

        if accelerator.distributed_type != DistributedBackend.FSDP:
            raise RuntimeError(
                "`save_fsdp_lora_model` was called when FSDP was not being used."
            )
        if not wraps(model, FSDP):
            raise RuntimeError(
                "`save_fsdp_lora_model` was called but provided model is not an FSDP model."
            )
        if not wraps(model, LoraModel):
            raise RuntimeError(
                "`save_fsdp_lora_model` was called but provided model is not a LoRA model."
            )

        # okay now that validation is out of the way, we are free to implement saving
        sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_config):
            state = model.state_dict()

        # When training a LoRA with FSDP and Accelerate, you cannot directly merge the adapters into
        # the model wrapped by FSDP. To get around this limitation, we get a copy of the state dict
        # create an identical model on CPU, load the state dict into the CPU model, merge the adapters
        # and save the model to disk.
        if accelerator.is_main_process:
            # remove device_map from args list so we can load the model on CPU
            old_device_map = self.model.base_model_args.pop("device_map", None)
            model_copy = AutoModelForCausalLM.from_pretrained(
                **model.base_model_args, device_map="cpu"
            )
            model_copy = LoraModel(model_copy, model.lora_config, "default")
            model_copy.load_state_dict(state)
            model_copy.merge_and_unload(progressbar=True)
            model_copy.save_pretrained(output_dir, safe_serialization=True)
            model.config.to_json_file(f"{output_dir}/config.json")
            model.tokenizer.save_pretrained(output_dir)
            del model_copy
            if old_device_map:
                # return the previous device_map so it can be used later on if needed
                model.base_model_args["device_map"] = old_device_map

        dist.barrier()

    def save_full_state(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        output_dir,
        is_lora: bool,
        epoch: int,
        samples_seen: int,
    ):
        """
        Saves model, optimizer, and lr_scheduler state.
        TODO: save model config - decided not to do this.
        TODO: save tokenizer - decided not to do this.
        TODO: handle LoRA
        TODO: handle granite
        """
        if is_lora:
            raise NotImplementedError("Can't save full state for LoRA at the moment.")

        # if args.is_granite:
        #     raise NotImplementedError("Can't save full state for Granite models yet.")

        output_dir = Path(output_dir) / "full_state" / f"epoch_{epoch}"
        log_rank_0(
            f"\033[93mSaving full model state in {output_dir}\033[0m", to_print=True
        )

        # patch FSDP state dict method so it works correctly.
        def _get_state_dict_patched(model, unwrap=False):
            return get_state_dict_unpatched(model, unwrap=unwrap)

        if optimizer.distributed_training_framework == "fsdp":
            get_state_dict_unpatched = accelerator.get_state_dict
            accelerator.get_state_dict = _get_state_dict_patched

        accelerator.save_state(
            output_dir=output_dir,
            # max_shard_size="5GB",
            # safe_serialization=True,
        )

        # save metadata file for current training status
        if accelerator.is_main_process:
            # TODO: should we set the global_step here rather than calculating global_step
            #   based on samples_seen?
            metadata = {"current_epoch": epoch, "samples_seen": samples_seen}
            torch.save(metadata, output_dir / "training_metadata.json")
            log_rank_0(
                f"\033[93mSaving training state: {metadata}\033[0m", to_print=True
            )

        log_rank_0(f"\033[93mModel state saved in: {output_dir}\033[0m", to_print=True)

        # cleanup
        if optimizer.distributed_training_framework == "fsdp":
            accelerator.get_state_dict = get_state_dict_unpatched

    def save_hf_format_accelerate(
        self,
        keep_last_checkpoint_only,
        output_dir,
        model: Model,
        accelerator: Accelerator,
        samples_seen,
        use_dolomite: bool,
        is_lora=False,
    ):
        # Standard
        from tempfile import TemporaryDirectory

        # Build the subdirectory name
        subdir = (
            "last_epoch" if keep_last_checkpoint_only else f"samples_{samples_seen}"
        )

        log_rank_0(
            f"\033[93mSaving model in huggingface format at: {subdir}\033[0m",
            to_print=True,
        )
        start = time.time()

        if model.model_type in ("gpt_megatron", "gpt_dolomite"):
            convert_dolomite = False
        else:
            convert_dolomite = True

        # Build the final output directory path
        final_output_dir = Path(output_dir) / "hf_format" / subdir

        if use_dolomite and convert_dolomite:
            tmpdir = TemporaryDirectory("w")  # pylint: disable=consider-using-with
            output_dir = Path(tmpdir.name)
        else:
            output_dir = final_output_dir

        CONFIG_NAME = "config.json"
        output_config_file = output_dir / CONFIG_NAME

        # XXX(osilkin): LoRA + FSDP requires a different saving path than the others
        #               so we set this variable and use it to avoid those paths further down.
        is_fsdp_lora = (
            is_lora and accelerator.distributed_type == DistributedBackend.FSDP
        )
        if is_fsdp_lora:
            self.save_fsdp_lora_model(
                tokenizer=model.tokenizer,
                accelerator=accelerator,
                output_dir=output_dir,
            )

        get_state_dict_unpatched = accelerator.get_state_dict

        def _get_state_dict_patched(model, unwrap=False):
            return get_state_dict_unpatched(model, unwrap=unwrap)

        accelerator.get_state_dict = _get_state_dict_patched

        if not is_fsdp_lora and accelerator.is_main_process:
            if is_lora:
                model.module.merge_adapter()
                model_state = model.module.state_dict()

            output_dir.mkdir(parents=True, exist_ok=True)
            if not model.module.config.architectures and convert_dolomite:
                arch_added = False
                if model.model_type == "llama":
                    model.module.config.architectures = ["LlamaForCausalLM"]
                    arch_added = True
                elif model.model_type == "granite":
                    model.module.config.architectures = ["GraniteForCausalLM"]
                    arch_added = True
                if arch_added:
                    warnings.warn(
                        f"Adding architectures to ckpt: {model.module.config.architectures}",
                    )
                else:
                    warnings.warn(
                        f"Converting from dolomite, but no architecture field added to config.json",
                    )
            model.module.config.to_json_file(output_config_file)
            model.tokenizer.save_pretrained(output_dir)

            if is_lora:
                self.save_dict_accelerate(
                    accelerator,
                    model_state,
                    save_directory=output_dir,
                    max_shard_size="5GB",
                    safe_serialization=True,
                )
                model.module.unmerge_adapter()

        if not is_lora:
            accelerator.save_model(
                model,
                save_directory=output_dir,
                max_shard_size="5GB",
                safe_serialization=True,
            )

        if use_dolomite and convert_dolomite and accelerator.is_main_process:
            # export doesnt like the directory to exist
            if final_output_dir.exists():
                shutil.rmtree(final_output_dir)
            export_to_huggingface(
                pretrained_model_name_or_path=tmpdir.name,
                save_path=final_output_dir,
                model_type=model.model_type,
            )
            tmpdir.cleanup()

        log_rank_0(f"\033[93mModel saved in {final_output_dir}\033[0m", to_print=True)
        log_rank_0(f"saving took {time.time() - start} seconds")
        dist.barrier()

        accelerator.get_state_dict = get_state_dict_unpatched

    def save_dict_accelerate(
        self,
        accelerator: Accelerator,
        state_to_save,
        save_directory,
        max_shard_size="5GB",
        safe_serialization=True,
    ):
        old_get_state = accelerator.get_state_dict
        accelerator.get_state_dict = self._copy_no_lora_dict

        def skip_precheck_loops():
            return []

        # The save model does a loop over modules and params in order to determine how to get state dict. Since we already have the state dict directly, we want to bypass those checks.
        state_to_save.modules = skip_precheck_loops
        state_to_save.parameters = skip_precheck_loops

        accelerator.save_model(
            state_to_save,
            save_directory=save_directory,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
        )

        accelerator.get_state_dict = old_get_state

    def _copy_no_lora_dict(self, state_dict):
        # Standard
        from collections import OrderedDict
        from copy import deepcopy

        cleaned_state_dict = OrderedDict()
        for param_tensor in state_dict:
            if not "lora" in param_tensor:
                cleaned_state_dict[
                    param_tensor.replace(".base_layer", "").replace(
                        "base_model.model.", ""
                    )
                ] = deepcopy(state_dict[param_tensor]).cpu()
        return cleaned_state_dict
