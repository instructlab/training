# Standard
from copy import deepcopy
from pathlib import Path
import shutil
import time
import warnings

# Third Party
from instructlab.dolomite.hf_models import export_to_huggingface
from torch import distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
import torch

# First Party
from instructlab.training.accelerator import Accelerator
from instructlab.training.config import DistributedBackend
from instructlab.training.model import Model

# Local
from .utils import log_rank_0, wraps


class Checkpointer:
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        strategy="all",
    ):
        self.strategy = strategy.lower()
        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator

        # Map strategies to internal methods
        self._checkpoint_fn = {
            "full_state": self.save_full_state,
            "hf_format": self.save_hf_format_accelerate,
            "all": self.save_all_checkpoints,
        }.get(self.strategy, self._no_checkpoint)

    def checkpoint(self, *args, **kwargs):
        # Calls the method chosen at init
        return self._checkpoint_fn(*args, **kwargs)

    # pylint: disable=unused-argument
    def _no_checkpoint(self, *args, **kwargs):
        print("[None] Skipping checkpointing.")

    # pylint: disable=unused-argument
    def save_fsdp_lora_model(
        self,
        output_dir: Path,
        **kwargs,
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

        if self.accelerator.distributed_type != DistributedBackend.FSDP:
            raise RuntimeError(
                "`save_fsdp_lora_model` was called when FSDP was not being used."
            )
        if not wraps(self.model, FSDP):
            raise RuntimeError(
                "`save_fsdp_lora_model` was called but provided model is not an FSDP model."
            )
        if not wraps(self.model, LoraModel):
            raise RuntimeError(
                "`save_fsdp_lora_model` was called but provided model is not a LoRA model."
            )

        # okay now that validation is out of the way, we are free to implement saving
        sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, sd_config):
            state = self.model.state_dict()

        # When training a LoRA with FSDP and Accelerate, you cannot directly merge the adapters into
        # the model wrapped by FSDP. To get around this limitation, we get a copy of the state dict
        # create an identical model on CPU, load the state dict into the CPU model, merge the adapters
        # and save the model to disk.
        if self.accelerator.is_main_process:
            # Third Party
            from transformers import AutoModelForCausalLM

            # remove device_map from args list so we can load the model on CPU
            old_device_map = self.model.base_model_args.pop("device_map", None)
            model_copy = AutoModelForCausalLM.from_pretrained(
                **self.model.base_model_args, device_map="cpu"
            )
            model_copy = LoraModel(model_copy, self.model.lora_config, "default")
            model_copy.load_state_dict(state)
            model_copy.merge_and_unload(progressbar=True)
            model_copy.save_pretrained(output_dir, safe_serialization=True)
            self.model.config.to_json_file(f"{output_dir}/config.json")
            self.model.tokenizer.save_pretrained(output_dir)
            del model_copy
            if old_device_map:
                # return the previous device_map so it can be used later on if needed
                self.model.base_model_args["device_map"] = old_device_map

        dist.barrier()

    # pylint: disable=unused-argument
    def save_full_state(
        self,
        output_dir,
        epoch: int,
        samples_seen: int,
        **kwargs,
    ):
        """
        Saves model, optimizer, and lr_scheduler state.
        TODO: save model config - decided not to do this.
        TODO: save tokenizer - decided not to do this.
        TODO: handle LoRA
        TODO: handle granite
        """
        if self.model.lora_config is not None:
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

        if self.accelerator.distributed_framework == "fsdp":
            get_state_dict_unpatched = self.accelerator.get_state_dict
            self.accelerator.get_state_dict = _get_state_dict_patched

        self.accelerator.save_state(
            output_dir=output_dir,
            # max_shard_size="5GB",
            # safe_serialization=True,
        )

        # save metadata file for current training status
        if self.accelerator.is_main_process:
            # TODO: should we set the global_step here rather than calculating global_step
            #   based on samples_seen?
            metadata = {"current_epoch": epoch, "samples_seen": samples_seen}
            torch.save(metadata, output_dir / "training_metadata.json")
            log_rank_0(
                f"\033[93mSaving training state: {metadata}\033[0m", to_print=True
            )

        log_rank_0(f"\033[93mModel state saved in: {output_dir}\033[0m", to_print=True)

        # cleanup
        if self.accelerator.distributed_framework == "fsdp":
            self.accelerator.get_state_dict = get_state_dict_unpatched

    # pylint: disable=unused-argument
    def save_hf_format_accelerate(
        self,
        output_dir,
        epoch: int,
        samples_seen: int,
        last_epoch: bool = False,
        **kwargs,
    ):
        # Standard
        from tempfile import TemporaryDirectory

        # Build the subdirectory name
        subdir = "last_epoch" if last_epoch else f"samples_{samples_seen}"

        log_rank_0(
            f"\033[93mSaving model in huggingface format at: {subdir}\033[0m",
            to_print=True,
        )
        start = time.time()

        if self.model.model_type in ("gpt_megatron", "gpt_dolomite"):
            convert_dolomite = False
        else:
            convert_dolomite = True

        # Build the final output directory path
        final_output_dir = Path(output_dir) / "hf_format" / subdir

        if self.model.model_type == "dolomite" and convert_dolomite:
            tmpdir = TemporaryDirectory("w")  # pylint: disable=consider-using-with
            output_dir = Path(tmpdir.name)
        else:
            output_dir = final_output_dir

        CONFIG_NAME = "config.json"
        output_config_file = output_dir / CONFIG_NAME

        # XXX(osilkin): LoRA + FSDP requires a different saving path than the others
        #               so we set this variable and use it to avoid those paths further down.
        is_fsdp_lora = (
            self.model.lora_config is not None
            and self.accelerator.distributed_type == DistributedBackend.FSDP
        )
        if is_fsdp_lora:
            self.save_fsdp_lora_model(
                model=self.model,
                accelerator=self.accelerator,
                output_dir=output_dir,
            )

        get_state_dict_unpatched = self.accelerator.get_state_dict

        def _get_state_dict_patched(model, unwrap=False):
            return get_state_dict_unpatched(model, unwrap=unwrap)

        self.accelerator.get_state_dict = _get_state_dict_patched

        if not is_fsdp_lora and self.accelerator.is_main_process:
            if self.model.lora_config is not None:
                self.model.module.merge_adapter()
                model_state = self.model.module.state_dict()

            output_dir.mkdir(parents=True, exist_ok=True)
            if not self.model.module.config.architectures and convert_dolomite:
                arch_added = False
                if self.model.model_type == "llama":
                    self.model.module.config.architectures = ["LlamaForCausalLM"]
                    arch_added = True
                elif self.model.model_type == "granite":
                    self.model.module.config.architectures = ["GraniteForCausalLM"]
                    arch_added = True
                if arch_added:
                    warnings.warn(
                        f"Adding architectures to ckpt: {self.model.module.config.architectures}",
                    )
                else:
                    warnings.warn(
                        f"Converting from dolomite, but no architecture field added to config.json",
                    )
            self.model.module.config.to_json_file(output_config_file)
            self.model.tokenizer.save_pretrained(output_dir)

            if self.model.lora_config is not None:
                self.save_dict_accelerate(
                    self.accelerator,
                    model_state,
                    save_directory=output_dir,
                    max_shard_size="5GB",
                    safe_serialization=True,
                )
                self.model.module.unmerge_adapter()

        if self.model.lora_config is None:
            self.accelerator.save_model(
                self.model,
                save_directory=output_dir,
                max_shard_size="5GB",
                safe_serialization=True,
            )

        if (
            self.model.model_type == "dolomite"
            and convert_dolomite
            and self.accelerator.is_main_process
        ):
            # export doesnt like the directory to exist
            if final_output_dir.exists():
                shutil.rmtree(final_output_dir)
            export_to_huggingface(
                pretrained_model_name_or_path=tmpdir.name,
                save_path=final_output_dir,
                model_type=self.model.model_type,
            )
            tmpdir.cleanup()

        log_rank_0(f"\033[93mModel saved in {final_output_dir}\033[0m", to_print=True)
        log_rank_0(f"saving took {time.time() - start} seconds")
        dist.barrier()

        self.accelerator.get_state_dict = get_state_dict_unpatched

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

        cleaned_state_dict = OrderedDict()
        for param_tensor in state_dict:
            if not "lora" in param_tensor:
                cleaned_state_dict[
                    param_tensor.replace(".base_layer", "").replace(
                        "basemodel.model.", ""
                    )
                ] = deepcopy(state_dict[param_tensor]).cpu()
        return cleaned_state_dict

    def load_latest_full_state(self, output_dir: Path) -> None:
        """Loads accelerator state from most recently saved checkpoint
        in `output_dir/full_state`.

        Args:
            output_dir: Base output directory containing the full_state subdirectory
        """
        full_state_dir = output_dir / "full_state"

        if not full_state_dir.is_dir():
            return

        # picks checkpoint with the largest number of samples by splitting the "samples_NNNN" string on _
        # and comparing the number at the end of the string
        checkpoint_list = sorted(
            list(full_state_dir.iterdir()),
            reverse=True,
            key=lambda x: int(str(x).rsplit("_", maxsplit=1)[-1]),
        )

        if len(checkpoint_list) == 0:
            log_rank_0(
                f"\033[93mNo checkpoints to load from: {full_state_dir}\033[0m",
                to_print=True,
            )
            return

        latest_checkpoint = checkpoint_list[0]
        log_rank_0(
            f"\033[93mLoading checkpoint from: {latest_checkpoint}\033[0m",
            to_print=True,
        )
        self.accelerator.load_state(latest_checkpoint)

    def save_all_checkpoints(
        self,
        output_dir,
        epoch: int,
        samples_seen: int,
        last_epoch: bool = False,
    ):
        self.save_hf_format_accelerate(
            output_dir=output_dir,
            epoch=epoch,
            samples_seen=samples_seen,
            last_epoch=last_epoch,
        )
        self.save_full_state(
            output_dir=output_dir, epoch=epoch, samples_seen=samples_seen
        )
