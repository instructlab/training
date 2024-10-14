# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Tuple, Any

from accelerate import Accelerator, DistributedType
from peft import LoraModel
from torch import distributed as dist
from torch import nn
from torch.cuda import empty_cache
from transformers import PreTrainedModel
import torch

def wraps(module: nn.Module, wrapped_classes: Tuple[Any]) -> bool:
    """Checks if a module or its children are an instance of one of the provided classes.

    Args:
        module (nn.Module): A PyTorch module.
        wrapped_classes(Tuple): A tuple of potential classes the module could be.

    Returns:
        bool: True if the module or any of its children are instances of `transformers.PreTrainedModel`, False otherwise.
    """
    if isinstance(module, wrapped_classes):
        return True
    
    for m in module.children():
        if wraps(m, wrapped_classes):
            return True
    
    return False



class __SuperAccelerator(Accelerator):
    """
    Custom InstructLab Accelerator class that extends the `accelerate.Accelerator` object.
    We extend this class to modify some functionality embedded in the existing Accelerator
    which prevents us from being able to properly save LoRA models when using FSDP as the
    distributed backend.
    
    Warning: This is NOT a public API and is not intended to be supported beyond its
    internal usage in this library. Use at your own discretion.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cpu_model: PreTrainedModel = None
        self._is_lora = False

    def prepare(self, *args, **kwargs):
        # Extract out the model to make a copy on the cpu.
        # Make sure this only happens once per object lifecycle - we call accelerator.prepare
        # several times.
        num_times_found = 0
        using_lora = torch.ByteTensor([False]).cuda()
        if self.distributed_type == DistributedType.FSDP and self.is_main_process and not self._cpu_model:
            for arg in args:
                if isinstance(arg, nn.Module) and wraps(arg, PreTrainedModel) and wraps(arg, LoraModel):
                    using_lora[0] = True
                    num_times_found += 1
                    # cpu model setter logic will handle validation - but this may be a stupid idea and
                    # we should instead handle it here
                    self.cpu_model = arg
                    break
                    
        print(f'number of times we found a lora pretrained arg: {num_times_found}')
        dist.barrier()
        dist.all_reduce(using_lora, op=dist.ReduceOp.MAX)
        if using_lora[0]:
            self._is_lora = True
        return super().prepare(*args, **kwargs)

    @property
    def cpu_model(self) -> nn.Module:
        if self.is_main_process:
            return self._cpu_model
        return None

    @cpu_model.setter
    def cpu_model(self, model: nn.Module) -> nn.Module | None:
        """
        Given a model **BEFORE** it is sent to FSDP, we copy it and keep it on the CPU.
        The model is only stored for the main process, so calling on a non-main will return None.
        """
        if not self.is_main_process:
            # only one process in the group should ever store the model
            return

        # ensure the model is not on the GPU yet
        if any(p.is_cuda for p in model.parameters()):
            # while it is POSSIBLE to copy a model from the GPU to the CPU, we should avoid doing this
            # due to potential memory constraints.
            # 
            # As long as we correctly prepare the model through Accelerate, we should not hit this.
            raise RuntimeError('Copying a model from the GPU to the CPU is not supported.')

        self._cpu_model = deepcopy(model)

    def save_lora_fsdp(self, model: nn.Module, *args, **kwargs) -> None:
        """Extension of the `accelerate.Accelerator.save_model` method.

        This provides the ability to save a model in SafeTensors format when training a LoRA with FSDP.

        Args:
            model (nn.Module): The accelerator-wrapped model to save.
        """
        
        
        if self.distributed_type != DistributedType.FSDP:
            raise RuntimeError('`__SuperAccelerator.save_fsdp_lora` was called when FSDP was not being used.')
        if not self._is_lora:
            raise RuntimeError('`__SuperAccelerator.save_fsdp_lora` was called but was not configured to use LoRA')

        print('GETTING OLD MODEL STATE DICT')
        model_state = self.get_state_dict(model, unwrap=True)
        print('COPYING CPU MODEL')
        if self.is_main_process:
            tmp_model: LoraModel = deepcopy(self.cpu_model)
            print('LOADING STATE DICT INTO TEMP MODEL')
            tmp_model.load_state_dict(model_state)
            print('MERGING & UNLOADING TEMP MODEL')
            tmp_model.merge_and_unload(True)
            print('GETTING STATE DICT FROM TEMP MODEL')
            model_state = tmp_model.state_dict()

            old_get_state_dict = self.get_state_dict
            def _custom_get_state_dict(ref: Accelerator, *args, **kwargs):
                """
                Custom function to trick `accelerate.Accelerator` to get a state dict that will work
                when training with LoRA & FSDP.
                """
                print('RETURNING TEMP MODEL STATE DICT INTO ACCELERATORS SAVE MODEL')
                return model_state
        
            print('OVERWRITING get_state_dict WITH CUSTOM FN')
            self.get_state_dict = _custom_get_state_dict
            print('CALLING ACCELERATOR SAVE_MODEL')
            self.save_model(model, *args, **kwargs)
            print('RETURNING get_state_dict FUNCTION TO OLD')
            self.get_state_dict = old_get_state_dict

            print('DELETING TMP MODEL')
            del tmp_model
            empty_cache()
        

        
        print('SAVED SUCCESSFULLY')
        # from IPython import embed; embed()

