# Standard
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch
import os
import sys

# Third Party
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import pytest
import torch
import torch.nn as nn

# First Party
from instructlab.training.config import DistributedBackend, ModelTypes, Optimizers
from instructlab.training.model import Accelerator, Checkpointer, Model, setup_optimizer


# Define base model class at module level
class MockBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = MagicMock()
        # Add layers to base model
        layer0 = MagicMock()
        layer1 = MagicMock()
        self.layers = nn.ModuleList([layer0, layer1])


# Test fixtures
@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec="PreTrainedTokenizer")
    tokenizer.__len__.return_value = 1000
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def mock_config():
    config = MagicMock(spec=PretrainedConfig)
    config.vocab_size = 1000
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.architectures = ["LlamaForCausalLM"]
    return config


@pytest.fixture
def mock_model(mock_config, mock_tokenizer):
    # Create a mock model that matches the expected structure
    class MockBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(1000, 768)
            self.forward = MagicMock()
            # Add the projection layers directly to the base model
            self.q_proj = nn.Linear(768, 768)
            self.v_proj = nn.Linear(768, 768)

            def prepare_inputs_for_generation(*args, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            self.prepare_inputs_for_generation = prepare_inputs_for_generation

    class MockLlamaForCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = mock_config
            self.lora_config = (
                Model.create_lora_config(
                    lora_target_modules=["q_proj", "v_proj"],
                    lora_alpha=16,
                    lora_dropout=0.1,
                    lora_r=8,
                ),
            )
            self._no_split_modules = ["transformer"]
            self.__class__.__name__ = "LlamaForCausalLM"
            self.gradient_checkpointing_enable = MagicMock()
            self.gradient_checkpointing = False
            self.parameters = MagicMock(return_value=[])
            self.base_model_args = {}
            self.model_type = ModelTypes.CAUSALLM
            self.model = self
            self.module = self
            self.update_model = MagicMock()
            self.tokenizer = mock_tokenizer
            self.base_model = MockBaseModel()

            def named_modules_mock(*args, **kwargs):
                return [
                    ("base_model.q_proj", self.base_model.q_proj),
                    ("base_model.v_proj", self.base_model.v_proj),
                ]

            self.named_modules = named_modules_mock

            def get_submodule_mock(name):
                if name == "base_model":
                    return self.base_model
                elif name == "base_model.q_proj":
                    return self.base_model.q_proj
                elif name == "base_model.v_proj":
                    return self.base_model.v_proj
                return None

            self.get_submodule = get_submodule_mock

            def get_input_embeddings():
                return self.base_model.embed_tokens

            self.get_input_embeddings = get_input_embeddings

            # Override _apply to prevent recursion
            def _apply_mock(fn):
                return self

            self._apply = _apply_mock

            # Add prepare_inputs_for_generation to match base model
            def prepare_inputs_for_generation(*args, **kwargs):
                return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

            self.prepare_inputs_for_generation = prepare_inputs_for_generation

    model = MockLlamaForCausalLM()
    return model


@pytest.fixture
def mock_peft_model(mock_model):
    # Create a mock PEFT model that wraps the base model
    class MockPEFTModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.lora_config = MagicMock()
            self.lora_config = Model.create_lora_config(
                lora_target_modules=["q_proj", "v_proj"],
                lora_alpha=16,
                lora_dropout=0.1,
                lora_r=8,
            )
            self.lora_config.lora_alpha = 16
            self.lora_config.lora_dropout = 0.1
            self.lora_config.target_modules = ["q_proj", "v_proj"]
            self.lora_config.task_type = "CAUSAL_LM"

            def __getattr__(self, name):
                if name == "base_model":
                    return self.base_model
                return getattr(self.base_model, name)

            # Override _apply to prevent recursion
            def _apply_mock(fn):
                return self

            self._apply = _apply_mock

            # Add prepare_inputs_for_generation to match base model
            def prepare_inputs_for_generation(*args, **kwargs):
                return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

            self.prepare_inputs_for_generation = prepare_inputs_for_generation

    return MockPEFTModel(mock_model)


@pytest.fixture
def mock_dataloader():
    dataloader = MagicMock()
    dataloader.dataset = MagicMock()
    dataloader.dataset.__len__.return_value = 1000
    return dataloader


@pytest.fixture
def mock_distributed():
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.barrier"),
    ):
        yield


@pytest.fixture
def mock_model_path(tmp_path):
    return str(tmp_path / "model")


@pytest.fixture
def mock_output_dir(tmp_path):
    return str(tmp_path / "output")


# Model class tests
class TestModel:
    def test_model_initialization(self, mock_tokenizer, mock_model, mock_peft_model):
        with (
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ) as mock_from_pretrained,
            patch("peft.LoraModel", return_value=mock_peft_model),
        ):
            model = Model(
                model_path="test_path",
                output_dir="test_output",
                distributed_framework=DistributedBackend.FSDP,
                model_type=ModelTypes.CAUSALLM,
                noise_alpha=0.1,
                tokenizer=mock_tokenizer,
                flash_enabled=True,
                lora_config=Model.create_lora_config(
                    lora_target_modules=["q_proj", "v_proj"],
                    lora_alpha=16,
                    lora_dropout=0.1,
                    lora_r=8,
                ),
            )
            mock_from_pretrained.assert_called_once()
            assert model.model_type == ModelTypes.CAUSALLM
            assert model.lora_config.r == 8
            assert model.lora_config.lora_alpha == 16
            assert model.lora_config.lora_dropout == 0.1
            assert sorted(model.lora_config.target_modules) == sorted(
                ["q_proj", "v_proj"]
            )
            mock_model.gradient_checkpointing_enable.assert_called_once()

    def test_get_projection_layer_names(self, mock_model, mock_tokenizer):
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ):
            model = Model(
                model_path="test_path",
                output_dir="test_output",
                distributed_framework=DistributedBackend.FSDP,
                model_type=ModelTypes.CAUSALLM,
                noise_alpha=0.1,
                tokenizer=mock_tokenizer,
            )
            proj_layers = model.get_projection_layer_names()
            assert set(proj_layers) == {"q_proj", "v_proj"}

    def test_prepare_peft_model_fsdp(self, mock_model, mock_tokenizer):
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ):
            model = Model(
                model_path="test_path",
                output_dir="test_output",
                distributed_framework=DistributedBackend.FSDP.value,
                model_type=ModelTypes.CAUSALLM,
                noise_alpha=0.1,
                tokenizer=mock_tokenizer,
                lora_config=Model.create_lora_config(
                    lora_target_modules=["q_proj", "v_proj"],
                    lora_alpha=16,
                    lora_dropout=0.1,
                    lora_r=8,
                ),
            )
            with patch("peft.LoraModel") as mock_lora_model:
                model.prepare_peft_model()
                mock_lora_model.assert_called_once()

    def test_prepare_peft_model_deepspeed(self, mock_model, mock_tokenizer):
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ):
            # Mock the PeftModel check
            mock_model.is_loaded_in_8bit = False
            mock_model.is_loaded_in_4bit = False
            mock_model.__class__.__name__ = "LlamaForCausalLM"  # Not a PeftModel

            # Create a mock PeftModel class
            class MockPeftModel(nn.Module):
                pass

            model = Model(
                model_path="test_path",
                output_dir="test_output",
                distributed_framework=DistributedBackend.DEEPSPEED.value,
                model_type=ModelTypes.CAUSALLM,
                noise_alpha=0.1,
                tokenizer=mock_tokenizer,
                lora_config=Model.create_lora_config(
                    lora_target_modules=["q_proj", "v_proj"],
                    lora_alpha=16,
                    lora_dropout=0.1,
                    lora_r=8,
                ),
            )
            with (
                patch("peft.get_peft_model") as mock_get_peft_model,
                patch("peft.PeftModel", MockPeftModel),
            ):
                model.prepare_peft_model()
                mock_get_peft_model.assert_called_once()

    def test_create_lora_config(self, mock_tokenizer, mock_model):
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ):
            lora_config = Model.create_lora_config(
                lora_target_modules=["q_proj", "v_proj"],
                lora_alpha=16,
                lora_dropout=0.1,
                lora_r=8,
            )
            assert lora_config.r == 8
            assert lora_config.lora_alpha == 16
            assert lora_config.lora_dropout == 0.1
            assert sorted(lora_config.target_modules) == sorted(["q_proj", "v_proj"])

    def test_reconcile_tokenizer(self, mock_tokenizer, mock_model):
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ):
            model = Model(
                model_path="test_path",
                output_dir="test_output",
                distributed_framework=DistributedBackend.FSDP,
                model_type=ModelTypes.CAUSALLM,
                noise_alpha=0.1,
                tokenizer=mock_tokenizer,
            )
            model.reconcile_tokenizer()
            assert model.model.config.pad_token_id == mock_tokenizer.pad_token_id
            assert model.model.config.bos_token_id == mock_tokenizer.bos_token_id
            assert model.model.config.eos_token_id == mock_tokenizer.eos_token_id

    def test_supports_flash_attention(self):
        with (
            patch(
                "torch.cuda.get_device_capability", return_value=(8, 0)
            ) as mock_capability,
            patch(
                "torch.cuda.get_device_properties",
                return_value=MagicMock(gcnArchName="gfx90a:0"),
            ) as mock_props,
        ):
            assert Model.supports_flash_attention() is True
            mock_capability.assert_called_once()
            mock_props.assert_called_once()

    def test_check_flash_attn_enabled(self):
        # Test when flash attention is enabled and supported
        with patch.object(Model, "supports_flash_attention", return_value=True):
            assert Model.check_flash_attn_enabled(False, False) is True

        # Test when flash attention is enabled but not supported
        with patch.object(Model, "supports_flash_attention", return_value=False):
            with pytest.raises(
                RuntimeError,
                match="Trying to use Flash Attention on unsupported hardware",
            ):
                Model.check_flash_attn_enabled(False, False)

        # Test when flash attention is disabled but dolomite is enabled
        with pytest.raises(
            RuntimeError,
            match="Trying to use dolomite padding-free transformer without flash attention",
        ):
            Model.check_flash_attn_enabled(True, True)

        # Test when flash attention is disabled and dolomite is disabled
        assert Model.check_flash_attn_enabled(True, False) is False

    def test_setup_optimizer(self, mock_model, mock_tokenizer):
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained"
        ) as mock_from_pretrained:
            mock_model = MagicMock()
            mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
            mock_model.config = MagicMock()
            mock_model.config.vocab_size = 1000
            mock_model.__class__.__name__ = "LlamaForCausalLM"  # Set correct class name
            mock_from_pretrained.return_value = mock_model
            mock_tokenizer.__len__.return_value = 1000

            model = Model(
                model_path="instructlab/granite-7b-lab",
                output_dir="test_output",
                distributed_framework=DistributedBackend.FSDP,
                model_type=ModelTypes.CAUSALLM,
                noise_alpha=None,
                tokenizer=mock_tokenizer,
            )
            model.model = mock_model

            # Test FSDP with AdamW
            optimizer = setup_optimizer(
                model=model,
                cpu_offload=False,
                name=None,
                learning_rate=1e-4,
            )
            assert isinstance(optimizer, torch.optim.AdamW)

            # Test DeepSpeed with FusedAdam
            model.distributed_framework = DistributedBackend.DEEPSPEED
            with patch("instructlab.training.model.FusedAdam") as mock_fused_adam:
                optimizer = setup_optimizer(
                    model=model,
                    cpu_offload=False,
                    name=None,
                    learning_rate=1e-4,
                )
                mock_fused_adam.assert_called_once()

            # Test DeepSpeed with CPUAdam
            with patch("instructlab.training.model.DeepSpeedCPUAdam") as mock_cpu_adam:
                optimizer = setup_optimizer(
                    model=model,
                    cpu_offload=True,
                    name=None,
                    learning_rate=1e-4,
                )
                mock_cpu_adam.assert_called_once()

            # Test explicit optimizer selection
            with patch("instructlab.training.model.AdamW") as mock_adamw:
                optimizer = setup_optimizer(
                    model=model,
                    cpu_offload=False,
                    name=Optimizers.ADAMW,
                    learning_rate=1e-4,
                )
                mock_adamw.assert_called_once()

    def test_model_lora_initialization(
        self, mock_model_path, mock_output_dir, mock_tokenizer
    ):
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained"
        ) as mock_from_pretrained:
            # Create a simpler mock model
            mock_model = MagicMock()
            mock_model._no_split_modules = ["transformer"]
            mock_model.config = MagicMock()
            mock_model.config.vocab_size = 1000
            mock_model.config.pad_token_id = 0
            mock_model.config.bos_token_id = 1
            mock_model.config.eos_token_id = 2
            mock_model.config.architectures = ["LlamaForCausalLM"]
            mock_model.gradient_checkpointing = False
            mock_model.__class__.__name__ = "LlamaForCausalLM"
            mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
            mock_model.base_model_args = {}
            mock_model.model_type = ModelTypes.CAUSALLM
            mock_model.model = mock_model
            mock_model.module = mock_model
            mock_model.tokenizer = mock_tokenizer
            mock_model.base_model = MagicMock()
            mock_model.base_model.embed_tokens = MagicMock()
            mock_model.get_input_embeddings = MagicMock(
                return_value=mock_model.base_model.embed_tokens
            )
            mock_model.base_model.q_proj = nn.Linear(768, 768)
            mock_model.base_model.v_proj = nn.Linear(768, 768)

            # Add named_modules to support LoRA
            def named_modules_mock(*args, **kwargs):
                return [
                    ("base_model.v_proj", mock_model.base_model.v_proj),
                    ("base_model.q_proj", mock_model.base_model.q_proj),
                ]

            mock_model.named_modules = named_modules_mock
            mock_from_pretrained.return_value = mock_model

            with (
                patch("peft.LoraModel") as mock_peft_model,
                patch("peft.get_peft_model") as mock_get_peft_model,
                patch("peft.prepare_model_for_kbit_training") as mock_prepare_model,
            ):
                # Mock the PEFT model initialization
                mock_peft_model.return_value = mock_model
                mock_get_peft_model.return_value = mock_model
                mock_prepare_model.return_value = mock_model

                model = Model(
                    model_path=mock_model_path,
                    output_dir=mock_output_dir,
                    distributed_framework=DistributedBackend.FSDP,
                    model_type=ModelTypes.CAUSALLM,
                    noise_alpha=None,
                    tokenizer=mock_tokenizer,
                    lora_config=Model.create_lora_config(
                        lora_target_modules=["v_proj", "q_proj"],
                        lora_alpha=32,
                        lora_dropout=0.1,
                        lora_r=8,
                    ),
                )

                assert model.lora_config is not None
                assert model.lora_config.r == 8
                assert model.lora_config.lora_alpha == 32
                assert model.lora_config.lora_dropout == 0.1
                assert set(model.lora_config.target_modules) == {"v_proj", "q_proj"}

    def test_model_reconcile_tokenizer(
        self, mock_model_path, mock_output_dir, mock_tokenizer
    ):
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained"
        ) as mock_from_pretrained:
            mock_model = MagicMock()
            mock_model.config.vocab_size = 1000
            mock_model.config.pad_token_id = None
            mock_model.config.bos_token_id = None
            mock_model.config.eos_token_id = None
            mock_model.__class__.__name__ = "LlamaForCausalLM"  # Set a valid class name
            mock_model.gradient_checkpointing = False
            mock_from_pretrained.return_value = mock_model

            model = Model(
                model_path=mock_model_path,
                output_dir=mock_output_dir,
                distributed_framework=DistributedBackend.FSDP,
                model_type=ModelTypes.CAUSALLM,
                noise_alpha=None,
                tokenizer=mock_tokenizer,
            )

            model.reconcile_tokenizer()

            assert model.model.config.pad_token_id == mock_tokenizer.pad_token_id
            assert model.model.config.bos_token_id == mock_tokenizer.bos_token_id
            assert model.model.config.eos_token_id == mock_tokenizer.eos_token_id


# Accelerator class tests
class TestAccelerator:
    def test_accelerator_initialization(self, mock_model, mock_dataloader):
        mock_model.lora_config = None
        with (
            patch(
                "instructlab.training.utils.get_module_class_from_name",
                return_value=MockBaseModel,
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            patch(
                "torch.cuda.get_device_properties",
                return_value=MagicMock(gcnArchName="gfx90a:0"),
            ),
            patch("accelerate.utils.is_bf16_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda._initialized", True),
            patch("torch.cuda.is_initialized", return_value=True),
        ):
            accelerator = Accelerator(
                model=mock_model,
                samples_per_gpu=4,
                grad_accum=2,
                train_loader=mock_dataloader,
                save_samples=1000,
                distributed_framework=DistributedBackend.FSDP,
                fsdp_sharding_strategy="FULL_SHARD",
                fsdp_cpu_offload_params=True,
            )
            assert accelerator.samples_per_gpu == 4
            assert accelerator.grad_accum == 2
            assert accelerator.save_samples == 1000
            mock_model.update_model.assert_called_once()

    def test_setup_lr_scheduler(self, mock_model, mock_dataloader):
        mock_model.lora_config = None
        with (
            patch(
                "instructlab.training.utils.get_module_class_from_name",
                return_value=MockBaseModel,
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            patch(
                "torch.cuda.get_device_properties",
                return_value=MagicMock(gcnArchName="gfx90a:0"),
            ),
            patch("accelerate.utils.is_bf16_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda._initialized", True),
            patch("torch.cuda.is_initialized", return_value=True),
        ):
            accelerator = Accelerator(
                model=mock_model,
                samples_per_gpu=4,
                grad_accum=2,
                train_loader=mock_dataloader,
                save_samples=1000,
                distributed_framework=DistributedBackend.FSDP,
                fsdp_sharding_strategy="FULL_SHARD",
            )

            # Create a real AdamW optimizer
            params = [torch.nn.Parameter(torch.randn(2, 2))]
            optimizer = AdamW(params, lr=0.001)
            optimizer.param_groups = [{"lr": 0.001}]
            optimizer.state_dict = MagicMock(
                return_value={"param_groups": [{"lr": 0.001}]}
            )
            optimizer.get_lr = MagicMock(return_value=0.001)

            accelerator.setup_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler="cosine",
                num_epochs=10,
                num_warmup_steps=100,
            )
            assert hasattr(accelerator, "lr_scheduler")


# Checkpointer class tests
class TestCheckpointer:
    def test_checkpointer_initialization(self, mock_model):
        optimizer = MagicMock()
        accelerator = MagicMock()

        checkpointer = Checkpointer(
            model=mock_model,
            optimizer=optimizer,
            accelerator=accelerator,
            strategy="full_state",
        )
        assert checkpointer.strategy == "full_state"

    def test_save_full_state(self, mock_model, tmp_path, mock_distributed):
        optimizer = MagicMock()
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.save_state = MagicMock()
        mock_model.lora_config = None
        checkpointer = Checkpointer(
            model=mock_model,
            optimizer=optimizer,
            accelerator=accelerator,
            strategy="full_state",
        )

        output_dir = tmp_path / "test_output"
        os.makedirs(output_dir / "full_state" / "epoch_1", exist_ok=True)
        checkpointer.save_full_state(output_dir=output_dir, epoch=1, samples_seen=1000)
        accelerator.save_state.assert_called_once()

    def test_save_hf_format_accelerate(self, mock_model, tmp_path, mock_distributed):
        optimizer = MagicMock()
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.save_model = MagicMock()
        mock_model.lora_config = None
        mock_model.model_type = ModelTypes.CAUSALLM
        mock_model.module = mock_model  # Ensure module is set
        mock_model.tokenizer = MagicMock()  # Ensure tokenizer is set

        checkpointer = Checkpointer(
            model=mock_model,
            optimizer=optimizer,
            accelerator=accelerator,
            strategy="hf_format",
        )

        output_dir = tmp_path / "test_output"
        os.makedirs(output_dir / "hf_format" / "samples_1000", exist_ok=True)
        checkpointer.save_hf_format_accelerate(
            output_dir=output_dir, epoch=1, samples_seen=1000
        )
        accelerator.save_model.assert_called_once()
