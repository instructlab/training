# Standard
from unittest.mock import MagicMock, patch
import os

# Third Party
from peft import LoraConfig
from transformers import AutoTokenizer
import pytest
import torch

# First Party
from instructlab.training.config import DistributedBackend
from instructlab.training.model import CausalLMModel, LigerModel, Model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.__len__.return_value = 32000
    return tokenizer


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.model_type = "llama"
    return config


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.config.vocab_size = 32000
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.__class__.__name__ = "LlamaForCausalLM"
    model.base_model = MagicMock()
    model.base_model.embed_tokens = MagicMock()
    model.base_model.embed_tokens.forward = MagicMock()
    model.get_input_embeddings = MagicMock(return_value=MagicMock())

    # Add projection layers to the model
    model.named_modules.return_value = [
        ("model.layers.0.self_attn.q_proj", None),
        ("model.layers.0.self_attn.v_proj", None),
        ("model.layers.0.self_attn.k_proj", None),
        ("model.layers.0.self_attn.o_proj", None),
    ]
    return model


@pytest.fixture
def lora_config():
    return LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )


def test_model_initialization(mock_tokenizer, mock_model, mock_config):
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ),
    ):
        model = CausalLMModel(
            model_path="test_model",
            output_dir="test_output",
            distributed_framework=DistributedBackend.DEEPSPEED,
            noise_alpha=None,
            tokenizer=mock_tokenizer,
        )

        assert isinstance(model, CausalLMModel)
        assert model.distributed_framework == DistributedBackend.DEEPSPEED
        assert model.tokenizer == mock_tokenizer
        assert model.noise_alpha is None


def test_model_with_lora(mock_tokenizer, mock_model, mock_config, lora_config):
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ),
        patch("peft.get_peft_model", return_value=mock_model),
    ):
        model = CausalLMModel(
            model_path="test_model",
            output_dir="test_output",
            distributed_framework=DistributedBackend.DEEPSPEED,
            noise_alpha=None,
            tokenizer=mock_tokenizer,
            lora_config=lora_config,
        )

        assert model.lora_config == lora_config


def test_reconcile_tokenizer(mock_tokenizer, mock_model, mock_config):
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ),
    ):
        # Test case where tokenizer has more tokens than model
        mock_tokenizer.__len__.return_value = 33000

        model = CausalLMModel(
            model_path="test_model",
            output_dir="test_output",
            distributed_framework=DistributedBackend.DEEPSPEED,
            noise_alpha=None,
            tokenizer=mock_tokenizer,
        )

        # Verify that resize_token_embeddings was called
        mock_model.resize_token_embeddings.assert_called_once()


def test_model_train_mode(mock_tokenizer, mock_model, mock_config):
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ),
    ):
        model = CausalLMModel(
            model_path="test_model",
            output_dir="test_output",
            distributed_framework=DistributedBackend.DEEPSPEED,
            noise_alpha=None,
            tokenizer=mock_tokenizer,
        )

        # Test train mode
        model.train(True)
        mock_model.train.assert_called_once_with(True)

        # Test eval mode
        model.train(False)
        mock_model.train.assert_called_with(False)


def test_model_parameters(mock_tokenizer, mock_model, mock_config):
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ),
    ):
        model = CausalLMModel(
            model_path="test_model",
            output_dir="test_output",
            distributed_framework=DistributedBackend.DEEPSPEED,
            noise_alpha=None,
            tokenizer=mock_tokenizer,
        )

        # Test parameters method
        model.parameters()
        mock_model.parameters.assert_called_once()


def test_model_get_projection_layers(mock_tokenizer, mock_model, mock_config):
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ),
    ):
        model = CausalLMModel(
            model_path="test_model",
            output_dir="test_output",
            distributed_framework=DistributedBackend.DEEPSPEED,
            noise_alpha=None,
            tokenizer=mock_tokenizer,
        )

        proj_layers = model.projection_layer_names
        assert set(proj_layers) == {"q_proj", "v_proj", "k_proj", "o_proj"}


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_properties")
def test_model_flash_attention_support(mock_device_props, mock_cuda_available):
    # Test NVIDIA GPU support
    mock_device = MagicMock()
    mock_device.gcnArchName = "sm_80"  # Not an AMD GPU
    mock_device_props.return_value = mock_device

    # Test with supported NVIDIA architecture
    with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
        assert Model.supports_flash_attention() is True

    # Test with unsupported NVIDIA architecture
    with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        assert Model.supports_flash_attention() is False

    # Test AMD GPU support
    mock_device.gcnArchName = "gfx90a"
    mock_device_props.return_value = mock_device

    # Test with supported AMD architecture
    with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        assert Model.supports_flash_attention() is True


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_capability", return_value=(8, 0))
@patch("torch.cuda.get_device_properties")
def test_model_flash_attention_check(
    mock_device_props, mock_capability, mock_cuda_available
):
    # Mock device properties for NVIDIA GPU
    mock_device = MagicMock()
    mock_device.gcnArchName = "sm_80"
    mock_device_props.return_value = mock_device

    # Test enabling flash attention
    assert Model.check_flash_attn_enabled(False) is True

    # Test disabling flash attention
    assert Model.check_flash_attn_enabled(True) is False


# New tests for model initializations


def test_causal_lm_model_with_flash_attention(mock_tokenizer, mock_model, mock_config):
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ),
    ):
        model = CausalLMModel(
            model_path="test_model",
            output_dir="test_output",
            distributed_framework=DistributedBackend.DEEPSPEED,
            noise_alpha=None,
            tokenizer=mock_tokenizer,
            flash_enabled=True,
        )

        assert model.base_model_args["attn_implementation"] == "flash_attention_2"


def test_model_with_noise_alpha(mock_tokenizer, mock_model, mock_config):
    mock_model.__class__.__name__ = "LlamaForCausalLM"
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
        ),
    ):
        model = CausalLMModel(
            model_path="test_model",
            output_dir="test_output",
            distributed_framework=DistributedBackend.DEEPSPEED,
            noise_alpha=0.1,
            tokenizer=mock_tokenizer,
        )

        assert model.noise_alpha == 0.1
