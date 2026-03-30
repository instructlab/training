# Standard
import logging

# Third Party
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedModel
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
)

logger = logging.getLogger("instructlab.training")


def is_vlm_with_causal_lm(model_path: str, trust_remote_code: bool = False) -> bool:
    """Check if a model is a VLM that wraps a CausalLM text backbone.

    Returns True when the model needs VLM extraction to obtain the trainable
    CausalLM sub-model.  This covers two cases:

    1. The top-level config does NOT map to CausalLM, but ``text_config`` does
       (e.g. Ministral-3 / Mistral3ForConditionalGeneration).
    2. The top-level config IS in the CausalLM mapping, but the resolved class
       is actually a ``ForConditionalGeneration`` VLM (e.g. Gemma 3, which is
       dual-registered so ``AutoModelForCausalLM`` loads the full VLM).  These
       models still have an extractable CausalLM text backbone via
       ``text_config``.

    Args:
        model_path: HuggingFace model ID or local path.
        trust_remote_code: Whether to trust remote code when loading config.

    Returns:
        True if the model is a VLM wrapper around a CausalLM text backbone.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    text_config = getattr(config, "text_config", None)

    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        # The config maps to CausalLM, but check what class it actually
        # resolves to.  Some models (e.g. Gemma 3) are dual-registered and
        # AutoModelForCausalLM loads a ForConditionalGeneration VLM instead
        # of a text-only CausalLM.  Those still need extraction.
        resolved_cls = MODEL_FOR_CAUSAL_LM_MAPPING[config.__class__]
        is_actually_vlm = "ForConditionalGeneration" in resolved_cls.__name__
        if not is_actually_vlm:
            return False
        # It's a VLM disguised as CausalLM — fall through to check text_config
        if text_config is None:
            return False
        return text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING

    if text_config is None:
        return False

    return text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING


def is_vlm_for_direct_loading(model_path: str, trust_remote_code: bool = False) -> bool:
    """Check if a model is a VLM that should be loaded directly for text-only training.

    This handles models where:
    - No CausalLM class exists (top-level config NOT in MODEL_FOR_CAUSAL_LM_MAPPING)
    - No extractable text backbone (text_config also NOT in MODEL_FOR_CAUSAL_LM_MAPPING)
    - But the model IS in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING (can be loaded as VLM)

    Args:
        model_path: HuggingFace model ID or local path.
        trust_remote_code: Whether to trust remote code when loading config.

    Returns:
        True if the model should be loaded directly as a VLM for text-only training.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    # Skip if it can be loaded as CausalLM directly
    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        return False

    # Skip if extractable (text_config maps to CausalLM)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        return False

    # Check if it's loadable as a VLM
    return config.__class__ in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING


def load_vlm_for_text_training(
    model_path: str,  # pylint: disable=unused-argument
    load_kwargs: dict,
) -> PreTrainedModel:
    """Load a VLM model directly for text-only training.

    Used when no CausalLM variant exists. The VLM's forward() works with just
    input_ids + labels, producing logits and CE loss without vision inputs.

    Args:
        model_path: HuggingFace model ID or local path.
        load_kwargs: Keyword arguments for ``from_pretrained`` (must include
            ``pretrained_model_name_or_path``).

    Returns:
        The VLM ``PreTrainedModel`` loaded for text-only training.
    """
    filtered_kwargs = {
        k: v
        for k, v in load_kwargs.items()
        if not (k == "quantization_config" and v is None)
    }

    logger.info(
        "VLM config detected — loading VLM directly for text-only training "
        "(no CausalLM variant available)"
    )
    return AutoModelForImageTextToText.from_pretrained(**filtered_kwargs)


def _find_text_backbone(vlm_model) -> nn.Module:
    """Auto-detect the text backbone attribute in a VLM.

    Tries common attribute names first, then falls back to searching
    ``named_children()`` for modules whose class name contains
    ``ForCausalLM`` or ``TextModel``.

    Args:
        vlm_model: The loaded VLM model.

    Returns:
        The text backbone ``nn.Module``.

    Raises:
        ValueError: If no text backbone can be found.
    """
    inner = vlm_model.model if hasattr(vlm_model, "model") else vlm_model

    # Try common attribute names in order of prevalence
    for attr_name in ("language_model", "text_model", "llm"):
        if hasattr(inner, attr_name):
            return getattr(inner, attr_name)

    # Fallback: search named_children for recognisable patterns
    for name, module in inner.named_children():
        class_name = module.__class__.__name__
        if "ForCausalLM" in class_name or "TextModel" in class_name:
            logger.info(
                "Auto-detected text backbone via named_children: %s (%s)",
                name,
                class_name,
            )
            return module

    available = [name for name, _ in inner.named_children()]
    raise ValueError(
        f"Could not find text backbone in VLM model. "
        f"Available sub-modules on the inner model: {available}"
    )


def _dequantize_fp8_model(model: PreTrainedModel) -> None:
    """Dequantize FP8 weights in-place for FSDP compatibility.

    Some models (e.g. Ministral) ship with FP8 quantized weights that include
    scalar parameters like ``weight_scale_inv`` and ``activation_scale``.
    FSDP rejects scalar parameters, so we dequantize the weights back to
    bfloat16 and remove all FP8 scalar parameters before distributed wrapping.

    The original FP8 scales and quantization config are preserved on the model
    (as ``_fp8_scales`` and ``_fp8_quantization_config``) so that
    :func:`requantize_fp8_state_dict` can restore them at checkpoint save time.

    The dequantization formula is:
        real_weight = fp8_weight.to(bfloat16) * weight_scale_inv
    """
    # FP8 scalar parameter names to remove after dequantization.
    # weight_scale_inv: inverse scale for weight quantization
    # activation_scale: scale for activation quantization (inference only)
    _FP8_SCALAR_ATTRS = ("weight_scale_inv", "activation_scale")

    # Store original scales keyed by module path for requantization at save time.
    fp8_scales: dict[str, dict[str, torch.Tensor]] = {}

    dequantized_count = 0
    for mod_name, module in model.named_modules():
        has_fp8 = any(hasattr(module, attr) for attr in _FP8_SCALAR_ATTRS)
        if not has_fp8:
            continue

        # Capture original scales before removing them
        saved = {}
        for attr in _FP8_SCALAR_ATTRS:
            if hasattr(module, attr):
                saved[attr] = getattr(module, attr).detach().clone().cpu()
        if saved:
            fp8_scales[mod_name] = saved

        # Dequantize weight if scale is present
        if hasattr(module, "weight_scale_inv") and hasattr(module, "weight"):
            scale_inv = module.weight_scale_inv
            weight = module.weight
            dtype = torch.bfloat16
            dequantized = weight.to(dtype) * scale_inv.to(dtype)
            module.weight = nn.Parameter(
                dequantized, requires_grad=weight.requires_grad
            )

        # Remove all FP8 scalar parameters/buffers
        for attr in _FP8_SCALAR_ATTRS:
            if not hasattr(module, attr):
                continue
            if attr in dict(module.named_parameters(recurse=False)):
                delattr(module, attr)
            elif attr in dict(module.named_buffers(recurse=False)):
                setattr(module, attr, None)

        dequantized_count += 1

    if dequantized_count > 0:
        logger.info(
            "Dequantized %d FP8 layers to bfloat16 for FSDP compatibility",
            dequantized_count,
        )
        # Preserve scales and quantization config for checkpoint re-quantization.
        # Store on both the model and its config so the metadata survives
        # model wrapping (FSDP) and distributed broadcast.
        model._fp8_scales = fp8_scales
        cfg = getattr(model, "config", None)
        if cfg is not None:
            cfg._fp8_scales = fp8_scales
            if hasattr(cfg, "quantization_config"):
                model._fp8_quantization_config = cfg.quantization_config
                cfg._fp8_quantization_config = cfg.quantization_config
                cfg.quantization_config = None
        # Clear quantization metadata so downstream code doesn't treat
        # the model as quantized during training
        if hasattr(model, "hf_quantizer"):
            model.hf_quantizer = None
        if hasattr(model, "is_loaded_in_8bit"):
            model.is_loaded_in_8bit = False


def requantize_fp8_state_dict(
    state_dict: dict[str, torch.Tensor],
    fp8_scales: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Re-quantize a dequantized state dict back to FP8 for checkpoint saving.

    This is the inverse of :func:`_dequantize_fp8_model`.  It converts
    bfloat16 weights back to ``float8_e4m3fn`` and restores the original
    ``weight_scale_inv`` and ``activation_scale`` entries so the saved
    checkpoint matches the original FP8 format.

    Args:
        state_dict: The model state dict with bfloat16 weights.
        fp8_scales: The ``_fp8_scales`` dict stored by
            :func:`_dequantize_fp8_model`, mapping module paths to their
            original scale tensors.

    Returns:
        A new state dict with FP8 weights and restored scale entries.
    """
    out = {}
    for key, tensor in state_dict.items():
        out[key] = tensor

    for mod_path, scales in fp8_scales.items():
        weight_key = f"{mod_path}.weight"
        if weight_key not in out:
            continue

        weight = out[weight_key]

        # Re-quantize: fp8_weight = real_weight / weight_scale_inv
        if "weight_scale_inv" in scales:
            scale_inv = scales["weight_scale_inv"]
            requantized = (weight.to(torch.float32) / scale_inv.to(torch.float32)).to(
                torch.float8_e4m3fn
            )
            out[weight_key] = requantized
            out[f"{mod_path}.weight_scale_inv"] = scale_inv

        # Restore activation_scale as-is
        if "activation_scale" in scales:
            out[f"{mod_path}.activation_scale"] = scales["activation_scale"]

    return out


def extract_causal_lm_from_vlm(
    model_path: str,
    load_kwargs: dict,
    trust_remote_code: bool = False,
) -> "PreTrainedModel":  # noqa: F821
    """Load a VLM and extract its CausalLM text backbone.

    This is used when a model's config is not directly registered as a CausalLM
    but wraps one internally (e.g. Ministral-3-3B uses a VLM architecture
    around a Mistral CausalLM).

    Uses ``init_empty_weights`` to avoid allocating random-weight tensors for
    the CausalLM wrapper, then transfers the real sub-modules from the VLM.

    Args:
        model_path: HuggingFace model ID or local path.
        load_kwargs: Keyword arguments for ``from_pretrained`` (must include
            ``pretrained_model_name_or_path``).
        trust_remote_code: Whether to trust remote code when loading config.

    Returns:
        A standalone ``PreTrainedModel`` CausalLM containing the extracted
        text backbone weights.
    """
    # Third Party
    from accelerate import init_empty_weights

    # Filter out quantization_config=None and pretrained_model_name_or_path
    # (already passed positionally or via kwargs to avoid duplicates).
    filtered_kwargs = {
        k: v
        for k, v in load_kwargs.items()
        if not (k == "quantization_config" and v is None)
    }

    logger.info(
        "VLM config detected — loading full VLM to extract CausalLM text backbone"
    )
    vlm = AutoModelForImageTextToText.from_pretrained(**filtered_kwargs)

    # Auto-detect the text backbone sub-module
    backbone = _find_text_backbone(vlm)

    # Build a lightweight CausalLM shell using empty weights to avoid
    # allocating large random tensors, then attach the real sub-modules.
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    text_config = config.text_config

    # Propagate quantization_config from the VLM config to text_config
    # so FP8 dequantization can preserve and restore it at checkpoint time.
    vlm_quant_cfg = getattr(config, "quantization_config", None)
    if vlm_quant_cfg is not None and not hasattr(text_config, "quantization_config"):
        text_config.quantization_config = vlm_quant_cfg

    causal_lm_class = MODEL_FOR_CAUSAL_LM_MAPPING[text_config.__class__]

    with init_empty_weights():
        text_model = causal_lm_class(text_config)

    # Transfer real weights from the VLM
    text_model.model = backbone
    text_model.lm_head = vlm.lm_head

    # Copy quantization metadata so downstream PEFT/kbit_training works
    for attr in ("is_loaded_in_4bit", "is_loaded_in_8bit", "hf_quantizer"):
        if hasattr(vlm, attr):
            setattr(text_model, attr, getattr(vlm, attr))

    del vlm

    # Dequantize FP8 weights if present — FSDP rejects scalar parameters
    # like weight_scale_inv that come from FP8 quantized models.
    _dequantize_fp8_model(text_model)

    return text_model


def has_mrope(model_path: str, trust_remote_code: bool = False) -> bool:
    """Check if a model uses M-RoPE (multimodal rotary position embeddings).

    Models with M-RoPE produce 3D position_ids that are incompatible with
    Flash Attention 2's packed-sequence detection. This checks both
    ``rope_scaling`` and ``rope_parameters`` on both the top-level config
    and ``text_config``.

    Args:
        model_path: HuggingFace model ID or local path.
        trust_remote_code: Whether to trust remote code when loading config.

    Returns:
        True if any rope configuration contains an ``mrope_section`` key.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return _config_has_mrope(config)


def _config_has_mrope(config) -> bool:
    """Check if a config object uses M-RoPE."""
    candidates = []
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        for attr in ("rope_scaling", "rope_parameters"):
            rope = getattr(cfg, attr, None)
            if rope is not None:
                candidates.append(rope)
    return any(
        (isinstance(rope, dict) and "mrope_section" in rope)
        or (not isinstance(rope, dict) and hasattr(rope, "mrope_section"))
        for rope in candidates
    )


def needs_sdpa(model_path: str, trust_remote_code: bool = False) -> bool:
    """Check if a model requires SDPA instead of Flash Attention 2.

    Returns True when the model has characteristics incompatible with
    Flash Attention 2:
    - M-RoPE (multimodal rotary position embeddings) producing 3D position_ids
    - A timm-based vision tower (TimmWrapperModel rejects flash_attention_2)

    Args:
        model_path: HuggingFace model ID or local path.
        trust_remote_code: Whether to trust remote code when loading config.

    Returns:
        True if the model should use SDPA attention.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    # Check M-RoPE
    if _config_has_mrope(config):
        return True

    return False


def has_timm_vision_tower(model_path: str, trust_remote_code: bool = False) -> bool:
    """Check if a model has a timm-based vision tower.

    timm vision towers only support ``eager`` attention, so the vision config
    must be patched to use eager while the text model can use FA2/SDPA.

    Args:
        model_path: HuggingFace model ID or local path.
        trust_remote_code: Whether to trust remote code when loading config.

    Returns:
        True if the model has a timm-based vision tower.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return False
    model_type = getattr(vision_config, "model_type", "")
    if model_type in ("timm_wrapper", "gemma3n_vision"):
        return True
    try:
        # Third Party
        from transformers.models.auto import MODEL_MAPPING

        if vision_config.__class__ in MODEL_MAPPING:
            vision_cls = MODEL_MAPPING[vision_config.__class__]
            if "Timm" in vision_cls.__name__:
                return True
    except Exception:
        pass
    return False
