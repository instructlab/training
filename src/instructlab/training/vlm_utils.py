# Standard
import logging

# Third Party
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedModel
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
)

logger = logging.getLogger("instructlab.training")


def is_vlm_with_causal_lm(model_path: str) -> bool:
    """Check if a model is a VLM that wraps a CausalLM text backbone.

    Returns True when the model's top-level config does NOT map to a CausalLM
    but its ``text_config`` does — meaning the model needs VLM extraction to
    obtain the trainable CausalLM sub-model.

    Models that are dual-registered (top-level config maps directly to CausalLM)
    return False because ``AutoModelForCausalLM`` can load them without
    extraction.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        True if the model is a VLM wrapper around a CausalLM text backbone.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # If the top-level config maps to CausalLM, no extraction needed.
    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        return False

    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return False

    return text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING


def is_vlm_for_direct_loading(model_path: str) -> bool:
    """Check if a model is a VLM that should be loaded directly for text-only training.

    This handles models where:
    - No CausalLM class exists (top-level config NOT in MODEL_FOR_CAUSAL_LM_MAPPING)
    - No extractable text backbone (text_config also NOT in MODEL_FOR_CAUSAL_LM_MAPPING)
    - But the model IS in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING (can be loaded as VLM)

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        True if the model should be loaded directly as a VLM for text-only training.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

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


def extract_causal_lm_from_vlm(model_path: str, load_kwargs: dict) -> "PreTrainedModel":  # noqa: F821
    """Load a VLM and extract its CausalLM text backbone.

    This is used when a model's config is not directly registered as a CausalLM
    but wraps one internally (e.g. Ministral-3-3B uses a VLM architecture
    around a Mistral CausalLM).

    Args:
        model_path: HuggingFace model ID or local path.
        load_kwargs: Keyword arguments for ``from_pretrained`` (must include
            ``pretrained_model_name_or_path``).

    Returns:
        A standalone ``PreTrainedModel`` CausalLM containing the extracted
        text backbone weights.
    """
    # Filter out quantization_config=None — passing None explicitly causes
    # an AttributeError in HF's FP8 auto-dequant code path.
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

    # Build a standalone CausalLM from the text_config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = config.text_config
    causal_lm_class = MODEL_FOR_CAUSAL_LM_MAPPING[text_config.__class__]
    text_model = causal_lm_class(text_config)

    # Transfer weights
    text_model.model = backbone
    text_model.lm_head = vlm.lm_head

    del vlm
    return text_model


def has_mrope(model_path: str) -> bool:
    """Check if a model uses M-RoPE (multimodal rotary position embeddings).

    Models with M-RoPE produce 3D position_ids that are incompatible with
    Flash Attention 2's packed-sequence detection. This checks both
    ``rope_scaling`` and ``rope_parameters`` on both the top-level config
    and ``text_config``.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        True if any rope configuration contains an ``mrope_section`` key.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
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


def needs_sdpa(model_path: str) -> bool:
    """Check if a model requires SDPA instead of Flash Attention 2.

    Returns True when the model has characteristics incompatible with
    Flash Attention 2:
    - M-RoPE (multimodal rotary position embeddings) producing 3D position_ids
    - A timm-based vision tower (TimmWrapperModel rejects flash_attention_2)

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        True if the model should use SDPA attention.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Check M-RoPE
    if _config_has_mrope(config):
        return True

    return False


def has_timm_vision_tower(model_path: str) -> bool:
    """Check if a model has a timm-based vision tower.

    timm vision towers only support ``eager`` attention, so the vision config
    must be patched to use eager while the text model can use FA2/SDPA.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        True if the model has a timm-based vision tower.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
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
