"""
Architecture-agnostic access to a frozen LM's decoder stack, final norm, and head.

Every tool in this package hooks the residual stream at `layers[L]` and (for the
KL paths) re-applies `lm_head(final_norm(·))`. The module path to those parts is
NOT uniform across HF architectures:

    Llama / Qwen (text-only *ForCausalLM)
        lm.model.layers, lm.model.norm, lm.lm_head

    Gemma 3 4b/12b/27b (multimodal Gemma3ForConditionalGeneration — what
    `AutoModelForCausalLM` returns for model_type "gemma3")
        lm.model.language_model.layers, lm.model.language_model.norm, lm.lm_head
        (`lm.model` here is a Gemma3Model wrapper with NO .layers and NO .norm)

Hard-coding `lm.model.layers` therefore raises AttributeError on Gemma 3 and any
other vision-language wrapper. `lm.get_decoder()` returns the text decoder for
all of the above, so it is tried first; the explicit paths remain as fallbacks
for architectures that predate that API or return something unexpected.
"""
from __future__ import annotations

import torch.nn as nn

# Final-norm attribute names, in the order they are probed.
_NORM_ATTRS = ("norm", "final_layernorm", "ln_f")


def _has_layers(mod) -> bool:
    return mod is not None and isinstance(getattr(mod, "layers", None), nn.ModuleList)


def find_decoder(lm) -> nn.Module:
    """Return the module that owns the decoder-layer list (`.layers`)."""
    get_decoder = getattr(lm, "get_decoder", None)
    if callable(get_decoder):
        try:
            dec = get_decoder()
        except Exception:
            dec = None
        if _has_layers(dec):
            return dec

    base = getattr(lm, "model", None)
    if _has_layers(base):
        return base
    # Multimodal wrappers: lm.model.language_model (Gemma 3, transformers >= 4.52)
    if _has_layers(getattr(base, "language_model", None)):
        return base.language_model
    # Older multimodal layout: lm.language_model.model
    lang = getattr(lm, "language_model", None)
    if _has_layers(lang):
        return lang
    if _has_layers(getattr(lang, "model", None)):
        return lang.model

    raise AttributeError(
        f"Could not locate the decoder stack on {type(lm).__name__}: tried "
        "get_decoder(), .model, .model.language_model, .language_model[.model]. "
        "Unsupported architecture — add its path to geoae.lm_arch.find_decoder."
    )


def decoder_layers(lm) -> nn.ModuleList:
    """Return the decoder's `nn.ModuleList` of transformer blocks."""
    return find_decoder(lm).layers


def locate_lm_parts(lm):
    """
    Return (decoder, layers, final_norm, lm_head).

    `decoder` is the text-decoder module, `layers` its block list, `final_norm`
    the norm applied to the last residual before the head, and `lm_head` the
    unembedding. Raises AttributeError if any part cannot be found.
    """
    decoder = find_decoder(lm)

    final_norm = next(
        (getattr(decoder, a) for a in _NORM_ATTRS if getattr(decoder, a, None) is not None),
        None,
    )
    if final_norm is None:
        raise AttributeError(
            f"No final norm on {type(decoder).__name__} (tried {_NORM_ATTRS})."
        )

    lm_head = getattr(lm, "lm_head", None)
    if lm_head is None and callable(getattr(lm, "get_output_embeddings", None)):
        lm_head = lm.get_output_embeddings()
    if lm_head is None:
        raise AttributeError(
            f"No lm_head / output embeddings on {type(lm).__name__}."
        )

    return decoder, decoder.layers, final_norm, lm_head


def hidden_size(lm) -> int:
    """Residual-stream width of the text decoder (NOT the vision tower's)."""
    cfg = find_decoder(lm).config
    d = getattr(cfg, "hidden_size", None)
    if d is None:
        raise AttributeError(f"No hidden_size on {type(cfg).__name__}.")
    return int(d)


def describe(lm) -> str:
    """One-line summary used by extraction/training logs."""
    dec, layers, _, head = locate_lm_parts(lm)
    return (f"{type(lm).__name__} (decoder {type(dec).__name__}): "
            f"{len(layers)} layers, d={hidden_size(lm)}, "
            f"vocab={head.weight.shape[0]}")
