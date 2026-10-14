"""Forward hooks for splicing tensors into a frozen LM's residual stream."""
from __future__ import annotations

import inspect
from typing import Callable

from torch import Tensor

from geoae.lm_arch import decoder_layers


def install_padding_position_hook(model):
    """Keep GPT-2's absolute positions independent of batching/padding.

    HF generation prepares padding-aware positions, but ordinary GPT-2 forward
    calls default to physical sequence indices. GeoAE uses both for activation
    capture and causal interventions. Fill missing position_ids from a 2-D
    attention mask; retain explicitly supplied positions and slice the mask's
    full prefix down to the current input length on cached decoding steps.
    Returns a removable hook handle, or None for other architectures.
    """
    if getattr(getattr(model, "config", None), "model_type", None) != "gpt2":
        return None
    signature = inspect.signature(model.forward)
    position_arg = list(signature.parameters).index("position_ids")

    def pre_hook(module, args, kwargs):
        bound = signature.bind_partial(*args, **kwargs).arguments
        mask = bound.get("attention_mask")
        if bound.get("position_ids") is not None or mask is None or mask.ndim != 2:
            return None
        ids = bound.get("input_ids")
        embeds = bound.get("inputs_embeds")
        if ids is not None:
            length, device = ids.shape[-1], ids.device
        elif embeds is not None:
            length, device = embeds.shape[-2], embeds.device
        else:
            return None
        # A cached call needs the full-prefix mask to recover absolute token
        # positions. With a query-only mask, retain HF's cache-based default.
        if bound.get("past_key_values") is not None and mask.shape[-1] <= length:
            return None
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        positions = positions.masked_fill(mask == 0, 0)[:, -length:].to(device)
        if len(args) > position_arg:
            args = list(args)
            args[position_arg] = positions
            return tuple(args), kwargs
        return args, {**kwargs, "position_ids": positions}

    return model.register_forward_pre_hook(pre_hook, with_kwargs=True)


class SplicingHook:
    """
    Replaces the residual stream at a target layer with a provided tensor.

    Usage:
        hook = SplicingHook(model, layer_idx=27)
        hook.activate(replacement_fn)   # replacement_fn(original) -> modified
        logits = model(input_ids).logits
        hook.deactivate()

    replacement_fn receives the raw hidden-states tensor (B, T, D) and returns
    a tensor of the same shape to splice in.
    """

    def __init__(self, model, layer_idx: int):
        self._model = model
        self._layer_idx = layer_idx
        self._handle = None
        self._fn: Callable | None = None

    def activate(self, fn: Callable[[Tensor], Tensor]) -> None:
        self._fn = fn
        layers = decoder_layers(self._model)
        self._handle = layers[self._layer_idx].register_forward_hook(self._hook_fn)

    def deactivate(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._fn = None

    def _hook_fn(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        modified = self._fn(hs)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


class TokenIdTap:
    """
    Records the input_ids of the LM's current forward pass, for a token-bypass AE.

    A forward PRE-hook on the input embedding sees exactly the ids whose hidden
    states the layer hooks receive: (B, T) on a full forward, (B, 1) on a cached
    generation step. `ids_for(hs)` checks that and returns them flattened in the
    same row-major order the splice functions use for hs.reshape(B * T, D).

        tap = TokenIdTap(lm)
        hook.activate(neuronlens.make_z_gate(..., tap=tap))
        lm(input_ids)          # the splice reads tap.ids_for(hs) inside the hook
        tap.remove()
    """

    def __init__(self, model):
        self._ids = None
        emb = model.get_input_embeddings()
        self._handle = emb.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, module, args, kwargs):
        self._ids = args[0] if args else kwargs.get("input")

    def ids_for(self, hs: Tensor) -> Tensor:
        if self._ids is None:
            raise RuntimeError("TokenIdTap saw no forward pass (was the LM called with input_ids?)")
        B, T = hs.shape[:2]
        if tuple(self._ids.shape) != (B, T):
            raise RuntimeError(f"TokenIdTap ids {tuple(self._ids.shape)} do not match hidden states {(B, T)}")
        return self._ids.reshape(B * T).to(hs.device)

    def remove(self) -> None:
        self._handle.remove()
