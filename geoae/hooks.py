"""Forward hooks for splicing tensors into a frozen LM's residual stream."""
from __future__ import annotations

from typing import Callable

from torch import Tensor

from geoae.lm_arch import decoder_layers


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
