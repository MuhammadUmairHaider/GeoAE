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
