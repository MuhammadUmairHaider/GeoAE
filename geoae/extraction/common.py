"""Residual capture and safe document pooling shared by labelled extractors."""
from __future__ import annotations

import numpy as np
import torch

from geoae.lm_arch import decoder_layers


def register_hook(model, layer: int) -> tuple[list, object]:
    """Capture detached CPU block outputs; return the list and removable handle."""
    captured = []

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured.append(hs.detach().cpu())

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    return captured, handle


def output_dtype(model, dtype="auto") -> np.dtype:
    """Gemma residuals can exceed FP16 range; preserve them without clipping."""
    if dtype == "auto":
        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        is_gemma = "gemma" in model_type or "gemma" in type(model).__name__.lower()
        dtype = "float32" if is_gemma else "float16"
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("activation dtype must be auto, float16, or float32")
    return resolved


def pool(hs: torch.Tensor, pooling: str, dtype=np.float16) -> np.ndarray:
    """Reduce a (T, D) sequence to a document vector, rejecting storage overflow."""
    if pooling == "last":
        vec = hs[-1]
    elif pooling == "mean":
        vec = hs.float().mean(dim=0)
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    values = vec.float().numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite model activations in {pooling} pooling")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = values.astype(dtype)
    if not np.isfinite(stored).all():
        raise ValueError(f"Activation overflow storing {pooling} pooling as {np.dtype(dtype)}; "
                         "use --dtype float32")
    return stored
