"""Shared loading of GeoAE training checkpoints and frozen LMs.

Every tool that consumes a checkpoint (evaluation, interp suite, dbpedia
benchmarks) rebuilds the AE through `load_ae_checkpoint`, so architecture
fields added to the checkpoint config (nonlinearity, metric, …) are honoured
in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

import torch

from geoae.model import GeoAE


def load_ae_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | str,
) -> tuple[GeoAE, torch.Tensor, torch.Tensor, dict]:
    """
    Rebuild a GeoAE from a training checkpoint (base, e2e, or stream format —
    they share model_state / norm_mean / norm_std / config).

    Returns (ae, norm_mean, norm_std, ckpt): the AE in eval mode on `device`,
    the normalisation stats as float32 tensors on `device`, and the raw
    checkpoint dict for callers that need config / val metrics.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mc = ckpt["config"]["model"]
    ae = GeoAE(
        hidden_size=mc["hidden_size"],
        latent_dim=mc["latent_dim"],
        n_clusters=mc["n_clusters"],
        nonlinearity=mc.get("nonlinearity", "linear"),
        metric=mc.get("metric", "euclidean"),
    ).to(device)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()

    norm_mean = torch.as_tensor(ckpt["norm_mean"], dtype=torch.float32, device=device)
    norm_std = torch.as_tensor(ckpt["norm_std"], dtype=torch.float32, device=device)
    return ae, norm_mean, norm_std, ckpt


def ae_label(ckpt: dict) -> str:
    """Short human-readable description of a loaded checkpoint's architecture."""
    mc = ckpt["config"]["model"]
    nl = mc.get("nonlinearity", "linear")
    return (f"AE {mc['hidden_size']}→{mc['latent_dim']}→{mc['hidden_size']} "
            f"({nl}, K={mc['n_clusters']})")


def load_lm(
    model_name: str,
    device: torch.device | str | None = None,
    device_map: str | None = None,
):
    """
    Load a frozen bfloat16 causal LM for evaluation/extraction.

    Pass `device_map="auto"` for sharded multi-GPU placement, or `device` to
    put the whole model on one device. The model is set to eval mode with all
    parameters frozen.
    """
    from transformers import AutoModelForCausalLM

    kwargs = {"dtype": torch.bfloat16}
    if device_map is not None:
        kwargs["device_map"] = device_map
    lm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device is not None and device_map is None:
        lm = lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm
