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
        latent_norm=mc.get("latent_norm", "none"),
    ).to(device)
    # `sinkhorn_g` was added with the generalised balancing knobs; checkpoints
    # written before it are still loadable, but nothing else may be missing.
    missing, unexpected = ae.load_state_dict(ckpt["model_state"], strict=False)
    unknown = set(missing) - {"sinkhorn_g"}
    if unknown or unexpected:
        raise RuntimeError(
            f"checkpoint architecture mismatch: missing={sorted(unknown)} "
            f"unexpected={sorted(unexpected)}")
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


GATED_HINT = """\
`{name}` is a gated repository — accepted licence + HF token required.

  1. Accept the licence at https://huggingface.co/{name} (button at the top).
  2. Authenticate on this machine:
         hf auth login                 # paste a token from hf.co/settings/tokens
     or set HF_TOKEN=<token> in the environment.

Original error: {err}"""


def _drop_vision_tower(lm) -> bool:
    """
    Delete the vision encoder from a multimodal wrapper (Gemma 3 4b/12b/27b load
    as Gemma3ForConditionalGeneration under AutoModelForCausalLM).

    Every GeoAE path feeds text `input_ids` only, and the wrapper's forward
    touches the tower solely when `pixel_values` is passed — so the tower is
    ~0.9 GB of bf16 weights that can never run. Returns True if anything was
    dropped.
    """
    dropped = False
    for owner in (lm, getattr(lm, "model", None)):
        if owner is None:
            continue
        for attr in ("vision_tower", "multi_modal_projector"):
            if getattr(owner, attr, None) is not None:
                setattr(owner, attr, None)
                dropped = True
    return dropped


def load_lm(
    model_name: str,
    device: torch.device | str | None = None,
    device_map: str | None = None,
    text_only: bool = True,
):
    """
    Load a frozen bfloat16 causal LM for evaluation/extraction.

    Pass `device_map="auto"` for sharded multi-GPU placement, or `device` to
    put the whole model on one device. The model is set to eval mode with all
    parameters frozen.

    `text_only` drops the vision encoder of multimodal checkpoints (Gemma 3);
    set it False if a caller ever needs image inputs.
    """
    from transformers import AutoModelForCausalLM

    from geoae.hf_auth import ensure_hf_login
    ensure_hf_login()   # exports HF_TOKEN; no-op if already set. Gated repos need it.

    kwargs = {"dtype": torch.bfloat16}
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        lm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as e:
        msg = str(e)
        if ("GatedRepo" in type(e).__name__ or "gated" in msg.lower()
                or "401" in msg or "403" in msg):
            raise RuntimeError(GATED_HINT.format(name=model_name, err=msg)) from e
        raise
    if text_only and _drop_vision_tower(lm):
        print(f"[load_lm] {model_name}: dropped vision tower (text-only use)")
    if device is not None and device_map is None:
        lm = lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm
