"""Shared loading of GeoAE training checkpoints and frozen LMs.

Every tool that consumes a checkpoint (evaluation, interp suite, dbpedia
benchmarks) rebuilds the AE through `load_ae_checkpoint`, so architecture
fields added to the checkpoint config (nonlinearity, metric, …) are honoured
in exactly one place.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from geoae.model import GeoAE


def load_ae_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | str,
    allow_token_bias: bool = False,
) -> tuple[GeoAE, torch.Tensor, torch.Tensor, dict]:
    """
    Rebuild a GeoAE from a training checkpoint (base, e2e, or stream format —
    they share model_state / norm_mean / norm_std / config).

    A token-bias checkpoint (model.token_bias) needs the current token ids to
    decode and, in bypass mode, encode. Tools written against `ae.encoder` / `ae.decoder` would
    silently drop the bias, so they are refused unless the caller passes
    allow_token_bias=True — which asserts it uses ae.encode / ae.decode with ids.

    Returns (ae, norm_mean, norm_std, ckpt): the AE in eval mode on `device`,
    the normalisation stats as float32 tensors on `device`, and the raw
    checkpoint dict for callers that need config / val metrics.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mc = ckpt["config"]["model"]
    tb_rows = tb_vocab = 0
    if "tb_table" in ckpt["model_state"]:
        if not allow_token_bias:
            raise SystemExit(
                f"{ckpt_path} is a TOKEN-BYPASS AE (model.token_bias={mc.get('token_bias')!r}): "
                f"token-aware encoding/decoding is required. This tool has not been converted to "
                f"ae.encode(x, tok) / ae.decode(z, tok); it would silently drop the bias.")
        tb_rows = ckpt["model_state"]["tb_table"].shape[0]
        tb_vocab = ckpt["model_state"]["tb_index"].shape[0]
    lc = ckpt["config"].get("loss", {})
    tc = ckpt["config"].get("train", {})

    # ------------------------------------------------------------------
    # Restore the ASSIGNMENT rule, not just the architecture.
    #
    # These fields used to be left at GeoAE's constructor defaults -- uniform
    # balancing, rho=1.0, eta=1.0, tau=1.0, ema_hard=False -- no matter what the
    # run actually trained with. A checkpoint trained with balance="zipf",
    # rho=0.6, eta=0.05 and tau annealed to 0.1 therefore reloaded with a
    # DIFFERENT forward(): ~14% of `out.Q` labels changed. Evaluations that use
    # nearest-centroid (`dist2.argmin`) were unaffected -- which is every eval in
    # geoae/interp except clustering_quality's assignment-entropy row -- but
    # anything reading `out.Q` off a reloaded model was not measuring the
    # trained rule.
    #
    # tau is annealed per epoch and saved at the top level of the checkpoint, so
    # prefer that over recomputing. zipf_alpha is a plain float set per epoch
    # (train.py: model.zipf_alpha = zipf_alpha_schedule(epoch, cfg)) and is NOT
    # in state_dict, so recompute it from the saved epoch.
    # ------------------------------------------------------------------
    tau = ckpt.get("tau")
    if tau is None:
        tau = lc.get("tau_end", lc.get("tau_start", 1.0))

    zipf_alpha = 0.0
    if lc.get("balance") == "zipf":
        epoch = ckpt.get("epoch")
        a0 = lc.get("zipf_alpha_start", 0.0)
        a1 = lc.get("zipf_alpha_end", 0.0)
        start = tc.get("full_loss_start_epoch", 0)
        n_ep = tc.get("n_epochs", 1)
        if epoch is None or epoch < start:
            zipf_alpha = a0
        else:
            prog = (epoch - start) / max(n_ep - start, 1)
            zipf_alpha = a0 + (a1 - a0) * min(prog, 1.0)

    ae = GeoAE(
        hidden_size=mc["hidden_size"],
        latent_dim=mc["latent_dim"],
        n_clusters=mc["n_clusters"],
        nonlinearity=mc.get("nonlinearity", "linear"),
        metric=mc.get("metric", "euclidean"),
        latent_norm=mc.get("latent_norm", "none"),
        decoder_type=mc.get("decoder_type", "linear"),
        decoder_hidden_dim=mc.get("decoder_hidden_dim", 0),
        decoder_nonlinearity=mc.get("decoder_nonlinearity", "gelu"),
        decoder_output_bias=mc.get("decoder_output_bias", False),
        dist_scale=mc.get("dist_scale", "raw"),
        # --- assignment rule, restored from the checkpoint ---
        sinkhorn_iters=lc.get("sinkhorn_iters", 3),
        balance=lc.get("balance", "uniform"),
        balance_rho=lc.get("balance_rho", 1.0),
        balance_eta=lc.get("balance_eta", 1.0),
        zipf_alpha=zipf_alpha,
        tau=tau,
        ema_decay=tc.get("ema_decay", 0.99),
        ema_hard=tc.get("ema_hard", False),
        token_bias_rows=tb_rows,
        vocab_size=tb_vocab,
        token_bias_mode=mc.get("token_bias_mode", "bypass"),
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
    decoder = mc.get("decoder_type", "linear")
    extra = "" if decoder == "linear" else (
        f", decoder={decoder}/{mc.get('decoder_nonlinearity', 'gelu')}"
        f"/h{mc.get('decoder_hidden_dim', 0) or ('auto')}")
    return (f"AE {mc['hidden_size']}→{mc['latent_dim']}→{mc['hidden_size']} "
            f"({nl}{extra}, K={mc['n_clusters']})")


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
    dropped. Remove modules from their owning object: newer Transformers
    wrappers expose read-only properties forwarding to ``lm.model``.
    """
    dropped = False
    for owner in (lm, getattr(lm, "model", None)):
        if owner is None:
            continue
        for attr in ("vision_tower", "multi_modal_projector"):
            # A property on the outer wrapper is an alias, not the module's
            # owner. Assigning that property fails on Gemma3 in Transformers
            # 4.57. Remove the registered child on `model` instead, while
            # retaining support for older wrappers with direct attributes.
            if attr not in vars(owner) and attr not in getattr(owner, "_modules", {}):
                continue
            if getattr(owner, attr, None) is not None:
                setattr(owner, attr, None)
                dropped = True
    return dropped


def _local_lm_snapshot(model_name: str) -> str | None:
    """Resolve an explicitly supplied offline snapshot, preserving model IDs.

    Recovery jobs may stage a verified snapshot on their compute node rather
    than reread shared-cache shards. Validate every entry before using the
    optional environment mapping so a bad override cannot silently fall back
    to a different model or to downloading weights.
    """
    raw = os.environ.get("GEOAE_LM_SNAPSHOTS_JSON")
    if raw is None:
        return None
    try:
        snapshots = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("GEOAE_LM_SNAPSHOTS_JSON must contain a JSON object") from exc
    if not isinstance(snapshots, dict):
        raise ValueError("GEOAE_LM_SNAPSHOTS_JSON must contain a JSON object")
    for name, directory in snapshots.items():
        if not isinstance(name, str) or not name or not isinstance(directory, str):
            raise ValueError("GEOAE_LM_SNAPSHOTS_JSON entries must map model IDs to directory strings")
        path = Path(directory)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError(
                f"GEOAE_LM_SNAPSHOTS_JSON entry for {name!r} must be an absolute existing directory: "
                f"{directory!r}")
    return snapshots.get(model_name)


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

    `GEOAE_LM_SNAPSHOTS_JSON` optionally maps model IDs to absolute local
    snapshot directories; matching entries load with `local_files_only=True`.
    """
    from transformers import AutoModelForCausalLM

    snapshot = _local_lm_snapshot(model_name)
    from geoae.hf_auth import ensure_hf_login
    ensure_hf_login()   # exports HF_TOKEN; no-op if already set. Gated repos need it.

    kwargs = {"dtype": torch.bfloat16}
    if snapshot is not None:
        kwargs["local_files_only"] = True
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        lm = AutoModelForCausalLM.from_pretrained(snapshot or model_name, **kwargs)
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
    from geoae.hooks import install_padding_position_hook
    install_padding_position_hook(lm)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm
