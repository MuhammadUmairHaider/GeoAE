"""
Phase 7: Causal validation via activation splicing.

Three experiments:
  1. Loss recovered: how much of the information lost by mean-ablation is
     recovered by splicing in the AE reconstruction.
  2. Per-cluster ablation: ablate each cluster's contribution and measure
     which tokens lose the most prediction quality → "what does cluster k mean".
  3. Ablation specificity (optional): Gini coefficient of per-token loss changes.

Usage:
    python evaluate.py --checkpoint checkpoints/best_val.pt --n_eval 500
    python evaluate.py --checkpoint checkpoints/best_val.pt --experiment loss_recovered
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM

from geoae.model import GeoAE


# Re-exported for backward compatibility — canonical homes are geoae.hooks
# and geoae.checkpoint.
from geoae.hooks import SplicingHook  # noqa: F401
from geoae.checkpoint import load_ae_checkpoint, load_lm


def load_ae_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[GeoAE, dict, dict]:
    """
    Load GeoAE from a training checkpoint.
    Returns (model, norm_params, cfg_dict).
    """
    model, norm_mean, norm_std, ckpt = load_ae_checkpoint(ckpt_path, device)
    return model, {"mean": norm_mean, "std": norm_std}, ckpt["config"]


# ---------------------------------------------------------------------------
# Per-token CE helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def ce_per_token(logits: Tensor, input_ids: Tensor) -> Tensor:
    """
    Cross-entropy loss per position (next-token prediction).
    logits: (1, T, V), input_ids: (1, T)
    Returns (T-1,) tensor (no loss for the last token, no target for the first).
    """
    shift_logits = logits[0, :-1]                 # (T-1, V)
    shift_labels = input_ids[0, 1:]               # (T-1,)
    return F.cross_entropy(shift_logits, shift_labels, reduction="none")


# ---------------------------------------------------------------------------
# Experiment 1: Loss recovered
# ---------------------------------------------------------------------------

@torch.no_grad()
def experiment_loss_recovered(
    lm: AutoModelForCausalLM,
    ae: GeoAE,
    norm_params: dict,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    device: torch.device,
) -> dict:
    """
    For each text:
      (a) original forward pass → CE_orig
      (b) mean-ablated (replace activation with its mean across the batch) → CE_ablated
      (c) AE-spliced (replace with AE reconstruction) → CE_spliced

    loss_recovered = (CE_ablated - CE_spliced) / (CE_ablated - CE_orig + 1e-8)

    Returns dict with per-text and aggregate stats.
    """
    hook = SplicingHook(lm, layer_idx)

    results = []
    mean_param = norm_params["mean"]    # (D,)
    std_param  = norm_params["std"]     # (D,)

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)

        # (a) Original
        out_orig = lm(input_ids=input_ids)
        ce_orig = ce_per_token(out_orig.logits, input_ids).mean().item()

        # (b) Mean ablation: replace with the per-dim mean of the activation
        #     We capture the activation first, then splice its mean.
        captured_act: list[Tensor] = []
        def capture_fn(hs: Tensor) -> Tensor:
            captured_act.append(hs.detach())
            return hs
        hook.activate(capture_fn)
        lm(input_ids=input_ids)
        hook.deactivate()
        act = captured_act[0]               # (1, T, D)

        act_mean = act.mean(dim=1, keepdim=True).expand_as(act)
        hook.activate(lambda hs, m=act_mean: m.to(hs.device))
        out_ablated = lm(input_ids=input_ids)
        hook.deactivate()
        ce_ablated = ce_per_token(out_ablated.logits, input_ids).mean().item()

        # (c) AE splice
        ae_device = next(ae.parameters()).device

        def ae_splice_fn(hs: Tensor, act=act) -> Tensor:
            B, T, D = hs.shape
            # Move to AE's device for computation; result goes back to hs.device
            flat = hs.reshape(B * T, D).float().to(ae_device)
            mean = mean_param.to(ae_device)
            std  = std_param.to(ae_device)
            flat_normed = (flat - mean) / std
            ae_out = ae(flat_normed)
            recon_raw = ae_out.x_hat * std + mean
            return recon_raw.reshape(B, T, D).to(hs.device).to(hs.dtype)

        hook.activate(ae_splice_fn)
        out_spliced = lm(input_ids=input_ids)
        hook.deactivate()
        ce_spliced = ce_per_token(out_spliced.logits, input_ids).mean().item()

        lr = (ce_ablated - ce_spliced) / (ce_ablated - ce_orig + 1e-8)
        results.append({
            "ce_orig": ce_orig,
            "ce_ablated": ce_ablated,
            "ce_spliced": ce_spliced,
            "loss_recovered": lr,
        })

    avg_lr = np.mean([r["loss_recovered"] for r in results])
    print(f"[eval] Loss recovered (mean over {len(texts)} texts): {avg_lr:.4f}")
    if avg_lr < 0.7:
        print("[eval] WARNING: loss_recovered < 0.7 — AE is not preserving enough structure")
    elif avg_lr >= 0.95:
        print("[eval] Excellent: loss_recovered >= 0.95")
    elif avg_lr >= 0.8:
        print("[eval] Good: loss_recovered >= 0.8")

    return {"per_text": results, "mean_loss_recovered": avg_lr}


# ---------------------------------------------------------------------------
# Experiment 2: Per-cluster ablation selectivity
# ---------------------------------------------------------------------------

@torch.no_grad()
def experiment_per_cluster_ablation(
    lm: AutoModelForCausalLM,
    ae: GeoAE,
    norm_params: dict,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    device: torch.device,
    top_k_tokens: int = 10,
) -> dict:
    """
    For each cluster k:
      - Ablate its contribution by subtracting Q[:, k:k+1] * c[k] from z.
      - Decode modified z, splice into forward pass.
      - Record per-token CE change vs original.

    Returns per-cluster: top affected tokens + their CE changes.
    """
    hook = SplicingHook(lm, layer_idx)
    mean_p = norm_params["mean"]
    std_p  = norm_params["std"]
    K = ae.n_clusters

    # Accumulate per-token CE changes across all texts, per cluster
    # token_changes[k] = list of (token_id, ce_change)
    token_changes: dict[int, list[tuple[int, float]]] = {k: [] for k in range(K)}

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(device)
        T = input_ids.shape[1]
        if T < 2:
            continue

        # Original per-token CE
        out_orig = lm(input_ids=input_ids)
        ce_orig = ce_per_token(out_orig.logits, input_ids)    # (T-1,)

        # Capture activation + latents once
        captured_hs: list[Tensor] = []
        hook.activate(lambda hs: (captured_hs.append(hs.detach()) or hs))
        lm(input_ids=input_ids)
        hook.deactivate()
        hs = captured_hs[0]                           # (1, T, D)
        hs_device = hs.device
        ae_device = next(ae.parameters()).device

        flat = hs.reshape(T, -1).float().to(ae_device)
        mean = mean_p.to(ae_device)
        std  = std_p.to(ae_device)
        flat_normed = (flat - mean) / std             # (T, D)
        ae_out = ae(flat_normed)
        z   = ae_out.z                                # (T, L)  on ae_device
        Q   = ae_out.Q                                # (T, K)  on ae_device

        centroids = ae.centroids                      # (K, L)  on ae_device

        for k in range(K):
            # Remove cluster k contribution from z
            qk = Q[:, k:k+1]                         # (T, 1)
            ck = centroids[k:k+1]                     # (1, L)
            z_ablated = z - qk * ck                   # (T, L)

            # Decode and splice
            recon_ablated = ae.decoder(z_ablated)                          # (T, D)
            recon_raw = recon_ablated * std + mean                         # (T, D)
            recon_4d = recon_raw.reshape(1, T, -1).to(hs_device).to(hs.dtype)

            hook.activate(lambda hs, r=recon_4d: r.to(hs.device))
            out_ablated = lm(input_ids=input_ids)
            hook.deactivate()

            ce_ablated_k = ce_per_token(out_ablated.logits, input_ids)   # (T-1,)
            ce_delta = (ce_ablated_k - ce_orig).cpu()                     # (T-1,)

            # Map to actual token ids (target tokens are input_ids[1:])
            target_ids = input_ids[0, 1:].cpu().tolist()
            for tok_id, delta in zip(target_ids, ce_delta.tolist()):
                token_changes[k].append((tok_id, delta))

    # Summarise: top_k_tokens most affected per cluster
    summary = {}
    for k in range(K):
        if not token_changes[k]:
            summary[k] = {"top_tokens": [], "mean_ce_change": 0.0, "gini": 0.0}
            continue

        changes = token_changes[k]
        sorted_changes = sorted(changes, key=lambda x: x[1], reverse=True)
        top = sorted_changes[:top_k_tokens]
        top_decoded = [(tokenizer.decode([tid]), delta) for tid, delta in top]

        all_deltas = np.array([d for _, d in changes])
        mean_change = float(all_deltas.mean())
        gini = _gini(np.abs(all_deltas))

        summary[k] = {
            "top_tokens": top_decoded,
            "mean_ce_change": mean_change,
            "gini": gini,
        }

    return summary


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    arr = arr.flatten()
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Splice sanity check (must pass before reporting any results)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sanity_check_splice(
    lm: AutoModelForCausalLM,
    layer_idx: int,
    tokenizer,
    device: torch.device,
    text: str = "The quick brown fox jumps over the lazy dog.",
) -> bool:
    """
    Splice in the ORIGINAL activation (identity operation) and verify
    logits are bit-identical (within fp16 rounding) to a no-hook pass.

    Returns True if the check passes.
    """
    hook = SplicingHook(lm, layer_idx)

    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    # No-hook baseline
    out_clean = lm(input_ids=input_ids)
    logits_clean = out_clean.logits.float()

    # Identity splice
    hook.activate(lambda hs: hs)
    out_spliced = lm(input_ids=input_ids)
    hook.deactivate()
    logits_spliced = out_spliced.logits.float()

    max_diff = (logits_clean - logits_spliced).abs().max().item()
    passed = max_diff < 1e-3

    if passed:
        print(f"[eval] Splice sanity check PASSED (max logit diff = {max_diff:.2e})")
    else:
        print(f"[eval] Splice sanity check FAILED (max logit diff = {max_diff:.2e}) — hook is broken!")

    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "sanity", "loss_recovered", "per_cluster"])
    parser.add_argument("--n_eval", type=int, default=200, help="Number of eval texts")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--out", default=None, help="Write JSON results to this path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from geoae.seeding import seed_everything
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)

    ae, norm_params, cfg_dict = load_ae_from_checkpoint(ckpt_path, device)
    model_name = cfg_dict["extraction"]["model_name"]

    print(f"[eval] Loading LM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = load_lm(model_name, device_map="auto")

    # Splice sanity check — always run first
    ok = sanity_check_splice(lm, args.layer, tokenizer, device)
    if not ok:
        print("[eval] Aborting — fix the splicing hook before running experiments.")
        return

    if args.experiment in ("all", "sanity"):
        if args.experiment == "sanity":
            return

    # Load eval texts (use a small slice of C4 validation)
    print(f"[eval] Loading {args.n_eval} eval texts …")
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", name="en", split="validation", streaming=True)
        texts = [ex["text"] for ex in ds.take(args.n_eval)]
    except Exception as e:
        print(f"[eval] Could not load eval texts from HF ({e}). Using hardcoded fallback.")
        texts = ["The cat sat on the mat. " * 20] * min(args.n_eval, 10)

    results = {}

    if args.experiment in ("all", "loss_recovered"):
        print("\n--- Experiment 1: Loss Recovered ---")
        results["loss_recovered"] = experiment_loss_recovered(
            lm, ae, norm_params, tokenizer, texts, args.layer, device
        )

    if args.experiment in ("all", "per_cluster"):
        print("\n--- Experiment 2: Per-cluster Ablation ---")
        cluster_summary = experiment_per_cluster_ablation(
            lm, ae, norm_params, tokenizer, texts[:min(50, len(texts))],
            args.layer, device
        )
        results["per_cluster"] = cluster_summary

        # Print top clusters by mean CE change
        ranked = sorted(cluster_summary.items(), key=lambda x: x[1]["mean_ce_change"], reverse=True)
        print("\nTop 10 clusters by mean CE change:")
        for k, info in ranked[:10]:
            top_toks = [f"{tok!r}({d:.3f})" for tok, d in info["top_tokens"][:3]]
            print(f"  cluster {k:3d}: mean_Δce={info['mean_ce_change']:.4f} "
                  f"gini={info['gini']:.3f}  top_tokens=[{', '.join(top_toks)}]")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[eval] Results written to {args.out}")


if __name__ == "__main__":
    main()
