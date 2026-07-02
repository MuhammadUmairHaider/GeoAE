"""
Direct Logit Attribution (DLA) for GeoAE cluster centroids.

For each centroid c_k, this measures the cluster's effect on the LM's output
*vocabulary* distribution by projecting the decoder direction through the
unembedding matrix (the "logit lens" / direct logit attribution used throughout
the SAE-interpretability literature). No forward passes and no data are needed —
it reads only the AE weights and the LM's unembedding.

Pipeline per cluster k:
  1. decoder(c_k)            -> direction in NORMALISED activation space   (D,)
  2. * norm_std              -> direction in RAW residual-stream space     (D,)
  3. * final_norm.weight     -> apply the final RMSNorm gain               (D,)
  4. @ W_U.T                 -> per-vocab logit contribution               (V,)
  5. - mean over vocab       -> centred: promoted (>0) vs suppressed (<0)

Why step 2 (the std multiply): the reconstruction actually spliced back into the
model is `x_hat * std + mean`. Because the decoder is linear and bias-free, the
marginal contribution of cluster k to that raw residual is
`q_k * decoder(c_k) * std`, so `decoder(c_k) * std` is the per-unit direction the
cluster injects into the residual stream.

Exactness: when the AE's target layer is the LAST transformer block (true for
layer 27 of Llama-3.2-3B, which has 28 layers), there are no intervening layers
between the splice point and the unembedding, so this attribution is EXACT up to
a single positive per-cluster scalar (the typical assignment q_k times the
RMSNorm rsqrt factor). That scalar scales every vocab entry of a cluster equally,
so it never changes *which* tokens a cluster promotes — only cross-cluster
magnitude comparisons (reported as `logit_norm`) carry that caveat. For earlier
layers this becomes the usual logit-lens approximation.

Usage:
    python -m geoae.interp.logit_attribution \
      --checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/best_val.pt \
      --top_k 15 \
      --out e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/dla.json

    # Compare DLA (causal, output-space) against the geometric closest tokens:
    python -m geoae.interp.logit_attribution \
      --checkpoint .../best_val.pt \
      --closest_tokens .../closest_tokens.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoTokenizer

from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.model import GeoAE


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_ae_and_norm(ckpt_path: Path, device: torch.device) -> tuple[GeoAE, Tensor, Tensor, dict]:
    print(f"[dla] Loading AE checkpoint: {ckpt_path}")
    ae, norm_mean, norm_std, ckpt = load_ae_checkpoint(ckpt_path, device)
    return ae, norm_mean, norm_std, ckpt["config"]


# ---------------------------------------------------------------------------
# Pull the unembedding + final norm out of the LM (the only LM weights DLA needs)
# ---------------------------------------------------------------------------

def get_unembedding(lm) -> tuple[Tensor, Tensor | None, int]:
    """Returns (W_U [V, D] float32, final_norm_weight [D] float32 or None, n_layers)."""
    W_U = lm.lm_head.weight.detach().float()                      # (V, D)
    final_norm = getattr(getattr(lm, "model", lm), "norm", None)
    norm_w = final_norm.weight.detach().float() if final_norm is not None else None
    n_layers = lm.config.num_hidden_layers
    return W_U, norm_w, n_layers


# ---------------------------------------------------------------------------
# Core DLA
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dla(
    ae: GeoAE,
    norm_std: Tensor,
    W_U: Tensor,
    final_norm_w: Tensor | None,
    apply_final_norm: bool,
) -> Tensor:
    """
    Returns centred logit-attribution matrix (K, V): row k = cluster k's effect on
    each vocab logit, centred so 0 = the cluster's average effect over the vocab.
    """
    device = W_U.device
    centroids = ae.centroids.detach().float().to(device)          # (K, L)

    # 1. centroid -> normalised activation space (decoder is linear, bias-free)
    directions = ae.decoder(centroids.to(next(ae.parameters()).device)).float().to(device)  # (K, D)

    # 2. -> raw residual-stream space
    directions = directions * norm_std.to(device)                 # (K, D)

    # 3. final RMSNorm gain (per-dim; the rsqrt scalar is omitted — see module docstring)
    if apply_final_norm and final_norm_w is not None:
        directions = directions * final_norm_w.to(device)         # (K, D)

    # 4. -> vocab logits
    logits = directions @ W_U.T                                   # (K, V)

    # 5. centre over vocab (a constant shift never changes the softmax)
    logits = logits - logits.mean(dim=1, keepdim=True)
    return logits


@torch.no_grad()
def find_duplicate_pairs(logits: Tensor, threshold: float, max_pairs: int = 50) -> list[tuple[int, int, float]]:
    """Cluster pairs whose centred logit-attribution vectors are near-identical
    (cosine >= threshold) — i.e. functionally redundant in output space."""
    norm = torch.nn.functional.normalize(logits, dim=1)           # (K, V)
    sim = norm @ norm.T                                           # (K, K)
    K = sim.shape[0]
    sim.fill_diagonal_(-1.0)
    iu, ju = torch.triu_indices(K, K, offset=1)
    vals = sim[iu, ju]
    mask = vals >= threshold
    pairs = [(int(iu[m]), int(ju[m]), float(vals[m])) for m in mask.nonzero(as_tuple=True)[0].tolist()]
    pairs.sort(key=lambda p: -p[2])
    return pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def topk_tokens(logits_row: Tensor, tokenizer, top_k: int) -> tuple[list, list]:
    """Returns (promoted, suppressed) as lists of [token_str, score]."""
    promoted_vals, promoted_idx = logits_row.topk(top_k, largest=True)
    suppressed_vals, suppressed_idx = logits_row.topk(top_k, largest=False)
    promoted = [[tokenizer.decode([t]), round(v, 4)]
                for t, v in zip(promoted_idx.tolist(), promoted_vals.tolist())]
    suppressed = [[tokenizer.decode([t]), round(v, 4)]
                  for t, v in zip(suppressed_idx.tolist(), suppressed_vals.tolist())]
    return promoted, suppressed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Direct Logit Attribution for GeoAE centroids.")
    parser.add_argument("--checkpoint", required=True, help="Path to AE .pt checkpoint")
    parser.add_argument("--top_k", type=int, default=15, help="Promoted/suppressed tokens per cluster")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <ckpt_dir>/dla.json)")
    parser.add_argument("--print_top_n", type=int, default=10,
                        help="Clusters to preview, ranked by logit-attribution strength")
    parser.add_argument("--no_final_norm", action="store_true",
                        help="Skip the final-RMSNorm gain (raw residual -> unembedding)")
    parser.add_argument("--dup_threshold", type=float, default=0.9,
                        help="Report cluster pairs with logit-attribution cosine >= this")
    parser.add_argument("--closest_tokens", default=None,
                        help="Optional closest_tokens.json to show geometric tokens side-by-side")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from geoae.seeding import seed_everything
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dla] Device: {device}")

    ckpt_path = Path(args.checkpoint)
    ae, norm_mean, norm_std, cfg = load_ae_and_norm(ckpt_path, device)
    K = ae.n_clusters
    target_layer = cfg["data"]["target_layer"]
    model_name = cfg["extraction"]["model_name"]

    # Only the unembedding + final norm are needed; load on CPU to spare VRAM.
    print(f"[dla] Loading LM weights (unembedding + final norm): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = load_lm(model_name)
    W_U, final_norm_w, n_layers = get_unembedding(lm)
    W_U = W_U.to(device)
    if final_norm_w is not None:
        final_norm_w = final_norm_w.to(device)
    del lm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    exact = (target_layer == n_layers - 1)
    apply_final_norm = not args.no_final_norm
    print(f"[dla] target_layer={target_layer}  n_layers={n_layers}  "
          f"{'EXACT (last layer)' if exact else 'logit-lens APPROXIMATION'}")
    print(f"[dla] final RMSNorm gain: {'applied' if (apply_final_norm and final_norm_w is not None) else 'skipped'}")

    logits = compute_dla(ae, norm_std, W_U, final_norm_w, apply_final_norm)   # (K, V)
    logit_norm = logits.norm(dim=1)                                          # (K,) effect strength

    # Optional geometric reference
    closest = None
    if args.closest_tokens:
        cp = Path(args.closest_tokens)
        if cp.exists():
            with open(cp) as f:
                raw = json.load(f)
            # Accept both schemas: v1 = {cluster: [items]}, v2 = {"clusters": {cluster: {...}}}.
            clusters_raw = raw.get("clusters", raw)
            # Map cluster -> list of geometric token strings (dedup, order preserved)
            closest = {}
            for k_str, entry in clusters_raw.items():
                if not str(k_str).lstrip("-").isdigit():
                    continue
                items = entry.get("top", []) if isinstance(entry, dict) else entry
                seen, toks = set(), []
                for it in items:
                    t = it.get("token")
                    if t is not None and t not in seen:
                        seen.add(t); toks.append(t)
                closest[int(k_str)] = toks
            print(f"[dla] Loaded geometric closest tokens for {len(closest)} clusters from {cp}")
        else:
            print(f"[dla] WARNING: --closest_tokens path not found: {cp}")

    # Build per-cluster records
    clusters = {}
    for k in range(K):
        promoted, suppressed = topk_tokens(logits[k], tokenizer, args.top_k)
        rec = {
            "logit_norm": round(float(logit_norm[k]), 4),
            "promoted": promoted,
            "suppressed": suppressed,
        }
        if closest is not None:
            rec["closest_geometric"] = closest.get(k, [])[:args.top_k]
        clusters[k] = rec

    dup_pairs = find_duplicate_pairs(logits, args.dup_threshold)

    out_path = Path(args.out) if args.out else ckpt_path.parent / "dla.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "checkpoint": str(ckpt_path),
            "model_name": model_name,
            "target_layer": target_layer,
            "n_layers": n_layers,
            "exact": exact,
            "n_clusters": K,
            "vocab_size": int(W_U.shape[0]),
            "applied_final_norm": bool(apply_final_norm and final_norm_w is not None),
            "top_k": args.top_k,
        },
        "clusters": {str(k): v for k, v in clusters.items()},
        "duplicate_pairs": [
            {"i": i, "j": j, "logit_cosine": round(c, 4),
             "i_promoted": [t for t, _ in clusters[i]["promoted"][:5]],
             "j_promoted": [t for t, _ in clusters[j]["promoted"][:5]]}
            for i, j, c in dup_pairs
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[dla] Saved -> {out_path}")

    # ---- Preview: strongest clusters by logit-attribution norm ----
    ranked = sorted(range(K), key=lambda k: -logit_norm[k].item())
    print(f"\n{'='*78}")
    print(f"  Top {args.print_top_n} clusters by logit-attribution strength")
    print(f"{'='*78}")
    for k in ranked[:args.print_top_n]:
        rec = clusters[k]
        prom = ", ".join(f"{t!r}" for t, _ in rec["promoted"][:8])
        supp = ", ".join(f"{t!r}" for t, _ in rec["suppressed"][:5])
        print(f"\ncluster {k:4d}  (‖logit‖={rec['logit_norm']:.2f})")
        print(f"  promotes : {prom}")
        print(f"  suppresses: {supp}")
        if closest is not None:
            geo = ", ".join(f"{t!r}" for t in rec.get("closest_geometric", [])[:8])
            print(f"  geometric : {geo}")

    # ---- Functional duplicates ----
    print(f"\n{'='*78}")
    print(f"  Functionally redundant cluster pairs (logit cosine >= {args.dup_threshold})")
    print(f"{'='*78}")
    if not dup_pairs:
        print("  none — no clusters share a near-identical output-space effect")
    else:
        print(f"  {len(dup_pairs)} pair(s) (showing up to 50):")
        for i, j, c in dup_pairs:
            shared = ", ".join(f"{t!r}" for t, _ in clusters[i]["promoted"][:4])
            print(f"  cluster {i:4d} ~ {j:4d}  cos={c:.3f}  promotes=[{shared}]")

    print(f"\n[dla] {sum(1 for k in range(K) if logit_norm[k] > 0)} clusters analysed; "
          f"full per-cluster promoted/suppressed lists in {out_path}")


if __name__ == "__main__":
    main()
