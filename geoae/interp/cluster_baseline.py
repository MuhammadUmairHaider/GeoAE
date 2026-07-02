"""
Raw k-means baseline: clusters residual stream activations directly.

No AE, no learned transformation. Answers the question:
  "Does training a GeoAE produce better clusters than just running
   k-means on the raw representation?"

Evaluation uses the same causal ablation logic as evaluate.py:
  for cluster k, subtract Q[:,k:k+1] * c[k] directly from the hidden state
  (no encoder/decoder — the raw space IS the latent space).

Usage:
    python -m geoae.interp.cluster_baseline fit   --layer 27 --n_clusters 128
    python -m geoae.interp.cluster_baseline eval  --layer 27 --n_clusters 128
    python -m geoae.interp.cluster_baseline both  --layer 27 --n_clusters 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.hooks import SplicingHook

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_kmeans(
    activations_dir: Path,
    layer: int,
    n_clusters: int,
    n_sample: int,
    seed: int,
    out_path: Path,
) -> np.ndarray:
    """
    Fit MiniBatchKMeans on a sample of raw activations.
    Returns centroids array (n_clusters, hidden_size).
    """
    from sklearn.cluster import MiniBatchKMeans

    npy_path = activations_dir / f"layer_{layer}.npy"
    norm_path = activations_dir / f"norm_params_layer{layer}.npz"

    mmap = np.load(str(npy_path), mmap_mode="r")
    N = mmap.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=min(n_sample, N), replace=False)
    idx.sort()  # sorted for sequential mmap reads

    print(f"[baseline] Loading {len(idx):,} samples from layer {layer} …")
    sample = mmap[idx].astype(np.float32)

    # Apply normalisation so comparison is fair (AE trains on normalised activations)
    if norm_path.exists():
        d = np.load(str(norm_path))
        mean = d["mean"].astype(np.float32)
        std  = d["std"].astype(np.float32)
        sample = (sample - mean) / std
        print(f"[baseline] Normalisation applied (mean/std from {norm_path.name})")
    else:
        print("[baseline] WARNING: no norm cache found; clustering raw (unnormalised) activations")
        mean = np.zeros(mmap.shape[1], np.float32)
        std  = np.ones(mmap.shape[1], np.float32)

    print(f"[baseline] Fitting MiniBatchKMeans  K={n_clusters} …")
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init=5,
        max_iter=300,
        verbose=1,
    )
    km.fit(sample)
    centroids = km.cluster_centers_.astype(np.float32)   # (K, D) in normalised space

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path),
             centroids=centroids,
             norm_mean=mean,
             norm_std=std,
             layer=layer,
             n_clusters=n_clusters)
    print(f"[baseline] Saved {out_path}")
    return centroids


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def ce_per_token(logits: Tensor, input_ids: Tensor) -> Tensor:
    return F.cross_entropy(logits[0, :-1], input_ids[0, 1:], reduction="none")


# ---------------------------------------------------------------------------
# Per-cluster ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_kmeans_clusters(
    lm,
    tokenizer,
    centroids_path: Path,
    texts: list[str],
    layer_idx: int,
    device: torch.device,
    top_k_tokens: int = 10,
) -> dict:
    data = np.load(str(centroids_path))
    centroids_np = data["centroids"]                     # (K, D) normalised
    norm_mean    = torch.from_numpy(data["norm_mean"])   # (D,)
    norm_std     = torch.from_numpy(data["norm_std"])    # (D,)
    K, D = centroids_np.shape
    centroids    = torch.from_numpy(centroids_np)        # (K, D)

    hook = SplicingHook(lm, layer_idx)
    token_changes: dict[int, list[tuple[int, float]]] = {k: [] for k in range(K)}

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(device)
        T = input_ids.shape[1]
        if T < 2:
            continue

        # Original CE
        out_orig = lm(input_ids=input_ids)
        ce_orig = ce_per_token(out_orig.logits, input_ids)   # (T-1,)

        # Capture raw hidden states at this layer
        captured = []
        hook.activate(lambda hs: (captured.append(hs.detach()) or hs))
        lm(input_ids=input_ids)
        hook.deactivate()

        hs = captured[0]                                     # (1, T, D) in model dtype
        hs_dev = hs.device

        # Normalise to match centroid space
        flat = hs[0].float()                                 # (T, D)
        mean = norm_mean.to(hs_dev)
        std  = norm_std.to(hs_dev)
        flat_normed = (flat - mean) / std                    # (T, D)

        # Hard k-means assignment
        c = centroids.to(hs_dev)                             # (K, D)
        dist2 = torch.cdist(flat_normed, c).pow(2)          # (T, K)
        assign = dist2.argmin(dim=1)                         # (T,)  hard assignment

        for k in range(K):
            mask = (assign == k).float().unsqueeze(1)       # (T, 1)
            if mask.sum() == 0:
                continue

            # Ablate: subtract centroid contribution for tokens in cluster k
            flat_ablated = flat_normed - mask * c[k:k+1]    # (T, D)

            # Denormalise and splice
            raw_ablated = flat_ablated * std + mean          # (T, D) back to model space
            spliced = raw_ablated.reshape(1, T, D).to(hs_dev).to(hs.dtype)

            hook.activate(lambda hs, r=spliced: r.to(hs.device))
            out_k = lm(input_ids=input_ids)
            hook.deactivate()

            ce_k = ce_per_token(out_k.logits, input_ids)    # (T-1,)
            ce_delta = (ce_k - ce_orig).cpu()

            target_ids = input_ids[0, 1:].cpu().tolist()
            for tok_id, delta in zip(target_ids, ce_delta.tolist()):
                token_changes[k].append((tok_id, delta))

    # Summarise
    summary = {}
    for k in range(K):
        if not token_changes[k]:
            summary[k] = {"top_tokens": [], "mean_ce_change": 0.0, "gini": 0.0}
            continue
        sorted_changes = sorted(token_changes[k], key=lambda x: x[1], reverse=True)
        top = [(tokenizer.decode([tid]), d) for tid, d in sorted_changes[:top_k_tokens]]
        all_d = np.array([d for _, d in token_changes[k]])
        gini = _gini(np.abs(all_d))
        summary[k] = {
            "top_tokens": top,
            "mean_ce_change": float(all_d.mean()),
            "gini": gini,
        }
    return summary


def _gini(arr: np.ndarray) -> float:
    arr = np.sort(arr.flatten())
    if arr.sum() == 0:
        return 0.0
    n = len(arr)
    return float((2 * (np.arange(1, n+1) * arr).sum()) / (n * arr.sum()) - (n+1)/n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "eval", "both"])
    parser.add_argument("--layer",       type=int, default=27)
    parser.add_argument("--n_clusters",  type=int, default=128)
    parser.add_argument("--n_sample",    type=int, default=500_000,
                        help="Tokens to sample for k-means fitting")
    parser.add_argument("--n_eval",      type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--activations_dir", default="activations")
    parser.add_argument("--model_name",  default="meta-llama/Llama-3.2-3B")
    args = parser.parse_args()

    from geoae.seeding import seed_everything
    seed_everything(args.seed)

    act_dir   = Path(args.activations_dir)
    ckpt_path = Path(f"checkpoints/baseline_layer{args.layer}_k{args.n_clusters}.npz")
    out_path  = Path(f"results_baseline_layer{args.layer}_k{args.n_clusters}.json")

    if args.phase in ("fit", "both"):
        fit_kmeans(act_dir, args.layer, args.n_clusters,
                   args.n_sample, args.seed, ckpt_path)

    if args.phase in ("eval", "both"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[baseline] Loading LM: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        lm = load_lm(args.model_name, device_map="auto")

        print(f"[baseline] Loading {args.n_eval} eval texts …")
        try:
            from datasets import load_dataset
            ds = load_dataset("allenai/c4", name="en", split="validation", streaming=True)
            texts = [ex["text"] for ex in ds.take(args.n_eval)]
        except Exception as e:
            print(f"[baseline] HF load failed ({e}), using fallback texts")
            texts = ["The quick brown fox jumps. " * 30] * 10

        summary = evaluate_kmeans_clusters(
            lm, tokenizer, ckpt_path, texts[:50],  # 50 texts for speed
            args.layer, device,
        )

        ranked = sorted(summary.items(), key=lambda x: x[1]["mean_ce_change"], reverse=True)
        print(f"\nTop 15 raw k-means clusters (layer {args.layer}, K={args.n_clusters}):")
        for k, info in ranked[:15]:
            top = [f"{t!r}({d:.3f})" for t, d in info["top_tokens"][:3]]
            print(f"  k={int(k):3d}  Δce={info['mean_ce_change']:+.4f}  "
                  f"gini={info['gini']:.3f}  [{', '.join(top)}]")

        ginis = [v["gini"] for v in summary.values()]
        print(f"\nGini mean: {np.mean(ginis):.4f}  "
              f">0.6: {sum(1 for g in ginis if g > 0.6)}/{len(ginis)}")

        with open(out_path, "w") as f:
            json.dump({"per_cluster": {str(k): v for k, v in summary.items()}}, f, indent=2)
        print(f"[baseline] Results → {out_path}")


if __name__ == "__main__":
    main()
