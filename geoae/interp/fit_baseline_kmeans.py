"""
Fit a raw k-means baseline on residual-stream activations, for a head-to-head
monosemanticity comparison against a learned GeoAE with the same K.

Fairness: the activations are normalised with the SAME mean/std the AE uses
(read straight from the AE checkpoint), so the baseline and the AE cluster the
identical normalised space — the only difference is the AE's learned encoder.

Output .npz (consumed by geoae.interp.closest_tokens --baseline_kmeans):
    centroids (K, D) in normalised space, norm_mean, norm_std, layer,
    n_clusters, model_name.

Usage:
    python -m geoae.interp.fit_baseline_kmeans \
      --checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/best_val.pt \
      --activations activations_diverse/layer_27.npy \
      --n_clusters 1000 --n_sample 1500000 \
      --out e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/baseline_kmeans_k1000.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from geoae.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="AE checkpoint (for norm + model_name + layer)")
    ap.add_argument("--activations", required=True, help="layer_<L>.npy to cluster")
    ap.add_argument("--n_clusters", type=int, default=1000)
    ap.add_argument("--n_sample", type=int, default=1_500_000, help="Tokens sampled for k-means")
    ap.add_argument("--batch_size", type=int, default=65536, help="MiniBatchKMeans batch (>= 64*K enforced)")
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--n_init", type=int, default=3)
    ap.add_argument("--verbose", type=int, default=1, help="MiniBatchKMeans verbosity (0=silent)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seed_everything(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(ckpt["norm_mean"], dtype=np.float32)
    std  = np.asarray(ckpt["norm_std"],  dtype=np.float32)
    cfg = ckpt["config"]
    layer = cfg["data"]["target_layer"]
    model_name = cfg["extraction"]["model_name"]
    print(f"[baseline-fit] Using AE norm from checkpoint (layer {layer}, {model_name})")

    mmap = np.load(args.activations, mmap_mode="r")
    N = mmap.shape[0]
    rng = np.random.RandomState(args.seed)
    idx = np.sort(rng.choice(N, size=min(args.n_sample, N), replace=False))
    print(f"[baseline-fit] Sampling {len(idx):,}/{N:,} activations and normalising …")
    from tqdm import tqdm
    D = mmap.shape[1]
    sample = np.empty((len(idx), D), dtype=np.float32)
    read_chunk = 100_000
    for s in tqdm(range(0, len(idx), read_chunk), desc="reading", unit="chunk"):
        blk = idx[s : s + read_chunk]
        sample[s : s + len(blk)] = (mmap[blk].astype(np.float32) - mean) / std

    from sklearn.cluster import MiniBatchKMeans
    # Large batch relative to K matters: with K=1000 a small batch starves most
    # centroids (few points each) and strands ~half as empty clusters, which would
    # unfairly handicap the baseline. A big batch + more iters + aggressive
    # reassignment of starved centroids gives k-means its best shot at the budget.
    batch_size = max(args.batch_size, 64 * args.n_clusters)
    print(f"[baseline-fit] Fitting MiniBatchKMeans K={args.n_clusters} "
          f"(batch={batch_size}, max_iter={args.max_iter}) …")
    km = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.seed,
                         batch_size=batch_size, n_init=args.n_init, max_iter=args.max_iter,
                         reassignment_ratio=0.05, verbose=args.verbose)
    km.fit(sample)
    centroids = km.cluster_centers_.astype(np.float32)
    # Fit-time probe only (undercounts vs the full collector run): how many
    # centroids are nearest to >=1 point on a large sample.
    probe = sample[: min(1_000_000, len(sample))]
    used = len(np.unique(km.predict(probe)))
    print(f"[baseline-fit] Fit done. {used}/{args.n_clusters} centroids win on a "
          f"{len(probe):,}-token probe (final live count comes from collect_closest_tokens).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), centroids=centroids, norm_mean=mean, norm_std=std,
             layer=layer, n_clusters=args.n_clusters, model_name=model_name)
    print(f"[baseline-fit] Saved -> {out}")


if __name__ == "__main__":
    main()
