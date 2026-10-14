"""Cluster geometry of an SAE's code space under its plain k-means codebook (eval_out/sae_plain_kmeans_fit.py), with the
metrics and the row sample of cq.json (geoae/interp/clustering_quality.py): 1,000,000 dump rows, sorted
RandomState(0).choice over all rows; labels = nearest centroid; each metric in the space's own Euclidean coordinates.

Dense codes of 1M rows x 16,384 features are 65 GB, so nothing is materialised: two streamed passes over the rows collect
the per-cluster sufficient statistics (count, sum, sum of squares in float64; then each row's distance to its cluster's
empirical mean) and the metrics are computed from them with clustering_quality's definitions:
  calinski_harabasz   sklearn formula (empirical cluster means)
  davies_bouldin      sklearn formula (mean distance to the empirical mean; centroid distances)
  separability_ratio  mean pairwise distance of empirical means / mean over clusters (n > 1) of the per-dimension std
  intra_cluster_var   mean over clusters (n > 1) of the per-dimension variance
  inter_centroid_dist stored centroids of the clusters that won rows
  silhouette          clustering_quality.silhouette on its own seeded 10,000-row subsample (rows gathered densely)
`--km` adds the balanced residual codebook scored through the same code path; it must reproduce cq.json's "base" entry.

    python -u eval_out/sae_geometry.py --activations_dir activations_gemma3_4b_sampled_struct5_10M --layer 33 \
        --sae scope:16k --codebook results/delta/g4b_l33_d5120/evals/sae_suite/controls/plain_kmeans_gs16k.npz \
        --km e2e/checkpoints/general/gemma3-4B/layer33/delta/balanced_kmeans_k2000_dpc_sampled_struct5.npz \
        --out results/delta/g4b_l33_d5120/evals/sae_suite/cq_gs16k_plain.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoae.interp.clustering_quality import (cluster_balance_entropy, effective_k, inter_centroid_dist, read_rows,  # noqa: E402
                                             silhouette)
from geoae.interp.sae_space import load_sae_space  # noqa: E402

CHUNK = 16384


@torch.no_grad()
def geometry(H: torch.Tensor, embed, C: torch.Tensor, seed: int = 0) -> dict:
    """H (n, d_in) rows on the GPU; embed(h_chunk) -> coordinates (m, F); C (K, F) stored centroids."""
    n, (K, F) = len(H), C.shape
    c2 = (C * C).sum(1)
    lab = torch.empty(n, dtype=torch.long, device=H.device)
    cnt = torch.zeros(K, dtype=torch.float64, device=H.device)
    s1 = torch.zeros(K, F, dtype=torch.float64, device=H.device)
    s2 = torch.zeros(K, F, dtype=torch.float64, device=H.device)
    sub = np.sort(np.random.default_rng(seed).choice(n, min(10_000, n), replace=False))  # silhouette()'s own draw
    sub_t = torch.from_numpy(sub).to(H.device)
    zsub = torch.empty(len(sub), F, device=H.device)
    for i in range(0, n, CHUNK):
        z = embed(H[i:i + CHUNK]).float()
        a = (c2[None, :] - 2.0 * (z @ C.T)).argmin(1)
        lab[i:i + CHUNK] = a
        zd = z.double()
        cnt.index_add_(0, a, torch.ones(len(a), dtype=torch.float64, device=H.device))
        s1.index_add_(0, a, zd)
        s2.index_add_(0, a, zd * zd)
        m = (sub_t >= i) & (sub_t < i + CHUNK)
        zsub[m] = z[sub_t[m] - i]
    live = cnt > 0
    nk = cnt[live]
    mean = s1[live] / nk[:, None]                      # empirical cluster means, float64
    var = (s2[live] / nk[:, None] - mean * mean).clamp_min(0)
    remap = torch.full((K,), -1, dtype=torch.long, device=H.device)
    remap[live] = torch.arange(int(live.sum()), device=H.device)
    dsum = torch.zeros(int(live.sum()), dtype=torch.float64, device=H.device)
    for i in range(0, n, CHUNK):
        z = embed(H[i:i + CHUNK]).double()
        k = remap[lab[i:i + CHUNK]]
        dsum.index_add_(0, k, (z - mean[k]).pow(2).sum(1).sqrt())
    k_live, N = int(live.sum()), float(n)
    gmean = s1.sum(0) / N
    extra = float((nk * (mean - gmean).pow(2).sum(1)).sum())
    intra = float(s2.sum() - (nk[:, None] * mean * mean).sum())
    ch = 1.0 if intra == 0.0 else extra * (N - k_live) / (intra * (k_live - 1.0))
    cd = torch.cdist(mean, mean)                       # float64
    s = dsum / nk
    cdi = cd.clone()
    cdi[cdi == 0] = float("inf")
    cdi.fill_diagonal_(float("inf"))                   # cdist's diagonal is ~1e-6, not exactly 0
    db = float(((s[:, None] + s[None, :]) / cdi).max(1).values.mean())
    iu = torch.triu_indices(k_live, k_live, 1, device=H.device)
    inter = float(cd[iu[0], iu[1]].mean())
    multi = nk > 1
    intra_std = float(var[multi].sqrt().mean(1).mean())
    labels = lab.cpu().numpy()
    icd_mean, icd_min, _ = inter_centroid_dist(C[live].cpu().numpy())
    bal_h, n_empty = cluster_balance_entropy(labels)
    # clustering_quality.silhouette(z, labels): default_rng(seed) subsample of 10,000, then sklearn's 5,000 with random_state
    order = np.random.default_rng(seed).choice(n, min(10_000, n), replace=False)
    pos = np.searchsorted(sub, order)
    sil = silhouette(zsub.cpu().numpy()[pos], labels[order], seed=seed) if n > 10_000 else float("nan")
    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch, "separability_ratio": inter / max(intra_std, 1e-8),
            "inter_centroid_dist_mean": icd_mean, "inter_centroid_dist_min": icd_min,
            "intra_cluster_var": float(var[multi].mean(1).mean()), "cluster_balance": float(bal_h),
            "n_empty_clusters": int(K - k_live), "effective_k": int(effective_k(labels, K)), "dim": int(F), "n_rows": int(n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--codebook", required=True, help="plain k-means on the SAE codes (centroids)")
    ap.add_argument("--km", default="", help="balanced residual k-means .npz: scored the same way as a check against cq.json")
    ap.add_argument("--n_sample", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    t0 = time.time()

    def log(*a):
        print(time.strftime("[%H:%M:%S]"), *a, f"({time.time() - t0:.0f}s)", flush=True)

    npy = Path(args.activations_dir) / f"layer_{args.layer}.npy"
    N = np.load(str(npy), mmap_mode="r").shape[0]
    idx = np.sort(np.random.RandomState(args.seed).choice(N, min(args.n_sample, N), replace=False))
    H = torch.from_numpy(read_rows(npy, idx).astype(np.float32)).to(dev)
    log(f"read {len(H):,} of {N:,} rows")
    res = {"meta": dict(vars(args), sample="sorted RandomState(seed).choice over all dump rows (as cq.json)",
                        labels="nearest stored centroid", metrics="clustering_quality definitions, streamed")}
    if args.km:
        z = np.load(args.km, allow_pickle=True)
        mu = torch.as_tensor(z["norm_mean"], dtype=torch.float32, device=dev)
        sd = torch.as_tensor(z["norm_std"], dtype=torch.float32, device=dev)
        res["base_check"] = geometry(H, lambda h: (h - mu) / sd, torch.as_tensor(z["centroids"], dtype=torch.float32, device=dev))
        log("base (check vs cq.json)", json.dumps(res["base_check"]))
        Path(args.out).write_text(json.dumps(res, indent=1))
    sae = load_sae_space(args.sae, dev, "sae")
    C = torch.as_tensor(np.load(args.codebook)["centroids"], dtype=torch.float32, device=dev)
    res["sae_plain"] = geometry(H, lambda h: sae.encode_raw(h, None), C)
    log("sae_plain", json.dumps(res["sae_plain"]))
    Path(args.out).write_text(json.dumps(res, indent=1))
    log("wrote", args.out)


if __name__ == "__main__":
    main()
