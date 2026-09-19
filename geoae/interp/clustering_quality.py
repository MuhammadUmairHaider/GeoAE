"""
Comprehensive clustering quality analysis.

Compares clustering quality across:
  - Raw k-means baseline (no AE, clusters directly in residual stream space)
  - Any number of AE checkpoints (linear/nonlinear, any latent dim)

Metrics computed:
  Geometric (latent space):
    silhouette          intra vs inter cluster similarity [-1,1] — higher better
    davies_bouldin      within/between cluster ratio — lower better
    calinski_harabasz   between/within variance ratio — higher better
    dunn_index          min inter / max intra cluster diameter — higher better
    separability_ratio  mean inter-centroid dist / mean intra-cluster std — higher better
    intra_var           mean within-cluster variance — lower = tighter clusters
    inter_centroid_dist mean pairwise centroid distance — higher = more spread
    cluster_balance     entropy of cluster size distribution — higher = more uniform
    assignment_entropy  softness of cluster assignments (0=hard, log(K)=uniform)
    effective_k         clusters with usage > 0.1/K
    effective_rank      effective rank of centroid matrix (exp of singular value entropy)

  Functional (from ablation results, if results.json available):
    gini_mean           mean Gini of per-token CE changes — higher = more selective
    gini_gt_06          fraction of clusters with Gini > 0.6
    mean_ce_change      mean CE change when a cluster is ablated — higher = more informative

Usage:
    python -m geoae.interp.clustering_quality \\
        --baseline   checkpoints/baseline_layer27_k128.npz \\
        --checkpoints \\
            checkpoints/layer_27/best_val.pt \\
            checkpoints/layer27_k128_d3072_linear/best_val.pt \\
            checkpoints/layer27_k128_d3072_gelu/best_val.pt \\
        --results \\
            results_baseline_layer27_k128.json \\
            results_layer27_k128_d2048_linear.json \\
            results_layer27_k128_d3072_linear.json \\
            results_layer27_k128_d3072_gelu.json \\
        --layer 27 --n_sample 50000
"""
from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path

import numpy as np
import torch




# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def silhouette(z: np.ndarray, labels: np.ndarray, max_samples: int = 10_000) -> float:
    from sklearn.metrics import silhouette_score
    if len(z) > max_samples:
        idx = np.random.choice(len(z), max_samples, replace=False)
        z, labels = z[idx], labels[idx]
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(z, labels, metric="euclidean", sample_size=min(5000, len(z))))


def davies_bouldin(z: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import davies_bouldin_score
    if len(set(labels)) < 2:
        return float("nan")
    return float(davies_bouldin_score(z, labels))


def calinski_harabasz(z: np.ndarray, labels: np.ndarray) -> float:
    """
    sklearn's calinski_harabasz_score, without its full float64 copy.

    sklearn upcasts the whole matrix to float64 before looping over clusters:
    +24 GB at n=1M, d=3072 and +49 GB at d=6144. That pushed the eval into
    memory pressure and systemd-oomd killed the tmux scope (2026-09-19). Same
    formula and the same per-cluster float64 accumulation, one cluster at a time.
    """
    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    k, n = len(uniq), len(z)
    if k < 2:
        return float("nan")
    mean = z.mean(axis=0, dtype=np.float64)
    order = np.argsort(inv, kind="stable")
    extra = intra = 0.0
    for idx in np.split(order, np.cumsum(counts)[:-1]):
        ck = z[idx].astype(np.float64)
        mk = ck.mean(axis=0)
        extra += len(ck) * float(((mk - mean) ** 2).sum())
        intra += float(((ck - mk) ** 2).sum())
    return 1.0 if intra == 0.0 else float(extra * (n - k) / (intra * (k - 1.0)))


def dunn_index(z: np.ndarray, labels: np.ndarray, max_samples: int = 5_000) -> float:
    """
    Dunn index = min inter-cluster distance / max intra-cluster diameter.
    Approximated on a subsample for speed.
    """
    if len(z) > max_samples:
        idx = np.random.choice(len(z), max_samples, replace=False)
        z, labels = z[idx], labels[idx]
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")

    # Per-cluster means and diameters
    centroids = {k: z[labels == k].mean(axis=0) for k in unique if (labels == k).sum() > 0}
    diameters = {}
    for k in unique:
        pts = z[labels == k]
        if len(pts) < 2:
            diameters[k] = 0.0
        else:
            # Approximate diameter as 2 * mean distance to centroid
            dists = np.linalg.norm(pts - centroids[k], axis=1)
            diameters[k] = float(2 * dists.mean())

    max_diam = max(diameters.values()) if diameters else 1e-8

    # Min inter-cluster centroid distance
    keys = list(centroids.keys())
    min_inter = float("inf")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = np.linalg.norm(centroids[keys[i]] - centroids[keys[j]])
            if d < min_inter:
                min_inter = d

    return float(min_inter / max(max_diam, 1e-8))


def separability_ratio(z: np.ndarray, labels: np.ndarray) -> float:
    """Mean inter-centroid distance / mean intra-cluster std."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")

    centroids = np.stack([z[labels == k].mean(axis=0) for k in unique if (labels == k).sum() > 0])
    # inter: mean pairwise centroid distance
    from sklearn.metrics import pairwise_distances
    pdist = pairwise_distances(centroids, metric="euclidean")
    n = len(centroids)
    inter = pdist[np.triu_indices(n, k=1)].mean()

    # intra: mean within-cluster std
    stds = []
    for k in unique:
        pts = z[labels == k]
        if len(pts) > 1:
            stds.append(pts.std(axis=0).mean())
    intra_std = np.mean(stds) if stds else 1e-8

    return float(inter / max(intra_std, 1e-8))


def inter_centroid_dist(centroids: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean, min, max) of pairwise centroid distances."""
    from sklearn.metrics import pairwise_distances
    pd = pairwise_distances(centroids, metric="euclidean")
    n = len(centroids)
    vals = pd[np.triu_indices(n, k=1)]
    return float(vals.mean()), float(vals.min()), float(vals.max())


def intra_cluster_variance(z: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-cluster variance (per dimension, then averaged)."""
    unique = np.unique(labels)
    vars_ = []
    for k in unique:
        pts = z[labels == k]
        if len(pts) > 1:
            vars_.append(pts.var(axis=0).mean())
    return float(np.mean(vars_)) if vars_ else float("nan")


def cluster_balance_entropy(labels: np.ndarray) -> tuple[float, float]:
    """
    Entropy of cluster size distribution, normalised by log(K).
    1.0 = perfectly balanced, 0.0 = all mass in one cluster.
    Also returns n_empty clusters.
    """
    unique, counts = np.unique(labels, return_counts=True)
    K_total = int(labels.max()) + 1
    n_empty = K_total - len(unique)
    probs = counts / counts.sum()
    H = -np.sum(probs * np.log(probs + 1e-10))
    H_max = math.log(len(unique))
    return float(H / max(H_max, 1e-10)), n_empty


def row_entropy(Q: np.ndarray) -> np.ndarray:
    """(B,) entropy of each row of a soft assignment matrix Q (B, K)."""
    eps = 1e-10
    return -(Q * np.log(Q + eps)).sum(axis=1)


def assignment_entropy(Q: np.ndarray) -> float:
    """Mean row entropy of soft assignment matrix Q (B, K)."""
    return float(row_entropy(Q).mean())


def effective_rank(centroids: np.ndarray) -> float:
    """
    Effective rank of the CENTERED centroid matrix = exp(entropy of squared
    singular values). Measures how many independent directions the centroids
    span around their mean. Centering matters: non-negative latents (GELU/ReLU)
    share a large mean offset that would otherwise dominate the spectrum.
    Range: 1 (all in one direction) to min(K, L) (fully spread).
    """
    _, s, _ = np.linalg.svd(centroids - centroids.mean(axis=0), full_matrices=False)
    s2 = s ** 2
    s2 = s2 / s2.sum()
    H = -np.sum(s2 * np.log(s2 + 1e-12))
    return float(math.exp(H))


def effective_k(labels: np.ndarray, K: int, threshold: float = 0.1) -> int:
    """Number of clusters with usage > threshold/K."""
    counts = np.bincount(labels, minlength=K)
    usage = counts / counts.sum()
    return int((usage > threshold / K).sum())


# ---------------------------------------------------------------------------
# Load latents for each mode
# ---------------------------------------------------------------------------

def read_rows(npy_path: Path, idx: np.ndarray) -> np.ndarray:
    """
    Rows `idx` of a 2-D .npy dump, read with preadv rather than through an mmap.

    Indexing an np.load(mmap_mode="r") array maps page-cache pages into this
    process, and on this kernel a fault maps whole cached folios around the
    touched row — MADV_RANDOM does not stop it. Gathering 1M scattered rows from
    the 60 GB L27 dump left 42.5 GB of the file mapped here at peak, on top of the
    latents, which is the memory pressure that got the eval's tmux scope killed
    by systemd-oomd. preadv copies exactly the bytes asked for; the page cache is
    still used but never mapped. Values are byte-identical to mmap indexing.
    """
    mm = np.load(str(npy_path), mmap_mode="r")
    if mm.ndim != 2 or not mm.flags.c_contiguous:
        raise ValueError(f"{npy_path}: expected a C-ordered 2-D array")
    dtype, D, offset = mm.dtype, mm.shape[1], int(mm.offset)
    del mm
    row = dtype.itemsize * D
    out = np.empty((len(idx), D), dtype=dtype)
    if len(idx) == 0:
        return out                      # memoryview.cast rejects zero-size buffers
    buf = memoryview(out).cast("B")
    fd = os.open(str(npy_path), os.O_RDONLY)
    try:
        for j, i in enumerate(idx):
            if os.preadv(fd, [buf[j * row:(j + 1) * row]], offset + int(i) * row) != row:
                raise IOError(f"{npy_path}: short read at row {int(i)}")
    finally:
        os.close(fd)
    return out


def load_raw_baseline(
    baseline_npz: Path,
    act_dir: Path,
    layer: int,
    n_sample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Returns (latents, hard_labels, centroids, label_str)."""
    data = np.load(str(baseline_npz))
    centroids = data["centroids"]     # (K, D) normalised
    norm_mean = data["norm_mean"]
    norm_std  = data["norm_std"]
    K = len(centroids)

    # Sample activations
    npy = act_dir / f"layer_{layer}.npy"
    N = np.load(str(npy), mmap_mode="r").shape[0]
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(N, min(n_sample, N), replace=False))
    sample = read_rows(npy, idx).astype(np.float32)
    sample = (sample - norm_mean) / norm_std   # normalise

    # Hard k-means assignment (nearest centroid)
    from sklearn.metrics import pairwise_distances_argmin
    labels = pairwise_distances_argmin(sample, centroids, metric="euclidean")

    label = f"Raw k-means  (D={centroids.shape[1]}, K={K})"
    return sample, labels, centroids, label


def load_ae_latents(
    ckpt_path: Path,
    act_dir: Path,
    layer: int,
    n_sample: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Returns (latents, hard_labels, per_row_assignment_entropy, centroids, label_str).

    Streams the sample: each batch is read from the mmap, normalised and encoded,
    and written into preallocated outputs. The earlier version held the full
    normalised input (11 GB at n=1M), a list of latent batches plus its
    concatenated copy (2 x 23 GB at latent_dim 6144) and the full soft Q (7.5 GB)
    at once — ~70 GB, which got the whole tmux scope killed by systemd-oomd once
    the dump's page cache was also mapped in (see `read_rows`). Q was only ever used
    for a mean row entropy, which is additive, so per-row entropies are kept
    instead. Batches are the same 4096 sorted rows as before, so BatchNorm and
    Sinkhorn see identical inputs and every metric is unchanged.
    """
    from geoae.checkpoint import load_ae_checkpoint
    ae, _, _, ckpt = load_ae_checkpoint(ckpt_path, device)
    mc = ckpt["config"]["model"]
    nl = mc.get("nonlinearity", "linear")

    norm_mean = ckpt["norm_mean"]
    norm_std  = ckpt["norm_std"]
    layer_from_cfg = ckpt["config"]["data"]["target_layer"]
    K = mc["n_clusters"]

    # Sample activations
    npy = act_dir / f"layer_{layer_from_cfg}.npy"
    N = np.load(str(npy), mmap_mode="r").shape[0]
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(N, min(n_sample, N), replace=False))

    # Encode in batches. Labels use nearest-centroid (dist2.argmin), matching the
    # k-means baseline's hard assignment — Sinkhorn Q.argmax would batch-balance
    # the labels by construction and depend on batch composition. Q is still
    # used for the assignment-entropy diagnostic.
    batch = 4096
    n = len(idx)
    z = np.empty((n, mc["latent_dim"]), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    H = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch):
            x_np = (read_rows(npy, idx[s:s+batch]).astype(np.float32) - norm_mean) / norm_std
            out = ae(torch.from_numpy(x_np).to(device))
            e = s + len(x_np)
            z[s:e] = out.z.cpu().numpy()
            H[s:e] = row_entropy(out.Q.cpu().numpy())
            labels[s:e] = out.dist2.argmin(dim=1).cpu().numpy()

    centroids = ae.centroids.cpu().numpy()

    label = (f"AE {mc['hidden_size']}→{mc['latent_dim']}→{mc['hidden_size']} "
             f"({nl}, K={K})")
    return z, labels, H, centroids, label


# ---------------------------------------------------------------------------
# Load functional metrics from results.json
# ---------------------------------------------------------------------------

def load_functional_metrics(results_path: Path | None) -> dict:
    if results_path is None or not results_path.exists():
        return {}
    with open(results_path) as f:
        data = json.load(f)
    clusters = data.get("per_cluster", {})
    if not clusters:
        return {}
    ginis = [v["gini"] for v in clusters.values()]
    changes = [v["mean_ce_change"] for v in clusters.values()]
    return {
        "gini_mean":    np.mean(ginis),
        "gini_gt_06":   np.mean([g > 0.6 for g in ginis]),
        "mean_ce_change": np.mean(changes),
    }


# ---------------------------------------------------------------------------
# Compute all metrics for one variant
# ---------------------------------------------------------------------------

def compute_metrics(
    z: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    Q: np.ndarray | None = None,
    functional: dict | None = None,
    assign_entropy: float | None = None,
) -> dict:
    K = centroids.shape[0]
    print("    silhouette …", end=" ", flush=True)
    sil = silhouette(z, labels)
    print(f"{sil:.4f}")

    print("    davies-bouldin …", end=" ", flush=True)
    db = davies_bouldin(z, labels)
    print(f"{db:.4f}")

    print("    calinski-harabasz …", end=" ", flush=True)
    ch = calinski_harabasz(z, labels)
    print(f"{ch:.1f}")

    print("    dunn index …", end=" ", flush=True)
    di = dunn_index(z, labels)
    print(f"{di:.6f}")

    print("    separability ratio …", end=" ", flush=True)
    sr = separability_ratio(z, labels)
    print(f"{sr:.4f}")

    # Centroid-matrix metrics must see only centroids that actually won points.
    # Every (z, labels) metric above is already restricted to surviving clusters
    # via np.unique(labels), but these two read the stored K x D matrix directly.
    # A collapsed baseline (raw k-means at K=2000 keeps ~270 live centroids) would
    # otherwise report the distance between two DEAD centroids as its minimum, and
    # inflate effective rank with centroids no token maps to.
    live_mask = np.bincount(labels, minlength=K) > 0
    live_centroids = centroids[live_mask]
    icd_mean, icd_min, icd_max = inter_centroid_dist(live_centroids)
    intra_var = intra_cluster_variance(z, labels)
    bal_H, n_empty = cluster_balance_entropy(labels)
    eff_k = effective_k(labels, K)
    eff_r = effective_rank(live_centroids)

    metrics = {
        "silhouette":        sil,
        "davies_bouldin":    db,
        "calinski_harabasz": ch,
        "dunn_index":        di,
        "separability_ratio": sr,
        "inter_centroid_dist_mean": icd_mean,
        "inter_centroid_dist_min":  icd_min,
        "intra_cluster_var": intra_var,
        "cluster_balance":   bal_H,
        "n_empty_clusters":  n_empty,
        "effective_k":       eff_k,
        "effective_rank":    eff_r,
    }
    if Q is not None:
        metrics["assignment_entropy"] = assignment_entropy(Q)
    elif assign_entropy is not None:
        metrics["assignment_entropy"] = assign_entropy
    if functional:
        metrics.update(functional)
    return metrics


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

METRIC_ROWS = [
    # (key, display name, direction, format)
    ("silhouette",              "Silhouette ↑",          "up",   ".4f"),
    ("davies_bouldin",          "Davies-Bouldin ↓",      "down", ".4f"),
    ("calinski_harabasz",       "Calinski-Harabasz ↑",   "up",   ".1f"),
    ("dunn_index",              "Dunn Index ↑",          "up",   ".6f"),
    ("separability_ratio",      "Separability Ratio ↑",  "up",   ".4f"),
    ("inter_centroid_dist_mean","Inter-centroid Dist ↑", "up",   ".4f"),
    ("inter_centroid_dist_min", "Min Centroid Dist ↑",   "up",   ".4f"),
    ("intra_cluster_var",       "Intra-cluster Var ↓",   "down", ".4f"),
    ("cluster_balance",         "Cluster Balance H ↑",   "up",   ".4f"),
    ("n_empty_clusters",        "Empty Clusters ↓",      "down", "d"),
    ("effective_k",             "Effective K ↑",         "up",   "d"),
    ("effective_rank",          "Effective Rank ↑",      "up",   ".2f"),
    ("assignment_entropy",      "Assignment Entropy",     None,   ".4f"),
    ("gini_mean",               "Gini Mean ↑",           "up",   ".4f"),
    ("gini_gt_06",              "Gini>0.6 Frac ↑",       "up",   ".3f"),
    ("mean_ce_change",          "Mean CE Change ↑",      "up",   ".4f"),
]


def print_table(labels: list[str], all_metrics: list[dict]) -> None:
    col_w = max(18, max(len(l) for l in labels) + 2)
    row_w = 28

    header = f"  {'Metric':<{row_w}}" + "".join(f"  {l[:col_w-2]:>{col_w}}" for l in labels)
    print("\n" + "="*(row_w + 4 + len(labels)*(col_w+2)))
    print(header)
    print("-"*(row_w + 4 + len(labels)*(col_w+2)))

    for key, name, direction, fmt in METRIC_ROWS:
        vals = [m.get(key, None) for m in all_metrics]
        if all(v is None for v in vals):
            continue

        # Find best value for highlighting
        numeric = [v for v in vals if v is not None and not math.isnan(v)]
        if numeric and direction == "up":
            best = max(numeric)
        elif numeric and direction == "down":
            best = min(numeric)
        else:
            best = None

        row = f"  {name:<{row_w}}"
        for v in vals:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cell = "N/A"
            elif fmt == "d":
                cell = f"{int(v):{fmt}}"
            else:
                cell = f"{v:{fmt}}"
            marker = " *" if (best is not None and v is not None and
                              not (isinstance(v, float) and math.isnan(v)) and
                              abs(v - best) < 1e-9 * max(abs(best), 1)) else "  "
            row += f"  {cell+marker:>{col_w}}"
        print(row)

    print("="*(row_w + 4 + len(labels)*(col_w+2)))
    print("  * = best value for that metric")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",    default=None,
                        help="Path to baseline k-means .npz (from cluster_baseline.py fit)")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="AE checkpoint .pt files")
    parser.add_argument("--results",     nargs="*", default=[],
                        help="results.json files (same order: baseline first, then AEs)")
    parser.add_argument("--layer",       type=int, default=27)
    parser.add_argument("--n_sample",    type=int, default=50_000)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--activations_dir", default="activations")
    parser.add_argument("--names", nargs="*", default=[],
                        help="Short column labels, in order (baseline first, then "
                             "checkpoints). Architecture strings are identical across "
                             "sibling runs, so pass run names to keep the table readable.")
    parser.add_argument("--out", default="clustering_quality_comparison.json")
    args = parser.parse_args()

    from geoae.seeding import seed_everything
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_dir = Path(args.activations_dir)

    # Parse results files into a list aligned with (baseline, *checkpoints)
    results_paths = [Path(r) if r else None for r in args.results]
    n_variants = (1 if args.baseline else 0) + len(args.checkpoints)
    while len(results_paths) < n_variants:
        results_paths.append(None)

    all_labels, all_metrics = [], []
    ri = 0   # results index

    # --- Baseline ---
    if args.baseline:
        print(f"\n[quality] Loading raw k-means baseline: {args.baseline}")
        z, labels, centroids, label = load_raw_baseline(
            Path(args.baseline), act_dir, args.layer, args.n_sample, args.seed
        )
        if ri < len(args.names):
            label = args.names[ri]
        print(f"[quality] Computing metrics for: {label}")
        func = load_functional_metrics(results_paths[ri])
        metrics = compute_metrics(z, labels, centroids, Q=None, functional=func)
        all_labels.append(label)
        all_metrics.append(metrics)
        ri += 1

    # --- AE variants ---
    for ckpt_path in args.checkpoints:
        print(f"\n[quality] Loading AE: {ckpt_path}")
        z, labels, H, centroids, label = load_ae_latents(
            Path(ckpt_path), act_dir, args.layer, args.n_sample, args.seed, device
        )
        if ri < len(args.names):
            label = args.names[ri]
        print(f"[quality] Computing metrics for: {label}")
        func = load_functional_metrics(results_paths[ri])
        metrics = compute_metrics(z, labels, centroids, functional=func,
                                  assign_entropy=float(H.mean()))
        # Free this model's 23 GB of latents BEFORE the next one is encoded;
        # rebinding `z` only releases it after the next load has finished.
        del z, labels, H
        all_labels.append(label)
        all_metrics.append(metrics)
        ri += 1

    # --- Print table ---
    print_table(all_labels, all_metrics)

    # Save
    out = {}
    for lbl, m in zip(all_labels, all_metrics):
        out[lbl] = {k: (v if not (isinstance(v, float) and math.isnan(v)) else None)
                    for k, v in m.items()}
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[quality] Full results saved → {out_path}")


if __name__ == "__main__":
    main()
