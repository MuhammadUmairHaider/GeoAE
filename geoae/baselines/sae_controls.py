"""Control helpers for the SAE suite: balanced k-means on SAE codes (C1), frequent-feature selection (C3),
non-sibling null pairs (C4).

``balanced_fit`` reproduces geoae.baselines.balanced_kmeans.main's fit (the procedure that produced the
base / base_tokmean codebooks: --init dpc --reinit peaks, Sinkhorn-balanced hard-EMA updates, tau 1.0 -> 0.1,
3 Sinkhorn iterations, EMA 0.9, batch 4096, 8 epochs, seed 42, the same density-peaks settings), but on a
ready-made matrix X (here: SAE codes) instead of normalised activations. Both use
the shared balanced codebook updater after their own initialization.
"""
from __future__ import annotations

import numpy as np
import torch

from geoae.baselines._balanced import balanced_updates
from geoae.training.initialization import density_peaks_select


def top_features(counts, k):
    """Indices of the k most frequent features (ties -> lower id)."""
    return np.argsort(-np.asarray(counts), kind="stable")[:k]


@torch.no_grad()
def balanced_fit(X, K, *, seed=42, epochs=8, batch_size=4096, tau_start=1.0, tau_end=0.1, sinkhorn_iters=3,
                 ema_decay=0.9, reinit_every=1000, density_pool=32768, density_knn=32, density_power=1.0,
                 peak_min_sep_frac=0.25, peak_refine_k=8, peak_reinit_pool=8192, log=print):
    """Returns centroids [K, F] (on X's device)."""
    if batch_size < K:
        raise ValueError("batch_size < K: Sinkhorn column marginal is meaningless")
    torch.manual_seed(seed)
    pool_n = min(density_pool, len(X))
    pool = X[torch.randperm(len(X), device=X.device)[:pool_n]]
    C, diag = density_peaks_select(pool, K, seed=seed, density_power=density_power,
                                   min_sep_frac=peak_min_sep_frac, knn_k=density_knn, refine_k=peak_refine_k)
    C = C.clone()
    del pool

    def report(epoch, centroids, tau, n_reinit):
        log(f"  epoch {epoch}/{epochs}: tau {tau:.3f} reinit so far {n_reinit}")

    return balanced_updates(
        X, C, epochs=epochs, batch_size=batch_size, tau_start=tau_start, tau_end=tau_end,
        sinkhorn_iters=sinkhorn_iters, ema_decay=ema_decay, reinit_every=reinit_every,
        reinit="peaks", density_power=density_power, density_knn=density_knn,
        peak_min_sep_frac=peak_min_sep_frac, peak_refine_k=peak_refine_k,
        peak_reinit_pool=peak_reinit_pool, on_epoch=report,
    )


def sample_pairs_nonsibling(labels, family_of, n, seed):
    """Up to n seeded label pairs (a, b) whose families differ. Returns (pairs, n_candidates)."""
    labels = list(labels)
    cand = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:] if family_of[a] != family_of[b]]
    rng = np.random.default_rng(seed)
    take = rng.choice(len(cand), min(n, len(cand)), replace=False) if cand else []
    return [cand[i] for i in sorted(take)], len(cand)


def summarize_pairs(pair_reports, thr=0.9):
    """Rates over pair records from analyze_coding: interpretation / geometry mix, decoder cosine, same-feature rate."""
    n = len(pair_reports)
    cos = np.array([p["decoder_cosine"] for p in pair_reports if p["decoder_cosine"] is not None], float)
    from collections import Counter
    interp = Counter(p["interpretation"] for p in pair_reports)
    geom = Counter(p["feature_geometry"] for p in pair_reports)
    return {
        "n_pairs": n, "n_both_assigned": int(len(cos)),
        "interpretation_counts": dict(interp), "geometry_counts": dict(geom),
        "same_feature_rate": geom.get("same_feature", 0) / max(n, 1),
        "rate_top1_cos_ge_thr": float((cos >= thr).sum() / max(n, 1)),
        "rate_top1_cos_ge_thr_among_assigned": float((cos >= thr).mean()) if len(cos) else None,
        "distinct_parallel_rate": geom.get("distinct_parallel", 0) / max(n, 1),
        "decoder_cosine": {"mean": float(cos.mean()), "median": float(np.median(cos)),
                           "quantiles_5_25_75_95": [float(x) for x in np.quantile(cos, [.05, .25, .75, .95])]}
        if len(cos) else None,
        "threshold": thr,
    }
