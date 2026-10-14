"""Plain (unbalanced) k-means on the TOKEN-REMOVED residual, x - b[token], with density-peaks seeds (Llama-3.2-3B L27,
final dump). Completes the 2 x 2 of {residual, token-removed} x {plain, balanced}, all seeded like the AE.

  - fit: Lloyd's algorithm on every 6th row of the training 95%, density-peaks init with the AE's settings, no balancing,
    empty clusters left in place; standardization and token table of the balanced token-removed baseline;
  - usage on the 500,000 held-out rows, for this fit and the balanced token-removed baseline;
  - geometry with clustering_quality's own metric functions on the exact 1M-row sample cq.json uses
    (RandomState(0) over all rows), for all four k-means variants. clustering_quality itself refuses token-removed
    codebooks, which is why the paper has no token-removed geometry. The two residual rows double as a check:
    they should reproduce cq.json (CH 199.8) and cq_plain_kmeans_density_peaks.json (CH 196.1);
  - saves the centroids with the token_bias field, so probe_chance_corrected and token_structure apply the subtraction.

    PYTHONPATH=. .venv/bin/python eval_out/paper_plain_kmeans_tokmean.py      (on the box)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.interp.clustering_quality import (calinski_harabasz, davies_bouldin, dunn_index, read_rows,
                                             separability_ratio, silhouette)
from geoae.seeded_init import density_peaks_select
from geoae.token_bias import TokenBiasLookup

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
dev = torch.device("cuda")
K, STRIDE, MAX_ITER, TOL = 2000, 6, 100, 1e-4
D = Path("activations_sampled_10M")
DL = Path("e2e/checkpoints/general/llama3.2-3B/layer27/delta")
KM = DL / "balanced_kmeans_k2000_dpc_sampled.npz"
KMT = DL / "balanced_kmeans_k2000_dpc_sampled_tokmean.npz"
KMP = DL / "plain_kmeans_k2000_density_peaks_sampled.npz"
OUT_NPZ = DL / "plain_kmeans_k2000_density_peaks_sampled_tokmean.npz"
OUT = Path("results/delta/d6144/evals/plain_kmeans_tokmean.json")

A = np.load(D / "layer_27.npy", mmap_mode="r")
TOK = np.load(D / "rows_tok.npy", mmap_mode="r")
N = A.shape[0]
v0 = int(N * 0.95)
C_b, m_b, s_b, _, _, _ = load_baseline_kmeans(KM, dev)
C_p, _, _, _, _, _ = load_baseline_kmeans(KMP, dev)
C_bt, m_t, s_t, _, _, _ = load_baseline_kmeans(KMT, dev, allow_token_bias=True)
tb_path = str(np.load(KMT, allow_pickle=True)["token_bias"])
tb = TokenBiasLookup(tb_path, dev)


def rows_to(rows: np.ndarray, token_removed: bool, dtype=torch.float16) -> torch.Tensor:
    out = torch.empty(len(rows), A.shape[1], dtype=dtype, device=dev)
    for i in range(0, len(rows), 65536):
        r = rows[i:i + 65536]
        h = torch.from_numpy(np.asarray(A[r], dtype=np.float32)).to(dev)
        if token_removed:
            t = torch.from_numpy(np.asarray(TOK[r], dtype=np.int64)).to(dev)
            x = (h - m_t) / s_t - tb(t)
        else:
            x = (h - m_b) / s_b
        out[i:i + len(h)] = x.to(dtype)
    return out


def assign(X: torch.Tensor, C: torch.Tensor, chunk: int = 65536):
    C = C.float()
    c2 = (C * C).sum(1)
    lab = torch.empty(len(X), dtype=torch.long, device=dev)
    inertia = 0.0
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].float()
        m, a = (c2[None, :] - 2.0 * (x @ C.T)).min(1)
        lab[i:i + chunk] = a
        inertia += float((m + (x * x).sum(1)).sum())
    return lab, inertia / len(X)


def usage(lab: torch.Tensor) -> dict:
    counts = torch.bincount(lab, minlength=K).double()
    p = counts / counts.sum()
    s = counts.sort(descending=True).values
    nz = p[p > 0]
    return {"live_clusters": int((counts > 0).sum()), "empty_clusters": int((counts == 0).sum()),
            "top10_share": float(s[:10].sum() / counts.sum()),
            "usage_perplexity": float(math.exp(-(nz * nz.log()).sum())),
            "clusters_for_half_the_tokens": int((s.cumsum(0) < 0.5 * counts.sum()).sum()) + 1}


def lloyd(X: torch.Tensor, C: torch.Tensor):
    C = C.float().clone()
    prev, hist = None, []
    for it in range(MAX_ITER):
        lab, inertia = assign(X, C)
        changed = 1.0 if prev is None else float((lab != prev).float().mean())
        hist.append({"iter": it, "inertia": inertia, "changed": changed,
                     "live": int(torch.bincount(lab, minlength=K).gt(0).sum())})
        if prev is not None and changed < TOL:
            break
        prev = lab
        sums = torch.zeros_like(C)
        for i in range(0, len(X), 262144):
            sums.index_add_(0, lab[i:i + 262144], X[i:i + 262144].float())
        cnt = torch.bincount(lab, minlength=K).float()
        live = cnt > 0
        C[live] = sums[live] / cnt[live, None]
    return C, hist


def geometry(z: np.ndarray, labels: np.ndarray) -> dict:
    return {"silhouette": silhouette(z, labels, seed=0), "davies_bouldin": davies_bouldin(z, labels),
            "calinski_harabasz": calinski_harabasz(z, labels), "dunn_index": dunn_index(z, labels),
            "separability_ratio": separability_ratio(z, labels)}


t0 = time.time()
res = {"meta": {"K": K, "stride": STRIDE, "max_iter": MAX_ITER, "tol": TOL, "token_bias": tb_path,
                "space": "standardized residual minus token mean (normalization and table of the balanced token-removed baseline)",
                "geometry_sample": "1,000,000 rows, RandomState(0) over all rows (same draw as cq.json)"}}

# 1) fit plain k-means on the token-removed training rows, density-peaks seeds
X = rows_to(np.arange(0, v0, STRIDE), token_removed=True)
print(f"train rows {len(X):,} loaded ({time.time() - t0:.0f}s)", flush=True)
g = torch.Generator(device="cpu").manual_seed(0)
perm = torch.randperm(len(X), generator=g)
C0, _ = density_peaks_select(X[perm[:32768].to(dev)].float(), K, seed=0, density_power=1.0,
                             min_sep_frac=0.25, knn_k=32, refine_k=8)
C_pt, hist = lloyd(X, C0)
res["fit"] = {"iterations": len(hist), "history": hist, "seconds": round(time.time() - t0, 1)}
print(f"plain token-removed k-means: {len(hist)} iterations, last changed {hist[-1]['changed']:.4f}", flush=True)
src = np.load(KMT, allow_pickle=True)
np.savez(str(OUT_NPZ), centroids=C_pt.cpu().numpy().astype(np.float32), norm_mean=src["norm_mean"],
         norm_std=src["norm_std"], layer=src["layer"], n_clusters=K, model_name=src["model_name"],
         init="dpc", reinit="none", space="raw", ae_checkpoint="", token_bias=tb_path)
del X
torch.cuda.empty_cache()

# 2) usage on the held-out rows
Xh = rows_to(np.arange(v0, N), token_removed=True)
res["heldout_usage"] = {"plain_tokmean": usage(assign(Xh, C_pt)[0]), "balanced_tokmean": usage(assign(Xh, C_bt)[0])}
print("held-out usage:", res["heldout_usage"], flush=True)
del Xh
torch.cuda.empty_cache()

# 3) geometry on the cq.json sample, residual pair first (checks), then token-removed pair
idx = np.sort(np.random.RandomState(0).choice(N, 1_000_000, replace=False))
res["geometry"] = {}
for token_removed, arms in ((False, {"balanced_residual": C_b, "plain_residual": C_p}),
                            (True, {"balanced_tokmean": C_bt, "plain_tokmean": C_pt})):
    Z = rows_to(idx, token_removed=token_removed, dtype=torch.float32)
    z = Z.cpu().numpy()
    for name, C in arms.items():
        lab = assign(Z, C)[0].cpu().numpy()
        t1 = time.time()
        res["geometry"][name] = geometry(z, lab)
        print(f"geometry {name}: " + ", ".join(f"{k} {v:.4f}" for k, v in res["geometry"][name].items())
              + f" ({time.time() - t1:.0f}s)", flush=True)
    del Z, z
    torch.cuda.empty_cache()

OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT, f"({time.time() - t0:.0f}s total)")
