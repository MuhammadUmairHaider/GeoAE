"""Plain (unbalanced) k-means on the final Llama-3.2-3B layer-27 dump: how concentrated are its clusters?

The paper's claim "plain k-means collapses" was last measured on an older dump (MiniBatchKMeans, see the
docstring of geoae/interp/fit_balanced_kmeans.py). This re-measures it under the paper's protocol:
  - space: the standardized residual stream, with the normalization of the residual (balanced) k-means baseline;
  - fit on a strided subsample of the first 95% of rows, evaluate on the held-out last 5% (500,000 rows);
  - Lloyd's algorithm to convergence with no balancing and no re-initialization of empty clusters
    (textbook k-means), from two initializations: k-means++ and the density-peaks init of the residual baseline.
The balanced residual k-means baseline is evaluated on the same held-out rows for reference.

    .venv/bin/python eval_out/paper_plain_kmeans_collapse.py      (on the box; ~5 min on one A100)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.interp.fit_balanced_kmeans import kmeanspp_init
from geoae.seeded_init import density_peaks_select

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
dev = torch.device("cuda")
K = 2000
STRIDE = 6            # ~1.58M training rows
MAX_ITER = 100
TOL = 1e-4            # stop when fewer than 0.01% of assignments change
D = Path("activations_sampled_10M")
KM = Path("e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled.npz")
OUT = Path("results/delta/d6144/evals/plain_kmeans_collapse.json")

A = np.load(D / "layer_27.npy", mmap_mode="r")
N = A.shape[0]
v0 = int(N * 0.95)
C_bal, mu, sd, k_bal, _, _ = load_baseline_kmeans(KM, dev)
assert k_bal == K


def standardized(rows: np.ndarray) -> torch.Tensor:
    out = torch.empty(len(rows), A.shape[1], dtype=torch.float16, device=dev)
    for i in range(0, len(rows), 65536):
        h = torch.from_numpy(np.asarray(A[rows[i:i + 65536]], dtype=np.float32)).to(dev)
        out[i:i + len(h)] = ((h - mu) / sd).half()
    return out


def assign(X: torch.Tensor, C: torch.Tensor, chunk: int = 65536):
    c2 = (C * C).sum(1)
    lab = torch.empty(len(X), dtype=torch.long, device=dev)
    inertia = 0.0
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].float()
        d = c2[None, :] - 2.0 * (x @ C.T)
        m, a = d.min(1)
        lab[i:i + chunk] = a
        inertia += float((m + (x * x).sum(1)).sum())
    return lab, inertia / len(X)


def usage(lab: torch.Tensor) -> dict:
    counts = torch.bincount(lab, minlength=K).double()
    p = counts / counts.sum()
    s = counts.sort(descending=True).values
    nz = p[p > 0]
    return {"live_clusters": int((counts > 0).sum()), "empty_clusters": int((counts == 0).sum()),
            "top1_share": float(s[0] / counts.sum()), "top10_share": float(s[:10].sum() / counts.sum()),
            "top100_share": float(s[:100].sum() / counts.sum()),
            "usage_perplexity": float(math.exp(-(nz * nz.log()).sum())),
            "clusters_for_half_the_tokens": int((s.cumsum(0) < 0.5 * counts.sum()).sum()) + 1}


def lloyd(X: torch.Tensor, C: torch.Tensor) -> tuple[torch.Tensor, list]:
    C = C.float().clone()
    prev = None
    hist = []
    for it in range(MAX_ITER):
        lab, inertia = assign(X, C)
        changed = 1.0 if prev is None else float((lab != prev).float().mean())
        hist.append({"iter": it, "inertia": inertia, "changed": changed, "live": int(torch.bincount(lab, minlength=K).gt(0).sum())})
        if prev is not None and changed < TOL:
            break
        prev = lab
        sums = torch.zeros_like(C).index_add_(0, lab, X.float()) if len(X) <= 400_000 else None
        if sums is None:
            sums = torch.zeros_like(C)
            for i in range(0, len(X), 262144):
                sums.index_add_(0, lab[i:i + 262144], X[i:i + 262144].float())
        cnt = torch.bincount(lab, minlength=K).float()
        live = cnt > 0
        C[live] = sums[live] / cnt[live, None]   # empty clusters keep their position (textbook k-means)
    return C, hist


t0 = time.time()
train_rows = np.arange(0, v0, STRIDE)
held_rows = np.arange(v0, N)
X = standardized(train_rows)
Xh = standardized(held_rows)
print(f"loaded train {len(X):,} held-out {len(Xh):,} rows in {time.time() - t0:.0f}s", flush=True)

res = {"meta": {"K": K, "train_rows": len(X), "heldout_rows": len(Xh), "stride": STRIDE, "max_iter": MAX_ITER,
                "tol": TOL, "space": "standardized residual (normalization of the residual k-means baseline)",
                "dump": str(D), "baseline": str(KM)}}
lab_b, _ = assign(Xh, C_bal.float())
res["balanced_residual_kmeans"] = {"heldout": usage(lab_b)}
print("balanced residual k-means, held-out:", res["balanced_residual_kmeans"]["heldout"], flush=True)

g = torch.Generator(device="cpu").manual_seed(0)
for init in ("kmeanspp", "density_peaks"):
    t1 = time.time()
    perm = torch.randperm(len(X), generator=g)
    if init == "kmeanspp":
        C0 = kmeanspp_init(X[perm[:200_000].to(dev)].float(), K, 0)
    else:
        C0, _ = density_peaks_select(X[perm[:32768].to(dev)].float(), K, seed=0, density_power=1.0,
                                     min_sep_frac=0.25, knn_k=32, refine_k=8)
    C, hist = lloyd(X, C0)
    lab_t, _ = assign(X, C)
    lab_h, _ = assign(Xh, C)
    npz = Path(f"e2e/checkpoints/general/llama3.2-3B/layer27/delta/plain_kmeans_k2000_{init}_sampled.npz")
    src = np.load(KM, allow_pickle=True)
    np.savez(str(npz), centroids=C.cpu().numpy().astype(np.float32), norm_mean=src["norm_mean"], norm_std=src["norm_std"],
             layer=src["layer"], n_clusters=K, model_name=src["model_name"])
    res[f"plain_kmeans_{init}"] = {"centroids": str(npz), "train": usage(lab_t), "heldout": usage(lab_h), "iterations": len(hist),
                                   "final_inertia": hist[-1]["inertia"], "history": hist,
                                   "seconds": round(time.time() - t1, 1)}
    print(f"plain k-means ({init}), {len(hist)} iterations, held-out:", res[f"plain_kmeans_{init}"]["heldout"], flush=True)

OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT, f"({time.time() - t0:.0f}s total)")
