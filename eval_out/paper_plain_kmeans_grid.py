"""Plain (unbalanced) k-means with density-peaks seeds, on the residual AND on the token-removed residual, for any model.
The model-agnostic version of paper_plain_kmeans_collapse.py + paper_plain_kmeans_tokmean.py (written for Llama L27);
it fills the 2 x 2 {residual, token-removed} x {plain, balanced}, all seeded like the AE.

  - fits: Lloyd's algorithm on every 6th row of the training 95%, density-peaks init with the AE's settings, no balancing,
    empty clusters left in place; each space uses the standardization (and token table) of its balanced baseline;
  - usage on the held-out 5% for all four codebooks;
  - geometry with clustering_quality's own metric functions on the 1M-row sample cq.json uses (RandomState(0) over all
    rows). The balanced residual row should reproduce the model's cq.json "base" entry;
  - saves both plain codebooks (the token-removed one with its token_bias field) for probe_chance_corrected and
    token_structure.

    PYTHONPATH=. .venv/bin/python eval_out/paper_plain_kmeans_grid.py \
      --activations_dir activations_gemma3_4b_sampled_struct5_10M --layer 33 \
      --km e2e/checkpoints/general/gemma3-4B/layer33/delta/balanced_kmeans_k2000_dpc_sampled_struct5.npz \
      --kmt e2e/checkpoints/general/gemma3-4B/layer33/delta/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz \
      --out results/delta/g4b_l33_d5120/evals/plain_kmeans_grid.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.interp.clustering_quality import (calinski_harabasz, davies_bouldin, dunn_index, separability_ratio,
                                             silhouette)
from geoae.seeded_init import density_peaks_select
from geoae.token_bias import TokenBiasLookup

ap = argparse.ArgumentParser()
ap.add_argument("--activations_dir", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--km", required=True, help="balanced residual k-means .npz")
ap.add_argument("--kmt", required=True, help="balanced token-removed k-means .npz (with token_bias)")
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=6)
ap.add_argument("--max_iter", type=int, default=100)
ap.add_argument("--n_geometry", type=int, default=1_000_000)
ap.add_argument("--seed", type=int, default=0, help="seed of the density-peaks pool draw and selection (models use 42)")
ap.add_argument("--tag", default="", help="suffix for the saved codebooks, e.g. _seed42")
ap.add_argument("--k", type=int, default=0, help="fit this many clusters instead of the balanced codebooks' K (the balanced "
                                                 "rows are then skipped: they have another K)")
ap.add_argument("--spaces", default="residual,tokmean", help="which plain fits to make")
args = ap.parse_args()
SPACES = [x for x in args.spaces.split(",") if x]

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
dev = torch.device("cuda")
TOL = 1e-4
D = Path(args.activations_dir)
A = np.load(D / f"layer_{args.layer}.npy", mmap_mode="r")
TOK = np.load(D / "rows_tok.npy", mmap_mode="r")
N = A.shape[0]
v0 = int(N * 0.95)
C_b, m_b, s_b, K, _, _ = load_baseline_kmeans(Path(args.km), dev)
C_bt, m_t, s_t, Kt, _, _ = load_baseline_kmeans(Path(args.kmt), dev, allow_token_bias=True)
assert K == Kt
K_SRC = K
if args.k:
    K = args.k
SAME_K = K == K_SRC
tb_path = str(np.load(args.kmt, allow_pickle=True)["token_bias"])
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
    for it in range(args.max_iter):
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
res = {"meta": {"activations_dir": str(D), "layer": args.layer, "K": K, "stride": args.stride, "max_iter": args.max_iter,
                "tol": TOL, "k_fit": K, "spaces": SPACES, "km": args.km, "kmt": args.kmt, "token_bias": tb_path, "seed": args.seed,
                "geometry_sample": f"{args.n_geometry:,} rows, RandomState(0) over all rows (same draw as cq.json)"},
       "fits": {}, "heldout_usage": {}, "geometry": {}}
plain = {}
for space, token_removed, src_npz in (("residual", False, args.km), ("tokmean", True, args.kmt)):
    if space not in SPACES:
        continue
    X = rows_to(np.arange(0, v0, args.stride), token_removed=token_removed)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    perm = torch.randperm(len(X), generator=g)
    C0, _ = density_peaks_select(X[perm[:32768].to(dev)].float(), K, seed=args.seed, density_power=1.0,
                                 min_sep_frac=0.25, knn_k=32, refine_k=8)
    C, hist = lloyd(X, C0)
    plain[space] = C
    src = np.load(src_npz, allow_pickle=True)
    out_npz = Path(src_npz).with_name(Path(src_npz).name.replace("balanced_kmeans", "plain_kmeans").replace(f"_k{K_SRC}_", f"_k{K}_").replace(".npz", f"{args.tag}.npz"))
    extra = {"token_bias": tb_path} if token_removed else {}
    np.savez(str(out_npz), centroids=C.cpu().numpy().astype(np.float32), norm_mean=src["norm_mean"],
             norm_std=src["norm_std"], layer=src["layer"], n_clusters=K, model_name=src["model_name"],
             init="dpc", reinit="none", space="raw", ae_checkpoint="", **extra)
    res["fits"][space] = {"npz": str(out_npz), "iterations": len(hist), "last_changed": hist[-1]["changed"],
                          "history": hist}
    print(f"plain {space}: {len(hist)} iterations, last changed {hist[-1]['changed']:.4f} -> {out_npz} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del X
    torch.cuda.empty_cache()
    Xh = rows_to(np.arange(v0, N), token_removed=token_removed)
    bal = C_bt if token_removed else C_b
    res["heldout_usage"][f"plain_{space}"] = usage(assign(Xh, C)[0])
    if SAME_K:
        res["heldout_usage"][f"balanced_{space}"] = usage(assign(Xh, bal)[0])
    print("held-out usage:", {k: v for k, v in res["heldout_usage"].items() if k.endswith(space)}, flush=True)
    del Xh
    torch.cuda.empty_cache()

idx = np.sort(np.random.RandomState(0).choice(N, args.n_geometry, replace=False))
for space, token_removed, bal in (("residual", False, C_b), ("tokmean", True, C_bt)):
    if space not in SPACES:
        continue
    Z = rows_to(idx, token_removed=token_removed, dtype=torch.float32)
    z = Z.cpu().numpy()
    for name, C in (((f"balanced_{space}", bal),) if SAME_K else ()) + ((f"plain_{space}", plain[space]),):
        lab = assign(Z, C)[0].cpu().numpy()
        t1 = time.time()
        res["geometry"][name] = geometry(z, lab)
        print(f"geometry {name}: " + ", ".join(f"{k} {v:.4f}" for k, v in res["geometry"][name].items())
              + f" ({time.time() - t1:.0f}s)", flush=True)
    del Z, z
    torch.cuda.empty_cache()

Path(args.out).write_text(json.dumps(res, indent=1))
print("wrote", args.out, f"({time.time() - t0:.0f}s total)")
