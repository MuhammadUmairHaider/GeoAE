"""Representation dispersion (Li et al., 2026, "On the Predictive Power of Representation Dispersion in
Language Models"): the mean pairwise cosine distance among hidden vectors, for the base residual stream
and the GeoAE latent, on the held-out 5% of the Llama-3.2-3B L27 sampled dump.

Closed form, no pairwise matrix: for unit vectors u_i, sum_{i != j} u_i . u_j = ||sum_i u_i||^2 - n, so
    dispersion = 1 - (||S||^2 - n) / (n (n - 1)),  S = sum_i u_i.
Their Sec. 2.3 splits distances into within-cluster and between-cluster pairs; with per-cluster sums S_c
    within  = 1 - sum_c (||S_c||^2 - n_c) / sum_c n_c (n_c - 1)
    between = 1 - [(||S||^2 - n) - sum_c (||S_c||^2 - n_c)] / [n (n - 1) - sum_c n_c (n_c - 1)].
Each space uses its own codebook: balanced k-means on the residual, balanced k-means after token
centring, GeoAE's prototypes on its latent. Every number is reported raw (the paper's metric) and
mean-centred, since the GELU latent is mostly positive and a shared offset alone shrinks raw cosines.
Uncertainty: five document-disjoint folds (doc id mod 5); intervals are the fold range.

    scripts/delta/py eval_out/paper_dispersion.py [--out results/delta/d6144/evals/dispersion.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.losses import pairwise_sq_dist
from geoae.token_bias import TokenBiasLookup

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="activations_sampled_10M")
ap.add_argument("--layer", type=int, default=27)
ap.add_argument("--ckpt", default="checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt")
ap.add_argument("--km", default="e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled.npz")
ap.add_argument("--km_tok", default="e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled_tokmean.npz")
ap.add_argument("--val_frac", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=16384)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--out", default="results/delta/d6144/evals/dispersion.json")
args = ap.parse_args()

t0 = time.time()
dev = torch.device("cuda")
torch.set_grad_enabled(False)
d = Path(args.dump)
A = np.load(d / f"layer_{args.layer}.npy", mmap_mode="r")
N = A.shape[0]
v0 = int(N * (1.0 - args.val_frac))
H = np.asarray(A[v0:], dtype=np.float16)                      # held-out rows only
tok = np.asarray(np.load(d / "rows_tok.npy", mmap_mode="r")[v0:], dtype=np.int64)
doc = np.asarray(np.load(d / "rows_doc.npy", mmap_mode="r")[v0:], dtype=np.int64)
pos = np.asarray(np.load(d / "rows_pos.npy", mmap_mode="r")[v0:], dtype=np.int64)
n = len(H)
fold = doc % args.folds
print(f"[disp] {n:,} held-out rows from {len(np.unique(doc)):,} docs (rows {v0:,}..{N:,}), read {time.time()-t0:.0f}s")

C, m_b, s_b, K, _, _ = load_baseline_kmeans(Path(args.km), dev)
C2, m_t, s_t, K2, _, _ = load_baseline_kmeans(Path(args.km_tok), dev, allow_token_bias=True)
tb = TokenBiasLookup(str(np.load(args.km_tok, allow_pickle=True)["token_bias"]), dev)
ae, m_a, s_a, _ = load_ae_checkpoint(args.ckpt, dev, allow_token_bias=True)
ae.eval()


def spaces(h, t):
    """{name: (vectors, cluster ids)} for one batch."""
    x = (h - m_b) / s_b
    xt = (h - m_t) / s_t - tb(t)
    z = ae.encode((h - m_a) / s_a, t)
    lab_b = pairwise_sq_dist(x, C, metric="euclidean").argmin(1)
    return {
        "residual_raw": (h, lab_b),            # the paper's object: un-normalised final-layer states
        "residual": (x, lab_b),                # normalised residual, where balanced k-means clusters
        "token_centred": (xt, pairwise_sq_dist(xt, C2, metric="euclidean").argmin(1)),
        "geoae": (z, pairwise_sq_dist(z, ae.centroids, metric=ae.metric).argmin(1)),
    }


def batches():
    for i in range(0, n, args.batch):
        h = torch.from_numpy(H[i:i + args.batch]).to(dev).float()
        t = torch.from_numpy(tok[i:i + args.batch]).to(dev)
        f = torch.from_numpy(fold[i:i + args.batch]).to(dev)
        yield i, h, t, f


# pass 1: per-space means (for the centred variant)
mean = {}
for _, h, t, _ in batches():
    for k, (v, _) in spaces(h, t).items():
        mean[k] = mean.get(k, 0) + v.double().sum(0)
mean = {k: (v / n).float() for k, v in mean.items()}

# pass 2: per-fold, per-cluster sums of unit vectors
F = args.folds
acc = {}
for _, h, t, f in batches():
    for k, (v, lab) in spaces(h, t).items():
        kk = K if k != "geoae" else ae.n_clusters
        for var, w in (("raw", v), ("centred", v - mean[k])):
            u = torch.nn.functional.normalize(w, dim=1).double()
            a = acc.setdefault((k, var), {"S": torch.zeros(F, kk, u.shape[1], dtype=torch.float64, device=dev),
                                          "n": torch.zeros(F, kk, dtype=torch.float64, device=dev)})
            idx = f * kk + lab
            a["S"].view(F * kk, -1).index_add_(0, idx, u)
            a["n"].view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
print(f"[disp] encoded and accumulated in {time.time()-t0:.0f}s")


def stats(S, cnt):
    """S (K, d), cnt (K,) -> dispersion, within, between, within/between pair shares."""
    n_ = cnt.sum()
    tot = (S.sum(0) ** 2).sum() - n_                 # sum_{i != j} u_i.u_j
    w = ((S ** 2).sum(1) - cnt).sum()
    wp = (cnt * (cnt - 1)).sum()
    tp = n_ * (n_ - 1)
    disp = 1 - tot / tp
    within = 1 - w / wp
    between = 1 - (tot - w) / (tp - wp)
    return {"dispersion": disp.item(), "within": within.item(), "between": between.item(),
            "between_over_within": (between / within).item()}


res = {"meta": {"paper": "Li et al. 2026, arXiv 2506.24106", "rows": n, "dump": str(d), "val_start": v0,
                "checkpoint": args.ckpt, "km": args.km, "km_tok": args.km_tok, "folds": F,
                "metric": "mean pairwise cosine distance; within/between use each space's own codebook"},
       "token_level": {}, "doc_level": {}}
for (k, var), a in acc.items():
    full = stats(a["S"].sum(0), a["n"].sum(0))
    per = [stats(a["S"][i], a["n"][i]) for i in range(F)]
    full["fold_range"] = {m: [min(p[m] for p in per), max(p[m] for p in per)] for m in full}
    res["token_level"].setdefault(k, {})[var] = full

# document level, as in the paper: one vector per text (here the latest sampled position of each doc)
order = np.lexsort((pos, doc))
last = order[np.r_[np.flatnonzero(np.diff(doc[order])), len(order) - 1]]
hd = torch.from_numpy(H[last]).to(dev).float()
td = torch.from_numpy(tok[last]).to(dev)
fd = fold[last]
for k, (v, _) in spaces(hd, td).items():
    for var, w in (("raw", v), ("centred", v - v.mean(0))):
        u = torch.nn.functional.normalize(w, dim=1).double()

        def disp(uu):
            m_ = len(uu)
            return (1 - ((uu.sum(0) ** 2).sum() - m_) / (m_ * (m_ - 1))).item()
        per = [disp(u[torch.from_numpy(fd == i).to(dev)]) for i in range(F)]
        res["doc_level"].setdefault(k, {})[var] = {"dispersion": disp(u), "fold_range": [min(per), max(per)],
                                                   "n_docs": int(len(last))}

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1))
print(f"[disp] wrote {out} ({time.time()-t0:.0f}s)\n")
print(f"{'space':15s} {'variant':8s} {'disp(tok)':>10s} {'within':>8s} {'between':>8s} {'b/w':>6s} {'disp(doc)':>10s}")
for k in res["token_level"]:
    for var in ("raw", "centred"):
        r, dd = res["token_level"][k][var], res["doc_level"][k][var]
        print(f"{k:15s} {var:8s} {r['dispersion']:10.4f} {r['within']:8.4f} {r['between']:8.4f} "
              f"{r['between_over_within']:6.3f} {dd['dispersion']:10.4f}   folds b/w {r['fold_range']['between_over_within'][0]:.3f}-{r['fold_range']['between_over_within'][1]:.3f}")
