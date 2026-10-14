"""Concepts as ranges along a shared direction (NeuronLens, generalised from neuron axes to any direction).

For each space (normalised residual with balanced k-means; token-centred residual with its k-means;
GeoAE latent with its prototypes), on the
held-out 5% of the Llama-3.2-3B L27 dump:
  1. centre the rows; take each cluster's centred member mean mu_c (clusters with >= MIN_N held-out rows);
  2. pair every cluster with the cluster whose mean points most nearly the same way (max cosine);
  3. project both clusters' members onto their shared direction u = normalise(mu_a/|mu_a| + mu_b/|mu_b|)
     and measure how well that single coordinate separates them: d' = |m_a - m_b| / sqrt((s_a^2 + s_b^2)/2),
     AUROC ~ Phi(d'/sqrt 2) under equal-variance Gaussians;
  4. report the share of each pair's centroid gap that lies along the shared direction (radial share).
A "range pair" is two clusters that point the same way (cosine >= COS) yet separate along that one
direction (d' >= DPRIME): two concepts at different ranges of one direction.

    scripts/delta/py eval_out/paper_ranges_along_directions.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.losses import pairwise_sq_dist
from geoae.token_bias import TokenBiasLookup

MIN_N, COS, DPRIME = 50, 0.9, 2.0
dev = torch.device("cuda")
torch.set_grad_enabled(False)
D = Path("activations_sampled_10M")
A = np.load(D / "layer_27.npy", mmap_mode="r")
v0 = int(A.shape[0] * 0.95)
H = np.asarray(A[v0:], dtype=np.float16)
tok = np.asarray(np.load(D / "rows_tok.npy", mmap_mode="r")[v0:], dtype=np.int64)
# RANGES_KM / RANGES_KMT / RANGES_CK / RANGES_OUT override the K=2000 d6144 defaults (e.g. for the K=4000 arm).
KM = os.environ.get("RANGES_KM", "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled.npz")
KMT = os.environ.get("RANGES_KMT", "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled_tokmean.npz")
CK = os.environ.get("RANGES_CK", "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt")
C, m_b, s_b, K, _, _ = load_baseline_kmeans(Path(KM), dev)
C2, m_t, s_t, _, _, _ = load_baseline_kmeans(Path(KMT), dev, allow_token_bias=True)
tb = TokenBiasLookup(str(np.load(KMT, allow_pickle=True)["token_bias"]), dev)
ae, m_a, s_a, _ = load_ae_checkpoint(CK, dev, allow_token_bias=True)
ae.eval()


def spaces(h, t):
    x = (h - m_b) / s_b
    xt = (h - m_t) / s_t - tb(t)
    z = ae.encode((h - m_a) / s_a, t)
    return {"residual": (x, pairwise_sq_dist(x, C, metric="euclidean").argmin(1)),
            "token_centred": (xt, pairwise_sq_dist(xt, C2, metric="euclidean").argmin(1)),
            "geoae": (z, pairwise_sq_dist(z, ae.centroids, metric=ae.metric).argmin(1))}


def batches(bs=16384):
    for i in range(0, len(H), bs):
        yield torch.from_numpy(H[i:i + bs].copy()).to(dev).float(), torch.from_numpy(tok[i:i + bs]).to(dev)


# pass 1: means and per-cluster sums of centred vectors
tot, S, n = {}, {}, {}
for h, t in batches():
    for k, (v, lab) in spaces(h, t).items():
        tot[k] = tot.get(k, 0) + v.double().sum(0)
        S.setdefault(k, torch.zeros(K, v.shape[1], dtype=torch.float64, device=dev)).index_add_(0, lab, v.double())
        n.setdefault(k, torch.zeros(K, dtype=torch.float64, device=dev)).index_add_(0, lab, torch.ones_like(lab, dtype=torch.float64))
N = len(H)
pairs = {}
for k in S:
    mean = tot[k] / N
    mu = S[k] / n[k].clamp_min(1)[:, None] - mean                 # centred member means
    ok = n[k] >= MIN_N
    U = torch.nn.functional.normalize(mu, dim=1)
    G = U @ U.T
    G[~ok] = -2
    G[:, ~ok] = -2
    G.fill_diagonal_(-2)
    best_cos, best = G.max(1)
    a = torch.nonzero(ok).squeeze(1)
    b = best[a]
    key = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], 1).unique(dim=0)   # unique unordered pairs
    pa, pb = key[:, 0], key[:, 1]
    u = torch.nn.functional.normalize(U[pa] + U[pb], dim=1)
    gap = mu[pa] - mu[pb]
    radial_share = ((gap * u).sum(1) ** 2) / (gap ** 2).sum(1)
    pairs[k] = {"mean": mean.float(), "pa": pa, "pb": pb, "u": u.float(), "cos": (U[pa] * U[pb]).sum(1),
                "norm_ratio": torch.maximum(mu[pa].norm(dim=1), mu[pb].norm(dim=1)) / torch.minimum(mu[pa].norm(dim=1), mu[pb].norm(dim=1)),
                "radial_share": radial_share, "best_cos_all": best_cos[ok]}

# pass 2: moments of each pair's members projected on the pair's shared direction
mom = {k: torch.zeros(len(p["pa"]), 6, dtype=torch.float64, device=dev) for k, p in pairs.items()}
for h, t in batches():
    for k, (v, lab) in spaces(h, t).items():
        p = pairs[k]
        proj = ((v - p["mean"]) @ p["u"].T).double()                # (B, n_pairs)
        ina = (lab[:, None] == p["pa"][None, :]).double()
        inb = (lab[:, None] == p["pb"][None, :]).double()
        mom[k] += torch.stack([ina.sum(0), (ina * proj).sum(0), (ina * proj ** 2).sum(0),
                               inb.sum(0), (inb * proj).sum(0), (inb * proj ** 2).sum(0)], 1)

res, tables = {}, {}
for k, p in pairs.items():
    m = mom[k]
    ma, mb = m[:, 1] / m[:, 0], m[:, 4] / m[:, 3]
    va, vb = m[:, 2] / m[:, 0] - ma ** 2, m[:, 5] / m[:, 3] - mb ** 2
    dprime = (ma - mb).abs() / ((va + vb) / 2).sqrt()
    auroc = 0.5 * (1 + torch.erf(dprime / 2))                     # Phi(d'/sqrt2)
    rng = (p["cos"] >= COS) & (dprime >= DPRIME)
    res[k] = {"clusters_with_partner": int(len(p["best_cos_all"])), "unique_pairs": int(len(p["pa"])),
              "median_best_partner_cos": p["best_cos_all"].median().item(),
              "pairs_cos_ge": int((p["cos"] >= COS).sum()),
              "range_pairs": int(rng.sum()),
              "range_pairs_share_of_pairs": (rng.sum() / len(p["pa"])).item(),
              "median_dprime_among_parallel": dprime[p["cos"] >= COS].median().item() if (p["cos"] >= COS).any() else None,
              "median_radial_share_among_parallel": p["radial_share"][p["cos"] >= COS].median().item() if (p["cos"] >= COS).any() else None}
    order = torch.argsort(dprime * rng.double(), descending=True)[: int(rng.sum())]
    tables[k] = [{"a": int(p["pa"][i]), "b": int(p["pb"][i]), "cos": round(p["cos"][i].item(), 3),
                  "dprime": round(dprime[i].item(), 2), "auroc": round(auroc[i].item(), 3),
                  "radial_share": round(p["radial_share"][i].item(), 3), "norm_ratio": round(p["norm_ratio"][i].item(), 2),
                  "n_a": int(m[i, 0]), "n_b": int(m[i, 3]), "mean_a": round(ma[i].item(), 2), "mean_b": round(mb[i].item(), 2)}
                 for i in order.tolist()]

out = Path(os.environ.get("RANGES_OUT", "results/delta/d6144/evals/ranges_along_directions.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"meta": {"min_n": MIN_N, "cos": COS, "dprime": DPRIME, "rows": N}, "summary": res,
                           "range_pairs": tables}, indent=1))
for k, r in res.items():
    print(k, json.dumps(r))
print("wrote", out)
