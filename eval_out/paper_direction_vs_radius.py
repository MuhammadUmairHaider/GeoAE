"""Where does cluster separation live: in directions or in radii?

For each space (normalised residual with balanced k-means, token-centred residual with its k-means, GeoAE
latent with its prototypes), centre the held-out rows and compute the between/within variance ratio of
the space's own clusters for three views of each vector v:
    full     v                 (Euclidean separation, what Calinski-Harabasz measures)
    angular  v / ||v||         (direction only, what cosine measures)
    radial   log ||v||         (length only)
Also the angle between cluster centroids (mean pairwise cosine distance of centred centroids).

    scripts/delta/py eval_out/paper_direction_vs_radius.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.losses import pairwise_sq_dist
from geoae.token_bias import TokenBiasLookup

dev = torch.device("cuda")
torch.set_grad_enabled(False)
D = Path("activations_sampled_10M")
A = np.load(D / "layer_27.npy", mmap_mode="r")
v0 = int(A.shape[0] * 0.95)
H = np.asarray(A[v0:], dtype=np.float16)
tok = np.asarray(np.load(D / "rows_tok.npy", mmap_mode="r")[v0:], dtype=np.int64)
# DVR_KM / DVR_KMT / DVR_CK / DVR_OUT override the K=2000 d6144 defaults (e.g. for the K=4000 arms).
KM = os.environ.get("DVR_KM", "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled.npz")
KMT = os.environ.get("DVR_KMT", "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled_tokmean.npz")
CK = os.environ.get("DVR_CK", "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt")
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
        yield (torch.from_numpy(H[i:i + bs].copy()).to(dev).float(), torch.from_numpy(tok[i:i + bs]).to(dev))


mean = {}
for h, t in batches():
    for k, (v, _) in spaces(h, t).items():
        mean[k] = mean.get(k, 0) + v.double().sum(0)
mean = {k: (v / len(H)).float() for k, v in mean.items()}

acc = {}
for h, t in batches():
    for k, (v, lab) in spaces(h, t).items():
        v = v - mean[k]
        nrm = v.norm(dim=1, keepdim=True)
        views = {"full": v.double(), "angular": (v / nrm).double(), "radial": nrm.log().double()}
        for name, w in views.items():
            a = acc.setdefault((k, name), {"S": torch.zeros(K, w.shape[1], dtype=torch.float64, device=dev),
                                           "Q": torch.zeros(K, dtype=torch.float64, device=dev),
                                           "n": torch.zeros(K, dtype=torch.float64, device=dev)})
            a["S"].index_add_(0, lab, w)
            a["Q"].index_add_(0, lab, (w ** 2).sum(1))
            a["n"].index_add_(0, lab, torch.ones_like(lab, dtype=torch.float64))

res = {}
for (k, name), a in acc.items():
    n, S, Q = a["n"], a["S"], a["Q"]
    live = n > 0
    tot = Q.sum() - (S.sum(0) ** 2).sum() / n.sum()
    within = (Q[live] - (S[live] ** 2).sum(1) / n[live]).sum()
    res.setdefault(k, {})[name] = ((tot - within) / within).item()
    if name == "full":
        cen = S[live] / n[live, None]
        cu = torch.nn.functional.normalize(cen, dim=1)
        m_ = len(cu)
        res[k]["centroid_cosine_dispersion"] = (1 - ((cu.sum(0) ** 2).sum() - m_) / (m_ * (m_ - 1))).item()
        res[k]["centroid_norm_cv"] = (cen.norm(dim=1).std() / cen.norm(dim=1).mean()).item()

for k in ("residual", "token_centred", "geoae"):
    a = acc[(k, "angular")]
    n, S = a["n"], a["S"]
    big = n > 1
    wc = 1 - ((S[big] ** 2).sum(1) - n[big]) / (n[big] * (n[big] - 1))      # per-cluster within cosine distance
    res[k]["within_cos_pair_weighted"] = (((S[big] ** 2).sum(1) - n[big]).sum() * -1 / (n[big] * (n[big] - 1)).sum() + 1).item()
    res[k]["within_cos_point_weighted"] = ((wc * n[big]).sum() / n[big].sum()).item()
    res[k]["within_cos_cluster_mean"] = wc.mean().item()
    res[k]["top10_share"] = (n.sort(descending=True).values[:10].sum() / n.sum()).item()
    order = n.argsort(descending=True)[:10]
    res[k]["within_cos_top10_clusters"] = (1 - ((S[order] ** 2).sum(1) - n[order]) / (n[order] * (n[order] - 1))).mean().item()
print(f"{'space':14s} {'pair-wtd':>9s} {'point-wtd':>10s} {'cluster-mean':>13s} {'top10 share':>12s} {'top10 within':>13s}   (within-cluster cosine distance; lower = tighter)")
for k in ("residual", "token_centred", "geoae"):
    r = res[k]
    print(f"{k:14s} {r['within_cos_pair_weighted']:9.3f} {r['within_cos_point_weighted']:10.3f} {r['within_cos_cluster_mean']:13.3f} {r['top10_share']:12.3f} {r['within_cos_top10_clusters']:13.3f}")
Path(os.environ.get("DVR_OUT", "results/delta/d6144/evals/direction_vs_radius.json")).write_text(json.dumps(res, indent=1))
print(f"{'space':14s} {'B/W full':>9s} {'B/W angle':>10s} {'B/W radius':>11s} {'centroid angle disp':>20s} {'centroid norm CV':>17s}")
for k, r in res.items():
    print(f"{k:14s} {r['full']:9.3f} {r['angular']:10.3f} {r['radial']:11.3f} {r['centroid_cosine_dispersion']:20.3f} {r['centroid_norm_cv']:17.3f}")
