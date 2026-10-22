"""Plain Lloyd k-means re-fit on a GeoAE latent (token-bypass OR no-bypass AE), with held-out usage and 1M-row geometry
of both the AE's trained codebook and the re-fit one.

The AE's own codebook is trained jointly with its latent (Sinkhorn balancing, pull, separation hinge). This fits a fresh
codebook on the same latent z with exactly the plain baselines' procedure (experiments/paper/analysis/paper_plain_kmeans_grid.py): Lloyd's
algorithm on every --stride-th row of the training 95% of the dump, density-peaks seeds with the AE's settings (pool
32,768), no balancing, no re-seeding, empty clusters left in place, <= 100 iterations, stop when < 1e-4 of the labels
change. Generalises geoae/baselines/atlas_latent_kmeans.py (bypass only, Atlas readout) to either AE type.

Outputs:
  --codebook  centroids in the latent format; geoae/evaluation/alignment.py reads it as an aekm arm
              (name=aekm:<ckpt>+<codebook>)
  --out       fit history, held-out usage of both codebooks, NMI between their held-out partitions, and geometry
              (clustering_quality's metric functions) on the 1M rows cq.json uses (RandomState(0) over all rows);
              the "trained" geometry entry should reproduce the AE's cq.json row

    python -u geoae/baselines/latent_kmeans.py --activations_dir <dump> --layer 14 \
        --ckpt checkpoints/llama3.2-3B/layer14/<run>/step_0015250.pt \
        --codebook e2e/checkpoints/general/llama3.2-3B/layer14/delta/plain_kmeans_k2000_latent_<arm>_seed42.npz \
        --out results/.../latent_plain_fit_<arm>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geoae.checkpoint import load_ae_checkpoint  # noqa: E402
from geoae.evaluation.geometry import calinski_harabasz
from geoae.evaluation.geometry import davies_bouldin
from geoae.evaluation.geometry import dunn_index
from geoae.evaluation.geometry import effective_rank
from geoae.evaluation.geometry import inter_centroid_dist
from geoae.evaluation.geometry import separability_ratio
from geoae.evaluation.geometry import silhouette
from geoae.training.initialization import density_peaks_select  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--activations_dir", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--codebook", required=True, help="where to save the latent codebook .npz")
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=6)
ap.add_argument("--max_iter", type=int, default=100)
ap.add_argument("--seed", type=int, default=42, help="density-peaks pool draw and selection (mid-layer plain fits: 42)")
ap.add_argument("--n_geometry", type=int, default=1_000_000)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
dev = torch.device("cuda")
TOL = 1e-4
BLOCK = 65536
t0 = time.time()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, f"({time.time() - t0:.0f}s)", flush=True)


ae, mean, std, ckpt = load_ae_checkpoint(args.ckpt, dev, allow_token_bias=True)
ae.eval()
BYPASS = bool(ae.has_token_bias)
C_tr = ae.centroids.float()
K, dz = C_tr.shape
D = Path(args.activations_dir)
A = np.load(D / f"layer_{args.layer}.npy", mmap_mode="r")
TOK = np.load(D / "rows_tok.npy", mmap_mode="r") if BYPASS else None
N = A.shape[0]
v0 = int(N * 0.95)
log(f"{args.ckpt}: {'token-bypass' if BYPASS else 'no-bypass'} AE, latent {dz}, K {K}; dump {N:,} rows")


def encode(h: np.ndarray, t) -> torch.Tensor:
    h = torch.from_numpy(np.asarray(h, dtype=np.float32)).to(dev)
    t = None if t is None else torch.from_numpy(np.asarray(t, dtype=np.int64)).to(dev)
    return ae.encode((h - mean) / std, t)


def encode_range(lo: int, hi: int, stride: int) -> torch.Tensor:
    """z of dump rows lo, lo + stride, ... < hi, read in contiguous slabs -> float16 on the GPU."""
    assert lo % stride == 0
    rows = np.arange(lo, hi, stride)
    out = torch.empty(len(rows), dz, dtype=torch.float16, device=dev)
    slab = stride * BLOCK
    j = 0
    for s in range(lo, hi, slab):
        e = min(s + slab, hi)
        h = np.asarray(A[s:e])[::stride]
        t = None if TOK is None else np.asarray(TOK[s:e])[::stride]
        out[j:j + len(h)] = encode(h, t).half()
        j += len(h)
    assert j == len(rows)
    return out


def assign(X: torch.Tensor, C: torch.Tensor, chunk: int = BLOCK):
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


def usage(lab) -> dict:
    lab = torch.as_tensor(lab, device=dev).long()
    counts = torch.bincount(lab, minlength=K).double()
    p = counts / counts.sum()
    s = counts.sort(descending=True).values
    nz = p[p > 0]
    return {"live_clusters": int((counts > 0).sum()), "empty_clusters": int((counts == 0).sum()),
            "top10_share": float(s[:10].sum() / counts.sum()),
            "usage_perplexity": float(math.exp(-(nz * nz.log()).sum())),
            "clusters_for_half_the_tokens": int((s.cumsum(0) < 0.5 * counts.sum()).sum()) + 1}


def nmi_labels(a: torch.Tensor, b: torch.Tensor) -> float:
    """Arithmetic-mean NMI between two partitions of the same rows."""
    J = torch.bincount(a * K + b, minlength=K * K).double().reshape(K, K)
    P = J / J.sum()
    pa, pb = P.sum(1), P.sum(0)
    nz = P > 0
    mi = float((P[nz] * (P[nz] / (pa[:, None] * pb[None, :])[nz]).log()).sum())
    h = lambda p: float(-(p[p > 0] * p[p > 0].log()).sum())
    return mi / max((h(pa) + h(pb)) / 2, 1e-12)


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


res = {"meta": {"ckpt": args.ckpt, "bypass": BYPASS, "activations_dir": str(D), "layer": args.layer, "K": K,
                "latent_dim": dz, "stride": args.stride, "max_iter": args.max_iter, "tol": TOL, "seed": args.seed,
                "fit_rows": f"every {args.stride}th row of [0, {v0:,})", "heldout_rows": f"[{v0:,}, {N:,})",
                "geometry_sample": f"{args.n_geometry:,} rows, RandomState(0) over all rows (same draw as cq.json)"}}

# ---- fit
X = encode_range(0, v0, args.stride)
log(f"encoded {len(X):,} training rows -> z {tuple(X.shape)}, {X.numel() * 2 / 2**30:.1f} GiB")
g = torch.Generator(device="cpu").manual_seed(args.seed)
perm = torch.randperm(len(X), generator=g)
C0, _ = density_peaks_select(X[perm[:32768].to(dev)].float(), K, seed=args.seed, density_power=1.0,
                             min_sep_frac=0.25, knn_k=32, refine_k=8)
C, hist = lloyd(X, C0)
res["fit"] = {"iterations": len(hist), "last_changed": hist[-1]["changed"], "history": hist,
              "train_inertia_plain": assign(X, C)[1], "train_inertia_trained": assign(X, C_tr)[1]}
log(f"Lloyd: {len(hist)} iterations, last changed {hist[-1]['changed']:.5f}, live {hist[-1]['live']}; inertia plain "
    f"{res['fit']['train_inertia_plain']:.2f} vs trained {res['fit']['train_inertia_trained']:.2f}")
del X
torch.cuda.empty_cache()

cfg = ckpt["config"]
cb = Path(args.codebook)
cb.parent.mkdir(parents=True, exist_ok=True)
np.savez(str(cb), centroids=C.cpu().numpy().astype(np.float32), norm_mean=mean.cpu().numpy(),
         norm_std=std.cpu().numpy(), layer=args.layer, n_clusters=K,
         model_name=cfg.get("extraction", {}).get("model_name", ""), init="dpc", reinit="none", space="latent",
         ae_checkpoint=args.ckpt, fit="plain Lloyd")
log("saved", cb)

# ---- held-out usage and agreement
Xh = encode_range(v0, N, 1)
lp, lt = assign(Xh, C)[0], assign(Xh, C_tr)[0]
res["heldout_usage"] = {"plain": usage(lp), "trained": usage(lt)}
res["heldout_nmi_trained_vs_plain"] = nmi_labels(lt, lp)
log("held-out usage", json.dumps(res["heldout_usage"]), "NMI(trained, plain)", f"{res['heldout_nmi_trained_vs_plain']:.3f}")
del Xh, lp, lt
torch.cuda.empty_cache()

# ---- geometry on cq.json's 1M rows, nearest-centroid labels (fp32)
torch.backends.cuda.matmul.allow_tf32 = False
idx = np.sort(np.random.RandomState(0).choice(N, min(args.n_geometry, N), replace=False))
z = np.empty((len(idx), dz), dtype=np.float32)
labs = {"trained": np.empty(len(idx), np.int64), "plain": np.empty(len(idx), np.int64)}
for s in range(0, len(idx), 16384):
    r = idx[s:s + 16384]
    v = encode(A[r], None if TOK is None else TOK[r]).float()
    z[s:s + len(r)] = v.cpu().numpy()
    labs["trained"][s:s + len(r)] = torch.cdist(v, C_tr).argmin(1).cpu().numpy()
    labs["plain"][s:s + len(r)] = torch.cdist(v, C.float()).argmin(1).cpu().numpy()
log(f"geometry sample encoded {z.shape}")
res["geometry"] = {}
for name, cent in (("trained", C_tr), ("plain", C.float())):
    lab = labs[name]
    t1 = time.time()
    cent = cent.cpu().numpy()
    live = np.unique(lab)
    icd = inter_centroid_dist(cent[live])
    res["geometry"][name] = {"silhouette": silhouette(z, lab, seed=0), "davies_bouldin": davies_bouldin(z, lab),
                             "calinski_harabasz": calinski_harabasz(z, lab), "dunn_index": dunn_index(z, lab),
                             "separability_ratio": separability_ratio(z, lab), "effective_rank": effective_rank(cent[live]),
                             "inter_centroid_dist_mean": icd[0], "inter_centroid_dist_min": icd[1],
                             "live_clusters_in_sample": int(len(live))}
    log(f"geometry {name}: " + ", ".join(f"{k} {v:.4f}" for k, v in res["geometry"][name].items()) + f" ({time.time() - t1:.0f}s)")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
log("wrote", args.out)
