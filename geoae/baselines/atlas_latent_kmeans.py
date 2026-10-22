"""Plain k-means re-fit on the FINAL GeoAE latent, read out on the FineWeb-Atlas cache.

The token-bypass AE's own codebook is trained jointly with its latent (Sinkhorn, pull, separation hinge). This fits a
fresh codebook on the same latent z with exactly the plain baselines' procedure (experiments/paper/analysis/paper_plain_kmeans_grid.py):
Lloyd's algorithm on every 6th row of the training 95% of the dump, density-peaks seeds with the AE's settings (pool
32,768, seed 0 as the Gemma plain fits), no balancing, no re-seeding, empty clusters left in place, <= 100 iterations,
stop when < 1e-4 of the labels change.

Then every Atlas cache row is labelled by nearest centroid in z (concept_probe.assign, 'aekm' arm) as `bypass_plain`,
next to `bypass` (the trained codebook; identical to the original assign.npz, which atlas_concept_stats checks when the
files are merged), with the spine arrays of geoae/evaluation/atlas_assign.py.

Outputs:
  --codebook                       centroids in the fit_balanced_kmeans --space latent format (concept_probe reads it)
  <out_dir>/assign_latent_plain.npz labels bypass, bypass_plain + spine   (atlas_concept_stats --assign a.npz,b.npz)
  <out_dir>/latent_plain_fit.json  fit history, held-out usage of both codebooks, usage on the Atlas sampled rows

    python -u geoae/baselines/atlas_latent_kmeans.py --activations_dir activations_gemma3_4b_sampled_struct5_10M --layer 33 \
        --ckpt checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt \
        --km e2e/checkpoints/general/gemma3-4B/layer33/delta/balanced_kmeans_k2000_dpc_sampled_struct5.npz \
        --codebook e2e/checkpoints/general/gemma3-4B/layer33/delta/plain_kmeans_k2000_dpc_sampled_struct5_latent_bypass.npz \
        --cache /scratch/dogar/geoae/atlas_g4b_l33 --out_dir results/delta/g4b_l33_d5120/evals/sae_suite/atlas_concepts/plain_kmeans
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
import geoae.analysis.atlas_siblings as AS  # noqa: E402
import geoae.analysis.concept_probe as CP  # noqa: E402
from geoae.training.initialization import density_peaks_select  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--activations_dir", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--ckpt", required=True, help="the token-bypass GeoAE checkpoint")
ap.add_argument("--km", required=True, help="balanced residual k-means .npz (layer / model_name for the saved codebook)")
ap.add_argument("--codebook", required=True, help="where to save the latent codebook .npz")
ap.add_argument("--cache", required=True)
ap.add_argument("--rows_dir", default=None)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--stride", type=int, default=6)
ap.add_argument("--max_iter", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--label_only", action="store_true", help="skip the fit: load --codebook and redo the Atlas labels")
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
dev = torch.device("cuda")
TOL = 1e-4
BLOCK = 65536
t0 = time.time()


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, f"({time.time() - t0:.0f}s)", flush=True)


ae, mean, std, _ = load_ae_checkpoint(args.ckpt, dev, allow_token_bias=True)
ae.eval()
assert ae.has_token_bias, "expected the token-bypass AE"
C_tr = ae.centroids.float()
K, dz = C_tr.shape
D = Path(args.activations_dir)
A = np.load(D / f"layer_{args.layer}.npy", mmap_mode="r")
TOK = np.load(D / "rows_tok.npy", mmap_mode="r")
N = A.shape[0]
v0 = int(N * 0.95)


def encode(h: np.ndarray, t: np.ndarray) -> torch.Tensor:
    h = torch.from_numpy(np.asarray(h, dtype=np.float32)).to(dev)
    t = torch.from_numpy(np.asarray(t, dtype=np.int64)).to(dev)
    return ae.encode((h - mean) / std, t)


def encode_range(lo: int, hi: int, stride: int) -> torch.Tensor:
    """z of dump rows lo, lo + stride, ... < hi, read in contiguous slabs (fast on GPFS) -> float16 on the GPU."""
    assert lo % stride == 0
    rows = np.arange(lo, hi, stride)
    out = torch.empty(len(rows), dz, dtype=torch.float16, device=dev)
    slab = stride * BLOCK
    j = 0
    for s in range(lo, hi, slab):
        e = min(s + slab, hi)
        h = np.asarray(A[s:e])[::stride]
        t = np.asarray(TOK[s:e])[::stride]
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
            "eff_k_exp_entropy": float(math.exp(-(nz * nz.log()).sum())),
            "eff_k_inverse_simpson": float(1.0 / (nz * nz).sum()),
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


if args.label_only:
    C = torch.as_tensor(np.load(args.codebook)["centroids"], dtype=torch.float32, device=dev)
    res = json.loads((Path(args.out_dir) / "latent_plain_fit.json").read_text())
    log("label only: loaded", args.codebook, tuple(C.shape))
else:
  res = {"meta": {"ckpt": args.ckpt, "activations_dir": str(D), "layer": args.layer, "K": K, "latent_dim": dz,
                  "stride": args.stride, "max_iter": args.max_iter, "tol": TOL, "seed": args.seed,
                  "fit_rows": f"every {args.stride}th row of [0, {v0:,})", "heldout_rows": f"[{v0:,}, {N:,})"}}

  # ---- fit
  X = encode_range(0, v0, args.stride)
  log(f"encoded {len(X):,} training rows -> z {tuple(X.shape)}, {X.numel() * 2 / 2**30:.1f} GiB")
  g = torch.Generator(device="cpu").manual_seed(args.seed)
  perm = torch.randperm(len(X), generator=g)
  C0, _ = density_peaks_select(X[perm[:32768].to(dev)].float(), K, seed=args.seed, density_power=1.0,
                               min_sep_frac=0.25, knn_k=32, refine_k=8)
  log("density-peaks seeds", tuple(C0.shape))
  C, hist = lloyd(X, C0)
  res["fit"] = {"iterations": len(hist), "last_changed": hist[-1]["changed"], "history": hist,
                "train_usage_plain": usage(assign(X, C)[0]), "train_usage_trained": usage(assign(X, C_tr)[0])}
  log(f"Lloyd: {len(hist)} iterations, last changed {hist[-1]['changed']:.5f}, live {hist[-1]['live']}")
  del X
  torch.cuda.empty_cache()

  src = np.load(args.km, allow_pickle=True)
  cb = Path(args.codebook)
  cb.parent.mkdir(parents=True, exist_ok=True)
  np.savez(str(cb), centroids=C.cpu().numpy().astype(np.float32), norm_mean=mean.cpu().numpy(),
           norm_std=std.cpu().numpy(), layer=src["layer"], n_clusters=K, model_name=src["model_name"],
           init="dpc", reinit="none", space="latent", ae_checkpoint=args.ckpt, fit="plain Lloyd")
  log("saved", cb)

  # ---- held-out usage
  Xh = encode_range(v0, N, 1)
  res["heldout_usage"] = {"bypass_plain": usage(assign(Xh, C)[0]), "bypass_trained": usage(assign(Xh, C_tr)[0])}
  log("held-out usage", json.dumps(res["heldout_usage"]))
  del Xh
  torch.cuda.empty_cache()

# ---- Atlas labels (same blocks and assignment code as atlas_token_detail / atlas_assign_extra; full fp32 as they ran:
# TF32 flips near-tie rows, 2,960 of 899,676 for the trained codebook)
torch.backends.cuda.matmul.allow_tf32 = False
cache = AS.AtlasCache(args.cache, args.rows_dir)
row_pos = np.asarray(np.load(Path(args.rows_dir or args.cache) / "rows_pos.npy", mmap_mode="r"), dtype=np.int64)
tok_all = np.asarray(cache.tok, dtype=np.int64)
chunk_all = np.asarray(cache.row_chunk, dtype=np.int64)
arms = {"bypass": ("ae", (ae, mean, std)), "bypass_plain": ("aekm", (ae, mean, std, C))}
lab = {}
for name, arm in arms.items():
    parts = [CP.assign(np.asarray(cache.rows[i:i + BLOCK]), arm, dev, ids=tok_all[i:i + BLOCK])
             for i in range(0, cache.n_rows, BLOCK)]
    lab[name] = np.concatenate(parts).astype(np.int32)
    log(f"labels {name}: {len(np.unique(lab[name])):,} distinct")
ntok = np.asarray(cache.n_tok, dtype=np.int64)
sampled = row_pos != ntok[chunk_all] - 1
res["atlas_sampled_usage"] = {n: usage(l[sampled]) for n, l in lab.items()}
res["atlas_agreement_bypass_vs_plain"] = float((lab["bypass"] == lab["bypass_plain"]).mean())
log("Atlas sampled-row usage", json.dumps(res["atlas_sampled_usage"]))

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "assign_latent_plain.npz", **lab, row_chunk=chunk_all.astype(np.int32),
                    row_pos=row_pos.astype(np.int32), row_tok=tok_all.astype(np.int32),
                    chunk_doc=cache.doc.astype(np.int64), chunk_ntok=ntok.astype(np.int32))
(out / "latent_plain_fit.json").write_text(json.dumps(res, indent=1))
log("wrote", out / "assign_latent_plain.npz", "and", out / "latent_plain_fit.json")
