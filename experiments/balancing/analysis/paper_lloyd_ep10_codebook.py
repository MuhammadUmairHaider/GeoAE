"""Plain k-means on the epoch-10 GeoAE latent, before any clustering training: run B's starting codebook, re-fitted.

Run B (no Sinkhorn, dpc_lloyd init, no re-seeding) fitted plain k-means on its latent at the start of epoch 11 and only
then began the clustering phase. That codebook was not saved on its own, so this re-fits it with the same procedure on
B's end-of-epoch-10 checkpoint: encode 1,583,334 training rows (every 6th row of the first 95%), density-peaks seeds
from 32,768 of them (the trainer's settings, seed 42), then Lloyd's algorithm for up to 100 iterations with empty clusters
left in place. The rows differ from the trainer's random draw, so this is the same procedure, not the identical codebook.

The codebook is written into a copy of the checkpoint (centroids + centroids_initialized), so the paper's own evaluators
(probe_chance_corrected --models, token_structure --arms, clustering_quality --checkpoints) score it as an AE arm.

    PYTHONPATH=. .venv/bin/python experiments/balancing/analysis/paper_lloyd_ep10_codebook.py --ckpt <step_0002890.pt> --out_ckpt <codebook.pt>
"""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.training.initialization import density_peaks_select

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out_ckpt", required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--activations_dir", default="activations_sampled_10M")
ap.add_argument("--layer", type=int, default=27)
ap.add_argument("--stride", type=int, default=6)
ap.add_argument("--max_iter", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
dev = torch.device("cuda")
D = Path(args.activations_dir)
A = np.load(resolve_input_path(D / f"layer_{args.layer}.npy"), mmap_mode="r")
TOK = np.load(resolve_input_path(D / "rows_tok.npy"), mmap_mode="r")
N = A.shape[0]
v0 = int(N * 0.95)
ae, mu, sd, _ = load_ae_checkpoint(args.ckpt, dev, allow_token_bias=True)
ae.eval()
K = ae.n_clusters
t0 = time.time()


def encode(rows: np.ndarray) -> torch.Tensor:
    out = None
    for i in range(0, len(rows), 32768):
        r = rows[i:i + 32768]
        h = torch.from_numpy(np.asarray(A[r], dtype=np.float32)).to(dev)
        t = torch.from_numpy(np.asarray(TOK[r], dtype=np.int64)).to(dev)
        z = ae.encode((h - mu) / sd, t)
        if out is None:
            out = torch.empty(len(rows), z.shape[1], dtype=torch.float16, device=dev)
        out[i:i + len(z)] = z.half()
    return out


def assign(Z: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    c2 = (C * C).sum(1)
    lab = torch.empty(len(Z), dtype=torch.long, device=dev)
    for i in range(0, len(Z), 65536):
        lab[i:i + 65536] = (c2[None, :] - 2.0 * (Z[i:i + 65536].float() @ C.T)).argmin(1)
    return lab


def usage(lab: torch.Tensor) -> dict:
    counts = torch.bincount(lab, minlength=K).double()
    p = counts / counts.sum()
    s = counts.sort(descending=True).values
    nz = p[p > 0]
    return {"live_clusters": int((counts > 0).sum()), "empty_clusters": int((counts == 0).sum()),
            "top10_share": float(s[:10].sum() / counts.sum()),
            "usage_perplexity": float(math.exp(-(nz * nz.log()).sum()))}


Z = encode(np.arange(0, v0, args.stride))
print(f"encoded {len(Z):,} training latents ({time.time() - t0:.0f}s)", flush=True)
g = torch.Generator(device="cpu").manual_seed(args.seed)
perm = torch.randperm(len(Z), generator=g)
C, _ = density_peaks_select(Z[perm[:32768].to(dev)].float(), K, seed=args.seed, density_power=1.0,
                            min_sep_frac=0.25, knn_k=32, refine_k=8)
C = C.float().clone()
prev, hist = None, []
for it in range(args.max_iter):
    lab = assign(Z, C)
    changed = 1.0 if prev is None else float((lab != prev).float().mean())
    cnt = torch.bincount(lab, minlength=K)
    hist.append({"iter": it, "changed": changed, "non_empty": int((cnt > 0).sum())})
    if prev is not None and changed < 1e-4:
        break
    prev = lab
    sums = torch.zeros_like(C)
    for i in range(0, len(Z), 262144):
        sums.index_add_(0, lab[i:i + 262144], Z[i:i + 262144].float())
    live = cnt > 0
    C[live] = sums[live] / cnt[live, None].float()
res = {"ckpt": args.ckpt, "train_rows": len(Z), "iterations": len(hist), "history": hist,
       "train_usage": usage(assign(Z, C))}
print(f"Lloyd: {len(hist)} iterations, last changed {hist[-1]['changed']:.4f}, train usage {res['train_usage']}", flush=True)
del Z
torch.cuda.empty_cache()
Zh = encode(np.arange(v0, N))
res["heldout_usage"] = usage(assign(Zh, C))
print("held-out usage:", res["heldout_usage"], flush=True)

ck = torch.load(resolve_input_path(args.ckpt), map_location="cpu", weights_only=False)
st = ck["model_state"]
assert st["centroids"].shape == C.shape, (st["centroids"].shape, C.shape)
st["centroids"] = C.cpu().to(st["centroids"].dtype)
st["centroids_initialized"] = torch.ones_like(st["centroids_initialized"], dtype=torch.bool) \
    if st["centroids_initialized"].dtype == torch.bool else torch.ones_like(st["centroids_initialized"])
ck.pop("opt_state", None)   # the optimizer is not needed for evaluation
ck["note"] = f"{args.ckpt} with a re-fitted plain k-means codebook (experiments/balancing/analysis/paper_lloyd_ep10_codebook.py)"
torch.save(ck, args.out_ckpt)
res["out_ckpt"] = args.out_ckpt
Path(args.out_json).write_text(json.dumps(res, indent=1))
print("wrote", args.out_ckpt, args.out_json, f"({time.time() - t0:.0f}s)")
