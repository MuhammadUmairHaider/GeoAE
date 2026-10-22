"""What do a space's clusters encode on IN-DISTRIBUTION rows? Raw homogeneity / completeness / NMI of nearest-centroid
labels against the dump's own row metadata, on the held-out 5% of a dump (contiguous rows, never trained on).

Targets: current token (rows_tok), position bucket (log2 buckets of rows_pos: [4,8), [8,16), ..., [1024,2048)),
source domain (rows_src: code / edu_web / math / pile / structured / web / wiki / wiki_de / es / fr) and document
(rows_doc). Rows are kept only if their cluster holds >= --min_rows kept rows in every arm (fixed point; reporting rule
"more than 5" = 6). Also: effective clusters (exp of label entropy) of early rows (position < 32) vs the same number of
random held-out rows, per arm.

Arms use geoae/evaluation/alignment.py's spec: name=km:<codebook.npz> | name=ae:<ckpt.pt> | name=aekm:<ckpt.pt>+<latent codebook.npz>.

    python -u experiments/paper/analysis/dump_structure.py --activations_dir <dump> --layer 17 --arms "..." --out dump_structure.json
"""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from geoae.checkpoint import load_ae_checkpoint  # noqa: E402
from geoae.analysis.closest_tokens import load_baseline_kmeans  # noqa: E402
from geoae.token_bias import TokenBiasLookup  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--activations_dir", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--arms", required=True)
ap.add_argument("--min_rows", type=int, default=6)
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_grad_enabled(False)
dev = torch.device("cuda")

arms = {}
for name, spec in (a.split("=", 1) for a in args.arms.split(",")):
    kind, path = spec.split(":", 1)
    if kind in ("ae", "aekm"):
        ae, mu, sd, _ = load_ae_checkpoint(path.split("+", 1)[0], dev, allow_token_bias=True)
        ae.eval()
        C = ae.centroids.float() if kind == "ae" else torch.as_tensor(np.load(resolve_input_path(path.split("+", 1)[1]))["centroids"],
                                                                        dtype=torch.float32, device=dev)
        arms[name] = ("ae", ae, mu, sd, C, None)
    else:
        C, m, s, *_ = load_baseline_kmeans(Path(path), dev, allow_erasure=True)
        meta = np.load(resolve_input_path(path), allow_pickle=True)
        tb = TokenBiasLookup(str(meta["token_bias"]), dev) if "token_bias" in meta.files else None
        arms[name] = ("km", None, m, s, C.float(), tb)

D = Path(args.activations_dir)
A = np.load(resolve_input_path(D / f"layer_{args.layer}.npy"), mmap_mode="r")
N = A.shape[0]
v0 = int(N * 0.95)
tok = np.asarray(np.load(resolve_input_path(D / "rows_tok.npy"), mmap_mode="r")[v0:], dtype=np.int64)
pos = np.asarray(np.load(resolve_input_path(D / "rows_pos.npy"), mmap_mode="r")[v0:], dtype=np.int64)
src = np.asarray(np.load(resolve_input_path(D / "rows_src.npy"), mmap_mode="r", allow_pickle=True)[v0:])
doc = np.asarray(np.load(resolve_input_path(D / "rows_doc.npy"), mmap_mode="r")[v0:], dtype=np.int64)
n = N - v0
print(f"held-out rows [{v0:,}, {N:,}) = {n:,}; positions {pos.min()}..{pos.max()}", flush=True)

labels = {k: np.empty(n, np.int64) for k in arms}
for s0 in range(0, n, 32768):
    h = torch.from_numpy(np.asarray(A[v0 + s0:v0 + s0 + 32768], dtype=np.float32)).to(dev)
    t = torch.from_numpy(tok[s0:s0 + 32768]).to(dev)
    for name, (kind, ae, m, s, C, tb) in arms.items():
        if kind == "ae":
            v = ae.encode((h - m) / s, t if ae.has_token_bias else None)
        else:
            v = (h - m) / s
            if tb is not None:
                v = v - tb(t)
        labels[name][s0:s0 + len(h)] = torch.cdist(v.float(), C).argmin(1).cpu().numpy()


def enc(v):
    return np.unique(v, return_inverse=True)[1]


def hcn(y, c):
    y, c = enc(y), enc(c)
    ny, nc = y.max() + 1, c.max() + 1
    J = np.bincount(y.astype(np.int64) * nc + c, minlength=ny * nc).reshape(ny, nc).astype(float)
    P = J / J.sum()
    py, pc = P.sum(1), P.sum(0)
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / (py[:, None] * pc[None, :])[nz])).sum())
    H = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    hy, hc = H(py), H(pc)
    return {"homogeneity": mi / hy, "completeness": mi / hc, "nmi": mi / ((hy + hc) / 2), "clusters": int(nc)}


def eff_k(lab):
    p = np.bincount(enc(lab)) / len(lab)
    return float(math.exp(-(p * np.log(p)).sum()))


keep = np.ones(n, bool)
while True:
    new = keep.copy()
    for lab in labels.values():
        c = enc(lab)
        new &= np.bincount(c[keep], minlength=c.max() + 1)[c] >= args.min_rows
    if (new == keep).all():
        break
    keep = new
bucket = np.floor(np.log2(np.maximum(pos, 1))).astype(np.int64)
targets = {"current_token": tok, "position_bucket": bucket, "source": src, "document": doc}
early = pos < 32
rnd = np.random.RandomState(0).choice(n, int(early.sum()), replace=False)
res = {"meta": dict(vars(args), heldout_rows=int(n), kept_rows=int(keep.sum()), early_rows=int(early.sum()),
                    position_buckets="floor(log2(position))"), "arms": {}}
for name, lab in labels.items():
    r = {t: hcn(y[keep], lab[keep]) for t, y in targets.items()}
    r["eff_k_early_pos_lt32"] = eff_k(lab[early])
    r["eff_k_random_matched"] = eff_k(lab[rnd])
    res["arms"][name] = r
    print(f"{name:<16} " + "  ".join(f"{t}: {v['homogeneity']:.3f}/{v['completeness']:.3f}/{v['nmi']:.3f}"
                                      for t, v in r.items() if isinstance(v, dict))
          + f"  effK early {r['eff_k_early_pos_lt32']:.0f} vs random {r['eff_k_random_matched']:.0f}", flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print("wrote", args.out, f"(kept {keep.sum():,} of {n:,} rows)")
