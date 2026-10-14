"""Are the concept-probe rows in distribution for each space? Compares probe-cache rows with held-out dump rows.

Per probe rung and per space (same arm spec as eval_out/probe_dense_subsets.py: km / ae / aekm):
  dist_med_ratio, dist_p90_ratio  nearest-centroid distance of the rung's rows / the same quantile on held-out dump rows
  beyond_dump_p99                 share of the rung's rows farther from every centroid than 99% of dump rows
  eff_k, eff_k_dump               effective clusters (exp of label entropy) of the rung's rows, and of the same number of
                                  random held-out dump rows (the count is n-dependent, so it is matched)
Plus the standardized input norm (residual space, the first km arm's standardization) as a ratio to the dump's median.
Rows: token rungs subsampled to --n_token with RandomState(0) as in the probes; dump rows = --n_dump random rows of the
held-out 5% (RandomState(1)).

    python -u eval_out/probe_ood_check.py --activations_dir <dump> --layer 17 --cache <cache_ids/...> \
        --arms "res_plain=km:...,geoae=ae:...,geoae_lplain=aekm:<ckpt>+<codebook>" --out ood.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoae.checkpoint import load_ae_checkpoint  # noqa: E402
from geoae.interp import concept_probe as CP  # noqa: E402
from geoae.interp.closest_tokens import load_baseline_kmeans  # noqa: E402
from geoae.token_bias import TokenBiasLookup  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--activations_dir", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--arms", required=True)
ap.add_argument("--n_dump", type=int, default=100_000)
ap.add_argument("--n_token", type=int, default=80_000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.set_grad_enabled(False)
dev = torch.device("cuda")

arms = {}
for name, spec in (a.split("=", 1) for a in args.arms.split(",")):
    kind, path = spec.split(":", 1)
    if kind in ("ae", "aekm"):
        ck = path.split("+", 1)[0]
        ae, mu, sd, _ = load_ae_checkpoint(ck, dev, allow_token_bias=True)
        ae.eval()
        C = ae.centroids.float() if kind == "ae" else torch.as_tensor(np.load(path.split("+", 1)[1])["centroids"],
                                                                        dtype=torch.float32, device=dev)
        arms[name] = ("ae", ae, mu, sd, C, None)
    else:
        C, m, s, *_ = load_baseline_kmeans(Path(path), dev, allow_erasure=True)
        meta = np.load(path, allow_pickle=True)
        tb = TokenBiasLookup(str(meta["token_bias"]), dev) if "token_bias" in meta.files else None
        arms[name] = ("km", None, m, s, C.float(), tb)
norm_arm = next(n for n, a in arms.items() if a[0] == "km" and a[5] is None)


def embed(arm, H: np.ndarray, ids):
    kind, ae, m, s, C, tb = arm
    out_d, out_l = [], []
    for i in range(0, len(H), 16384):
        h = torch.from_numpy(np.asarray(H[i:i + 16384], dtype=np.float32)).to(dev)
        t = None if ids is None else torch.from_numpy(np.asarray(ids[i:i + 16384], dtype=np.int64)).to(dev)
        if kind == "ae":
            if ae.has_token_bias and t is None:
                return None, None
            v = ae.encode((h - m) / s, t if ae.has_token_bias else None)
        else:
            v = (h - m) / s
            if tb is not None:
                if t is None:
                    return None, None
                v = v - tb(t)
        d = torch.cdist(v.float(), C)
        mn, lab = d.min(1)
        out_d.append(mn.cpu().numpy())
        out_l.append(lab.cpu().numpy())
    return np.concatenate(out_d), np.concatenate(out_l)


def input_norm(H: np.ndarray) -> np.ndarray:
    _, _, m, s, _, _ = arms[norm_arm]
    out = []
    for i in range(0, len(H), 65536):
        h = torch.from_numpy(np.asarray(H[i:i + 65536], dtype=np.float32)).to(dev)
        out.append(((h - m) / s).norm(dim=1).cpu().numpy())
    return np.concatenate(out)


def eff_k(lab: np.ndarray) -> float:
    p = np.bincount(lab) / len(lab)
    p = p[p > 0]
    return float(math.exp(-(p * np.log(p)).sum()))


D = Path(args.activations_dir)
A = np.load(D / f"layer_{args.layer}.npy", mmap_mode="r")
TOK = np.load(D / "rows_tok.npy", mmap_mode="r")
N = A.shape[0]
v0 = int(N * 0.95)
ridx = np.sort(np.random.RandomState(1).choice(np.arange(v0, N), args.n_dump, replace=False))
Hd, Td = np.asarray(A[ridx], dtype=np.float32), np.asarray(TOK[ridx], dtype=np.int64)
dump = {n: embed(a, Hd, Td) for n, a in arms.items()}
nd = input_norm(Hd)
res = {"meta": vars(args), "dump": {"input_norm_median": float(np.median(nd)),
                                    "arms": {n: {"dist_median": float(np.median(d)), "dist_p90": float(np.quantile(d, .9)),
                                                 "dist_p99": float(np.quantile(d, .99)), "eff_k": eff_k(l)}
                                             for n, (d, l) in dump.items()}}, "rungs": {}}
print(f"dump: input norm median {np.median(nd):.1f}; " + "; ".join(
    f"{n} dist med {np.median(d):.2f} effK {eff_k(l):.0f}" for n, (d, l) in dump.items()), flush=True)
rng_m = np.random.RandomState(2)
for rung, fname, key, grain, _ in CP.LADDER:
    if not (Path(args.cache) / fname).exists() or rung in ("pos_fine", "ner_fine", "ravel_continent", "ravel_language"):
        continue                                  # same rows as pos_coarse / ner_coarse / ravel_country
    H, y = CP.load_rung(args.cache, fname, key, 0)
    ids = CP.load_rung_ids(args.cache, fname)
    if grain == "token" and len(H) > args.n_token:
        sub = np.random.RandomState(0).choice(len(H), args.n_token, replace=False)
        H, ids = H[sub], (None if ids is None else ids[sub])
    n = len(H)
    msub = rng_m.choice(args.n_dump, min(n, args.n_dump), replace=False)
    nr = input_norm(H)
    r = {"n": int(n), "grain": grain, "input_norm_ratio_median": float(np.median(nr) / np.median(nd)),
         "input_norm_ratio_p90": float(np.quantile(nr, .9) / np.quantile(nd, .9)), "arms": {}}
    for name, arm in arms.items():
        d, lab = embed(arm, H, ids)
        if d is None:
            continue
        dd, dl = dump[name]
        r["arms"][name] = {"dist_med_ratio": float(np.median(d) / np.median(dd)),
                           "dist_p90_ratio": float(np.quantile(d, .9) / np.quantile(dd, .9)),
                           "beyond_dump_p99": float((d > np.quantile(dd, .99)).mean()),
                           "eff_k": eff_k(lab), "eff_k_dump": eff_k(dl[msub])}
    res["rungs"][rung] = r
    print(f"{rung:<15} n={n:>6} norm x{r['input_norm_ratio_median']:.2f}  " + "  ".join(
        f"{a}: dist x{v['dist_med_ratio']:.2f} >p99 {100 * v['beyond_dump_p99']:.0f}% effK {v['eff_k']:.0f}/{v['eff_k_dump']:.0f}"
        for a, v in r["arms"].items()), flush=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print("wrote", args.out)
