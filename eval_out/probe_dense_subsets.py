"""Chance-corrected NMI probes recomputed on the rows that land in well-populated clusters.

The probe's partition of a rung's rows has many tiny clusters (GeoAE puts the 2,650 DBpedia-14 texts in 227 clusters,
85 of them singletons). This rescoring asks whether the scores, and the ranking of the spaces, hold once those rows
are set aside. "Cluster size" is the number of THIS rung's rows a cluster receives (not its training usage).

Rows, assignment and scoring are exactly eval_out/probe_chance_corrected.py's (anchor rows excluded, token rungs
subsampled to 80k with RandomState(0), nearest-centroid labels via concept_probe.assign, arithmetic-mean NMI,
null = mean NMI over label shuffles, cNMI = (NMI - null) / (1 - null)); the "all" subset reproduces the probe files.
Subsets, each scored with its own null (labels shuffled within the kept rows):
  all            every row
  min5 / min10   drop rows whose cluster holds fewer than 5 / 10 of the rung's rows
  top90/75/50    keep the most populated clusters, whole, until they cover >= 90 / 75 / 50% of the rows
  common_min5/10 rows that survive min5 / min10 in EVERY arm (the same rows for all spaces)
  common_fpN     (--fp_min N) rows whose cluster holds >= N KEPT rows in every arm, iterated to a fixed point
  pair_fpN_a+b   (--pairs a:b,... --pair_min N) the same fixed point over the two arms of each pair only (both scored on it)
                 (eval_out/atlas_nmi_filtered.filter_rows; N = 6 is the reporting rule's "more than 5")
Also per subset: homogeneity MI/H(label) and completeness MI/H(cluster), raw and chance-corrected ((MI - MI0) / (H - MI0),
MI0 = shuffled MI; NMI is their harmonic mean), rows and clusters kept, and purity (share of rows whose cluster's majority label is their own) with
its shuffled value.

    PYTHONPATH=. .venv/bin/python eval_out/probe_dense_subsets.py --out results/delta/nosinkhorn/probe_dense_subsets.json
    (Gemma 3 4B L33: --cache cache/gemma3-4b/l33 --arms res_balanced=km:...,res_plain=km:...,tok_plain=km:...,geoae=ae:...)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoae.checkpoint import load_ae_checkpoint  # noqa: E402
from geoae.interp import concept_probe as CP  # noqa: E402
from geoae.interp.closest_tokens import load_baseline_kmeans  # noqa: E402
from geoae.seeded_init import anchor_row_indices  # noqa: E402
from geoae.token_bias import TokenBiasLookup  # noqa: E402

D = "e2e/checkpoints/general/llama3.2-3B/layer27/delta/"
K = "checkpoints/llama3.2-3B/layer27/"
ARMS = [
    ("res_balanced", "km", D + "balanced_kmeans_k2000_dpc_sampled.npz"),
    ("res_plain", "km", D + "plain_kmeans_k2000_dpc_sampled_seed42.npz"),
    ("tok_plain", "km", D + "plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz"),
    ("geoae", "ae", K + "k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt"),
    ("A_nosk_reinit", "ae", K + "k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nosinkhorn/step_0014450.pt"),
    ("B_nosk_lloyd", "ae", K + "k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nosinkhorn_lloyd_noreinit/step_0014450.pt"),
]
SUBSETS = ["all", "min5", "min10", "top90", "top75", "top50", "common_min5", "common_min10"]

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache_ids")
ap.add_argument("--n_token", type=int, default=80000)
ap.add_argument("--n_shuffle", type=int, default=10)
ap.add_argument("--arms", default="", help="name=km:<codebook.npz>, name=ae:<checkpoint.pt> or name=aekm:<checkpoint.pt>+"
                                           "<latent codebook.npz> (eval_out/latent_plain_kmeans.py), comma-separated; "
                                           "default: the Llama L27 arms above")
ap.add_argument("--fp_min", type=int, default=0, help="also score the fixed-point common subset common_fp<N>")
ap.add_argument("--pairs", default="", help="a:b,c:d -> also score each pair on its own fixed-point subset pair_fp<N>_a+b")
ap.add_argument("--pair_min", type=int, default=10)
ap.add_argument("--rungs", default="", help="comma list: score only these ladder rungs (default: every cached rung)")
ap.add_argument("--no_anchor_exclusion", action="store_true",
                help="keep every row (by default the 25-per-class seeded-init anchor rows are dropped)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
if args.arms:
    ARMS = [(n, *spec.split(":", 1)) for n, spec in (a.split("=", 1) for a in args.arms.split(","))]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # CPU fallback (12 Oct: probes while every GPU trains)

arms = {}
for name, kind, path in ARMS:
    if kind == "saekm":     # name=saekm:<sae spec>@<code-space codebook.npz> (eval_out/sae_plain_kmeans_fit.py)
        from geoae.interp.sae_space import load_sae_kmeans
        arms[name] = ("saekm", load_sae_kmeans(path, dev, name))
    elif kind == "aekm":
        ck, cb = path.split("+", 1)
        ae, mu, sd, _ = load_ae_checkpoint(ck, dev, allow_token_bias=True)
        ae.eval()
        C = torch.as_tensor(np.load(cb)["centroids"], dtype=torch.float32, device=dev)
        arms[name] = ("aekm", (ae, mu, sd, C))
    elif kind == "ae":
        ae, mu, sd, _ = load_ae_checkpoint(path, dev, allow_token_bias=True)
        ae.eval()
        arms[name] = ("ae", (ae, mu, sd))
    else:
        C, m, s, *_ = load_baseline_kmeans(Path(path), dev, allow_erasure=True)
        meta = np.load(path, allow_pickle=True)
        tbl = TokenBiasLookup(str(meta["token_bias"]), dev) if "token_bias" in meta.files else None
        arms[name] = ("km", (C, m, s, None, tbl))


def sae_assign(H, km, chunk=16384):
    """Nearest code-space centroid of each row's SAE codes (raw residual in; sae_space.CodeKMeans)."""
    return np.concatenate([km.assign_labels(torch.from_numpy(np.asarray(H[i:i + chunk])).to(dev).float()).cpu().numpy()
                           for i in range(0, len(H), chunk)])


def enc(v):
    return np.unique(v, return_inverse=True)[1]


def info(y, c):
    """(H(label), H(cluster), MI) in nats."""
    ny, nc = y.max() + 1, c.max() + 1
    J = np.bincount(y * nc + c, minlength=ny * nc).reshape(ny, nc).astype(float)
    N = J.sum()
    py, pc, P = J.sum(1) / N, J.sum(0) / N, J / N
    nz = P > 0
    mi = (P[nz] * np.log(P[nz] / (py[:, None] * pc[None, :])[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(h(py)), float(h(pc)), float(mi)


def nmi(y, c):
    """Arithmetic-mean NMI (eval_out/probe_chance_corrected.nmi_weighted with unit weights)."""
    hy, hc, mi = info(y, c)
    return mi / max((hy + hc) / 2, 1e-12)


def purity(y, c):
    J = np.zeros((c.max() + 1, y.max() + 1))
    np.add.at(J, (c, y), 1)
    return float(J.max(1).sum() / len(y))


def score(y, c, rng):
    y, c = enc(y), enc(c)
    if len(np.unique(y)) < 2:
        return None
    perms = [rng.permutation(y) for _ in range(args.n_shuffle)]
    nm = nmi(y, c)
    null = float(np.mean([nmi(p, c) for p in perms]))
    # NMI (arithmetic) = V-measure = harmonic mean of homogeneity MI/H(label) (each cluster holds one label) and
    # completeness MI/H(cluster) (each label sits in one cluster). Shuffling keeps both marginals, so the chance-
    # corrected forms share the shuffled MI: (MI - MI0) / (H - MI0); cNMI = (MI - MI0) / (mean H - MI0) as above.
    hy, hc, mi = info(y, c)
    mi0 = float(np.mean([info(p, c)[2] for p in perms]))
    hom, com = mi / hy, mi / max(hc, 1e-12)
    homc, comc = (mi - mi0) / (hy - mi0), (mi - mi0) / max(hc - mi0, 1e-12)
    return dict(n=int(len(y)), clusters=int(c.max() + 1), nmi=nm, null=null, cnmi=(nm - null) / (1 - null),
                H_label=hy, H_cluster=hc, mi=mi, mi_null=mi0, homogeneity=hom, completeness=com,
                homogeneity_c=homc, completeness_c=comc,
                purity=purity(y, c), purity_null=float(np.mean([purity(p, c) for p in perms[:3]])))


def fixed_point(labs, n_min):
    """Keep-mask: every kept row's cluster holds >= n_min kept rows in every arm (iterated to a fixed point)."""
    keep = np.ones(len(next(iter(labs.values()))), bool)
    while True:
        new = keep.copy()
        for c in labs.values():
            new &= np.bincount(c[keep], minlength=c.max() + 1)[c] >= n_min
        if (new == keep).all():
            return keep
        keep = new


def keep_mask(c, subset):
    size = np.bincount(c)[c]
    if subset == "all":
        return np.ones(len(c), bool)
    if subset.startswith("min"):
        return size >= int(subset[3:])
    frac = int(subset[3:]) / 100
    cnt = np.bincount(c)
    order = np.argsort(-cnt, kind="stable")
    cum = np.cumsum(cnt[order])
    n_keep = int(np.searchsorted(cum, frac * len(c)) + 1)
    return np.isin(c, order[:n_keep])


excl = None if args.no_anchor_exclusion else anchor_row_indices(args.cache, None, 25, 5, 42, "", 25)
only = set(args.rungs.split(",")) if args.rungs else None
if args.fp_min:
    SUBSETS = SUBSETS + [f"common_fp{args.fp_min}"]
out = {"meta": {"arms": {n: p for n, _, p in ARMS}, "subsets": SUBSETS, "n_shuffle": args.n_shuffle,
                "anchor_exclusion": not args.no_anchor_exclusion}, "rungs": {}}
for rung, fname, key, grain, _ in CP.LADDER:
    if not (Path(args.cache) / fname).exists() or (only and rung not in only):
        continue
    H, y = CP.load_rung(args.cache, fname, key, 0)
    ids = CP.load_rung_ids(args.cache, fname)
    if excl and rung in excl:
        keep = np.ones(len(y), bool)
        keep[excl[rung][excl[rung] < len(y)]] = False
        H, y = H[keep], y[keep]
        ids = None if ids is None else ids[keep]
    if grain == "token" and len(H) > args.n_token:
        sub = np.random.RandomState(0).choice(len(H), args.n_token, replace=False)
        H, y = H[sub], y[sub]
        ids = None if ids is None else ids[sub]
    y = enc(np.asarray(y).astype(str))
    labs = {n: enc(sae_assign(H, a[1]) if a[0] == "saekm" else CP.assign(H, a, dev, ids=ids)) for n, a in arms.items()}
    masks = {n: {s: keep_mask(c, s) for s in SUBSETS if not s.startswith("common")} for n, c in labs.items()}
    common = {f"common_{t}": np.logical_and.reduce([masks[n][t] for n in labs]) for t in ("min5", "min10")}
    if args.fp_min:
        common[f"common_fp{args.fp_min}"] = fixed_point(labs, args.fp_min)
    res = {"grain": grain, "n": int(len(y)), "common_rows": {k: int(v.sum()) for k, v in common.items()}, "arms": {}}
    for n, c in labs.items():
        rng = np.random.default_rng(0)
        res["arms"][n] = {s: score(y[m], c[m], rng) for s, m in list(masks[n].items()) + list(common.items())}
    for pair in filter(None, args.pairs.split(",")):
        a, b = pair.split(":")
        m = fixed_point({a: labs[a], b: labs[b]}, args.pair_min)
        name = f"pair_fp{args.pair_min}_{a}+{b}"
        res["common_rows"][name] = int(m.sum())
        for n in (a, b):
            res["arms"][n][name] = score(y[m], labs[n][m], np.random.default_rng(0))
    out["rungs"][rung] = res
    t = res["arms"]
    print(f"{rung:<16} n={len(y):>6} common_min10={common['common_min10'].sum():>6}  " + "  ".join(
        f"{n}: all {t[n]['all']['cnmi']:.3f} min5 {t[n]['min5']['cnmi']:.3f} top50 {t[n]['top50']['cnmi']:.3f}"
        for n in list(t)[:3]), flush=True)
Path(args.out).write_text(json.dumps(out, indent=1))
print("wrote", args.out)
