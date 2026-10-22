"""Atlas NMI: chance-corrected NMI between every representation space's discrete assignment and FineWeb-Atlas labels.

The Atlas analogue of the topic14 probe, on an Atlas activation cache written by geoae/extraction/atlas_siblings_extract.py
(Gemma-3-4B layer 33 with --dtype float32 --include_last). Arms are the ones of geoae/evaluation/chance_corrected_probe.py:
--models (token-bypass AEs), --baselines (k-means npz, incl. base_tokmean), --saes (matched / Gemma Scope SAEs;
cluster = top-1 active feature, -1 = its own label, matched SAEs get the row's token id).

Label sets (each scored separately; a chunk qualifies only when it carries exactly ONE label of the set):
  atlas14_s{0,1,2}  the three 14-concept content draws of draws.json (the SAME concept ids as the Llama analysis)
  doctype           the 31 `document`-type concepts
(A top-20 tone set is deliberately not built: tone labels are dense, so single-label chunks are a biased sliver.)
Grains: `token` = every kept token row of a qualifying chunk (subsampled to <= --n_token rows, fixed seed), `last` =
only each chunk's final token row (needs the cache to have been extracted with --include_last).

Per arm: nmi, null (mean NMI over 5 label permutations), cnmi = (nmi - null) / (1 - null), live clusters, 95% CI
est + quantile(boots - mean(boots), [.025, .975]) over --n_boot multinomial row weights (NOT raw percentiles of the
bootstraps: those are biased upward for many-cluster arms), the unassigned fraction for SAEs, and `f1`, the mean over
classes of the best single-cluster F1. `leak` = cNMI(label; current token id) per grain.

    python -u geoae/evaluation/atlas_nmi.py --cache /scratch/dogar/geoae/atlas_g4b_l33 --models bypass=<ckpt> \
        --baselines base=<npz>,base_tokmean=<npz> --saes "bt10k=matched:<last.pt>,..." --out atlas_nmi.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geoae.analysis.atlas_siblings as AS  # noqa: E402

DEFAULT_DRAWS = "results/delta/d6144/interventions/atlas_classes/draws.json"
GRAINS = ("token", "last")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ pure helpers
def exclusive_labels(M):
    """chunk x unit 0/1 membership (scipy sparse) -> (chunks, labels): the chunks carrying exactly one unit, and which."""
    R = M.tocsr()
    nnz = np.diff(R.indptr)
    chunks = np.flatnonzero(nnz == 1)
    return chunks, R.indices[R.indptr[chunks]].astype(np.int64)


def select_rows(row_start, chunks, labels, grain):
    """(row indices, labels) of the qualifying chunks. token: all rows of each chunk (chunk order); last: the final
    row of each chunk that has one."""
    chunks = np.asarray(chunks, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if grain == "token":
        lens = row_start[chunks + 1] - row_start[chunks]
        return AS.rows_of_chunks(row_start, chunks), np.repeat(labels, lens)
    if grain == "last":
        has = row_start[chunks + 1] > row_start[chunks]
        c = chunks[has]
        return row_start[c + 1] - 1, labels[has]
    raise ValueError(grain)


def subsample(n, cap, seed=0):
    """Sorted fixed-seed subset of range(n) of size <= cap (all rows when n <= cap)."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, cap, replace=False))


def ci_centered(est, boots, q=(0.025, 0.975)):
    """est + quantile(boots - mean(boots), q): a bootstrap-centred interval (raw percentiles are biased upward for
    many-cluster arms because a resample's NMI is inflated by duplicated rows)."""
    b = np.asarray(boots, dtype=float)
    lo, hi = np.quantile(b - b.mean(), q)
    return float(est + lo), float(est + hi)


def best_cluster_f1(y, c):
    """Mean over classes of max over clusters of F1 (predictor: 'row is in that cluster'); y, c dense int ids.
    Returns (mean, per-class array)."""
    ny, nc = int(y.max()) + 1, int(c.max()) + 1
    J = np.bincount(y * nc + c, minlength=ny * nc).reshape(ny, nc).astype(float)
    f1 = 2 * J / np.maximum(J.sum(1)[:, None] + J.sum(0)[None, :], 1)
    per = f1.max(1)
    present = J.sum(1) > 0
    return float(per[present].mean()), per


def _enc(v):
    return np.unique(v, return_inverse=True)[1]


def score_labeling(yi, lab, nmi_weighted, perms, W, ones):
    """One arm's scores for dense class ids yi and arbitrary integer cluster labels lab."""
    c = _enc(lab)
    nm = nmi_weighted(yi, c, ones)
    null = float(np.mean([nmi_weighted(p, c, ones) for p in perms]))
    boots = [(nmi_weighted(yi, c, W[b]) - null) / (1 - null) for b in range(len(W))]
    cn = (nm - null) / (1 - null)
    lo, hi = ci_centered(cn, boots)
    f1, _ = best_cluster_f1(yi, c)
    return dict(nmi=nm, null=null, cnmi=cn, live=int(c.max() + 1), ci_lo=lo, ci_hi=hi, f1=f1, boots=boots)


# ------------------------------------------------------------------------------------------------ label sets
def build_label_sets(cache, draws, conc):
    """name -> dict(names, chunks, labels (dense class idx), n_per_class_chunks) for every label set."""
    sets = {}
    for d in draws["draws"]:
        units = [c["ids"] for c in d["classes"]]
        M = cache.membership(units)
        ch, lab = exclusive_labels(M)
        sets[f"atlas14_s{d['seed']}"] = dict(names=[c["name"] for c in d["classes"]], unit_ids=units, chunks=ch, labels=lab)
    doc = conc[conc.concept_type == "document"]
    units = [[int(i)] for i in doc.concept_id]
    M = cache.membership(units)
    ch, lab = exclusive_labels(M)
    sets["doctype"] = dict(names=doc.name.tolist(), unit_ids=units, chunks=ch, labels=lab)
    for v in sets.values():
        v["chunks_per_class"] = np.bincount(v["labels"], minlength=len(v["names"])).tolist()
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="Atlas activation cache dir (atlas_siblings_extract.py)")
    ap.add_argument("--rows_dir", default=None, help="directory holding rows*.npy copies (e.g. /dev/shm) ")
    ap.add_argument("--draws", default=DEFAULT_DRAWS)
    ap.add_argument("--atlas_dir", default="/scratch/dogar/geoae/atlas")
    ap.add_argument("--models", default="")
    ap.add_argument("--baselines", default="")
    ap.add_argument("--saes", default="")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--n_token", type=int, default=80000)
    ap.add_argument("--min_class_chunks", type=int, default=100, help="classes below this are reported (kept anyway)")
    ap.add_argument("--sets", default="", help="comma list of label sets; empty = all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd
    import torch

    import geoae.evaluation.chance_corrected_probe as PCC
    import geoae.analysis.concept_probe as CP

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = AS.AtlasCache(args.cache, args.rows_dir)
    cmeta = json.load(open(Path(args.cache) / "meta.json"))
    log(f"cache {args.cache}: {cache.n_chunks:,} chunks, {cache.n_rows:,} rows, d={cache.rows.shape[1]}, dtype={cache.rows.dtype}"
        f", include_last={cmeta.get('args', {}).get('include_last')}")
    if cache.rows.dtype != np.float32:
        log("WARNING: cache rows are not float32")
    row_pos = np.load(Path(args.rows_dir or args.cache) / "rows_pos.npy", mmap_mode="r")
    draws = json.load(open(args.draws))
    conc = pd.read_parquet(Path(args.atlas_dir) / "fineweb-concept-atlas.parquet")
    sets = build_label_sets(cache, draws, conc)
    only = set(filter(None, args.sets.split(",")))

    arms, saes = PCC.load_arms(args.models, args.baselines, args.saes, dev)
    log(f"arms: {list(arms)}  saes: {list(saes)}")
    rng = np.random.default_rng(0)
    out = {"meta": dict(cache=str(args.cache), draws=str(args.draws), cache_meta=cmeta, n_boot=args.n_boot,
                        n_token=args.n_token, ci="est + quantile(boots - mean(boots), [.025,.975])",
                        draws_sha256=draws.get("sha256")), "label_sets": {}}
    for name, ls in sets.items():
        if only and name not in only:
            continue
        low = [(n, k) for n, k in zip(ls["names"], ls["chunks_per_class"]) if k < args.min_class_chunks]
        rec = dict(names=ls["names"], unit_ids=ls["unit_ids"], n_chunks=int(len(ls["chunks"])),
                   chunks_per_class=ls["chunks_per_class"], classes_below_min=low, grains={})
        log(f"{name}: {rec['n_chunks']:,} qualifying chunks; per class {ls['chunks_per_class']}"
            + (f"  BELOW {args.min_class_chunks}: {low}" if low else ""))
        for grain in GRAINS:
            idx, y = select_rows(cache.row_start, ls["chunks"], ls["labels"], grain)
            if grain == "last":
                ok = row_pos[idx] == (cache.n_tok[cache.row_chunk[idx]] - 1)
                if not ok.any():
                    raise SystemExit("no chunk's last row stores its final token: extract with --include_last")
                if not ok.all():
                    log(f"  last grain: dropping {int((~ok).sum())} chunks without a stored final-token row")
                idx, y = idx[ok], y[ok]
            if grain == "token" and len(idx) > args.n_token:
                s = subsample(len(idx), args.n_token)
                idx, y = idx[s], y[s]
            H = np.asarray(cache.rows[idx])
            if not np.isfinite(H).all():
                raise SystemExit("non-finite rows in the cache")
            ids = np.asarray(cache.tok[idx], dtype=np.int64)
            yi = _enc(y)
            ones = np.ones(len(yi))
            W = rng.multinomial(len(yi), ones / len(yi), size=args.n_boot).astype(float)
            perms = [rng.permutation(yi) for _ in range(5)]
            g = dict(n_rows=int(len(yi)), n_chunks=int(len(np.unique(cache.row_chunk[idx]))),
                     rows_per_class=np.bincount(y, minlength=len(ls["names"])).tolist(), arms={})
            c = _enc(ids)
            nm = PCC.nmi_weighted(yi, c, ones)
            null = float(np.mean([PCC.nmi_weighted(p, c, ones) for p in perms]))
            g["leak"] = (nm - null) / (1 - null)
            for n, mdl in arms.items():
                g["arms"][n] = score_labeling(yi, CP.assign(H, mdl, dev, ids=ids), PCC.nmi_weighted, perms, W, ones)
            for n, sae in saes.items():
                lab = PCC.sae_assign(H, sae, dev, ids)
                g["arms"][n] = dict(score_labeling(yi, lab, PCC.nmi_weighted, perms, W, ones), unassigned=float((lab < 0).mean()))
            rec["grains"][grain] = g
            log(f"  {name:<12} {grain:<5} n={g['n_rows']:>6} leak={g['leak']:.3f}  "
                + "  ".join(f"{n}={a['cnmi']:.3f}" for n, a in g["arms"].items()))
        out["label_sets"][name] = rec
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out))      # incremental: a killed job keeps finished sets
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
