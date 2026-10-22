"""Homogeneity, completeness and raw NMI of every codebook against single-label FineWeb-Atlas classes, read at the last
word of each chunk, on the rows whose cluster holds MORE THAN --min_rows - 1 scored rows in EVERY codebook.

Rows: one per chunk, the last token of the chunk's last word (atlas_siblings_extract.py --last_word cache). Label sets as
geoae/evaluation/atlas_nmi.py: the three 14-concept content draws of draws.json and the 31 document-type concepts; a chunk
qualifies when it carries exactly one class of the set. Filter (user, 7 Oct): a cluster with few scored rows is pure by
construction and inflates homogeneity and NMI, so a row is kept only when its cluster holds >= --min_rows of the kept
rows in every codebook (intersection over codebooks), applied repeatedly until no row is dropped (dropping rows can push
another cluster below the threshold). All codebooks are then scored on the same rows.
Scores (sklearn): homogeneity h = 1 - H(class|cluster)/H(class), completeness c = 1 - H(cluster|class)/H(cluster),
NMI = V-measure = 2hc/(h+c); purity = share of rows in their cluster's majority class. No chance correction. Uncertainty: document bootstrap (documents resampled with
replacement, their kept rows repeated), 95% percentile CI of each score and of each codebook's difference to --ref.

    scripts/py geoae/evaluation/atlas_alignment.py --assign <atlas_lastword>/assign.npz --chunks <cache>/chunks.npz \
        --concepts <atlas_concept_meta.json> --out <atlas_lastword>/nmi_filtered.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix

DEFAULT_DRAWS = "results/delta/d6144/interventions/atlas_classes/draws.json"
METRICS = ("purity", "homogeneity", "completeness", "nmi")


def exclusive(label_ptr, label_ids, units):
    """(chunks, class) for chunks carrying exactly one of the classes (each class = a list of concept ids)."""
    lut = np.full(int(label_ids.max()) + 1, -1, dtype=np.int64)
    for j, u in enumerate(units):
        lut[np.asarray(u, dtype=np.int64)] = j
    n = len(label_ptr) - 1
    chunk_of = np.repeat(np.arange(n), np.diff(label_ptr))
    j = lut[label_ids]
    pairs = np.unique(np.stack([chunk_of[j >= 0], j[j >= 0]], 1), axis=0)      # distinct (chunk, class)
    one = np.bincount(pairs[:, 0], minlength=n)[pairs[:, 0]] == 1
    return pairs[one, 0], pairs[one, 1]


def filter_rows(labels, min_rows):
    """Boolean keep-mask over rows: every kept row's cluster holds >= min_rows kept rows in every codebook (fixed point).
    labels: dict arm -> (n,) int cluster ids. -> (keep, rounds)"""
    n = len(next(iter(labels.values())))
    keep = np.ones(n, dtype=bool)
    rounds = 0
    while True:
        rounds += 1
        new = keep.copy()
        for lab in labels.values():
            inv = np.unique(lab, return_inverse=True)[1]
            size = np.bincount(inv[keep], minlength=inv.max() + 1)
            new &= size[inv] >= min_rows
        if (new == keep).all():
            return keep, rounds
        keep = new


def load_assign(paths):
    """Merge assign files (atlas_assign_extra.py) row by row; their spines must match, an arm in two files must agree."""
    out = {}
    for p in filter(None, paths.split(",")):
        f = np.load(p)
        for k in f.files:
            v = f[k]
            if k in out:
                assert np.array_equal(out[k], v), f"{p}: '{k}' differs from an earlier --assign file"
            out[k] = v
    return out


def score(y, c):
    out = dict(zip(METRICS[1:], map(float, homogeneity_completeness_v_measure(y, c))))
    out["purity"] = float(contingency_matrix(y, c, sparse=True).max(axis=0).sum() / len(y))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assign", required=True, help="assign.npz of the last-word cache (one row per chunk); a comma list "
                    "merges files with identical row spines (e.g. assign.npz + an SAE assign file)")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--draws", default=DEFAULT_DRAWS)
    ap.add_argument("--arms", default="base,plain,base_tokmean,plain_tokmean,bypass")
    ap.add_argument("--ref", default="bypass", help="differences are ref - arm")
    ap.add_argument("--min_rows", type=int, default=6, help="kept clusters hold >= this many scored rows (6 = more than 5)")
    ap.add_argument("--per_space", action="store_true",
                    help="the original probe filter (min10 of probe_dense_subsets.py): each codebook keeps ITS OWN rows whose "
                         "cluster holds >= --min_rows of the set's scored rows (no intersection, no iteration); the bootstrap "
                         "resamples the same documents for every codebook")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]

    z = np.load(args.chunks)
    A = load_assign(args.assign)
    assert (A["row_chunk"] == np.arange(len(A["row_chunk"]))).all(), "expects one row per chunk (a --last_word cache)"
    assert len(A["row_chunk"]) == len(z["label_ptr"]) - 1, "assign and chunks describe different chunk counts"
    doc_all = z["doc_int_id"].astype(np.int64)
    draws = json.load(open(args.draws))
    meta = json.load(open(args.concepts))
    sets = {f"atlas14_s{d['seed']}": [c["ids"] for c in d["classes"]] for d in draws["draws"]}
    sets["doctype"] = [[int(i)] for i, t in zip(meta["concept_id"], meta["concept_type"]) if t == "document"]

    filt = ("per codebook: row kept iff its cluster holds >= min_rows of the set's scored rows in THAT codebook (no "
            "intersection); no chance correction" if args.per_space else
            "row kept iff its cluster holds >= min_rows kept rows in every arm (fixed point); no chance correction")
    res = {"meta": dict(vars(args), filter=filt), "sets": {}}
    for name, units in sets.items():
        chunks, y_all = exclusive(z["label_ptr"], z["label_ids"], units)
        labels = {a: A[a][chunks].astype(np.int64) for a in arms}
        if args.per_space:
            own = {}
            for a, v in labels.items():
                inv = np.unique(v, return_inverse=True)[1]
                own[a] = np.bincount(inv)[inv] >= args.min_rows
            keep, rounds = np.logical_or.reduce(list(own.values())), 0     # union: the rows any codebook scores
        else:
            keep, rounds = filter_rows(labels, args.min_rows)
            own = {a: keep for a in arms}
        y, doc = y_all[keep], doc_all[chunks][keep]
        lab = {a: v[keep] for a, v in labels.items()}
        mine = {a: own[a][keep] for a in arms}                               # each codebook's scored rows
        point = {a: score(y[mine[a]], lab[a][mine[a]]) for a in arms}
        for a in arms:
            point[a]["clusters"] = int(len(np.unique(lab[a][mine[a]])))
            point[a]["kept_share"] = float(mine[a].sum() / len(chunks))
        udoc, dinv = np.unique(doc, return_inverse=True)
        rows_of = np.split(np.argsort(dinv, kind="stable"), np.cumsum(np.bincount(dinv))[:-1])
        rng = np.random.default_rng(args.seed)
        boot = {a: {m: [] for m in METRICS} for a in arms}
        for _ in range(args.n_boot):
            pick = rng.integers(0, len(udoc), len(udoc))
            idx = np.concatenate([rows_of[d] for d in pick])
            for a in arms:
                ia = idx[mine[a][idx]]
                s = score(y[ia], lab[a][ia])
                for m in METRICS:
                    boot[a][m].append(s[m])
        ci = {a: {m: [float(x) for x in np.quantile(boot[a][m], [0.025, 0.975])] for m in METRICS} for a in arms}
        diff = {}
        for a in arms:
            if a == args.ref:
                continue
            diff[a] = {}
            for m in METRICS:
                d = np.asarray(boot[args.ref][m]) - np.asarray(boot[a][m])
                diff[a][m] = dict(point=point[args.ref][m] - point[a][m],
                                  ci=[float(x) for x in np.quantile(d, [0.025, 0.975])], share_ref_higher=float((d > 0).mean()))
        res["sets"][name] = dict(n_classes=len(units), n_chunks_single_label=int(len(chunks)), n_kept=int(keep.sum()),
                                 kept_share=float(keep.mean()), filter_rounds=rounds,
                                 n_classes_kept=int(len(np.unique(y))), n_documents_kept=int(len(udoc)),
                                 point=point, ci=ci, diff_ref_minus_arm=diff)
        print(f"== {name}: kept {keep.sum():,}/{len(chunks):,} rows ({keep.mean():.0%}, {rounds} rounds, "
              f"{len(np.unique(y))} classes, {len(udoc):,} documents)")
        neg = {a: int((v < 0).sum()) for a, v in lab.items() if (v < 0).any()}
        if neg:
            print(f"   kept rows labelled -1 (no active feature; scored as one cluster): {neg}")
        for a in arms:
            p = point[a]
            dtxt = "" if a == args.ref else "  | ref-arm " + "  ".join(
                f"{m[:4]} {diff[a][m]['point']:+.3f} [{diff[a][m]['ci'][0]:+.3f}, {diff[a][m]['ci'][1]:+.3f}]" for m in METRICS)
            print(f"   {a:14s} pur {p['purity']:.3f}  h {p['homogeneity']:.3f}  c {p['completeness']:.3f}  nmi {p['nmi']:.3f}  "
                  f"({p['clusters']} clusters){dtxt}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
