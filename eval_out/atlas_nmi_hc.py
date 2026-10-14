"""NMI, homogeneity and completeness of every codebook against single-label FineWeb-Atlas classes, at three readouts.

Label sets as eval_out/atlas_nmi.py builds them: the three 14-concept content draws of draws.json (atlas14_s0/1/2) and
the 31 document-type concepts (doctype); a chunk qualifies when it carries exactly ONE class of the set. Readouts:
  lastword  one row per chunk, the last token of its last word (atlas_siblings_extract.py --last_word cache)
  random    the ~8 random non-final positions per chunk of the original cache
  final     the chunk's final token (97% punctuation) of the original cache
Scores (sklearn): homogeneity h = 1 - H(class|cluster)/H(class) (each cluster holds one class), completeness
c = 1 - H(cluster|class)/H(cluster) (each class sits in one cluster), NMI = V-measure = 2hc/(h+c) (arithmetic
normalisation). Each is also chance-corrected, x_c = (x - null) / (1 - null), null = mean over 5 class permutations.

    scripts/delta/py eval_out/atlas_nmi_hc.py --orig_assign A.npz,B.npz --lastword_assign C.npz \
        --chunks <cache>/chunks.npz --concepts <atlas_concept_meta.json> --out nmi_hc.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure

DEFAULT_DRAWS = "results/delta/d6144/interventions/atlas_classes/draws.json"
ARMS = ("base", "base_tokmean", "plain", "plain_tokmean", "bypass")


def exclusive(label_ptr, label_ids, units):
    """(chunks, class) for chunks carrying exactly one of the classes (each class = a list of concept ids)."""
    lut = np.full(int(label_ids.max()) + 1, -1, dtype=np.int64)
    for j, u in enumerate(units):
        lut[np.asarray(u, dtype=np.int64)] = j
    n = len(label_ptr) - 1
    chunk_of = np.repeat(np.arange(n), np.diff(label_ptr))
    j = lut[label_ids]
    pairs = np.unique(np.stack([chunk_of[j >= 0], j[j >= 0]], 1), axis=0)      # distinct (chunk, class)
    cnt = np.bincount(pairs[:, 0], minlength=n)
    one = cnt[pairs[:, 0]] == 1
    return pairs[one, 0], pairs[one, 1]


def scores(y, c, rng):
    h, cm, v = homogeneity_completeness_v_measure(y, c)
    null = np.array([homogeneity_completeness_v_measure(rng.permutation(y), c) for _ in range(5)]).mean(0)
    raw = np.array([h, cm, v])
    cc = (raw - null) / (1 - null)
    return dict(homogeneity=float(h), completeness=float(cm), nmi=float(v),
                homogeneity_cc=float(cc[0]), completeness_cc=float(cc[1]), nmi_cc=float(cc[2]),
                clusters_used=int(len(np.unique(c))))


def small_cluster_variants(y, c, min_sizes, rng, ami=True):
    """Scores with tiny clusters neutralised. A cluster is small when it holds < m of the SCORED rows. merge: all rows of
    small clusters become one 'rest' cluster (same rows for every arm; the rest cluster is impure); drop: those rows are
    removed. ami: adjusted mutual information of the raw labels (analytic chance correction for the cluster sizes)."""
    sizes = np.bincount(np.unique(c, return_inverse=True)[1])
    inv = np.unique(c, return_inverse=True)[1]
    out = {}
    for m in min_sizes:
        small = sizes[inv] < m
        merged = np.where(small, -1, c)
        out[f"merge_lt{m}"] = dict(scores(y, merged, rng), rows_in_small=float(small.mean()))
        if (~small).sum() > 1:
            out[f"drop_lt{m}"] = dict(scores(y[~small], c[~small], rng), rows_kept=int((~small).sum()))
    if ami:
        out["ami"] = float(adjusted_mutual_info_score(y, c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_assign", required=True, help="comma list of assign npz over the original cache (merged)")
    ap.add_argument("--lastword_assign", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--draws", default=DEFAULT_DRAWS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_sizes", default="", help="e.g. 2,5,10,20: also score with clusters of < m scored rows merged "
                                                       "into one / dropped, plus AMI")
    ap.add_argument("--grains", default="lastword,random,final")
    args = ap.parse_args()
    min_sizes = [int(x) for x in args.min_sizes.split(",") if x]

    z = np.load(args.chunks)
    draws = json.load(open(args.draws))
    meta = json.load(open(args.concepts))
    sets = {f"atlas14_s{d['seed']}": [c["ids"] for c in d["classes"]] for d in draws["draws"]}
    sets["doctype"] = [[int(i)] for i, t in zip(meta["concept_id"], meta["concept_type"]) if t == "document"]

    orig = {}
    for p in args.orig_assign.split(","):
        a = np.load(p)
        for k in a.files:
            orig.setdefault(k, a[k])
    lw = np.load(args.lastword_assign)
    assert (lw["row_chunk"] == np.arange(len(lw["row_chunk"]))).all(), "last-word cache must hold one row per chunk"
    rc, rp, nt = orig["row_chunk"].astype(np.int64), orig["row_pos"].astype(np.int64), orig["chunk_ntok"].astype(np.int64)
    final = rp == nt[rc] - 1

    res = {"meta": vars(args), "sets": {}}
    for name, units in sets.items():
        chunks, y_chunk = exclusive(z["label_ptr"], z["label_ids"], units)
        cls = np.full(len(z["label_ptr"]) - 1, -1, dtype=np.int64)
        cls[chunks] = y_chunk
        res["sets"][name] = {"n_classes": len(units), "n_chunks": int(len(chunks))}
        for grain in args.grains.split(","):
            if grain == "lastword":
                rows, src = chunks, lw
            else:
                sel = (cls[rc] >= 0) & (final if grain == "final" else ~final)
                rows, src = np.flatnonzero(sel), orig
            y = cls[chunks] if grain == "lastword" else cls[rc[rows]]
            out = {"n_rows": int(len(rows))}
            for arm in ARMS:
                out[arm] = scores(y, src[arm][rows], np.random.default_rng(0))
                if min_sizes:
                    out[arm]["small"] = small_cluster_variants(y, src[arm][rows], min_sizes, np.random.default_rng(0))
            res["sets"][name][grain] = out
            print(f"{name:11s} {grain:8s} n={len(rows):7,}  " + "  ".join(
                f"{a}: h {out[a]['homogeneity']:.3f} c {out[a]['completeness']:.3f} nmi {out[a]['nmi']:.3f} "
                f"(cc {out[a]['nmi_cc']:.3f})" for a in ARMS), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
