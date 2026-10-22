"""Raw cross-document neighbour precision per Atlas concept (no chance correction): for every concept c, the share of the k
neighbour slots of its positive rows whose chunk also carries c, i.e. precision_c = hits_c / (k * m_c) of metric 2 in
geoae/analysis/atlas_concept_stats.py without the division by expected_c. Reads the graph of geoae/analysis/atlas_knn_graph.py
(neighbours from other documents only) and the concept list of a concept_stats run (concept_table.csv).

Summaries per space: mean and median of precision_c over concepts, the same as neighbours out of k, and the pooled share
over all (query, concept) pairs. Comparisons ref - arm: mean difference of precision_c over concepts, share of concepts
where ref is higher, and a document-bootstrap 95% CI of the mean difference (documents resampled with replacement; a
query is weighted by its document's count).

    scripts/py geoae/analysis/atlas_knn_precision.py --graph <dir>/knn_graph.npz --chunks <cache>/chunks.npz \
        --table <dir>/concept_table.csv --out <dir>/knn_precision.json
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--table", required=True, help="concept_table.csv of atlas_concept_stats.py (its concept ids)")
    ap.add_argument("--arms", default="base,base_tokmean,bypass")
    ap.add_argument("--ref", default="bypass")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]

    G = np.load(args.graph)
    z = np.load(args.chunks)
    concepts = np.array([int(r["concept_id"]) for r in csv.DictReader(open(args.table))], dtype=np.int64)
    g_chunk, g_doc = G["chunk"].astype(np.int64), G["doc"].astype(np.int64)
    R, k = len(g_chunk), G[f"nbr_{arms[0]}"].shape[1]
    ptr, ids = z["label_ptr"], z["label_ids"].astype(np.int64)
    n_chunks = len(ptr) - 1
    col = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    col[concepts] = np.arange(len(concepts))
    chunk_of = np.repeat(np.arange(n_chunks), np.diff(ptr))
    keep = col[ids] >= 0
    pc_chunk, pc_c = chunk_of[keep], col[ids[keep]]                     # (chunk, concept) label pairs, eligible concepts
    member = np.sort(pc_chunk * len(concepts) + pc_c)
    row_of_chunk = np.full(n_chunks, -1, dtype=np.int64)
    row_of_chunk[g_chunk] = np.arange(R)
    pr_row, pr_c = row_of_chunk[pc_chunk], pc_c                         # positive (graph row, concept) pairs
    ok = pr_row >= 0
    pr_row, pr_c = pr_row[ok], pr_c[ok]
    udoc, dinv = np.unique(g_doc, return_inverse=True)
    pr_d = dinv[pr_row]
    C, D = len(concepts), len(udoc)
    m_cd = np.bincount(pr_c * D + pr_d, minlength=C * D).reshape(C, D).astype(np.float64)  # dense: C*D small enough
    hits = {}
    for a in arms:
        nch = g_chunk[G[f"nbr_{a}"]]                                      # (R, k) neighbour chunks
        q = nch[pr_row] * C + pr_c[:, None]
        pos = np.minimum(np.searchsorted(member, q), len(member) - 1)
        h = (member[pos] == q).sum(1)
        hits[a] = np.bincount(pr_c * D + pr_d, weights=h, minlength=C * D).reshape(C, D)

    def prec(w):
        m = m_cd @ w
        return {a: (hits[a] @ w) / (k * m) for a in arms}, m

    p, m = prec(np.ones(D))
    res = {"meta": dict(vars(args), n_rows=R, k=k, n_concepts=C, n_documents=D), "arms": {}, "diff_ref_minus_arm": {}}
    for a in arms:
        res["arms"][a] = dict(mean=float(p[a].mean()), median=float(np.median(p[a])), mean_of_k=float(k * p[a].mean()),
                              pooled=float(hits[a].sum() / (k * m.sum())))
    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.n_boot):
        w = np.bincount(rng.integers(0, D, D), minlength=D).astype(np.float64)
        pb, _ = prec(w)
        boots.append({a: float(np.nanmean(pb[a])) for a in arms})
    for a in arms:
        res["arms"][a]["mean_ci"] = [float(x) for x in np.quantile([b[a] for b in boots], [0.025, 0.975])]
        if a == args.ref:
            continue
        d = p[args.ref] - p[a]
        db = [b[args.ref] - b[a] for b in boots]
        res["diff_ref_minus_arm"][a] = dict(mean=float(d.mean()), ci=[float(x) for x in np.quantile(db, [0.025, 0.975])],
                                            share_ref_higher=float((d > 0).mean()), share_tied=float((d == 0).mean()))
    for a in arms:
        r = res["arms"][a]
        dtxt = "" if a == args.ref else (f"  | {args.ref} - {a}: {res['diff_ref_minus_arm'][a]['mean']:+.4f} "
                                         f"[{res['diff_ref_minus_arm'][a]['ci'][0]:+.4f}, {res['diff_ref_minus_arm'][a]['ci'][1]:+.4f}], "
                                         f"{args.ref} higher on {res['diff_ref_minus_arm'][a]['share_ref_higher']:.0%}")
        print(f"{a:14s} mean {r['mean']:.4f} [{r['mean_ci'][0]:.4f}, {r['mean_ci'][1]:.4f}]  median {r['median']:.4f}  "
              f"= {r['mean_of_k']:.2f} of {k}  pooled {r['pooled']:.4f}{dtxt}")
    terc = np.quantile(m, [1 / 3, 2 / 3])
    size = np.digitize(m, terc)                                          # 0 / 1 / 2 = small / mid / large concepts
    res["by_size_tercile"] = {}
    for t, name in enumerate(("small", "mid", "large")):
        sel = size == t
        res["by_size_tercile"][name] = dict(n=int(sel.sum()), positives_range=[float(m[sel].min()), float(m[sel].max())],
                                            **{f"mean_{a}": float(p[a][sel].mean()) for a in arms},
                                            **{f"share_{args.ref}_higher_than_{a}": float((p[args.ref][sel] > p[a][sel]).mean())
                                               for a in arms if a != args.ref})
        print(f"  {name:5s} concepts ({sel.sum()}, {m[sel].min():.0f}-{m[sel].max():.0f} positive chunks): "
              + "  ".join(f"{a} {p[a][sel].mean():.4f}" for a in arms) + "  | "
              + "  ".join(f"{args.ref}>{a} on {(p[args.ref][sel] > p[a][sel]).mean():.0%}" for a in arms if a != args.ref))
    res["per_concept"] = {"concept_id": concepts.tolist(), "positives": m.tolist(), **{a: p[a].tolist() for a in arms}}
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
