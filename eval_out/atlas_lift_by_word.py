"""Cross-document cluster lift (metric 1 of eval_out/atlas_concept_stats.py) split by whether the two rows are the SAME
current token or DIFFERENT tokens. Tests whether a codebook's lift comes from co-clustering repeated words (a concept's
own vocabulary) rather than grouping different words of one concept.

For concept c, over ordered pairs of its positive sampled rows from different documents:
  all   P(same cluster)                        = (S_k - S_kd) / (P - P_d)            (reproduces concept_table.csv)
  diff  P(same cluster | different token)      = (S_k - S_kd - S_kt + S_kdt) / (P - P_d - P_t + P_dt)
  same  P(same cluster | same token)           = (S_kt - S_kdt) / (P_t - P_dt)
with S_x = sum over groups x of n^2 (k cluster, d document, t token) and P_x the same without the cluster; each divided
by the same quantity over ALL sampled rows (chance). Concepts = the concept_id column of a concept_table.csv.

    python -u eval_out/atlas_lift_by_word.py --assign merged.npz --chunks chunks.npz --table concept_table.csv \
        --arms base,base_tokmean,bypass,plain,plain_tokmean,bypass_plain --out lift_by_word.json
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--assign", required=True)
ap.add_argument("--chunks", required=True)
ap.add_argument("--table", required=True)
ap.add_argument("--arms", required=True)
ap.add_argument("--compare", default="bypass:plain_tokmean,bypass_plain:plain_tokmean,bypass:base_tokmean,bypass:base")
ap.add_argument("--out", required=True)
args = ap.parse_args()

A = np.load(args.assign)
Z = np.load(args.chunks)
row_chunk = A["row_chunk"].astype(np.int64)
row_pos = A["row_pos"].astype(np.int64)
ntok = A["chunk_ntok"].astype(np.int64)
s = np.flatnonzero(row_pos != ntok[row_chunk] - 1)                 # sampled rows (final-token rows excluded)
chunk = row_chunk[s]
_, doc = np.unique(A["chunk_doc"].astype(np.int64)[chunk], return_inverse=True)
_, tok = np.unique(A["row_tok"].astype(np.int64)[s], return_inverse=True)
doc, tok = doc.reshape(-1).astype(np.int64), tok.reshape(-1).astype(np.int64)
with open(args.table) as f:
    cids = np.array([int(r["concept_id"]) for r in csv.DictReader(f)], dtype=np.int64)
cmap = np.full(int(Z["label_ids"].max()) + 1, -1, dtype=np.int64)
cmap[cids] = np.arange(len(cids))
C = len(cids)

# positive (sampled row, concept) pairs, concepts restricted to the table
lp = Z["label_ptr"].astype(np.int64)
per_row = np.diff(lp)[chunk]
row_rep = np.repeat(np.arange(len(s)), per_row)
starts = np.repeat(lp[chunk], per_row)
offs = np.arange(len(row_rep)) - np.repeat(np.cumsum(per_row) - per_row, per_row)
cc = cmap[Z["label_ids"].astype(np.int64)[starts + offs]]
keep = cc >= 0
row_rep, cc = row_rep[keep], cc[keep]
n_pos = np.bincount(cc, minlength=C)
print(f"{len(s):,} sampled rows, {C} concepts, {len(cc):,} positive (row, concept) pairs", flush=True)

ND, NT = int(doc.max()) + 1, int(tok.max()) + 1


def sq_by_group(group_of_c, *fields):
    """per concept (or a single global group when group_of_c is None): sum over groups of n^2."""
    span = float(np.prod([float(n) for _, n in fields])) * (C if group_of_c is not None else 1)
    assert span < 2 ** 62, "composite key overflows int64"
    key = np.zeros(len(fields[0][0]), dtype=np.int64)
    for f, n in fields:
        key = key * n + f
    if group_of_c is None:
        _, cnt = np.unique(key, return_counts=True)
        return float((cnt.astype(np.float64) ** 2).sum())
    key = group_of_c * int(np.prod([n for _, n in fields], dtype=np.float64)) + key
    u, cnt = np.unique(key, return_counts=True)
    per = int(np.prod([n for _, n in fields], dtype=np.float64))
    return np.bincount(u // per, weights=cnt.astype(np.float64) ** 2, minlength=C)


def rates(lab_rows, cpos):
    """(all, diff, same) same-cluster rates: per concept (cpos = concept per pair) or global (cpos None)."""
    d, t, k = doc[lab_rows], tok[lab_rows], lab[lab_rows]
    one = np.zeros(len(lab_rows), dtype=np.int64)
    if cpos is None:
        P = float(len(lab_rows)) ** 2
    else:
        P = n_pos.astype(np.float64) ** 2
    g = cpos
    Sk = sq_by_group(g, (k, K))
    Skd = sq_by_group(g, (k, K), (d, ND))
    Skt = sq_by_group(g, (k, K), (t, NT))
    Skdt = sq_by_group(g, (k, K), (d, ND), (t, NT))
    Pd = sq_by_group(g, (d, ND))
    Pt = sq_by_group(g, (t, NT))
    Pdt = sq_by_group(g, (d, ND), (t, NT))
    del one
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((Sk - Skd) / (P - Pd), (Sk - Skd - Skt + Skdt) / (P - Pd - Pt + Pdt), (Skt - Skdt) / (Pt - Pdt))


res = {"n_sampled_rows": int(len(s)), "n_concepts": C, "arms": {}, "compare": {}}
lifts = {}
all_rows = np.arange(len(s))
for arm in args.arms.split(","):
    lab = A[arm].astype(np.int64)[s]
    lab = np.where(lab < 0, lab.max() + 1, lab)
    K = int(lab.max()) + 1
    G = rates(all_rows, None)
    R = rates(row_rep, cc)
    L = {m: R[i] / G[i] for i, m in enumerate(("all", "diff", "same"))}
    lifts[arm] = L
    res["arms"][arm] = {m: {"geo_mean": float(np.exp(np.nanmean(np.log(L[m])))), "chance": G[i]}
                        for i, m in enumerate(("all", "diff", "same"))}
    print(f"{arm:14s} lift all {res['arms'][arm]['all']['geo_mean']:.3f}  diff-token "
          f"{res['arms'][arm]['diff']['geo_mean']:.3f}  same-token {res['arms'][arm]['same']['geo_mean']:.3f}", flush=True)

# reproduction check against the table's lift column
with open(args.table) as f:
    rows = list(csv.DictReader(f))
for arm in lifts:
    col = f"lift_{arm}"
    if col in rows[0]:
        ref = np.array([float(r[col]) for r in rows])
        res["arms"][arm]["max_abs_rel_diff_vs_table"] = float(np.nanmax(np.abs(lifts[arm]["all"] / ref - 1)))
        print(f"  {arm}: max relative difference from {col} = {res['arms'][arm]['max_abs_rel_diff_vs_table']:.2e}")

rng = np.random.default_rng(0)
boot = rng.integers(0, C, size=(10000, C))
for spec in args.compare.split(","):
    x, y = spec.split(":")
    if x not in lifts or y not in lifts:
        continue
    out = {}
    for m in ("all", "diff", "same"):
        dl = np.log(lifts[x][m] / lifts[y][m])
        ok = np.isfinite(dl)
        d = dl[ok]
        bi = boot[:, :len(d)] % len(d)
        ci = np.exp(np.percentile(d[bi].mean(1), [2.5, 97.5]))
        out[m] = {"n": int(ok.sum()), "gmr": float(np.exp(d.mean())), "ci95_concept_boot": ci.tolist(),
                  "win": float((d > 0).mean())}
    res["compare"][f"{x}_vs_{y}"] = out
    print(f"{x} vs {y}: " + "  ".join(f"{m} {o['gmr']:.3f} [{o['ci95_concept_boot'][0]:.3f}, "
                                        f"{o['ci95_concept_boot'][1]:.3f}] win {o['win']:.2f}" for m, o in out.items()))
json.dump(res, open(args.out, "w"), indent=1)
print("wrote", args.out)
