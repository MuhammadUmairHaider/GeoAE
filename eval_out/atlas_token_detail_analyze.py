"""CPU analyses of the token-level Atlas labels dumped by eval_out/atlas_token_detail.py (numpy only).

For every label set (atlas14_s0..2, doctype) and arm, chance-corrected NMI (cnmi = (nmi - null) / (1 - null), null =
mean NMI over 5 label permutations) on:
  repro      the atlas_nmi.py token grain (all rows of qualifying chunks, 80k fixed-seed subsample) - must match it
  one_row    ONE sampled (non-final) row per chunk, 10 draws: no within-chunk correlation
  position   sampled rows binned by token position (bins subsampled to a common n), plus final-token rows
  toktype    sampled rows by current-token type (punct/number, function word, word continuation, content word start)
and, on all rows of qualifying chunks:
  pairs      P(same cluster | same chunk / same doc other chunk / same class other doc / different class); the
             concept lift = P(same class, other doc) / P(different class) separates concept grouping from document
             co-clustering
  clusters   'concept clusters': >= 20 rows, top class share >= 0.5, rows from >= 5 documents
Reference arms: doc_hash / chunk_hash = cluster by document / chunk id modulo 2000 (pure document memorisation).

    python3 eval_out/atlas_token_detail_analyze.py --assign assign.npz --chunks chunks.npz --draws draws.json \
        --doctype atlas_doctype_ids.json --vocab gemma3_vocab.json --out atlas_token_detail.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

ARMS = ("base", "base_tokmean", "bypass", "mat10k", "mat10k_km", "bt10k", "bt10k_km")
STOP = set("""a an the and or but if then else of in on at to for from by with without into onto over under about
above below between among through during before after since until while as than that this these those there here
is are was were be been being am do does did done have has had having can could will would shall should may might
must not no nor so too very just also only even still yet already again ever never always often i me my mine we us
our ours you your yours he him his she her hers it its they them their theirs who whom whose which what when where
why how all any both each few more most other some such own same one s t can't don't won't isn't aren't wasn't
weren't up down out off""".split())


def nmi(y, c):
    ny, nc = y.max() + 1, c.max() + 1
    J = np.bincount(y * nc + c, minlength=ny * nc).reshape(ny, nc).astype(float)
    N = J.sum()
    py, pc, P = J.sum(1) / N, J.sum(0) / N, J / N
    nz = P > 0
    mi = (P[nz] * np.log(P[nz] / (py[:, None] * pc[None, :])[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(mi / max((h(py) + h(pc)) / 2, 1e-12))


def dense(v):
    return np.unique(v, return_inverse=True)[1]


def cnmi(y, c, seed=0):
    y, c = dense(y), dense(c)
    rng = np.random.default_rng(seed)
    m = nmi(y, c)
    null = float(np.mean([nmi(rng.permutation(y), c) for _ in range(5)]))
    return (m - null) / (1 - null)


def boot_diff(y, a, b, n_boot=200, seed=0):
    """cnmi(a) - cnmi(b) with a centred bootstrap interval over rows (paired: same resample for both arms)."""
    y, a, b = dense(y), dense(a), dense(b)
    est = cnmi(y, a) - cnmi(y, b)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        d.append(cnmi(y[s], a[s], 1) - cnmi(y[s], b[s], 1))
    d = np.asarray(d)
    lo, hi = np.quantile(d - d.mean(), [0.025, 0.975])
    return dict(est=est, lo=est + lo, hi=est + hi)


def n_same(*keys):
    """Number of ordered row pairs (i != j) agreeing on every key."""
    k = np.zeros(len(keys[0]), dtype=np.int64)
    for v in keys:
        v = dense(v)
        k = k * (int(v.max()) + 1) + v
    c = np.unique(k, return_counts=True)[1].astype(float)
    return float((c * (c - 1)).sum())


def pair_stats(cl, ch, dc, y):
    N = len(cl)
    tot = N * (N - 1.0)
    s_cl, s_cly = n_same(cl), n_same(cl, y)
    out = {}
    out["same_chunk"] = n_same(cl, ch) / max(n_same(ch), 1)
    out["same_doc_other_chunk"] = (n_same(cl, dc) - n_same(cl, ch)) / max(n_same(dc) - n_same(ch), 1)
    out["same_class_other_doc"] = (s_cly - n_same(cl, y, dc)) / max(n_same(y) - n_same(y, dc), 1)
    out["diff_class"] = (s_cl - s_cly) / max(tot - n_same(y), 1)
    out["concept_lift"] = out["same_class_other_doc"] / max(out["diff_class"], 1e-12)
    out["doc_lift"] = out["same_doc_other_chunk"] / max(out["diff_class"], 1e-12)
    out["chunk_lift"] = out["same_chunk"] / max(out["diff_class"], 1e-12)
    # share of same-class same-cluster pairs that come from one document
    out["class_pairs_within_doc"] = n_same(cl, y, dc) / max(s_cly, 1)
    return out


def concept_clusters(cl, y, dc, min_rows=20, min_share=0.5, min_docs=5):
    cl = dense(cl)
    n_k = np.bincount(cl)
    J = np.bincount(cl * (y.max() + 1) + y, minlength=len(n_k) * (y.max() + 1)).reshape(len(n_k), -1)
    share = J.max(1) / np.maximum(n_k, 1)
    dd = dense(dc)
    D = int(dd.max()) + 1
    nd = np.bincount(np.unique(cl * D + dd) // D, minlength=len(n_k))
    pure = (n_k >= min_rows) & (share >= min_share)
    cross = pure & (nd >= min_docs)
    return dict(n_pure=int(pure.sum()), n_cross_doc=int(cross.sum()),
                rows_pure=float(n_k[pure].sum() / len(cl)), rows_cross_doc=float(n_k[cross].sum() / len(cl)),
                classes_covered=int(len(np.unique(J[cross].argmax(1)))) if cross.any() else 0)


def tok_type(vocab):
    """id -> 0 punct/number/space, 1 function word, 2 word continuation, 3 content word start."""
    t = np.zeros(len(vocab), dtype=np.int8)
    for i, s in enumerate(vocab):
        start = s.startswith("▁")
        w = s.lstrip("▁")
        if not any(ch.isalpha() for ch in w):
            t[i] = 0
        elif start and w.lower() in STOP:
            t[i] = 1
        elif not start:
            t[i] = 2
        else:
            t[i] = 3
    return t


TOKTYPE = ("punct_num", "function", "continuation", "content_start")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--doctype", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = np.load(args.assign)
    z = np.load(args.chunks)
    arms = [a for a in ARMS if a in A.files]
    row_chunk = A["row_chunk"].astype(np.int64)
    row_pos = A["row_pos"].astype(np.int64)
    row_tok = A["row_tok"].astype(np.int64)
    doc = z["doc_int_id"].astype(np.int64)
    ntok = z["n_tok"].astype(np.int64)
    row_start = z["row_start"].astype(np.int64)
    is_last = row_pos == ntok[row_chunk] - 1
    ttype = tok_type(json.load(open(args.vocab)))[row_tok]
    lab = {a: A[a].astype(np.int64) for a in arms}
    lab["doc_hash"] = doc[row_chunk] % 2000
    lab["chunk_hash"] = row_chunk % 2000
    allarms = arms + ["doc_hash", "chunk_hash"]

    ptr, ids = z["label_ptr"], z["label_ids"]
    chunk_of = np.repeat(np.arange(len(doc)), np.diff(ptr))
    draws = json.load(open(args.draws))
    sets = {f"atlas14_s{d['seed']}": [c["ids"] for c in d["classes"]] for d in draws["draws"]}
    sets["doctype"] = [[i] for i in json.load(open(args.doctype))["ids"]]

    res = {}
    for sname, units in sets.items():
        M = np.zeros((len(doc), len(units)), dtype=bool)
        for k, u in enumerate(units):
            M[chunk_of[np.isin(ids, u)], k] = True
        chunks = np.flatnonzero(M.sum(1) == 1)
        clab = np.full(len(doc), -1)
        clab[chunks] = M[chunks].argmax(1)
        rows = np.concatenate([np.arange(row_start[c], row_start[c + 1]) for c in chunks])
        y_all = clab[row_chunk]
        R = {}
        print(f"== {sname}: {len(chunks)} chunks, {len(rows)} rows", flush=True)

        # repro of atlas_nmi token grain (80k fixed-seed subsample of all rows, chunk order)
        r = rows if len(rows) <= 80000 else rows[np.sort(np.random.RandomState(0).choice(len(rows), 80000, replace=False))]
        R["repro"] = {a: cnmi(y_all[r], lab[a][r]) for a in allarms}
        print("  repro", {a: round(v, 3) for a, v in R["repro"].items()}, flush=True)

        # one sampled row per chunk, 10 draws
        samp = rows[~is_last[rows]]
        rng = np.random.default_rng(0)
        per = {a: [] for a in allarms}
        first = None
        for d in range(10):
            key = rng.random(len(samp))
            order = np.lexsort((key, row_chunk[samp]))
            s = samp[order]
            pick = s[np.r_[True, row_chunk[s][1:] != row_chunk[s][:-1]]]
            first = pick if first is None else first
            for a in allarms:
                per[a].append(cnmi(y_all[pick], lab[a][pick]))
        R["one_row"] = {a: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for a, v in per.items()}
        R["one_row_diff"] = {f"bypass-{b}": boot_diff(y_all[first], lab["bypass"][first], lab[b][first])
                             for b in ("base_tokmean", "base", "mat10k") if b in lab}
        print("  one_row", {a: round(v["mean"], 3) for a, v in R["one_row"].items()}, flush=True)
        print("  one_row_diff", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in R["one_row_diff"].items()},
              flush=True)

        # position bins (sampled rows), common n per bin; final-token rows separately
        edges = [4, 16, 32, 64, 96, 10 ** 9]
        bins = [(samp[(row_pos[samp] >= lo) & (row_pos[samp] < hi)], f"{lo}-{hi if hi < 10**9 else 'end'}")
                for lo, hi in zip(edges[:-1], edges[1:])]
        last = rows[is_last[rows]]
        bins.append((last, "final"))
        n_min = min(len(b) for b, _ in bins)
        R["position"] = {"n_per_bin": int(n_min)}
        for b, name in bins:
            b = np.sort(np.random.RandomState(1).choice(b, n_min, replace=False))
            R["position"][name] = {a: cnmi(y_all[b], lab[a][b]) for a in allarms}
        print("  position n", n_min, {k: round(v["bypass"] - v["base_tokmean"], 3) for k, v in R["position"].items()
                                      if isinstance(v, dict)}, flush=True)

        # token type (sampled rows), common n
        groups = [(samp[ttype[samp] == t], TOKTYPE[t]) for t in range(4)]
        n_min = min(len(g) for g, _ in groups)
        R["toktype"] = {"n_per_type": int(n_min), "share": {nm: float(len(g) / len(samp)) for g, nm in groups}}
        for g, name in groups:
            g = np.sort(np.random.RandomState(2).choice(g, n_min, replace=False))
            R["toktype"][name] = {a: cnmi(y_all[g], lab[a][g]) for a in allarms}
        print("  toktype n", n_min, {k: round(v["bypass"] - v["base_tokmean"], 3) for k, v in R["toktype"].items()
                                     if isinstance(v, dict) and "bypass" in v}, flush=True)

        # pair decomposition + concept clusters on all rows
        yr, chr_, dcr = y_all[rows], row_chunk[rows], doc[row_chunk[rows]]
        R["pairs"] = {a: pair_stats(lab[a][rows], chr_, dcr, yr) for a in allarms}
        R["clusters"] = {a: concept_clusters(lab[a][rows], yr, dcr) for a in allarms if not a.endswith("_hash")}
        print("  concept_lift", {a: round(v["concept_lift"], 2) for a, v in R["pairs"].items()},
              "chunk_lift", {a: round(v["chunk_lift"], 1) for a, v in R["pairs"].items()}, flush=True)
        res[sname] = R
        json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
