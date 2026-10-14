"""Homogeneity, completeness and raw NMI of codebooks against probe labels (or the current / next token), on the rows
whose cluster holds >= --min_rows rows in EVERY codebook of a group (the reporting rule: never NMI alone, tiny clusters
filtered, the plain token-removed k-means control in every group).

Input: the --save_labels npz of eval_out/probe_chance_corrected.py (per rung: <rung>__y, <rung>__<arm>) or of
eval_out/token_structure.py (cur, next, <arm>). Groups: "name=arm1,arm2,...;name2=..." — the filter of
eval_out/atlas_nmi_filtered.filter_rows is applied within a group (intersection over its arms, to a fixed point), so
every arm of a group is scored on the same rows; --ref (the last arm of each group by default) is compared with the rest.
Scores from the weighted contingency table: h = I/H(Y), c = I/H(C), NMI = 2I/(H(Y)+H(C)) (= V-measure, arithmetic
normalisation, sklearn's homogeneity_completeness_v_measure on unit weights). Row bootstrap (multinomial weights),
95% percentile CIs of each score and of ref - arm.

    scripts/delta/py eval_out/probe_hc_filtered.py --labels probe_labels.npz --mode probe \
        --groups "K2000=plain,plain_tokmean,bypass;K4000=plain4k,plain_tokmean4k,bypass4k" --out probe_hc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_nmi_filtered import filter_rows  # noqa: E402

METRICS = ("homogeneity", "completeness", "nmi")


def _H(counts):
    p = counts[counts > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


class HC:
    """h / c / NMI of class ids y against cluster ids c for any row weights (bootstrap) from one encoding."""

    def __init__(self, y, c):
        self.y = np.unique(y, return_inverse=True)[1]
        self.c = np.unique(c, return_inverse=True)[1]
        self.pair = np.unique(self.y.astype(np.int64) * (self.c.max() + 1) + self.c, return_inverse=True)[1]

    def __call__(self, w=None):
        w = np.ones(len(self.y)) if w is None else w
        hy, hc = _H(np.bincount(self.y, w)), _H(np.bincount(self.c, w))
        i = hy + hc - _H(np.bincount(self.pair, w))
        h = i / hy if hy > 0 else 1.0
        c = i / hc if hc > 0 else 1.0
        nmi = 2 * i / (hy + hc) if hy + hc > 0 else 1.0
        return dict(homogeneity=h, completeness=c, nmi=nmi)


def score_group(y, labels, arms, ref, min_rows, n_boot, rng):
    keep, rounds = filter_rows({a: labels[a] for a in arms}, min_rows)
    out = dict(n_rows=int(len(y)), n_kept=int(keep.sum()), kept_share=float(keep.mean()), filter_rounds=rounds,
               n_classes_kept=int(len(np.unique(y[keep]))), arms={}, diff_ref_minus_arm={})
    if keep.sum() < 2:
        return out
    hc = {a: HC(y[keep], labels[a][keep]) for a in arms}
    n = int(keep.sum())
    boots = {a: {m: [] for m in METRICS} for a in arms}
    for _ in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
        for a in arms:
            s = hc[a](w)
            for m in METRICS:
                boots[a][m].append(s[m])
    for a in arms:
        p = hc[a]()
        out["arms"][a] = dict(p, clusters=int(len(np.unique(labels[a][keep]))),
                              ci={m: [float(x) for x in np.quantile(boots[a][m], [.025, .975])] for m in METRICS})
    for a in arms:
        if a == ref:
            continue
        out["diff_ref_minus_arm"][a] = {m: dict(point=out["arms"][ref][m] - out["arms"][a][m],
                                                ci=[float(x) for x in np.quantile(np.subtract(boots[ref][m], boots[a][m]),
                                                                                  [.025, .975])]) for m in METRICS}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mode", choices=("probe", "tokstruct"), required=True)
    ap.add_argument("--groups", required=True, help="name=arm,arm,...;name=...  (the last arm of a group is its ref)")
    ap.add_argument("--rungs", default="", help="probe mode: comma list of rungs (default all in the file)")
    ap.add_argument("--targets", default="cur,next", help="tokstruct mode: which token labels to score")
    ap.add_argument("--min_rows", type=int, default=6)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Z = np.load(args.labels)
    groups = {}
    for g in filter(None, args.groups.split(";")):
        name, arms = g.split("=", 1)
        groups[name] = [a for a in arms.split(",") if a]
    if args.mode == "probe":
        rungs = sorted({k.split("__")[0] for k in Z.files if k.endswith("__y")})
        if args.rungs:
            rungs = [r for r in args.rungs.split(",") if r in rungs]
        tasks = {r: (Z[f"{r}__y"], {k.split("__", 1)[1]: Z[k] for k in Z.files if k.startswith(f"{r}__")
                                    and k not in (f"{r}__y", f"{r}__ids")}) for r in rungs}
    else:
        arm_labels = {k: Z[k] for k in Z.files if k not in ("cur", "next")}
        tasks = {t: (Z[t], arm_labels) for t in args.targets.split(",") if t in Z.files}
    rng = np.random.default_rng(args.seed)
    res = {"meta": vars(args), "tasks": {}}
    for task, (y, labels) in tasks.items():
        res["tasks"][task] = {}
        for gname, arms in groups.items():
            if any(a not in labels for a in arms):
                continue
            r = score_group(np.asarray(y), labels, arms, arms[-1], args.min_rows, args.n_boot, rng)
            res["tasks"][task][gname] = r
            if not r["arms"]:
                print(f"{task:16s} {gname:6s} kept {r['n_kept']:,}/{r['n_rows']:,} -> too few rows", flush=True)
                continue
            print(f"{task:16s} {gname:6s} kept {r['n_kept']:>7,}/{r['n_rows']:<7,} ({r['kept_share']:.0%}) " + "  ".join(
                f"{a}: h {v['homogeneity']:.3f} c {v['completeness']:.3f} nmi {v['nmi']:.3f}" for a, v in r["arms"].items()),
                flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
