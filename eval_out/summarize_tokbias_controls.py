"""Is the token-bypass AE's gain the ENCODER, or just the token-mean subtraction?

Arms (chance-corrected NMI, eval_out/probe_tokbias_controls.json, cache_ids/):
  d6144_new     parent AE
  tokbias       token-bypass AE: encoder on x - b[tok]
  km_tokmean    balanced k-means on x - b[tok], SAME table, dpc init, no encoder  <- matched control
  km_tok64      balanced k-means, 64 token directions projected out
  km_pca64      balanced k-means, 64 top-variance directions projected out (All-but-the-Top)
  km_new        balanced k-means on x
Every k-means arm has two seeds (42, 7); its cell is the seed mean, and each contrast
shows the gap between its two per-seed values next to a 95% paired bootstrap CI
(centred on the point estimate; see the note in main()).
A contrast is called (✓) only if the CI excludes 0 AND |Δ| exceeds that seed gap.

    .venv/bin/python eval_out/summarize_tokbias_controls.py
"""
from __future__ import annotations

import json

import numpy as np

PROBE = "eval_out/probe_tokbias_controls.json"
AES = ["d6144_new", "tokbias"]
KMS = ["km_tokmean", "km_tok64", "km_pca64", "km_new"]
SEEDS = ["", "_s7"]
CONTRASTS = [("tokbias", "km_tokmean"), ("km_tokmean", "km_new"), ("km_tokmean", "km_tok64"),
             ("tokbias", "d6144_new")]


def cells(A, arm):
    """(seed-mean cNMI, per-seed cNMI list, per-seed bootstrap arrays)."""
    names = [arm] if arm in AES else [arm + s for s in SEEDS]
    vals = [A[n]["cnmi"] for n in names]
    boots = [np.array(A[n]["boots"]) for n in names]
    return float(np.mean(vals)), vals, boots


def main():
    d = json.load(open(PROBE))
    arms = AES + KMS
    print("=" * 96)
    print("CHANCE-CORRECTED NMI — k-means arms are the mean of 2 seeds [seed gap]; leak = cNMI(label; token id)")
    print("=" * 96)
    print(f"  {'rung':16s} {'leak':>5s} " + "".join(f"{a:>17s}" for a in arms))
    grp = {"token": {a: [] for a in arms}, "sequence": {a: [] for a in arms}}
    for r, v in d.items():
        A = v["arms"]
        row = []
        for a in arms:
            m, vals, _ = cells(A, a)
            grp[v["grain"]][a].append(m)
            row.append(f"{m:.3f}" + (f" [{abs(vals[0] - vals[1]):.3f}]" if len(vals) > 1 else " " * 8))
        leak = "  —" if v["leak"] is None else f"{v['leak']:.2f}"
        print(f"  {r:16s} {leak:>5s} " + "".join(f"{c:>17s}" for c in row))
    for g, cols in grp.items():
        print(f"  {'MEAN ' + g:22s} " + "".join(f"{np.mean(cols[a]):17.3f}" for a in arms))

    for a, b in CONTRASTS:
        print(f"\n  {a} − {b}   (Δ [95% CI] (seed gap); ✓ = CI excludes 0 and |Δ| > seed gap)")
        tot = {"token": [], "sequence": []}
        for r, v in d.items():
            A = v["arms"]
            ma, va, ba = cells(A, a)
            mb, vb, bb = cells(A, b)
            k = max(len(ba), len(bb))
            per_seed = [va[min(i, len(va) - 1)] - vb[min(i, len(vb) - 1)] for i in range(k)]
            boots = np.mean([ba[min(i, len(ba) - 1)] - bb[min(i, len(bb) - 1)] for i in range(k)], 0)
            m = float(np.mean(per_seed))
            gap = abs(per_seed[0] - per_seed[-1])
            # Centred (basic) bootstrap: NMI on a with-replacement resample is biased UP,
            # more so for arms that spread a small rung over many clusters, so the raw
            # percentile interval can miss its own point estimate. Use the resampling
            # SPREAD around the point estimate instead.
            lo, hi = m + np.quantile(boots - boots.mean(), [.025, .975])
            ok = (lo > 0 or hi < 0) and abs(m) > gap
            tot[v["grain"]].append(m)
            print(f"    {r:16s} {m:+.3f} [{lo:+.3f},{hi:+.3f}] ({gap:.3f}) {'✓' if ok else ''}")
        print(f"    MEAN token {np.mean(tot['token']):+.3f}   MEAN sequence {np.mean(tot['sequence']):+.3f}")


if __name__ == "__main__":
    main()
