"""Matched base (hb) vs the token-bypass AE (z) vs the plain base (h), per intervention.

    h    edit the residual directly
    hb   edit (h - mean)/std - b[tok] at every position, add b[tok] back (no encoder)
    z    the token-bypass AE: encode (x - b[tok]), edit, decode + b[tok]

The hb runs re-run h on the bypass AE's own joint-correct doc sets; z comes from the
earlier bypass-AE files. The pairing is only valid if the re-run h reproduces the old h
exactly — checked and reported first.

    z − hb   the ENCODER's contribution to edit quality (the fair base comparison)
    hb − h   what removing the token mean does to a plain residual edit

Steering sign is flipped so + = better everywhere. Paired over concepts (Wilcoxon).

    .venv/bin/python eval_out/summarize_tokbias_hb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_tokbias_interventions as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETS = [  # (title, hb file (h + hb), z file (bypass AE), sign)
    ("DB14 steering", "results/steer_db14_d6144_tokbias_hb.json", "results/steer_db14_d6144_tokbias.json", -1),
    ("DB14 range interventions", "results/range_intervention_db14_d6144_tokbias_dprime_hb.json",
     "results/range_intervention_db14_d6144_tokbias_dprime.json", 1),
    ("bias_in_bios range interventions", "results/range_intervention_biasbios_d6144_tokbias_dprime_hb.json",
     "results/range_intervention_biasbios_d6144_tokbias_dprime.json", 1),
]


def _p(v):
    return stats.wilcoxon(v).pvalue if np.any(v != 0) else 1.0


def one(title, f_hb, f_z, sign):
    print(f"\n{'=' * 96}\n{title.upper()} — + = first arm better; p = Wilcoxon over concepts\n{'=' * 96}")
    if not (ROOT / f_hb).exists() or not (ROOT / f_z).exists():
        print(f"  (missing: {f_hb if not (ROOT / f_hb).exists() else f_z})")
        return
    N = json.load(open(ROOT / f_hb))["concepts"]
    Z = json.load(open(ROOT / f_z))["concepts"]
    cs = [c for c in N if c in Z]
    hk = [(c, k) for c in cs for k in N[c] if k.startswith("h_") and k in Z[c]]
    same = sum(N[c][k] == Z[c][k] for c, k in hk)
    print(f"  pairing check: re-run h identical to the z file's h in {same}/{len(hk)} cells"
          + ("" if same == len(hk) else "  !! NOT identical — hb and z are on different doc sets or runs"))
    ops = [k[3:] for k in N[cs[0]] if k.startswith("hb_") and "z_" + k[3:] in Z[cs[0]]]
    print(f"  {'operator':22s} {'h':>7s} {'hb':>7s} {'z':>7s} | {'z − hb (encoder)':>22s} | {'hb − h (removal)':>22s}")
    per_zhb, per_hbh = [], []
    for op in ops:
        sel = lambda D, s: sign * np.array([D[c][f"{s}_{op}"]["selectivity"] for c in cs])
        h, hb, z = sel(N, "h"), sel(N, "hb"), sel(Z, "z")
        # z is read against the AE's own reconstruction and h/hb against the base model, as in
        # every z − h table: compare selectivities, which are drops from each arm's own reference.
        d1, d2 = z - hb, hb - h
        per_zhb.append(d1); per_hbh.append(d2)
        print(f"  {op:22s} {h.mean():+7.3f} {hb.mean():+7.3f} {z.mean():+7.3f} | "
              f"{d1.mean():+7.3f} p={_p(d1):.3f} {int((d1 > 0).sum()):2d}/{len(d1):<3d} | "
              f"{d2.mean():+7.3f} p={_p(d2):.3f} {int((d2 > 0).sum()):2d}/{len(d2):<3d}")
    for name, sel_ops in (("POOLED all operators", ops),
                          ("POOLED steering ops", [o for o in ops if o.startswith("st_") or title.endswith("steering")])):
        j = [ops.index(o) for o in sel_ops]
        a = np.mean([per_zhb[i] for i in j], 0)
        b = np.mean([per_hbh[i] for i in j], 0)
        print(f"  {name:22s} {'':23s} | {a.mean():+7.3f} p={_p(a):.3f} {int((a > 0).sum()):2d}/{len(a):<3d} | "
              f"{b.mean():+7.3f} p={_p(b):.3f} {int((b > 0).sum()):2d}/{len(b):<3d}")


def main():
    print("L27 d6144 sampled — token-bypass AE vs its MATCHED BASE (edit h − b[tok], add b[tok] back)")
    for s in SETS:
        one(*s)
    T.number_hb()


if __name__ == "__main__":
    main()
