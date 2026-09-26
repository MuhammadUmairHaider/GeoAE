"""Current-token vs predicted-next-token erasure: which one carries the gain?

Reads the three erasure bases (their subspace overlap and held-out R^2 at rank 64)
and eval_out/probe_token_split.json (probe_chance_corrected over two k-means seeds
per arm). Every number is chance-corrected NMI; contrasts are seed-averaged, with a
95% paired bootstrap CI over probe examples, next to the SEED GAP of the same
contrast (seed 42 contrast minus seed 7 contrast), which the bootstrap does not see.
A contrast is called only when its CI excludes 0 AND |mean| > its seed gap.

    .venv/bin/python eval_out/summarize_token_split.py
"""
from __future__ import annotations

import json

import numpy as np

E = "e2e/checkpoints/general/llama3.2-3B/layer27"
BASES = {"both": f"{E}/token_erasure_sampled.npz", "cur": f"{E}/token_erasure_sampled_cur.npz",
         "pred": f"{E}/token_erasure_sampled_pred.npz"}
PROBE = "eval_out/probe_token_split.json"
ARMS = ["raw", "cur64", "pred64", "both64", "pca64"]
SEEDS = ["s42", "s7"]
CONTRASTS = [("cur64", "pred64"), ("both64", "cur64"), ("both64", "pred64"),
             ("cur64", "pca64"), ("pred64", "pca64"), ("both64", "pca64"),
             ("cur64", "raw"), ("pred64", "raw"), ("both64", "raw")]
R = 64


def bases():
    print("=" * 78 + "\nERASURE BASES — rank 64, held-out val rows (from each basis file's report)\n" + "=" * 78)
    U, rep = {}, {}
    for k, p in BASES.items():
        d = np.load(p, allow_pickle=True)
        U[k] = d["U_token"][:, :R]
        U.setdefault("pca", d["U_pca"][:, :R])
        rep[k] = {(r["basis"], r["rank"]): r for r in json.loads(str(d["report"]))}
    ref = rep["both"]
    rows = [("none", ref[("none", 0)]), ("cur", rep["cur"][("token", R)]),
            ("pred", rep["pred"][("token", R)]), ("both", ref[("token", R)]), ("pca", ref[("pca", R)])]
    print(f"  {'basis':6s} | {'var removed':>11s} | {'R2 cur':>7s} | {'R2 pred':>7s} | {'R2 doc':>7s} | {'doc var kept':>12s}")
    for n, r in rows:
        print(f"  {n:6s} | {r['var_removed']:11.3f} | {r['r2_cur']:7.3f} | {r['r2_pred']:7.3f} | "
              f"{r['r2_doc']:7.3f} | {r['doc_var_kept']:12.3f}")
    print(f"\n  subspace overlap (mean squared cosine of principal angles, 1 = same subspace, "
          f"{R / 3072:.3f} = random)")
    names = ["cur", "pred", "both", "pca"]
    print("  " + " " * 6 + "".join(f"{n:>8s}" for n in names))
    for a in names:
        print(f"  {a:6s}" + "".join(f"{float(np.sum((U[a].T @ U[b]) ** 2) / R):8.3f}" for b in names))


def probe():
    d = json.load(open(PROBE))
    print("\n" + "=" * 78 + "\nCONCEPT PROBE — chance-corrected NMI, mean of 2 k-means seeds (seed gap in brackets)\n" + "=" * 78)
    arms_in = next(iter(d.values()))["arms"]
    ae = [a for a in arms_in if not any(a.startswith(x + "_") for x in ARMS)]
    print(f"  {'rung':16s}" + "".join(f"{a:>15s}" for a in ARMS) + "".join(f"{a[:10]:>11s}" for a in ae))
    grp = {"token": {a: [] for a in ARMS}, "sequence": {a: [] for a in ARMS}}
    for rung, v in d.items():
        A = v["arms"]
        cells = []
        for a in ARMS:
            s = [A[f"{a}_{sd}"]["cnmi"] for sd in SEEDS]
            grp[v["grain"]][a].append(np.mean(s))
            cells.append(f"{np.mean(s):7.3f} [{abs(s[0] - s[1]):.3f}]")
        print(f"  {rung:16s}" + "".join(f"{c:>15s}" for c in cells) + "".join(f"{A[a]['cnmi']:11.3f}" for a in ae))
    for g, v in grp.items():
        print(f"  {'MEAN ' + g:16s}" + "".join(f"{np.mean(v[a]):15.3f}" for a in ARMS))

    print("\n" + "=" * 78 + "\nCONTRASTS — seed-averaged cNMI difference [95% CI] (seed gap); "
          "✓ = CI excludes 0 and |Δ| > seed gap\n" + "=" * 78)
    for a, b in CONTRASTS:
        print(f"\n  {a} − {b}")
        tot = {"token": [], "sequence": []}
        for rung, v in d.items():
            A = v["arms"]
            boots = np.mean([np.array(A[f"{a}_{s}"]["boots"]) - np.array(A[f"{b}_{s}"]["boots"]) for s in SEEDS], 0)
            pt = [A[f"{a}_{s}"]["cnmi"] - A[f"{b}_{s}"]["cnmi"] for s in SEEDS]
            m, gap = float(np.mean(pt)), abs(pt[0] - pt[1])
            lo, hi = m + np.quantile(boots - boots.mean(), [.025, .975])   # centred: resampled NMI is biased up
            ok = (lo > 0 or hi < 0) and abs(m) > gap
            tot[v["grain"]].append(m)
            print(f"    {rung:16s} {m:+.3f} [{lo:+.3f},{hi:+.3f}] ({gap:.3f}) {'✓' if ok else ''}")
        print(f"    {'MEAN token':16s} {np.mean(tot['token']):+.3f}   {'MEAN sequence':16s} {np.mean(tot['sequence']):+.3f}")


if __name__ == "__main__":
    bases()
    probe()
