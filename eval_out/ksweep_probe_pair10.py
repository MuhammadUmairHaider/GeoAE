"""K-sweep probes under Table 3's rule: rows whose cluster holds >= 10 kept rows in BOTH spaces of the pair
(token-removed k-means, GeoAE), iterated to a fixed point (eval_out/atlas_nmi_filtered.filter_rows); raw h/c/NMI
(eval_out/probe_hc_filtered.HC). Also checks that the cached rows reproduce Table 3's GeoAE min10 values."""
import json, sys
import numpy as np
sys.path.insert(0, "eval_out")
from probe_hc_filtered import HC
from atlas_nmi_filtered import filter_rows

Z = np.load("results/delta/g4b_l33_d5120/evals/kcost/probe_labels.npz")
T3 = json.load(open("results/delta/g4b_l33_nobypass/evals/probe_dense_subsets.json"))["rungs"]
TASKS = ["domain", "topic4", "topic14", "topic20"]
out = {}
for t in TASKS:
    y = Z[f"{t}__y"]
    # check: GeoAE (same checkpoint) on its own min10 rows must equal Table 3's file
    b = Z[f"{t}__bypass"]
    _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 10
    h = HC(y[keep], b[keep])()["homogeneity"]
    ref = T3[t]["arms"]["geoae"]["min10"]
    print(f"{t:8s} rows {len(y)} vs T3 {T3[t]['n']} | geoae min10 h {h:.4f} vs T3 {ref['homogeneity']:.4f} (n {keep.sum()} vs {ref['n']})")
    # Table 3's own K=2000 pair values for reference
    p = T3[t]["arms"]
    print(f"         T3 K2000 pair_fp10: tok_plain {p['tok_plain']['pair_fp10_tok_plain+geoae']['homogeneity']:.3f} geoae {p['geoae']['pair_fp10_tok_plain+geoae']['homogeneity']:.3f}")
    out[t] = {}
    for K, km, ae in ((4000, "plain_tokmean4k", "bypass4k"), (8000, "plain_tokmean8k", "bypass8k")):
        L = {km: Z[f"{t}__{km}"], ae: Z[f"{t}__{ae}"]}
        keep, rounds = filter_rows(L, 10)
        s = {a: HC(y[keep], L[a][keep])() for a in (km, ae)}
        out[t][K] = dict(kept=float(keep.mean()), km=s[km], ae=s[ae])
        print(f"         K{K}: kept {keep.mean():.0%} ({rounds} rounds)  km h {s[km]['homogeneity']:.3f}  geoae h {s[ae]['homogeneity']:.3f}")
json.dump(out, open(sys.argv[1], "w"), indent=1)
