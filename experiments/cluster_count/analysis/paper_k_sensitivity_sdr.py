"""K2000 vs K4000 same-direction removal (default d6144; or pass label=path ...): matched-damage sibling/random dNLL, sibling/target ratio
(median over cases of sibling dNLL / L, as in paper_atlas_sibling_removal.sparing_ratio), inner/outer split.
Matched level L per case = 0.5 x min over arms of max target dNLL (same as paper_same_direction_removal)."""

from geoae.paths import resolve_input_path
import json
import sys
import numpy as np

ARMS = ["dir_range_z", "dir_all_z", "dir_range_h", "nl_h", "nl_hb", "nl_z"]
rng = np.random.default_rng(0)


def boot(v, f):
    v = np.asarray(v)
    b = [f(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
    return f(v), np.quantile(b, .025), np.quantile(b, .975)


def load(path):
    d = json.load(open(resolve_input_path(path)))
    grid = d["meta"]["grid"]
    out = []
    for c in d["cases"]:
        cv = {}
        for a in ARMS:
            pts = [(0., 0., 0.)] + [tuple(c["res"][a][f"{s:g}"][g]["dnll"] for g in ("target", "sibling", "random")) for s in grid[a]]
            pts = np.asarray(pts, float)
            cv[a] = pts[np.argsort(pts[:, 0], kind="stable")]
        L = 0.5 * min(cv[a][:, 0].max() for a in ARMS)
        row = {"pos": c["T_position"], "L": L}
        for a in ARMS:
            row[a] = (float(np.interp(L, cv[a][:, 0], cv[a][:, 1])), float(np.interp(L, cv[a][:, 0], cv[a][:, 2])))
        out.append(row)
    return out


res = {}
RUNS = [a.split("=", 1) for a in sys.argv[1:]] or [
    ("K2000", "results/delta/d6144/evals/same_direction_removal.json"),
    ("K4000", "results/delta/d6144k4000/evals/same_direction_removal.json")]
for K, p in RUNS:
    rows = load(p)
    rows = [r for r in rows if r["L"] > 0]
    print(f"\n== {K}: n={len(rows)} L median {np.median([r['L'] for r in rows]):.3f}")
    print(f"{'arm':12s} | sib mean [CI]          | rnd mean [CI]          | sib/tgt median [CI]    | inner sib/L  outer sib/L (medians) | inner sib  outer sib (means)")
    res[K] = {}
    for a in ARMS:
        sib = [r[a][0] for r in rows]; rnd = [r[a][1] for r in rows]; ratio = [r[a][0] / r["L"] for r in rows]
        inn = [r for r in rows if r["pos"] == "inner"]; out = [r for r in rows if r["pos"] == "outer"]
        s, r_, q = boot(sib, np.mean), boot(rnd, np.mean), boot(ratio, np.median)
        ri = np.median([r[a][0] / r["L"] for r in inn]); ro = np.median([r[a][0] / r["L"] for r in out])
        si = np.mean([r[a][0] for r in inn]); so = np.mean([r[a][0] for r in out])
        res[K][a] = dict(sib=s, rnd=r_, ratio=q, ri=ri, ro=ro, si=si, so=so, n_in=len(inn), n_out=len(out),
                         Li=np.median([r["L"] for r in inn]), Lo=np.median([r["L"] for r in out]))
        print(f"{a:12s} | {s[0]:.3f} [{s[1]:.3f},{s[2]:.3f}] | {r_[0]:.4f} [{r_[1]:.4f},{r_[2]:.4f}] | {q[0]:.3f} [{q[1]:.3f},{q[2]:.3f}] | {ri:.3f} {ro:.3f} | {si:.3f} {so:.3f}")
    print(f"  inner n={res[K]['nl_h']['n_in']} L med {res[K]['nl_h']['Li']:.3f}; outer n={res[K]['nl_h']['n_out']} L med {res[K]['nl_h']['Lo']:.3f}")
    # share of cases where range-z sibling damage < 10% of target damage
    print("  dir_range_z cases with sib < 0.1*L:", sum(r['dir_range_z'][0] < 0.1 * r['L'] for r in rows), "/", len(rows))
