"""Direction-projection faithfulness tables (geoae/interp/range_intervention_compare.py --proj).

Per concept and method, ONE assigned direction (or the span of k directions) is removed from the UNEDITED base model's residual
stream by orthogonal projection, and the gold-label confidence is read on the concept's own documents (target) and on the
sampled other-class documents. Every method shares the model, the documents and the starting confidence, so no scale choice
can favour one of them.
  target removed = -(mean edited - mean unedited confidence on the concept's documents), as % of the unedited confidence
  collateral     = mean over other-class documents of max(0, unedited - edited), as % of their unedited confidence
  selectivity    = target removed - collateral                                             (higher = better)
Operators: pjc = projection of (h - m), m = class-balanced mean of the fit documents at the answer position (default);
           pj  = plain projection h - (h.d)d.
Concept averages are 10%-trimmed means (floor(trim * n) cut per side); differences between two methods are trimmed means of
the per-concept paired differences with a 95% bootstrap CI over concepts (10k, seed 0).

    python eval_out/paper_projection_table.py [--arm g4b_l33_d5120] [--op pjc] [--scale rel|abs] [--tex]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="g4b_l33_d5120")
ap.add_argument("--op", default="pjc", choices=["pjc", "pj"])
ap.add_argument("--scale", default="rel", choices=["rel", "abs"])
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--tag", default="", help="file suffix, e.g. _smoke")
ap.add_argument("--pairs", default="geoae:sae_f1,geoae:sae_f5,geoae:sae_km,geoae:km,geoae:kmt,geoae:nobypass,"
                                   "geoae_z5:sae_f5,nobypass_z5:sae_f5,geoae:label,sae_f5:label")
ap.add_argument("--tex", action="store_true")
args = ap.parse_args()
B = 10000
DS = [("db14", "DBpedia-14"), ("biasbios", "Bias in Bios"), ("ag_news", "AG News"), ("emotion6", "Emotion"), ("news20", "20NG"),
      ("atlas_s0", "FineWeb-Atlas, draw 0")]
METHODS = [("label", "Supervised"), ("km", "k-means"), ("kmt", "Tok.-rem. k-means"), ("sae_km", "SAE k-means"),
           ("nobypass", "No-bypass AE"), ("geoae", "GeoAE"),
           ("h_n1", "Residual, top-1 dim"), ("sae_f1", "SAE, top-1 feature"), ("nobypass_z1", "No-bypass AE, top-1 dim"),
           ("geoae_z1", "GeoAE, top-1 dim"),
           ("h_n5", "Residual, top-5 dims"), ("sae_f5", "SAE, top-5 features"), ("nobypass_z5", "No-bypass AE, top-5 dims"),
           ("geoae_z5", "GeoAE, top-5 dims")]


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def ci(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return np.nan, np.nan
    bs = tmean(d[np.random.default_rng(0).integers(0, len(d), (B, len(d)))])
    return tuple(np.quantile(bs, [.025, .975]))


def metrics(c, name):
    """(start conf, target removed, collateral, selectivity) for one concept; NaNs -> zero effect if the method had no direction."""
    ref = c["ref"]["h"]; nt = len(c["eval_idx"]["tgt"])
    r = np.array(ref["doc_conf"], float)
    key = f"h_{args.op}_{name}"
    if key not in c:
        if c.get("proj", {}).get(name, {}).get("degenerate"):
            return 100 * r[:nt].mean(), 0.0, 0.0, 0.0
        return None
    e = np.array(c[key]["doc_conf"], float)
    rem, coll = r[:nt].mean() - e[:nt].mean(), np.maximum(0, r[nt:] - e[nt:]).mean()
    if args.scale == "rel":
        rem, coll = rem / r[:nt].mean(), coll / r[nt:].mean()
    return 100 * r[:nt].mean(), 100 * rem, 100 * coll, 100 * (rem - coll)


root = Path("results/delta") / args.arm / "interventions" / "proj"
data = {}
for ds, nm in DS:
    p = root / f"direction_projection_{ds}{args.tag}.json"
    if p.exists():
        d = json.loads(p.read_text())
        data[ds] = (nm, d, "summary" in d)
unit = "% of the unedited confidence" if args.scale == "rel" else "confidence points"
print(f"# Direction projection, {args.arm}, operator {args.op}, {unit}; {args.trim:.0%}-trimmed means over concepts\n")
tab = {}
for ds, (nm, d, done) in data.items():
    C = list(d["concepts"].values())
    print(f"## {nm} ({len(C)} concepts{'' if done else ', RUNNING'}; unedited target confidence "
          f"{np.mean([100 * np.mean(c['ref']['h']['doc_conf'][:len(c['eval_idx']['tgt'])]) for c in C]):.1f}%)\n")
    print("| Method | Target removed | Collateral | Selectivity | Distinct units | Coverage / purity |")
    print("|---|---|---|---|---|---|")
    for m, lab in METHODS:
        rows = [metrics(c, m) for c in C]
        rows = np.array([r for r in rows if r is not None])
        if not len(rows):
            continue
        info = [c["proj"][m] for c in C if m in c.get("proj", {})]
        units = len({i["unit"] for i in info if "unit" in i})
        cov = (f"{np.mean([i['coverage'] for i in info]):.2f} / {np.mean([i['purity'] for i in info]):.2f}"
               if info and "coverage" in info[0] else "")
        tab[(ds, m)] = rows
        print(f"| {lab} | {tmean(rows[:, 1]):.1f} | {tmean(rows[:, 2]):.1f} | {tmean(rows[:, 3]):+.1f} | "
              f"{units if units else ''} | {cov} |")
    print()
print("## Paired differences in selectivity (first − second), trimmed mean [95% CI], concepts better / worse\n")
print("| Pair | " + " | ".join(nm for _, (nm, _, _) in data.items()) + " | Pooled |")
print("|---|" + "---|" * (len(data) + 1))
names = dict(METHODS)
for pr in args.pairs.split(","):
    a, b = pr.split(":")
    cells, pooled = [], {}
    for ds in data:
        if (ds, a) in tab and (ds, b) in tab and len(tab[(ds, a)]) == len(tab[(ds, b)]):
            dd = tab[(ds, a)][:, 3] - tab[(ds, b)][:, 3]
            lo, hi = ci(dd)
            pooled[ds] = dd
            cells.append(f"{tmean(dd):+.1f} [{lo:+.1f}, {hi:+.1f}] {int((dd > 0).sum())}/{int((dd < 0).sum())}")
        else:
            cells.append("")
    if pooled:
        allx = np.concatenate(list(pooled.values()))
        rng = np.random.default_rng(0)
        bs = tmean(np.concatenate([v[rng.integers(0, len(v), (B, len(v)))] for v in pooled.values()], axis=1))
        lo, hi = np.quantile(bs, [.025, .975])
        cells.append(f"{tmean(allx):+.1f} [{lo:+.1f}, {hi:+.1f}] {int((allx > 0).sum())}/{int((allx < 0).sum())}")
    print(f"| {names[a]} − {names[b]} | " + " | ".join(cells) + " |")

if args.tex:
    cols = [m for m in ("km", "kmt", "sae_km", "nobypass", "geoae", "sae_f1", "sae_f5") if any((ds, m) in tab for ds in data)]
    print("\n```latex")
    print("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule")
    print("Dataset & " + " & ".join(names[m] for m in cols) + " \\\\\n\\midrule")
    for ds, (nm, _, _) in data.items():
        vals = {m: tmean(tab[(ds, m)][:, 3]) for m in cols if (ds, m) in tab}
        best = max(vals.values())
        print(nm + " & " + " & ".join((f"\\textbf{{{vals[m]:.1f}}}" if vals[m] == best else f"{vals[m]:.1f}") if m in vals else "--"
                                       for m in cols) + " \\\\")
    print("\\bottomrule\n\\end{tabular}\n```")
