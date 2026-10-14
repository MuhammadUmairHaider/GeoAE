"""Removal edits compared at matched deterioration, residual (h) vs latent (z), and transport vs range.

Per concept and operator, the edit traces a curve through (0,0) and alpha = 0.5, 1, 2 in
(collateral, removal) space, collateral = accuracy drop on the other classes. We read removal at a
fixed collateral C and collateral at a fixed removal E by linear interpolation along that curve,
then average over concepts (concepts that never reach the level are dropped and counted), with a
paired bootstrap over concepts for z - h and transport - range."""
import json
import sys
from pathlib import Path

import numpy as np

OPS = ("range", "transport", "global", "salient")
ALPHAS = ("0.5", "1.0", "2.0")
rng = np.random.default_rng(0)


def curve(c, space, op, y="tgt_drop", x="comp_drop"):
    pts = [(0.0, 0.0)] + [(c[f"{space}_st_{op}_a{a}"][x], c[f"{space}_st_{op}_a{a}"][y]) for a in ALPHAS if f"{space}_st_{op}_a{a}" in c]
    return pts


def y_at_x(pts, x0):
    """Interpolate y where the curve first reaches x0 (x need not be monotone: use first crossing)."""
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if xa <= x0 <= xb and xb > xa:
            return ya + (yb - ya) * (x0 - xa) / (xb - xa)
        if xa == x0:
            return ya
    return np.nan


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) < 3:
        return np.nan, np.nan, np.nan, len(v)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    d = np.asarray(boots) - np.mean(boots)
    return v.mean(), v.mean() + np.quantile(d, .025), v.mean() + np.quantile(d, .975), len(v)


def fmt(t, pct=True):
    m, lo, hi, n = t
    if not np.isfinite(m):
        return f"   n/a (n={n})"
    k = 100 if pct else 1
    return f"{m * k:+6.1f} [{lo * k:+.1f},{hi * k:+.1f}] n={n}"


def analyse(path, label):
    d = json.loads(Path(path).read_text())
    al = [f"{a:.1f}" for a in d["meta"]["alphas"]]
    C = [c for c in d["concepts"].values() if all(f"{sp}_st_{op}_a{a}" in c for sp in "hz" for op in OPS for a in al)]
    print(f"  strengths {al}")
    base_ppl = d["meta"]["base_ppl"]
    print(f"\n######## {label}  ({len(C)} concepts)")
    # 1. removal at matched collateral
    for x0 in (0.10, 0.20, 0.30):
        print(f"  removal at collateral {x0:.0%}:")
        vals = {}
        for op in OPS:
            h = [y_at_x(curve(c, "h", op), x0) for c in C]
            z = [y_at_x(curve(c, "z", op), x0) for c in C]
            vals[op] = (h, z)
            mh, mz = np.nanmean(h), np.nanmean(z)
            print(f"    {op:9s} h {mh:6.1%}  z {mz:6.1%}   z-h {fmt(ci(np.subtract(z, h)))}")
        for sp, i in (("h", 0), ("z", 1)):
            print(f"    transport-range ({sp}) {fmt(ci(np.subtract(vals['transport'][i], vals['range'][i])))}")
    # 2. collateral at matched removal (inverse curve)
    for e0 in (0.50, 0.70, 0.85):
        print(f"  collateral at removal {e0:.0%} (lower is better):")
        vals = {}
        for op in OPS:
            h = [y_at_x([(t, cc) for cc, t in curve(c, "h", op)], e0) for c in C]
            z = [y_at_x([(t, cc) for cc, t in curve(c, "z", op)], e0) for c in C]
            vals[op] = (h, z)
            print(f"    {op:9s} h {np.nanmean(h):6.1%}  z {np.nanmean(z):6.1%}   z-h {fmt(ci(np.subtract(z, h)))}")
        for sp, i in (("h", 0), ("z", 1)):
            print(f"    transport-range ({sp}) {fmt(ci(np.subtract(vals['transport'][i], vals['range'][i])))}")
    # 3. removal at matched perplexity rise (relative to each space's own unedited baseline)
    for r0 in (0.02, 0.05):
        print(f"  removal at perplexity +{r0:.0%}:")
        for op in OPS:
            h = [y_at_x(curve(c, "h", op, x="ppl_rise"), r0 * base_ppl) for c in C]
            z = [y_at_x(curve(c, "z", op, x="ppl_rise"), r0 * base_ppl) for c in C]
            print(f"    {op:9s} h {np.nanmean(h):6.1%}  z {np.nanmean(z):6.1%}   z-h {fmt(ci(np.subtract(z, h)))}")


for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    if Path(path).exists():
        analyse(path, label)
    else:
        print(f"\n######## {label}: missing {path}")
