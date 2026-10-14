"""Confidence-based control edits compared at EQUAL target removal (and equal collateral), base vs GeoAE.

Answers "is GeoAE's alpha-0.5 win just a bigger edit?". Per concept, space and operator (range / plain steering), the
alpha grid 0, 0.5, 1, 2 traces a curve of (target confidence removed, collateral loss), both as % of that space's own
unedited confidence (same definitions as paper_control_conf_tables.py: collateral counts per-doc losses only). We read
collateral at fixed removal levels and removal at fixed collateral levels by linear interpolation along that curve
(first crossing; concepts whose curve never reaches the level are dropped and counted), then compare GeoAE - base per
concept: 10%-trimmed mean, 95% bootstrap CI over concepts stratified by dataset (10k, seed 0).
Usage: python eval_out/paper_control_conf_matched.py [--trim 0.1]"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--tag", default="conf", help="result-file tag: conf (alphas 0.5 1 2) or conf_small (0.1-0.5)")
ap.add_argument("--rem_levels", default=None, help="comma list of % removed at which to read collateral")
ap.add_argument("--coll_levels", default=None, help="comma list of % collateral at which to read removal")
args = ap.parse_args()
SMALL = args.tag == "conf_small"
REM = [float(x) for x in (args.rem_levels or ("10,20,30,40" if SMALL else "30,50,70")).split(",")]
COLL = [float(x) for x in (args.coll_levels or ("1,2,5" if SMALL else "5,10,20")).split(",")]
rng = np.random.default_rng(0)
B = 10000
ALPHAS = None  # set after parsing
DSN = {"db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News"}
ALPHAS = ("0.1", "0.2", "0.3", "0.4", "0.5") if SMALL else ("0.5", "1.0", "2.0")


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def load(arm, ds):
    p = Path(f"results/delta/{arm}/interventions/range_intervention_{ds}_dprime_{args.tag}.json")
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return list(d["concepts"].values()) if "summary" in d else []


def curve(c, s, mode):
    """[(collateral %, removed %)] for alpha 0, 0.5, 1, 2 — both relative to the space's own unedited confidence."""
    ref, nt = c["ref"][s], len(c["eval_idx"]["tgt"])
    r = np.array(ref["doc_conf"]); pts = [(0.0, 0.0)]
    for a in ALPHAS:
        e = np.array(c[f"{s}_st_{mode}_a{a}"]["doc_conf"])
        removed = 100 * (r[:nt].mean() - e[:nt].mean()) / r[:nt].mean()
        coll = 100 * np.maximum(0, r[nt:] - e[nt:]).mean() / r[nt:].mean()
        pts.append((coll, removed))
    return pts


def y_at_x(pts, x0):
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if xa <= x0 <= xb and xb > xa:
            return ya + (yb - ya) * (x0 - xa) / (xb - xa)
    return np.nan


def compare(strata, mode, level, read):
    """read='coll': collateral at removal = level; read='rem': removal at collateral = level. Per-concept z - h."""
    dif, hs, zs = {}, [], []
    for g, C in strata.items():
        d = []
        for c in C:
            v = {}
            for s in ("h", "z"):
                pts = curve(c, s, mode)
                v[s] = y_at_x([(y, x) for x, y in pts], level) if read == "coll" else y_at_x(pts, level)
            if np.isfinite(v["h"]) and np.isfinite(v["z"]):
                d.append(v["z"] - v["h"]); hs.append(v["h"]); zs.append(v["z"])
        dif[g] = np.array(d)
    groups = [g for g in dif if len(dif[g])]
    if not groups:
        return None
    allx = np.concatenate([dif[g] for g in groups])
    boots = tmean(np.concatenate([dif[g][rng.integers(0, len(dif[g]), (B, len(dif[g])))] for g in groups], axis=1))
    lo, hi = np.quantile(boots, [.025, .975])
    n_all = sum(len(C) for C in strata.values())
    return tmean(np.array(hs)), tmean(np.array(zs)), tmean(allx), lo, hi, (allx > 1e-9).sum(), (allx < -1e-9).sum(), len(allx), n_all


f = lambda v, fmt=".1f": format(v, fmt).replace("-", "−")
for arm, model in [("d6144", "Llama-3.2-3B, layer 27"), ("g4b_l33_d5120", "Gemma 3 4B, layer 33"),
                   ("qwen25_l35_d4096", "Qwen2.5-3B, layer 35")]:
    strata = {g: load(arm, g) for g in DSN}
    strata = {g: C for g, C in strata.items() if C}
    if not strata:
        print(f"\n### {model}: no *_dprime_{args.tag}.json results yet"); continue
    n = sum(len(C) for C in strata.values())
    print(f"\n### {model} (pooled: {', '.join(DSN[g] for g in strata)}; {n} concepts)\n")
    print("| Method | Compared at | Base | GeoAE | Δ GeoAE − base [95% CI] | GeoAE better / worse | Concepts reaching the level |")
    print("|---|---|---|---|---|---|---|")
    for mode, name in (("range", "Range steering"), ("global", "Plain steering")):
        for read, levels, lab, better in (("coll", REM, "collateral at {:g}% removed", "lower"),
                                          ("rem", COLL, "removed at {:g}% collateral", "higher")):
            for lv in levels:
                r = compare(strata, mode, lv, read)
                if r is None:
                    continue
                h, z, d, lo, hi, w, l, k, nn = r
                good = (d < 0) if better == "lower" else (d > 0)
                sig = "**" if (lo > 0 or hi < 0) else ""
                tag = "" if not sig else (" (GeoAE better)" if good else " (base better)")
                wins, losses = (l, w) if better == "lower" else (w, l)
                print(f"| {name} | {lab.format(lv)} | {f(h)} | {f(z)} | {sig}{f(d, '+.1f')} [{f(lo, '+.1f')}, {f(hi, '+.1f')}]{sig}{tag} "
                      f"| {wins} / {losses} | {k} of {nn} |")
