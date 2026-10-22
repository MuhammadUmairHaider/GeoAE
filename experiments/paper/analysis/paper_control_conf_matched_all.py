"""Equal-damage comparison in CONFIDENCE units over all control datasets, base (residual) vs GeoAE latent.

Same method as experiments/paper/analysis/paper_control_conf_matched.py (curve, y_at_x, compare are copied unchanged): per concept, space
and operator (range / plain steering) the alpha grid 0, 0.5, 1, 2 traces (collateral loss %, target removed %), both
relative to the space's own unedited confidence; collateral is read at fixed removal and removal at fixed collateral by
linear interpolation (first crossing; concepts that never reach the level are dropped and counted); Δ = GeoAE - base per
concept, 10%-trimmed mean, 95% bootstrap CI over concepts stratified by dataset (10k, seed 0).
Datasets and file lookup as experiments/paper/analysis/paper_control_conf_all.py. Pools: Curated = DBpedia-14, Bias in Bios, AG News, Emotion,
20 Newsgroups, TREC; Curated + Atlas adds the three FineWeb-Atlas draws. Jigsaw (binary) is not pooled.
Usage: python experiments/paper/analysis/paper_control_conf_matched_all.py [--trim 0.1] [--rem_levels 30,50,70] [--coll_levels 5,10,20]
"""

from geoae.paths import resolve_input_path
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--rem_levels", default="30,50,70")
ap.add_argument("--coll_levels", default="5,10,20")
args = ap.parse_args()
REM = [float(x) for x in args.rem_levels.split(",")]
COLL = [float(x) for x in args.coll_levels.split(",")]
ALPHAS = ("0.5", "1.0", "2.0")
rng = np.random.default_rng(0)
B = 10000
ROOT = Path("results/delta")
CURATED = ["db14", "biasbios", "ag_news", "emotion6", "news20", "trec6"]
NAMES = {"db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News", "emotion6": "Emotion",
         "news20": "20 Newsgroups", "trec6": "TREC", "atlas": "FineWeb-Atlas"}


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def _usable(p):
    if not resolve_input_path(p).exists():
        return None
    d = json.loads(resolve_input_path(p).read_text())
    if "summary" not in d or not d["concepts"] or "ref" not in next(iter(d["concepts"].values())):
        return None
    return d


def load(arm, ds):
    if ds == "atlas":
        cands = [[ROOT / arm / "interventions" / "atlas_classes" / f"range_intervention_atlas_s{s}_dprime{t}.json"
                  for t in ("_conf", "")] for s in (0, 1, 2)]
    else:
        cands = [[ROOT / arm / "interventions" / f"range_intervention_{ds}_dprime{t}.json" for t in ("_conf", "")]]
    recs = []
    for group in cands:
        for p in group:
            d = _usable(p)
            if d is not None:
                recs += list(d["concepts"].values())
                break
    return recs


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
for arm, model in [("d6144", "Llama-3.2-3B, layer 27"), ("g4b_l33_d5120", "Gemma 3 4B, layer 33")]:
    data = {ds: load(arm, ds) for ds in CURATED + ["atlas"]}
    for pool_name, pool in (("Curated", CURATED), ("Curated + Atlas", CURATED + ["atlas"])):
        strata = {g: data[g] for g in pool if data[g]}
        n = sum(len(C) for C in strata.values())
        missing = [NAMES[g] for g in pool if not data[g]]
        print(f"\n### {model} — {pool_name} ({n} concepts{'; missing: ' + ', '.join(missing) if missing else ''})\n")
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
