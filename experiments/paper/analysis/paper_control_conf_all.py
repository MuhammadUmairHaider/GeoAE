"""Control tables in CONFIDENCE units across all control datasets, base (residual) vs GeoAE latent; default α 0.5 (+ range removal).

Metric definitions are copied from experiments/paper/analysis/paper_control_conf_tables.py (same defaults): per concept, edit and space,
against that space's own unedited run, using per-document gold-label confidences (rec["ref"], doc_conf):
  target removed  = -(mean edited - mean unedited conf on the concept's own docs), as % of the unedited conf (--scale rel)
  collateral loss = mean over other-class docs of max(0, unedited - edited), as % of their unedited conf
  selectivity     = target removed - collateral loss                                   (higher = better)
Concept averages are trimmed means (floor(trim * n) cut per side, default 10%); Δ = trimmed mean of per-concept paired
GeoAE - base selectivity; 95% CI = bootstrap over concepts, stratified by dataset in pooled rows (10k, seed 0).

Result files (first match that is finished and carries per-doc confidences):
  results/delta/<arm>/interventions/range_intervention_<ds>_dprime_conf.json, then ..._dprime.json
  FineWeb-Atlas: .../interventions/atlas_classes/range_intervention_atlas_s{0,1,2}_dprime_conf.json, then ..._dprime.json
Jigsaw (binary) is reported for its meaningful direction only (removing "toxic"), never pooled and never bolded.
Pooled rows: Curated = DBpedia-14, Bias in Bios, AG News, Emotion, 20 Newsgroups, TREC; Curated + Atlas.

Usage: python experiments/paper/analysis/paper_control_conf_all.py [--alphas 0.5] [--trim 0.10] [--scale rel|abs] [--collateral loss|signed]
"""

from geoae.paths import resolve_input_path
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--alphas", default="0.5")
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--scale", choices=["rel", "abs"], default="rel")
ap.add_argument("--collateral", choices=["loss", "signed"], default="loss")
args = ap.parse_args()
rng = np.random.default_rng(0)
B = 10000
ROOT = Path("results/delta")


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def metrics(c, s, op):
    """(unedited target conf, target change, collateral loss, selectivity) in points (same as paper_control_conf_tables)."""
    ref, rec = c["ref"][s], c[f"{s}_{op}"]
    nt = len(c["eval_idx"]["tgt"])
    r, e = np.array(ref["doc_conf"]), np.array(rec["doc_conf"])
    t_before, t_after = r[:nt].mean(), e[:nt].mean()
    coll = np.maximum(0, r[nt:] - e[nt:]).mean() if args.collateral == "loss" else (r[nt:] - e[nt:]).mean()
    tchg = t_after - t_before
    if args.scale == "rel":
        tchg, coll = tchg / t_before, coll / r[nt:].mean()
    return 100 * t_before, 100 * tchg, 100 * coll, 100 * (-tchg - coll)


def _usable(p):
    if not resolve_input_path(p).exists():
        return None
    d = json.loads(resolve_input_path(p).read_text())
    if "summary" not in d or not d["concepts"] or "ref" not in next(iter(d["concepts"].values())):
        return None
    return d


def load(arm, ds):
    """Concept records with per-doc confidences; Atlas draws pooled (keys kept apart). Returns (records, source files)."""
    if ds == "atlas":
        cands = [[ROOT / arm / "interventions" / "atlas_classes" / f"range_intervention_atlas_s{s}_dprime{t}.json"
                  for t in ("_conf", "")] for s in (0, 1, 2)]
    else:
        cands = [[ROOT / arm / "interventions" / f"range_intervention_{ds}_dprime{t}.json" for t in ("_conf", "")]]
    recs, used = [], []
    for group in cands:
        for p in group:
            d = _usable(p)
            if d is not None:
                cs = d["concepts"]
                if ds == "jigsaw2":
                    toxic = str(d["meta"].get("classes", ["clean", "toxic"]).index("toxic"))
                    cs = {toxic: cs[toxic]} if toxic in cs else {}
                recs += list(cs.values()); used.append(p.name)
                break
    return recs, used


ALS = [f"{float(a):.1f}" for a in args.alphas.split(",")]
OPS = [("rm_range_comp", "Range removal", "–")] + [(f"st_{m}_a{a}", name, a.rstrip("0").rstrip(".") if a.endswith(".0") else a)
                                                   for m, name in (("range", "Range steering"), ("global", "Plain steering")) for a in ALS]
DATASETS = [("db14", "DBpedia-14"), ("biasbios", "Bias in Bios"), ("ag_news", "AG News"), ("emotion6", "Emotion"),
            ("news20", "20 Newsgroups"), ("trec6", "TREC question type"), ("atlas", "FineWeb-Atlas (3 random draws)"),
            ("jigsaw2", "Jigsaw: remove toxic")]
CURATED = ["db14", "biasbios", "ag_news", "emotion6", "news20", "trec6"]
f = lambda v, fmt=".1f": format(v, fmt).replace("-", "−")
print(f"Confidence units; collateral = {args.collateral}; scale = {args.scale}; {args.trim:.0%}-trimmed means over concepts; α {', '.join(ALS)}")
for arm, model in [("d6144", "Llama-3.2-3B, layer 27"), ("g4b_l33_d5120", "Gemma 3 4B, layer 33")]:
    data = {ds: load(arm, ds) for ds, _ in DATASETS}
    print(f"\n### {model}\n")
    print("| Dataset (concepts) | Method | Target confidence removed | Collateral loss | Selectivity | Δ GeoAE − base [95% CI] | GeoAE better / worse |")
    print("|---|---|---|---|---|---|---|")
    groups = [([ds], name) for ds, name in DATASETS] + [(CURATED, "Curated"), (CURATED + ["atlas"], "Curated + Atlas")]
    for group, gname in groups:
        strata = {g: data[g][0] for g in group if data[g][0]}
        pooled = len(group) > 1
        if not strata:
            print(f"| {gname} | pending | | | | | |")
            continue
        n = sum(len(v) for v in strata.values())
        nm = f"**{gname} ({n})**" if pooled else f"{gname} ({n})"
        if pooled and len(strata) < len(group):
            nm += " (missing: " + ", ".join(dict(DATASETS)[g] for g in group if g not in strata) + ")"
        if group == ["atlas"] and len(data["atlas"][1]) < 3:
            nm += f" ({len(data['atlas'][1])} of 3 draws)"
        for i, (op, name, al) in enumerate(OPS):
            M = {g: {s: np.array([metrics(c, s, op) for c in strata[g]]) for s in ("h", "z")} for g in strata}
            H = np.concatenate([M[g]["h"] for g in strata]); Z = np.concatenate([M[g]["z"] for g in strata])
            dif = {g: M[g]["z"][:, 3] - M[g]["h"][:, 3] for g in strata}
            allx = np.concatenate(list(dif.values()))
            boots = tmean(np.concatenate([d[rng.integers(0, len(d), (B, len(d)))] for d in dif.values()], axis=1))
            lo, hi = np.quantile(boots, [.025, .975])
            sig = "**" if (lo > 0 or hi < 0) and n >= 14 and group != ["jigsaw2"] else ""
            delta = f"{sig}{f(tmean(allx), '+.1f')} [{f(lo, '+.1f')}, {f(hi, '+.1f')}]{sig}"
            wl = f"{(allx > 1e-9).sum()} / {(allx < -1e-9).sum()}"
            arrow = lambda j, sgn=1: f"{f(sgn * tmean(H[:, j]))} → {f(sgn * tmean(Z[:, j]))}"
            meth = name + ("" if al == "–" else f", α {al}")
            print(f"| {nm if i == 0 else ''} | {meth} | {arrow(1, -1)} | {arrow(2)} | {arrow(3)} | {delta} | {wl} |")
    print("\nSources: " + "; ".join(f"{ds}: {', '.join(data[ds][1]) or 'none'}" for ds, _ in DATASETS))
