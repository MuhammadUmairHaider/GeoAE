"""Control tables (range removal / range steering / plain steering), residual (base) vs GeoAE latent, trimmed mean over concepts.

Per dataset and model: removed %, collateral %, selectivity (= removed - collateral) for base and GeoAE, and
Delta = trimmed mean of the PER-CONCEPT paired difference (GeoAE - base) in selectivity, with a 95% bootstrap CI over
concepts (stratified by dataset in pooled rows; 10k resamples, seed 0). Level columns are trimmed separately, so they need
not subtract to Delta. Trim = fraction cut from EACH side (floor(trim * n) concepts; 0.10 default, 0.25 = interquartile mean).

Datasets: DBpedia-14, Bias in Bios, AG News, Emotion, 20 Newsgroups, TREC (question type), Jigsaw (binary, never pooled,
never bolded), FineWeb-Atlas classes (3 random draws, one stratum). Pooled rows: "Curated" = the six labelled multi-class
datasets; "Curated + Atlas". Missing result files are reported as pending.

Usage: python eval_out/paper_control_tables_trimmed.py [--trim 0.10] [--min_eval 0]
  --min_eval N drops concepts with fewer than N eval docs where the file records per-class counts (meta.n_eval_by_class).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("trim_pos", nargs="?", type=float, default=None, help="(legacy positional trim)")
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--min_eval", type=int, default=0)
args = ap.parse_args()
TRIM = args.trim_pos if args.trim_pos is not None else args.trim
rng = np.random.default_rng(0)
B = 10000


def TM(x):
    """Trimmed mean along the last axis (scipy.stats.trim_mean definition: cut floor(TRIM n) from each end)."""
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(TRIM * n)
    return x[..., k:n - k].mean(axis=-1)


ARMS = [("d6144", "Llama-3.2-3B, layer 27"), ("g4b_l33_d5120", "Gemma 3 4B, layer 33")]
DATASETS = [("db14", "DBpedia-14"), ("biasbios", "Bias in Bios"), ("ag_news", "AG News"), ("emotion6", "Emotion"),
            ("news20", "20 Newsgroups"), ("trec6", "TREC question type"), ("jigsaw2", "Jigsaw (binary)"),
            ("atlas", "FineWeb-Atlas (3 draws)")]
CURATED = ["db14", "biasbios", "ag_news", "emotion6", "news20", "trec6"]
OPS = [("rm_range_comp", "Range removal", "–"), ("st_range_a0.5", "Range steering", "0.5"), ("st_range_a1.0", "Range steering", "1"),
       ("st_global_a0.5", "Plain steering", "0.5"), ("st_global_a1.0", "Plain steering", "1")]
ROOT = Path("results/delta")


def _files(arm, ds):
    if ds == "atlas":
        return [ROOT / arm / "interventions" / "atlas_classes" / f"range_intervention_atlas_s{s}_dprime.json" for s in (0, 1, 2)]
    return [ROOT / arm / "interventions" / f"range_intervention_{ds}_dprime{suf}.json" for suf in ("", "_a0p5")]


def load(arm, ds):
    """Concept records (merged across the alpha-0.5 companion file; Atlas draws kept apart), with eval counts when known."""
    out, dropped = {}, 0
    for f in _files(arm, ds):
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if "summary" not in d:          # unfinished run
            continue
        n_eval = d["meta"].get("n_eval_by_class") or {}
        for c, v in d["concepts"].items():
            key = f"{f.stem}:{c}" if ds == "atlas" else c
            if args.min_eval and c in n_eval and int(n_eval[c]) < args.min_eval:
                dropped += 1
                continue
            out.setdefault(key, {}).update(v)
    return list(out.values()), dropped


cmp_rows = []
print(f"Trim {TRIM:.0%} each side; min eval docs per concept: {args.min_eval or 'none'}")
for arm, mname in ARMS:
    data = {ds: load(arm, ds) for ds, _ in DATASETS}
    print(f"\n### {mname}\n")
    print("| Dataset (concepts) | Method | α | Removed: base | Removed: GeoAE | Collateral: base | Collateral: GeoAE "
          "| Selectivity: base | Selectivity: GeoAE | Δ GeoAE − base [95% CI] | GeoAE better / worse |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    groups = [([ds], name) for ds, name in DATASETS] + [(CURATED, "Curated"), (CURATED + ["atlas"], "Curated + Atlas")]
    for group, gname in groups:
        strata = {g: data[g][0] for g in group if data[g][0]}
        n = sum(len(v) for v in strata.values())
        pooled = len(group) > 1
        label = f"**{gname} ({n})**" if pooled else f"{gname} ({n})"
        if pooled and len(strata) < len(group):
            label += " (partial: " + ", ".join(g for g in group if g not in strata) + " pending)"
        if not strata:
            print(f"| {gname} | pending | | | | | | | | | |")
            continue
        for i, (op, name, al) in enumerate(OPS):
            S = {g: [v for v in V if "z_" + op in v and "h_" + op in v] for g, V in strata.items()}
            S = {g: V for g, V in S.items() if V}
            if not S:
                continue
            V = sum(S.values(), [])
            col = lambda s, k: TM(np.array([100 * v[s + op][k] for v in V]))
            dif = {g: np.array([100 * (v["z_" + op]["selectivity"] - v["h_" + op]["selectivity"]) for v in S[g]]) for g in S}
            allx = np.concatenate(list(dif.values()))
            boots = TM(np.concatenate([d[rng.integers(0, len(d), (B, len(d)))] for d in dif.values()], axis=1))
            lo, hi = np.quantile(boots, [.025, .975])
            sig = "**" if (lo > 0 or hi < 0) and len(allx) >= 14 and "jigsaw2" not in group else ""
            est = TM(allx)
            print(f"| {label if i == 0 else ''} | {name} | {al} | {col('h_','tgt_drop'):.1f} | {col('z_','tgt_drop'):.1f} "
                  f"| {col('h_','comp_drop'):.1f} | {col('z_','comp_drop'):.1f} | {col('h_','selectivity'):.1f} | {col('z_','selectivity'):.1f} "
                  f"| {sig}{est:+.1f} [{lo:+.1f}, {hi:+.1f}]{sig} | {(allx > 0).sum()} / {(allx < 0).sum()} |")
            cmp_rows.append((mname.split(",")[0], gname, f"{name} {al}", allx.mean(), est, lo, hi, len(allx)))
    dropped = {ds: data[ds][1] for ds, _ in DATASETS if data[ds][1]}
    if dropped:
        print(f"\nConcepts dropped by --min_eval {args.min_eval}: {dropped}")

print(f"\n### mean vs {TRIM:.0%}-trimmed mean of Δ\n")
print("| Model | Dataset | Method | n | Δ mean | Δ trimmed [95% CI] |")
print("|---|---|---|---|---|---|")
for r in cmp_rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[7]} | {r[3]:+.1f} | {r[4]:+.1f} [{r[5]:+.1f}, {r[6]:+.1f}] |")
