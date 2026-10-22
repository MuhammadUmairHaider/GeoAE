"""Paper LaTeX tables for the control experiments in confidence units (one table per method), straight from the result JSONs.

Columns: Dataset | Delta Target (Base, AE) | Collateral (Base, AE) | Delta Sel. (AE - Base). Rows grouped by model.
Metric code identical to experiments/paper/analysis/paper_control_conf_tables.py / paper_control_conf_all.py (defaults: --scale rel,
--collateral loss): per concept, against each space's own unedited run, from per-document gold-label confidences —
  Delta Target = change in mean confidence on the concept's own docs, % of the unedited confidence (negative = removed)
  Collateral   = mean over other-class docs of max(0, unedited - edited), % of their unedited confidence
  Delta Sel.   = per-concept (AE selectivity - Base selectivity), selectivity = -Delta Target - Collateral
Column means are 10%-trimmed over concepts; Delta Sel. is the trimmed mean of per-concept differences; bold = 95% bootstrap
CI over concepts excludes 0 (stratified by dataset in the pooled row; 10k, seed 0), only for rows with >= 14 concepts.
Files: results/delta/<arm>/interventions/range_intervention_<ds>_dprime_conf.json, else ..._dprime.json (must carry rec["ref"]).

Usage: python experiments/paper/analysis/paper_control_conf_latex.py [--datasets db14,biasbios,ag_news,emotion6,news20] [--alpha 0.5]
         > results/delta/control_conf_latex_a05.tex
"""

from geoae.paths import resolve_input_path
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", default="db14,biasbios,ag_news,emotion6,news20")
ap.add_argument("--alpha", default="0.5")
ap.add_argument("--trim", type=float, default=0.10)
args = ap.parse_args()
rng = np.random.default_rng(0)
B = 10000
ROOT = Path("results/delta")
ARMS = [("d6144", "Llama-3.2-3B"), ("g4b_l33_d5120", "Gemma-3-4B"), ("qwen25_l35_d4096", "Qwen2.5-3B")]
NAMES = {"db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News", "emotion6": "Emotion",
         "news20": "20 Newsgroups", "trec6": "TREC"}
SHORT = {"news20": "20NG"}          # row labels that must stay narrow to fit one ACL column
DS = args.datasets.split(",")
A = f"{float(args.alpha):.1f}"


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def metrics(c, s, op):
    """(target change %, collateral %, selectivity) relative to the space's own unedited confidence."""
    ref, rec = c["ref"][s], c[f"{s}_{op}"]
    nt = len(c["eval_idx"]["tgt"])
    r, e = np.array(ref["doc_conf"]), np.array(rec["doc_conf"])
    tchg = (e[:nt].mean() - r[:nt].mean()) / r[:nt].mean()
    coll = np.maximum(0, r[nt:] - e[nt:]).mean() / r[nt:].mean()
    return 100 * tchg, 100 * coll, 100 * (-tchg - coll)


def load(arm, ds):
    for t in ("_conf", ""):
        p = ROOT / arm / "interventions" / f"range_intervention_{ds}_dprime{t}.json"
        if resolve_input_path(p).exists():
            d = json.loads(resolve_input_path(p).read_text())
            if "summary" in d and d["concepts"] and "ref" in next(iter(d["concepts"].values())):
                return list(d["concepts"].values()), p.name
    return [], None


def tex(x, sign=False):
    s = f"{x:+.1f}" if sign else f"{x:.1f}"
    return "$-$" + s[1:] if s.startswith("-") else s


sources = []
counts = {}
body = {}
for op_key, op_title, label in (("st_range", "Range steering", "tab:control-range"), ("st_global", "Plain steering", "tab:control-plain")):
    op = f"{op_key}_a{A}"
    rows = []
    for mi, (arm, model) in enumerate(ARMS):
        data = {ds: load(arm, ds) for ds in DS}
        if op_key == "st_range":
            sources.append(f"% {model}: " + "; ".join(f"{ds}={data[ds][1] or 'MISSING'}" for ds in DS))
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{model}}}}} \\")
        for group in [[ds] for ds in DS] + [DS]:
            strata = {g: data[g][0] for g in group if data[g][0]}
            pooled = len(group) > 1
            name = r"\textit{Pooled}" if pooled else SHORT.get(group[0], NAMES[group[0]])
            if not strata or (pooled and len(strata) < len(group)):
                rows.append(f"{name} & \\multicolumn{{5}}{{c}}{{pending}} \\\\")
                continue
            M = {g: {s: np.array([metrics(c, s, op) for c in strata[g]]) for s in ("h", "z")} for g in strata}
            H = np.concatenate([M[g]["h"] for g in strata]); Z = np.concatenate([M[g]["z"] for g in strata])
            dif = {g: M[g]["z"][:, 2] - M[g]["h"][:, 2] for g in strata}
            allx = np.concatenate(list(dif.values()))
            boots = tmean(np.concatenate([d[rng.integers(0, len(d), (B, len(d)))] for d in dif.values()], axis=1))
            lo, hi = np.quantile(boots, [.025, .975])
            n = len(allx)
            counts["pooled" if pooled else NAMES[group[0]]] = n
            d = tex(tmean(allx), True)
            if (lo > 0 or hi < 0) and n >= 14:
                d = rf"\textbf{{{d}}}"
            rows.append(f"{name} & {tex(tmean(H[:, 0]))} & {tex(tmean(Z[:, 0]))} & {tex(tmean(H[:, 1]))} & "
                        f"{tex(tmean(Z[:, 1]))} & {d} \\\\")
        if mi < len(ARMS) - 1:
            rows.append(r"\midrule")
    body[op_key] = (op_title, label, rows)

cnt = ", ".join(f"{NAMES[ds]}: {counts.get(NAMES[ds], '?')}" for ds in DS) + f", pooled: {counts.get('pooled', '?')}"
print("% Generated by experiments/paper/analysis/paper_control_conf_latex.py from per-document confidence records:")
print("\n".join(sources))
print(r"% Requires \usepackage{booktabs}. Fits one ACL column (\columnwidth).")
for op_key in ("st_range", "st_global"):
    title, label, rows = body[op_key]
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{$\Delta$ Target} & \multicolumn{2}{c}{Collateral} & $\Delta$ Sel. \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}")
    print(r"Dataset & Base & AE & Base & AE & AE $-$ Base \\")
    print(r"\midrule")
    print("\n".join(rows))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{" + title + rf" ($\alpha={args.alpha}$): change in the model's confidence in the correct label when a concept "
          r"is edited, base residual stream vs.\ GeoAE latent. $\Delta$ Target: change in confidence on the concept's own "
          r"documents; Collateral: confidence lost on other-class documents; both as \% of each space's unedited confidence. "
          r"$\Delta$ Sel.: difference in selectivity ($-\Delta$ Target $-$ Collateral), AE $-$ Base, in percentage points. "
          + ("20NG: 20 Newsgroups. " if "news20" in DS else "") +
          rf"Positive favours GeoAE. Means over concepts are 10\% trimmed ({cnt}); $\Delta$ Sel.\ is the trimmed mean of "
          r"per-concept differences, so it need not equal the difference of the column means. Bold: 95\% bootstrap CI "
          r"excludes zero (marked only for rows with at least 14 concepts).}")
    print(r"\label{" + label + "}")
    print(r"\end{table}")
    print()
