"""Plain-steering control table for the paper: three models, true-positive documents, accuracy points, plain means.

One row per (model, dataset); the three FineWeb-Atlas draws are pooled into one row. Spaces: base residual (h), GeoAE latent
(z, its reconstruction spliced in) and, on Gemma only, the Gemma Scope 16k codes (s, all features) and their single top
feature (s1). Documents are true positives of every unedited model that the row compares:
  Llama / Qwen  base model and AE splice        scripts/argo/control_tp2_plain.sbatch   -> <arm>/interventions/tp2/*_hz_tp2.json
  Gemma         base, AE splice and SAE splice  TP3=1 scripts/argo/control_three_way.sbatch -> <arm>/interventions/sae/*_hzs_tp3.json
                                                (+ ..._s1_tp3.json for the top-1 arm, merged after checking the documents match)
Metric (the definitions of eval_out/paper_control_three_way.py --metric acc --trim 0), per concept and space, against the
space's OWN unedited run:
  Target drop  = unedited - edited accuracy on the concept's documents            (higher = more of the concept removed)
  Collateral   = unedited - edited accuracy on the other-class documents          (lower = less damage)
  selectivity  = Target drop - Collateral; Delta Sel. = selectivity of the space - selectivity of the base residual
Both are printed as positive DROPS, so the header arrows read naturally (Target drop up, Collateral down). --signed prints
the earlier layout instead: signed changes (Delta Target and Collateral both negative) and no arrows on those two columns,
because 'higher is better' for a negative collateral reads as 'more collateral is better'.
Rows are plain means over concepts; where several edited spaces are compared (Gemma) the most selective one is bold.
A dataset may be listed with document redraws (news20,news20_s1,...: eval_out/control_datasets_redraw.py; db14_s1,...:
scripts/argo/control_new_draws.sh, the Hub split shuffled with another seed). They hold the
same classes, so each class is first averaged over the draws that scored it and counts as ONE concept; FineWeb-Atlas draws
hold different classes and are pooled as before.
Statistics (stderr and --stats): the mean per-concept difference with a 95% bootstrap CI
over concepts (10k resamples, seed 0; stratified by dataset in the pooled rows) and the number of concepts better / worse.

    python eval_out/paper_control_plain_table.py > results/delta/control_plain_tp_a05.tex
    python eval_out/paper_control_plain_table.py --stats results/delta/control_plain_tp_a05.md > /dev/null
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", default="0.5")
ap.add_argument("--datasets", default="db14,biasbios,ag_news,emotion6,news20,atlas_s0,atlas_s1,atlas_s2")
ap.add_argument("--stats", default="", help="also write the statistics as a Markdown file")
ap.add_argument("--allow_partial", action="store_true", help="print rows of runs that have not finished (marked)")
ap.add_argument("--json", default="", help="also write the per-row statistics (mean, 95% interval, concept counts) as JSON; read by eval_out/paper_figures.py")
ap.add_argument("--signed", action="store_true", help="signed changes (negative = accuracy lost) instead of positive drops")
ap.add_argument("--appendix", default="", help="also write the appendix table (documents, starting confidence, intervals) as LaTeX rows")
args = ap.parse_args()
ROOT = Path("results/delta")
KEY = f"st_global_a{float(args.alpha):.1f}"
NAMES = {"db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News", "emotion6": "Emotion", "news20": "20NG"}
# (label, arm, file pattern, spaces in column order, extra top-k file pattern or None)
# GEOAE_NE100=1: the 12 Oct no-balancing / 100-epoch models (arm folders <arm>_ne100, same file names)
import os
SFX = os.environ.get("GEOAE_SFX") or ("_ne100" if os.environ.get("GEOAE_NE100") == "1" else "")     # GEOAE_SFX=_ns100: the no-staging models
MODELS = [
    ("Llama-3.2-3B", "d6144" + SFX, "tp2/range_intervention_{ds}_dprime_hz_tp2.json", ["h", "z"], None),
    ("Gemma 3 4B", "g4b_l33_d5120" + SFX, "sae/range_intervention_{ds}_dprime_hzs_tp3.json", ["h", "s", "s1", "z"],
     "sae/range_intervention_{ds}_dprime_s1_tp3.json"),
    ("Qwen2.5-3B", "qwen25_l35_d4096" + SFX, "tp2/range_intervention_{ds}_dprime_hz_tp2.json", ["h", "z"], None),
]
COLS = ["h", "s", "s1", "z"]
HEAD = {"h": "Base", "s": "SAE", "s1": "SAE$_{1}$", "z": "GeoAE"}
B = 10000


def metrics(c, s):
    """(Delta Target, Collateral, selectivity) in accuracy points for one concept and space."""
    ref, rec = c["ref"][s], c[f"{s}_{KEY}"]
    nt = c.get("nt_s1", len(c["eval_idx"]["tgt"])) if s == "s1" else len(c["eval_idx"]["tgt"])
    r, e = np.array(ref["doc_correct"], float), np.array(rec["doc_correct"], float)
    dt, co = e[:nt].mean() - r[:nt].mean(), e[nt:].mean() - r[nt:].mean()
    return 100 * dt, 100 * co, 100 * (-dt + co)


def ci(d, strata):
    d, strata = np.asarray(d, float), np.asarray(strata)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    parts = [d[strata == g][rng.integers(0, (strata == g).sum(), (B, (strata == g).sum()))] for g in np.unique(strata)]
    return tuple(np.quantile(np.concatenate(parts, axis=1).mean(axis=1), [.025, .975]))


TOP1_NOTE = {}


def load(arm, pat, extra, ds):
    p = ROOT / arm / "interventions" / pat.format(ds=ds)
    q = Path(str(p).replace("_dprime_hzs_tp3", "_dprime_hzss1_tp3"))          # one run with all four spaces
    if SFX and extra and q.exists() and "summary" in json.loads(q.read_text()):
        return json.loads(q.read_text()), ""
    if not p.exists():
        return None, f"missing {p}"
    D = json.loads(p.read_text())
    done = "summary" in D
    # 12 Oct runs: a single run with all four spaces (SUBS=h,z,s,s1 -> ..._hzss1_...) replaces the pair of files where
    # the separate top-1 run had drawn different documents; it carries s1 itself, so nothing is merged.
    if extra and not (ROOT / arm / "interventions" / extra.format(ds=ds)).exists():
        return None, f"missing {ROOT / arm / 'interventions' / extra.format(ds=ds)}"
    if extra:
        X = json.loads((ROOT / arm / "interventions" / extra.format(ds=ds)).read_text())
        for k, c in D["concepts"].items():
            x = X["concepts"][k]
            if SFX and x["eval_idx"] != c["eval_idx"]:
                # 12 Oct runs: every run re-checks which pool documents are true positives, and a borderline document can
                # flip between two runs (seen: 79 vs 80 documents in one Atlas class). The top-1 arm is then scored on its
                # own, almost identical documents; the overlap is checked and reported instead of asserted.
                ov = min(len(set(x["eval_idx"][t]) & set(c["eval_idx"][t])) / max(len(c["eval_idx"][t]), 1) for t in ("tgt", "comp"))
                if ov < 0.9:      # other-class documents are re-drawn when the pool differs: wait for the single four-space run
                    return None, f"{arm} {ds}: top-1 run on other documents ({ov:.0%} shared); needs the hzss1 run"
                TOP1_NOTE.setdefault((arm, ds), []).append(ov)
                c["nt_s1"] = len(x["eval_idx"]["tgt"])
            else:
                assert x["eval_idx"] == c["eval_idx"], f"{arm} {ds} concept {k}: the top-1 run scored different documents"
                assert x["ref"]["s1"]["doc_correct"] == c["ref"]["s"]["doc_correct"], f"{arm} {ds} concept {k}: SAE references differ"
            c["ref"]["s1"] = x["ref"]["s1"]
            c.update({kk: v for kk, v in x.items() if kk.startswith("s1_")})
    return D, ("" if done else "RUNNING")


n1 = lambda v: "--" if v is None else "$" + format(0.0 if abs(v) < 0.05 else v, ".1f") + "$"
def ns(v, bold=False):
    if v is None:
        return "--"
    t = format(0.0 if abs(v) < 0.05 else v, "+.1f")
    return "$\\mathbf{" + t + "}$" if bold else "$" + t + "$"


rows_tex, stats, notes, app = [], [], [], []
for label, arm, pat, spaces, extra in MODELS:
    C, G, meta, REP = [], [], {}, {}
    for ds in args.datasets.split(","):
        D, state = load(arm, pat, extra, ds)
        if D is None or (state and not args.allow_partial):
            notes.append(f"{label} {ds}: {state}")
            continue
        g = "FineWeb-Atlas" if ds.startswith("atlas_s") else NAMES[re.sub(r"_s\d+$", "", ds)]
        m = D["meta"]
        scored = list(D["concepts"])
        meta.setdefault(g, []).append((ds, state, len(scored), sum(m["n_eval_by_class"][k] for k in scored), m))
        for k, c in D["concepts"].items():
            if all(f"{s}_{KEY}" in c for s in spaces):
                key = (ds, k) if ds.startswith("atlas_s") else (g, k)       # document redraws: one concept per class
                if key not in REP:
                    REP[key] = []; C.append(REP[key]); G.append(g)
                REP[key].append(c)
    if not C:
        continue
    G = np.array(G)
    M = {s: np.array([np.mean([metrics(c, s) for c in rep], axis=0) for rep in C]) for s in spaces}
    rows_tex.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
    groups = list(dict.fromkeys(G)) + ["All concepts"]
    for g in groups:
        idx = np.arange(len(C)) if g == "All concepts" else np.where(G == g)[0]
        dsel = {s: (M[s][idx, 2] - M["h"][idx, 2]) for s in spaces if s != "h"}
        best = max(dsel, key=lambda s: dsel[s].mean())
        sgn = 1.0 if args.signed else -1.0          # M holds signed changes; the default table shows positive drops
        cell = lambda s, j: n1(sgn * M[s][idx, j].mean() if s in spaces else None)
        name = ("\\quad " + g) if g != "All concepts" else f"\\quad \\emph{{All ({len(idx)} concepts)}}"
        rows_tex.append(name + " & " + " & ".join(cell(s, 0) for s in COLS) + " & " + " & ".join(cell(s, 1) for s in COLS) + " & "
                        + " & ".join(ns(dsel[s].mean() if s in dsel else None, bold=(len(dsel) > 1 and s == best and dsel[s].mean() > 0)) for s in COLS[1:]) + " \\\\")
        docs = (sum(nd for lst in meta.values() for _, _, _, nd, _ in lst) if g == "All concepts"
                else sum(nd for _, _, _, nd, _ in meta[g]))
        conf = {s: 100 * float(np.mean([np.mean([c["ref"]["s" if s == "s1" else s]["tgt_conf"] for c in C[i]]) for i in idx])) for s in spaces if s != "s1"}
        lo_z, hi_z = ci(dsel["z"], G[idx])
        app.append((label, g, len(idx), docs, conf, dsel["z"].mean(), lo_z, hi_z, int((dsel["z"] > 1e-9).sum()), int((dsel["z"] < -1e-9).sum())))
        for s, d in dsel.items():
            lo, hi = ci(d, G[idx])
            stats.append((label, g, len(idx), HEAD[s].replace("$_{1}$", " top-1") + " − Base", d.mean(), lo, hi, int((d > 1e-9).sum()), int((d < -1e-9).sum())))
        if "s" in spaces:
            d = M["z"][idx, 2] - M["s"][idx, 2]
            lo, hi = ci(d, G[idx])
            stats.append((label, g, len(idx), "GeoAE − SAE", d.mean(), lo, hi, int((d > 1e-9).sum()), int((d < -1e-9).sum())))
    rows_tex.append("\\addlinespace[2pt]")
    for g, lst in meta.items():
        for ds, state, nc, nd, m in lst:
            notes.append(f"{label} | {g} ({ds}){' RUNNING' if state else ''} | concepts {nc} | evaluation documents {nd} | unedited accuracy "
                         f"base {100 * m['base_acc']:.1f} / AE {100 * m['zrecon_acc']:.1f}"
                         + (f" / SAE {100 * m['srecon_acc']:.1f}" if "srecon_acc" in m else "")
                         + f" | unedited confidence base {100 * m['base_conf']:.1f} / AE {100 * m['zrecon_conf']:.1f}"
                         + (f" / SAE {100 * m['srecon_conf']:.1f}" if "srecon_conf" in m else "")
                         + f" | perplexity base {m['base_ppl']:.2f} / AE {m['zrecon_ppl']:.2f}" + (f" / SAE {m['srecon_ppl']:.2f}" if "srecon_ppl" in m else ""))

if rows_tex and rows_tex[-1].startswith("\\addlinespace"):
    rows_tex.pop()
print("% Generated by eval_out/paper_control_plain_table.py (plain steering, alpha " + args.alpha + ", accuracy points, plain means over concepts).")
print("\\begin{tabular}{@{}l rrrr rrrr rrr@{}}\n\\toprule")
H1, H2 = (("$\\Delta$ Target", "Collateral") if args.signed else ("Target drop $\\uparrow$", "Collateral $\\downarrow$"))
print(f" & \\multicolumn{{4}}{{c}}{{{H1}}} & \\multicolumn{{4}}{{c}}{{{H2}}} & \\multicolumn{{3}}{{c}}{{$\\Delta$ Sel.\\ vs.\\ Base $\\uparrow$}} \\\\")
print("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(l){10-12}")
print("Dataset & " + " & ".join([HEAD[s] for s in COLS] * 2 + [HEAD[s] for s in COLS[1:]]) + " \\\\\n\\midrule")
print("\n".join(rows_tex))
print("\\bottomrule\n\\end{tabular}")

md = ["# Plain steering on true-positive documents, alpha " + args.alpha + " (accuracy points, plain means over concepts)", "",
      "| Model | Rows | Concepts | Selectivity difference | Mean [95% CI] | Concepts better / worse |", "|---|---|---|---|---|---|"]
for label, g, n, name, mean, lo, hi, w, l in stats:
    md.append(f"| {label} | {g} | {n} | {name} | {mean:+.1f} [{lo:+.1f}, {hi:+.1f}] | {w} / {l} |")
md += ["", "## Runs", ""] + [f"- {x}" for x in notes]
print("\n".join(md), file=sys.stderr)
if args.stats:
    Path(args.stats).write_text("\n".join(md) + "\n")
if args.json:
    Path(args.json).write_text(json.dumps({"alpha": args.alpha, "unit": "accuracy points", "rows": [
        {"model": label, "rows": g, "concepts": n, "difference": name, "mean": float(mean), "lo": float(lo), "hi": float(hi),
         "better": w, "worse": l} for label, g, n, name, mean, lo, hi, w, l in stats]}, indent=1) + "\n")
if args.appendix:
    out, last = [], None
    for label, g, n, docs, conf, mean, lo, hi, w, l in app:
        if label != last:
            if last is not None:
                out.append("\\addlinespace[2pt]")
            out.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
            last = label
        name = "\\quad " + (g if g != "All concepts" else "\\emph{All}")
        cs = " & ".join(f"${conf[s]:.0f}$" if s in conf else "--" for s in ("h", "s", "z"))
        sig = lo > 0 or hi < 0
        d = f"{mean:+.1f}"
        nd = f"{docs:,}".replace(",", "{,}")
        out.append(f"{name} & ${n}$ & ${nd}$ & {cs} & " + ("$\\mathbf{" + d + "}$" if sig else "$" + d + "$")
                   + f" & $[{lo:+.1f}, {hi:+.1f}]$ & ${w}$ / ${l}$ \\\\")
    Path(args.appendix).write_text("\n".join(out) + "\n")
