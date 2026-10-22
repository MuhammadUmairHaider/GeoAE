"""Control tables for THREE spaces edited on the same documents: base residual (h), token-bypass AE latent (z) and SAE
codes (s). Input: range_intervention_compare.py --substrates h,z,s files (experiments/paper/argo/control_three_way.sbatch).

Metric definitions are those of experiments/paper/analysis/paper_control_conf_all.py: per concept, edit and space, against that space's OWN
unedited run (base model for h, the space's reconstruction spliced in for z and s)
  --metric conf  per-document gold-label confidence:
                 target change = mean edited - mean unedited on the concept's own documents; collateral = mean over
                 other-class documents of max(0, unedited - edited); --scale rel (the paper's) expresses both as % of the
                 unedited confidence of those documents, --scale abs in confidence points
  --metric acc   accuracy points: target change = edited - unedited accuracy on the concept's documents; collateral =
                 unedited - edited accuracy on the other-class documents
  selectivity = -target change - collateral                                             (higher = better)
Concept averages are 10%-trimmed means; differences between two spaces are trimmed means of per-concept paired differences
with a 95% bootstrap CI over concepts, stratified by dataset when several are pooled (10k, seed 0).

--tag _tp3 reads the three-way TRUE-POSITIVE runs (TP3=1: every document is correct under the base model, the AE splice
and the SAE splice, so all three spaces start at 100% accuracy on the same documents).

    python experiments/paper/analysis/paper_control_three_way.py --dataset atlas_s0 [--scale rel|abs] [--metric conf|acc] [--tag _tp3]
    python experiments/paper/analysis/paper_control_three_way.py --dataset atlas_s0,atlas_s1,atlas_s2 --tag _tp3 --metric acc
"""

from geoae.paths import resolve_input_path
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="one dataset or a comma list (pooled)")
ap.add_argument("--arm", default="g4b_l33_d5120")
ap.add_argument("--metric", default="conf", choices=["conf", "acc"])
ap.add_argument("--scale", default="rel", choices=["rel", "abs"])
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--tag", default="")
ap.add_argument("--tex", action="store_true", help="LaTeX table in the layout of the paper's SAE control table: one row per "
                                                   "dataset (the Atlas draws pooled into one row), range and plain steering at --alpha")
ap.add_argument("--alpha", default="0.5")
ap.add_argument("--extra", default="", help="SAE top-k feature arms to add, e.g. s1 or s1,s5 (only the k features of largest "
                                            "d-prime are edited); read from ..._dprime_<extra_tag><tag>.json and merged after "
                                            "checking that both runs scored the same documents")
ap.add_argument("--extra_tag", default="s1s5")
ap.add_argument("--range_all", default="", help="file tag of a run with --percent 1.0 (range steering with NO coordinate selection: "
                                                "the range gate on every coordinate), e.g. hzs_alldims; merged as its own block")
args = ap.parse_args()
EXTRA = [x for x in args.extra.split(",") if x]
SP = ([("h", "Base"), ("s", "SAE")] + [(x, f"SAE top-{x[1:]}") for x in EXTRA] + [("z", "AE")])
PAIRS = ([(a, "h") for a, _ in SP[1:]] + [("z", a) for a, _ in SP[1:-1]])
LAB = dict(SP)
OPS = [("st_range", "Range steering"), ("st_rangeall", "Range steering, no coordinate selection"),
       ("st_global", "Plain steering"), ("rm_range_comp", "Range removal")]
f1 = lambda v: format(v, ".1f").replace("-", "−")
fs = lambda v: format(v, "+.1f").replace("-", "−")
B = 10000


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def ci(d, strata):
    """95% CI of the trimmed mean of d, bootstrapping concepts within each dataset."""
    d, strata = np.asarray(d, float), np.asarray(strata)
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    parts = [d[strata == g][rng.integers(0, (strata == g).sum(), (B, (strata == g).sum()))] for g in np.unique(strata)]
    return tuple(np.quantile(tmean(np.concatenate(parts, axis=1)), [.025, .975]))


def metrics(c, s, key):
    """(unedited level on the target docs, target change, collateral, selectivity)."""
    ref, rec = c["ref"][s], c[f"{s}_{key}"]
    nt = len(c["eval_idx"]["tgt"])
    f = "doc_conf" if args.metric == "conf" else "doc_correct"
    r, e = np.array(ref[f], float), np.array(rec[f], float)
    tchg = e[:nt].mean() - r[:nt].mean()
    coll = np.maximum(0, r[nt:] - e[nt:]).mean() if args.metric == "conf" else r[nt:].mean() - e[nt:].mean()
    if args.metric == "conf" and args.scale == "rel":
        tchg, coll = tchg / r[:nt].mean(), coll / r[nt:].mean()
    return 100 * r[:nt].mean(), 100 * tchg, 100 * coll, 100 * (-tchg - coll)


root = Path("results/delta") / args.arm / "interventions" / "sae"
C, strata, metas = [], [], {}
for ds in args.dataset.split(","):
    p = root / f"range_intervention_{ds}_dprime_hzs{args.tag}.json"
    if not resolve_input_path(p).exists():
        print(f"(missing {p})")
        continue
    D = json.loads(resolve_input_path(p).read_text())
    if EXTRA:      # merge the top-k arms of a second run on the same pool; refuse if the documents differ
        px = root / f"range_intervention_{ds}_dprime_{args.extra_tag}{args.tag}.json"
        X = json.loads(resolve_input_path(px).read_text())
        flips, dmax = 0, 0.0
        for k, c in D["concepts"].items():
            x = X["concepts"][k]
            assert x["eval_idx"] == c["eval_idx"], f"{ds} concept {k}: the two runs scored different documents"
            a, b = np.array(c["ref"]["s"]["doc_conf"]), np.array(x["ref"][EXTRA[0]]["doc_conf"])
            flips += int((np.array(c["ref"]["s"]["doc_correct"]) != np.array(x["ref"][EXTRA[0]]["doc_correct"])).sum())
            dmax = max(dmax, float(np.abs(a - b).max()))
            for arm in EXTRA:
                c["ref"][arm] = x["ref"][arm]
                c.update({kk: v for kk, v in x.items() if kk.startswith(arm + "_")})
        assert X["meta"]["n_eval_by_class"] == D["meta"]["n_eval_by_class"], f"{ds}: evaluation sets differ in size"
        print(f"% {ds}: top-k run merged; unedited SAE-splice reference of the two runs: {flips} prediction flips, "
              f"max |confidence difference| {dmax:.4f}", file=sys.stderr)
    if args.range_all:      # range steering without coordinate selection, from a second run on the same pool
        Rr = json.loads((resolve_input_path(root / f"range_intervention_{ds}_dprime_{args.range_all}{args.tag}.json")).read_text())
        assert Rr["meta"]["percent"] == 1.0, f"{ds}: the --range_all run did not use --percent 1.0"
        flips = 0
        for k, c in D["concepts"].items():
            x = Rr["concepts"][k]
            assert x["eval_idx"] == c["eval_idx"], f"{ds} concept {k}: the two runs scored different documents"
            for sp in "hsz":
                flips += int((np.array(c["ref"][sp]["doc_correct"]) != np.array(x["ref"][sp]["doc_correct"])).sum())
                c.update({kk.replace("_st_range_a", "_st_rangeall_a"): v for kk, v in x.items() if kk.startswith(f"{sp}_st_range_a")})
        print(f"% {ds}: no-selection range run merged; unedited references of the two runs differ on {flips} predictions", file=sys.stderr)
    metas[ds] = (D["meta"], "summary" in D, list(D["concepts"]))
    for c in D["concepts"].values():
        C.append(c); strata.append(ds)
strata = np.array(strata)
if args.tex:
    TEXNAME = {"db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News", "emotion6": "Emotion", "news20": "20NG"}
    group = np.array(["FineWeb-Atlas" if d.startswith("atlas_s") else TEXNAME.get(d, d) for d in strata])
    order = list(dict.fromkeys(group))
    n1 = lambda v: "$" + format(0.0 if abs(v) < 0.05 else v, ".1f") + "$"
    ns = lambda v: "$" + format(0.0 if abs(v) < 0.05 else v, "+.1f") + "$"
    n = len(SP)
    head = {"h": "Base", "s": "SAE", "z": "AE", **{x: f"SAE$_{{{x[1:]}}}$" for x in EXTRA}}
    arms = [a for a, _ in SP]
    print("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l" + "r" * (3 * n - 1) + "}\n\\toprule")
    print(f" & \\multicolumn{{{n}}}{{c}}{{$\\Delta$ Target}} & \\multicolumn{{{n}}}{{c}}{{Collateral}} & \\multicolumn{{{n - 1}}}{{c}}{{$\\Delta$ Sel.\\ vs.\\ Base}} \\\\")
    print(f"\\cmidrule(lr){{2-{n + 1}}}\\cmidrule(lr){{{n + 2}-{2 * n + 1}}}\\cmidrule(lr){{{2 * n + 2}-{3 * n}}}")
    print("Dataset & " + " & ".join([head[a] for a in arms] * 2 + [head[a] for a in arms[1:]]) + " \\\\")
    counts = {}
    for op, title in OPS[:3]:
        key = f"{op}_a{args.alpha}"
        if not any(f"h_{key}" in c for c in C):
            continue
        print(f"\\midrule\n\\multicolumn{{{3 * n}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for g in order:
            idx = [i for i in range(len(C)) if group[i] == g]
            have = [a for a in arms if all(f"{a}_{key}" in C[i] for i in idx)]      # a top-k arm has no no-selection variant
            M = {sp: np.array([metrics(C[i], sp, key) for i in idx]) for sp in have}
            counts[g] = len(idx)
            print(f"{g} & " + " & ".join(n1(tmean(M[sp][:, 1])) if sp in have else "--" for sp in arms) + " & "
                  + " & ".join(n1(-tmean(M[sp][:, 2])) if sp in have else "--" for sp in arms) + " & "
                  + " & ".join(ns(tmean(M[sp][:, 3] - M["h"][:, 3])) if sp in have else "--" for sp in arms[1:]) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")
    print("% concepts per row: " + ", ".join(f"{g} {n}" for g, n in counts.items())
          + f"; metric {args.metric}{'' if args.metric == 'acc' else ' ' + args.scale}; alpha {args.alpha}; files *_hzs{args.tag}.json")
    raise SystemExit
unit = ("accuracy points" if args.metric == "acc" else
        "% of each space's own unedited confidence" if args.scale == "rel" else "confidence points")
print(f"# Three-space control{' (three-way true positives)' if 'tp3' in args.tag else ''}, {args.arm}: {args.dataset}; {unit}\n")
print("| Dataset | Concepts scored | Evaluation documents | Unedited accuracy, Base / SAE / AE | Perplexity, Base / SAE / AE |")
print("|---|---|---|---|---|")
for ds, (m, done, scored) in metas.items():
    ne = [m["n_eval_by_class"][k] for k in scored]           # the scored concepts' own documents
    print(f"| {ds}{'' if done else ' (RUNNING)'} | {len(scored)} | {sum(ne)} ({min(ne)}–{max(ne)} per class) | "
          f"{100 * m['base_acc']:.0f}% / {100 * m.get('srecon_acc', float('nan')):.0f}% / {100 * m['zrecon_acc']:.0f}% | "
          f"{m['base_ppl']:.2f} / {m.get('srecon_ppl', float('nan')):.2f} / {m['zrecon_ppl']:.2f} |")
print()
keys = sorted({k[2:] for c in C for k in c if k[:2] == "h_" and isinstance(c[k], dict) and "doc_conf" in c[k]})
for op, title in OPS:
    for key in [k for k in keys if k == op or k.startswith(op + "_a")]:
        a = key.split("_a")[-1] if "_a" in key else ""
        sp_k = [(a, n) for a, n in SP if all(f"{a}_{key}" in c for c in C)]
        idx = list(range(len(C)))
        if not sp_k or sp_k[0][0] != "h":
            continue
        M = {s: np.array([metrics(C[i], s, key) for i in idx]) for s, _ in sp_k}
        st = strata[idx]
        print(f"## {title}" + (f", strength {a}" if a else "") + f" ({len(idx)} concepts)\n")
        print("| Space | Unedited | Target change | Collateral | Selectivity |\n|---|---|---|---|---|")
        for s, name in sp_k:
            print(f"| {name} | {f1(tmean(M[s][:, 0]))} | {f1(tmean(M[s][:, 1]))} | {f1(tmean(M[s][:, 2]))} | {f1(tmean(M[s][:, 3]))} |")
        print("\n| Selectivity difference | Trimmed mean [95% CI] | Concepts better / worse |\n|---|---|---|")
        for a_, b_ in [pq for pq in PAIRS if pq[0] in M and pq[1] in M]:
            lab = f"{LAB[a_]} − {LAB[b_]}"
            d = M[a_][:, 3] - M[b_][:, 3]
            lo, hi = ci(d, st)
            print(f"| {lab} | {fs(tmean(d))} [{fs(lo)}, {fs(hi)}] | {int((d > 1e-9).sum())} / {int((d < -1e-9).sum())} |")
        print()
