"""Control tables in CONFIDENCE units (change in the model's probability of the gold label), base vs GeoAE.

Confidence = softmax probability of the gold label's first token at the answer position (shared.predict gold_conf),
read from the *_dprime_conf.json reruns (per-doc records). Per concept and edit, against each space's own unedited run:
  target change   = mean edited conf - mean unedited conf on the concept's own docs (negative = edit lowered it)
  collateral loss = mean over other-class docs of max(0, unedited - edited)   (default: only losses count; a removed
                    label's probability mass flows to the other labels and raises their confidence, which is not harm)
                    --collateral signed: mean unedited - mean edited instead (rises cancel losses)
  selectivity     = target drop - collateral loss                      (higher = better)
Concept averages are 10%-trimmed means (floor(0.1 n) cut per side); Delta = trimmed mean of per-concept paired
GeoAE - base selectivity; 95% CI = bootstrap over concepts, stratified by dataset for the pooled rows (10k, seed 0).
--scale rel (default): target change and collateral loss as % of the space's own unedited confidence on those docs
  (the GeoAE splice starts from lower confidence than the base model, so absolute points understate its removal);
--scale abs: percentage points.
Usage: python eval_out/paper_control_conf_tables.py [--trim 0.1] [--collateral loss|signed] [--scale rel|abs]"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--collateral", choices=["loss", "signed"], default="loss")
ap.add_argument("--scale", choices=["rel", "abs"], default="rel")
ap.add_argument("--tag", default="conf", help="result-file tag: conf (alphas 0.5 1 2 + range removal) or conf_small")
ap.add_argument("--alphas", default=None, help="comma list of alphas to tabulate (default: 0.5,1.0 for conf, all five for conf_small)")
ap.add_argument("--compact", action="store_true", help="narrow tables: 'base → GeoAE' cells, starting confidence in its own table")
args = ap.parse_args()
rng = np.random.default_rng(0)
B = 10000


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def load(arm, ds):
    p = Path(f"results/delta/{arm}/interventions/range_intervention_{ds}_dprime_{args.tag}.json")
    if p.exists() and "summary" not in json.loads(p.read_text()):
        return []                       # unfinished run
    return list(json.load(open(p))["concepts"].values()) if p.exists() else []


def metrics(c, s, op):
    """(unedited target conf, target change, collateral loss, selectivity) in points for one concept/space/edit."""
    ref, rec = c["ref"][s], c[f"{s}_{op}"]
    nt = len(c["eval_idx"]["tgt"])
    r, e = np.array(ref["doc_conf"]), np.array(rec["doc_conf"])
    t_before, t_after = r[:nt].mean(), e[:nt].mean()
    coll = np.maximum(0, r[nt:] - e[nt:]).mean() if args.collateral == "loss" else (r[nt:] - e[nt:]).mean()
    tchg = t_after - t_before
    if args.scale == "rel":
        tchg, coll = tchg / t_before, coll / r[nt:].mean()
    return 100 * t_before, 100 * tchg, 100 * coll, 100 * (-tchg - coll)


ALS = (args.alphas.split(",") if args.alphas else (["0.1", "0.2", "0.3", "0.4", "0.5"] if args.tag == "conf_small" else ["0.5", "1.0"]))
ALS = [f"{float(a):.1f}" for a in ALS]
OPS = ([("rm_range_comp", "Range removal", "–")] if args.tag == "conf" else []) + \
      [(f"st_{m}_a{a}", name, a.rstrip("0").rstrip(".") if a.endswith(".0") else a)
       for m, name in (("range", "Range steering"), ("global", "Plain steering")) for a in ALS]
DSN = {"db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News"}
print(f"collateral = {args.collateral}; scale = {args.scale}; {args.trim:.0%}-trimmed means over concepts")
starting = {}
for arm, model in [("d6144", "Llama-3.2-3B, layer 27"), ("g4b_l33_d5120", "Gemma 3 4B, layer 33"),
                   ("qwen25_l35_d4096", "Qwen2.5-3B, layer 35")]:
    if not any(load(arm, ds) for ds in DSN):
        print(f"\n### {model}: no *_dprime_{args.tag}.json results yet")
        continue
    before = starting.setdefault(model, [])
    print(f"\n### {model}\n")
    if args.compact:
        print("| Dataset | Method | Target confidence removed | Collateral loss | Selectivity | Δ GeoAE − base [95% CI] | GeoAE better / worse |")
        print("|---|---|---|---|---|---|---|")
    else:
        print("| Dataset (concepts) | Method | α | Conf. before: base | Conf. before: GeoAE | Target change: base | Target change: GeoAE "
              "| Collateral loss: base | Collateral loss: GeoAE | Selectivity: base | Selectivity: GeoAE "
              "| Δ GeoAE − base [95% CI] | GeoAE better / worse |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for group in (["db14"], ["biasbios"], ["ag_news"], ["db14", "biasbios", "ag_news"]):
        strata = {g: load(arm, g) for g in group}
        if not all(strata.values()):
            missing = [g for g in group if not strata[g]]
            print(f"| {' + '.join(DSN[g] for g in group)} | (missing: {', '.join(missing)}) |" + " |" * 11)
            continue
        n = sum(len(v) for v in strata.values())
        nm = f"**All three ({n})**" if len(group) > 1 else f"{DSN[group[0]]} ({n})"
        for i, (op, name, al) in enumerate(OPS):
            M = {g: {s: np.array([metrics(c, s, op) for c in strata[g]]) for s in ("h", "z")} for g in group}
            H = np.concatenate([M[g]["h"] for g in group]); Z = np.concatenate([M[g]["z"] for g in group])
            dif = {g: M[g]["z"][:, 3] - M[g]["h"][:, 3] for g in group}
            allx = np.concatenate([dif[g] for g in group])
            boots = tmean(np.concatenate([dif[g][rng.integers(0, len(dif[g]), (B, len(dif[g])))] for g in group], axis=1))
            lo, hi = np.quantile(boots, [.025, .975])
            sig = "**" if (lo > 0 or hi < 0) and n >= 14 else ""
            f = lambda v, fmt=".1f": format(v, fmt).replace("-", "−")
            delta = f"{sig}{f(tmean(allx), '+.1f')} [{f(lo, '+.1f')}, {f(hi, '+.1f')}]{sig}"
            wl = f"{(allx > 1e-9).sum()} / {(allx < -1e-9).sum()}"
            if args.compact:
                arrow = lambda j, sgn=1: f"{f(sgn * tmean(H[:, j]))} → {f(sgn * tmean(Z[:, j]))}"
                meth = name + ("" if al == "–" else f", α {al}")
                print(f"| {nm if i == 0 else ''} | {meth} | {arrow(1, -1)} | {arrow(2)} | {arrow(3)} | {delta} | {wl} |")
                if i == 0:
                    before.append((nm, tmean(H[:, 0]), tmean(Z[:, 0])))
                continue
            cols = " | ".join(f"{f(tmean(H[:, j]))} | {f(tmean(Z[:, j]))}" for j in range(4))
            print(f"| {nm if i == 0 else ''} | {name} | {al} | {cols} | {delta} | {wl} |")
if args.compact and starting:
    print("\n### Starting confidence in the correct label on the target documents (%, before any edit)\n")
    models = [m for m in starting if len(starting[m]) == max(len(v) for v in starting.values())]
    print("| Dataset | " + " | ".join(f"{m.split(',')[0]}: base → GeoAE" for m in models) + " |")
    print("|---|" + "---|" * len(models))
    for k in range(len(starting[models[0]])):
        nm = starting[models[0]][k][0]
        print(f"| {nm} | " + " | ".join(f"{starting[m][k][1]:.1f} → {starting[m][k][2]:.1f}" for m in models) + " |")
