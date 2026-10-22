"""Tables for geoae/evaluation/chain_control.py (one or both reconstructions spliced in; each arm edits one space of the chain).

Within an order the reference is the unedited chain, identical for every arm, so numbers are plain confidence points
(--scale rel divides by the order's unedited confidence: one denominator for all arms of that order).
  target change = mean edited - mean unedited gold-label confidence on the concept's own documents
  collateral    = mean over the sampled other-class documents of max(0, unedited - edited)
  selectivity   = -target change - collateral                                      (higher = better)
Concept averages are 10%-trimmed means (a plain mean below 10 concepts). A difference between two arms is paired on the
same documents. Per dataset its 95% CI is a DOCUMENT bootstrap (target and other-class documents resampled within each
concept, one draw for both arms; --n_boot reps, seed 0): uncertainty about these concepts' documents. The pooled table
(several files) adds a CONCEPT bootstrap stratified by dataset (10,000 reps) and a Wilcoxon signed-rank test over concepts.
Controls: @rand = the same number of coordinates drawn at random, @shuf = class labels permuted before fitting.

    python experiments/paper/analysis/paper_chain_control_table.py results/delta/g4b_l33_d5120/interventions/chain/chain_control_*.json
"""

from geoae.paths import resolve_input_path
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from geoae.evaluation.chain_control import selectivity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--scale", choices=["abs", "rel"], default="abs")
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--sae_arm", default="s", help="the SAE arm of the AE-vs-SAE tables (s = top 30 percent of features, s_top = top-K)")
args = ap.parse_args()
NAME = {"h_first": "Residual, before", "h_last": "Residual, after", "z": "AE latent", "s": "SAE codes", "s_top": "SAE top-K features"}
OPN = {"st_range": "Range steering", "st_global": "Plain steering", "rm_range_comp": "Range removal"}
LAB = {"base": "Base", "ae": "AE spliced", "ae_only": "AE spliced", "sae": "SAE spliced", "sae_only": "SAE spliced",
       "ae_sae": "Chain AE → SAE", "sae_ae": "Chain SAE → AE"}
opn = lambda op: next(v for k, v in OPN.items() if op.startswith(k))


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def sel_boot(ref, run, nt, ti, ci):
    """selectivity for bootstrap index matrices ti (B, nt), ci (B, nc) -> (B,)"""
    r, e = np.asarray(ref, float), np.asarray(run, float)
    rt, et, rc, ec = r[:nt][ti], e[:nt][ti], r[nt:][ci], e[nt:][ci]
    tchg, coll = et.mean(1) - rt.mean(1), np.maximum(0, rc - ec).mean(1)
    if args.scale == "rel":
        tchg, coll = tchg / rt.mean(1), coll / rc.mean(1)
    return 100 * (-tchg - coll)


def paired(C, o1, k1, o2, k2):
    """selectivity(order o1, run key k1) - selectivity(o2, k2) per concept, and the document-bootstrap CI of its trimmed mean."""
    rng = np.random.default_rng(0)
    pt, bs = [], []
    for c in C.values():
        if k1 not in c["runs"] or k2 not in c["runs"]:
            continue
        nt, nc = len(c["eval_idx"]["tgt"]), len(c["eval_idx"]["comp"])
        r1, r2, e1, e2 = c["ref"][o1]["doc_conf"], c["ref"][o2]["doc_conf"], c["runs"][k1]["doc_conf"], c["runs"][k2]["doc_conf"]
        pt.append(selectivity(r1, e1, nt, args.scale)[2] - selectivity(r2, e2, nt, args.scale)[2])
        ti, ci = rng.integers(0, nt, (args.n_boot, nt)), rng.integers(0, nc, (args.n_boot, nc))
        bs.append(sel_boot(r1, e1, nt, ti, ci) - sel_boot(r2, e2, nt, ti, ci))
    if not pt:
        return None
    pt = np.array(pt)
    b = tmean(np.array(bs).T)
    return pt, float(np.quantile(b, .025)), float(np.quantile(b, .975))


def cell(res):
    if res is None:
        return ""
    pt, lo, hi = res
    return f"{tmean(pt):+.1f} [{lo:+.1f}, {hi:+.1f}] {int((pt > 0).sum())}/{int((pt < 0).sum())}"


pooled = {}                                                  # (comparison label, op) -> list of per-concept arrays, one per file
for f in args.files:
    J = json.load(open(resolve_input_path(f)))
    M, C = J["meta"], J["concepts"]
    stages = M.get("order_stages") or M["orders"]
    if isinstance(stages, list):
        stages = {o: o.split("_") for o in stages}
    arms = M["signature"]["arms"] if "signature" in M else M["arms"]      # meta["arms"] is the raw option string in new files
    arms = arms.split(",") if isinstance(arms, str) else arms
    orders = list(stages)
    refkey = {o: o for o in orders}
    ops = sorted({k.split("|")[2] for c in C.values() for k in c["runs"] if "@" not in k}, key=lambda o: (o.startswith("rm"), o))
    has_ctl = any("@" in k for c in C.values() for k in c["runs"])
    unit = "confidence points" if args.scale == "abs" else "% of the order's unedited confidence"
    print(f"# {M['dataset']}: {len(C)} concepts{'' if J.get('done') else ' (RUNNING)'}, "
          f"{sorted(set(M['n_eval_by_class'].values()))} evaluation documents per class; {unit}\n")
    if M.get("dropped_classes"):
        print("Dropped classes: " + "; ".join(f"{k} ({v})" for k, v in M["dropped_classes"].items()) + "\n")
    if M.get("checks"):
        ch = M["checks"]
        print(f"Checks: chain input vs fitted activations, max relative L2 error {ch['input_rel_err']:.1e} "
              f"(min cosine {ch.get('input_min_cos', float('nan')):.5f}); identity edit, max confidence change "
              f"{max(ch['identity_max_abs_diff'].values()):.1e}\n")
    print("| Unedited model | Accuracy | Confidence | Perplexity |\n|---|---|---|---|")
    for k in ("base", "ae", "ae_only", "sae", "sae_only", "ae_sae", "sae_ae"):
        if k in M["unedited"]:
            u = M["unedited"][k]
            print(f"| {LAB[k]} | {u['acc']:.1%} | {100 * u['conf']:.1f} | {u['ppl']:.2f} |" if u.get("ppl") else f"| {LAB[k]} | {u['acc']:.1%} | {100 * u['conf']:.1f} | |")
    for o in orders:
        print(f"\n## {LAB[o]} (every row starts from this unedited model)\n")
        print("| Edit | Edited space | Coordinates | Target change | Collateral | Selectivity |"
              + (" Random coordinates | Shuffled labels |" if has_ctl else ""))
        print("|---|---|---|---|---|---|" + ("---|---|" if has_ctl else ""))
        for op in ops:
            for arm in arms:
                k = f"{o}|{arm}|{op}"
                rows = [selectivity(c["ref"][o]["doc_conf"], c["runs"][k]["doc_conf"], len(c["eval_idx"]["tgt"]), args.scale)
                        for c in C.values() if k in c["runs"]]
                if not rows:
                    continue
                R = np.array(rows)
                nco = [c["info"][f"{o}|{arm}"].get("n_coords", c["info"][f"{o}|{arm}"].get("n_salient")) for c in C.values() if f"{o}|{arm}" in c["info"]]
                dim = next(c["info"][f"{o}|{arm}"]["dim"] for c in C.values() if f"{o}|{arm}" in c["info"])
                coords = "all" if op.startswith("st_global") else f"{int(np.median(nco)):,} of {dim:,}"
                ctl = ""
                if has_ctl:
                    cs = []
                    for t in ("rand", "shuf"):
                        rr = [selectivity(c["ref"][o]["doc_conf"], c["runs"][k + "@" + t]["doc_conf"], len(c["eval_idx"]["tgt"]), args.scale)[2]
                              for c in C.values() if k + "@" + t in c["runs"]]
                        cs.append(f"{tmean(rr):+.1f}" if rr else "")
                    ctl = f" {cs[0]} | {cs[1]} |"
                print(f"| {opn(op)} | {NAME[arm]} | {coords} | {tmean(R[:, 0]):.1f} | {tmean(R[:, 1]):.1f} | {tmean(R[:, 2]):+.1f} |{ctl}")
    S = args.sae_arm
    comps = []                                               # (label, order of the AE arm, order of the SAE arm)
    if "ae" in orders and "sae" in orders:
        comps.append(("Single splice (each against its own model)", "ae", "sae"))
    comps += [(f"Same chain, {LAB[o][6:]}", o, o) for o in orders if len(stages[o]) == 2]
    if "ae_sae" in orders and "sae_ae" in orders:
        comps += [("Each edited last", "sae_ae", "ae_sae"), ("Each edited first", "ae_sae", "sae_ae")]
    if comps:
        print(f"\n## AE latent minus {NAME[S]}: selectivity difference [95% CI], concepts AE better / worse\n")
        print("| Edit | " + " | ".join(c[0] for c in comps) + " |\n|---|" + "---|" * len(comps))
        for op in ops:
            if S == "s_top" and op.startswith("st_global"):
                continue
            cells = []
            for lab, oa, os_ in comps:
                res = paired(C, oa, f"{oa}|z|{op}", os_, f"{os_}|{S}|{op}")
                cells.append(cell(res))
                if res is not None:
                    pooled.setdefault((f"AE − {NAME[S]}: {lab}", op), []).append(res[0])
            print(f"| {opn(op)} | " + " | ".join(cells) + " |")
    print("\n## Each space minus the residual after the chain (same model), selectivity difference [95% CI], better / worse\n")
    sp = [a for a in ("z", "s", "s_top") if a in arms]
    print("| Edit | Order | " + " | ".join(NAME[a] for a in sp) + " |\n|---|---|" + "---|" * len(sp))
    for op in ops:
        for o in orders:
            cells = []
            for a in sp:
                res = paired(C, o, f"{o}|{a}|{op}", o, f"{o}|h_last|{op}")
                cells.append(cell(res))
                if res is not None:
                    pooled.setdefault((f"{NAME[a]} − residual after, {LAB[o]}", op), []).append(res[0])
            print(f"| {opn(op)} | {LAB[o]} | " + " | ".join(cells) + " |")
    if has_ctl:
        print("\n## Real edit minus its controls (range steering), selectivity difference [95% CI], better / worse\n")
        print("| Order | Edited space | Minus random coordinates | Minus shuffled labels |\n|---|---|---|---|")
        op = next(o for o in ops if o.startswith("st_range"))
        for o in orders:
            for a in arms:
                k = f"{o}|{a}|{op}"
                cells = [cell(paired(C, o, k, o, k + "@" + t)) for t in ("rand", "shuf")]
                if any(cells):
                    print(f"| {LAB[o]} | {NAME[a]} | {cells[0]} | {cells[1]} |")
    print()

if len(args.files) > 1 and pooled:
    from scipy.stats import wilcoxon
    print(f"# Pooled over {len(args.files)} datasets: trimmed mean of per-concept differences [95% CI, concept bootstrap stratified by "
          f"dataset], better / worse, Wilcoxon p\n")
    print("| Comparison | Edit | Concepts | Difference [95% CI] | Better / worse | Wilcoxon p |\n|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    for (lab, op), parts in pooled.items():
        allx = np.concatenate(parts)
        bs = tmean(np.concatenate([p[rng.integers(0, len(p), (10000, len(p)))] for p in parts], axis=1))
        lo, hi = np.quantile(bs, [.025, .975])
        p = wilcoxon(allx).pvalue if (allx != 0).sum() >= 6 else float("nan")
        print(f"| {lab} | {opn(op)} | {len(allx)} | {tmean(allx):+.1f} [{lo:+.1f}, {hi:+.1f}] | {int((allx > 0).sum())} / {int((allx < 0).sum())} | {p:.2g} |")
