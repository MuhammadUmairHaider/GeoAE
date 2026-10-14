"""SAE control tables restricted to what the SAE-spliced model still classifies correctly (user, 10 Oct).

Source runs (Gemma 3 4B L33, alpha 0.5; geoae/interp/range_intervention_compare.py):
  SAE run   results/delta/g4b_l33_d5120/interventions/sae/range_intervention_<ds>_dprime_gs16k.json   substrates h, s
  paper run results/delta/g4b_l33_d5120/interventions/range_intervention_<ds>_dprime[_conf].json      substrates h, z
Every evaluation document is correct for the base model and for the AE splice (the joint-correct pools). Here:
  1. a class is REMOVED when fewer than --min_tp of its target documents are correct under the unedited SAE splice;
  2. (--docs tp, default) the remaining concepts are scored only on documents the SAE splice also gets right: target
     documents of the concept, and other-class documents that are SAE-correct and not from a removed class.
     (--docs all: step 1 only, every document of the remaining classes.)
Base and SAE are scored on the SAME restricted documents, so SAE - Base is paired per concept. The AE column comes from the
paper run on the same remaining classes with all of its documents (that run has no SAE reference, so its documents cannot
be filtered by SAE correctness; the two runs' document sets differ by a few documents).
Metric = eval_out/paper_control_conf_all.py: target change and collateral loss in % of each space's own unedited
confidence (--scale rel) or in confidence points (--scale abs); selectivity = -target change - collateral; 10% trimmed
means over concepts; 95% bootstrap CI of the trimmed mean of per-concept paired differences (10k, seed 0).

    python eval_out/paper_control_sae_tp.py [--min_tp 10] [--docs tp|all] [--scale rel|abs]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--min_tp", type=int, default=10)
ap.add_argument("--docs", choices=["tp", "all"], default="tp")
ap.add_argument("--scale", choices=["rel", "abs"], default="rel")
ap.add_argument("--trim", type=float, default=0.10)
args = ap.parse_args()
D = "results/delta/g4b_l33_d5120/interventions/"
DS = [("db14", "DBpedia-14", "range_intervention_db14_dprime_conf.json"),
      ("biasbios", "Bias in Bios", "range_intervention_biasbios_dprime_conf.json"),
      ("ag_news", "AG News", "range_intervention_ag_news_dprime_conf.json"),
      ("emotion6", "Emotion", "range_intervention_emotion6_dprime.json"),
      ("news20", "20NG", "range_intervention_news20_dprime.json")]
OPS = [("st_range_a0.5", "Range steering"), ("st_global_a0.5", "Plain steering"), ("rm_range_comp", "Range removal")]


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def ci(d):
    d = np.asarray(d, float)
    b = tmean(d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))])
    return np.quantile(b, [.025, .975])


def metrics(c, s, op, tm, cm):
    """(start conf of target docs, target change, collateral, selectivity) on target docs tm and other-class docs cm."""
    nt = len(c["eval_idx"]["tgt"])
    r, e = np.array(c["ref"][s]["doc_conf"], float), np.array(c[f"{s}_{op}"]["doc_conf"], float)
    rt, et, rc, ec = r[:nt][tm], e[:nt][tm], r[nt:][cm], e[nt:][cm]
    tchg, coll = et.mean() - rt.mean(), np.maximum(0, rc - ec).mean()
    if args.scale == "rel":
        tchg, coll = tchg / rt.mean(), coll / rc.mean()
    return 100 * rt.mean(), 100 * tchg, 100 * coll, 100 * (-tchg - coll)


unit = "% of each space's unedited confidence" if args.scale == "rel" else "confidence points"
print(f"# SAE control at alpha 0.5, classes with < {args.min_tp} SAE-correct target documents removed; documents: "
      f"{'SAE-correct only' if args.docs == 'tp' else 'all'}; {unit}\n")
out = {}
for ds, nm, old in DS:
    S = json.load(open(D + f"sae/range_intervention_{ds}_dprime_gs16k.json"))
    Z = json.load(open(D + old))
    classes = S["meta"]["classes"]
    cls_of = {}                                             # eval-document index -> class id
    for k, c in S["concepts"].items():
        for i in c["eval_idx"]["tgt"]:
            cls_of[i] = int(k)
    tp = {int(k): int(sum(c["ref"]["s"]["doc_correct"][:len(c["eval_idx"]["tgt"])])) for k, c in S["concepts"].items()}
    removed = sorted(k for k, v in tp.items() if v < args.min_tp)
    kept = [k for k in S["concepts"] if int(k) not in removed]
    masks = {}
    for k in kept:
        c = S["concepts"][k]; nt = len(c["eval_idx"]["tgt"])
        ok = np.array(c["ref"]["s"]["doc_correct"], bool)
        notrem = np.array([cls_of.get(i, -1) not in removed for i in c["eval_idx"]["comp"]])
        masks[k] = (ok[:nt], ok[nt:] & notrem) if args.docs == "tp" else (np.ones(nt, bool), notrem)
    nt_m = [int(masks[k][0].sum()) for k in kept]; nc_m = [int(masks[k][1].sum()) for k in kept]
    print(f"## {nm}: {len(kept)} of {len(S['concepts'])} classes kept"
          + (f"; removed: {', '.join(f'{classes[k]} ({tp[k]})' for k in removed)}" if removed else "")
          + f"\ndocuments per concept: target {np.mean(nt_m):.0f} (min {min(nt_m)}), other-class {np.mean(nc_m):.0f} (min {min(nc_m)})")
    h0 = tmean([metrics(S["concepts"][k], "h", OPS[0][0], *masks[k])[0] for k in kept])
    s0 = tmean([metrics(S["concepts"][k], "s", OPS[0][0], *masks[k])[0] for k in kept])
    zk = [k for k in kept if k in Z["concepts"] and "z" in Z["concepts"][k]["ref"]]
    full = lambda c: (np.ones(len(c["eval_idx"]["tgt"]), bool), np.ones(len(c["eval_idx"]["comp"]), bool))
    z0 = tmean([metrics(Z["concepts"][k], "z", OPS[0][0], *full(Z["concepts"][k]))[0] for k in zk])
    print(f"unedited confidence on the target documents: Base {h0:.1f}, SAE {s0:.1f}, AE {z0:.1f} (AE: its own run, all documents)\n")
    print("| Edit | Target change, Base / SAE | Collateral, Base / SAE | Selectivity, SAE − Base [95% CI] | better / worse | AE − Base (same classes) |")
    print("|---|---|---|---|---|---|")
    for op, lab in OPS:
        H = np.array([metrics(S["concepts"][k], "h", op, *masks[k]) for k in kept])
        Sm = np.array([metrics(S["concepts"][k], "s", op, *masks[k]) for k in kept])
        d = Sm[:, 3] - H[:, 3]; lo, hi = ci(d)
        dz = np.array([metrics(Z["concepts"][k], "z", op, *full(Z["concepts"][k]))[3]
                       - metrics(Z["concepts"][k], "h", op, *full(Z["concepts"][k]))[3] for k in zk])
        zl, zh = ci(dz)
        out[(ds, op)] = (tmean(d), lo, hi, tmean(dz))
        print(f"| {lab} | {tmean(H[:, 1]):.1f} / {tmean(Sm[:, 1]):.1f} | {tmean(H[:, 2]):.1f} / {tmean(Sm[:, 2]):.1f} | "
              f"{tmean(d):+.1f} [{lo:+.1f}, {hi:+.1f}] | {int((d > 0).sum())} / {int((d < 0).sum())} | {tmean(dz):+.1f} [{zl:+.1f}, {zh:+.1f}] |")
    print()
