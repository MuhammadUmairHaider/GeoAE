"""Analysis of the Atlas-class range interventions (geoae.interp.range_intervention_compare on atlas_s0/1/2), per draw and
pooled over draws (concepts of all draws are concatenated; SEs are over concepts).

  (a) per-alpha selectivity tables exactly like the DB14 ones: removed (target accuracy drop) h->z, collateral (other-class
      drop) h->z, selectivity z-h +- SE over concepts, z/h wins, ppl rise h/z; ops global/salient/range/transport + rm_range_comp.
  (b) matched deterioration (reuses eval_out/paper_matched_deterioration.py: curve / y_at_x / ci): removal at fixed
      collateral, collateral at fixed removal, removal at fixed ppl rise; z-h and transport-range, bootstrap over concepts.
  (c) collateral structure from the additive comp_by_class field: does collateral land on RELATED concepts? Per target class
      the Spearman correlation across the other classes between their accuracy drop and relatedness to the target (MiniLM
      "name: description" cosine; Atlas co-occurrence PMI), averaged over target classes with a bootstrap over target
      classes, separately for h and z (+ paired z-h); and the share of the total collateral (docs lost) carried by the
      single most-hit other class, h vs z.

    python eval_out/paper_atlas_class_interventions.py                       # draws 0,1,2 present in the directory
    python eval_out/paper_atlas_class_interventions.py --seeds 0 --tag _smoke
Writes <dir>/summary<tag>.json and prints the report.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OPS = ("global", "salient", "range", "transport")
REL = ("minilm_cos", "cooc_pmi")
MIN_DOCS_LOST = 3.0          # a target enters the concentration statistic only if h and z each lose >= this many docs in total
N_BOOT = 2000


def load_matched_module():
    """eval_out/paper_matched_deterioration.py runs its CLI loop at import; import it with an empty argv."""
    spec = importlib.util.spec_from_file_location("pmd", Path(__file__).with_name("paper_matched_deterioration.py"))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


PMD = load_matched_module()
BRNG = np.random.default_rng(0)


# ------------------------------------------------------------------------------------------------ (a) tables
def keys_of(meta):
    als = [f"{a:.1f}" for a in meta["alphas"]]
    return ["rm_range_comp"] + [f"st_{op}_a{a}" for a in als for op in OPS]


def summ(rows):
    """rows: [(h_rec, z_rec)] over concepts -> the tool's summary statistics."""
    f = lambda i, k: np.array([r[i][k] for r in rows])
    d = f(1, "selectivity") - f(0, "selectivity")
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return dict(n=len(rows), h_sel=float(f(0, "selectivity").mean()), z_sel=float(f(1, "selectivity").mean()),
                z_minus_h=float(d.mean()), z_minus_h_se=se, z_wins=int((d > 1e-9).sum()), h_wins=int((d < -1e-9).sum()),
                h_removed=float(f(0, "tgt_drop").mean()), z_removed=float(f(1, "tgt_drop").mean()),
                h_collateral=float(f(0, "comp_drop").mean()), z_collateral=float(f(1, "comp_drop").mean()),
                h_ppl_rise=float(f(0, "ppl_rise").mean()), z_ppl_rise=float(f(1, "ppl_rise").mean()))


def table(concepts, keys, title):
    print(f"\n--- {title}  ({len(concepts)} concepts) ---")
    print(f"{'intervention':<20} | {'removed h->z':>13} | {'collateral h->z':>15} | {'selectivity h->z':>17} | "
          f"{'z-h +-SE':>14} | {'z/h wins':>8} | {'ppl+ h/z':>11}")
    out = {}
    for k in keys:
        rows = [(c[f"h_{k}"], c[f"z_{k}"]) for c in concepts if f"h_{k}" in c and f"z_{k}" in c]
        if not rows:
            continue
        s = out[k] = summ(rows)
        print(f"{k:<20} | {s['h_removed']:5.2f}->{s['z_removed']:5.2f} | {s['h_collateral']:6.2f}->{s['z_collateral']:6.2f}  | "
              f"{s['h_sel']:+6.3f}->{s['z_sel']:+6.3f}  | {s['z_minus_h']:+6.3f}+-{s['z_minus_h_se']:5.3f} | "
              f"{s['z_wins']:>3}/{s['h_wins']:<4} | {s['h_ppl_rise']:+5.2f}/{s['z_ppl_rise']:+5.2f}")
    return out


# ------------------------------------------------------------------------------------------------ (b) matched
def nanmean(v):
    v = np.asarray(v, float)
    return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float("nan")


def matched(C, base_ppl, alphas):
    """Same quantities as paper_matched_deterioration.analyse, returned as a dict (and printed)."""
    PMD.ALPHAS = tuple(f"{a:.1f}" for a in alphas)
    ops = PMD.OPS
    C = [c for c in C if all(f"{sp}_st_{op}_a{a}" in c for sp in "hz" for op in ops for a in PMD.ALPHAS)]
    res = {"n_concepts": len(C)}
    print(f"\n--- matched deterioration ({len(C)} concepts; alphas {list(PMD.ALPHAS)}) ---")

    def block(name, levels, getter, pct_label):
        res[name] = {}
        for lv in levels:
            print(f"  {pct_label(lv)}:")
            vals, res[name][str(lv)] = {}, {}
            for op in ops:
                h = [getter("h", c, op, lv) for c in C]
                z = [getter("z", c, op, lv) for c in C]
                vals[op] = (h, z)
                d = PMD.ci(np.subtract(z, h))
                mh, mz = nanmean(h), nanmean(z)
                res[name][str(lv)][op] = dict(h=mh, z=mz, z_minus_h=[float(x) for x in d])
                print(f"    {op:9s} h {mh:6.1%}  z {mz:6.1%}   z-h {PMD.fmt(d)}")
            for sp, i in (("h", 0), ("z", 1)):
                d = PMD.ci(np.subtract(vals["transport"][i], vals["range"][i]))
                res[name][str(lv)][f"transport_minus_range_{sp}"] = [float(x) for x in d]
                print(f"    transport-range ({sp}) {PMD.fmt(d)}")

    block("removal_at_collateral", (0.10, 0.20, 0.30),
          lambda sp, c, op, x0: PMD.y_at_x(PMD.curve(c, sp, op), x0), lambda x: f"removal at collateral {x:.0%}")
    block("collateral_at_removal", (0.50, 0.70, 0.85),
          lambda sp, c, op, e0: PMD.y_at_x([(t, cc) for cc, t in PMD.curve(c, sp, op)], e0),
          lambda x: f"collateral at removal {x:.0%} (lower is better)")
    block("removal_at_ppl_rise", (0.02, 0.05),
          lambda sp, c, op, r0: PMD.y_at_x(PMD.curve(c, sp, op, x="ppl_rise"), r0 * base_ppl),
          lambda x: f"removal at perplexity +{x:.0%}")
    return res


# ------------------------------------------------------------------------------------------------ (c) collateral structure
def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return np.nan
    return float(spearmanr(a[m], b[m])[0])


def boot_mean(v, paired_other=None):
    """mean over targets (nan dropped) and a percentile CI from resampling targets with replacement."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(mean=float(v.mean()) if len(v) else None, lo=None, hi=None, n=int(len(v)))
    bs = np.array([v[BRNG.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
    return dict(mean=float(v.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=int(len(v)))


def target_records(draws, key):
    """Per (draw, target class): drop rows for h and z, the docs-lost vectors, relatedness rows. Skips targets without the key."""
    out = []
    for d in draws:
        for c, rec in d["res"]["concepts"].items():
            c = int(c)
            hr, zr = rec.get(f"h_{key}"), rec.get(f"z_{key}")
            if not hr or not zr or "comp_by_class" not in hr or "comp_by_class" not in zr:
                continue
            others = sorted(set(hr["comp_by_class"]) & set(zr["comp_by_class"]), key=int)
            o = np.array([int(x) for x in others])
            n = np.array([hr["comp_by_class"][x]["n"] for x in others], float)
            dh = np.array([hr["comp_by_class"][x]["drop"] for x in others], float)
            dz = np.array([zr["comp_by_class"][x]["drop"] for x in others], float)
            out.append(dict(seed=d["seed"], cls=c, others=o, n=n, dh=dh, dz=dz,
                            rel={r: np.array([d["rel"][r][c][int(x)] for x in o]) for r in REL}))
    return out


def collateral_structure(draws, keys):
    res = {}
    print("\n--- collateral structure (does collateral land on related concepts? how concentrated is it?) ---")
    print("rho = mean over target classes of Spearman(drop on other class, relatedness to target); +- = 95% bootstrap over targets")
    print(f"{'op':<20}{'rel':<11}{'rho h':>8}{'rho z':>8}{'z-h [95% CI]':>24}{'n':>5} | {'top-class share h':>18}{'z':>7}{'z-h [95% CI]':>22}{'n':>5}")
    for key in keys:
        T = target_records(draws, key)
        if not T:
            continue
        res[key] = {}
        for rname in REL:
            rh = np.array([spearman(t["dh"], t["rel"][rname]) for t in T])
            rz = np.array([spearman(t["dz"], t["rel"][rname]) for t in T])
            both = np.isfinite(rh) & np.isfinite(rz)
            bh, bz, bd = boot_mean(rh[both]), boot_mean(rz[both]), boot_mean((rz - rh)[both])
            # pooled over all (target, other) cells, cluster bootstrap over targets
            ch = spearman(np.concatenate([t["dh"] for t in T]), np.concatenate([t["rel"][rname] for t in T]))
            cz = spearman(np.concatenate([t["dz"] for t in T]), np.concatenate([t["rel"][rname] for t in T]))
            res[key][rname] = dict(rho_h=bh, rho_z=bz, rho_z_minus_h=bd, pooled_cells_rho_h=ch, pooled_cells_rho_z=cz)
            if rname == REL[0]:
                # concentration: docs lost per other class, positive part
                Lh = [np.clip(t["n"] * t["dh"], 0, None) for t in T]
                Lz = [np.clip(t["n"] * t["dz"], 0, None) for t in T]
                use = [i for i in range(len(T)) if Lh[i].sum() >= MIN_DOCS_LOST and Lz[i].sum() >= MIN_DOCS_LOST]
                sh = np.array([Lh[i].max() / Lh[i].sum() for i in use])
                sz = np.array([Lz[i].max() / Lz[i].sum() for i in use])
                uniform = float(np.mean([1.0 / len(T[i]["others"]) for i in use])) if use else None
                ch_, cz_, cd_ = boot_mean(sh), boot_mean(sz), boot_mean(sz - sh)
                res[key]["top_class_share"] = dict(h=ch_, z=cz_, z_minus_h=cd_, uniform_share=uniform, min_docs_lost=MIN_DOCS_LOST,
                                                   n_targets_total=len(T))
            ts = res[key].get("top_class_share")
            f = lambda b: ("n/a" if b["mean"] is None else f"{b['mean']:+.2f}") + ("" if b["lo"] is None else f" [{b['lo']:+.2f},{b['hi']:+.2f}]")
            tail = ""
            if rname == REL[0] and ts:
                tail = f" | {ts['h']['mean'] if ts['h']['mean'] is not None else float('nan'):>18.2f}{ts['z']['mean'] if ts['z']['mean'] is not None else float('nan'):>7.2f}{f(ts['z_minus_h']):>22}{ts['h']['n']:>5}"
            print(f"{key:<20}{rname:<11}{(bh['mean'] if bh['mean'] is not None else float('nan')):>8.2f}"
                  f"{(bz['mean'] if bz['mean'] is not None else float('nan')):>8.2f}{f(bd):>24}{bd['n']:>5}{tail}")
    return res


# ------------------------------------------------------------------------------------------------ main
def load_draws(d, seeds, tag):
    out = []
    for s in seeds:
        f = d / f"range_intervention_atlas_s{s}_dprime{tag}.json"
        r = d / f"s{s}" / "relatedness.json"
        if not f.exists() or not r.exists():
            print(f"[skip] draw {s}: missing {f if not f.exists() else r}")
            continue
        res = json.load(open(f))
        if "summary" not in res:
            print(f"[skip] draw {s}: {f.name} is incomplete (no summary)")
            continue
        out.append(dict(seed=s, res=res, rel=json.load(open(r)), meta=res["meta"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/delta/d6144/interventions/atlas_classes")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="", help="_smoke for the smoke outputs")
    args = ap.parse_args()
    d = Path(args.dir)
    draws = load_draws(d, args.seeds, args.tag)
    if not draws:
        raise SystemExit("no complete range_intervention_atlas_s*_dprime*.json found")
    summary = {"seeds": [x["seed"] for x in draws], "per_draw": {}, "pooled": {}, "smoke": args.tag == "_smoke"}
    alphas = draws[0]["meta"]["alphas"]
    keys = keys_of(draws[0]["meta"])
    for x in draws:
        m = x["meta"]
        print(f"\n######## draw {x['seed']}: {len(x['res']['concepts'])} classes intervened; base acc {m['base_acc']}, "
              f"AE-splice acc {m['zrecon_acc']}, ppl {m['base_ppl']} / {m['zrecon_ppl']}; n_fit/n_eval/n_comp {m['n_fit']}/{m['n_eval']}/{m['n_comp']}")
        C = list(x["res"]["concepts"].values())
        summary["per_draw"][str(x["seed"])] = dict(
            classes=[m["classes"][int(c)] for c in x["res"]["concepts"]],
            n_eval_by_class={m["classes"][int(c)]: m["n_eval_by_class"][c] for c in x["res"]["concepts"]},
            table=table(C, keys, f"draw {x['seed']} selectivity"))
        summary["per_draw"][str(x["seed"])]["matched"] = matched(C, m["base_ppl"], alphas)
        summary["per_draw"][str(x["seed"])]["collateral_structure"] = collateral_structure(
            [x], ["rm_range_comp"] + [f"st_{op}_a{a:.1f}" for a in alphas for op in OPS])
    print(f"\n######## pooled over draws {summary['seeds']}")
    Cp = [c for x in draws for c in x["res"]["concepts"].values()]
    summary["pooled"]["table"] = table(Cp, keys, "pooled selectivity")
    summary["pooled"]["matched"] = matched(Cp, float(np.mean([x["meta"]["base_ppl"] for x in draws])), alphas)
    cs_keys = ["rm_range_comp"] + [f"st_{op}_a{a:.1f}" for a in alphas for op in OPS]
    summary["pooled"]["collateral_structure"] = collateral_structure(draws, cs_keys)
    out = d / f"summary{args.tag}.json"
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\n[analysis] wrote {out}")


if __name__ == "__main__":
    main()
