
from geoae.paths import resolve_input_path
import json, re, statistics as st
from pathlib import Path
R = Path("results/delta")
ARMS = ["d768", "d3072", "d6144", "d12288", "d6144k4000"]
def L(p):
    try: return json.load(open(resolve_input_path(p)))
    except Exception: return None
S = json.load(open(resolve_input_path(R / "summary.json")))
out = {"arms": ARMS, "summary": S, "probe": {}, "tokstruct": {}, "cjudge": {}, "concept": {}, "transfer": {}, "geometry": {}, "usage": {}, "steer": {}, "number": {}}
usage = L("logs/panel_usage.json") or {}
for a in ARMS:
    k = 4000 if "k4000" in a else 2000; w = int(re.match(r"d(\d+)", a).group(1))
    for d, v in usage.items():
        if d.endswith(f"k{k}_bnh_b32k_lam1_d{w}_dpc_sampled_tokbias"): out["usage"][a] = v
    p = L(R / a / "evals/probe.json") or {}
    out["probe"][a] = {r: {"grain": v["grain"], "n": v["n"], "leak": v["leak"],
        "arms": {n: {"cnmi": x["cnmi"], "ci": [x["cnmi"] - 1.96 * st.pstdev(x["boots"]), x["cnmi"] + 1.96 * st.pstdev(x["boots"])]} for n, x in v["arms"].items()}} for r, v in p.items()}
    out["tokstruct"][a] = L(R / a / "evals/token_structure.json")
    cj = (L(R / a / "steering/cluster_steer_judge_llmfluency.json") or L(R / a / "steering/cluster_steer_judge.json")) or {}
    out["cjudge"][a] = cj.get("net_fluent")
    cs = L(R / a / "steering/cluster_steering.json") or {}
    out["transfer"][a] = {arm: {dd: v["transfer"] for dd, v in by.items()} for arm, by in (cs.get("matched_disrupt") or {}).items()}
    cc = (L(R / a / "concept/concept_steer_judge_llmfluency.json") or L(R / a / "concept/concept_steer_judge.json")) or {}
    if cc:
        sc = {}
        for scope, v in cc["scopes"].items():
            best = cc["best_alpha"]
            sc[scope] = {"usable": {h: v["per_alpha"][h][best[h]]["usable"] for h in best if h in v["per_alpha"] and best[h] in v["per_alpha"][h]},
                         "fluent": {h: v["per_alpha"][h][best[h]]["fluent_share"] for h in best if h in v["per_alpha"] and best[h] in v["per_alpha"][h]},
                         "pairwise": v["pairwise"]}
        out["concept"][a] = {"best_alpha": cc["best_alpha"], "scopes": sc}
    sd = L(R / a / "interventions/steer_db14.json")
    if sd: out["steer"][a] = sd["meta"].get("summary_selectivity")
    nr = L(R / a / "interventions/number_review.json")
    if nr: out["number"][a] = {k2: {"d": v["difference"], "ci": v["noun_bootstrap_95"]} for k2, v in nr["paired_alpha1_comparisons"].items()}
    if nr:
        av = nr["averages_over_suppression_directions"]
        out.setdefault("number_abs", {})[a] = {k2: {m: v[m] for m in ("strict_target_flip", "complement_drop", "neutral_top1_change", "neutral_kl")} for k2, v in av.items()}
    # per-operator intervention summaries (removal + every steering operator/strength)
    OPK = ("h_tgt_drop", "h_comp_drop", "z_tgt_drop", "z_comp_drop", "z_minus_h", "z_minus_h_se", "z_wins", "n", "h_ppl_rise", "z_ppl_rise")
    out.setdefault("ops", {})[a] = {}
    for ds in ("db14", "biasbios"):
        rs = L(R / a / f"interventions/range_intervention_{ds}_dprime.json")
        if rs:
            sm = rs["summary"]
            out["ops"][a][ds] = {k2: {kk: v[kk] for kk in OPK} for k2, v in sm.items() if k2.startswith(("rm_", "st_"))}
    g = {}
    lg = R / a / "logs/geometry/geometry.log"
    if resolve_input_path(lg).exists():
        cur = None
        for line in resolve_input_path(lg).read_text().splitlines():
            m = re.match(r"\[geo\] ([a-z_0-9]+): ([\d,]+) points, (\d+) classes \((\w+)\)", line)
            if m: cur = m.group(1); g[cur] = {"n": int(m.group(2).replace(",", "")), "classes": int(m.group(3)), "grain": m.group(4)}
            m = re.search(r"kNN-10 label agreement: (.*)", line)
            if m and cur: g[cur]["knn"] = {x.split()[0]: float(x.split()[1]) for x in re.findall(r"[a-z_]+ [0-9.]+", m.group(1))}
    out["geometry"][a] = g
import sys
# ---- linear-probe tests (TPP, bias probe), matched-base number control (hb), all-hubs transfer
import numpy as _np
out["tpp"], out["biasprobe"], out["number_hb"], out["transfer_allhubs"] = {}, {}, {}, {}
for a in ARMS:
    out["tpp"][a] = {}
    for ds in ("dbpedia", "biasbios"):
        t = L(R / a / f"probes/tpp_{ds}.json")
        if not t: continue
        sp = {}
        for name, v in t["spaces"].items():
            cs = [c for c in v["concepts"].values() if c.get("status") == "evaluated"]
            imp = [c["importance"]["k_at_50pct_drop_frac"] for c in cs if c["importance"]["k_at_50pct_drop_frac"] is not None]
            rnd = [c["random"]["k_at_50pct_drop_frac"] for c in cs if c["random"]["k_at_50pct_drop_frac"] is not None]
            cd = [c["importance"]["complement_drop_at_k50"] for c in cs if c["importance"].get("complement_drop_at_k50") is not None]
            sp[name] = {"n_dims": v["n_dims"], "probe_acc": v["probe_accuracy"], "k50_frac": float(_np.mean(imp)) if imp else None,
                        "n_reached": len(imp), "n_classes": len(cs), "random_k50_frac": float(_np.mean(rnd)) if rnd else None,
                        "collateral_at_k50": float(_np.mean(cd)) if cd else None}
        out["tpp"][a][ds] = sp
    b = L(R / a / "probes/biasbios_probe.json")
    if b:
        sp = {}
        for name, v in b["spaces"].items():
            ab = v["ablations"]; e = {"prof_bal": v["test_baseline"]["task_balanced_accs"], "gender_bal": v["test_baseline"]["spur_balanced_accs"], "n_dims": v["n_dims"]}
            for key in ("gender_probe", "gender_dprime"):
                nc = ab[key]["operating_points"]["near_chance"]
                e[key] = {"frac": nc.get("fraction"), "k": nc.get("k"),
                          "prof_bal_after": (nc.get("test") or {}).get("task_balanced_accs"), "gender_bal_after": (nc.get("test") or {}).get("spur_balanced_accs")}
            rn = [ab[k]["operating_points"]["near_chance"].get("fraction") for k in ab if k.startswith("random_") and "operating_points" in ab[k]]
            e["random_frac"] = float(_np.mean([x if x is not None else 1.0 for x in rn])) if rn else None
            sp[name] = e
        out["biasprobe"][a] = sp
    hb = L(R / a / "interventions/range_number_dprime_hb.json")
    if hb:
        arms_ = hb["arms"]; res = {}
        for pre in ("h", "hb", "z"):
            for op in ("range", "transport"):
                for al in ("0.5", "1.0", "2.0"):
                    ks = [k for k in arms_ if k.startswith(pre + ":") and k.endswith(f":{op}:a{al}")]
                    if ks: res[f"{pre}:{op}:a{al}"] = float(_np.mean([arms_[k]["all"]["target_drop"] for k in ks]))
        out["number_hb"][a] = res
    ah = L(R / a / "steering/cluster_steering_allhubs.json")
    if ah: out["transfer_allhubs"][a] = {arm: {dd: v["transfer"] for dd, v in by.items()} for arm, by in ah["matched_disrupt"].items()}
output = Path(sys.argv[1] if len(sys.argv) > 1 else "results/width/report_data.json")
output.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(output, "w"), indent=1, default=lambda o: None)
print("ok", {a: (len(out["probe"][a]), bool(out["concept"].get(a)), len(out["geometry"][a]), bool(out["usage"].get(a))) for a in ARMS})
