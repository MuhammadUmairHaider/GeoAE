"""Per-class sizes of the three-way TRUE-POSITIVE document pools (correct under the base model, the AE splice and the SAE
splice; range_intervention_compare.py --tp_all, scripts/argo/control_three_way.sbatch with TP3=1).

Reads the pool file (documents tried per class, how many passed base + AE and entered the SAE filter, how many survived it)
and, when the control run has started, the result file's fit / evaluation allocation after the scoring-time re-check.

    python eval_out/tp3_pool_report.py [--datasets db14,biasbios,ag_news,emotion6,news20] [--arm g4b_l33_d5120] [--min_eval 20]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", default="db14,biasbios,ag_news,emotion6,news20")
ap.add_argument("--arm", default="g4b_l33_d5120")
ap.add_argument("--min_eval", type=int, default=20)
ap.add_argument("--classes", action="store_true", help="also print one row per class")
args = ap.parse_args()
NAMES = {"db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News", "emotion6": "Emotion", "news20": "20NG"}

rows, detail = [], {}
for ds in args.datasets.split(","):
    if ds.startswith("atlas_s"):
        pool = Path(f"results/delta/d6144/interventions/atlas_classes/s{ds[7:]}/joint_correct_tp3_g4b_gs16k.json")
    else:
        pool = Path(f"results/delta/joint_correct/joint_correct_{ds}_l33_{args.arm}_tp3_gs16k.json")
    res = Path(f"results/delta/{args.arm}/interventions/sae/range_intervention_{ds}_dprime_hzs_tp3.json")
    if not pool.exists():
        rows.append(f"| {NAMES.get(ds, ds)} | pool not built yet | | | | | |")
        continue
    P = json.loads(pool.read_text())
    names, built = P["classes"], P["classes_built"]
    got = Counter(d["label"] for d in P["docs"])
    tried = {int(k): v for k, v in P.get("tried_by_class", {}).items()}
    xs = {int(k): v for k, v in P.get("extra_stats", {}).get("sae", {}).items()}
    R = json.loads(res.read_text()) if res.exists() else None
    m = R["meta"] if R else {}
    nf, ne = m.get("n_fit_by_class", {}), m.get("n_eval_by_class", {})
    scored = set(R["concepts"]) if R else set()
    state = "pool only" if R is None else ("done" if "summary" in R else f"control running, {len(scored)} concepts")
    enough = [c for c in built if (ne.get(str(c), 0) if R else got[c]) >= (args.min_eval if R else 2 * args.min_eval)]
    rows.append(f"| {NAMES.get(ds, ds)} | {len(built)} | {sum(tried.get(c, 0) for c in built):,} | "
                f"{sum(xs.get(c, {}).get('entered', 0) for c in built):,} | {sum(xs.get(c, {}).get('survived', 0) for c in built):,} | "
                f"{sum(got[c] for c in built):,} | "
                + (f"{sum(ne.get(str(c), 0) for c in enough):,} in {len(enough)} classes | " if R else f"({len(enough)} classes with ≥ {2 * args.min_eval} documents) | ")
                + state + " |")
    detail[ds] = [(names[c], tried.get(c, 0), xs.get(c, {}).get("entered", 0), xs.get(c, {}).get("survived", 0), got[c],
                   nf.get(str(c), ""), ne.get(str(c), ""), "" if not R else ("yes" if str(c) in scored else
                   ("skipped" if ne.get(str(c), 0) < args.min_eval else "pending"))) for c in built]

print("| Dataset | Classes | Documents tried | Right under base and AE | Also right under SAE | Kept in pool | Evaluation documents used | State |")
print("|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
if args.classes:
    for ds, d in detail.items():
        print(f"\n## {NAMES.get(ds, ds)}\n")
        print("| Class | Tried | Right under base and AE | Also right under SAE | Kept in pool | Fit | Evaluation | Scored |")
        print("|---|---|---|---|---|---|---|---|")
        for r in d:
            print("| " + " | ".join(str(x) for x in r) + " |")
