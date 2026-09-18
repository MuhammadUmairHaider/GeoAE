"""
Multi-judge auto-interp with Kendall's W, following VQLC (arXiv 2602.02726 §5.3).

WHY THIS EXISTS. A single LLM judge is not a measuring instrument until you know
how much of its output is the judge rather than the data. Measured here on
identical clusters and identical prompts, the `level` taxonomy assigned "syntax"
to 20% of clusters under gpt-4o-mini, 48% under gemini-2.5-flash-lite and 70%
under deepseek-v3.1. Any conclusion drawn from one judge's absolute numbers is
that judge's opinion.

WHAT IS REPORTED

  per-judge means            each judge scored separately, so drift is visible
  Kendall's W per metric     concordance of the judges' RANKINGS of the clusters,
                             0 = no agreement beyond chance, 1 = identical order.
                             Tie-corrected, since 1-5 rating scales tie heavily.
  consensus intrusion        word intrusion resolved by MAJORITY VOTE; items
                             where fewer than `--min_agree` judges agree are
                             DROPPED rather than counted, so a coin-flip between
                             judges cannot masquerade as a score.
  resolved fraction          how much of the sample survived that filter

A high mean with a low W means the judges agree on nothing and the mean is an
artifact of averaging. Report both or neither.

    python -m geoae.interp.judge_agreement results/ct_a.json results/ct_b.json \
      --judges google/gemini-2.5-flash-lite,openai/gpt-4o-mini,deepseek/deepseek-chat-v3.1 \
      --n_clusters 150
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def kendalls_w(ratings: np.ndarray) -> float:
    """
    Kendall's W for m judges x n items, with the standard tie correction.

    ratings[j, i] = judge j's score for item i. Scores are converted to ranks
    within each judge, so judges using different parts of the scale still agree
    if they order items the same way — which is the property we care about.
    """
    from scipy.stats import rankdata
    m, n = ratings.shape
    if m < 2 or n < 2:
        return float("nan")
    R = np.vstack([rankdata(row) for row in ratings])          # ties -> midranks
    Rsum = R.sum(axis=0)
    S = ((Rsum - Rsum.mean()) ** 2).sum()
    # tie correction: T_j = sum over tie groups of (t^3 - t)
    T = 0.0
    for row in R:
        _, cnt = np.unique(row, return_counts=True)
        T += float(((cnt ** 3) - cnt).sum())
    denom = (m ** 2) * (n ** 3 - n) - m * T
    return float(12.0 * S / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_paths", nargs="+")
    ap.add_argument("--judges", required=True,
                    help="comma list of >=3 models; VQLC used 3 (Claude Haiku, "
                         "Gemini Flash, DeepSeek)")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--checkpoint", default=None, help="override centroid source for hard negatives")
    ap.add_argument("--n_clusters", type=int, default=150)
    ap.add_argument("--min_agree", type=int, default=2,
                    help="drop an intrusion item unless this many judges agree")
    ap.add_argument("--cache_dir", default="cache/llm_judge")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--sample", default="stratified")
    ap.add_argument("--min_assigned", type=int, default=50)
    ap.add_argument("--out", default="results/judge_agreement.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if len(judges) < 3:
        print(f"[agree] WARNING: {len(judges)} judges. W is unstable below 3; "
              f"VQLC used 3 and reported W of 0.30-0.78.")

    import geoae.interp.llm_judge as J
    out = {}
    for path in args.json_paths:
        name = Path(path).name
        per_judge = {}
        for model in judges:
            ns = argparse.Namespace(
                json_paths=[path], checkpoint=args.checkpoint, provider=args.provider,
                model=model, n_clusters=args.n_clusters, sample=args.sample,
                min_assigned=args.min_assigned, n_explain=8, n_pos=6, n_neg=6,
                n_neighbours=4, n_intruder_trials=3, intruder_k=5,
                null_control=False, concurrency=args.concurrency,
                cache_dir=args.cache_dir, out=None, dry_run=False, seed=args.seed,
                anchor_mean_k=5, report_from=None,
            )
            llm = J.LLM(args.provider, model, Path(args.cache_dir))
            import random
            res = J.judge_run(Path(path), ns, llm, random.Random(args.seed))
            per_judge[model] = {c["cluster"]: c for c in res["clusters"]}
            print(f"[agree] {name} | {model}: {len(res['clusters'])} clusters, "
                  f"{llm.n_calls} calls / {llm.n_cached} cached")

        common = sorted(set.intersection(*[set(d) for d in per_judge.values()]))
        print(f"[agree] {name}: {len(common)} clusters scored by all {len(judges)} judges\n")

        rec = {"n_common": len(common), "judges": judges, "per_judge": {}, "W": {}}
        for metric in ("semantic_coh", "generality", "mono_llm", "intruder_acc"):
            # Filter PER METRIC to the clusters every judge scored. Dropping the
            # whole metric because one judge returned null on one cluster (an
            # unparsable response) silently deleted every row for an arm.
            ok = [c for c in common
                  if all(isinstance(per_judge[m][c].get(metric), (int, float))
                         for m in judges)]
            if len(ok) < 10:
                print(f"[agree]   {metric}: only {len(ok)} clusters scored by all judges — skipped")
                continue
            M = np.asarray([[per_judge[m][c][metric] for c in ok] for m in judges], dtype=float)
            rec["per_judge"][metric] = {j: float(M[i].mean()) for i, j in enumerate(judges)}
            rec["W"][metric] = kendalls_w(M)
            rec.setdefault("n_scored", {})[metric] = len(ok)

        # consensus intrusion: majority vote per cluster, unresolved items dropped
        res_n = drop = 0
        acc = []
        for c in common:
            vals = [per_judge[j][c].get("intruder_acc") for j in judges]
            if any(v is None for v in vals):
                continue
            # a cluster "passes" for a judge if it beat chance on its trials
            votes = [v > (1.0 / 6) for v in vals]
            top = Counter(votes).most_common(1)[0]
            if top[1] < args.min_agree:
                drop += 1
                continue
            res_n += 1
            acc.append(float(top[0]))
        rec["consensus_intrusion_pass_rate"] = float(np.mean(acc)) if acc else None
        rec["resolved_fraction"] = res_n / max(res_n + drop, 1)

        print(f"=== {name}")
        # Chance level for W is NOT 0. Under independent judges it sits near 1/m
        # (measured: 0.376 for m=3, n=60 uniform-random scores). VQLC reports
        # 0.30-0.78; the low end of that range is therefore indistinguishable
        # from no agreement, which their text notes for AG News + LLaMA (0.300).
        rng = np.random.RandomState(0)
        null = float(np.mean([kendalls_w(rng.rand(len(judges), max(len(common), 2)))
                              for _ in range(20)]))
        rec["W_null"] = null
        print(f"{'metric':<16}" + "".join(f"{j.split('/')[-1][:16]:>18}" for j in judges)
              + f"{'Kendall W':>12}")
        for metric, w in rec["W"].items():
            pj = rec["per_judge"][metric]
            print(f"{metric:<16}" + "".join(f"{pj[j]:>18.3f}" for j in judges) + f"{w:>12.3f}")
        print(f"{'  [chance W]':<16}" + " " * (18 * len(judges)) + f"{null:>12.3f}")
        print(f"\n  consensus intrusion pass-rate {rec['consensus_intrusion_pass_rate']}"
              f"   resolved {100*rec['resolved_fraction']:.0f}% of clusters "
              f"(>={args.min_agree}/{len(judges)} judges agree)\n")
        out[name] = rec

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[agree] wrote {args.out}")


if __name__ == "__main__":
    main()
