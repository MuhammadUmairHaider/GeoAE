"""Descriptive, matched-damage summaries for the base-selected removal run.

Match each primary comparison separately, preserving the original interpolation
convention. Average the two removal directions within each unordered pair;
report no case-wise significance tests because pairs can share a cluster.
This script requires only the Python standard library.
"""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path


GROUPS = {
    "primary": ["nl_h", "nl_z", "dir_range_x", "dir_range_z"],
    "neuronlens": ["nl_h", "nl_hb", "nl_z"],
    "direction_ranges": ["dir_range_x", "dir_range_z"],
    "direction_ranges_sensitivity": ["dir_range_x", "dir_range_h_native", "dir_range_z"],
    "range_vs_whole": ["dir_range_x", "dir_all_x", "dir_range_z", "dir_all_z"],
}


def describe(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)} if values else {}


def curve(case, arm):
    points = [(0., {"sibling": {"dnll": 0., "kl": 0.}, "random": {"dnll": 0., "kl": 0.}})]
    points.extend((r["target"]["dnll"], r) for r in case["res"][arm].values())
    return sorted(points, key=lambda p: p[0])


def at_level(points, level):
    xs = [p[0] for p in points]
    hi = bisect.bisect_right(xs, level)
    lo = max(0, hi-1)
    hi = min(hi, len(points)-1)
    weight = (level-xs[lo])/(xs[hi]-xs[lo]) if xs[hi] != xs[lo] else 0.
    return {g: {metric: (1-weight)*points[lo][1][g][metric] + weight*points[hi][1][g][metric]
                for metric in ("dnll", "kl")} for g in ("sibling", "random")}


def summarize_records(records, arms):
    by_pair = defaultdict(list)
    for rec in records:
        by_pair[tuple(sorted((rec["T"], rec["S"])))].append(rec)
    pairs = []
    for pair, cases in by_pair.items():
        pair_arms = {arm: {g: {metric: statistics.mean(c["arms"][arm][g][metric] for c in cases)
                              for metric in ("dnll", "kl")}
                          for g in ("sibling", "random")} for arm in arms}
        pairs.append({"pair": list(pair), "removal_directions": len(cases),
                      "target_level_mean": statistics.mean(c["target_level"] for c in cases), "arms": pair_arms})
    stats = {arm: {g: {metric: describe([p["arms"][arm][g][metric] for p in pairs])
                      for metric in ("dnll", "kl")}
                  for g in ("sibling", "random")} for arm in arms}
    return {"n_cases": len(records), "n_pairs": len(pairs),
            "n_complete_pairs": sum(p["removal_directions"] == 2 for p in pairs),
            "n_incomplete_pairs": sum(p["removal_directions"] != 2 for p in pairs),
            "target_levels": describe([r["target_level"] for r in records]),
            "arms": stats, "pairs": pairs, "cases": records}


def matched_group(cases, arms):
    records, skipped, absolute = [], [], {}
    curves = [{arm: curve(c, arm) for arm in arms} for c in cases]
    for case, cv in zip(cases, curves):
        level = .5 * min(max(x[0] for x in cv[a]) for a in arms)
        if level <= 0:
            skipped.append({"T": case["T"], "S": case["S"], "level": level})
            continue
        records.append({"T": case["T"], "S": case["S"], "target_level": level,
                        "arms": {a: at_level(cv[a], level) for a in arms}})
    result = summarize_records(records, arms)
    result["skipped_cases"] = skipped
    for level in (.25, .5, 1.):
        rs = [{"T": c["T"], "S": c["S"], "target_level": level,
               "arms": {a: at_level(cv[a], level) for a in arms}}
              for c, cv in zip(cases, curves) if all(max(p[0] for p in cv[a]) >= level for a in arms)]
        absolute[str(level)] = summarize_records(rs, arms)
        absolute[str(level)]["reach_by_arm"] = {
            a: sum(max(p[0] for p in cv[a]) >= level for cv in curves) for a in arms}
        kept = {(r["T"], r["S"]) for r in rs}
        absolute[str(level)]["excluded_cases"] = [
            {"T": c["T"], "S": c["S"]} for c in cases if (c["T"], c["S"]) not in kept]
    result["absolute_target_levels"] = absolute
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/delta/d6144/evals/base_selected_removal.json")
    ap.add_argument("--out", default="results/delta/d6144/evals/base_selected_comparison.json")
    args = ap.parse_args()
    raw = json.loads(resolve_input_path(Path(args.input)).read_text())
    assert raw["meta"]["args"]["selection_space"] == "residual"
    cases = raw["cases"]
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for c in cases:
        parent[find(c["T"])] = find(c["S"])
    components = defaultdict(list)
    for x in parent:
        components[find(x)].append(x)
    out = {"meta": {"source": args.input, "selection": raw["meta"]["args"],
                    "selected_pairs": len(raw["meta"]["pairs"]), "tested_cases": len(cases),
                    "tested_pairs": len({tuple(sorted((c["T"], c["S"]))) for c in cases}),
                    "components": list(components.values()), "docs": raw["meta"]["docs_streamed"],
                    "rows": raw["meta"]["n_rows"], "skipped_cases": raw["meta"]["skipped_cases"],
                    "matching": "Per case, interpolate at half the smallest maximum target loss increase within each comparison group.",
                    "aggregation": "Average available removal directions within a pair, then equal-weight pairs; complete/incomplete pair counts reported; descriptive statistics only.",
                    "position_in_raw_output": "T_position uses independently fitted normalized base direction; T_position_by_space records other spaces."},
           "groups": {name: matched_group(cases, arms) for name, arms in GROUPS.items()},
           "max_target_loss_increase": {a: describe([max(r["target"]["dnll"] for r in c["res"][a].values()) for c in cases])
                                        for a in cases[0]["res"]},
           "standard_setting": raw["summary"]["standard"]}
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=False))
    for name, group in out["groups"].items():
        print(name, "pairs", group["n_pairs"], "cases", group["n_cases"], "target", group["target_levels"])
        for arm, m in group["arms"].items():
            print(f"  {arm:22s} sibling {m['sibling']['dnll'].get('mean', float('nan')):.4f} random {m['random']['dnll'].get('mean', float('nan')):.4f}")
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
