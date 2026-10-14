"""Descriptive matched-target summaries, without assuming monotone edit sweeps."""
from __future__ import annotations

import statistics


def first_crossing(points, level, metric="dnll"):
    """Interpolate the first upward crossing in actual strength order.

    The origin is the exact no-edit condition. No sorting by measured target
    loss and no extrapolation: either can hide a nonmonotone intervention curve.
    """
    previous = {g: {metric: 0.0} for g in ("target", "sibling", "random")}
    previous_strength = 0.0
    for strength, current in points:
        low, high = previous["target"][metric], current["target"][metric]
        if high > low and low <= level <= high:
            fraction = (level - low) / (high - low)
            result = {g: previous[g][metric] + fraction *
                      (current[g][metric] - previous[g][metric])
                      for g in ("target", "sibling", "random")}
            result["strength"] = previous_strength + fraction * (strength - previous_strength)
            result["segment"] = [previous_strength, strength]
            return result
        previous, previous_strength = current, strength
    return None


def summarize(cases, arms, metric="dnll", fixed_levels=(0.25, 0.5, 1.0)):
    def points(case, arm):
        return [(r["p"], r["metrics"]) for r in case["res"][arm]]

    def maximum(case, arm):
        return max([0.0] + [r["metrics"]["target"][metric] for r in case["res"][arm]])

    def aggregate(records):
        # Keep both orientations together; exclude partial pairs from headline means.
        grouped = {}
        for record in records:
            grouped.setdefault(record["pair_id"], []).append(record)
        complete = {k: v for k, v in grouped.items() if len(v) == 2}
        pair_values = {
            arm: {g: [statistics.mean(r["values"][arm][g] for r in pair)
                      for pair in complete.values()]
                  for g in ("target", "sibling", "random")}
            for arm in arms
        }
        return {
            "n_cases_reaching": len(records), "n_complete_pairs": len(complete),
            "n_complete_cases": 2 * len(complete),
            "n_incomplete_cases": len(records) - 2 * len(complete),
            "L_mean": statistics.mean(r["level"] for pair in complete.values() for r in pair)
                      if complete else None,
            "L_min": min((r["level"] for pair in complete.values() for r in pair), default=None),
            "L_max": max((r["level"] for pair in complete.values() for r in pair), default=None),
            "arms": {a: {g: statistics.mean(v) if v else None for g, v in gs.items()}
                     for a, gs in pair_values.items()},
            "records": records,
        }

    matched = []
    skipped = []
    for case in cases:
        level = 0.5 * min(maximum(case, arm) for arm in arms)
        values = {arm: first_crossing(points(case, arm), level, metric) for arm in arms}
        if level <= 1e-8 or any(v is None for v in values.values()):
            skipped.append({"T": case["T"], "S": case["S"], "max_target":
                            {a: maximum(case, a) for a in arms}})
            continue
        matched.append({"pair_id": case["pair_id"], "T": case["T"], "S": case["S"],
                        "level": level, "values": values})
    output = {"metric": metric, "arms_compared": arms,
              "rule": "first upward crossing in strength order; pair means over complete pairs",
              "matched": aggregate(matched), "skipped": skipped, "fixed": {}}
    for level in fixed_levels:
        records = []
        reach = {a: 0 for a in arms}
        for case in cases:
            values = {a: first_crossing(points(case, a), level, metric) for a in arms}
            for a in arms:
                reach[a] += values[a] is not None
            if all(v is not None for v in values.values()):
                records.append({"pair_id": case["pair_id"], "T": case["T"], "S": case["S"],
                                "level": level, "values": values})
        output["fixed"][str(level)] = {**aggregate(records), "reach_by_arm": reach}
    return output
