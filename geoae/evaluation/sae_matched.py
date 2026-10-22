"""Secondary descriptive matching on already observed held-out removal curves.

The primary experiment selects intervention strengths using fitting data.
This postprocessing conditions on *observed* held-out target damage and linearly
interpolates collateral between those measured strengths. It does not rerun a
matched intervention or provide an independently held-out calibrated estimate.

python -m geoae.evaluation.sae_matched removal.json --out observed_matching.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from geoae.evaluation.sae_removal import family_bootstrap


DEFAULT_LEVELS = (.025, .05, .1)
_GROUPS = ("target", "sibling", "random")
_METRICS = ("delta_nll", "kl")


def sae_range_operator_diagnostic(removal_analysis):
    """Flag whether control-mean replacement can create an inactive coefficient.

    This is an algebraic flag from stored fit interval metadata, not a claim
    about the number of zero coefficients actually present on evaluation rows.
    It does not change the launched intervention or any selection/matching rule.
    """
    targets = {}
    for row in removal_analysis["records"]:
        geometry = row.get("sae_feature_range")
        if row["arm"] != "sae_range" or not geometry:
            continue
        includes_zero = geometry["lo"] <= 0 <= geometry["hi"]
        positive_control_mean = geometry["control_mean"] > 0
        targets[(row["pair_id"], row["target"])] = {
            "pair_id": row["pair_id"], "target": row["target"],
            "zero_in_fit_gate": bool(includes_zero),
            "positive_control_mean": bool(positive_control_mean),
            "can_activate_inactive_coefficient": bool(includes_zero and positive_control_mean),
        }
    return {
        "n_targets": len(targets),
        "n_zero_in_fit_gate": sum(r["zero_in_fit_gate"] for r in targets.values()),
        "n_positive_control_mean": sum(r["positive_control_mean"] for r in targets.values()),
        "n_can_activate_inactive_coefficient": sum(r["can_activate_inactive_coefficient"] for r in targets.values()),
        "operator": "range-gated movement toward fit control coefficient mean; no coefficient>0 eligibility gate",
        "limitation": "potential from interval algebra; observed zero-coefficient counts are not stored",
        "targets": list(targets.values()),
    }


def interpolate_observed_curve(records, level: float, *, min_observed_points: int = 2):
    """Interpolate a single intervention's alpha-ordered measured curve.

    At least ``min_observed_points`` distinct positive-alpha measured points are
    required in addition to the known zero-edit origin. Damage must be monotone
    over the whole available observed curve. No sorting by damage, envelope
    repair, or extrapolation is permitted. Duplicate alphas must agree exactly
    about outcomes, sample counts, and selected features or the curve is refused.
    """
    if not np.isfinite(level) or level < 0 or min_observed_points < 1:
        raise ValueError("level must be finite/nonnegative and point minimum positive")
    available = [r for r in records if r.get("status") == "matched" and "evaluation" in r]
    available.sort(key=lambda r: r["alpha"])
    points = []
    for row in available:
        alpha = row["alpha"]
        if not np.isfinite(alpha) or alpha < 0:
            return {"status": "nonfinite", "alpha_interpolated": None}
        values = np.asarray([row["evaluation"][g][m] for g in _GROUPS for m in _METRICS], dtype=float)
        if not np.isfinite(values).all():
            return {"status": "nonfinite", "alpha_interpolated": None}
        if alpha == 0:
            if np.any(values != 0):
                return {"status": "inconsistent_origin", "alpha_interpolated": None}
            continue
        if points and alpha == points[-1]["alpha"]:
            previous = points[-1]
            if row["evaluation"] != previous["evaluation"] or row.get("selected_features") != previous.get("selected_features"):
                return {"status": "inconsistent_duplicate_alpha", "alpha_interpolated": None}
            continue
        points.append(row)
    if len(points) < min_observed_points:
        return {"status": "insufficient_observed_points", "alpha_interpolated": None,
                "n_observed_points": len(points)}
    reference = points[0]
    if any(any(row["evaluation"][g]["n"] != reference["evaluation"][g]["n"] for g in _GROUPS)
           or row.get("selected_features") != reference.get("selected_features") for row in points[1:]):
        return {"status": "inconsistent_curve", "alpha_interpolated": None}
    strengths = np.asarray([0.] + [r["alpha"] for r in points], dtype=float)
    damages = np.asarray([0.] + [r["evaluation"]["target"]["delta_nll"] for r in points], dtype=float)
    if np.any(np.diff(damages) < 0):
        return {"status": "nonmonotonic_observed_damage", "alpha_interpolated": None,
                "n_observed_points": len(points)}
    if level > damages[-1]:
        return {"status": "unreachable_observed_damage", "alpha_interpolated": None,
                "n_observed_points": len(points), "observed_damage_max": float(damages[-1])}
    exact = np.flatnonzero(damages == level)
    if len(exact):
        lower = upper = int(exact[0])
        weight = 0.
    else:
        upper = int(np.flatnonzero(damages > level)[0])
        lower = upper - 1
        weight = float((level - damages[lower]) / (damages[upper] - damages[lower]))
    alpha = float(strengths[lower] + weight * (strengths[upper] - strengths[lower]))
    evaluation = {}
    for group in _GROUPS:
        evaluation[group] = {"n": reference["evaluation"][group]["n"]}
        for metric in _METRICS:
            values = [0.] + [r["evaluation"][group][metric] for r in points]
            evaluation[group][metric] = float(values[lower] + weight * (values[upper] - values[lower]))
    # Write the exact conditioning level rather than a roundoff-only mismatch.
    evaluation["target"]["delta_nll"] = float(level)
    return {
        "status": "observed_interpolated", "alpha_interpolated": alpha,
        "n_observed_points": len(points), "evaluation": evaluation,
        "observed_bracket_alpha": [float(strengths[lower]), float(strengths[upper])],
        "observed_bracket_target_damage": [float(damages[lower]), float(damages[upper])],
        "bracket_includes_zero_origin": bool(lower == 0),
    }


def analyze_observed_matching(removal_analysis, levels=DEFAULT_LEVELS, *,
                              bootstrap_samples=500, seed=0, min_observed_points=2, sae_arm="sae"):
    """Condition both arms on equal observed damage, using the paired intersection.

    ``sae_arm='sae_range'`` compares the auxiliary SAE coefficient range operator.
    The ``sae`` statistics key denotes the selected SAE operation, named by each
    comparison's explicit ``arm`` field. Returned matches retain every unreachable, nonmonotone, and insufficient
    curve as an explicit status. Comparisons resample decision families, keeping
    all sibling targets together. Interpolated per-target outcomes are treated
    as fixed within each bootstrap family; within-document sampling uncertainty
    and interpolation-model uncertainty are not estimated.
    """
    levels = tuple(sorted(set(float(level) for level in levels)))
    if not levels or not np.isfinite(levels).all() or min(levels) < 0:
        raise ValueError("levels must contain finite nonnegative values")
    if sae_arm not in ("sae", "sae_range"):
        raise ValueError("sae_arm must be sae or sae_range")
    grouped = defaultdict(list)
    for row in removal_analysis["records"]:
        if row["arm"] not in (sae_arm, "geoae_range"):
            continue
        grouped[(row["pair_id"], row["target"], row["arm"], row["feature_budget"])].append(row)
    matches, lookup = [], {}
    for (pair_id, target, arm, budget), rows in grouped.items():
        families = {r["family"] for r in rows}
        if len(families) != 1:
            raise ValueError("one intervention curve cannot span different families")
        family = next(iter(families))
        for level in levels:
            row = {"pair_id": pair_id, "target": target, "family": family, "arm": arm,
                   "feature_budget": budget, "level": level,
                   **interpolate_observed_curve(rows, level, min_observed_points=min_observed_points)}
            matches.append(row)
            if row["status"] == "observed_interpolated":
                lookup[(pair_id, target, arm, budget, level)] = row
    budgets = sorted({budget for _, _, arm, budget in grouped if arm == sae_arm})
    comparisons = []
    for budget in budgets:
        for level in levels:
            sae_rows = [r for r in matches if r["arm"] == sae_arm and r["feature_budget"] == budget
                        and r["level"] == level and r["status"] == "observed_interpolated"]
            paired = []
            for sae in sae_rows:
                geo = lookup.get((sae["pair_id"], sae["target"], "geoae_range", None, level))
                if geo is not None:
                    paired.append((sae, geo))
            item = {"arm": sae_arm, "feature_budget": budget, "level": level,
                    "n_paired_targets": len(paired), "n_families": len({s["family"] for s, _ in paired}),
                    "n_unpaired_sae_targets": len(sae_rows) - len(paired),
                    "sae": {}, "geoae_range": {}, "difference": {}}
            for group in _GROUPS:
                for metric in _METRICS:
                    values = {name: defaultdict(list) for name in ("sae", "geoae_range", "difference")}
                    for sae, geo in paired:
                        a, b = sae["evaluation"][group][metric], geo["evaluation"][group][metric]
                        values["sae"][sae["family"]].append(a)
                        values["geoae_range"][sae["family"]].append(b)
                        values["difference"][sae["family"]].append(a - b)
                    for name, family_values in values.items():
                        item[name][f"{group}_{metric}"] = family_bootstrap(family_values, bootstrap_samples, seed)
            comparisons.append(item)
    return {
        "metadata": {
            "analysis": "secondary descriptive interpolation conditional on equal observed held-out target damage",
            "sae_operation": sae_arm,
            "levels": list(levels), "level_declaration": "fixed reporting levels; not chosen to maximize observed effects",
            "matching_metric": "observed_heldout_next_token_delta_nll_nats",
            "alpha_selection_for_measured_points": "primary fit-only calibration",
            "strengths_rerun_at_observed_match": False, "independently_heldout_strength_calibration": False,
            "curve_rule": "strict alpha-ordered observed target-damage monotonicity; no extrapolation",
            "min_positive_alpha_observed_points": min_observed_points,
            "origin": "known zero intervention: target and collateral damage zero",
            "bootstrap_unit": "decision_family", "bootstrap_samples": bootstrap_samples, "seed": seed,
            "uncertainty_limit": "family resampling only; no within-document or interpolation-model uncertainty",
        },
        "matches": matches, "comparisons": comparisons,
        "status_counts": dict(Counter(r["status"] for r in matches)),
        "operator_diagnostic": sae_range_operator_diagnostic(removal_analysis) if sae_arm == "sae_range" else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("removal_json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sae_arm", choices=("sae", "sae_range"), default="sae")
    args = parser.parse_args()
    source = json.loads(Path(args.removal_json).read_text())
    analysis = analyze_observed_matching(source.get("analysis", source), sae_arm=args.sae_arm)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source": args.removal_json, "analysis": analysis}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
