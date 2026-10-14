"""Observed matching is descriptive, monotone, bounded, and family-paired."""
from __future__ import annotations

import json

import pytest

from geoae.interp.sae_matched import analyze_observed_matching, interpolate_observed_curve, sae_range_operator_diagnostic


def point(alpha, target_damage, sibling_damage, *, target="a", family="family", arm="sae", budget=1):
    return dict(pair_id=target, target=target, family=family, arm=arm,
                feature_budget=budget if arm in ("sae", "sae_range") else None,
                status="matched", alpha=alpha, selected_features=[4] if arm in ("sae", "sae_range") else [],
                evaluation={
                    "target": {"n": 10, "delta_nll": target_damage, "kl": target_damage / 2},
                    "sibling": {"n": 8, "delta_nll": sibling_damage, "kl": sibling_damage / 2},
                    "random": {"n": 20, "delta_nll": sibling_damage / 4, "kl": sibling_damage / 8},
                })


def test_alpha_ordered_interpolation_of_every_outcome_at_equal_observed_damage():
    # Input order does not determine interpolation; alpha ordering does.
    result = interpolate_observed_curve([point(3, .08, .4), point(1, .02, .1)], .05)
    assert result["status"] == "observed_interpolated"
    assert result["alpha_interpolated"] == pytest.approx(2)
    assert result["evaluation"]["target"]["delta_nll"] == .05
    assert result["evaluation"]["sibling"]["delta_nll"] == pytest.approx(.25)
    assert result["evaluation"]["random"]["kl"] == pytest.approx(.03125)
    assert result["observed_bracket_alpha"] == [1., 3.]
    assert not result["bracket_includes_zero_origin"]
    origin = interpolate_observed_curve([point(1, .02, .1), point(3, .08, .4)], .01)
    assert origin["alpha_interpolated"] == pytest.approx(.5)
    assert origin["evaluation"]["sibling"]["delta_nll"] == pytest.approx(.05)
    assert origin["bracket_includes_zero_origin"]


def test_observed_matching_refuses_extrapolation_and_nonmonotonic_damage():
    rows = [point(1, .02, .1), point(3, .08, .4)]
    assert interpolate_observed_curve(rows, .1)["status"] == "unreachable_observed_damage"
    # Sorting by damage would hide this reversal and manufacture a matched case.
    rows = [point(1, .08, .1), point(3, .02, .4)]
    assert interpolate_observed_curve(rows, .01)["status"] == "nonmonotonic_observed_damage"
    # The physical zero origin prevents negative-damage points being accepted.
    rows = [point(1, -.01, .1), point(3, .08, .4)]
    assert interpolate_observed_curve(rows, .01)["status"] == "nonmonotonic_observed_damage"


def test_requires_two_distinct_measured_positive_strengths_and_consistent_samples():
    rows = [point(1, .02, .1)]
    assert interpolate_observed_curve(rows, .01)["status"] == "insufficient_observed_points"
    assert interpolate_observed_curve(rows * 2, .01)["status"] == "insufficient_observed_points"
    rows = [point(1, .02, .1), point(3, .08, .4)]
    rows[1]["evaluation"]["target"]["n"] = 11
    assert interpolate_observed_curve(rows, .01)["status"] == "inconsistent_curve"


def test_duplicate_alphas_cannot_create_conflicting_curves():
    rows = [point(1, .02, .1), point(1, .03, .1), point(3, .08, .4)]
    assert interpolate_observed_curve(rows, .01)["status"] == "inconsistent_duplicate_alpha"
    rows = [point(1, .02, .1), point(3, .08, .4)]
    rows[1]["selected_features"] = [5]
    assert interpolate_observed_curve(rows, .01)["status"] == "inconsistent_curve"


def test_pair_intersection_excludes_unreachable_other_arm_and_reports_statuses():
    rows = [point(1, .04, .2), point(2, .08, .4),
            point(1, .04, .1, arm="geoae_range"), point(2, .08, .2, arm="geoae_range"),
            point(1, .04, 100., target="unpaired"), point(2, .08, 200., target="unpaired"),
            point(1, .01, .1, target="unpaired", arm="geoae_range"),
            point(2, .02, .2, target="unpaired", arm="geoae_range")]
    result = analyze_observed_matching({"records": rows}, levels=(.05,), bootstrap_samples=0)
    comparison = result["comparisons"][0]
    assert comparison["n_paired_targets"] == 1
    assert comparison["n_unpaired_sae_targets"] == 1
    assert comparison["sae"]["target_delta_nll"]["mean"] == .05
    assert comparison["geoae_range"]["target_delta_nll"]["mean"] == .05
    assert comparison["sae"]["sibling_delta_nll"]["mean"] == pytest.approx(.25)
    assert comparison["geoae_range"]["sibling_delta_nll"]["mean"] == pytest.approx(.125)
    assert result["status_counts"]["unreachable_observed_damage"] == 1
    assert not result["metadata"]["strengths_rerun_at_observed_match"]
    assert not result["metadata"]["independently_heldout_strength_calibration"]
    json.dumps(result, allow_nan=False)


def test_bootstrap_groups_all_sibling_targets_and_weights_families_equally():
    rows = []
    for i in range(21):
        family = "many_siblings" if i < 20 else "other"
        value = 10. if i < 20 else 0.
        for alpha in (1, 2):
            rows.append(point(alpha, .05 * alpha, value * alpha,
                              target=str(i), family=family))
            rows.append(point(alpha, .05 * alpha, 0.,
                              target=str(i), family=family, arm="geoae_range"))
    result = analyze_observed_matching({"records": rows}, levels=(.05,), bootstrap_samples=1000)
    comparison = result["comparisons"][0]
    assert comparison["n_paired_targets"] == 21
    assert comparison["n_families"] == 2
    assert comparison["difference"]["sibling_delta_nll"]["mean"] == 5.
    assert comparison["difference"]["sibling_delta_nll"]["ci95"] == [0., 10.]


def test_single_family_has_no_confidence_interval_and_flat_curve_uses_first_hit():
    rows = [point(1, .05, .1), point(2, .05, .2)]
    result = interpolate_observed_curve(rows, .05)
    assert result["alpha_interpolated"] == 1
    assert result["evaluation"]["sibling"]["delta_nll"] == .1
    analysis = analyze_observed_matching({"records": rows + [
        point(1, .05, .1, arm="geoae_range"), point(2, .05, .2, arm="geoae_range")]},
        levels=(.05,), bootstrap_samples=100)
    assert analysis["comparisons"][0]["difference"]["sibling_delta_nll"]["ci95"] is None


def test_nonfinite_and_mixed_family_curves_rejected():
    rows = [point(1, .02, .1), point(3, .08, .4)]
    rows[1]["evaluation"]["sibling"]["delta_nll"] = float("nan")
    assert interpolate_observed_curve(rows, .01)["status"] == "nonfinite"
    rows = [point(1, .02, .1), point(3, .08, .4, family="different")]
    with pytest.raises(ValueError, match="families"):
        analyze_observed_matching({"records": rows})


def test_auxiliary_range_operator_is_compared_separately_from_feature_ablation():
    rows = []
    for alpha in (1, 2):
        rows.extend([point(alpha, .05 * alpha, .2 * alpha),
                     point(alpha, .05 * alpha, 0., arm="sae_range"),
                     point(alpha, .05 * alpha, .1 * alpha, arm="geoae_range")])
    primary = analyze_observed_matching({"records": rows}, levels=(.05,), bootstrap_samples=0)
    auxiliary = analyze_observed_matching({"records": rows}, levels=(.05,), bootstrap_samples=0, sae_arm="sae_range")
    assert primary["comparisons"][0]["arm"] == "sae"
    assert auxiliary["comparisons"][0]["arm"] == "sae_range"
    assert auxiliary["metadata"]["sae_operation"] == "sae_range"
    assert primary["comparisons"][0]["difference"]["sibling_delta_nll"]["mean"] == pytest.approx(.1)
    assert auxiliary["comparisons"][0]["difference"]["sibling_delta_nll"]["mean"] == pytest.approx(-.1)
    assert {r["arm"] for r in auxiliary["matches"]} == {"sae_range", "geoae_range"}


def test_operator_diagnostic_flags_zero_activation_without_changing_matching():
    rows = [point(1, .05, .1, arm="sae_range"), point(2, .1, .2, arm="sae_range")]
    for row in rows:
        row["sae_feature_range"] = {"lo": -1., "hi": 2., "control_mean": .5}
    diagnostic = sae_range_operator_diagnostic({"records": rows})
    assert diagnostic["n_targets"] == 1  # Two strengths are the same target.
    assert diagnostic["n_can_activate_inactive_coefficient"] == 1
    baseline = interpolate_observed_curve(rows, .05)
    for row in rows:
        row["sae_feature_range"] = {"lo": .1, "hi": 2., "control_mean": .5}
    diagnostic = sae_range_operator_diagnostic({"records": rows})
    assert diagnostic["n_can_activate_inactive_coefficient"] == 0
    assert interpolate_observed_curve(rows, .05) == baseline
