"""Removal protocol tests: leakage, matching, residual preservation, and grouping."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from geoae.interp.sae_removal import (
    calibrate_strength, family_bootstrap, fit_range, run_removal,
    score_edits, select_features, summarize_removal,
)


class Decoder(torch.nn.Module):
    """Deliberately imperfect reconstruction with a token bypass."""
    def decode(self, z, tok):
        return torch.cat((z, torch.zeros_like(z)), 1) + tok[:, None].float() * 7


class SAE:
    decoder_directions = torch.tensor([[1., 0.], [0., 1.]])


class Head(torch.nn.Module):
    def forward(self, h):
        return torch.stack((h[:, 0], -h[:, 0], h[:, 1]), 1)


def fixture():
    # Each split contains two samples for each of two siblings and controls.
    latent = torch.tensor([1.9, 2.1, 5.9, 6.1, -.1, .1] * 2)[:, None]
    h = torch.cat((latent + .25, torch.full_like(latent, .7)), 1)
    a = torch.tensor([True, True, False, False, False, False] * 2)
    b = torch.tensor([False, False, True, True, False, False] * 2)
    controls = torch.ones(12, dtype=torch.bool)  # Must be excluded by module.
    fit = torch.arange(12) < 6
    features = torch.cat((latent.clamp_min(0), torch.zeros_like(latent)), 1)
    return dict(h=h, z=latent, tok=torch.arange(12), nxt=torch.zeros(12, dtype=torch.long),
                memberships={"a": a, "b": b}, pairs=[{"a": "a", "b": "b", "family": "one"}],
                fit_mask=fit, eval_mask=~fit, controls=controls, ae=Decoder(),
                norm_mean=torch.zeros(2), norm_std=torch.ones(2), sae=SAE(),
                features=features, norm32=torch.nn.Identity(), head32=Head())


def test_select_features_uses_fit_only_and_excludes_absence_features():
    features = torch.tensor([[3., 1., 0.], [3., 1., 0.], [0., 2., 1.], [0., 2., 1.],
                             [0., 0., 1000.], [1000., 1000., 0.]])
    target = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
    control = ~target
    fit = torch.arange(6) < 4
    ids, scores = select_features(features, target, control, fit, 3)
    assert ids == [0]
    assert scores[0] > 0
    changed = features.clone()
    changed[4:] = torch.tensor([[10000., 10000., 10000.], [0., 0., 0.]])
    assert select_features(changed, target, control, fit, 3) == (ids, scores)


def test_range_fitting_uses_common_center_and_fit_only():
    args = fixture()
    controls = args["controls"] & ~args["memberships"]["a"] & ~args["memberships"]["b"]
    fit = fit_range(args["z"], args["memberships"]["a"], args["memberships"]["b"],
                    controls, args["fit_mask"])
    assert fit["mean"] == pytest.approx(2.)
    assert fit["lo"] == pytest.approx(1.8)
    assert fit["hi"] == pytest.approx(2.2)
    assert fit["fit_sibling_gate_fraction"] == 0
    assert fit["sibling_alignment"] == pytest.approx(1.)
    changed = args["z"].clone()
    changed[6:] = 10000
    second = fit_range(changed, args["memberships"]["a"], args["memberships"]["b"],
                       controls, args["fit_mask"])
    for key in ("lo", "hi", "mean", "sd"):
        assert second[key] == fit[key]


def test_calibration_no_extrapolation_or_damage_sorting():
    result = calibrate_strength([0., 1., 2.], [0., .2, .6], .4)
    assert result["status"] == "matched"
    assert result["alpha"] == pytest.approx(1.5)
    assert calibrate_strength([0., 1., 2.], [0., .2, .6], .7)["status"] == "unreachable"
    assert calibrate_strength([0., 1., 2.], [0., .3, .2], .1)["status"] == "nonmonotonic"
    assert calibrate_strength([0., 1., 2.], [0., 0., .6], 0)["alpha"] == 0
    assert calibrate_strength([0., 1.], [0., np.nan], .1)["status"] == "nonfinite"


def test_scoring_matches_nll_and_kl_and_unchanged_rows_are_exact_zero():
    h = torch.tensor([[1., .4], [2., .3]], dtype=torch.float16)
    labels = torch.tensor([0, 2])
    def edit(rows, clean, alpha):
        result = clean.clone()
        result[rows == 0, 0] -= alpha
        return result
    scored = score_edits(h, labels, [0, 1], edit, torch.nn.Identity(), Head(), [0., .5], 1)
    clean_lp = torch.log_softmax(Head()(h.float()), -1)
    edited = h.float().clone()
    edited[0, 0] -= .5
    edited_lp = torch.log_softmax(Head()(edited), -1)
    expected_delta = clean_lp[0, 0] - edited_lp[0, 0]
    expected_kl = (clean_lp[0].exp() * (clean_lp[0] - edited_lp[0])).sum()
    assert scored[.5]["delta_nll"][0] == pytest.approx(float(expected_delta))
    assert scored[.5]["kl"][0] == pytest.approx(float(expected_kl))
    assert scored[.5]["delta_nll"][1] == 0.
    assert scored[.5]["kl"][1] == 0.
    assert np.array_equal(scored[0.]["delta_nll"], [0., 0.])


def test_removal_retains_reconstruction_error_range_spares_sibling():
    args = fixture()
    original_controls = args["controls"].clone()
    result = run_removal(**args, feature_budgets=(1,), target_damage=(.1,),
                         strengths=(0., .5, 1., 2.), min_rows=2, max_rows_per_group=None,
                         bootstrap_samples=10)
    assert torch.equal(args["controls"], original_controls)
    assert result["metadata"]["calibration_reuses_feature_and_range_fit_rows"]
    rows = {r["arm"]: r for r in result["records"] if r["target"] == "a"}
    assert all(row["status"] == "matched" for row in rows.values())
    assert rows["geoae_range"]["evaluation"]["sibling"]["delta_nll"] == 0
    assert rows["geoae_range"]["evaluation"]["sibling"]["kl"] == 0
    assert rows["geoae_direction"]["evaluation"]["sibling"]["delta_nll"] > 0
    assert rows["sae"]["evaluation"]["sibling"]["delta_nll"] > 0
    # At alpha 1, the target first coordinate is .25: the AE's missing .25/.7
    # residual and huge token bypass both survive/cancel correctly.
    curve = next(c for c in result["curves"] if c["target"] == "a" and c["arm"] == "geoae_range")
    fit_h = args["h"][:2]
    edited_h = torch.tensor([[.25, .7], [.25, .7]])
    clean_lp = torch.log_softmax(Head()(fit_h), -1)
    edited_lp = torch.log_softmax(Head()(edited_h), -1)
    expected = (clean_lp[:, 0] - edited_lp[:, 0]).mean()
    assert curve["calibration_target_delta_nll"][2] == pytest.approx(float(expected), abs=2e-6)
    assert rows["geoae_range"]["calibration_attained"]["n"] == 2
    assert result["paired_comparisons"][0]["n_paired_targets"] == 2
    json.dumps(result, allow_nan=False)


def test_evaluation_damage_never_selects_matching_strength():
    args = fixture()
    first = run_removal(**args, feature_budgets=(1,), target_damage=(.1,),
                        min_rows=2, max_rows_per_group=None, bootstrap_samples=0)
    # Change only eval raw residuals, including next-token predictive evidence.
    args["h"] = args["h"].clone()
    args["h"][6:, 1] += 10
    second = run_removal(**args, feature_budgets=(1,), target_damage=(.1,),
                         min_rows=2, max_rows_per_group=None, bootstrap_samples=0)
    assert [(r["status"], r["alpha"]) for r in first["records"]] == [
        (r["status"], r["alpha"]) for r in second["records"]]
    a = next(r for r in first["records"] if r["status"] == "matched")
    b = next(r for r in second["records"] if r["status"] == "matched")
    assert a["evaluation"]["target"]["delta_nll"] != b["evaluation"]["target"]["delta_nll"]


def test_interpolation_is_refined_on_fit_and_failed_matches_are_excluded():
    kwargs = dict(feature_budgets=(1,), target_damage=(.1,), strengths=(0., 2.),
                  min_rows=2, max_rows_per_group=None, bootstrap_samples=0)
    refined = run_removal(**fixture(), **kwargs)
    row = next(r for r in refined["records"] if r["arm"] == "geoae_range" and r["target"] == "a")
    assert row["status"] == "matched"
    assert row["calibration_refinement_steps"] > 0
    assert abs(row["calibration_attained"]["delta_nll"] - .1) <= row["calibration_tolerance"]
    unrefined = run_removal(**fixture(), **kwargs, calibration_max_steps=0)
    row = next(r for r in unrefined["records"] if r["arm"] == "geoae_range" and r["target"] == "a")
    assert row["status"] == "calibration_unresolved"
    assert "evaluation" not in row


def test_auxiliary_sae_range_gate_preserves_error_and_spares_unchanged_sibling():
    kwargs = dict(feature_budgets=(1,), target_damage=(.1,), min_rows=2,
                  max_rows_per_group=None, bootstrap_samples=0, include_sae_range=True)
    args = fixture()
    result = run_removal(**args, **kwargs)
    row = next(r for r in result["records"] if r["arm"] == "sae_range" and r["target"] == "a")
    assert row["status"] == "matched"
    assert row["selected_features"] == [0]
    assert row["sae_feature_range"]["fit_sibling_gate_fraction"] == 0
    assert row["sae_feature_range"]["control_mean"] == pytest.approx(.05)
    assert row["evaluation"]["sibling"]["delta_nll"] == 0
    assert row["evaluation"]["sibling"]["kl"] == 0
    assert row["evaluation"]["random"]["kl"] == 0
    curve = next(r for r in result["curves"] if r["arm"] == "sae_range" and r["target"] == "a")
    clean_lp = torch.log_softmax(Head()(args["h"][:2]), -1)
    # Original .25/.7 reconstruction error remains; target coefficient moves
    # toward .05 rather than zero, and no SAE reconstruction is substituted.
    edited_lp = torch.log_softmax(Head()(torch.tensor([[.30, .7], [.30, .7]])), -1)
    assert curve["calibration_target_delta_nll"][3] == pytest.approx(
        float((clean_lp[:, 0] - edited_lp[:, 0]).mean()), abs=1e-6)
    changed = fixture()
    changed["features"][6:, 0] += 10
    second = run_removal(**changed, **kwargs)
    second_row = next(r for r in second["records"] if r["arm"] == "sae_range" and r["target"] == "a")
    assert second_row["sae_feature_range"] == row["sae_feature_range"]
    assert second_row["alpha"] == row["alpha"]
    assert second_row["evaluation"]["target"]["delta_nll"] == 0


def test_overlapping_splits_rejected_and_missing_pairs_reported():
    args = fixture()
    args["eval_mask"] = torch.ones(12, dtype=torch.bool)
    with pytest.raises(ValueError, match="overlap"):
        run_removal(**args)
    args = fixture()
    args["pairs"] = [{"a": "a", "b": "missing"}]
    result = run_removal(**args, min_rows=2, bootstrap_samples=0)
    assert result["excluded_pairs"][0]["reason"] == "missing_membership"
    assert result["records"] == []


def test_family_bootstrap_keeps_sibling_dependence_equal_family_weight():
    # Twenty highly dependent targets still count as one family, not twenty.
    result = family_bootstrap({"large": [10.] * 20, "small": [0.]}, samples=1000, seed=2)
    assert result["n_families"] == 2
    assert result["mean"] == 5
    assert result["ci95"] == [0., 10.]


def test_paired_summary_uses_only_intersection_of_reachable_targets():
    def row(target, arm, family, damage):
        return dict(status="matched", target=target, pair_id=target, family=family,
                    arm=arm, feature_budget=1 if arm == "sae" else None,
                    requested_damage=.1,
                    evaluation={group: {"delta_nll": damage, "kl": damage}
                                for group in ("target", "sibling", "random")})
    result = summarize_removal([row("a", "sae", "family1", .4),
                               row("b", "sae", "family2", 100.),
                               row("a", "geoae_range", "family1", .1)], bootstrap_samples=0)
    paired = result["paired_comparisons"][0]
    assert paired["n_paired_targets"] == 1
    assert paired["n_unpaired_sae_targets"] == 1
    assert paired["sibling_delta_nll"]["mean"] == pytest.approx(.3)
