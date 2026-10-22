"""Synthetic frozen-SAE tests that check selection leakage and budget semantics."""
import json

import numpy as np
import pytest
import torch

from geoae.baselines.sae_coding import analyze_coding
from geoae.baselines.sae_coding import analyze_geoae_geometry
from geoae.baselines.sae_coding import reconstruction_report


def _coding_data(shared=True):
    # Two independent splits with five rows each of A, B, and random controls.
    labels = torch.tensor([0] * 5 + [1] * 5 + [2] * 5).repeat(2)
    memberships = {"a": labels == 0, "b": labels == 1}
    fit = torch.arange(30) < 15
    evaluation = ~fit
    controls = labels == 2
    offset = torch.tensor([-0.1, -0.05, 0.0, 0.05, 0.1]).repeat(6)
    z = torch.zeros(30, 3)
    if shared:
        z[:, 0] = torch.where(labels == 0, 1.0, torch.where(labels == 1, 3.0, 0.0)) + offset
    else:
        z[:, 0] = torch.where(labels == 0, 2.0, 0.0) + offset
        z[:, 1] = torch.where(labels == 1, 2.0, 0.0) + offset
    z[:, 2] = offset
    pairs = [{"a": "a", "b": "b", "family": "decision", "geoae_direction": 7}]
    return z, memberships, pairs, fit, evaluation, controls


def _analyze(data, directions=None, **kwargs):
    z, memberships, pairs, fit, evaluation, controls = data
    if directions is None:
        directions = torch.eye(3)
    return analyze_coding(z, memberships, pairs, directions, fit, evaluation,
                          controls=controls, budgets=(1, 2), **kwargs)


def test_shared_atom_can_separate_sibling_intensities_on_heldout_rows():
    report = _analyze(_coding_data())
    assert report["labels"]["a"]["feature_id"] == 0
    assert report["labels"]["b"]["feature_id"] == 0
    pair = report["pairs"][0]
    assert pair["feature_geometry"] == "same_feature"
    assert pair["interpretation"] == "same_feature_intensity_separation"
    assert pair["sibling_discrimination"]["orientation_fit"] == -1
    assert pair["sibling_discrimination"]["heldout"]["auc"] == 1.0
    family = report["families"][0]
    assert family["sae_union_atoms_by_handle_budget"]["1"] == 1
    assert family["geoae_direction_count"] == 1
    assert family["heldout_assignment_coverage"] == 1
    json.dumps(report, allow_nan=False)


def test_changing_only_eval_rows_cannot_change_selected_handles_or_orientation():
    data = _coding_data()
    baseline = _analyze(data)
    z, members, pairs, fit, evaluation, controls = data
    z[evaluation & members["a"], 1] = 1000.0
    z[evaluation & members["b"], 1] = 3000.0
    z[evaluation & members["a"], 0] = 20.0
    z[evaluation & members["b"], 0] = 2.0
    report = _analyze(data)
    for label in ("a", "b"):
        assert report["labels"][label]["feature_id"] == baseline["labels"][label]["feature_id"]
        assert report["labels"][label]["handles_by_budget"] == baseline["labels"][label]["handles_by_budget"]
    discrimination = report["pairs"][0]["sibling_discrimination"]
    assert discrimination["orientation_fit"] == -1
    assert discrimination["heldout"]["auc"] == 0.0
    assert not discrimination["heldout_separated"]


@pytest.mark.parametrize("second_direction, category", [([0.99, 0.1, 0.0], "distinct_parallel"),
                                                       ([0.0, 1.0, 0.0], "distinct_unrelated"),
                                                       ([-1.0, 0.0, 0.0], "distinct_unrelated")])
def test_distinct_feature_geometry_uses_signed_raw_decoder_cosine(second_direction, category):
    directions = torch.tensor([[1.0, 0.0, 0.0], second_direction, [0.0, 0.0, 1.0]])
    report = _analyze(_coding_data(shared=False), directions)
    pair = report["pairs"][0]
    assert (pair["feature_a"], pair["feature_b"]) == (0, 1)
    assert pair["feature_geometry"] == category
    assert pair["interpretation"] == category
    assert pair["sibling_discrimination"]["heldout"]["auc"] == 1.0
    assert report["families"][0]["sae_union_atoms_by_handle_budget"]["1"] == 2


def test_fit_association_without_eval_support_is_not_forced_into_category():
    data = _coding_data()
    z, members, pairs, fit, evaluation, controls = data
    z[evaluation, 0] = 0.0
    report = _analyze(data)
    assert report["pairs"][0]["feature_geometry"] == "same_feature"
    assert report["pairs"][0]["interpretation"] == "unresolved"
    assert report["labels"]["a"]["heldout"]["auc"] == 0.5
    assert report["families"][0]["heldout_assignment_coverage"] == 0.0


def test_distinct_atom_contrast_is_invariant_to_equivalent_feature_decoder_rescaling():
    baseline = _analyze(_coding_data(shared=False))
    data = _coding_data(shared=False)
    data[0][:, 1] *= 100.0
    directions = torch.eye(3)
    directions[1] /= 100.0
    rescaled = _analyze(data, directions)
    first = baseline["pairs"][0]["sibling_discrimination"]
    second = rescaled["pairs"][0]["sibling_discrimination"]
    assert first["score"] == "raw_contribution_norm_contrast"
    assert second["orientation_fit"] == first["orientation_fit"]
    assert second["heldout"]["auc"] == first["heldout"]["auc"]
    assert second["heldout"]["mean_gap"] == pytest.approx(first["heldout"]["mean_gap"])
    assert second["heldout"]["dprime"] == pytest.approx(first["heldout"]["dprime"], rel=1e-5)


def test_missing_fit_rows_and_zero_decoder_atom_are_unassigned():
    data = _coding_data()
    assert _analyze(data, min_fit_examples=6)["labels"]["a"]["feature_id"] is None
    directions = torch.eye(3)
    directions[0] = 0.0
    report = _analyze(data, directions)
    assert report["labels"]["a"]["feature_id"] is None
    assert report["pairs"][0]["interpretation"] == "unresolved"


def test_masks_accept_numpy_and_overlapping_members_exclude_pair_overlap():
    data = list(_coding_data())
    data[1]["a"][5] = True  # Fit overlap should be excluded from scalar discrimination.
    data[1]["a"][20] = True
    for index in (3, 4, 5):
        data[index] = data[index].numpy()
    data[1] = {label: mask.numpy() for label, mask in data[1].items()}
    report = _analyze(tuple(data))
    pair = report["pairs"][0]
    assert pair["overlapping_member_rows"] == 2
    assert pair["sibling_discrimination"]["fit"]["n_negative"] == 4
    assert pair["sibling_discrimination"]["heldout"]["n_negative"] == 4


def test_auroc_gives_half_credit_for_tied_scores():
    data = _coding_data()
    z, members, pairs, fit, evaluation, controls = data
    z[evaluation & members["a"], 0] = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    z[evaluation & controls, 0] = 0.0
    report = _analyze(data)
    assert report["labels"]["a"]["heldout"]["auc"] == pytest.approx(0.8)


def test_reject_split_leakage_nonfinite_values_and_nonboolean_masks():
    data = list(_coding_data())
    data[4] = data[3].clone()
    with pytest.raises(ValueError, match="disjoint"):
        _analyze(tuple(data))
    data = list(_coding_data())
    data[0][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        _analyze(tuple(data))
    data = list(_coding_data())
    data[3] = data[3].long()
    with pytest.raises(ValueError, match="boolean"):
        _analyze(tuple(data))


class _AffineSAE:
    decoder_directions = torch.tensor([[0.1, 0.0], [0.0, 10.0], [1.0, 0.0]])
    bias = torch.tensor([[3.0, 4.0]])
    parameter_count = 17

    def decode(self, z):
        return z @ self.decoder_directions + self.bias


def test_reconstruction_budget_ranks_contribution_norm_not_coefficient_alone():
    sae = _AffineSAE()
    z = torch.tensor([[20.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    h = sae.decode(z)
    report = reconstruction_report(h, z, sae, budgets=(1, 2, 20), batch_size=1)
    # Coefficient 20 contributes norm 2; coefficient 1 contributes norm 10.
    assert report["active_budget_reconstructions"][0]["metrics"]["all"]["raw_mse"] == 1.0
    assert report["active_budget_reconstructions"][1]["metrics"]["all"]["raw_mse"] == 0.0
    assert report["full_sae"]["all"]["raw_mse"] == 0.0
    assert report["observed_l0"]["all"] == 1.5
    assert report["active_budget_reconstructions"][0]["observed_retained_l0"]["all"] == 1.0
    assert report["active_budget_reconstructions"][2]["maximum_retained_atoms"] == 3
    assert report["dictionary"] == {"feature_atoms": 3, "decoder_scalar_parameters": 6,
                                    "total_scalar_parameters": 17}
    json.dumps(report, allow_nan=False)


def test_raw_mse_uses_one_control_reference_and_reports_geoae_separately():
    sae = _AffineSAE()
    z = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
    h = sae.decode(z)
    controls = np.array([True, True, False])
    report = reconstruction_report(h, z, sae, controls=controls, budgets=(1,),
                                   geoae_reconstruction=h + 1, batch_size=2)
    assert report["normalization"]["raw_variance_per_component"] == 0.5
    assert report["bias_only"]["controls"]["raw_mse"] == 1.0
    assert report["bias_only"]["noncontrols"]["raw_mse"] == 8.0
    assert report["bias_only"]["noncontrols"]["normalized_mse"] == 16.0
    assert report["supplied_geoae_full"]["all"]["raw_mse"] == 1.0
    assert report["supplied_geoae_full"]["all"]["normalized_mse"] == 2.0


def test_zero_reference_variance_produces_json_null_normalized_mse():
    sae = _AffineSAE()
    z = torch.zeros(2, 3)
    report = reconstruction_report(sae.decode(z), z, sae, budgets=(1,))
    assert report["full_sae"]["all"]["normalized_mse"] is None
    assert report["full_sae"]["noncontrols"]["raw_mse"] is None
    json.dumps(report, allow_nan=False)


def test_reconstruction_rejects_empty_controls_and_wrong_decoder_shape():
    sae = _AffineSAE()
    z = torch.zeros(2, 3)
    h = sae.decode(z)
    with pytest.raises(ValueError, match="at least one row"):
        reconstruction_report(h, z, sae, controls=torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="same number of rows"):
        reconstruction_report(h[:1], z, sae)


def test_matched_geoae_budgets_preserve_per_row_bias_and_report_unequal_dictionaries():
    sae = _AffineSAE()
    z = torch.tensor([[20.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    h = sae.decode(z)
    # A two-atom GeoAE uses the same contributions but includes row-dependent bypass.
    geo_z = torch.tensor([[2.0, 1.0], [0.0, 2.0]])
    geo_directions = torch.tensor([[1.0, 0.0], [0.0, 10.0]])
    geo_bias = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    geo_full = geo_z @ geo_directions + geo_bias
    report = reconstruction_report(
        h, z, sae, budgets=(1, 2), geoae_reconstruction=geo_full,
        geoae_features=geo_z, geoae_decoder_directions=geo_directions,
        geoae_bias=geo_bias, geoae_parameter_count=11, batch_size=1,
    )
    assert report["geoae_dictionary"] == {"feature_atoms": 2, "decoder_scalar_parameters": 4,
                                          "total_scalar_parameters": 11}
    assert report["dictionary"]["feature_atoms"] == 3
    assert report["geoae_observed_l0"]["all"] == 1.5
    assert report["geoae_bias_only"]["all"]["raw_mse"] == 108.0
    paired = report["matched_active_budget_reconstructions"]
    assert paired[0]["sae"]["metrics"]["all"]["raw_mse"] == 1.0
    assert paired[0]["geoae"]["metrics"]["all"]["raw_mse"] == 3.0
    assert paired[1]["geoae"]["metrics"]["all"]["raw_mse"] == 2.0
    assert report["geoae_affine_full"] == report["supplied_geoae_full"]
    json.dumps(report, allow_nan=False)


def test_geoae_global_bias_is_valid_and_partial_budget_inputs_are_rejected():
    sae = _AffineSAE()
    z = torch.tensor([[20.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    h = sae.decode(z)
    report = reconstruction_report(
        h, z, sae, budgets=(1, 3), geoae_features=z,
        geoae_decoder_directions=sae.decoder_directions, geoae_bias=sae.bias,
    )
    assert report["active_budget_reconstructions"] == report["geoae_active_budget_reconstructions"]
    with pytest.raises(ValueError, match="together"):
        reconstruction_report(h, z, sae, geoae_features=z)


def test_geoae_shared_latent_scalar_is_centered_and_selected_only_on_fit():
    z, members, pairs, fit, evaluation, controls = _coding_data()
    report = analyze_geoae_geometry(z, members, pairs, fit.numpy(), evaluation.numpy(), controls.numpy())
    pair = report["pairs"][0]
    assert report["space"] == "geoae_latent_euclidean"
    assert pair["fit_geo_shared_cos"] == pytest.approx(1.0)
    assert pair["meaningful_shared_scalar"]
    assert pair["orientation_fit"] == -1
    assert pair["eval_auc"] == 1.0
    assert pair["eval_dprime"] > 20
    assert pair["eval_2sd_interval_overlap"]["intersection_over_union"] == 0.0
    shifted = analyze_geoae_geometry(z + torch.tensor([20.0, 10.0, 30.0]), members, pairs,
                                     fit, evaluation, controls)["pairs"][0]
    assert shifted["fit_geo_shared_cos"] == pytest.approx(pair["fit_geo_shared_cos"])
    assert shifted["eval_dprime"] == pytest.approx(pair["eval_dprime"], rel=1e-5)
    z[evaluation & members["a"], 0] = 20.0
    z[evaluation & members["b"], 0] = 2.0
    reversed_pair = analyze_geoae_geometry(z, members, pairs, fit, evaluation, controls)["pairs"][0]
    assert reversed_pair["orientation_fit"] == -1
    assert reversed_pair["eval_auc"] == 0.0
    assert reversed_pair["fit_shared_direction"]
    json.dumps(report, allow_nan=False)


def test_geoae_nonparallel_and_degenerate_means_do_not_claim_shared_ranges():
    z, members, pairs, fit, evaluation, controls = _coding_data(shared=False)
    pair = analyze_geoae_geometry(z, members, pairs, fit, evaluation, controls)["pairs"][0]
    assert pair["fit_geo_shared_cos"] == pytest.approx(0.0, abs=1e-6)
    assert pair["status"] == "measured"
    assert not pair["meaningful_shared_scalar"]
    z[:] = 0.0
    z[members["a"], 0] = 1.0
    z[members["b"], 0] = -1.0
    pair = analyze_geoae_geometry(z, members, pairs, fit, evaluation, controls)["pairs"][0]
    assert pair["fit_geo_shared_cos"] == -1.0
    assert pair["status"] == "degenerate_shared_direction"
    assert pair["degenerate"] and pair["eval_auc"] is None
    z[members["a"]] = 0.0
    pair = analyze_geoae_geometry(z, members, pairs, fit, evaluation, controls)["pairs"][0]
    assert pair["status"] == "degenerate_centered_mean"
    assert not pair["meaningful_shared_scalar"]
