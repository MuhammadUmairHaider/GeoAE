"""Focused control protocol: frozen documents, readout semantics, and calibration."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from geoae.interp.ssl_concept_controls import (
    Readout, calibrate_strength, data_fingerprint, direction_edit, document_means,
    fit_direction, run_experiment, validate_data,
)


class ImperfectAE:
    """A fixed token bypass and a missing residual component."""
    def token_bias(self, tok):
        return torch.stack((tok.float() % 2 * .4, tok.float() % 3 * .1), 1)

    def encode(self, x, tok):
        return (x - self.token_bias(tok))[:, :1]

    def decode(self, z, tok):
        return torch.cat((z, torch.zeros_like(z)), 1) + self.token_bias(tok)


class Head(torch.nn.Module):
    def forward(self, h):
        return torch.stack((h[:, 0], -h[:, 0], h[:, 1]), 1)


def fixture():
    h, tok, nxt, docs, labels, splits = [], [], [], [], [], []
    for part in ("fit", "calibration", "test"):
        for label in (1, 0, -1):
            for doc in range(3):
                for position in range(2):
                    center = 2. if label == 1 else -2. if label == 0 else 0.
                    h.append([center + .08 * (doc - 1) + .025 * position, .7])
                    tok.append(doc * 2 + position)
                    nxt.append(0 if label == 1 else 1 if label == 0 else 2)
                    docs.append(f"{part}/{label}/{doc}")
                    labels.append(label)
                    splits.append(part)
    return {"h": np.asarray(h, np.float32), "tok": np.asarray(tok, np.int64),
            "nxt": np.asarray(nxt, np.int64), "doc": np.asarray(docs),
            "pair": np.asarray(["tools/procedures"] * len(h)),
            "label": np.asarray(labels), "split": np.asarray(splits)}


def run(data, **kwargs):
    return run_experiment(data, ImperfectAE(), torch.zeros(2), torch.ones(2),
                          Readout(torch.nn.Identity(), Head(), chunk_size=8),
                          direction_kinds=("difference",), target_nll=(.1,),
                          alpha_grid=(0., .5, 1., 2.), bootstrap=20, **kwargs)


def test_range_operator_spares_sibling_and_zero_is_exact_identity():
    x = torch.tensor([[1.9, 2.], [2.1, 2.], [5.9, 2.], [6.1, 2.], [10., 2.]])
    fitted = fit_direction(x, np.arange(5), [1, 1, 0, 0, 0], [0, 0, 1, 1, 0],
                           [1, 1, 1, 1, 1])
    edited, changed = direction_edit(x, fitted, 1.)
    assert changed.tolist() == [True, True, False, False, False]
    assert torch.equal(edited[2:], x[2:])
    assert edited[:2, 0].tolist() == pytest.approx([6., 6.])
    all_edit, all_changed = direction_edit(x, fitted, 1., gated=False)
    assert all_changed.all()
    assert torch.equal(all_edit[:, 1], x[:, 1])
    zero, zero_changed = direction_edit(x, fitted, 0)
    assert zero is x
    assert not zero_changed.any()


def test_fit_uses_fit_only_and_equal_document_weight():
    # The many positions in document t0 must not dominate document t1.
    x = torch.tensor([[0., 0.]] * 10 + [[4., 0.], [6., 0.], [8., 0.], [999., 99.]])
    docs = np.asarray(["t0"] * 10 + ["t1", "s0", "s1", "test"])
    target = np.asarray([True] * 11 + [False, False, True])
    fit = np.arange(14) < 13
    sibling = ~target
    first = fit_direction(x, docs, target, sibling, fit)
    assert first["target_mean"] == pytest.approx(-2.)
    assert first["sibling_mean"] == pytest.approx(-7.)
    changed = x.clone()
    changed[-1] = torch.tensor([-9999., -9999.])
    second = fit_direction(changed, docs, target, sibling, fit)
    assert torch.equal(first["u"], second["u"])
    assert first["lo"] == second["lo"]
    assert first["hi"] == second["hi"]
    means, ids = document_means(torch.tensor([1., 3., 8.]), ["a", "a", "b"])
    assert ids.tolist() == ["a", "b"]
    assert means.tolist() == [2., 8.]


def test_shared_geometry_requires_real_fit_background():
    x = torch.tensor([[2., 1.], [3., 1.], [5., 1.], [6., 1.], [0., 0.], [0., 0.]])
    kwargs = dict(x=x, docs=np.arange(6), target=[1, 1, 0, 0, 0, 0],
                  sibling=[0, 0, 1, 1, 0, 0], fit=[1] * 6, kind="shared")
    with pytest.raises(ValueError, match="background"):
        fit_direction(**kwargs)
    fit = fit_direction(**kwargs, background=[0, 0, 0, 0, 1, 1])
    assert fit["u"][0] > .9
    assert fit["fit_sibling_gate_fraction"] == 0.


def test_global_document_split_leakage_and_ambiguous_memberships_rejected():
    data = fixture()
    validate_data(data)
    data["doc"][18] = data["doc"][0]
    with pytest.raises(ValueError, match="split leakage"):
        validate_data(data)
    data = fixture()
    data["doc"][6] = data["doc"][0]
    with pytest.raises(ValueError, match="ambiguous"):
        validate_data(data)


def test_calibration_uses_strength_order_never_clamps_or_damage_sorts():
    result = calibrate_strength([0., 1., 2.], [0., .2, .6], .4)
    assert result["alpha"] == pytest.approx(1.5)
    assert result["bracket"] == [1., 2.]
    assert calibrate_strength([0., 1., 2.], [0., .2, .6], .7)["status"] == "unreachable"
    nonmonotonic = calibrate_strength([0., 1., 2., 3.], [0., .3, .1, .6], .4)
    assert nonmonotonic["bracket"] == [2., 3.]
    assert not nonmonotonic["calibration_curve_monotone"]
    assert calibrate_strength([0., 1.], [0., np.nan], .1)["status"] == "nonfinite"
    with pytest.raises(ValueError, match="strictly increase"):
        calibrate_strength([0., 1., 1.], [0., .1, .2], .1)


def test_readout_has_fp32_nll_softcap_and_exact_untouched_zero():
    h = torch.tensor([[1., .4], [2., .3]], dtype=torch.float16)
    nxt = torch.tensor([0, 2])
    scorer = Readout(torch.nn.Identity(), Head(), chunk_size=1, logit_softcap=2.)
    logits = 2. * torch.tanh(Head()(h.float()) / 2.)
    expected = -torch.log_softmax(logits, -1).gather(1, nxt[:, None]).squeeze(1)
    assert torch.equal(scorer.nll(h, nxt), expected)
    edited = h.float().clone()
    edited[0, 0] -= .5
    delta = scorer.delta_nll(h, edited, nxt, torch.tensor([True, False]))
    assert delta[1] == 0.
    assert delta[0] == pytest.approx(float(scorer.nll(edited[:1], nxt[:1]) - expected[:1]))


def test_baseline_costs_both_target_directions_and_range_selectivity():
    result = run(fixture(), substrates=("base", "hb", "ae_recon", "ae_delta"))
    records = result["records"]
    assert {record["target_label"] for record in records} == {0, 1}
    matched = [record for record in records if record["status"] == "matched"]
    assert matched
    assert all(abs(record["calibration_attained"] - .1) <= .005 for record in matched)
    for record in matched:
        if record["arm"].endswith("/range"):
            assert record["evaluation"]["sibling"]["delta_nll"]["mean"] == 0.
            assert record["evaluation"]["random"]["delta_nll"]["mean"] == 0.
        if record["arm"].startswith(("base/", "hb/", "ae_delta/")):
            for group in ("target", "sibling", "random"):
                assert record["evaluation"][group]["total_delta_nll_vs_raw"] == record["evaluation"][group]["delta_nll"]
    recon_range = next(record for record in matched if record["arm"] == "ae_recon/difference/range")
    assert recon_range["evaluation"]["sibling"]["total_delta_nll_vs_raw"]["mean"] != 0.
    assert result["paired_comparisons"]
    assert all(record["test_target_match_status"] == "within_tolerance" for record in result["paired_comparisons"])
    json.dumps(result, allow_nan=False)


def test_test_data_never_selects_alpha_and_transfer_mismatch_is_explicit():
    first_data = fixture()
    first = run(first_data, substrates=("base", "ae_recon"))
    changed = {name: value.copy() for name, value in first_data.items()}
    changed["h"][changed["split"] == "test", 0] += 100.
    second = run(changed, substrates=("base", "ae_recon"))
    key = lambda record: (record["target_label"], record["arm"], record.get("alpha"), record["status"])
    assert list(map(key, first["records"])) == list(map(key, second["records"]))
    assert first["data_sha256"] != second["data_sha256"]
    assert any(record.get("test_match_status") == "unmatched_target_damage" for record in second["records"])
    # Fitted directions are also fixed before calibration/test inspection.
    assert [g["fit"] for g in first["geometry"]] == [g["fit"] for g in second["geometry"]]


def test_zero_strength_has_exact_local_identity_on_imperfect_ae_baseline():
    result = run_experiment(fixture(), ImperfectAE(), torch.zeros(2), torch.ones(2),
                            Readout(torch.nn.Identity(), Head()), target_nll=(0.,),
                            direction_kinds=("difference",), bootstrap=0)
    for record in result["records"]:
        assert record["alpha"] == 0.
        assert record["calibration_attained"] == 0.
        for group in ("target", "sibling", "random"):
            assert record["evaluation"][group]["delta_nll"]["mean"] == 0.
            assert record["evaluation"][group]["changed_fraction"]["mean"] == 0.
            assert record["evaluation"][group]["edit_l2"]["mean"] == 0.
        if record["arm"].startswith("ae_recon/"):
            assert record["evaluation"]["target"]["total_delta_nll_vs_raw"]["mean"] != 0.


def test_insufficient_documents_excluded_and_single_doc_uncertainty_not_fabricated():
    data = fixture()
    result = run(data, min_docs=4)
    assert not result["records"]
    assert len(result["excluded"]) == 2
    from geoae.interp.ssl_concept_controls import _summary
    summary = _summary(torch.tensor([1., 2., 3.]), ["one"] * 3, bootstrap=100)
    assert summary["ci95"] is None
    assert summary["n_docs"] == 1
    assert data_fingerprint(data) == data_fingerprint({key: value.copy() for key, value in data.items()})
