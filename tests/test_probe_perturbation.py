"""Regression checks for the literal k50 input-coordinate ablation metric."""
import json
import sys

import numpy as np
import pytest
import torch

from geoae.interp import probe_perturbation as tpp


def test_literal_half_drop_and_collateral_at_crossing():
    r = tpp.summarize_curve([0, 2, 4, 6], [.8, .6, .4, .2], [.9, .8, .7, .1], 6)
    assert r["threshold"] == .4
    assert r["k_at_50pct_drop"] == 4
    assert r["k_at_50pct_drop_frac"] == pytest.approx(2/3)
    assert r["previous_tested_k"] == 2
    assert r["complement_drop_at_k50"] == pytest.approx(.2)
    assert r["selectivity"] == pytest.approx(.2)
    # Full ablation instead gives (.8-.2)-(.9-.1) = -.2: a different quantity.


@pytest.mark.parametrize("target,status", [([.8, .7, .6], "not_reached"),
                                           ([0., 0., 0.], "undefined_baseline"),
                                           ([None, None, None], "undefined_baseline")])
def test_unreached_and_invalid_baselines_are_not_full_removal(target, status):
    r = tpp.summarize_curve([0, 5, 10], target, [.9, .6, .2], 10)
    assert r["k_at_50pct_drop"] is None
    assert r["k_at_50pct_drop_frac"] is None
    assert r["selectivity"] is None
    assert r["threshold_status"] == status
    json.dumps(r, allow_nan=False)


def test_first_sampled_crossing_when_curve_recovers():
    r = tpp.summarize_curve([0, 1, 2, 3], [1., .4, .9, .2], [1., .9, .8, .1], 3)
    assert r["k_at_50pct_drop"] == 1


def test_class_ranking_is_softmax_offset_invariant():
    W = np.array([[10., 2., 1.], [10., -2., 0.], [10., 0., -1.]])
    common = np.array([900., -40., 77.])
    np.testing.assert_array_equal(tpp.rank_dims_global(W), tpp.rank_dims_global(W+common))
    for c in range(3):
        ranking = tpp.rank_dims_per_class(W, c)
        np.testing.assert_array_equal(ranking, tpp.rank_dims_per_class(W+common, c))
        assert ranking[0] != 0  # zero-effect coordinates can tie at the bottom


@pytest.mark.parametrize("bias", [True, False])
def test_incremental_sweep_equals_direct_ablation(bias):
    rng = np.random.RandomState(22)
    torch.manual_seed(22)
    X = rng.randn(90, 9).astype(np.float32)
    y = np.tile(np.arange(3), 30)
    probe = tpp.LinearProbe(9, 3, bias=bias)
    ranking = rng.permutation(9)
    r = tpp.perturbation_sweep(X, y, probe, ranking, 1, n_steps=9)
    target, complement = [], []
    for k in r["ks"]:
        changed = torch.from_numpy(X.copy())
        changed[:, ranking[:k]] = 0
        correct = probe(changed).argmax(1).numpy() == y
        target.append(float(correct[y == 1].mean()))
        complement.append(float(correct[y != 1].mean()))
    np.testing.assert_allclose(r["tgt_accs"], target, atol=1e-7)
    np.testing.assert_allclose(r["comp_accs"], complement, atol=1e-7)
    expected = tpp.summarize_curve(r["ks"], target, complement, 9)
    assert r["k_at_50pct_drop"] == expected["k_at_50pct_drop"]


def test_validation_budget_is_independent_of_test_curve():
    val = tpp.summarize_curve([0, 2, 4], [1., .8, .4], [1., .9, .8], 4)
    test = tpp.summarize_curve([0, 2, 4], [.8, .3, .2], [.9, .8, .6], 4)
    r = tpp.validation_operating_point(val, test)
    assert r["k"] == 4  # test's own descriptive k50 is 2
    assert r["test"]["complement_drop"] == pytest.approx(.3)
    test["tgt_accs"] = [.9, .8, .7]
    assert tpp.validation_operating_point(val, test)["k"] == 4


def test_random_threshold_of_mean_is_not_mean_trial_threshold(monkeypatch):
    first = tpp.summarize_curve([0, 1, 2], [1., 0., 0.], [1., .8, .3], 2)
    second = tpp.summarize_curve([0, 1, 2], [1., 1., 0.], [1., .8, .3], 2)
    runs = iter([first, second])
    monkeypatch.setattr(tpp, "perturbation_sweep", lambda *a, **kw: next(runs))
    r = tpp.random_perturbation_sweep(np.zeros((2, 2)), np.array([0, 1]), None, 0, n_trials=2)
    assert r["k_at_50pct_drop"] == 1
    assert r["median_trial_k50"] == 1.5
    assert r["trial_k50"] == [1, 2]
    runs = iter([first, tpp.summarize_curve([0, 1, 2], [1., .8, .7], [1., .8, .3], 2)])
    r = tpp.random_perturbation_sweep(np.zeros((2, 2)), np.array([0, 1]), None, 0, n_trials=2)
    assert r["median_trial_k50"] is None
    assert r["n_trials_reached"] == 1


def test_scaling_ignores_validation_and_test_outliers():
    X = np.array([[1., 2.], [3., 2.], [10000., -50.]], dtype=np.float32)
    tr, te = tpp.transform_raw(X, X[2:], np.array([0, 1]))
    np.testing.assert_allclose(tr[:2], [[-1., 0.], [1., 0.]])


@pytest.mark.parametrize("ranking", ["contrast", "dprime"])
def test_cli_uses_common_cohort_and_preserves_missing_baselines(tmp_path, monkeypatch, ranking):
    rng = np.random.RandomState(3)
    for split, n in [("train", 160), ("test", 40)]:
        y = np.tile([0, 0, 3, 3], n//4)
        X = rng.randn(n, 6).astype(np.float32)
        X[:, 0] += 2 * (y == 3)
        np.save(tmp_path / ("layer_27.npy" if split == "train" else "layer_27_test.npy"), X)
        np.save(tmp_path / f"labels_{split}.npy", y)
    (tmp_path / "meta.json").write_text(json.dumps({"layer": 27, "model": "tiny",
                                                  "classes": ["a", "b", "c", "d"]}))
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["tpp", "--act_dir", str(tmp_path), "--spaces", "raw",
        "--device", "cpu", "--probe_epochs", "3", "--n_steps", "6", "--n_random_trials", "1",
        "--out", str(out), "--plot_dir", str(tmp_path / "plots"), "--ranking", ranking])
    tpp.main()
    r = json.loads(out.read_text())
    assert r["status"] == "complete"
    assert r["spaces"]["raw"]["n_test"] == 40
    for name in ["b", "c"]:
        concept = r["spaces"]["raw"]["concepts"][name]
        if ranking == "contrast":
            assert concept["importance"]["threshold_status"] == "undefined_baseline"
        else:
            assert concept["status"] == "missing_fitting_group"
    assert (tmp_path / "plots" / "k50_and_curves.png").exists()


def test_nonfinite_cache_is_not_silently_clamped(tmp_path):
    for suffix in ["", "_test"]: np.save(tmp_path/f"layer_27{suffix}.npy", [[np.inf, 1.]])
    for split in ["train", "test"]: np.save(tmp_path/f"labels_{split}.npy", [0])
    with pytest.raises(ValueError, match="non-finite"):
        tpp.load_activations(tmp_path, 27)


def test_joint_correct_mapping_validates_source_order_and_strips_text(tmp_path, monkeypatch):
    import datasets
    source = datasets.Dataset.from_dict({"content": [" A ", "B", " C ", "D"], "label": [0, 1, 2, 3]})
    ordered = source.shuffle(seed=7)
    np.save(tmp_path / "labels_test.npy", np.asarray(ordered["label"]))
    (tmp_path / "meta.json").write_text(json.dumps({"seed": 7, "n_test": 4}))
    jc = tmp_path / "joint.json"
    jc.write_text(json.dumps({"docs": [{"text": "  B  "}]}))
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: {"test": source})
    indices = tpp.load_joint_correct_indices(str(jc), tmp_path)
    assert indices.tolist() == [list(ordered["content"]).index("B")]
    np.save(tmp_path / "labels_test.npy", np.zeros(4, dtype=np.int64))
    with pytest.raises(ValueError, match="ordering"):
        tpp.load_joint_correct_indices(str(jc), tmp_path)
