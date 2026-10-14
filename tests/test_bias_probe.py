"""Regression tests for BiasBios split, selection, and ablation failures."""
import copy
import json
import sys

import numpy as np
import pytest
import torch

from geoae.bias import probe as bp


def test_deduplication_preserves_alignment_and_prevents_shared_texts():
    train = np.array(["b", "a", "b", "c"])
    test = np.array(["a", "d", "d", "e"])
    ti, te = bp.deduplicate_indices(train, test)
    assert ti.tolist() == [0, 1, 3]
    assert te.tolist() == [1, 3]
    assert not set(train[ti]) & set(test[te])


def test_split_is_shared_despite_global_rng_consumption():
    p = np.tile(np.arange(4), 30)
    g = np.tile([0, 1], 60)
    a, b, method = bp.shared_split(p, g, 42)
    np.random.rand(1000)
    c, d, _ = bp.shared_split(p, g, 42)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert not set(a) & set(b)
    assert method == "profession_gender"


def test_scaling_uses_only_probe_training_rows():
    train = np.array([[1., 5.], [3., 5.], [1000., -20.]], dtype=np.float32)
    test = np.array([[-500., 20.]], dtype=np.float32)
    scaled, _ = bp.standardize([train, test], np.array([0, 1]))
    np.testing.assert_allclose(scaled[:2], [[-1., 0.], [1., 0.]])
    train[2] = -10000
    changed, _ = bp.standardize([train, test * 20], np.array([0, 1]))
    np.testing.assert_array_equal(changed[:2], scaled[:2])


def test_ranking_ignores_softmax_common_weight_offset():
    weights = np.array([[100., 1., 0.], [100., -1., .5]])
    expected = np.array([1, 2, 0])
    np.testing.assert_array_equal(bp.rank_dims_global(weights), expected)
    np.testing.assert_array_equal(bp.rank_dims_global(weights + [1000., 20., 30.]), expected)


@pytest.mark.parametrize("bias", [True, False])
def test_incremental_sweep_matches_direct_coordinate_ablation(bias):
    torch.manual_seed(12)
    rng = np.random.RandomState(12)
    X = rng.randn(81, 13).astype(np.float32)
    task = rng.randint(0, 3, len(X))
    gender = np.array([0] * 61 + [1] * 20)
    pt = bp.LinearProbe(13, 3, bias=bias)
    pg = bp.LinearProbe(13, 2, bias=bias)
    ranking = rng.permutation(13)
    curve = bp.cross_perturbation_sweep(X, task, pt, gender, pg, ranking, n_steps=13)
    for i, k in enumerate(curve["ks"]):
        changed = torch.from_numpy(X.copy())
        changed[:, ranking[:k]] = 0
        for prefix, probe, labels, n in [("task", pt, task, 3), ("spur", pg, gender, 2)]:
            metric = bp.classification_metrics(probe(changed), torch.from_numpy(labels), n)
            assert curve[f"{prefix}_accs"][i] == pytest.approx(metric["accuracy"])
            assert curve[f"{prefix}_balanced_accs"][i] == pytest.approx(metric["balanced_accuracy"])
    assert curve["spur_balanced_accs"][-1] == .5
    assert curve["spur_accs"][-1] != .5  # constant prediction on imbalanced data


def test_threshold_selection_uses_validation_and_reports_unreached():
    val = {"ks": [0, 1, 2, 3], "n_dims": 3,
           "spur_balanced_accs": [.98, .8, .7, .6],
           "task_accs": [.8, .7, .6, .5], "task_balanced_accs": [.75, .65, .55, .45]}
    test = copy.deepcopy(val)
    test["spur_balanced_accs"] = [.99, .5, .49, .5]
    points = bp.select_operating_points(val, test)
    assert points["half_excess"]["k"] == 2  # halfway from .98 to .5 is .74, not .49
    assert points["near_chance"] == {"reached_on_validation": False, "k": None, "test": None}
    test["spur_balanced_accs"] = [.51, .51, .51, .51]
    assert bp.select_operating_points(val, test)["half_excess"]["k"] == 2


def test_below_chance_is_not_called_information_erasure():
    curve = {"ks": [0, 1], "n_dims": 1, "spur_balanced_accs": [.95, .1],
             "task_accs": [.8, .7], "task_balanced_accs": [.8, .7]}
    assert not bp.select_operating_points(curve, curve)["near_chance"]["reached_on_validation"]


def test_full_raw_cli_with_noncontiguous_profession_labels(tmp_path, monkeypatch):
    rng = np.random.RandomState(7)
    for split, n in [("train", 160), ("test", 40)]:
        p = np.tile([0, 0, 27, 27], n // 4)
        g = np.tile([0, 1, 0, 1], n // 4)
        X = rng.randn(n, 8).astype(np.float32)
        X[:, 0] += 3 * g
        X[:, 1] += 3 * (p == 27)
        np.save(tmp_path / ("layer_27.npy" if split == "train" else "layer_27_test.npy"), X)
        np.save(tmp_path / f"labels_profession_{split}.npy", p)
        np.save(tmp_path / f"labels_gender_{split}.npy", g)
    (tmp_path / "meta.json").write_text(json.dumps({"layer": 27, "model": "tiny",
                                                  "n_train": 160, "n_test": 40}))
    np.savez(tmp_path / "hashes.npz", train=np.array([f"train{i}" for i in range(160)]),
             test=np.array([f"test{i}" for i in range(40)]))
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["bias-probe", "--act_dir", str(tmp_path),
        "--source_hashes", str(tmp_path / "hashes.npz"), "--spaces", "raw", "--device", "cpu",
        "--probe_epochs", "3", "--n_steps", "8", "--n_random_trials", "1", "--out", str(output)])
    bp.main()
    result = json.loads(output.read_text())
    assert result["status"] == "complete"
    assert result["schema_version"] == 2
    assert result["spaces"]["raw"]["n_dims"] == 8
    split = result["split"]
    assert not set(split["fit_source_rows"]) & set(split["validation_source_rows"])
    for arm in result["spaces"]["raw"]["ablations"].values():
        assert arm["test"]["spur_balanced_accs"][-1] == .5
        for point in arm["operating_points"].values():
            assert point["k"] is None or point["k"] in arm["validation"]["ks"]
