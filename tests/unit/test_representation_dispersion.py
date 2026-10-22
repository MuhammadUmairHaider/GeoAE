"""Compare streaming paper metrics against explicit small cosine matrices."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

from geoae.evaluation.dispersion import ClusterStatistics
from geoae.evaluation.dispersion import dispersion_metrics
from geoae.evaluation.dispersion import vendi_score


def brute_force(vectors, labels):
    vectors, labels = np.asarray(vectors), np.asarray(labels)
    distances = cdist(vectors, vectors, metric="cosine")
    off_diagonal = ~np.eye(len(vectors), dtype=bool)
    within_pairs = off_diagonal & (labels[:, None] == labels[None, :])
    between_pairs = off_diagonal & (labels[:, None] != labels[None, :])
    occupied = np.unique(labels)
    centroids = np.stack([vectors[labels == c].mean(0) for c in occupied])
    per_cluster = np.array([
        cdist(vectors[labels == c], centroids[i:i + 1], metric="cosine").mean()
        for i, c in enumerate(occupied)
    ])
    centroid_distances = cdist(centroids, centroids, metric="cosine")
    return {
        "dispersion": distances[off_diagonal].mean() if off_diagonal.any() else None,
        "within": per_cluster.mean(),
        "within_sample_weighted": np.average(per_cluster, weights=[sum(labels == c) for c in occupied]),
        "between": centroid_distances[~np.eye(len(occupied), dtype=bool)].mean() if len(occupied) > 1 else None,
        "pairwise_within": distances[within_pairs].mean() if within_pairs.any() else None,
        "pairwise_between": distances[between_pairs].mean() if between_pairs.any() else None,
    }


@pytest.mark.parametrize("centred", [False, True])
def test_paper_and_pair_metrics_match_explicit_cosine(centred):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(12, 5)) * np.arange(1, 13)[:, None] + 2
    labels = np.array([0] * 2 + [2] * 3 + [6] * 7)
    folds = np.arange(len(x)) % 3
    stat = ClusterStatistics(8, 5, 3)
    for start, stop in ((0, 4), (4, 9), (9, 12)):
        stat.add_vectors(x[start:stop], labels[start:stop], folds[start:stop])
    mean = stat.mean() if centred else None
    y = torch.tensor(x) - mean if centred else torch.tensor(x)
    for start, stop in ((0, 3), (3, 10), (10, 12)):
        stat.add_units(y[start:stop], labels[start:stop], folds[start:stop], "test")
    for fold in (None, 0, 1, 2):
        mask = np.ones(len(x), dtype=bool) if fold is None else folds == fold
        expected = brute_force(y[mask].numpy(), labels[mask])
        actual = stat.metrics("test", fold=fold, centre=mean)
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, abs=2e-12) if value is not None else actual[key] is None
    full = stat.metrics("test", centre=mean)
    assert full["nonempty_clusters"] == 3
    assert full["empty_clusters"] == 5
    assert full["within"] != pytest.approx(full["within_sample_weighted"])
    # The old row-pair partition has a different definition than the paper.
    assert full["within"] != pytest.approx(full["pairwise_within"])
    json.dumps(full, allow_nan=False)


def test_batch_additivity_and_independent_cluster_ranges():
    x = torch.tensor([[1., 2.], [3., 1.], [-1., 4.], [2., 7.]], dtype=torch.float64)
    labels = torch.tensor([0, 7, 7, 2])
    whole, split = ClusterStatistics(9, 2), ClusterStatistics(9, 2)
    whole.add(x, labels)
    for index in range(len(x)):
        split.add(x[index:index + 1], labels[index:index + 1])
    for key, value in whole.metrics().items():
        assert split.metrics()[key] == pytest.approx(value) if isinstance(value, float) else split.metrics()[key] == value
    with pytest.raises(ValueError, match="cluster range"):
        ClusterStatistics(3, 2).add(x, labels)


def test_empty_singleton_and_one_cluster_null_denominators():
    empty = ClusterStatistics(4, 2, 2).metrics()
    assert empty["rows"] == 0
    assert all(empty[name] is None for name in ("dispersion", "within", "between", "pairwise_within", "pairwise_between"))
    single = ClusterStatistics(4, 2, 2)
    single.add([[1., 2.]], torch.tensor([2]))
    result = single.metrics()
    assert result["within"] == pytest.approx(0, abs=1e-12)
    assert result["dispersion"] is None
    assert result["between"] is None
    assert result["between_over_within"] is None
    assert result["singleton_clusters"] == 1
    assert single.metrics(fold=1)["rows"] == 0
    one_cluster = ClusterStatistics(1, 2)
    one_cluster.add([[1., 0.], [0., 1.]], torch.tensor([0, 0]))
    assert one_cluster.metrics()["dispersion"] == pytest.approx(1)
    assert one_cluster.metrics()["pairwise_between"] is None
    json.dumps(empty, allow_nan=False)
    json.dumps(result, allow_nan=False)


def test_zero_empirical_centroid_does_not_silently_drop_group():
    stat = ClusterStatistics(3, 2)
    stat.add([[1., 0.], [-1., 0.], [0., 2.]], torch.tensor([0, 0, 1]))
    result = stat.metrics()
    assert result["zero_centroid_clusters"] == 1
    assert not result["centroid_metrics_defined"]
    assert all(result[key] is None for key in ("within", "within_sample_weighted", "between", "between_over_within"))
    assert result["dispersion"] == pytest.approx(4 / 3)
    assert result["pairwise_within"] == pytest.approx(2)
    assert result["pairwise_between"] == pytest.approx(1)


@pytest.mark.parametrize("row, message", [([0., 0.], "zero"), ([float("nan"), 1.], "nonfinite"), ([float("inf"), 1.], "nonfinite")])
def test_invalid_rows_reject_cosine(row, message):
    stat = ClusterStatistics(1, 2)
    with pytest.raises(ValueError, match=message):
        stat.add([row], torch.tensor([0]))
    assert stat.counts.sum() == 0


def test_centred_degenerate_row_is_explicit_error():
    stat = ClusterStatistics(1, 2)
    vectors = torch.tensor([[1., 2.], [1., 2.]])
    stat.add_vectors(vectors, torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="zero representation rows in centred"):
        stat.add_units(vectors.double() - stat.mean(), torch.tensor([0, 0]), variant="centred")


def test_two_pass_count_mismatch_and_invalid_counts_reject():
    stat = ClusterStatistics(2, 2)
    stat.add_vectors([[1., 1.]], torch.tensor([0]))
    stat.add_units([[1., 1.]], torch.tensor([1]))
    with pytest.raises(ValueError, match="do not match"):
        stat.metrics()
    with pytest.raises(ValueError, match="counts must be integers"):
        dispersion_metrics(torch.ones(1, 2), torch.ones(1, 2), torch.tensor([1.5]), torch.ones(1))


def test_vendi_rank_and_input_guards():
    assert vendi_score(torch.eye(3)) == pytest.approx(3)
    assert vendi_score([[1., 0.], [-1., 0.], [4., 0.]]) == pytest.approx(1)
    assert vendi_score(torch.empty(0, 2)) is None
    with pytest.raises(ValueError, match="zero"):
        vendi_score([[0., 0.]])
    with pytest.raises(ValueError, match="limited"):
        vendi_score(torch.eye(3), max_rows=2)


def _wrapper():
    path = Path(__file__).resolve().parents[2] / "experiments/paper/analysis/paper_dispersion.py"
    spec = importlib.util.spec_from_file_location("paper_dispersion_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_import_safe_and_sampling_document_overlap():
    module = _wrapper()  # Import must neither parse pytest's args nor touch model/dump files.
    args = module.parse_args([])
    assert args.km_tok is None and args.device == "auto" and args.max_rows == 0
    assert args.out.endswith("dispersion_v2.json")
    docs = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    full, meta = module.select_rows(docs, 4, 0, 42)
    assert full.tolist() == list(range(5, 12))
    assert meta["excluded_train_overlap_rows"] == 1
    assert meta["excluded_train_overlap_documents"] == 1
    first, _ = module.select_rows(docs, 4, 4, 42)
    again, _ = module.select_rows(docs, 4, 4, 42)
    assert np.array_equal(first, again)
    assert np.array_equal(first, np.sort(first)) and set(first).issubset(full)
    latest = module.latest_sampled_rows(np.array([5, 3, 5, 3, 4]), np.array([2, 8, 9, 1, 5]))
    assert latest.tolist() == [1, 2, 4]


@pytest.mark.parametrize("argv", [["--batch", "0"], ["--folds", "0"], ["--val_frac", "1"],
                                   ["--max_rows", "-1"], ["--vendi_rows", "2049"], ["--seed", "-1"]])
def test_cli_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        _wrapper().parse_args(argv)


def test_checkpoint_layer_and_holdout_fraction_guards():
    module = _wrapper()
    checkpoint = {"config": {"data": {"target_layer": 27, "val_frac": 0.05}}}
    module.validate_checkpoint_data(checkpoint, 27, 0.05)
    module.validate_checkpoint_data({}, 27, 0.05)  # Older checkpoints may omit metadata.
    with pytest.raises(ValueError, match="checkpoint layer"):
        module.validate_checkpoint_data(checkpoint, 14, 0.05)
    with pytest.raises(ValueError, match="training split"):
        module.validate_checkpoint_data(checkpoint, 27, 0.1)
