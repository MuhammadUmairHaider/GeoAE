"""CPU checks for SugarCrepe++ distances without unit-vector normalization."""

import numpy as np
import pytest

from experiments.compositionality.analysis.sugarcrepe_geometry import normalized_euclidean_pairs
from experiments.compositionality.analysis.sugarcrepe_geometry import raw_euclidean_pairs


def test_raw_ranking_retains_vector_scale_and_differs_from_normalized_ranking():
    # The paraphrase points closer to the anchor but lies farther away in raw space.
    embeddings = np.array([[1, 0], [3, 4], [0, 2]], dtype=np.float32)
    indices = np.array([[0, 1, 2]])

    raw = raw_euclidean_pairs(embeddings, indices)
    normalized = normalized_euclidean_pairs(embeddings, indices)

    np.testing.assert_allclose(raw["d_paraphrase"], [np.sqrt(20)], rtol=1e-14)
    np.testing.assert_allclose(raw["d_distractor"], [np.sqrt(5)], rtol=1e-14)
    np.testing.assert_allclose(raw["d_paraphrase_distractor"], [np.sqrt(13)], rtol=1e-14)
    assert raw["pass"].tolist() == [False]
    assert normalized["pass"].tolist() == [True]
    for key in ("d_paraphrase", "d_distractor", "d_paraphrase_distractor"):
        assert raw[key].dtype == np.float64


def test_zero_vectors_are_valid_raw_embeddings():
    embeddings = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.float64)

    raw = raw_euclidean_pairs(embeddings, np.array([[0, 1, 2]]))

    np.testing.assert_array_equal(raw["d_paraphrase"], [1])
    np.testing.assert_array_equal(raw["d_distractor"], [2])
    assert raw["pass"].tolist() == [True]
    assert raw["strict_pass"].tolist() == [True]


def test_exact_raw_ties_count_as_inclusive_passes():
    embeddings = np.array([
        [0, 0], [3, 4], [-3, 4],
        [0, 0], [0, 0], [0, 0],
    ], dtype=np.float64)

    raw = raw_euclidean_pairs(embeddings, np.array([[0, 1, 2], [3, 4, 5]]))

    np.testing.assert_array_equal(raw["d_paraphrase"], [5, 0])
    np.testing.assert_array_equal(raw["d_distractor"], [5, 0])
    np.testing.assert_array_equal(raw["d_paraphrase_distractor"], [6, 0])
    for key in ("pass", "reverse_pass", "both_pass", "ties"):
        assert raw[key].tolist() == [True, True]
    for key in ("strict_pass", "strict_both_pass"):
        assert raw[key].tolist() == [False, False]


def test_raw_near_tie_has_no_epsilon_tolerance():
    embeddings = np.array([[0], [1], [np.nextafter(1.0, 0.0)]], dtype=np.float64)

    raw = raw_euclidean_pairs(embeddings, np.array([[0, 1, 2]]))

    assert raw["d_paraphrase"][0] > raw["d_distractor"][0]
    assert raw["pass"].tolist() == [False]
    assert raw["ties"].tolist() == [False]


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_raw_nonfinite_embeddings_are_rejected(nonfinite):
    embeddings = np.array([[0, 0], [1, 0], [nonfinite, 1]], dtype=np.float64)

    with pytest.raises(ValueError):
        raw_euclidean_pairs(embeddings, np.array([[0, 1, 2]]))
