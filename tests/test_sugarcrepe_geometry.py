"""CPU checks for the SugarCrepe++ caption-triplet geometry metric."""

import numpy as np
import pytest

from eval_out.sugarcrepe_geometry import build_text_index, normalized_euclidean_pairs


def test_normalization_can_reverse_the_raw_euclidean_ranking():
    # The paraphrase has a larger norm but a closer direction to the anchor.
    embeddings = np.array([[1, 0], [3, 4], [0, 2]], dtype=np.float32)
    assert np.linalg.norm(embeddings[0] - embeddings[1]) > np.linalg.norm(
        embeddings[0] - embeddings[2]
    )

    result = normalized_euclidean_pairs(embeddings, np.array([[0, 1, 2]]))

    np.testing.assert_allclose(result["d_paraphrase"], [np.sqrt(0.8)], rtol=1e-14)
    np.testing.assert_allclose(result["d_distractor"], [np.sqrt(2)], rtol=1e-14)
    np.testing.assert_allclose(result["d_paraphrase_distractor"], [np.sqrt(0.4)], rtol=1e-14)
    assert result["pass"].tolist() == [True]
    for key in ("d_paraphrase", "d_distractor", "d_paraphrase_distractor"):
        assert result[key].dtype == np.float64


def test_independent_positive_rescaling_does_not_change_distances_or_decisions():
    embeddings = np.array([[1, 2, 3], [-2, 1, 4], [4, 1, -1], [1, -3, 2]], dtype=np.float64)
    indices = np.array([[0, 1, 2], [3, 2, 1], [1, 0, 3]])
    original = normalized_euclidean_pairs(embeddings, indices)
    rescaled = normalized_euclidean_pairs(embeddings * np.array([2, 0.5, 8, 0.25])[:, None], indices)

    for key in original:
        np.testing.assert_allclose(rescaled[key], original[key], rtol=1e-14, atol=1e-14)
    for key in ("d_paraphrase", "d_distractor", "d_paraphrase_distractor"):
        assert np.all((original[key] >= 0) & (original[key] <= 2))


def test_anchor_reverse_and_strict_decisions_have_distinct_semantics():
    root3 = np.sqrt(3.0)
    embeddings = np.array([
        # Equal anchor distances; the two alternatives are farther apart.
        [1, 0], [0, 1], [0, -1],
        # Anchor passes, but the distractor is close to the paraphrase.
        [1, 0], [0, 1], [-0.5, root3 / 2],
        # Anchor fails, while the reverse comparison passes.
        [1, 0], [0, 1], [root3 / 2, -0.5],
        # Anchor passes strictly, and the reverse comparison is an exact tie.
        [0, 1], [1, 0], [0, -1],
        # Both comparisons pass strictly.
        [1, 0], [1, 0], [-1, 0],
        # All three strings can have identical representations.
        [1, 0], [1, 0], [1, 0],
    ], dtype=np.float64)
    result = normalized_euclidean_pairs(embeddings, np.arange(len(embeddings)).reshape(-1, 3))

    expected = {
        "pass": [True, True, False, True, True, True],
        "reverse_pass": [True, False, True, True, True, True],
        "both_pass": [True, False, False, True, True, True],
        "strict_pass": [False, True, False, True, True, False],
        "strict_both_pass": [False, False, False, False, True, False],
        "ties": [True, False, False, False, False, True],
    }
    for key, values in expected.items():
        np.testing.assert_array_equal(result[key], values)
        assert result[key].dtype == np.bool_
    assert result["d_paraphrase"][0] == result["d_distractor"][0]
    assert result["d_paraphrase"][3] == result["d_paraphrase_distractor"][3]
    assert result["d_paraphrase"][4] == 0
    assert result["d_distractor"][4] == 2


def test_near_tie_is_not_treated_as_an_exact_tie():
    embeddings = np.array([[1, 0], [0, 1], [1e-12, 1]], dtype=np.float64)
    result = normalized_euclidean_pairs(embeddings, np.array([[0, 1, 2]]))

    assert result["d_paraphrase"][0] > result["d_distractor"][0]
    assert result["pass"].tolist() == [False]
    assert result["ties"].tolist() == [False]


@pytest.mark.parametrize("bad_vector", [[0, 0], [np.nan, 1], [np.inf, 1], [-np.inf, 1]])
def test_zero_and_nonfinite_embeddings_are_rejected(bad_vector):
    embeddings = np.array([[1, 0], [0, 1], bad_vector], dtype=np.float64)
    with pytest.raises(ValueError):
        normalized_euclidean_pairs(embeddings, np.array([[0, 1, 2]]))


def test_text_index_deduplicates_exact_strings_in_first_occurrence_order():
    rows = [
        {"caption": "One", "caption2": " One", "negative_caption": "Two"},
        {"caption": "Two", "caption2": "One", "negative_caption": "one"},
        {"caption": "One", "caption2": "One", "negative_caption": "Two"},
    ]
    texts, indices = build_text_index(rows)

    assert texts == ["One", " One", "Two", "one"]
    np.testing.assert_array_equal(indices, [[0, 1, 2], [2, 0, 3], [0, 0, 2]])
    assert indices.shape == (len(rows), 3)
    assert np.issubdtype(indices.dtype, np.integer)


@pytest.mark.parametrize("field", ["caption", "caption2", "negative_caption"])
@pytest.mark.parametrize("invalid", ["", " ", "\n\t", None, 42, [], {}])
def test_text_index_rejects_blank_or_nonstring_captions(field, invalid):
    row = {"caption": "Anchor", "caption2": "Paraphrase", "negative_caption": "Distractor"}
    row[field] = invalid
    with pytest.raises(ValueError):
        build_text_index([row])
