"""clustering_quality's width-neutral metrics: silhouette_full and standardized_spread. CPU only."""
import numpy as np
import pytest
from sklearn.metrics import silhouette_score

from geoae.interp.clustering_quality import (calinski_harabasz, separability_ratio, silhouette_full,
                                             standardized_spread)

STD_KEYS = ("within_var_share", "separability_radii", "inter_centroid_dist_rel")


def _data(seed=0, n=600, d=9, k=12, ids=None):
    rng = np.random.RandomState(seed)
    lab = rng.randint(0, k, n)
    z = (rng.randn(k, d)[lab] * 1.5 + rng.randn(n, d)).astype(np.float32)
    return z, (lab if ids is None else np.asarray(ids)[lab])


def _sil(z, labels):
    return silhouette_full(z, labels, n_score=len(z), device="cpu")[0]


def test_matches_sklearn():
    z, labels = _data()
    assert _sil(z, labels) == pytest.approx(silhouette_score(z, labels, metric="euclidean"), abs=1e-4)


def test_non_contiguous_label_ids():          # e.g. AE clusters with some ids never used
    z, labels = _data(seed=1, k=3, ids=[3, 17, 1999])
    assert _sil(z, labels) == pytest.approx(silhouette_score(z, labels, metric="euclidean"), abs=1e-4)


def test_singleton_cluster_scores_zero():
    z, labels = _data(seed=2)
    labels = labels.copy()
    labels[0] = 99
    assert _sil(z, labels) == pytest.approx(silhouette_score(z, labels, metric="euclidean"), abs=1e-4)


def test_chunking_does_not_change_the_value():
    z, labels = _data(seed=3)
    small = silhouette_full(z, labels, n_score=len(z), device="cpu", score_chunk=37, ref_chunk=101)[0]
    assert small == pytest.approx(_sil(z, labels), abs=1e-6)


def test_translation_invariant():             # an uncentred latent must not cost float32 precision
    z, labels = _data(seed=4)
    assert _sil(z + 1000.0, labels) == pytest.approx(_sil(z, labels), abs=1e-3)


def test_shares_and_calinski_harabasz():
    z, labels = _data(seed=5)
    s = standardized_spread(z, labels)
    n, k = len(z), len(np.unique(labels))
    assert s["within_var_share"] + s["between_var_share"] == pytest.approx(1.0, abs=1e-9)
    assert s["between_var_share"] / s["within_var_share"] == pytest.approx(
        calinski_harabasz(z, labels) * (k - 1) / (n - k), rel=1e-6)


def test_scale_invariant():
    z, labels = _data(seed=6)
    a, b = standardized_spread(z, labels), standardized_spread(z * 7.3, labels)
    for key in STD_KEYS:
        assert b[key] == pytest.approx(a[key], rel=1e-4)
    assert _sil(z * 7.3, labels) == pytest.approx(_sil(z, labels), rel=1e-4)


def test_width_neutral_where_separability_ratio_is_not():
    # Tiling the dimensions scales every distance by sqrt(2): nothing about the clustering changes.
    z, labels = _data(seed=7)
    z2 = np.concatenate([z, z], axis=1)
    a, b = standardized_spread(z, labels), standardized_spread(z2, labels)
    for key in STD_KEYS:
        assert b[key] == pytest.approx(a[key], rel=1e-4)
    assert _sil(z2, labels) == pytest.approx(_sil(z, labels), rel=1e-4)
    assert separability_ratio(z2, labels) / separability_ratio(z, labels) == pytest.approx(np.sqrt(2), rel=1e-3)


def test_single_cluster_is_nan():
    z = np.random.RandomState(8).randn(20, 3).astype(np.float32)
    labels = np.zeros(20, int)
    assert all(np.isnan(v) for v in silhouette_full(z, labels, device="cpu"))
    assert all(np.isnan(v) for v in standardized_spread(z, labels).values())
