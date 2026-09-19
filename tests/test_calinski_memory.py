"""clustering_quality.calinski_harabasz must equal sklearn's without its float64 copy. CPU only."""
import numpy as np
import pytest
from sklearn.metrics import calinski_harabasz_score

from geoae.interp.clustering_quality import calinski_harabasz


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_sklearn(dtype):
    rng = np.random.RandomState(0)
    z = (rng.randn(3000, 17) + rng.randint(0, 5, 3000)[:, None]).astype(dtype)
    labels = rng.randint(0, 40, 3000)
    assert calinski_harabasz(z, labels) == pytest.approx(calinski_harabasz_score(z, labels), rel=1e-9)


def test_non_contiguous_label_ids():          # e.g. AE clusters with some ids never used
    rng = np.random.RandomState(1)
    z = rng.randn(500, 4).astype(np.float32)
    labels = rng.choice([3, 17, 1999], 500)
    assert calinski_harabasz(z, labels) == pytest.approx(calinski_harabasz_score(z, labels), rel=1e-9)


def test_degenerate_cases():
    assert np.isnan(calinski_harabasz(np.ones((10, 3), np.float32), np.zeros(10, int)))
    z = np.repeat(np.eye(2, dtype=np.float32), 5, axis=0)     # zero within-cluster spread
    assert calinski_harabasz(z, np.repeat([0, 1], 5)) == 1.0 == calinski_harabasz_score(z, np.repeat([0, 1], 5))
