"""read_rows must return exactly what mmap indexing returns, without mapping the file. CPU only."""
import numpy as np
import pytest

from geoae.interp.clustering_quality import read_rows


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_read_rows_matches_mmap_indexing(tmp_path, dtype):
    rng = np.random.RandomState(0)
    arr = rng.randn(1000, 37).astype(dtype)
    p = tmp_path / "layer_0.npy"
    np.save(p, arr)
    idx = np.sort(rng.choice(1000, 250, replace=False))
    got = read_rows(p, idx)
    want = np.load(p, mmap_mode="r")[idx]
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.tobytes() == np.asarray(want).tobytes()


def test_read_rows_empty_and_edges(tmp_path):
    arr = np.arange(40, dtype=np.float32).reshape(10, 4)
    p = tmp_path / "x.npy"
    np.save(p, arr)
    assert read_rows(p, np.array([0, 9])).tolist() == [arr[0].tolist(), arr[9].tolist()]
    assert read_rows(p, np.array([], dtype=np.int64)).shape == (0, 4)


def test_read_rows_rejects_short_read(tmp_path):
    p = tmp_path / "x.npy"
    np.save(p, np.zeros((4, 3), np.float32))
    with pytest.raises(IOError, match="short read"):
        read_rows(p, np.array([10]))
