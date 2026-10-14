"""eval_out/atlas_nmi.py helpers on tiny synthetic data."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from geoae.interp import atlas_siblings as AS

_p = Path(__file__).resolve().parents[1] / "eval_out" / "atlas_nmi.py"
sys.path.insert(0, str(_p.parent))
_spec = importlib.util.spec_from_file_location("atlas_nmi", _p)
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)


def membership():
    # 5 chunks; concept ids 0..3 ; unit0 = {0, 1}, unit1 = {2}, unit2 = {3}
    ptr = np.array([0, 1, 3, 4, 6, 6])
    ids = np.array([0, 2, 3, 1, 1, 0])        # chunk0 {0}; chunk1 {2,3}; chunk2 {1}; chunk3 {1,0}; chunk4 {}
    return AS.labels_csr(ptr, ids, 5, [[0, 1], [2], [3]], 4)


def test_exclusive_qualification():
    ch, lab = AN.exclusive_labels(membership())
    # chunk1 carries two units, chunk4 none; chunk3 carries two ids of ONE unit -> still exclusive
    assert ch.tolist() == [0, 2, 3]
    assert lab.tolist() == [0, 0, 0]


def test_row_to_label_and_last_grain():
    row_start = np.array([0, 2, 5, 6, 9, 9])
    ch, lab = np.array([0, 2, 3]), np.array([1, 0, 2])
    idx, y = AN.select_rows(row_start, ch, lab, "token")
    assert idx.tolist() == [0, 1, 5, 6, 7, 8] and y.tolist() == [1, 1, 0, 2, 2, 2]
    idx, y = AN.select_rows(row_start, ch, lab, "last")
    assert idx.tolist() == [1, 5, 8] and y.tolist() == [1, 0, 2]
    # a chunk with no stored rows is skipped in the last grain
    idx, y = AN.select_rows(row_start, np.array([4, 0]), np.array([0, 1]), "last")
    assert idx.tolist() == [1] and y.tolist() == [1]


def test_subsample_fixed():
    assert AN.subsample(5, 10).tolist() == [0, 1, 2, 3, 4]
    s = AN.subsample(100, 10)
    assert len(s) == 10 and (np.diff(s) > 0).all() and s.tolist() == AN.subsample(100, 10).tolist()


def test_ci_centered():
    b = np.array([0.5, 0.6, 0.7, 0.8, 0.9] * 20)
    lo, hi = AN.ci_centered(0.3, b)
    q = np.quantile(b - b.mean(), [.025, .975])
    assert np.isclose(lo, 0.3 + q[0]) and np.isclose(hi, 0.3 + q[1])
    assert lo < 0.3 < hi            # centred on the estimate, not on the (inflated) bootstrap mean


def test_best_cluster_f1():
    y = np.array([0, 0, 0, 0, 1, 1])
    c = np.array([0, 0, 0, 1, 1, 1])
    m, per = AN.best_cluster_f1(y, c)
    # class0 vs cluster0: tp3, |class|4, |cluster|3 -> F1 = 6/7; class1 vs cluster1: tp2, 2, 3 -> 4/5
    assert np.allclose(per, [6 / 7, 4 / 5]) and np.isclose(m, (6 / 7 + 4 / 5) / 2)
    m, per = AN.best_cluster_f1(y, np.array([0, 1, 2, 3, 4, 5]))     # all singletons: F1 = 2/(n_k+1)
    assert np.allclose(per, [2 / 5, 2 / 3])


def test_score_labeling_runs():
    sys.path.insert(0, str(_p.parent))
    import probe_chance_corrected as PCC
    rng = np.random.default_rng(0)
    yi = rng.integers(0, 3, 300)
    ones = np.ones(300)
    W = rng.multinomial(300, ones / 300, size=20).astype(float)
    perms = [rng.permutation(yi) for _ in range(5)]
    r = AN.score_labeling(yi, yi * 2, PCC.nmi_weighted, perms, W, ones)       # perfect clustering
    assert r["cnmi"] > 0.99 and r["f1"] > 0.99 and r["live"] == 3
    assert r["ci_lo"] <= r["cnmi"] <= r["ci_hi"] + 1e-9
