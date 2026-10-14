"""Range geometry of a concept pair: planted shared-direction / orthogonal structure, bootstrap, aggregation."""
import numpy as np
import pytest
import torch

from geoae.interp import atlas_geometry as G

D = 24


def _rows(centre, n, sd, gen):
    return torch.as_tensor(centre, dtype=torch.float32) + sd * torch.randn(n, D, generator=gen)


def _planted(mag_a, mag_b, sd=0.5, n=400, seed=0, direction=None):
    gen = torch.Generator().manual_seed(seed)
    v = torch.zeros(D)
    v[0] = 1.0
    if direction is not None:
        v = direction
    mean = torch.zeros(D)
    return (_rows(mag_a * v, n, sd, gen), _rows(mag_b * v, n, sd, gen), _rows(mag_a * v, n, sd, gen),
            _rows(mag_b * v, n, sd, gen), mean, v)


def test_same_direction_different_magnitude_is_a_range_pair():
    fa, fb, ea, eb, mean, v = _planted(1.0, 3.0)
    s = G.pair_stats(fa, fb, ea, eb, mean)
    assert s["cos"] > 0.99
    assert s["dprime"] == pytest.approx(2.0 / 0.5, rel=0.15)           # (3 - 1) / sd along u
    assert s["radial_share"] > 0.98                                    # the whole gap lies along u
    assert s["cos"] >= G.COS and s["dprime"] >= G.DPRIME
    assert abs(s["cos_eval"] - s["cos"]) < 0.02 and abs(s["insample_dprime"] - s["dprime"]) < 0.3


def test_orthogonal_directions_have_low_cosine_and_no_gap_along_u():
    gen = torch.Generator().manual_seed(1)
    fa = _rows(np.eye(D)[0] * 2, 300, 0.5, gen)
    fb = _rows(np.eye(D)[1] * 2, 300, 0.5, gen)
    ea = _rows(np.eye(D)[0] * 2, 300, 0.5, gen)
    eb = _rows(np.eye(D)[1] * 2, 300, 0.5, gen)
    s = G.pair_stats(fa, fb, ea, eb, torch.zeros(D))
    assert abs(s["cos"]) < 0.1
    assert s["radial_share"] < 0.05 and s["cos"] < G.COS               # the gap is orthogonal to the shared direction
    assert s["dprime"] < 1.0                                            # so the diagonal u cannot separate the two


def test_global_mean_is_subtracted_before_the_cosine():
    fa, fb, ea, eb, mean, v = _planted(1.0, 3.0)
    off = torch.zeros(D)
    off[5] = 40.0                                                       # a big common offset must not create parallelism
    s0 = G.pair_stats(fa, fb, ea, eb, mean)
    s1 = G.pair_stats(fa + off, fb + off, ea + off, eb + off, off)
    assert s1["cos"] == pytest.approx(s0["cos"], abs=1e-4) and s1["dprime"] == pytest.approx(s0["dprime"], rel=1e-3)
    s2 = G.pair_stats(fa + off, fb + off, ea + off, eb + off, mean)     # without centring both means point along the offset
    assert s2["cos"] > 0.99 and s2["dprime"] < s0["dprime"]


def test_dprime_uses_population_variance_and_is_symmetric():
    a = torch.tensor([0.0, 2.0])
    b = torch.tensor([4.0, 6.0])
    assert G.dprime_of(a, b) == pytest.approx(4.0 / 1.0)                # var = 1 each (population)
    assert G.dprime_of(b, a) == pytest.approx(4.0)


def _docs(n_rows, per_doc):
    return torch.arange(n_rows) // per_doc


def test_bootstrap_brackets_the_point_estimate_and_is_reproducible():
    fa, fb, ea, eb, mean, v = _planted(1.0, 3.0, n=400, seed=2)
    pt = G.pair_stats(fa, fb, ea, eb, mean)
    sets = [(x, _docs(len(x), 8)) for x in (fa, fb, ea, eb)]
    b1 = G.bootstrap_pair(*sets, mean, 300, torch.Generator().manual_seed(0))
    b2 = G.bootstrap_pair(*sets, mean, 300, torch.Generator().manual_seed(0))
    for k in ("cos", "dprime", "radial_share"):
        np.testing.assert_allclose(b1[k], b2[k])
        assert len(b1[k]) == 300
    lo, hi = np.percentile(b1["dprime"], [2.5, 97.5])
    assert lo < pt["dprime"] < hi
    assert hi - lo > 0.1                                                # resampling documents does change d'
    assert np.percentile(b1["cos"], 2.5) > 0.95
    assert np.median(b1["dprime"]) == pytest.approx(pt["dprime"], rel=0.1)


def test_document_bootstrap_is_wider_than_row_bootstrap_when_rows_share_a_document_offset():
    gen = torch.Generator().manual_seed(3)
    v = torch.zeros(D)
    v[0] = 1.0
    sets_rows = []
    for mag in (1.0, 2.0, 1.0, 2.0):                                    # fa, fb, ea, eb
        off = torch.randn(40, 1, generator=gen) * 0.6 * v                # one offset per document along the direction
        rows = (mag * v + off.repeat_interleave(10, 0) + 0.3 * torch.randn(400, D, generator=gen))
        sets_rows.append(rows)
    mean = torch.zeros(D)
    width = {}
    for per_doc in (1, 10):
        sets = [(x, _docs(len(x), per_doc)) for x in sets_rows]
        b = G.bootstrap_pair(*sets, mean, 400, torch.Generator().manual_seed(1))
        width[per_doc] = np.subtract(*np.percentile(b["dprime"], [97.5, 2.5]))
    assert width[10] > 1.3 * width[1]


def test_group_summary_counts_range_pairs_and_ci():
    cos = np.array([0.95, 0.99, 0.5, 0.92, 0.1])
    dp = np.array([3.0, 1.0, 5.0, 2.5, 0.2])
    rad = np.array([0.9, 0.8, 0.3, 0.95, 0.5])
    s = G.summarise_group(cos, dp, rad, B=500, seed=0)
    assert s["n"] == 5 and s["range_pairs"] == 2 and s["parallel_pairs"] == 3
    assert s["range_share"]["value"] == pytest.approx(0.4)
    lo, hi = s["range_share"]["ci"]
    assert 0.0 <= lo <= 0.4 <= hi <= 1.0
    assert s["cos_median"]["value"] == pytest.approx(0.92) and s["dprime_median_among_parallel"] == pytest.approx(2.5)
    nan = G.summarise_group([0.95, np.nan], [3.0, 1.0], [0.9, 0.9], B=50)
    assert nan["n"] == 1 and nan["n_dropped_nonfinite"] == 1
    assert G.summarise_group([], [], [], B=10)["range_pairs"] == 0


def test_paired_diff_and_boot_ci_edge_cases():
    d = G.paired_diff([1, 1, 0, 1], [0, 1, 0, 0], B=300, seed=0)
    assert d["mean_diff"] == pytest.approx(0.5) and d["ci"][0] <= 0.5 <= d["ci"][1] and d["n"] == 4
    p, lo, hi = G.boot_ci([], np.mean)
    assert np.isnan(p) and np.isnan(lo)
    assert G.boot_ci([2.0, 2.0, 2.0], np.median)[0] == 2.0


def test_in_sample_statistics_equal_the_original_script_formulas():
    """paper_ranges_along_directions.py, lines 70-123, written out for one pair and compared with pair_stats()."""
    gen = torch.Generator().manual_seed(7)
    rows_a = _rows(np.r_[1.5, 0.5, np.zeros(D - 2)], 120, 0.7, gen)
    rows_b = _rows(np.r_[3.0, 1.0, np.zeros(D - 2)], 90, 0.9, gen)
    rest = _rows(np.zeros(D), 300, 1.0, gen)                          # other rows: the global mean covers ALL rows
    mean = torch.cat([rows_a, rows_b, rest]).mean(0)
    # --- original, verbatim maths: pass 1
    S = torch.stack([rows_a.double().sum(0), rows_b.double().sum(0)])
    n = torch.tensor([len(rows_a), len(rows_b)], dtype=torch.float64)
    mu = S / n.clamp_min(1)[:, None] - mean.double()
    U = torch.nn.functional.normalize(mu, dim=1)
    pa, pb = 0, 1
    u = torch.nn.functional.normalize(U[[pa]] + U[[pb]], dim=1)
    gap = mu[[pa]] - mu[[pb]]
    radial = ((gap * u).sum(1) ** 2) / (gap ** 2).sum(1)
    cos = (U[[pa]] * U[[pb]]).sum(1)
    # --- pass 2: moments of the members projected on u
    m = torch.zeros(1, 6, dtype=torch.float64)
    for rows, which in ((rows_a, 0), (rows_b, 1)):
        proj = ((rows - mean) @ u.float().T).double()
        base = 3 * which
        m[:, base] += len(rows)
        m[:, base + 1] += proj.sum(0)
        m[:, base + 2] += (proj ** 2).sum(0)
    ma, mb = m[:, 1] / m[:, 0], m[:, 4] / m[:, 3]
    va, vb = m[:, 2] / m[:, 0] - ma ** 2, m[:, 5] / m[:, 3] - mb ** 2
    dprime = (ma - mb).abs() / ((va + vb) / 2).sqrt()
    # --- ours, with the same rows duplicated into both halves so the in-sample numbers are the only difference
    st = G.pair_stats(rows_a[:60], rows_b[:45], rows_a[60:], rows_b[45:], mean)
    assert st["insample_cos"] == pytest.approx(float(cos), abs=1e-5)
    assert st["insample_radial_share"] == pytest.approx(float(radial), abs=1e-5)
    assert st["insample_dprime"] == pytest.approx(float(dprime), rel=1e-4)


def test_threshold_sweep_is_monotone():
    cos = np.array([0.95, 0.8, 0.6, 0.3])
    dp = np.array([3.0, 1.5, 0.7, 0.1])
    s = G.summarise_group(cos, dp, np.ones(4), B=20)["threshold_sweep_share"]
    assert s["cos>=0.9,d'>=2"] == 0.25 and s["cos>=0.5,d'>=0.5"] == 0.75
    assert s["cos>=0.5,d'>=0.5"] >= s["cos>=0.7,d'>=0.5"] >= s["cos>=0.9,d'>=2"]


def test_row_docs_and_chunk_means_follow_the_gather_order():
    from types import SimpleNamespace
    # chunks 0..3 with 2, 0, 3, 1 rows; documents 10, 10, 11, 12 (chunk 1 has no rows)
    cache = SimpleNamespace(row_start=np.array([0, 2, 2, 5, 6]), doc=np.array([10, 10, 11, 12]))
    chunks = np.array([2, 0, 1, 3])                                   # gather order: chunk 2's rows first
    X = torch.arange(6, dtype=torch.float32)[:, None] * torch.ones(1, 3)   # rows 0..5 of chunk 2, 0, 3 -> values below
    X = torch.tensor([[3.0], [4.0], [5.0], [0.0], [1.0], [6.0]]).repeat(1, 3)  # chunk2: 3,4,5  chunk0: 0,1  chunk3: 6
    d = G.row_docs(cache, chunks)
    assert d.tolist() == [1, 1, 1, 0, 0, 2]                           # docs: 11 -> 1, 10 -> 0, 12 -> 2 (np.unique order)
    m, md = G.chunk_means(cache, chunks, X)
    assert m[:, 0].tolist() == [4.0, 0.5, 6.0]                        # chunk 1 (no rows) dropped
    assert md.tolist() == [1, 0, 2]
