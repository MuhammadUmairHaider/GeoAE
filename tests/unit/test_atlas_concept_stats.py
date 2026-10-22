"""geoae/analysis/atlas_concept_stats.py on a tiny synthetic Atlas (12 chunks, 6 documents, 57 rows, 6 multi-label concepts, 4
clusters, a k=3 cross-document kNN graph): metric 1 (cross-document cluster lift) and metric 2 (neighbour enrichment) against
brute-force enumeration of row pairs, with unit weights and with integer document weights. A weight w_d is checked by
PHYSICALLY DUPLICATING the document into w_d distinct copies (--copy_pairs across: pairs inside one copy excluded, pairs across
copies kept; exclude: every pair inside the original document excluded; for metric 2 only the queries are duplicated, the
graph is held fixed), plus the GMR / win helpers and the CLI end to end.

The input / output options are covered at the bottom: multi-file --assign and --graph (merge by key, identical-arm dedupe,
conflicting arrays, row mismatches, --graph_prefer), --m1_compare / --m2_compare parsing and the decision rule they leave
alone, --unassigned_as_cluster (mapping, shares, brute-force lifts) and --out_prefix."""
import copy
import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_p = Path(__file__).resolve().parents[2] / "geoae/analysis/atlas_concept_stats.py"
_spec = importlib.util.spec_from_file_location("atlas_concept_stats", _p)
AC = importlib.util.module_from_spec(_spec)
sys.modules["atlas_concept_stats"] = AC
_spec.loader.exec_module(AC)

CHUNK_DOC = np.array([10, 10, 10, 20, 30, 30, 40, 40, 50, 50, 50, 60])
NTOK = np.array([6] * 10 + [3, 6])               # rows at positions 1..n-1, the last one is the final-token row
CARRIES = {0: {0, 1, 3, 4, 6, 8, 9}, 1: {1, 2, 4, 5, 7, 9, 10, 11}, 2: {0, 2, 3, 5, 6, 10}, 3: {5, 11},
           4: set(range(12)), 5: {7}}            # concept -> chunks; 4 is on every chunk, 5 on one chunk
CTYPES = ["content", "entity", "tone", "document", "content", "content"]
ARMS = ("base", "base_tokmean", "bypass", "mat10k_km")
KNN = ("base", "base_tokmean", "bypass", "mat10k")
K = 3
SETS = "primary:3:2:0.9,robust:2:2:0.95"          # primary = concepts 0,1,2; robust adds 3


def world(seed=0):
    rng = np.random.default_rng(seed)
    n_chunks = len(CHUNK_DOC)
    W = SimpleNamespace(n_all=6, chunk_doc=CHUNK_DOC, ntok=NTOK)
    W.row_chunk = np.repeat(np.arange(n_chunks), NTOK - 1)
    W.row_pos = np.concatenate([np.arange(1, n) for n in NTOK])
    W.n_rows = len(W.row_chunk)
    W.labels = {a: rng.integers(0, 4, W.n_rows).astype(np.int32) for a in ARMS}
    W.labels["mat10k"] = rng.integers(-1, 9, W.n_rows).astype(np.int32)     # unused for clustering: may hold -1
    lists = [[c for c, ch in CARRIES.items() if k in ch] for k in range(n_chunks)]
    lists = [list(rng.permutation(x)) for x in lists]
    lists[0].append(lists[0][0])                                            # a duplicated label must count once
    W.label_ptr = np.r_[0, np.cumsum([len(x) for x in lists])].astype(np.int64)
    W.label_ids = np.array([c for x in lists for c in x], dtype=np.int32)
    W.sampled = np.flatnonzero(W.row_pos != NTOK[W.row_chunk] - 1)
    W.doc_ids = np.unique(CHUNK_DOC[W.row_chunk[W.sampled]])
    W.D = len(W.doc_ids)
    W.row_d = np.searchsorted(W.doc_ids, CHUNK_DOC[W.row_chunk])           # dense document of every cache row
    # graph: 30 of the sampled rows, k=3 neighbours from other documents (positions in the sample), one list per arm
    W.g_rows = np.sort(rng.choice(W.sampled, 30, replace=False))
    W.g_chunk = W.row_chunk[W.g_rows]
    W.g_d = W.row_d[W.g_rows]
    W.nbrs = {}
    for a in KNN:
        nb = np.zeros((30, K), dtype=np.int32)
        for i in range(30):
            nb[i] = rng.choice(np.flatnonzero(W.g_d != W.g_d[i]), K, replace=False)
        W.nbrs[a] = nb
    return W


def eligible_ids(set_name):
    return {"primary": [0, 1, 2], "robust": [0, 1, 2, 3]}[set_name]


def build(W, union=(0, 1, 2, 3)):
    """The library's glue from raw arrays to the pair structures (the same calls as main())."""
    union_ids = np.array(union)
    nc, nd, pc, pcon = AC.concept_counts(W.label_ptr, W.label_ids, W.chunk_doc, W.n_all)
    loc = np.full(W.n_all, -1)
    loc[union_ids] = np.arange(len(union_ids))
    keep = loc[pcon] >= 0
    ptr, ids = AC.chunk_csr(pc[keep], loc[pcon[keep]], len(CHUNK_DOC))
    pair_row, pair_c = AC.expand_rows(W.row_chunk[W.sampled], ptr, ids)
    return SimpleNamespace(union_ids=union_ids, ptr=ptr, ids=ids, pair_row=pair_row, pair_c=pair_c,
                           s_doc=W.row_d[W.sampled], nc=nc, nd=nd)


# ------------------------------------------------------------------------------------------ brute force (no shared code)
def same_unit(ci, cj, mode):
    """Are two duplicated entries 'the same document'? across: the same copy; exclude: any copy of the same document."""
    return ci == cj if mode == "across" else ci[0] == cj[0]


def brute_lift_parts(W, arm, c, w, mode="across"):
    """A_c, B_c, G by enumerating ordered row pairs of the document-duplicated sample (copy = (document, i))."""
    lab = W.labels[arm]
    ent = [(r, (int(W.row_d[r]), a)) for r in W.sampled for a in range(int(w[W.row_d[r]]))]
    A = B = same = pairs = 0
    for ri, ci in ent:
        for rj, cj in ent:
            if same_unit(ci, cj, mode):
                continue
            pairs += 1
            same += int(lab[ri] == lab[rj])
            if W.row_chunk[ri] in CARRIES[c] and W.row_chunk[rj] in CARRIES[c]:
                B += 1
                A += int(lab[ri] == lab[rj])
    return A, B, (same / pairs if pairs else np.nan)


def plain_lift_parts(W, arm, c):
    """The spec's unweighted formulas, from the n_ckd counts."""
    lab = W.labels[arm]
    n_ckd = Counter((int(lab[r]), int(W.row_d[r])) for r in W.sampled if W.row_chunk[r] in CARRIES[c])
    n_ck, n_cd = Counter(), Counter()
    for (k, d), v in n_ckd.items():
        n_ck[k] += v
        n_cd[d] += v
    A = sum(v * v for v in n_ck.values()) - sum(v * v for v in n_ckd.values())
    B = sum(n_ck.values()) ** 2 - sum(v * v for v in n_cd.values())
    N_kd = Counter((int(lab[r]), int(W.row_d[r])) for r in W.sampled)
    N_k, N_d = Counter(), Counter()
    for (k, d), v in N_kd.items():
        N_k[k] += v
        N_d[d] += v
    N = len(W.sampled)
    G = (sum(v * v for v in N_k.values()) - sum(v * v for v in N_kd.values())) / (N * N - sum(v * v for v in N_d.values()))
    return A, B, G


def brute_enrich(W, arm, c, w, mode="across"):
    """(precision, expected, enrich) of concept c by enumerating the document-duplicated queries and their pools."""
    ent = [(i, (int(W.g_d[i]), a)) for i in range(len(W.g_rows)) for a in range(int(w[W.g_d[i]]))]
    carries = lambda i: W.g_chunk[i] in CARRIES[c]
    queries = [e for e in ent if carries(e[0])]
    if not queries:
        return np.nan, np.nan, np.nan
    hits = sum(sum(carries(j) for j in W.nbrs[arm][i]) for i, _ in queries)
    precision = hits / (K * len(queries))
    expected = 0.0
    for _, cp in queries:
        pool = [e for e in ent if not same_unit(e[1], cp, mode)]
        expected += sum(carries(e[0]) for e in pool) / len(pool)
    expected /= len(queries)
    return precision, expected, (precision / expected if expected > 0 else np.nan)


def brute_lift(W, arm, c, w, mode="across"):
    A, B, G = brute_lift_parts(W, arm, c, w, mode)
    return (A / B) / G if B else np.nan


def metric1_obj(W, mode="across"):
    b = build(W)
    labels = {a: W.labels[a][W.sampled].astype(np.int64) for a in ARMS}
    return AC.Metric1(b.pair_c, b.pair_row, b.s_doc, labels, len(b.union_ids), W.D, mode), b


def metric2_obj(W, arms=KNN, mode="across"):
    b = build(W)
    return AC.Metric2(W.g_chunk, W.g_d, {a: W.nbrs[a].astype(np.int64) for a in arms}, b.ptr, b.ids, b.union_ids,
                      W.n_all, W.D, mode), b


# ------------------------------------------------------------------------------------------ concepts and pairs
def test_sampled_mask_drops_the_final_token_row():
    W = world()
    m = AC.sampled_mask(W.row_chunk, W.row_pos, NTOK)
    assert m.sum() == (NTOK - 2).sum() == 4 * 11 + 1 and len(W.sampled) == m.sum()
    # the final row of every chunk is the one at position n_tok - 1
    assert not m[np.cumsum(NTOK - 1) - 1].any()


def test_concept_counts_and_eligibility():
    W = world()
    nc, nd, pchunk, pcon = AC.concept_counts(W.label_ptr, W.label_ids, W.chunk_doc, W.n_all)
    for c, chunks in CARRIES.items():
        assert nc[c] == len(chunks)                                   # the duplicated label of chunk 0 is not double counted
        assert nd[c] == len({int(CHUNK_DOC[k]) for k in chunks})
    assert len(pchunk) == sum(len(v) for v in CARRIES.values())
    assert (np.diff(pchunk) >= 0).all()                               # sorted by chunk
    el = {n: AC.eligible_mask(nc, nd, 12, *p) for n, p in AC.parse_sets(SETS).items()}
    assert np.flatnonzero(el["primary"]).tolist() == eligible_ids("primary")
    assert np.flatnonzero(el["robust"]).tolist() == eligible_ids("robust")
    # prevalence cap is inclusive: concept 4 (prevalence 1.0) is out at 0.95 and in at 1.0
    assert AC.eligible_mask(nc, nd, 12, 3, 2, 1.0)[4] and not AC.eligible_mask(nc, nd, 12, 3, 2, 0.95)[4]


def test_parse_sets():
    assert AC.parse_sets("primary:100:30:0.2,robust:50:20:0.2") == {"primary": (100, 30, 0.2), "robust": (50, 20, 0.2)}
    for bad in ("a:1:2", "a:1:2:0.1,a:3:4:0.2", ""):
        with pytest.raises(ValueError):
            AC.parse_sets(bad)


def test_positive_pairs_match_a_loop():
    W = world()
    b = build(W)
    got = sorted(zip(b.pair_row.tolist(), b.pair_c.tolist()))
    want = sorted((i, c) for i, r in enumerate(W.sampled) for c in (0, 1, 2, 3) if W.row_chunk[r] in CARRIES[c])
    assert got == want


def test_doc_weights_are_multinomial_and_seeded():
    w = AC.doc_weights(6, 50, 3)
    assert w.shape == (50, 6) and (w.sum(1) == 6).all() and (w >= 0).all()
    assert (AC.doc_weights(6, 50, 3) == w).all() and not (AC.doc_weights(6, 50, 4) == w).all()


# ------------------------------------------------------------------------------------------ metric 1
def test_metric1_unit_weights_match_brute_force_and_plain_formulas():
    W = world()
    m1, _ = metric1_obj(W)
    parts = m1.parts(np.ones(W.D))
    for arm in ARMS:
        A, B, G = parts[arm]
        for c in range(4):
            bA, bB, bG = brute_lift_parts(W, arm, c, np.ones(W.D, dtype=int))
            pA, pB, pG = plain_lift_parts(W, arm, c)
            assert (A[c], B[c]) == (bA, bB) == (pA, pB)               # exact: integers in float64
            assert G == pytest.approx(bG, rel=1e-12) and G == pytest.approx(pG, rel=1e-12)
        lift = m1.lift(np.ones(W.D))[arm]
        ok = B > 0
        assert np.allclose(lift[ok], (A[ok] / B[ok]) / G, rtol=1e-12)


@pytest.mark.parametrize("mode", AC.COPY_PAIRS)
@pytest.mark.parametrize("w", [np.array([2, 0, 1, 3, 0, 0]), np.array([1, 1, 1, 1, 1, 1]), np.array([0, 4, 0, 1, 1, 0]),
                               np.array([6, 0, 0, 0, 0, 0])])
def test_metric1_integer_weights_equal_physical_duplication(w, mode):
    W = world()
    m1, _ = metric1_obj(W, mode)
    parts = m1.parts(w.astype(np.float64))
    for arm in ARMS:
        A, B, G = parts[arm]
        for c in range(4):
            bA, bB, bG = brute_lift_parts(W, arm, c, w, mode)
            assert (A[c], B[c]) == (bA, bB)
            assert G == pytest.approx(bG, rel=1e-12) or (np.isnan(G) and np.isnan(bG))


def test_unit_weights_are_the_plain_definition_in_both_modes():
    W = world()
    for mode in AC.COPY_PAIRS:
        m1, _ = metric1_obj(W, mode)
        for arm, (A, B, G) in m1.parts(np.ones(W.D)).items():
            for c in range(4):
                assert (A[c], B[c]) == plain_lift_parts(W, arm, c)[:2]
        m2, _ = metric2_obj(W, mode=mode)
        ref, _ = metric2_obj(W)
        assert np.array_equal(m2.common(np.ones(W.D))[1], ref.common(np.ones(W.D))[1])


def test_metric1_undefined_without_cross_document_pairs():
    W = world()
    m1, _ = metric1_obj(W)
    w = np.zeros(W.D)
    w[0] = 1.0                                                        # one document, one copy: no cross-document pair
    assert all(np.isnan(x).all() for x in m1.lift(w).values())


def test_negative_cluster_labels_are_rejected():
    W = world()
    b = build(W)
    labels = {"mat10k": W.labels["mat10k"][W.sampled].astype(np.int64)}
    assert labels["mat10k"].min() < 0
    with pytest.raises(SystemExit):
        AC.Metric1(b.pair_c, b.pair_row, b.s_doc, labels, 4, W.D)


# ------------------------------------------------------------------------------------------ metric 2
def test_knn_hits_match_set_membership():
    W = world()
    b = build(W)
    member = AC.membership_keys(b.ptr, b.ids, b.union_ids, W.n_all)
    assert (np.diff(member) > 0).all()
    pr, pc = AC.expand_rows(W.g_chunk, b.ptr, b.ids)
    for arm in KNN:
        nbr = W.nbrs[arm].astype(np.int64)
        got = AC.knn_hits(W.g_chunk, nbr, pr, pc, member, b.union_ids, W.n_all, block=7)    # blocks smaller than P
        want = [sum(W.g_chunk[j] in CARRIES[int(b.union_ids[c])] for j in nbr[i]) for i, c in zip(pr, pc)]
        assert got.tolist() == want


def test_metric2_unit_weights_match_brute_force():
    W = world()
    m2, b = metric2_obj(W)
    ones = np.ones(W.D)
    m_w, expected = m2.common(ones)
    enrich = m2.enrich(ones)
    for c in range(4):
        n_pos = sum(W.g_chunk[i] in CARRIES[c] for i in range(len(W.g_rows)))
        assert m_w[c] == n_pos == m2.n_pos[c]
        for arm in KNN:
            prec, exp, enr = brute_enrich(W, arm, c, np.ones(W.D, dtype=int))
            assert expected[c] == pytest.approx(exp, rel=1e-12)
            assert m2.precision(arm, ones, m_w)[c] == pytest.approx(prec, rel=1e-12)
            assert enrich[arm][c] == pytest.approx(enr, rel=1e-12)


@pytest.mark.parametrize("mode", AC.COPY_PAIRS)
@pytest.mark.parametrize("w", [np.array([2, 0, 1, 3, 0, 0]), np.array([0, 4, 0, 1, 1, 0]), np.array([6, 0, 0, 0, 0, 0]),
                               np.array([1, 1, 1, 1, 1, 1])])
def test_metric2_integer_weights_equal_duplicated_queries(w, mode):
    W = world()
    m2, _ = metric2_obj(W, mode=mode)
    wf = w.astype(np.float64)
    m_w, expected = m2.common(wf)
    enrich = m2.enrich(wf)
    if mode == "exclude" and w.max() == 6:                  # one document holds all weight: no other-document pool
        assert all(np.isnan(x).all() for x in enrich.values())
        return
    for c in range(4):
        for arm in KNN:
            prec, exp, enr = brute_enrich(W, arm, c, w, mode)
            if np.isnan(enr):
                assert np.isnan(enrich[arm][c])
                continue
            assert expected[c] == pytest.approx(exp, rel=1e-12)
            assert m2.precision(arm, wf, m_w)[c] == pytest.approx(prec, rel=1e-12)
            assert enrich[arm][c] == pytest.approx(enr, rel=1e-12)


# ------------------------------------------------------------------------------------------ statistics helpers
def test_log_ratios_gmr_win():
    x = np.array([2.0, 4.0, 1.0, 0.0, np.nan, 3.0, 5.0])
    y = np.array([1.0, 2.0, 2.0, 1.0, 1.0, np.inf, 5.0])
    delta, ok = AC.log_ratios(x, y)
    assert ok.tolist() == [True, True, True, False, False, False, True]
    gmr, med, win, wins, losses, ties = AC.gmr_win(delta)
    # valid concepts: ratios 2, 2, 1/2, 1 -> mean log = ln(2) / 4, median log = ln(2) / 2
    assert gmr == pytest.approx(2 ** 0.25) and med == pytest.approx(2 ** 0.5)
    assert (wins, losses, ties) == (2, 1, 1) and win == pytest.approx(0.5)
    assert np.isnan(AC.gmr_win(np.array([]))[0])


def test_boot_gmr_win_skips_concepts_undefined_in_a_rep():
    nan = np.nan
    xb = np.array([[2, 4, 1, nan], [2, 2, 2, 2], [nan, nan, nan, nan], [2, 0, 4, 8]], dtype=float)
    yb = np.ones((4, 4))
    yb[0] = [1, 2, 2, 1]
    gm, win, cnt = AC.boot_gmr_win(xb, yb)
    assert cnt.tolist() == [3, 4, 0, 3]
    assert gm[0] == pytest.approx(2 ** (1 / 3)) and win[0] == pytest.approx(2 / 3)
    assert gm[1] == pytest.approx(2.0) and win[1] == 1.0
    assert np.isnan(gm[2]) and np.isnan(win[2])
    assert gm[3] == pytest.approx((2 * 4 * 8) ** (1 / 3)) and win[3] == 1.0       # the 0 is dropped in that rep only
    assert AC.interval(gm) == pytest.approx(np.percentile(gm[~np.isnan(gm)], [2.5, 97.5]).tolist())
    assert AC.interval([np.nan]) is None and AC.sd([1.0]) is None


def test_compare_statistics_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    n, B = 60, 25
    x = np.exp(rng.normal(0.3, 0.5, n))
    y = np.exp(rng.normal(0.0, 0.5, n))
    x[3], y[7] = 0.0, np.nan                                         # two dropped concepts
    xb = x * np.exp(rng.normal(0, 0.05, (B, n)))
    yb = y * np.exp(rng.normal(0, 0.05, (B, n)))
    out = AC.compare(x, y, xb, yb, np.random.default_rng(0), 400)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    delta = np.log(x[ok] / y[ok])
    assert (out["n_concepts"], out["n_valid"], out["n_dropped"]) == (n, n - 2, 2)
    assert out["gmr"] == pytest.approx(np.exp(delta.mean())) and out["median_ratio"] == pytest.approx(np.exp(np.median(delta)))
    assert out["win"] == pytest.approx((delta > 0).mean())
    assert out["wilcoxon_p"] == pytest.approx(stats.wilcoxon(delta).pvalue)
    assert out["sign_p"] == pytest.approx(stats.binomtest(int((delta > 0).sum()), len(delta), 0.5).pvalue)
    lo, hi = out["gmr_ci_doc"]
    assert lo < out["gmr"] < hi and out["gmr_sd_doc"] > 0 and out["win_ci_doc"][0] <= out["win_ci_doc"][1]
    clo, chi = out["gmr_ci_concept"]
    assert clo < out["gmr"] < chi
    assert AC.compare(x, y, xb, yb, np.random.default_rng(0), 400) == out                  # seeded: reproducible
    # constant bootstrap = point estimate: degenerate interval; below min_test no tests are run
    c = AC.compare(x[:9], y[:9], np.tile(x[:9], (B, 1)), np.tile(y[:9], (B, 1)), np.random.default_rng(0), 50)
    assert c["wilcoxon_p"] is None and c["sign_p"] is None and c["gmr_ci_concept"] is None
    assert c["gmr_ci_doc"] == pytest.approx([c["gmr"], c["gmr"]]) and c["gmr_sd_doc"] == pytest.approx(0, abs=1e-12)


def test_tests_over_concepts_drop_ties():
    stats = pytest.importorskip("scipy.stats")
    d = np.array([0.5, 0.2, 0.9, -0.1, 0.0, 0.0, 0.3, 0.4])                  # 5 wins, 1 loss, 2 ties
    wil, sign = AC.tests_over_concepts(d)
    assert sign == pytest.approx(stats.binomtest(5, 6, 0.5).pvalue)         # ties are dropped from the sign test
    assert wil == pytest.approx(stats.wilcoxon(d).pvalue)
    assert AC.tests_over_concepts(np.zeros(5)) == (None, None) and AC.tests_over_concepts(np.array([])) == (None, None)


def test_make_strata_and_arm_summary():
    types = np.array(["content"] * 6 + ["tone"] * 3 + ["entity"])
    n_chunks = np.array([100, 100, 110, 120, 200, 400, 150, 160, 900, 1000])
    strata, (lo, hi) = AC.make_strata(types, n_chunks)
    assert strata["type:content"].tolist() == [0, 1, 2, 3, 4, 5] and "type:document" not in strata
    sizes = sorted(len(strata[k]) for k in strata if k.startswith("size:"))
    assert sum(sizes) == 10 and all(n_chunks[strata["size:T1_small"]] <= lo) and all(n_chunks[strata["size:T3_large"]] > hi)
    s = AC.arm_summary(np.array([1.0, 2.0, 4.0, 0.0, np.nan]), np.array([[1.0, 2.0, 4.0, 0.0, np.nan]] * 3))
    assert (s["n_finite"], s["n_positive"]) == (4, 3) and s["median"] == 1.5 and s["gmean"] == pytest.approx(2.0)
    assert s["gmean_ci_doc"] == pytest.approx([2.0, 2.0])


# ------------------------------------------------------------------------------------------ the CLI end to end
def write_world(tmp, W, graph=True, graph_arms=KNN):
    np.savez(tmp / "assign.npz", row_chunk=W.row_chunk.astype(np.int32), row_pos=W.row_pos.astype(np.int32),
             row_tok=np.zeros(W.n_rows, dtype=np.int32), chunk_doc=CHUNK_DOC.astype(np.int64),
             chunk_ntok=NTOK.astype(np.int32), **W.labels)
    np.savez(tmp / "chunks.npz", doc_int_id=CHUNK_DOC.astype(np.int32), n_tok=NTOK.astype(np.int32),
             row_start=np.r_[0, np.cumsum(NTOK - 1)].astype(np.int64), label_ptr=W.label_ptr, label_ids=W.label_ids)
    (tmp / "meta.json").write_text(json.dumps(dict(
        concept_id=list(range(6)), concept_type=CTYPES, name=[f'concept "{i}", x' for i in range(6)],
        lcc=["A"] * 6, chunk_prevalence=[len(CARRIES[c]) / 12 for c in range(6)])))
    if graph:
        np.savez(tmp / "knn_graph.npz", rows=W.g_rows.astype(np.int64), chunk=W.g_chunk.astype(np.int32),
                 doc=CHUNK_DOC[W.g_chunk].astype(np.int64), tok=np.zeros(len(W.g_rows), dtype=np.int32),
                 **{f"nbr_{a}": W.nbrs[a] for a in graph_arms})


def run_main(tmp, extra=(), n_boot=7, prefix=""):
    """main() on the files of write_world(tmp); `extra` comes last, so it overrides --assign / --graph etc."""
    out = tmp / "out"
    AC.main(["--assign", str(tmp / "assign.npz"), "--chunks", str(tmp / "chunks.npz"), "--concepts", str(tmp / "meta.json"),
             "--graph", str(tmp / "knn_graph.npz"), "--sets", SETS, "--n_boot", str(n_boot), "--n_concept_boot", "50",
             "--seed", "5", "--out_dir", str(out), *extra])
    return out, json.loads((out / f"{prefix}concept_stats.json").read_text())


def nz(v):
    return np.nan if v is None else v


@pytest.mark.parametrize("mode", AC.COPY_PAIRS)
def test_cli_end_to_end_matches_brute_force(tmp_path, capsys, mode):
    pytest.importorskip("scipy.stats")
    W = world()
    write_world(tmp_path, W)
    out, res = run_main(tmp_path, extra=("--copy_pairs", mode))
    assert res["meta"]["args"]["copy_pairs"] == mode
    printed = capsys.readouterr().out
    assert "=== set primary" in printed and "=== set robust" in printed and "bypass_vs_base" in printed
    assert "decision rule" in printed

    # concept table: primary set, spec columns (+ n_graph_pos), unit-weight values equal brute force
    rows = list(csv.DictReader(open(out / "concept_table.csv")))
    assert [r["concept_id"] for r in rows] == ["0", "1", "2"] and rows[0]["name"] == 'concept "0", x'
    assert list(rows[0])[:7] == ["concept_id", "concept_type", "name", "n_chunks", "n_docs", "n_pos_rows", "n_graph_pos"]
    assert list(rows[0])[7:] == [f"lift_{a}" for a in ARMS] + [f"enrich_{a}" for a in KNN]
    for r in rows:
        c = int(r["concept_id"])
        assert r["concept_type"] == CTYPES[c] and int(r["n_chunks"]) == len(CARRIES[c])
        assert int(r["n_docs"]) == len({int(CHUNK_DOC[k]) for k in CARRIES[c]})
        assert int(r["n_pos_rows"]) == sum(W.row_chunk[s] in CARRIES[c] for s in W.sampled)
        assert int(r["n_graph_pos"]) == sum(W.g_chunk[i] in CARRIES[c] for i in range(len(W.g_rows)))
        for a in ARMS:
            assert float(r[f"lift_{a}"]) == pytest.approx(brute_lift(W, a, c, np.ones(W.D, dtype=int)), rel=1e-9, nan_ok=True)
        for a in KNN:
            assert float(r[f"enrich_{a}"]) == pytest.approx(brute_enrich(W, a, c, np.ones(W.D, dtype=int))[2], rel=1e-9,
                                                              nan_ok=True)

    # JSON structure
    assert res["meta"]["metric2"]["ran"] is True and res["meta"]["metric2"]["k"] == K
    assert res["meta"]["n_documents"] == 6 and res["meta"]["n_sampled_rows"] == len(W.sampled)
    assert {"load", "metric1_boot", "metric2_boot", "stats"} <= set(res["meta"]["timings_s"])
    assert res["sets"]["primary"]["n_concepts"] == 3 and res["sets"]["robust"]["n_concepts"] == 4
    assert res["sets"]["primary"]["params"] == dict(min_chunks=3, min_docs=2, max_prevalence=0.9)
    cmp1 = res["sets"]["primary"]["metric1"]["comparisons"]
    assert set(cmp1) == {"bypass_vs_base", "bypass_vs_tokrm", "bypass_vs_sae"}
    assert cmp1["bypass_vs_sae"]["x"] == "bypass" and cmp1["bypass_vs_sae"]["y"] == "mat10k_km"
    assert res["sets"]["primary"]["metric2"]["comparisons"]["bypass_vs_sae"]["y"] == "mat10k"
    assert res["sets"]["primary"]["decision_rule"]["passed"] in (True, False, None)
    ov = cmp1["bypass_vs_base"]["overall"]
    assert ov["n_concepts"] == 3 and len(ov["boot_gmr"]) == 7

    # the per-rep document-bootstrap GMR of the overall comparison = brute force on physically duplicated documents
    Wt = AC.doc_weights(W.D, 7, 5)
    brute = {"metric1": brute_lift, "metric2": lambda W, a, c, w, m: brute_enrich(W, a, c, w, m)[2]}
    for b in range(7):
        for key, fn in brute.items():
            d = []
            for c in (0, 1, 2):
                x, y = fn(W, "bypass", c, Wt[b], mode), fn(W, "base", c, Wt[b], mode)
                if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
                    d.append(np.log(x / y))
            want = np.exp(np.mean(d)) if d else np.nan
            got = nz(res["sets"]["primary"][key]["comparisons"]["bypass_vs_base"]["overall"]["boot_gmr"][b])
            assert got == pytest.approx(want, rel=1e-9, nan_ok=True)
    # strata exist for the types present and for the size terciles
    assert set(cmp1["bypass_vs_base"]["strata"]) >= {"type:content", "type:entity", "type:tone"}
    assert "type:document" not in cmp1["bypass_vs_base"]["strata"]                      # no document concept in primary
    assert res["sets"]["robust"]["metric1"]["comparisons"]["bypass_vs_base"]["strata"]["type:document"]["n_concepts"] == 1
    # the bootstrap-bias diagnostics are reported next to the percentile CI
    assert {"gmr_boot_mean_doc", "gmr_ci_doc_centered"} <= set(ov)
    # reproducible
    out2, res2 = run_main(tmp_path, extra=("--copy_pairs", mode))
    for s in ("primary", "robust"):
        assert res2["sets"][s] == res["sets"][s]


def test_cli_skips_metric2_without_a_graph(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    write_world(tmp_path, world(1), graph=False)
    out, res = run_main(tmp_path)
    assert res["meta"]["metric2"]["ran"] is False and "does not exist" in res["meta"]["metric2"]["reason"]
    assert "skipped" in res["sets"]["primary"]["metric2"] and "metric1" in res["sets"]["primary"]
    assert res["sets"]["primary"]["decision_rule"]["passed"] is None
    header = next(csv.reader(open(out / "concept_table.csv")))
    assert "n_graph_pos" not in header and not any(h.startswith("enrich_") for h in header)
    assert "metric 2 SKIPPED" in capsys.readouterr().out


def test_cli_with_a_partial_graph_skips_comparisons_needing_missing_arms(tmp_path):
    pytest.importorskip("scipy.stats")
    write_world(tmp_path, world(2), graph_arms=("base", "bypass"))               # killed graph job: two arms finished
    out, res = run_main(tmp_path)
    m2 = res["sets"]["primary"]["metric2"]
    assert res["meta"]["metric2"]["arms_missing_from_graph"] == ["base_tokmean", "mat10k"]
    assert set(m2["comparisons"]) == {"bypass_vs_base"} and set(m2["skipped"]) == {"bypass_vs_tokrm", "bypass_vs_sae"}
    assert set(res["sets"]["primary"]["metric1"]["comparisons"]) == {"bypass_vs_base", "bypass_vs_tokrm", "bypass_vs_sae"}


def test_cli_rejects_an_inconsistent_graph(tmp_path):
    W = world(3)
    write_world(tmp_path, W)
    G = dict(np.load(tmp_path / "knn_graph.npz"))
    G["nbr_base"][0, 0] = np.flatnonzero(W.g_d == W.g_d[0])[0]                    # a neighbour in the query's own document
    np.savez(tmp_path / "knn_graph.npz", **G)
    with pytest.raises(SystemExit, match="own document"):
        run_main(tmp_path)


# ------------------------------------------------------------------------------------------ multi-file --assign
def save_assign(path, W, arms, **override):
    """An assign npz holding the spine of world W and the label arrays `arms`; `override` replaces any key."""
    d = dict(row_chunk=W.row_chunk.astype(np.int32), row_pos=W.row_pos.astype(np.int32),
             row_tok=np.zeros(W.n_rows, dtype=np.int32), chunk_doc=CHUNK_DOC.astype(np.int64),
             chunk_ntok=NTOK.astype(np.int32))
    d.update({a: W.labels[a] for a in arms})
    d.update(override)
    np.savez(path, **d)
    return str(path)


def test_load_assign_merges_files_by_key_and_dedupes_identical_arms(tmp_path):
    W = world()
    f1 = save_assign(tmp_path / "a1.npz", W, ["base", "bypass"])
    f2 = save_assign(tmp_path / "a2.npz", W, ["bypass", "mat10k_km", "base_tokmean"],
                     bypass=W.labels["bypass"].astype(np.int64))              # same labels in another dtype: no conflict
    A, holders = AC.load_assign([f1, f2], ["base", "bypass", "mat10k_km", "base_tokmean"])
    for a in ARMS:
        assert np.array_equal(A[a], W.labels[a])
    assert np.array_equal(A["row_chunk"], W.row_chunk) and np.array_equal(A["row_pos"], W.row_pos)
    assert np.array_equal(A["chunk_doc"], CHUNK_DOC) and np.array_equal(A["chunk_ntok"], NTOK)
    assert holders == {"base": [f1], "bypass": [f1, f2], "mat10k_km": [f2], "base_tokmean": [f2]}
    A1, h1 = AC.load_assign(Path(f1), ["base"])                              # a single path: the pre-merge call
    assert set(A1) == set(AC.ASSIGN_SPINE) | {"base"} and h1 == {"base": [f1]}


def test_load_assign_compares_only_the_requested_arms(tmp_path):
    W = world()
    bad = W.labels["base"].copy()
    bad[0] += 1
    f1 = save_assign(tmp_path / "a1.npz", W, ["base", "bypass"])
    f2 = save_assign(tmp_path / "a2.npz", W, ["bypass"], base=bad)
    A, holders = AC.load_assign([f1, f2], ["bypass"])                        # `base` differs but is not asked for
    assert holders == {"bypass": [f1, f2]} and "base" not in A
    with pytest.raises(SystemExit, match=r"'base' is in both .* with different labels"):
        AC.load_assign([f1, f2], ["base"])


def test_load_assign_rejects_an_arm_with_different_labels_in_two_files(tmp_path):
    W = world()
    bad = W.labels["bypass"].copy()
    bad[[3, 9]] += 1
    f1 = save_assign(tmp_path / "a1.npz", W, ["bypass"])
    f2 = save_assign(tmp_path / "a2.npz", W, ["bypass"], bypass=bad)
    with pytest.raises(SystemExit, match=r"'bypass' is in both .* with different labels \(2 of 57 rows differ\)"):
        AC.load_assign([f1, f2], ["bypass"])
    f3 = save_assign(tmp_path / "a3.npz", W, ["bypass"], bypass=W.labels["bypass"][:-1])
    with pytest.raises(SystemExit, match="different shapes"):
        AC.load_assign([f1, f3], ["bypass"])


@pytest.mark.parametrize("key", AC.ASSIGN_SPINE)
def test_load_assign_rejects_files_over_different_rows(tmp_path, key):
    W = world()
    f1 = save_assign(tmp_path / "a1.npz", W, ["base"])
    wrong = {"row_chunk": np.roll(W.row_chunk, 1), "row_pos": np.roll(W.row_pos, 1), "chunk_doc": CHUNK_DOC[::-1].copy(),
             "chunk_ntok": NTOK[::-1].copy()}[key]
    assert not np.array_equal(wrong, {"row_chunk": W.row_chunk, "row_pos": W.row_pos, "chunk_doc": CHUNK_DOC,
                                      "chunk_ntok": NTOK}[key])
    f2 = save_assign(tmp_path / "a2.npz", W, ["bypass"], **{key: wrong})
    with pytest.raises(SystemExit, match=f"{key} differs from"):
        AC.load_assign([f1, f2], ["base", "bypass"])


def test_load_assign_missing_spine_key_and_missing_arm(tmp_path):
    W = world()
    f1 = save_assign(tmp_path / "a1.npz", W, ["base"])
    np.savez(tmp_path / "nospine.npz", base=W.labels["base"])
    with pytest.raises(SystemExit, match=r"missing key\(s\)"):
        AC.load_assign([f1, str(tmp_path / "nospine.npz")], ["base"])
    with pytest.raises(SystemExit, match=r"no cluster arm\(s\) \['nope'\]"):
        AC.load_assign([f1], ["base", "nope"])
    with pytest.raises(SystemExit, match="no file"):
        AC.load_assign([], ["base"])


# ------------------------------------------------------------------------------------------ multi-file --graph
def graph_ctx(W):
    """The cache arguments of load_graph that follow (paths, knn_arms)."""
    return (W.n_rows, W.row_chunk.astype(np.int64), CHUNK_DOC.astype(np.int64),
            AC.sampled_mask(W.row_chunk, W.row_pos, NTOK), W.doc_ids)


def save_graph(path, W, arms, nbrs=None, **override):
    d = dict(rows=W.g_rows.astype(np.int64), chunk=W.g_chunk.astype(np.int32), doc=CHUNK_DOC[W.g_chunk].astype(np.int64),
             tok=np.zeros(len(W.g_rows), dtype=np.int32))
    d.update({f"nbr_{a}": (nbrs or W.nbrs)[a] for a in arms})
    d.update(override)
    np.savez(path, **d)
    return str(path)


def load_graph(W, paths, arms=KNN, prefer=None):
    return AC.load_graph(paths, list(arms), *graph_ctx(W), prefer=prefer)


def tweak_nbrs(W, arm="bypass"):
    """W.nbrs[arm] with the first two neighbours of row 0 swapped (an order change only) and one neighbour of row 1 replaced
    by another document's row: 3 entries differ, 1 row has a different neighbour SET."""
    nb = W.nbrs[arm].copy()
    nb[0, [0, 1]] = nb[0, [1, 0]]
    nb[1, 2] = next(v for v in range(len(W.g_rows)) if W.g_d[v] != W.g_d[1] and v not in nb[1])
    return nb


def test_load_graph_merges_files_by_arm_and_dedupes_identical_arms(tmp_path):
    W = world()
    g1 = save_graph(tmp_path / "g1.npz", W, ["base", "bypass"])
    g2 = save_graph(tmp_path / "g2.npz", W, ["bypass", "mat10k", "base_tokmean"])
    G = load_graph(W, [g1, g2])
    assert list(G["nbrs"]) == list(KNN) and G["missing"] == [] and G["files"] == [g1, g2]
    for a in KNN:
        assert np.array_equal(G["nbrs"][a], W.nbrs[a]) and G["nbrs"][a].dtype == np.int64
    assert np.array_equal(G["rows"], W.g_rows) and np.array_equal(G["chunk"], W.g_chunk) and np.array_equal(G["doc"], W.g_d)
    assert G["arm_files"] == {"base": g1, "bypass": g1, "base_tokmean": g2, "mat10k": g2}
    assert set(G["duplicates"]) == {"bypass"}
    assert G["duplicates"]["bypass"] == dict(used=g1, prefer=None, also_in=[g2], n_entries_differ={g2: 0},
                                             n_rows_differ={g2: 0})
    G1 = AC.load_graph(g1, list(KNN), *graph_ctx(W))                          # a single path (str): the pre-merge call
    assert list(G1["nbrs"]) == ["base", "bypass"] and G1["missing"] == ["base_tokmean", "mat10k"]
    assert G1["duplicates"] == {} and G1["files"] == [g1]


def test_load_graph_differing_duplicates_need_graph_prefer(tmp_path, capsys):
    W = world()
    nb2 = dict(W.nbrs, bypass=tweak_nbrs(W))
    g1 = save_graph(tmp_path / "g1.npz", W, ["base", "bypass"])
    g2 = save_graph(tmp_path / "g2.npz", W, ["bypass", "mat10k"], nbrs=nb2)
    assert (nb2["bypass"] != W.nbrs["bypass"]).sum() == 3
    with pytest.raises(SystemExit, match=r"different neighbours \(3 of 90 neighbour entries differ\).*--graph_prefer"):
        load_graph(W, [g1, g2], arms=("bypass",))
    first = load_graph(W, [g1, g2], arms=("bypass",), prefer="first")
    assert np.array_equal(first["nbrs"]["bypass"], W.nbrs["bypass"]) and first["arm_files"] == {"bypass": g1}
    log = capsys.readouterr().out
    assert "3 of 90 neighbour entries (1 of 30 rows with a different neighbour set) differ" in log
    assert f"between {g1} (used, --graph_prefer first) and {g2}" in log
    last = load_graph(W, [g1, g2], arms=("bypass",), prefer="last")
    assert np.array_equal(last["nbrs"]["bypass"], nb2["bypass"]) and last["arm_files"] == {"bypass": g2}
    assert last["duplicates"]["bypass"] == dict(used=g2, prefer="last", also_in=[g1], n_entries_differ={g1: 3},
                                                n_rows_differ={g1: 1})
    assert f"between {g2} (used, --graph_prefer last) and {g1}" in capsys.readouterr().out
    # arms found in one file only do not care about the preference; an arm in no file is reported as missing
    both = load_graph(W, [g1, g2], prefer="last")
    assert both["arm_files"] == {"base": g1, "bypass": g2, "mat10k": g2} and both["missing"] == ["base_tokmean"]
    with pytest.raises(ValueError):
        load_graph(W, [g1, g2], prefer="middle")


def test_load_graph_a_duplicate_with_another_k_is_a_difference(tmp_path, capsys):
    W = world()
    g1 = save_graph(tmp_path / "g1.npz", W, ["bypass"])
    g2 = save_graph(tmp_path / "g2.npz", W, ["bypass"], nbrs={"bypass": W.nbrs["bypass"][:, :2]})
    with pytest.raises(SystemExit, match="different shapes"):
        load_graph(W, [g1, g2], arms=("bypass",))
    got = load_graph(W, [g1, g2], arms=("bypass",), prefer="first")
    assert got["nbrs"]["bypass"].shape == (30, K) and "shapes differ" in capsys.readouterr().out
    assert got["duplicates"]["bypass"]["n_entries_differ"] == {g2: None}


def test_load_graph_rejects_files_over_different_rows_and_invalid_files(tmp_path):
    W = world()
    g1 = save_graph(tmp_path / "g1.npz", W, ["base"])
    other = np.sort(np.random.default_rng(9).choice(W.sampled, 30, replace=False))
    assert not np.array_equal(other, W.g_rows)
    g2 = save_graph(tmp_path / "g2.npz", W, ["bypass"], rows=other, chunk=W.row_chunk[other].astype(np.int32),
                    doc=CHUNK_DOC[W.row_chunk[other]].astype(np.int64))       # valid on its own, other rows than g1
    with pytest.raises(SystemExit, match="rows/chunk/doc differ from"):
        load_graph(W, [g1, g2])
    # every file is validated like a single one: a bad chunk column, a neighbour in the query's own document
    g3 = save_graph(tmp_path / "g3.npz", W, ["bypass"], chunk=np.roll(W.g_chunk, 1).astype(np.int32))
    with pytest.raises(SystemExit, match="disagree with the cache"):
        load_graph(W, [g1, g3])
    bad = W.nbrs["mat10k"].copy()
    bad[0, 0] = np.flatnonzero(W.g_d == W.g_d[0])[0]
    g4 = save_graph(tmp_path / "g4.npz", W, ["mat10k"], nbrs={"mat10k": bad})
    with pytest.raises(SystemExit, match="own document"):
        load_graph(W, [g1, g4])
    with pytest.raises(SystemExit, match="no file"):
        load_graph(W, [])


def redraw_nbrs(W, arm="bypass", seed=99):
    """W.nbrs[arm] with every row's neighbours drawn again (other documents only): a graph with other metric values."""
    rng = np.random.default_rng(seed)
    nb = np.zeros_like(W.nbrs[arm])
    for i in range(len(W.g_rows)):
        nb[i] = rng.choice(np.flatnonzero(W.g_d != W.g_d[i]), K, replace=False)
    return nb


# ------------------------------------------------------------------------------------------ comparisons
def test_parse_compare_names_order_and_defaults():
    assert AC.parse_compare("a=x:y, b = u : v") == {"a": ("x", "y"), "b": ("u", "v")}
    assert list(AC.parse_compare("z=x:y,a=u:v")) == ["z", "a"]               # the order seeds the concept bootstrap
    assert AC.parse_compare("") == {} and AC.parse_compare(" , ") == {}
    assert AC.DEFAULT_M1_COMPARE == ("bypass_vs_base=bypass:base,bypass_vs_tokrm=bypass:base_tokmean,"
                                     "bypass_vs_sae=bypass:mat10k_km")
    assert AC.DEFAULT_M2_COMPARE == ("bypass_vs_base=bypass:base,bypass_vs_tokrm=bypass:base_tokmean,"
                                     "bypass_vs_sae=bypass:mat10k")
    # the pre-registered comparisons (the module's former COMPARISONS table), in this order
    assert list(AC.DEFAULT_COMPARE_SPECS["m1"].items()) == [("bypass_vs_base", ("bypass", "base")),
                                                            ("bypass_vs_tokrm", ("bypass", "base_tokmean")),
                                                            ("bypass_vs_sae", ("bypass", "mat10k_km"))]
    assert list(AC.DEFAULT_COMPARE_SPECS["m2"].items()) == [("bypass_vs_base", ("bypass", "base")),
                                                            ("bypass_vs_tokrm", ("bypass", "base_tokmean")),
                                                            ("bypass_vs_sae", ("bypass", "mat10k"))]
    assert AC.DECISION_COMPARISON == "bypass_vs_base"
    assert AC.split_list(" a, b,,c ,") == ["a", "b", "c"]


def test_analyse_set_defaults_to_the_preregistered_comparisons():
    pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    n, B = 12, 6
    types = np.array(["content"] * 8 + ["tone"] * 4)
    n_chunks = np.arange(100, 112)
    pt, boot = {}, {}
    for m, arms in (("m1", ARMS), ("m2", KNN)):
        pt[m] = {a: np.exp(rng.normal(0, 0.5, n)) for a in arms}
        boot[m] = {a: pt[m][a] * np.exp(rng.normal(0, 0.05, (B, n))) for a in arms}
    default = AC.analyse_set(0, np.arange(n), types, n_chunks, pt, boot, 50, 0)
    assert default == AC.analyse_set(0, np.arange(n), types, n_chunks, pt, boot, 50, 0, AC.DEFAULT_COMPARE_SPECS)
    assert set(default["metric1"]["comparisons"]) == {"bypass_vs_base", "bypass_vs_tokrm", "bypass_vs_sae"}
    assert default["metric1"]["comparisons"]["bypass_vs_sae"]["y"] == "mat10k_km"
    assert default["metric2"]["comparisons"]["bypass_vs_sae"]["y"] == "mat10k"
    assert "skipped" not in default["decision_rule"] and default["decision_rule"]["passed"] in (True, False)
    no_graph = AC.analyse_set(0, np.arange(n), types, n_chunks, {"m1": pt["m1"]}, {"m1": boot["m1"]}, 50, 0)
    assert no_graph["decision_rule"]["passed"] is None and "skipped" not in no_graph["decision_rule"]   # a metric not run
    assert no_graph["metric2"] == {"skipped": "no neighbour graph"}


@pytest.mark.parametrize("bad", ["a", "a=x", "a=x:", "a=:y", "=x:y", "a=x:y:z", "a=x=y:z", "a=x:y,a=u:v", "a=x:x"])
def test_parse_compare_rejects_malformed_items(bad):
    with pytest.raises(ValueError, match="--m2_compare"):
        AC.parse_compare(bad, "--m2_compare")


def test_cli_rejects_a_malformed_comparison(tmp_path):
    write_world(tmp_path, world())
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, extra=("--m1_compare", "bypass_vs_base=bypass"))
    assert e.value.code == 2                                                  # argparse error, no traceback


# ------------------------------------------------------------------------------------------ --unassigned_as_cluster
def test_fshare_never_rounds_a_nonzero_share_down_to_zero():
    assert AC.fshare(0) == AC.fshare(0.0) == "0%" and AC.fshare(1.0) == "100.0%" and AC.fshare(0.1234) == "12.3%"
    assert AC.fshare(3.376e-05) == "0.0034%" and AC.fshare(1 / 799_676) == "0.00013%" and AC.fshare(0.0005) == "0.05%"


def test_unassigned_to_cluster_maps_minus_one_to_one_extra_cluster():
    L, n, cid = AC.unassigned_to_cluster(np.array([0, 2, -1, 1, -1]))
    assert L.tolist() == [0, 2, 3, 1, 3] and (n, cid) == (2, 3)
    same = np.array([4, 0, 1])
    L, n, cid = AC.unassigned_to_cluster(same)
    assert L.tolist() == [4, 0, 1] and (n, cid) == (0, None)                 # nothing to map: unchanged
    L, n, cid = AC.unassigned_to_cluster(np.array([-1, -1, -1]))
    assert L.tolist() == [0, 0, 0] and (n, cid) == (3, 0)                    # nothing assigned: one cluster, id 0
    with pytest.raises(SystemExit, match="below -1"):
        AC.unassigned_to_cluster(np.array([0, -2]), "mat10k")


def test_metric1_unassigned_as_cluster_matches_brute_force_on_the_mapped_labels():
    W = world()
    b = build(W)
    raw = W.labels["mat10k"][W.sampled].astype(np.int64)
    n_un = int((raw < 0).sum())
    assert 0 < n_un < len(raw)
    labels = {"mat10k": raw, "base": W.labels["base"][W.sampled].astype(np.int64)}
    with pytest.raises(SystemExit, match="--unassigned_as_cluster"):          # without the flag: today's error, now hinting it
        AC.Metric1(b.pair_c, b.pair_row, b.s_doc, labels, 4, W.D)
    m1 = AC.Metric1(b.pair_c, b.pair_row, b.s_doc, labels, 4, W.D, "across", True)
    assert m1.unassigned["mat10k"] == dict(n=n_un, share=n_un / len(raw), cluster_id=int(raw.max()) + 1)
    assert m1.unassigned["base"] == dict(n=0, share=0.0, cluster_id=None)
    assert m1.arms["mat10k"].n_clusters == int(raw.max()) + 2 and (raw < 0).any()  # the input array is not modified
    mapped = W.labels["mat10k"].astype(np.int64).copy()
    mapped[mapped < 0] = raw.max() + 1
    Wm = copy.copy(W)
    Wm.labels = dict(W.labels, mat10k=mapped)
    for w in (np.ones(W.D, dtype=int), np.array([2, 0, 1, 3, 0, 0])):
        parts = m1.parts(w.astype(np.float64))
        for c in range(4):
            bA, bB, bG = brute_lift_parts(Wm, "mat10k", c, w)
            A, B, G = parts["mat10k"]
            assert (A[c], B[c]) == (bA, bB) and G == pytest.approx(bG, rel=1e-12)
    # a flag with nothing to map changes nothing
    ref = AC.Metric1(b.pair_c, b.pair_row, b.s_doc, {"base": labels["base"]}, 4, W.D)
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in
               zip(ref.lift(np.ones(W.D))["base"], m1.lift(np.ones(W.D))["base"]))


def test_cli_unassigned_as_cluster_records_the_share_and_prints_it(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    W = world()
    write_world(tmp_path, W)
    raw = W.labels["mat10k"][W.sampled].astype(np.int64)
    n_un = int((raw < 0).sum())
    args = ("--cluster_arms", "base,mat10k", "--m1_compare", "sae_vs_base=mat10k:base")
    with pytest.raises(SystemExit, match="--unassigned_as_cluster"):
        run_main(tmp_path, extra=args)
    capsys.readouterr()
    out, res = run_main(tmp_path, extra=args + ("--unassigned_as_cluster",))
    un = res["meta"]["unassigned_as_cluster"]
    assert un["enabled"] is True and un["n_rows"] == {"base": 0, "mat10k": n_un}
    assert un["share_of_sampled_rows"] == pytest.approx({"base": 0.0, "mat10k": n_un / len(raw)})
    assert un["cluster_id"] == {"base": None, "mat10k": int(raw.max()) + 1}
    printed = capsys.readouterr().out
    assert re.search(r"^   arm +unassigned +median +geo-mean", printed, re.M)
    row = re.search(r"^   mat10k +(\d+\.\d)% +\d", printed, re.M)
    assert row and float(row.group(1)) == pytest.approx(100 * n_un / len(raw), abs=0.05)
    assert re.search(r"^   base +0% +\d", printed, re.M)
    sections = re.findall(r"-- metric2.*?(?=\n   ! =)", printed, re.S)           # the metric-2 tables have no such column
    assert len(sections) == 2 and all("unassigned" not in sec for sec in sections)
    assert "unassigned (-1) sampled rows kept as one extra cluster" in printed
    # the concept table's lifts are brute force on the mapped labels
    mapped = W.labels["mat10k"].astype(np.int64).copy()
    mapped[mapped < 0] = raw.max() + 1
    Wm = copy.copy(W)
    Wm.labels = dict(W.labels, mat10k=mapped)
    rows = list(csv.DictReader(open(out / "concept_table.csv")))
    assert len(rows) == 3
    for r in rows:
        want = brute_lift(Wm, "mat10k", int(r["concept_id"]), np.ones(W.D, dtype=int))
        assert float(r["lift_mat10k"]) == pytest.approx(want, rel=1e-9, nan_ok=True)
    assert res["sets"]["primary"]["metric1"]["comparisons"]["sae_vs_base"]["x"] == "mat10k"


# ------------------------------------------------------------------------------------------ the CLI options end to end
def test_cli_out_prefix_names_the_outputs(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    write_world(tmp_path, world(4))
    out, res = run_main(tmp_path, extra=("--out_prefix", "scope_"), prefix="scope_")
    assert sorted(p.name for p in out.iterdir()) == ["scope_concept_stats.json", "scope_concept_table.csv"]
    assert res["meta"]["args"]["out_prefix"] == "scope_"
    printed = capsys.readouterr().out
    assert f"wrote {out / 'scope_concept_table.csv'}" in printed and f"and {out / 'scope_concept_stats.json'}" in printed
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, extra=("--out_prefix", "sub/x_"))
    assert e.value.code == 2


def test_cli_defaults_equal_the_explicit_default_options(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    write_world(tmp_path, world(5))
    out, res = run_main(tmp_path)
    printed = capsys.readouterr().out
    assert "unassigned" not in printed                                        # no extra column unless the flag is on
    meta = res["meta"]
    assert meta["unassigned_as_cluster"]["enabled"] is False
    assert set(meta["unassigned_as_cluster"]["share_of_sampled_rows"].values()) == {0.0}
    assert meta["arms_added_for_comparisons"] == {"--cluster_arms": [], "--knn_arms": []}
    assert meta["cluster_arms"] == list(ARMS) and meta["metric2"]["knn_arms"] == list(KNN)
    assert meta["comparisons"] == {m: {n: list(xy) for n, xy in c.items()} for m, c in AC.DEFAULT_COMPARE_SPECS.items()}
    assert meta["inputs"]["graph_prefer"] is None and meta["inputs"]["graph_duplicate_arms"] == {}
    out2, res2 = run_main(tmp_path, extra=("--m1_compare", AC.DEFAULT_M1_COMPARE, "--m2_compare", AC.DEFAULT_M2_COMPARE,
                                           "--graph_prefer"))                 # the bare flag means `first`
    assert res2["meta"]["inputs"]["graph_prefer"] == "first"
    assert res2["sets"] == res["sets"]
    assert (out2 / "concept_table.csv").read_text() == (out / "concept_table.csv").read_text()


def test_cli_custom_comparisons_add_their_arms_and_leave_the_decision_rule_alone(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    write_world(tmp_path, world(5))
    _, ref = run_main(tmp_path)
    capsys.readouterr()
    out, res = run_main(tmp_path, extra=(
        "--cluster_arms", "base", "--knn_arms", "base",
        "--m1_compare", "first=bypass:base,second=bypass:base_tokmean,sae_vs_tok=mat10k_km:base_tokmean",
        "--m2_compare", "first=bypass:base,second=bypass:base_tokmean"))
    printed = capsys.readouterr().out
    meta = res["meta"]
    assert meta["cluster_arms"] == ["base", "bypass", "base_tokmean", "mat10k_km"]
    assert meta["arms_added_for_comparisons"] == {"--cluster_arms": ["bypass", "base_tokmean", "mat10k_km"],
                                                  "--knn_arms": ["bypass", "base_tokmean"]}
    assert "comparison arm(s) ['bypass', 'base_tokmean', 'mat10k_km'] added to --cluster_arms" in printed
    assert "comparison arm(s) ['bypass', 'base_tokmean'] added to --knn_arms" in printed
    p = res["sets"]["primary"]
    assert set(p["metric1"]["comparisons"]) == {"first", "second", "sae_vs_tok"} and p["metric1"]["skipped"] == {}
    assert set(p["metric2"]["comparisons"]) == {"first", "second"}
    assert set(p["metric1"]["arms"]) == {"base", "bypass", "base_tokmean", "mat10k_km"}      # every arm used is summarised
    assert set(p["metric2"]["arms"]) == {"base", "bypass", "base_tokmean"}
    assert (p["metric1"]["comparisons"]["sae_vs_tok"]["x"], p["metric1"]["comparisons"]["sae_vs_tok"]["y"]) == (
        "mat10k_km", "base_tokmean")
    for m in ("metric1", "metric2"):                                          # renamed defaults at the same positions
        assert p[m]["comparisons"]["first"] == ref["sets"]["primary"][m]["comparisons"]["bypass_vs_base"]
        assert p[m]["comparisons"]["second"] == ref["sets"]["primary"][m]["comparisons"]["bypass_vs_tokrm"]
    assert p["metric1"]["arms"]["mat10k_km"] == ref["sets"]["primary"]["metric1"]["arms"]["mat10k_km"]
    # no comparison named bypass_vs_base: the decision rule is skipped with a note, not computed from another comparison
    d = p["decision_rule"]
    assert d["passed"] is None and d["skipped"] == "no bypass_vs_base comparison configured for metric1, metric2"
    assert d["ci_lower"] == {"metric1": None, "metric2": None}
    assert "decision rule (bypass_vs_base, doc-bootstrap GMR CI above 1 for both metrics): skipped, no bypass_vs_base" in printed
    assert re.search(r"^   mat10k_km +\d", printed, re.M) and "sae_vs_tok (mat10k_km vs base_tokmean)" in printed
    # the rule is tied to the NAME: re-pointing the name re-points the rule
    _, moved = run_main(tmp_path, extra=("--m1_compare", "bypass_vs_base=base_tokmean:base",
                                         "--m2_compare", "bypass_vs_base=base_tokmean:base"))
    mp = moved["sets"]["primary"]
    named = mp["metric1"]["comparisons"]["bypass_vs_base"]
    assert mp["decision_rule"]["ci_lower"]["metric1"] == named["overall"]["gmr_ci_doc"][0]
    assert named["x"] == "base_tokmean"
    assert "skipped" not in mp["decision_rule"]


def test_cli_decision_rule_is_skipped_only_for_the_metric_without_the_comparison(tmp_path):
    pytest.importorskip("scipy.stats")
    write_world(tmp_path, world(5))
    _, res = run_main(tmp_path, extra=("--m2_compare", "other=bypass:base"))
    d = res["sets"]["primary"]["decision_rule"]
    assert d["passed"] is None and d["skipped"] == "no bypass_vs_base comparison configured for metric2"
    assert d["ci_lower"]["metric1"] is not None and d["ci_lower"]["metric2"] is None


def test_cli_split_assign_and_graph_files_reproduce_the_single_file_run(tmp_path):
    pytest.importorskip("scipy.stats")
    W = world(6)
    write_world(tmp_path, W)
    out, ref = run_main(tmp_path)                                             # one assign file, one graph file
    split = tmp_path / "split"
    split.mkdir()
    write_world(split, W)
    a1 = save_assign(split / "a1.npz", W, ["base", "bypass"])
    a2 = save_assign(split / "a2.npz", W, ["bypass", "base_tokmean", "mat10k_km"])             # bypass is in both files
    g1 = save_graph(split / "g1.npz", W, ["base", "bypass"])
    g2 = save_graph(split / "g2.npz", W, ["bypass", "base_tokmean", "mat10k"])
    out2, res = run_main(split, extra=("--assign", f"{a1},{a2}", "--graph", f"{g1},{g2}"))
    assert res["sets"] == ref["sets"]
    assert (out2 / "concept_table.csv").read_text() == (out / "concept_table.csv").read_text()
    inputs = res["meta"]["inputs"]
    assert inputs["assign_files"] == [a1, a2] and inputs["assign_arm_files"]["bypass"] == [a1, a2]
    assert inputs["graph_files"] == [g1, g2] and inputs["graph_arm_files"]["base_tokmean"] == g2
    assert inputs["graph_duplicate_arms"]["bypass"]["n_entries_differ"] == {g2: 0}

    # the same arm with different neighbours in the two graph files: an error unless --graph_prefer says which copy
    nb2 = dict(W.nbrs, bypass=redraw_nbrs(W))
    n_diff = int((nb2["bypass"] != W.nbrs["bypass"]).sum())
    assert n_diff > 0
    g2b = save_graph(split / "g2b.npz", W, ["bypass", "base_tokmean", "mat10k"], nbrs=nb2)
    with pytest.raises(SystemExit, match="--graph_prefer"):
        run_main(split, extra=("--assign", f"{a1},{a2}", "--graph", f"{g1},{g2b}"))
    _, first = run_main(split, extra=("--assign", f"{a1},{a2}", "--graph", f"{g1},{g2b}", "--graph_prefer", "first"))
    assert first["sets"] == ref["sets"]                                       # g1's bypass = the original graph
    Wb = copy.copy(W)
    Wb.nbrs = nb2
    (tmp_path / "ref2").mkdir()
    write_world(tmp_path / "ref2", Wb)
    _, ref2 = run_main(tmp_path / "ref2")                                     # one graph file holding the tweaked bypass
    _, last = run_main(split, extra=("--assign", f"{a1},{a2}", "--graph", f"{g1},{g2b}", "--graph_prefer", "last"))
    assert last["sets"] == ref2["sets"] and last["sets"] != ref["sets"]
    assert last["meta"]["inputs"]["graph_duplicate_arms"]["bypass"]["n_entries_differ"] == {g1: n_diff}
    assert last["meta"]["inputs"]["graph_arm_files"]["bypass"] == g2b and last["meta"]["inputs"]["graph_prefer"] == "last"

    # conflicting labels in the two assign files: an error (there is no preference for labels)
    bad = W.labels["bypass"].copy()
    bad[5] += 1
    a2b = save_assign(split / "a2b.npz", W, ["bypass", "base_tokmean", "mat10k_km"], bypass=bad)
    with pytest.raises(SystemExit, match="different labels"):
        run_main(split, extra=("--assign", f"{a1},{a2b}", "--graph", f"{g1},{g2}"))


def test_cli_graph_list_with_a_missing_file(tmp_path):
    pytest.importorskip("scipy.stats")
    write_world(tmp_path, world(7))
    here, nope1, nope2 = (str(tmp_path / n) for n in ("knn_graph.npz", "nope1.npz", "nope2.npz"))
    with pytest.raises(SystemExit, match="do not exist"):                      # partial lists could mix up arm sources
        run_main(tmp_path, extra=("--graph", f"{here},{nope1}"))
    _, res = run_main(tmp_path, extra=("--graph", f"{nope1},{nope2}"))
    assert res["meta"]["metric2"]["ran"] is False and "none of" in res["meta"]["metric2"]["reason"]
