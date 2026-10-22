"""Pair rules, exclusive-chunk counting, the document split and null sampling of the Atlas sibling test."""
import numpy as np
import pytest

import geoae.analysis.atlas_siblings as AS


# ---------------------------------------------------------------- names
def test_name_tokens_strip_plurals_and_punctuation():
    assert AS.name_tokens("Horror films") == ("horror", "film")
    assert AS.name_tokens("Children's literature") == ("children", "s", "literature")
    assert AS.name_tokens("Mathematics") == ("mathematic",)         # light stemmer: only used for containment tests
    assert AS.name_tokens("Class") == ("class",)                      # 'ss' endings are kept


def test_name_contains_is_token_wise_and_symmetric():
    assert AS.name_contains("Climate change", "Climate change policy")
    assert AS.name_contains("Climate change policy", "Climate change")
    assert AS.name_contains("Horror films", "Horror film")
    assert AS.name_contains("Film", "Horror films")
    assert not AS.name_contains("Gas", "Gasoline")                    # substring but not a token
    assert not AS.name_contains("Beer", "Distilled spirits")
    assert not AS.name_contains("Power tools", "Machine tools")        # share a token, neither contains the other
    assert not AS.name_contains("", "Anything")


def test_merge_duplicate_names_is_case_insensitive_and_keeps_order():
    names = ["Art exhibitions", "Jazz", "art  EXHIBITIONS", "Soul", "jazz", "Art exhibitions"]
    counts = [10, 5, 99, 7, 1, 3]
    units = AS.merge_duplicate_names(names, counts)
    assert units == [[0, 2, 5], [1, 4], [3]]
    assert AS.primary_member(units[0], counts) == 2                   # most frequent member supplies description / LCC
    assert AS.primary_member([1, 4], counts) == 1


# ---------------------------------------------------------------- packed co-occurrence
def _pack(C):
    n = len(C)
    return np.array([C[i, j] for i in range(n) for j in range(i, n)], dtype=np.uint32)


def test_packed_rows_matches_dense_symmetric_matrix():
    rng = np.random.default_rng(0)
    n = 9
    C = rng.integers(0, 50, (n, n))
    C = np.triu(C) + np.triu(C, 1).T
    packed = _pack(C)
    assert len(packed) == n * (n + 1) // 2
    rows = AS.packed_rows(packed, n, [0, 4, 8])
    np.testing.assert_array_equal(rows, C[[0, 4, 8]])
    i = np.array([5, 2, 7])
    j = np.array([2, 5, 7])
    np.testing.assert_array_equal(AS.packed_index(i, j, n)[:2], AS.packed_index(j, i, n)[:2])
    assert packed[AS.packed_index(np.array([3]), np.array([6]), n)][0] == C[3, 6]


def test_ppmi_zeroes_diagonal_rare_and_negative_pmi():
    n_total = 1000.0
    marg = np.array([100.0, 100.0, 100.0, 10.0])
    counts = np.array([[100, 50, 5, 30.0]])            # concept 0 with itself, 1, 2 (rare), 3
    own = np.array([100.0])
    P = AS.ppmi_profiles(counts, own, marg, n_total, min_count=20, zero_cols=[0])
    assert P[0, 0] == 0.0                                           # diagonal
    assert P[0, 1] == pytest.approx(np.log(50 * 1000 / (100 * 100)))
    assert P[0, 2] == 0.0                                           # below min_count although PMI = log(0.5 * 10) > 0
    assert P[0, 3] == pytest.approx(np.log(30 * 1000 / (100 * 10)))
    # a rare co-occurrence is zeroed even when its PMI is large, and a below-chance one is clipped at zero
    big = AS.ppmi_profiles(np.array([[100, 19.0, 0, 0]]), own, marg, n_total, min_count=20)
    assert big[0, 1] == 0.0
    below = AS.ppmi_profiles(np.array([[100, 25.0, 0, 0]]), own, np.array([100.0, 500.0, 1, 1]), n_total, min_count=20)
    assert below[0, 1] == 0.0                                       # 25 observed vs 50 expected -> negative PMI -> 0
    assert (AS.ppmi_profiles(counts, own, marg, n_total) >= 0).all()


def test_cosine_matrix_and_first_order_stats():
    X = np.array([[1.0, 0], [2.0, 0], [0, 3.0], [0, 0]])
    S = AS.cosine_matrix(X)
    assert S[0, 1] == pytest.approx(1.0) and S[0, 2] == pytest.approx(0.0) and S[3, 3] == 0.0
    ci = np.array([100.0, 10.0])
    cij = np.array([[100.0, 5.0], [5.0, 10.0]])
    ov, jac = AS.first_order_stats(cij, ci)
    assert ov[0, 1] == pytest.approx(5 / 10)                         # share of the SMALLER concept's chunks
    assert jac[0, 1] == pytest.approx(5 / (100 + 10 - 5))


# ---------------------------------------------------------------- pair rules
def test_top_k_pairs_mutual_flags_and_masking():
    #           0     1     2     3
    sim = np.array([[1.0, 0.9, 0.2, 0.1],
                    [0.9, 1.0, 0.3, 0.8],
                    [0.2, 0.3, 1.0, 0.7],
                    [0.1, 0.8, 0.7, 1.0]])
    allowed = np.ones((4, 4), bool)
    np.fill_diagonal(allowed, False)
    pairs = AS.top_k_pairs(sim, allowed, 1)
    # 0 -> 1, 1 -> 0, 2 -> 3, 3 -> 1
    assert pairs == {(0, 1, True), (2, 3, False), (1, 3, False)}
    allowed[0, 1] = allowed[1, 0] = False                            # mask the best pair
    pairs = AS.top_k_pairs(sim, allowed, 1)
    assert (0, 1, True) not in pairs and not any(p[:2] == (0, 1) for p in pairs)
    allowed[:] = False                                               # nothing allowed -> nothing returned
    assert AS.top_k_pairs(sim, allowed, 2) == set()


def test_synonym_guard_blocks_containment_and_high_jaccard():
    names = ["Climate change", "Climate change policy", "Gas", "Gasoline", "Beer", "Distilled spirits"]
    jac = np.zeros((6, 6))
    jac[4, 5] = jac[5, 4] = 0.4
    ok = AS.synonym_guard(names, jac, 0.15)
    assert not ok[0, 1] and not ok[1, 0]                             # containment
    assert ok[2, 3]                                                  # 'Gas' is only a substring of 'Gasoline'
    assert not ok[4, 5]                                              # Jaccard above the threshold
    assert ok[0, 4] and not ok[2, 2]                                 # diagonal never allowed


# ---------------------------------------------------------------- chunk bookkeeping
def _toy_labels():
    # 6 chunks; concept ids 0..5. units: u0 = {0, 3} (merged duplicate), u1 = {1}, u2 = {2}
    labels = [[0], [0, 1], [1], [3, 1], [2, 0], [5]]
    return AS.ragged_to_csr(labels)


def test_ragged_to_csr_roundtrip():
    ptr, flat = _toy_labels()
    assert ptr.tolist() == [0, 1, 3, 4, 6, 8, 9]
    assert flat.tolist() == [0, 0, 1, 1, 3, 1, 2, 0, 5]
    p2, f2 = AS.ragged_to_csr([])
    assert p2.tolist() == [0] and len(f2) == 0


def test_exclusive_counts_merge_ids_and_report_shared():
    ptr, flat = _toy_labels()
    M = AS.labels_csr(ptr, flat, 6, [[0, 3], [1], [2]], 6)
    # u0 chunks: 0, 1, 3 (via id 3), 4 ; u1 chunks: 1, 2, 3 ; u2 chunks: 4
    assert M.indices[M.indptr[0]:M.indptr[1]].tolist() == [0, 1, 3, 4]
    a_only, b_only, both = AS.chunk_sets(M, 0, 1)
    assert a_only.tolist() == [0, 4] and b_only.tolist() == [2] and both.tolist() == [1, 3]
    ex = AS.exclusive_counts(M, [(0, 1), (1, 2), (0, 2)])
    assert ex.tolist() == [[2, 1, 2], [3, 1, 0], [3, 0, 1]]
    # a chunk carrying two ids of the SAME unit counts once
    ptr2, flat2 = AS.ragged_to_csr([[0, 3], [1]])
    M2 = AS.labels_csr(ptr2, flat2, 2, [[0, 3], [1]], 6)
    assert M2.nnz == 2 and M2.data.tolist() == [1, 1]


def test_split_half_is_deterministic_document_level_and_balanced():
    docs = np.repeat(np.arange(20000), 3)                            # 3 chunks per document
    s1 = AS.split_half(docs, 0)
    np.testing.assert_array_equal(s1, AS.split_half(docs, 0))
    assert (s1.reshape(-1, 3) == s1.reshape(-1, 3)[:, :1]).all()     # every chunk of a doc on the same side
    assert abs(s1.mean() - 0.5) < 0.02
    assert (AS.split_half(docs, 0) != AS.split_half(docs, 1)).mean() > 0.3   # the seed changes the split
    # consecutive ids must not alternate: hash, not parity
    assert (AS.split_half(np.arange(1000), 0)[:-1] != AS.split_half(np.arange(1000), 0)[1:]).mean() < 0.7


# ---------------------------------------------------------------- nulls
def test_sample_null_pairs_respects_every_constraint():
    rng = np.random.default_rng(0)
    n = 300
    lcc = np.array(list("ABC"))[rng.integers(0, 3, n)]
    log_count = rng.normal(8, 1, n)
    sib = []
    for _ in range(25):
        L = rng.choice(list("ABC"))
        idx = np.flatnonzero(lcc == L)
        a, b = rng.choice(idx, 2, replace=False)
        sib.append((int(a), int(b)))
    forbidden = {frozenset(p) for p in sib}
    eligible = np.ones(n, bool)
    eligible[:10] = False

    def allowed(x, y):
        return (x + y) % 7 != 0

    for mode in ("same", "cross"):
        out = AS.sample_null_pairs(sib, log_count, lcc, eligible, forbidden, mode, np.random.default_rng(1), 0.1, allowed)
        assert len(out) == len(sib) and all(o is not None for o in out)
        seen = set()
        for (a, b), (x, y, tol) in zip(sib, out):
            assert eligible[x] and eligible[y] and x != y
            assert lcc[x] == lcc[a]
            assert (lcc[y] == lcc[a]) == (mode == "same")
            assert abs(log_count[x] - log_count[a]) <= tol + 1e-12 and abs(log_count[y] - log_count[b]) <= tol + 1e-12
            assert frozenset((x, y)) not in forbidden and allowed(x, y)
            assert frozenset((x, y)) not in seen
            seen.add(frozenset((x, y)))
    # deterministic given the generator
    a1 = AS.sample_null_pairs(sib, log_count, lcc, eligible, forbidden, "same", np.random.default_rng(5), 0.1, allowed)
    a2 = AS.sample_null_pairs(sib, log_count, lcc, eligible, forbidden, "same", np.random.default_rng(5), 0.1, allowed)
    assert a1 == a2


def test_sample_null_pairs_reports_none_when_impossible():
    lcc = np.array(["A", "A", "A"])
    out = AS.sample_null_pairs([(0, 1)], np.zeros(3), lcc, np.ones(3, bool), set(), "cross",
                               np.random.default_rng(0), 0.1, None, max_tries=30)
    assert out == [None]                                              # there is no other letter to draw y from


# ---------------------------------------------------------------- frozen list
def test_pair_list_hash_is_order_invariant_and_tag_sensitive():
    p1 = [dict(a=1, b=2, tag="E"), dict(a=5, b=3, tag="C")]
    p2 = [dict(a=3, b=5, tag="C"), dict(a=2, b=1, tag="E")]
    assert AS.pair_list_sha256(p1) == AS.pair_list_sha256(p2)
    assert AS.pair_list_sha256(p1) != AS.pair_list_sha256([dict(a=1, b=2, tag="both"), dict(a=5, b=3, tag="C")])
    assert AS.pair_list_sha256(p1) != AS.pair_list_sha256(p1[:1])


def test_stream_fingerprint_depends_on_order_and_content():
    f = AS.stream_fingerprint([1, 2, 3], [0, 1, 0])
    assert f == AS.stream_fingerprint(np.array([1, 2, 3]), np.array([0, 1, 0]))
    assert f != AS.stream_fingerprint([2, 1, 3], [1, 0, 0])
    assert f != AS.stream_fingerprint([1, 2, 4], [0, 1, 0])


# ---------------------------------------------------------------- cache helpers
def test_rows_of_chunks_concatenates_row_ranges_in_chunk_order():
    row_start = np.array([0, 3, 3, 7, 10])                           # chunk 1 has no rows
    np.testing.assert_array_equal(AS.rows_of_chunks(row_start, np.array([2, 0])), [3, 4, 5, 6, 0, 1, 2])
    np.testing.assert_array_equal(AS.rows_of_chunks(row_start, np.array([1])), [])
    np.testing.assert_array_equal(AS.rows_of_chunks(row_start, np.array([3, 0, 2])), [7, 8, 9, 0, 1, 2, 3, 4, 5, 6])
    assert AS.rows_of_chunks(row_start, np.array([], dtype=int)).dtype == np.int64


def test_cap_chunks_is_a_sorted_subset():
    ch = np.arange(100, 200)
    rng = np.random.default_rng(0)
    out = AS.cap_chunks(ch, 30, rng)
    assert len(out) == 30 and (np.diff(out) > 0).all() and set(out) <= set(ch)
    assert AS.cap_chunks(ch, 500, rng) is ch


def test_atlas_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n_rows_per = [2, 0, 3, 1]
    row_start = np.r_[0, np.cumsum(n_rows_per)]
    R = int(row_start[-1])
    np.save(tmp_path / "rows.npy", rng.normal(size=(R, 4)).astype(np.float16))
    for nm, dt in (("rows_tok", np.int32), ("rows_nxt", np.int32), ("rows_chunk", np.int32)):
        np.save(tmp_path / f"{nm}.npy", np.arange(R, dtype=dt))
    lab_ptr, lab_ids = AS.ragged_to_csr([[0], [0, 1], [1], [2]])
    np.savez(tmp_path / "chunks.npz", doc_int_id=np.array([7, 7, 8, 9]), chunk_id=np.arange(4), n_tok=np.full(4, 50),
             row_start=row_start, label_ptr=lab_ptr, label_ids=lab_ids, tok_ptr=np.arange(5), tok_flat=np.arange(4),
             fingerprint=np.asarray("abc"))
    c = AS.AtlasCache(tmp_path)
    assert c.n_chunks == 4 and c.n_rows == R and c.fingerprint == "abc"
    idx, h, tok, nxt = c.gather(np.array([2, 0]))
    assert idx.tolist() == [2, 3, 4, 0, 1] and h.shape == (5, 4) and tok.tolist() == [2, 3, 4, 0, 1]
    M = c.membership([[0], [1]], n_concepts=3)
    assert AS.exclusive_counts(M, [(0, 1)]).tolist() == [[1, 1, 1]]
    fit = c.fit_mask(0)
    assert fit[0] == fit[1]                                           # same document, same side
    sets = AS.pair_chunk_sets(M, 0, 1, fit)
    assert sorted(np.concatenate([sets["a_fit"], sets["a_eval"]]).tolist()) == [0] and sets["both"].tolist() == [1]


def test_pair_exclusive_report_threshold_and_merged_units(tmp_path):
    n = 12
    labels = [[0]] * 5 + [[1]] * 3 + [[0, 1]] * 2 + [[7]] + [[9]]          # 0 only x5, 1 only x3, both x2
    ptr, ids = AS.ragged_to_csr(labels)
    row_start = np.arange(n + 1)
    np.save(tmp_path / "rows.npy", np.zeros((n, 2), np.float16))
    for nm in ("rows_tok", "rows_nxt", "rows_chunk"):
        np.save(tmp_path / f"{nm}.npy", np.zeros(n, np.int32))
    np.savez(tmp_path / "chunks.npz", doc_int_id=np.arange(n), chunk_id=np.zeros(n), n_tok=np.full(n, 9),
             row_start=row_start, label_ptr=ptr, label_ids=ids, tok_ptr=np.arange(n + 1), tok_flat=np.zeros(n),
             fingerprint=np.asarray("x"))
    cache = AS.AtlasCache(tmp_path)
    recs = [dict(a=0, b=1, a_ids=[0], b_ids=[1]), dict(a=0, b=7, a_ids=[0, 9], b_ids=[7])]
    rep, uid, M = AS.pair_exclusive_report(cache, recs, 3, n_concepts=10)
    assert (rep[0]["a_only"], rep[0]["b_only"], rep[0]["shared"], rep[0]["passes"]) == (5, 3, 2, True)
    assert rep[0]["a_total"] == 7 and rep[0]["b_total"] == 5
    assert rep[1]["a_only"] == 8 and rep[1]["b_only"] == 1 and not rep[1]["passes"]      # merged unit {0, 9}
    assert AS.pair_exclusive_report(cache, recs, 1, n_concepts=10)[0][1]["passes"]
