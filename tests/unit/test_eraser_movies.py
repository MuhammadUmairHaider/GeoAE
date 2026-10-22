"""CPU-only tests for geoae.analysis.eraser_movies (no model, no network)."""
import numpy as np
import pytest

import geoae.analysis.eraser_movies as em


def test_alignment_leading_space_and_split_word():
    lines = [["the", "movie", "unbelievably", "bad"], ["ok"]]
    text, spans = em.text_with_spans(lines)
    assert text == "the movie unbelievably bad\nok"
    assert [text[a:b] for a, b in spans] == ["the", "movie", "unbelievably", "bad", "ok"]
    # BOS (0,0); 'the'; ' movie' with leading space; word split in 3 subwords; ' bad'; '\n'; 'ok'
    i = text.index("unbelievably")
    offs = [(0, 0), (0, 3), (3, 9), (i, i + 2), (i + 2, i + 7), (i + 7, i + 12), (i + 12, i + 16), (i + 16, i + 17), (i + 17, i + 19)]
    al = em.align_offsets(text, offs, spans)
    assert al[0] is None and al[7] is None
    assert al[1] == (0, 1) and al[2] == (1, 2)
    assert al[3] == al[4] == al[5] == (2, 3)
    assert al[6] == (3, 4) and al[8] == (4, 5)
    kp, pr, pt = em.alignment_pairs(al)
    assert list(kp) == [1, 2, 3, 4, 5, 6, 8]
    sc, cnt = em.aggregate_to_tokens(np.array([1, 2, 3, 6, 9, 4, 5.0]), pr, pt, 5)
    assert list(cnt) == [1, 1, 3, 1, 1]
    assert sc[2] == pytest.approx(6.0) and sc[4] == 5.0


def test_merged_token_spans_two_eraser_tokens():
    text, spans = em.text_with_spans([["a", "b"]])
    al = em.align_offsets(text, [(0, 3)], spans)
    assert al == [(0, 2)]


def test_evidence_index_convention():
    toks = "a bad movie but great acting".split()
    ev = [[{"docid": "d", "start_token": 1, "end_token": 3, "text": "bad movie"}],
          [{"docid": "d", "start_token": 4, "end_token": 6, "text": "great acting"}]]
    assert em.check_evidence_convention(ev, toks) == (2, 0)
    m = em.rationale_mask(ev, len(toks), "d")
    assert list(m.nonzero()[0]) == [1, 2, 4, 5]          # end exclusive
    ev[0][0]["end_token"] = 2
    assert em.check_evidence_convention(ev, toks) == (2, 1)


def test_build_text_remove_and_keep():
    lines = [["a", "b", "c"], ["d"], ["e", "f"]]
    keep = np.array([1, 0, 1, 0, 1, 1], bool)
    assert em.build_text(lines) == "a b c\nd\ne f"
    assert em.build_text(lines, keep) == "a c\ne f"            # order kept, empty line dropped
    assert em.build_text(lines, ~keep) == "b\nd"


def test_probe_decomposition_equals_doc_logit():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(17, 6))
    w, b = rng.normal(size=6), 0.3
    mu, sd = rng.normal(size=6), rng.uniform(0.5, 2, size=6)
    s = em.probe_row_scores(R, w, mu, sd)
    doc_logit = ((R.mean(0) - mu) / sd) @ w + b
    assert s.mean() + b == pytest.approx(doc_logit)


def test_cluster_weights_toy():
    w = em.cluster_weights([3, 0, 1], [0, 3, 1])
    K = 3
    assert w[0] == pytest.approx(np.log(4 / (4 + K)) - np.log(1 / (4 + K)))
    assert w[0] > 0 > w[1] and w[2] == pytest.approx(0.0)


def test_auprc_orientation_flips_ranking():
    mask = np.array([1, 1, 0, 0, 0, 0], bool)
    s = np.array([5, 4, 1, 0, -1, -2.0])
    hi = em.oriented_auprc(s, mask, +1)
    lo = em.oriented_auprc(s, mask, -1)
    assert hi == pytest.approx(1.0) and lo < 0.5
    rng = np.random.default_rng(0)
    assert list(em.rank_order(s, rng)[:2]) == [0, 1]
    assert list(em.rank_order(-s, rng)[:2]) == [5, 4]
    assert em.top_mask(s, 0.34, rng).sum() == 3      # ceil(.34*6)


def test_aopc_arithmetic():
    per_bin, a = em.aopc(0.9, [0.8, 0.5, 0.1])
    assert per_bin == pytest.approx([0.1, 0.4, 0.8]) and a == pytest.approx(1.3 / 3)


def test_paired_stats_detects_shift():
    rng = np.random.default_rng(0)
    b = rng.normal(size=60)
    r = em.paired_stats(b + 0.5, b, 300, rng)
    assert r["mean_diff"] == pytest.approx(0.5) and r["ci_lo"] > 0 and r["wilcoxon_p"] < 1e-6
    assert em.paired_stats(b, b, 50, rng)["wilcoxon_p"] == 1.0
