"""Unit tests for concept_steer_generate / concept_steer_judge: pure numpy/torch logic only,
no model, no GPU, no network."""
from __future__ import annotations

import numpy as np
import torch

from geoae.interp.concept_steer_generate import (
    contrast_direction, cosine, decode_direction, fit_class_cluster_stats, label_delta,
    select_handle_clusters,
)
from geoae.interp.concept_steer_judge import (
    boot_mean_ci, centred_ci, dataset_predicate, discover_handles, filter_scope,
    format_per_concept_row, matched_at_fluent_share, net_fluent_scores, pick_candidates,
    present_pairs,
)
from geoae.model import GeoAE

# ---------------------------------------------------------------------------
# handle selection
# ---------------------------------------------------------------------------

def test_fit_class_cluster_stats_class_balanced_precision():
    # 2 classes, 4 clusters. Class 0 has 100 fit tokens, class 1 has 10 -- an UNBALANCED class
    # size, so precision must be class-balanced (rate-based), not a raw count ratio.
    # cluster 0: 50 tokens of class 0 (from 5 docs) + 5 tokens of class 1 (from 2 docs)
    # cluster 1: only class-0 tokens
    cls = np.array([0] * 100 + [1] * 10)
    docid = np.array(list(range(100)) + list(range(100, 110)))
    cluster = np.zeros(110, dtype=np.int64)
    cluster[:50] = 0                       # class 0, cluster 0 (50 tokens, docs 0..49 -> 50 docs)
    cluster[50:100] = 1                    # class 0, cluster 1 (50 tokens, docs 50..99 -> 50 docs)
    cluster[100:105] = 0                   # class 1, cluster 0 (5 tokens, docs 100..104 -> 5 docs)
    cluster[105:110] = 2                   # class 1, cluster 2 (5 tokens, docs 105..109 -> 5 docs)
    n_ck, n_docs_ck, p_ck = fit_class_cluster_stats(cls, docid, cluster, n_classes=2, K=4)

    assert n_ck[0].tolist() == [50, 50, 0, 0]
    assert n_ck[1].tolist() == [5, 0, 5, 0]
    # cluster 0: distinct docs for class 0 = 50 (docs 0-49), for class 1 = 2 (only 2 unique
    # docids among 100..104? no -- 5 distinct docs 100..104) -> 5
    assert n_docs_ck[0, 0] == 50 and n_docs_ck[1, 0] == 5

    # class-balanced rate: f_{0,0} = 50/100 = 0.5, f_{1,0} = 5/10 = 0.5 -> EQUAL despite raw
    # counts (50 vs 5) being 10x apart. So p_ck[class 0, cluster 0] must be 0.5, not ~0.91.
    assert np.isclose(p_ck[0, 0], 0.5)
    assert np.isclose(p_ck[1, 0], 0.5)
    # cluster 1 is pure class 0 -> p_ck[0,1] = 1.0
    assert np.isclose(p_ck[0, 1], 1.0)
    # cluster 2 is pure class 1 -> p_ck[1,2] = 1.0
    assert np.isclose(p_ck[1, 2], 1.0)
    # cluster 3: no tokens of either class -> precision 0 (denom 0, not NaN/inf)
    assert p_ck[0, 3] == 0.0 and p_ck[1, 3] == 0.0


def test_select_handle_clusters_hub_and_thresholds():
    n_ck = np.array([100, 50, 30, 25, 19])           # cluster 3 just above min_count=20 fails
    n_docs_ck = np.array([10, 10, 2, 10, 10])        # cluster 2 fails min_docs=5
    p_ck = np.array([0.9, 0.5, 0.99, 0.99, 0.99])
    hub = np.array([True, False, False, False, False])   # cluster 0 excluded as a hub

    order, w = select_handle_clusters(n_ck, n_docs_ck, p_ck, hub, min_count=20, min_docs=5, top_k=5)
    # eligible: cluster 1 (n=50>=20, docs=10>=5, not hub); cluster 2 fails min_docs;
    # cluster 3 (n=25>=20, docs=10>=5); cluster 4 fails min_count (19<20); cluster 0 is a hub.
    assert set(order.tolist()) == {1, 3}
    # ordered by DESCENDING p_ck: cluster 3 (0.99) before cluster 1 (0.5)
    assert order.tolist() == [3, 1]
    # weights proportional to n_ck, summing to 1: n_ck[3]=25, n_ck[1]=50 -> w = [25/75, 50/75]
    np.testing.assert_allclose(w, [25 / 75, 50 / 75])
    assert np.isclose(w.sum(), 1.0)


def test_select_handle_clusters_top_k_ordering():
    n_ck = np.array([30, 30, 30, 30])
    n_docs_ck = np.array([10, 10, 10, 10])
    p_ck = np.array([0.1, 0.9, 0.5, 0.7])
    hub = np.zeros(4, dtype=bool)
    order, w = select_handle_clusters(n_ck, n_docs_ck, p_ck, hub, min_count=1, min_docs=1, top_k=2)
    assert order.tolist() == [1, 3]                  # top-2 by descending p_ck: 0.9, 0.7
    np.testing.assert_allclose(w, [0.5, 0.5])         # equal n_ck -> equal weights


def test_select_handle_clusters_no_eligible_returns_empty():
    n_ck = np.array([5, 5])
    n_docs_ck = np.array([1, 1])
    p_ck = np.array([0.9, 0.9])
    hub = np.array([False, False])
    order, w = select_handle_clusters(n_ck, n_docs_ck, p_ck, hub, min_count=20, min_docs=5, top_k=5)
    assert order.size == 0 and w.size == 0


# ---------------------------------------------------------------------------
# label-direction scaling
# ---------------------------------------------------------------------------

def test_label_delta_norm_matches_alpha_times_R():
    g = torch.Generator().manual_seed(0)
    v_c = torch.randn(37, generator=g) * 5.3          # arbitrary norm, must be normalised away
    for R_c in (0.5, 2.0, 7.3):
        for alpha in (0.0, 0.3, 1.0, 2.5):
            d = label_delta(v_c, R_c, alpha)
            assert torch.isclose(d.norm(), torch.tensor(alpha * R_c), atol=1e-5)
            if alpha > 0:
                # direction is v_c / ||v_c||, unaffected by v_c's own scale
                torch.testing.assert_close(d / d.norm(), v_c / v_c.norm(), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# judge option determinism
# ---------------------------------------------------------------------------

def test_pick_candidates_deterministic_and_identical_across_calls():
    all_concepts = ["Company", "Artist", "Athlete", "Building", "Village", "Animal"]
    a = pick_candidates("db14", "Company", 3, all_concepts)
    b = pick_candidates("db14", "Company", 3, all_concepts)
    assert a == b                                     # same inputs -> byte-identical output
    cand, order = a
    assert cand[0] == "Company"                       # target is always cand[0]
    assert len(cand) == 4 and len(set(cand)) == 4      # 3 distinct distractors, no repeats
    assert sorted(order) == [0, 1, 2, 3]               # a genuine permutation

    # different prompt index or concept -> (almost certainly) a different draw
    c = pick_candidates("db14", "Company", 4, all_concepts)
    d = pick_candidates("db14", "Artist", 3, all_concepts)
    assert c != a or d != a


def test_pick_candidates_options_identical_regardless_of_handle_or_alpha():
    # The generation script never passes handle/alpha into pick_candidates, so by construction
    # the same (dataset, concept, prompt) call made while iterating different handles/alphas
    # returns the identical candidate set and order -- this test pins that invariant.
    all_concepts = ["accountant", "architect", "dj", "professor", "nurse"]
    calls = [pick_candidates("biasbios", "dj", 5, all_concepts) for _ in range(5)]
    assert all(c == calls[0] for c in calls)


# ---------------------------------------------------------------------------
# net_fluent / usable scoring
# ---------------------------------------------------------------------------

def test_net_fluent_scores_excludes_nonfluent_and_nets_alpha0():
    # 2 concepts x 3 prompts.
    # concept 0: all 3 prompts fluent; hit=[1,1,0], hit0=[0,0,0] -> net = mean([1,1,0]) = 2/3
    # concept 1: prompt 0 non-fluent (excluded); prompts 1,2 fluent, hit=[1,1], hit0=[1,0]
    #            -> diffs = [0, 1] -> net = 0.5 (the alpha-0 hit at prompt 1 nets to 0, not +1)
    hit = np.array([[1, 1, 0], [0, 1, 1]])
    hit0 = np.array([[0, 0, 0], [0, 1, 0]])
    distinct2 = np.array([[0.9, 0.9, 0.9], [0.3, 0.9, 0.9]])   # concept 1 prompt 0 non-fluent
    sc = net_fluent_scores(hit, hit0, distinct2, threshold=0.6)
    assert np.isclose(sc["fluent_share"][0], 1.0)
    assert np.isclose(sc["fluent_share"][1], 2 / 3)
    assert np.isclose(sc["net_hit"][0], 2 / 3)
    assert np.isclose(sc["net_hit"][1], 0.5)
    assert np.isclose(sc["usable"][0], 1.0 * (2 / 3))
    assert np.isclose(sc["usable"][1], (2 / 3) * 0.5)


def test_net_fluent_scores_zero_fluent_gives_zero_usable_not_nan():
    hit = np.array([[1, 1]])
    hit0 = np.array([[0, 0]])
    distinct2 = np.array([[0.1, 0.2]])          # nothing fluent
    sc = net_fluent_scores(hit, hit0, distinct2)
    assert sc["fluent_share"][0] == 0.0
    assert np.isnan(sc["net_hit"][0])
    assert sc["usable"][0] == 0.0                # NOT nan, even though net_hit is NaN


# ---------------------------------------------------------------------------
# centred bootstrap CI
# ---------------------------------------------------------------------------

def test_centred_ci_contains_the_estimate():
    rng = np.random.default_rng(0)
    m = 5.0
    boots = m + rng.normal(loc=0.0, scale=0.5, size=5000)   # bootstrap replicates centred on m
    lo, hi = centred_ci(m, boots)
    assert lo <= m <= hi


def test_boot_mean_ci_matches_direct_mean_and_ignores_nan():
    rng = np.random.default_rng(1)
    vals = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    m, lo, hi = boot_mean_ci(vals, rng, n_boot=1000)
    assert np.isclose(m, 2.5)
    assert lo <= m <= hi


def test_boot_mean_ci_all_nan_returns_nan():
    rng = np.random.default_rng(2)
    m, lo, hi = boot_mean_ci(np.array([np.nan, np.nan]), rng)
    assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# matched fluent-share interpolation
# ---------------------------------------------------------------------------

def test_matched_at_fluent_share_interpolates_monotonised_curve():
    # fluent share DECREASES as alpha grows: 1.0, 0.8, 0.6, 0.4 (already monotone)
    fluent_share = np.array([1.0, 0.8, 0.6, 0.4])
    net_hit = np.array([0.1, 0.3, 0.5, 0.7])           # increases with alpha (more disruption)
    v70 = matched_at_fluent_share(fluent_share, net_hit, 0.6)
    assert np.isclose(v70, 0.5)                         # exact grid point
    v_mid = matched_at_fluent_share(fluent_share, net_hit, 0.7)
    assert np.isclose(v_mid, 0.4)                        # halfway between 0.3 (0.8) and 0.5 (0.6)


def test_matched_at_fluent_share_forces_monotone_before_interpolating():
    # a non-monotone fluent share (a blip UP at alpha[2]) must be flattened via
    # np.minimum.accumulate before interpolation, matching the generation script's
    # ppl-cost handling in cluster_steer_generate.py.
    fluent_share = np.array([1.0, 0.7, 0.8, 0.5])       # blip at index 2
    values = np.array([0.0, 1.0, 2.0, 3.0])
    # minimum.accumulate -> [1.0, 0.7, 0.7, 0.5]; interpolating at 0.7 must hit the EARLIEST
    # (alpha-ascending) point with share 0.7, i.e. index 1 (value 1.0), or something on the
    # segment between index 1 and 3 after sorting by share -- never index 2's raw value alone.
    v = matched_at_fluent_share(fluent_share, values, 0.7)
    assert np.isfinite(v)


def test_matched_at_fluent_share_out_of_range_is_nan():
    fluent_share = np.array([1.0, 0.9, 0.8])
    values = np.array([0.0, 1.0, 2.0])
    assert np.isnan(matched_at_fluent_share(fluent_share, values, 0.1))
    assert np.isnan(matched_at_fluent_share(fluent_share, values, 1.5))


# ---------------------------------------------------------------------------
# per-dataset scope slicing
# ---------------------------------------------------------------------------

def test_filter_scope_pooled_is_identity():
    sc = {"concepts": [("db14", "Company"), ("biasbios", "dj")],
          "fluent_share": np.array([0.5, 0.6]), "net_hit": np.array([0.1, 0.2]),
          "usable": np.array([0.05, 0.12])}
    out = filter_scope(sc, None)
    assert out is sc                                    # no copy needed for the pooled scope


def test_filter_scope_restricts_to_one_dataset_and_realigns_arrays():
    sc = {"concepts": [("db14", "Company"), ("biasbios", "dj"), ("db14", "Artist")],
          "fluent_share": np.array([0.5, 0.6, 0.7]), "net_hit": np.array([0.1, 0.2, 0.3]),
          "usable": np.array([0.05, 0.12, 0.21])}
    out = filter_scope(sc, dataset_predicate("db14"))
    assert out["concepts"] == [("db14", "Company"), ("db14", "Artist")]
    np.testing.assert_allclose(out["fluent_share"], [0.5, 0.7])
    np.testing.assert_allclose(out["net_hit"], [0.1, 0.3])
    np.testing.assert_allclose(out["usable"], [0.05, 0.21])
    # a dataset with no concepts in `sc` -> empty, not an error
    empty = filter_scope(sc, dataset_predicate("nope"))
    assert empty["concepts"] == [] and len(empty["usable"]) == 0


# ---------------------------------------------------------------------------
# label_z: decoder-weight direction mapping, and shared alpha*R_c scaling
# ---------------------------------------------------------------------------

def test_decode_direction_uses_decoder_weight_and_token_bias_cancels():
    # geoae/model.py: decoder = nn.Linear(latent_dim, hidden_size, bias=False), so
    # decode_direction(ae, v_z) is exactly W_dec @ v_z. The token-bypass table is added by
    # ae.decode(z, tok) for a REAL round trip; it depends only on the token, so decoding two
    # latents at the SAME token and subtracting must equal decode_direction on their raw
    # difference -- the bias cancels, confirming a bare direction needs no token id at all.
    torch.manual_seed(0)
    ae = GeoAE(hidden_size=6, latent_dim=4, n_clusters=2, nonlinearity="linear",
               token_bias_rows=3, vocab_size=10).eval()
    tok_ids = torch.tensor([0, 1, 2])
    table = torch.randn(3, 6)
    ae.load_token_bias(tok_ids, table)

    z1, z2 = torch.randn(1, 4), torch.randn(1, 4)
    same_tok = torch.tensor([1])                        # SAME token for both decodes
    h1 = ae.decode(z1, same_tok)
    h2 = ae.decode(z2, same_tok)
    via_decode_diff = (h1 - h2).squeeze(0)
    via_weight_only = decode_direction(ae, (z1 - z2).squeeze(0))
    torch.testing.assert_close(via_decode_diff, via_weight_only, atol=1e-5, rtol=1e-5)

    # also matches a plain matmul against the raw weight, with NO token involved at all
    v_z = torch.randn(4)
    torch.testing.assert_close(decode_direction(ae, v_z), ae.decoder.weight @ v_z,
                               atol=1e-6, rtol=1e-6)


def test_label_z_direction_rescaled_to_alpha_R_like_every_other_handle():
    # label_z's edit is decode_direction(...) fed straight into label_delta, exactly like
    # label's own v_c -- so the final edit norm must be alpha * R_c regardless of the
    # decoder's own scale (decoder columns are unit-norm at init, but this must hold for any
    # decoder weight).
    torch.manual_seed(1)
    ae = GeoAE(hidden_size=12, latent_dim=8, n_clusters=3, nonlinearity="linear").eval()
    v_z = torch.randn(8) * 3.7
    u = decode_direction(ae, v_z)
    for R_c in (0.5, 4.0):
        for alpha in (0.0, 0.25, 1.0, 2.0):
            d = label_delta(u, R_c, alpha)
            assert torch.isclose(d.norm(), torch.tensor(alpha * R_c), atol=1e-4)


# ---------------------------------------------------------------------------
# cosine bookkeeping
# ---------------------------------------------------------------------------

def test_cosine_known_angles():
    a = torch.tensor([1.0, 0.0, 0.0])
    assert np.isclose(cosine(a, torch.tensor([1.0, 0.0, 0.0])), 1.0)
    assert np.isclose(cosine(a, torch.tensor([0.0, 1.0, 0.0])), 0.0, atol=1e-6)
    assert np.isclose(cosine(a, torch.tensor([-1.0, 0.0, 0.0])), -1.0)
    # scale-invariant
    assert np.isclose(cosine(a, torch.tensor([5.0, 0.0, 0.0])), 1.0)


# ---------------------------------------------------------------------------
# judge: handle discovery and present-only pairs
# ---------------------------------------------------------------------------

def test_discover_handles_fixed_order_and_extras_appended():
    steered = {
        "db14": {"Company": {"base": {}, "label": {}, "bypass": {}},
                 "Artist": {"label_z": {}, "label": {}}},
        "biasbios": {"dj": {"zzz_future_handle": {}, "base": {}}},
    }
    got = discover_handles(steered)
    # fixed DISPLAY_ORDER = (label, label_z, base, bypass), unrecognised names appended sorted
    assert got == ["label", "label_z", "base", "bypass", "zzz_future_handle"]


def test_discover_handles_only_present_subset():
    steered = {"db14": {"Company": {"label_z": {}, "bypass": {}}}}
    assert discover_handles(steered) == ["label_z", "bypass"]


def test_present_pairs_restricts_to_available_handles():
    # only label and base are present -> only pairs where BOTH sides are present survive
    assert present_pairs(["label", "base"]) == (("base", "label"),)
    all_four = present_pairs(["label", "label_z", "base", "bypass"])
    assert all_four == (
        ("label_z", "label"), ("label_z", "bypass"), ("bypass", "base"),
        ("bypass", "label"), ("base", "label"),
    )
    assert present_pairs([]) == ()


# ---------------------------------------------------------------------------
# per-concept table: usable printed per 100, like every other table
# ---------------------------------------------------------------------------

def test_format_per_concept_row_prints_usable_per_100_not_as_a_fraction():
    row = {"dataset": "db14", "concept": "Company", "precision_base": 0.6123,
           "precision_bypass": None, "usable_label": 0.275, "usable_base": None}
    line = format_per_concept_row(row, ["label", "base"])
    assert "label=27.5" in line                # NOT "label=0.3" (the un-multiplied fraction)
    assert "base=n/a" in line
    assert "prec base=0.61" in line
    assert "bypass=n/a" in line


# ---------------------------------------------------------------------------
# base_dir / bypass_dir: cluster target means used as a constant direction
# ---------------------------------------------------------------------------

def test_contrast_direction_toy_floats():
    # d_c = m_c - mean(m_c' over ALL concepts that have this handle), mirroring label's
    # v_c = mu_c - mu_d (whose grand mean also includes c itself).
    m = {"A": 10.0, "B": 20.0, "C": 30.0}
    d = contrast_direction(m)
    assert set(d) == {"A", "B", "C"}
    assert d["A"] == 10.0 - 20.0 and d["B"] == 0.0 and d["C"] == 30.0 - 20.0


def test_contrast_direction_skips_missing_handle():
    # a concept with no target (None) for this codebook contributes NOTHING to the mean and
    # gets no direction of its own -- this is what "skip it for that handle" means.
    m = {"A": 10.0, "B": None, "C": 30.0}
    d = contrast_direction(m)
    assert set(d) == {"A", "C"}                 # B is absent, not defaulted to 0
    mean_of_present = (10.0 + 30.0) / 2
    assert d["A"] == 10.0 - mean_of_present
    assert d["C"] == 30.0 - mean_of_present


def test_contrast_direction_empty_and_all_missing():
    assert contrast_direction({}) == {}
    assert contrast_direction({"A": None, "B": None}) == {}


def test_contrast_direction_works_on_torch_tensors():
    g = torch.Generator().manual_seed(0)
    m = {"A": torch.randn(5, generator=g), "B": None, "C": torch.randn(5, generator=g)}
    d = contrast_direction(m)
    mean = (m["A"] + m["C"]) / 2
    torch.testing.assert_close(d["A"], m["A"] - mean)
    torch.testing.assert_close(d["C"], m["C"] - mean)
    assert "B" not in d


def test_base_dir_direction_rescaled_to_alpha_R_like_every_other_handle():
    m = {"A": torch.tensor([1.0, 2.0, 3.0]), "B": torch.tensor([4.0, -1.0, 0.0]),
         "C": torch.tensor([0.0, 0.0, 5.0])}
    d = contrast_direction(m)
    for R_c in (0.5, 3.0):
        for alpha in (0.0, 0.4, 1.0):
            delta = label_delta(d["A"], R_c, alpha)
            assert torch.isclose(delta.norm(), torch.tensor(alpha * R_c), atol=1e-5)


def test_present_pairs_includes_dir_handles_only_when_both_sides_present():
    # base_dir/bypass_dir pairs are absent unless the SPECIFIC other side is present too
    assert present_pairs(["label", "base_dir"]) == (("base_dir", "label"),)
    assert present_pairs(["base_dir", "bypass_dir"]) == (("bypass_dir", "base_dir"),)
    everything = present_pairs(["label", "label_z", "base", "bypass", "base_dir", "bypass_dir"])
    for pair in (("bypass_dir", "base_dir"), ("bypass_dir", "label"), ("base_dir", "label"),
                ("bypass_dir", "bypass"), ("base_dir", "base")):
        assert pair in everything
    # a lone new handle with no partner present contributes nothing
    assert present_pairs(["base_dir"]) == ()
