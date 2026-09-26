"""Unit tests for concept_remove_generate / concept_remove_judge: pure numpy/torch logic
only, no model, no GPU, no network."""
from __future__ import annotations

import numpy as np
import torch

from geoae.interp.concept_remove_generate import (
    calibrate_threshold, class_balanced_mean_excluding, class_dprime_and_range, diagnostic_cell,
    llr_scores,
)
from geoae.interp.concept_remove_judge import (
    clean_removal_score, collateral_score, drop_and_selectivity, matched_at_removal_level,
)
from geoae.interp.concept_steer_judge import pick_candidates
from geoae.model import GeoAE

# ---------------------------------------------------------------------------
# d' / salient / range construction
# ---------------------------------------------------------------------------

def test_class_dprime_and_range_salient_and_empty_interval():
    # 5 coordinates. Coord 0: huge mean gap, small spread -> high d', clearly salient.
    # Coord 1: same means, huge spread -> low d', NOT salient.
    # Coords 2-4: identical mu_c == mu_rest -> d'=0, never salient.
    mu_c = np.array([10.0, 10.0, 0.0, 0.0, 0.0])
    sd_c = np.array([0.5, 5.0, 1.0, 1.0, 1.0])
    mu_rest = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sd_rest = np.array([0.5, 5.0, 1.0, 1.0, 1.0])
    d, salient, lo, hi = class_dprime_and_range(mu_c, sd_c, mu_rest, sd_rest, percent=0.2, tao=2.0)
    assert d[0] > d[1] > 0
    assert np.allclose(d[2:], 0.0)
    # top 20% of 5 coords = 1 coordinate -> exactly coord 0
    assert salient.tolist() == [True, False, False, False, False]
    # salient coord gets a finite range: mu_c -/+ tao*sd_c
    assert np.isclose(lo[0], 10.0 - 2.0 * 0.5)
    assert np.isclose(hi[0], 10.0 + 2.0 * 0.5)
    # UNTOUCHED (non-salient) coordinates get the EMPTY interval: lo=+inf, hi=-inf, so no
    # finite value can ever satisfy lo <= a <= hi.
    for j in (1, 2, 3, 4):
        assert lo[j] == np.inf and hi[j] == -np.inf
        assert not (lo[j] <= 0.0 <= hi[j])


def test_class_dprime_and_range_percent_rounds_and_clamps():
    mu_c = np.arange(10.0)
    sd_c = np.ones(10)
    mu_rest = np.zeros(10)
    sd_rest = np.ones(10)
    # percent so small it would round to 0 -> clamped to at least 1 salient coordinate
    _, salient, _, _ = class_dprime_and_range(mu_c, sd_c, mu_rest, sd_rest, percent=0.01, tao=1.0)
    assert salient.sum() == 1
    # percent >= 1 -> every coordinate salient, none left with an empty interval
    _, salient_all, lo_all, _ = class_dprime_and_range(mu_c, sd_c, mu_rest, sd_rest, percent=1.0, tao=1.0)
    assert salient_all.all()
    assert np.isfinite(lo_all).all()


def test_class_balanced_mean_excluding():
    # class-balanced: each class's OWN mean counts once, regardless of scale -- contrasts
    # with a token-weighted pool, which this deliberately is NOT.
    means = {0: np.array([10.0, 0.0]), 1: np.array([20.0, 0.0]), 2: np.array([30.0, 0.0])}
    comp0 = class_balanced_mean_excluding(means, exclude=0)
    assert np.allclose(comp0, [25.0, 0.0])          # mean of classes 1,2 only
    comp1 = class_balanced_mean_excluding(means, exclude=1)
    assert np.allclose(comp1, [20.0, 0.0])           # mean of classes 0,2


# ---------------------------------------------------------------------------
# z --z_mode delta: x + W_dec(edit(z) - z), identity edit -> x unchanged
# ---------------------------------------------------------------------------

def test_z_delta_mode_identity_leaves_x_unchanged():
    # Mirrors make_edit_splice's z/delta branch: x2 = x + ae.decoder(edit_fn(z) - z).
    torch.manual_seed(0)
    ae = GeoAE(hidden_size=6, latent_dim=4, n_clusters=2, nonlinearity="linear").eval()
    x = torch.randn(3, 6)
    z = ae.encode(x)
    identity = lambda a: a
    x2 = x + ae.decoder(identity(z) - z)
    torch.testing.assert_close(x2, x)


def test_z_delta_mode_matches_the_formula_for_a_real_edit():
    torch.manual_seed(1)
    ae = GeoAE(hidden_size=6, latent_dim=4, n_clusters=2, nonlinearity="linear").eval()
    x = torch.randn(3, 6)
    z = ae.encode(x)
    scale_by_2 = lambda a: a * 2.0
    x2 = x + ae.decoder(scale_by_2(z) - z)
    # scale_by_2(z) - z == z, so this must equal x + decoder(z)
    torch.testing.assert_close(x2, x + ae.decoder(z))
    assert not torch.allclose(x2, x)                # a real edit DOES move x


def test_z_delta_mode_token_bias_cancels_like_decode_direction():
    # The token-bypass table depends only on the CURRENT token, not on z, so it is irrelevant
    # to a bare latent DIFFERENCE (edit(z) - z) -- same reasoning as
    # concept_steer_generate.decode_direction. Confirmed here via an actual round trip at the
    # SAME token: decode(z2,tok) - decode(z1,tok) == decoder(z2 - z1), bias cancelling exactly.
    torch.manual_seed(2)
    ae = GeoAE(hidden_size=6, latent_dim=4, n_clusters=2, nonlinearity="linear",
               token_bias_rows=3, vocab_size=10).eval()
    ae.load_token_bias(torch.tensor([0, 1, 2]), torch.randn(3, 6))
    z1, z2 = torch.randn(1, 4), torch.randn(1, 4)
    same_tok = torch.tensor([1])
    diff_via_decode = (ae.decode(z2, same_tok) - ae.decode(z1, same_tok)).squeeze(0)
    diff_via_weight = ae.decoder((z2 - z1).squeeze(0))
    torch.testing.assert_close(diff_via_decode, diff_via_weight, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# clean_removal / collateral / selectivity scoring
# ---------------------------------------------------------------------------

def test_drop_and_selectivity_basic():
    # target: unsteered all judged as c (hit=1); edited half stop being judged as c.
    hit_tgt_uns = np.array([1, 1, 1, 1])
    hit_tgt_edit = np.array([1, 0, 1, 0])
    # complement: unsteered all correct (hit=1, own class); edited unaffected.
    hit_comp_uns = np.array([1, 1, 1])
    hit_comp_edit = np.array([1, 1, 1])
    sc = drop_and_selectivity(hit_tgt_uns, hit_tgt_edit, hit_comp_uns, hit_comp_edit)
    assert np.isclose(sc["tgt_drop"], 0.5)
    assert np.isclose(sc["comp_drop"], 0.0)
    assert np.isclose(sc["selectivity"], 0.5)


def test_clean_removal_excludes_nonfluent_and_nets_unsteered():
    # 4 target prompts. Unsteered: all judged as c (hit=1), all fluent -> removed_u = 0.
    hit_tgt_uns = np.array([1, 1, 1, 1])
    d2_tgt_uns = np.array([0.9, 0.9, 0.9, 0.9])
    # Edited: prompt0 not-c & fluent (counts); prompt1 not-c but NON-fluent (excluded);
    # prompt2 still c (doesn't count); prompt3 not-c & fluent (counts).
    hit_tgt_edit = np.array([0, 0, 1, 0])
    d2_tgt_edit = np.array([0.9, 0.2, 0.9, 0.9])
    cr = clean_removal_score(hit_tgt_edit, d2_tgt_edit, hit_tgt_uns, d2_tgt_uns)
    # removed_e = share fluent & not-c = 2/4 = 0.5; removed_u = 0 -> clean_removal = 0.5
    assert np.isclose(cr, 0.5)


def test_collateral_counts_nonfluent_or_offclass_and_nets_unsteered():
    # 4 complement prompts. Unsteered: all fluent and on their own class -> bad_u = 0.
    hit_comp_uns = np.array([1, 1, 1, 1])
    d2_comp_uns = np.array([0.9, 0.9, 0.9, 0.9])
    # Edited: prompt0 fine; prompt1 non-fluent (bad); prompt2 fluent but off-class (bad);
    # prompt3 fine.
    hit_comp_edit = np.array([1, 1, 0, 1])
    d2_comp_edit = np.array([0.9, 0.2, 0.9, 0.9])
    col = collateral_score(hit_comp_edit, d2_comp_edit, hit_comp_uns, d2_comp_uns)
    assert np.isclose(col, 0.5)                       # 2/4 bad, net of 0 unsteered


def test_clean_removal_and_collateral_zero_when_edit_matches_unsteered():
    hit = np.array([1, 0, 1])
    d2 = np.array([0.9, 0.9, 0.2])
    assert np.isclose(clean_removal_score(hit, d2, hit, d2), 0.0)
    assert np.isclose(collateral_score(hit, d2, hit, d2), 0.0)


def test_matched_at_removal_level_interpolates_increasing_removal():
    # removal INCREASES with alpha (the opposite convention from matched_at_fluent_share)
    removal = np.array([0.0, 0.3, 0.6])
    collateral = np.array([0.0, 0.1, 0.4])
    v = matched_at_removal_level(removal, collateral, 0.3)
    assert np.isclose(v, 0.1)
    v_mid = matched_at_removal_level(removal, collateral, 0.45)
    assert np.isclose(v_mid, 0.25)                    # halfway between 0.1 (0.3) and 0.4 (0.6)


def test_matched_at_removal_level_out_of_range_is_nan():
    removal = np.array([0.1, 0.2, 0.3])
    collateral = np.array([0.0, 0.1, 0.2])
    assert np.isnan(matched_at_removal_level(removal, collateral, 0.9))


def test_matched_at_removal_level_sorts_mixed_operator_cells():
    # cells from different operators arrive in arbitrary order; interpolation must use them
    # sorted by removal, not carry a running max across operators
    removal = np.array([0.4, 0.1, 0.2])
    collateral = np.array([0.3, 0.0, 0.1])
    assert np.isclose(matched_at_removal_level(removal, collateral, 0.15), 0.05)
    assert np.isclose(matched_at_removal_level(removal, collateral, 0.3), 0.2)


# ---------------------------------------------------------------------------
# judge options: identical across substrate/op/alpha (they never enter the key at all)
# ---------------------------------------------------------------------------

def test_options_identical_regardless_of_substrate_op_alpha():
    # concept_remove_judge builds candidates from (dataset, true_class, prompt_index) ONLY --
    # substrate/op/alpha are never part of the key, so calling pick_candidates for the "h"
    # rm_range_comp cell and the "z" st_transport_a2.0 cell of the SAME prompt position must
    # give byte-identical output, by construction (no substrate/op/alpha parameter exists to
    # vary it).
    all_classes = ["Company", "Artist", "Athlete", "Building", "Animal"]
    a = pick_candidates("db14", "Company", 3, all_classes)
    b = pick_candidates("db14", "Company", 3, all_classes)     # same call, standing in for a
    assert a == b                                              # different substrate/op/alpha


# ---------------------------------------------------------------------------
# LLR position gate
# ---------------------------------------------------------------------------

def test_llr_scores_higher_for_class_c_like_draws():
    # 4 coordinates, all salient. Class c centred at 5 with sd 1; rest centred at 0 with sd 1.
    mu_c = torch.tensor([5.0, 5.0, 5.0, 5.0])
    sd_c = torch.ones(4)
    mu_rest = torch.zeros(4)
    sd_rest = torch.ones(4)
    salient = torch.ones(4, dtype=torch.bool)
    near_c = mu_c.unsqueeze(0).repeat(3, 1)          # sits exactly on mu_c
    near_rest = mu_rest.unsqueeze(0).repeat(3, 1)     # sits exactly on mu_rest
    llr_c = llr_scores(near_c, mu_c, sd_c, mu_rest, sd_rest, salient)
    llr_rest = llr_scores(near_rest, mu_c, sd_c, mu_rest, sd_rest, salient)
    assert (llr_c > 0).all()                          # more likely class-c than rest
    assert (llr_rest < 0).all()                       # more likely rest than class-c
    assert (llr_c > llr_rest).all()


def test_llr_scores_only_sums_salient_coordinates():
    # coord 0 salient (favours c), coord 1 NOT salient (would favour rest if it counted) --
    # LLR must ignore coord 1 entirely.
    mu_c = torch.tensor([5.0, 0.0])
    sd_c = torch.tensor([1.0, 1.0])
    mu_rest = torch.tensor([0.0, 5.0])
    sd_rest = torch.tensor([1.0, 1.0])
    salient = torch.tensor([True, False])
    a = torch.tensor([[5.0, 5.0]])                    # coord0 at mu_c, coord1 at mu_rest
    llr = llr_scores(a, mu_c, sd_c, mu_rest, sd_rest, salient)
    # if coord 1 (non-salient) were included it would CANCEL coord 0's positive contribution
    # (a symmetric setup); since it must be excluded, llr stays strongly positive.
    assert llr.item() > 1.0


def test_llr_scores_no_salient_coords_is_zero():
    mu_c = torch.tensor([5.0, 5.0])
    sd_c = torch.ones(2)
    mu_rest = torch.zeros(2)
    sd_rest = torch.ones(2)
    salient = torch.zeros(2, dtype=torch.bool)
    a = torch.randn(4, 2)
    llr = llr_scores(a, mu_c, sd_c, mu_rest, sd_rest, salient)
    assert torch.equal(llr, torch.zeros(4))


# ---------------------------------------------------------------------------
# position gate g=0 -> x unchanged, in both h and z (delta) form
# ---------------------------------------------------------------------------

def test_h_position_gate_zero_leaves_x_unchanged():
    # Mirrors make_edit_splice's h branch: x2 = x + g*(edit_fn(x) - x).
    x = torch.randn(5, 6)
    edit_fn = lambda a: a * 10.0 + 3.0                # an aggressive, very much NOT identity edit
    g = torch.zeros(5)
    x2 = x + g[:, None] * (edit_fn(x) - x)
    torch.testing.assert_close(x2, x)
    # and a non-zero gate DOES change x, confirming the test isn't vacuous
    x2_on = x + torch.ones(5)[:, None] * (edit_fn(x) - x)
    assert not torch.allclose(x2_on, x)


def test_z_delta_position_gate_zero_leaves_x_unchanged():
    # Mirrors make_edit_splice's z/delta branch: x2 = x + g*ae.decoder(edit_fn(z) - z).
    torch.manual_seed(3)
    ae = GeoAE(hidden_size=6, latent_dim=4, n_clusters=2, nonlinearity="linear").eval()
    x = torch.randn(5, 6)
    z = ae.encode(x)
    edit_fn = lambda a: a * 5.0
    g = torch.zeros(5)
    x2 = x + g[:, None] * ae.decoder(edit_fn(z) - z)
    torch.testing.assert_close(x2, x)
    g_on = torch.ones(5)
    x2_on = x + g_on[:, None] * ae.decoder(edit_fn(z) - z)
    assert not torch.allclose(x2_on, x)


def test_z_splice_position_gate_zero_leaves_z_unchanged():
    # Mirrors make_edit_splice's z/splice branch: z2 = a + g*(edit_fn(a) - a).
    z = torch.randn(5, 4)
    edit_fn = lambda a: a * 5.0
    g = torch.zeros(5)
    z2 = z + g[:, None] * (edit_fn(z) - z)
    torch.testing.assert_close(z2, z)


# ---------------------------------------------------------------------------
# diagnostic_cell: HELD-document gate fire-rate bookkeeping
# ---------------------------------------------------------------------------

def test_diagnostic_cell_separates_class_c_from_rest():
    # 2 salient coordinates, class c centred at 5, rest at 0, sd 1 both. "own" positions sit
    # exactly on mu_c (always inside [lo,hi] and always LLR>0); "other" positions sit exactly
    # on mu_rest (always outside [lo,hi] -- 5 sd away -- and always LLR<0).
    mu_c = torch.tensor([5.0, 5.0])
    sd_c = torch.ones(2)
    mu_rest = torch.zeros(2)
    sd_rest = torch.ones(2)
    salient = torch.ones(2, dtype=torch.bool)
    lo, hi = mu_c - 2.0, mu_c + 2.0                   # tao=2 range, matches class_dprime_and_range
    own = mu_c.unsqueeze(0).repeat(6, 1)
    other = mu_rest.unsqueeze(0).repeat(6, 1)
    cell = diagnostic_cell(own, other, mu_c, sd_c, mu_rest, sd_rest, salient, lo, hi, {"fixed": 0.0})
    assert cell["coord_gate_target"] == 1.0
    assert cell["coord_gate_comp"] == 0.0
    assert cell["pos_gate"]["fixed"]["target"] == 1.0
    assert cell["pos_gate"]["fixed"]["comp"] == 0.0
    assert cell["pos_gate"]["fixed"]["thresh"] == 0.0
    assert cell["pos_gate_auroc"] == 1.0              # LLR perfectly separates own from other


def test_diagnostic_cell_multiple_labelled_thresholds_share_one_llr_computation():
    # fixed=0.0 and a permissive threshold=-100 (fires on everything) must both be reported,
    # under their own labels, from the SAME underlying LLR scores.
    mu_c = torch.tensor([5.0, 5.0])
    sd_c = torch.ones(2)
    mu_rest = torch.zeros(2)
    sd_rest = torch.ones(2)
    salient = torch.ones(2, dtype=torch.bool)
    lo, hi = mu_c - 2.0, mu_c + 2.0
    own = mu_rest.unsqueeze(0).repeat(4, 1)           # deliberately "rest"-like -> LLR < 0
    other = mu_rest.unsqueeze(0).repeat(4, 1)
    cell = diagnostic_cell(own, other, mu_c, sd_c, mu_rest, sd_rest, salient, lo, hi,
                           {"fixed": 0.0, "calibrated": -100.0})
    assert set(cell["pos_gate"]) == {"fixed", "calibrated"}
    assert cell["pos_gate"]["fixed"]["target"] == 0.0           # LLR<0 everywhere -> never fires at thresh 0
    assert cell["pos_gate"]["calibrated"]["target"] == 1.0      # but always fires at a very permissive thresh
    assert cell["pos_gate"]["fixed"]["thresh"] == 0.0
    assert cell["pos_gate"]["calibrated"]["thresh"] == -100.0
    # AUROC is threshold-independent and appears once, not per label
    assert "pos_gate_auroc" in cell and isinstance(cell["pos_gate_auroc"], float)


def test_diagnostic_cell_high_coord_gate_low_pos_gate_is_the_motivating_case():
    # This is exactly the failure mode that motivated --pos_gate llr: the coordinate range is
    # WIDE (tao effectively huge here) so it fires almost everywhere on BOTH pools, while the
    # LLR position gate still correctly favours class-c-like points.
    mu_c = torch.tensor([5.0, 5.0])
    sd_c = torch.ones(2)
    mu_rest = torch.tensor([4.0, 4.0])                # close together -> coordinate ranges overlap heavily
    sd_rest = torch.ones(2)
    salient = torch.ones(2, dtype=torch.bool)
    lo, hi = mu_c - 6.0, mu_c + 6.0                    # very wide range: fires almost everywhere
    own = mu_c.unsqueeze(0).repeat(6, 1)
    other = mu_rest.unsqueeze(0).repeat(6, 1)
    cell = diagnostic_cell(own, other, mu_c, sd_c, mu_rest, sd_rest, salient, lo, hi, {"fixed": 0.0})
    assert cell["coord_gate_target"] == 1.0
    assert cell["coord_gate_comp"] == 1.0             # coordinate gate is NON-selective here
    assert cell["pos_gate"]["fixed"]["target"] == 1.0
    assert cell["pos_gate"]["fixed"]["comp"] == 0.0    # position gate stays selective


def test_diagnostic_cell_empty_pool_is_nan_not_error():
    mu_c = torch.tensor([5.0])
    sd_c = torch.ones(1)
    mu_rest = torch.zeros(1)
    sd_rest = torch.ones(1)
    salient = torch.ones(1, dtype=torch.bool)
    lo, hi = mu_c - 2.0, mu_c + 2.0
    empty = torch.zeros(0, 1)
    own = mu_c.unsqueeze(0).repeat(3, 1)
    cell = diagnostic_cell(own, empty, mu_c, sd_c, mu_rest, sd_rest, salient, lo, hi, {"fixed": 0.0})
    assert np.isnan(cell["coord_gate_comp"])
    assert np.isnan(cell["pos_gate"]["fixed"]["comp"])
    assert np.isnan(cell["pos_gate_auroc"])           # can't compute AUROC with one empty class
    assert cell["coord_gate_target"] == 1.0            # the non-empty pool still scores fine


# ---------------------------------------------------------------------------
# --gate_target_rate calibration
# ---------------------------------------------------------------------------

def test_calibrate_threshold_puts_fire_rate_at_requested_share():
    rng = np.random.default_rng(0)
    llr_values = rng.normal(loc=0.0, scale=5.0, size=2000)
    for rate in (0.5, 0.8, 0.2):
        thresh = calibrate_threshold(llr_values, rate)
        fire_rate = float(np.mean(llr_values > thresh))
        assert abs(fire_rate - rate) < 0.02              # within 2pp on 2000 samples


def test_calibrate_threshold_accepts_torch_tensor():
    vals = torch.linspace(0.0, 100.0, 101)                # 0, 1, 2, ..., 100
    thresh = calibrate_threshold(vals, 0.8)
    # 80% of 101 values above thresh -> thresh sits near the 20th percentile (~20)
    fire_rate = float((vals > thresh).float().mean())
    assert abs(fire_rate - 0.8) < 0.02


def test_calibrate_threshold_monotone_in_rate():
    rng = np.random.default_rng(1)
    llr_values = rng.normal(size=500)
    t_strict = calibrate_threshold(llr_values, 0.2)        # fires on fewer -> higher threshold
    t_loose = calibrate_threshold(llr_values, 0.8)         # fires on more -> lower threshold
    assert t_strict > t_loose
