"""Unit tests for geoae/interp/chain_control.py and the extra-splice filter of _shared.build_joint_correct_set:
toy AE / SAE, no language model, no GPU, no network."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from geoae.interp import _shared as shared
from geoae.interp import neuronlens as nl
from geoae.interp.chain_control import (ALL_ARMS, ARM_ACTS, ORDERS, Chain, allocate, arms_of, control_rng, fit_edits,
                                        selectivity)

D, L, F, V, BOS = 12, 20, 40, 7, 0


class ToyAE:
    """Token-bypass AE in the standardised space: z = gelu((x - b[tok]) W_e), x_hat = z W_d + b[tok]."""
    has_token_bias = True

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.We, self.Wd = torch.randn(D, L, generator=g) / 3, torch.randn(L, D, generator=g) / 3
        self.b = torch.randn(V, D, generator=g) / 2

    def encode(self, x, tok=None):
        return torch.nn.functional.gelu((x - self.b[tok]) @ self.We)

    def decode(self, z, tok=None):
        return z @ self.Wd + self.b[tok]


class ToySAE:
    """JumpReLU SAE in raw units."""

    def __init__(self, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.We, self.Wd = torch.randn(D, F, generator=g) / 2, torch.randn(F, D, generator=g) / 4
        self.be, self.bd, self.thr = torch.randn(F, generator=g) / 4, torch.randn(D, generator=g) / 4, 0.3

    def encode_raw(self, x, tok=None):
        pre = x @ self.We + self.be
        return torch.relu(pre) * (pre > self.thr)

    def decode_raw(self, c, tok=None):
        return c @ self.Wd + self.bd


class FakeTap:
    def __init__(self, ids):
        self.ids = ids

    def ids_for(self, hs):
        return self.ids.reshape(-1)


@pytest.fixture
def setup():
    torch.manual_seed(0)
    mean, std = torch.randn(D), torch.rand(D) + 0.5
    chain = Chain(ToyAE(), mean, std, ToySAE(), BOS, torch.device("cpu"))
    x = torch.randn(30, D) * 2
    tok = torch.randint(1, V, (30,))
    tok[:3] = BOS
    return chain, x, tok


def test_arms_of_skips_arms_whose_stage_is_absent():
    assert arms_of(ORDERS["ae"], ALL_ARMS) == ["h_first", "z", "h_last"]
    assert arms_of(ORDERS["sae"], ALL_ARMS) == ["h_first", "s", "s_top", "h_last"]
    assert arms_of(ORDERS["ae_sae"], ALL_ARMS) == list(ALL_ARMS)


def test_unedited_chain_is_the_composition_in_the_stated_order(setup):
    chain, x, tok = setup
    ae = chain.ae_stage(x, tok)[1]
    sae = chain.sae_stage(x, tok)[1]
    assert torch.allclose(chain.rows(x, tok, ORDERS["ae"]), ae)
    assert torch.allclose(chain.rows(x, tok, ORDERS["sae"]), sae)
    assert torch.allclose(chain.rows(x, tok, ORDERS["ae_sae"]), chain.sae_stage(ae, tok)[1])
    assert torch.allclose(chain.rows(x, tok, ORDERS["sae_ae"]), chain.ae_stage(sae, tok)[1])
    assert not torch.allclose(chain.rows(x, tok, ORDERS["ae_sae"]), chain.rows(x, tok, ORDERS["sae_ae"]), atol=1e-3)


def test_ae_stage_matches_the_formula_in_standardised_space(setup):
    chain, x, tok = setup
    xn = (x - chain.mean) / chain.std
    z = chain.ae.encode(xn, tok)
    assert torch.allclose(chain.ae_stage(x, tok)[0], z)
    assert torch.allclose(chain.ae_stage(x, tok)[1], chain.ae.decode(z, tok) * chain.std + chain.mean)


def test_bos_rows_pass_through_the_sae_only(setup):
    chain, x, tok = setup
    y = chain.sae_stage(x, tok)[1]
    assert torch.equal(y[tok == BOS], x[tok == BOS])
    assert not torch.allclose(y[tok != BOS], x[tok != BOS], atol=1e-3)
    ae_out = chain.ae_stage(x, tok)[1]                      # the AE still acts at BOS, the SAE after it does not
    assert torch.equal(chain.rows(x, tok, ORDERS["ae_sae"])[tok == BOS], ae_out[tok == BOS])


@pytest.mark.parametrize("order", list(ORDERS))
def test_identity_edit_changes_nothing(setup, order):
    chain, x, tok = setup
    ref = chain.rows(x, tok, ORDERS[order])
    for arm in arms_of(ORDERS[order], ALL_ARMS):
        assert torch.equal(chain.rows(x, tok, ORDERS[order], arm, lambda a: a), ref), (order, arm)


@pytest.mark.parametrize("order", list(ORDERS))
def test_acts_are_exactly_what_each_arm_edit_receives(setup, order):
    """The statistics are fitted on Chain.acts; they must be the tensors the edit function is handed inside the chain."""
    chain, x, tok = setup
    acts = chain.acts(x.numpy(), tok.numpy(), ORDERS[order], chunk=7)
    assert set(acts) == {"h_first", "h_last"} | {{"ae": "z", "sae": "s"}[st] for st in ORDERS[order]}
    for arm in arms_of(ORDERS[order], ALL_ARMS):
        seen = []
        chain.rows(x, tok, ORDERS[order], arm, lambda a: (seen.append(a.clone()), a)[1])
        assert len(seen) == 1, (order, arm)
        assert np.allclose(seen[0].numpy(), acts[ARM_ACTS[arm]], atol=1e-5), (order, arm)


def test_an_edit_reaches_only_its_own_stage_and_then_passes_the_other_reconstruction(setup):
    chain, x, tok = setup
    bump = lambda a: a + 0.7
    z_edit_then_sae = chain.sae_stage(chain.ae_stage(x, tok, bump)[1], tok)[1]
    assert torch.allclose(chain.rows(x, tok, ORDERS["ae_sae"], "z", bump), z_edit_then_sae)
    s_edit_last = chain.sae_stage(chain.ae_stage(x, tok)[1], tok, bump)[1]
    assert torch.allclose(chain.rows(x, tok, ORDERS["ae_sae"], "s", bump), s_edit_last)
    assert torch.allclose(chain.rows(x, tok, ORDERS["ae_sae"], "s_top", bump), s_edit_last)      # s_top edits the same stage
    assert torch.allclose(chain.rows(x, tok, ORDERS["ae_sae"], "h_first", bump), chain.rows(x + 0.7, tok, ORDERS["ae_sae"]))
    assert torch.allclose(chain.rows(x, tok, ORDERS["ae_sae"], "h_last", bump), chain.rows(x, tok, ORDERS["ae_sae"]) + 0.7)


@pytest.mark.parametrize("order,arm", [("ae_sae", "s"), ("sae_ae", "z"), ("ae", "z"), ("sae", "s")])
def test_full_space_shift_in_the_last_stage_equals_the_same_shift_of_the_chain_output(setup, order, arm):
    """Linear decoder: shifting every coordinate of the LAST space by its class-mean difference is the residual shift by the
    class-mean difference of the chain outputs. The statistics come from the answer position, where every document
    carries the SAME token (the prompt's closing token), so the bypass AE's per-token mean cancels in the difference;
    with mixed tokens the two class-mean differences would also differ by the classes' mean token vectors."""
    chain, x, tok = setup
    tok = torch.full_like(tok, 3)                           # one token, as at the answer position
    is_c = np.arange(len(x)) < len(x) // 2
    acts = chain.acts(x.numpy(), tok.numpy(), ORDERS[order])
    A, H = acts[ARM_ACTS[arm]], acts["h_last"]
    r_code, r_res = A[is_c].mean(0) - A[~is_c].mean(0), H[is_c].mean(0) - H[~is_c].mean(0)
    inf = np.full(A.shape[1], np.inf, np.float32)
    out_code = chain.rows(x, tok, ORDERS[order], arm, nl.shift_edit(r_code, -inf, inf, 0.5, "cpu"))
    inf = np.full(H.shape[1], np.inf, np.float32)
    out_res = chain.rows(x, tok, ORDERS[order], "h_last", nl.shift_edit(r_res, -inf, inf, 0.5, "cpu"))
    assert torch.allclose(out_code, out_res, atol=1e-4)


def test_hook_fn_reshapes_uses_tap_ids_and_records_the_answer_position(setup):
    chain, x, tok = setup
    hs = x.reshape(5, 6, D).double()
    rec = []
    out = chain.hook_fn(ORDERS["sae_ae"], FakeTap(tok.reshape(5, 6)), record=rec)(hs)
    assert out.shape == hs.shape and out.dtype == hs.dtype
    assert torch.allclose(out.float().reshape(-1, D), chain.rows(x, tok, ORDERS["sae_ae"]), atol=1e-5)
    assert len(rec) == 1 and torch.allclose(rec[0], x.reshape(5, 6, D)[:, -1])
    chain.hook_fn(ORDERS["sae_ae"], FakeTap(tok.reshape(5, 6)), record=rec)(hs)
    assert len(rec) == 1                                    # only the first full-prompt pass is recorded
    one = []
    chain.hook_fn(ORDERS["ae"], FakeTap(tok[:5].reshape(5, 1)), record=one)(x[:5].reshape(5, 1, D))
    assert one == []                                        # cached generation steps (T == 1) are not


# ---------------------------------------------------------------------------
# edits and controls
# ---------------------------------------------------------------------------

def _fit_data(seed=0, n=120, d=50):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, d).astype(np.float32)
    is_c = np.arange(n) < n // 3
    A[is_c, :6] += 3.0                                      # 6 coordinates more active on the concept
    A[is_c, 6:10] -= 3.0                                    # 4 coordinates less active on it
    return A, is_c


KW = dict(percent=0.3, top_k=5, tao=2.0, alphas=[0.5, 1.0], ops=["rm_range", "st_range", "st_global"], device="cpu")


def test_fit_edits_keys_and_coordinate_counts():
    A, is_c = _fit_data()
    edits, info = fit_edits(A, is_c, "z", **KW)
    assert sorted(edits) == ["rm_range_comp", "st_global_a0.5", "st_global_a1.0", "st_range_a0.5", "st_range_a1.0"]
    assert info["n_coords"] == 15 and info["dim"] == 50
    assert info["in_range_concept"] > 0.9 and info["in_range_concept"] > info["in_range_others"]
    edits_top, info_top = fit_edits(A, is_c, "s_top", **KW)
    assert "st_global_a0.5" not in edits_top and info_top["n_coords"] == 5


def test_s_top_takes_only_coordinates_more_active_on_the_concept():
    A, is_c = _fit_data()
    x = torch.zeros(1, 50)
    x[0] = torch.as_tensor(A[is_c].mean(0))
    edits, _ = fit_edits(A, is_c, "s_top", **{**KW, "top_k": 8})
    moved = (edits["rm_range_comp"](x) != x)[0].numpy()
    more_active = A[is_c].mean(0) > A[~is_c].mean(0)
    assert moved[:6].all() and not moved[6:10].any()        # the 6 more-active coordinates; never the 4 less-active ones
    assert more_active[moved].all() and moved.sum() == 8    # the rest of the 8 are also (weakly) more active on the concept
    _, info = fit_edits(A, is_c, "s_top", **{**KW, "top_k": 8})
    assert info["n_coords"] <= 8


def test_range_edit_touches_only_in_range_values_of_selected_coordinates():
    A, is_c = _fit_data()
    edits, _ = fit_edits(A, is_c, "z", **KW)
    far = torch.full((1, 50), 1e3)                          # outside every range
    assert torch.equal(edits["st_range_a0.5"](far), far) and torch.equal(edits["rm_range_comp"](far), far)
    assert not torch.equal(edits["st_global_a0.5"](far), far)
    at = torch.as_tensor(A[is_c].mean(0))[None]
    assert (edits["st_range_a0.5"](at) != at).sum() == 15


def test_controls_are_seeded_and_differ_from_the_real_edit():
    A, is_c = _fit_data()
    real, info = fit_edits(A, is_c, "z", **KW)
    r1, i1 = fit_edits(A, is_c, "z", **KW, control="rand", rng=control_rng(42, 3, "ae_sae", "z", "rand"))
    r2, i2 = fit_edits(A, is_c, "z", **KW, control="rand", rng=control_rng(42, 3, "ae_sae", "z", "rand"))
    assert sorted(r1) == ["st_range_a0.5"]                  # controls: range steering at the first strength only
    assert i1["n_coords"] == info["n_coords"]               # same NUMBER of coordinates
    x = torch.as_tensor(A[is_c].mean(0))[None]
    assert torch.equal(r1["st_range_a0.5"](x), r2["st_range_a0.5"](x))
    assert not torch.equal(r1["st_range_a0.5"](x), real["st_range_a0.5"](x))
    other = fit_edits(A, is_c, "z", **KW, control="rand", rng=control_rng(42, 4, "ae_sae", "z", "rand"))[0]
    assert not torch.equal(other["st_range_a0.5"](x), r1["st_range_a0.5"](x))
    s1 = fit_edits(A, is_c, "z", **KW, control="shuf", rng=control_rng(42, 3, "ae", "z", "shuf"))[0]
    s2 = fit_edits(A, is_c, "z", **KW, control="shuf", rng=control_rng(42, 3, "ae", "z", "shuf"))[0]
    assert torch.equal(s1["st_range_a0.5"](x), s2["st_range_a0.5"](x))
    shift_real = (real["st_range_a0.5"](x) - x).abs().sum()
    shift_null = (s1["st_range_a0.5"](x) - x).abs().sum()
    assert shift_null < 0.5 * shift_real                    # a shuffled concept has almost no class gap to steer along


# ---------------------------------------------------------------------------
# document allocation, metric
# ---------------------------------------------------------------------------

def test_allocate_serves_evaluation_first_and_drops_short_classes():
    assert allocate(200, 200, 80, 65, 50, 40) == (80, 65)
    assert allocate(120, 120, 80, 65, 50, 40) == (55, 65)   # evaluation candidates first, the rest is fitted on
    assert allocate(95, 95, 80, 65, 50, 40) == (40, 55)
    assert allocate(89, 89, 80, 65, 50, 40) is None         # cannot give 40 fit + 50 evaluation documents
    assert allocate(300, 60, 80, 65, 50, 40, split=True) == (80, 60)
    assert allocate(300, 49, 80, 65, 50, 40, split=True) is None
    assert allocate(39, 300, 80, 65, 50, 40, split=True) is None


def test_selectivity_arithmetic_and_scales():
    ref = [0.8, 0.6, 0.5, 0.5, 0.4]                         # 2 target documents, 3 other-class documents
    run = [0.2, 0.4, 0.3, 0.6, 0.4]
    t, c, s = selectivity(ref, run, 2)
    assert t == pytest.approx(-40.0) and c == pytest.approx(100 * 0.2 / 3) and s == pytest.approx(40 - 100 * 0.2 / 3)
    tr, cr, sr = selectivity(ref, run, 2, scale="rel")
    assert tr == pytest.approx(100 * -0.4 / 0.7) and cr == pytest.approx(100 * (0.2 / 3) / (1.4 / 3))
    assert selectivity(ref, ref, 2) == (0.0, 0.0, 0.0)
    tb = selectivity(ref, run, 2, ti=np.array([0, 0]), ci=np.array([1, 1, 1]))
    assert tb[0] == pytest.approx(-60.0) and tb[1] == pytest.approx(0.0)      # a GAIN on other documents is not collateral


# ---------------------------------------------------------------------------
# joint-correct pool with extra splices
# ---------------------------------------------------------------------------

class FakeHook:
    def __init__(self):
        self.cur = None

    def activate(self, fn):
        self.cur = fn

    def deactivate(self):
        self.cur = None


def _pool(tmp_path, monkeypatch, extra, key, calls):
    """Texts are letter strings: the base model gets a text right iff it contains 'b', a splice named n iff it contains
    the letter n."""
    shared.set_dataset("ag_news")
    hook = FakeHook()

    def fake_predict(lm, tokenizer, texts, labels, device=None, batch_size=16, max_new=None):
        who = "b" if hook.cur is None else hook.cur
        calls.append((who, len(texts)))
        return {"correct": [who in t for t in texts], "gold_conf": [0.5] * len(texts), "pred": [""] * len(texts)}

    monkeypatch.setattr(shared, "predict", fake_predict)
    texts = ["bzsx", "bz", "bzs", "zsx", "bzsx2", "bsx", "bzx", "bzsx3", "b", "bzsx4", "bzsx5", "bzsx6"]
    ds = [{"text": t, "label": 0} for t in texts] + [{"text": t + "!", "label": 1} for t in texts]
    return shared.build_joint_correct_set(
        ds, shared.DATASET_CONFIGS["ag_news"], None, object(), "cpu", hook, "z", tmp_path / "pool.json", need=3,
        dataset_name="ag_news", model_name="toy", ae_sha="abc", batch_size=4, tag="t", classes=[0, 1],
        extra_recons=extra, extra_key=key)


def test_pool_keeps_only_documents_every_model_gets_right(tmp_path, monkeypatch):
    calls = []
    docs = _pool(tmp_path, monkeypatch, {"s": "s", "x": "x"}, "k1", calls)
    by = {c: [d["text"] for d in docs if d["label"] == c] for c in (0, 1)}
    assert by[0] == ["bzsx", "bzsx2", "bzsx3"] and by[1] == ["bzsx!", "bzsx2!", "bzsx3!"]      # in data order, `need` each
    assert all(all(ch in t for ch in "bzsx") for t in by[0] + by[1])
    saved = json.load(open(tmp_path / "pool.json"))
    assert saved["extra"] == "k1" and saved["extra_stats"]["s"]["0"]["entered"] >= saved["extra_stats"]["s"]["0"]["survived"]
    # an extra splice only sees documents that survived the base model, the AE splice and the earlier extras
    first_batch = calls[:4]
    assert first_batch == [("b", 4), ("z", 4), ("s", 3), ("x", 2)]


def test_pool_cache_is_tied_to_the_extra_filters(tmp_path, monkeypatch):
    calls = []
    a = _pool(tmp_path, monkeypatch, {"s": "s", "x": "x"}, "k1", calls)
    n = len(calls)
    b = _pool(tmp_path, monkeypatch, {"s": "s", "x": "x"}, "k1", calls)
    assert b == a and len(calls) == n                       # same key: loaded from the cache, nothing recomputed
    c = _pool(tmp_path, monkeypatch, {"s": "s"}, "k2", calls)
    assert len(calls) > n                                   # different filters: rebuilt
    assert [d["text"] for d in c if d["label"] == 0] == ["bzsx", "bzs", "bzsx2"]


def test_pool_without_extras_is_the_original_two_way_filter(tmp_path, monkeypatch):
    calls = []
    docs = _pool(tmp_path, monkeypatch, None, None, calls)
    assert [d["text"] for d in docs if d["label"] == 0] == ["bzsx", "bz", "bzs"]
    assert "extra" not in json.load(open(tmp_path / "pool.json"))
    assert {w for w, _ in calls} == {"b", "z"}
