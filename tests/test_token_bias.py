"""Token-bypass GeoAE (design B): the bias goes around the latent exactly, missing
ids get zero bias, ids are mandatory, checkpoints round-trip, unconverted tools are
refused, and the buffer yields (x, token) pairs. CPU only."""
import numpy as np
import pytest
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.config import Config
from geoae.data import ActivationBuffer, split_batch
from geoae.model import GeoAE
from geoae.token_bias import shrink_means


def _model(rows=3, vocab=10, D=6, L=8):
    torch.manual_seed(0)
    m = GeoAE(hidden_size=D, latent_dim=L, n_clusters=4, nonlinearity="gelu",
              token_bias_rows=rows, vocab_size=vocab)
    table = torch.randn(rows, D)
    m.load_token_bias(torch.tensor([2, 5, 7]), table)
    return m, table.half().float()


def test_bias_goes_around_the_latent():
    m, table = _model()
    x = torch.randn(4, 6)
    tok = torch.tensor([2, 5, 7, 9])                       # 9 is not in the table
    b = m.token_bias(tok)
    assert torch.allclose(b[:3], table) and torch.all(b[3] == 0)
    out = m(x, tok)
    assert torch.allclose(out.z, m.encoder(x - b))                    # encoder sees x - b
    assert torch.allclose(out.x_hat, m.decoder(out.z) + b)            # recon adds b back
    assert torch.allclose(m.encode(x, tok), out.z)
    assert torch.allclose(m.decode(out.z, tok), out.x_hat)


def test_token_ids_are_mandatory_and_plain_models_unchanged():
    m, _ = _model()
    with pytest.raises(ValueError, match="token ids"):
        m(torch.randn(2, 6))
    plain = GeoAE(hidden_size=6, latent_dim=8, n_clusters=4, nonlinearity="gelu")
    x = torch.randn(3, 6)
    assert not plain.has_token_bias and "tb_table" not in plain.state_dict()
    assert torch.allclose(plain.encode(x), plain.encoder(x))
    assert torch.allclose(plain(x).x_hat, plain.decoder(plain.encoder(x)))


def test_shrinkage():
    sums = torch.tensor([[30.0], [6.0], [1.0]])
    counts = torch.tensor([30, 3, 1])
    ids, table, cnt = shrink_means(sums, counts, shrink_k=30.0, min_count=2)
    assert ids.tolist() == [0, 1] and cnt.tolist() == [30, 3]
    assert torch.allclose(table[:, 0], torch.tensor([1.0 * 30 / 60, 2.0 * 3 / 33]))


def test_checkpoint_round_trip_and_guard(tmp_path):
    m, _ = _model()
    cfg = Config()
    cfg.model.hidden_size, cfg.model.latent_dim, cfg.model.n_clusters = 6, 8, 4
    cfg.model.nonlinearity, cfg.model.token_bias = "gelu", "some_table.npz"
    p = tmp_path / "ck.pt"
    torch.save({"model_state": m.state_dict(), "config": cfg.to_dict(), "tau": 1.0, "epoch": 1,
                "norm_mean": np.zeros(6, np.float32), "norm_std": np.ones(6, np.float32)}, p)
    with pytest.raises(SystemExit, match="TOKEN-BYPASS"):
        load_ae_checkpoint(p, "cpu")
    ae, *_ = load_ae_checkpoint(p, "cpu", allow_token_bias=True)
    tok = torch.tensor([2, 5, 9])
    x = torch.randn(3, 6)
    m.eval()
    assert torch.allclose(ae(x, tok).x_hat, m(x, tok).x_hat)


def test_buffer_yields_token_pairs(tmp_path):
    rng = np.random.RandomState(0)
    np.save(tmp_path / "layer_27.npy", rng.randn(40, 6).astype(np.float16))
    np.save(tmp_path / "rows_tok.npy", np.arange(40, dtype=np.int32))
    buf = ActivationBuffer(tmp_path, 27, val_frac=0.25, split="val", return_tokens=True)
    x, t = buf[3]
    assert x.shape == (6,) and t == 33                      # val split starts at row 30
    loader = torch.utils.data.DataLoader(buf, batch_size=4)
    xb, tb = split_batch(next(iter(loader)), "cpu")
    assert xb.shape == (4, 6) and tb.tolist() == [30, 31, 32, 33]
    plain = ActivationBuffer(tmp_path, 27, val_frac=0.25, split="val")
    xb2, tb2 = split_batch(next(iter(torch.utils.data.DataLoader(plain, batch_size=4))), "cpu")
    assert tb2 is None and torch.allclose(xb, xb2)


# ---------------------------------------------------------------------------
# Intervention splices (neuronlens factories, number-control patch) on a real
# tiny LM: the TokenIdTap must hand each hidden-state row its OWN token id.
# ---------------------------------------------------------------------------

def _tiny_lm_and_ae(vocab=40, hidden=16):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    lm = LlamaForCausalLM(LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2,
                                      num_key_value_heads=2)).eval()
    ae = GeoAE(hidden_size=hidden, latent_dim=24, n_clusters=4, nonlinearity="gelu",
               token_bias_rows=6, vocab_size=vocab)
    ae.load_token_bias(torch.arange(3, 9), torch.randn(6, hidden) * 3)
    return lm, ae.eval()


@torch.no_grad()
def test_z_splices_use_each_rows_own_token_id():
    from geoae.hooks import SplicingHook, TokenIdTap
    from geoae.interp import neuronlens as nl
    lm, ae = _tiny_lm_and_ae()
    mean, std = torch.zeros(16), torch.ones(16)
    ids = torch.tensor([[3, 4, 5, 20], [8, 7, 6, 3]])
    seen = {}
    grab = SplicingHook(lm, 1)
    grab.activate(lambda hs: seen.setdefault("h", hs.clone()) if "h" not in seen else hs)
    lm(input_ids=ids); grab.deactivate()
    h = seen["h"]
    want = ae.decode(ae.encode(h.reshape(-1, 16), ids.reshape(-1)), ids.reshape(-1)).reshape(h.shape)

    tap = TokenIdTap(lm)
    for fn in (nl.make_z_edit(None, ae, mean, std, "cpu", tap=tap),
               nl.make_z_steer(torch.zeros(ae.latent_dim).numpy(), 0.0, ae, mean, std, "cpu", tap=tap)):
        out = {}
        hook = SplicingHook(lm, 1)
        hook.activate(lambda hs, fn=fn: out.setdefault("r", fn(hs)))
        lm(input_ids=ids); hook.deactivate()
        torch.testing.assert_close(out["r"], want)
    # without a tap the bypass AE refuses instead of silently dropping the bias
    with pytest.raises(ValueError, match="tap"):
        nl.make_z_edit(None, ae, mean, std, "cpu")(h)
    tap.remove()


@torch.no_grad()
def test_tap_follows_cached_generation_steps():
    from geoae.hooks import SplicingHook, TokenIdTap
    lm, ae = _tiny_lm_and_ae()
    tap = TokenIdTap(lm)
    shapes = []
    hook = SplicingHook(lm, 1)
    hook.activate(lambda hs: (shapes.append((tuple(hs.shape[:2]), tap.ids_for(hs).shape[0])), hs)[1])
    lm.generate(input_ids=torch.tensor([[3, 4, 5]]), max_new_tokens=3, do_sample=False, pad_token_id=0)
    hook.deactivate(); tap.remove()
    assert shapes[0] == ((1, 3), 3) and all(s == ((1, 1), 1) for s in shapes[1:])


@torch.no_grad()
def test_number_patch_uses_last_position_id():
    from geoae.hooks import TokenIdTap
    from geoae.interp.range_number_control import make_last_patch
    lm, ae = _tiny_lm_and_ae()
    mean, std = torch.zeros(16), torch.ones(16)
    tap = TokenIdTap(lm)
    ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
    lm.model.embed_tokens(ids)                        # tap now holds these ids
    hs = torch.randn(2, 3, 16)
    patch, _ = make_last_patch(ae, mean, std, tap=tap)
    out = patch(hs)
    torch.testing.assert_close(out[:, -1], ae.decode(ae.encode(hs[:, -1], ids[:, -1]), ids[:, -1]))
    torch.testing.assert_close(out[:, :-1], hs[:, :-1])
    with pytest.raises(ValueError, match="tap"):
        make_last_patch(ae, mean, std)[0](hs)
    tap.remove()


def test_kmeans_control_subtracts_the_same_table(tmp_path):
    """The encoder-free control assigns x - b[tok] with the bypass AE's own table."""
    from geoae.interp.concept_probe import assign
    from geoae.token_bias import TokenBiasLookup
    table = np.zeros((2, 3), np.float32); table[0] = [5, 0, 0]; table[1] = [0, 5, 0]
    p = tmp_path / "tb.npz"
    np.savez(p, token_ids=np.array([4, 9]), table=table.astype(np.float16), vocab_size=12,
             norm_mean=np.zeros(3, np.float32), norm_std=np.ones(3, np.float32), shrink_k=30.0, min_count=10)
    tb = TokenBiasLookup(p, "cpu")
    got = tb(torch.tensor([4, 9, 2]))
    torch.testing.assert_close(got, torch.tensor([[5., 0, 0], [0, 5, 0], [0, 0, 0]]))
    C = torch.tensor([[0., 0, 0], [5, 0, 0]])
    H = np.array([[5., 0, 0], [5., 0, 0]], np.float32)
    km = ("km", (C, torch.zeros(3), torch.ones(3), None, tb))
    assert assign(H, km, "cpu", ids=np.array([4, 2])).tolist() == [0, 1]   # token 4's mean removed; 2 has none
    assert assign(H, km, "cpu") is None                                     # no ids -> not scored


@torch.no_grad()
def test_number_hb_arm_is_identity_unedited_and_edits_token_free_space(tmp_path):
    from geoae.hooks import TokenIdTap
    from geoae.interp.range_number_control import make_last_patch
    from geoae.token_bias import TokenBiasLookup
    lm, ae = _tiny_lm_and_ae()
    D = 16
    table = np.random.RandomState(0).randn(3, D).astype(np.float32)
    p = tmp_path / "tb.npz"
    np.savez(p, token_ids=np.array([3, 5, 8]), table=table.astype(np.float16), vocab_size=40,
             norm_mean=np.zeros(D, np.float32), norm_std=np.ones(D, np.float32), shrink_k=30.0, min_count=10)
    hb = TokenBiasLookup(p, "cpu")
    mean, std = torch.randn(D), torch.rand(D) + 0.5
    tap = TokenIdTap(lm)
    ids = torch.tensor([[1, 2, 3], [4, 6, 5]])
    lm.model.embed_tokens(ids)
    hs = torch.randn(2, 3, D)
    ident, _ = make_last_patch(ae, mean, std, tap=tap, hb=hb)
    torch.testing.assert_close(ident(hs), hs, atol=1e-5, rtol=1e-5)          # b added back exactly
    zero = lambda a: torch.zeros_like(a)                                       # edit: a -> 0
    out = make_last_patch(ae, mean, std, edit=zero, tap=tap, hb=hb)[0](hs)
    want = hb(ids[:, -1]) * std + mean                                         # only b[tok] survives
    torch.testing.assert_close(out[:, -1], want, atol=1e-5, rtol=1e-5)
    tap.remove()


@torch.no_grad()
def test_hb_edit_is_identity_unedited_and_edits_each_rows_token_free_space(tmp_path):
    from geoae.hooks import SplicingHook, TokenIdTap
    from geoae.interp import neuronlens as nl
    from geoae.token_bias import TokenBiasLookup
    lm, _ = _tiny_lm_and_ae()
    D = 16
    table = np.random.RandomState(1).randn(3, D).astype(np.float32)
    p = tmp_path / "tb.npz"
    np.savez(p, token_ids=np.array([3, 5, 8]), table=table.astype(np.float16), vocab_size=40,
             norm_mean=np.zeros(D, np.float32), norm_std=np.ones(D, np.float32), shrink_k=30.0, min_count=10)
    tb = TokenBiasLookup(p, "cpu")
    mean, std = torch.randn(D), torch.rand(D) + 0.5
    ids = torch.tensor([[3, 4, 5], [8, 7, 3]])
    tap = TokenIdTap(lm)
    out = {}
    for name, fn in (("ident", nl.make_hb_edit(None, tb, mean, std, "cpu", tap)),
                     ("zero", nl.make_hb_edit(torch.zeros_like, tb, mean, std, "cpu", tap)),
                     ("h", lambda hs: hs)):
        hook = SplicingHook(lm, 1)
        hook.activate(lambda hs, fn=fn, name=name: out.setdefault(name, (hs.clone(), fn(hs)))[1])
        lm(input_ids=ids); hook.deactivate()
    hs, ident = out["ident"]
    torch.testing.assert_close(ident, hs, atol=1e-5, rtol=1e-5)
    want = (tb(ids.reshape(-1)) * std + mean).reshape(2, 3, D)            # only b[tok] survives, per position
    torch.testing.assert_close(out["zero"][1], want, atol=1e-5, rtol=1e-5)
    tap.remove()


def test_cluster_steering_divergences():
    from geoae.interp.cluster_steering import js_rows, kl_rows
    p = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    q = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
    js = js_rows(p, q)
    assert abs(float(js[0])) < 1e-6 and abs(float(js[1]) - float(np.log(2))) < 1e-4   # identical / disjoint
    assert abs(float(kl_rows(p, q)[0])) < 1e-6 and float(kl_rows(p, q)[1]) > 10
