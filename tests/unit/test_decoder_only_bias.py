"""Decoder-only token means must preserve their semantics through checkpoints.

Tiny CPU models distinguish this experiment from the existing bypass: the
encoder consumes the full activation, while reconstruction retains the token
mean. No real activation dump or token-bias table is read.
"""
import numpy as np
import pytest
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.config import Config
from geoae.model import GeoAE
from geoae.training.common import build_model
from geoae.training.common import load_resume
from geoae.training.common import make_optimizer


D, L, K, VOCAB = 3, 5, 2, 8
TOKEN_IDS = torch.tensor([2, 4])
TABLE = torch.tensor([[4.0, -2.0, 1.0], [-6.0, 8.0, 2.0]])


def _model(mode="decoder_only"):
    ae = GeoAE(hidden_size=D, latent_dim=L, n_clusters=K,
               nonlinearity="gelu", token_bias_rows=len(TOKEN_IDS),
               vocab_size=VOCAB, token_bias_mode=mode)
    ae.load_token_bias(TOKEN_IDS, TABLE)
    return ae.eval()


def _cfg(table_path="embedded_table.npz", mode="decoder_only"):
    cfg = Config()
    cfg.model.hidden_size, cfg.model.latent_dim, cfg.model.n_clusters = D, L, K
    cfg.model.nonlinearity = "gelu"
    cfg.model.token_bias = str(table_path)
    cfg.model.token_bias_mode = mode
    return cfg


def _checkpoint(path, ae, cfg, opt=None, legacy=False):
    config = cfg.to_dict()
    if legacy:
        config["model"].pop("token_bias_mode")
    ck = {"model_state": ae.state_dict(), "config": config,
          "tau": 0.7, "epoch": 3, "step": 19,
          "norm_mean": np.zeros(D, np.float32), "norm_std": np.ones(D, np.float32)}
    if opt is not None:
        ck["opt_state"] = opt.state_dict()
    torch.save(ck, path)


def _write_table(path):
    np.savez(path, token_ids=TOKEN_IDS.numpy(), table=TABLE.half().numpy(),
             counts=np.array([50, 80]), vocab_size=VOCAB,
             norm_mean=np.zeros(D, np.float32), norm_std=np.ones(D, np.float32),
             shrink_k=30.0, min_count=10)


def test_encoder_receives_full_x_and_is_independent_of_token_id():
    ae = _model()
    # Identical activations with sharply different token means must have the
    # same representation; omitted IDs are valid for encoding alone.
    x = torch.tensor([[0.5, -0.25, 1.0]]).repeat(3, 1)
    tok = torch.tensor([2, 4, 7])
    expected = ae.encoder(x)
    torch.testing.assert_close(ae.encode(x, tok), expected)
    torch.testing.assert_close(ae.encode(x), expected)
    torch.testing.assert_close(expected[0], expected[1])
    torch.testing.assert_close(expected[0], expected[2])
    assert not torch.allclose(expected, ae.encoder(x - ae.token_bias(tok)))


def test_decoder_adds_each_tokens_mean_and_unknown_tokens_get_zero():
    ae = _model()
    z = torch.zeros(3, L)
    tok = torch.tensor([2, 4, 7])
    expected = torch.cat([TABLE, torch.zeros(1, D)])
    torch.testing.assert_close(ae.decode(z, tok), expected)
    assert ae.has_token_bias
    assert "tb_table" in ae.state_dict()
    assert "tb_table" not in dict(ae.named_parameters())
    with pytest.raises(ValueError, match="token"):
        ae.decode(z)
    with pytest.raises(ValueError, match="token"):
        ae(torch.randn(3, D))


def test_forward_agrees_with_encode_and_decode_and_fits_residual_target():
    ae = _model()
    x = torch.tensor([[0.5, -0.25, 1.0], [1.25, -0.5, 0.75], [-0.5, 2.0, 0.0]])
    tok = torch.tensor([2, 4, 7])
    ae.init_centroids_from_data(ae.encode(x))
    out = ae(x, tok)
    torch.testing.assert_close(out.z, ae.encode(x))
    torch.testing.assert_close(out.x_hat, ae.decode(out.z, tok))
    # Adding the fixed bias makes the reconstruction objective fit x-b,
    # despite retaining x as the input to the encoder.
    torch.testing.assert_close(out.x_hat - x,
                               ae.decoder(out.z) - (x - ae.token_bias(tok)))
    other = ae(x, tok.flip(0))
    torch.testing.assert_close(out.dist2, other.dist2)
    torch.testing.assert_close(out.Q, other.Q)


def test_identical_weights_have_distinct_mode_fingerprints():
    import hashlib
    from geoae.evaluation.cohorts import ae_fingerprint

    bypass, decoder_only = _model("bypass"), _model()
    decoder_only.load_state_dict(bypass.state_dict())
    # Bypass retains its existing cache identity; the changed reconstruction
    # semantics must invalidate that identity for decoder_only.
    legacy = hashlib.sha1()
    for name, value in sorted(bypass.state_dict().items()):
        legacy.update(name.encode())
        legacy.update(value.detach().float().cpu().numpy().tobytes())
    assert ae_fingerprint(bypass) == legacy.hexdigest()[:12]
    assert ae_fingerprint(decoder_only) != ae_fingerprint(bypass)


def test_builder_and_checkpoint_restore_mode_without_external_table(tmp_path):
    table_path = tmp_path / "tiny_table.npz"
    _write_table(table_path)
    cfg = _cfg(table_path)
    cfg.validate()
    ae = build_model(cfg, torch.device("cpu")).eval()
    assert ae.token_bias_mode == "decoder_only"
    path = tmp_path / "decoder_only.pt"
    _checkpoint(path, ae, cfg)
    # Restore must use embedded buffers and saved mode, not reload the .npz.
    table_path.unlink()
    with pytest.raises(SystemExit, match="TOKEN"):
        load_ae_checkpoint(path, "cpu")
    restored, *_ = load_ae_checkpoint(path, "cpu", allow_token_bias=True)
    assert restored.token_bias_mode == "decoder_only"
    x, tok = torch.randn(3, D), torch.tensor([2, 4, 7])
    torch.testing.assert_close(restored.encode(x), ae.encode(x))
    torch.testing.assert_close(restored(x, tok).x_hat, ae(x, tok).x_hat)


def test_legacy_checkpoint_without_mode_keeps_subtract_then_add_behavior(tmp_path):
    ae = _model("bypass")
    cfg = _cfg(mode="bypass")
    path = tmp_path / "old_bypass.pt"
    _checkpoint(path, ae, cfg, legacy=True)
    restored, *_ = load_ae_checkpoint(path, "cpu", allow_token_bias=True)
    assert restored.token_bias_mode == "bypass"
    x, tok = torch.randn(3, D), torch.tensor([2, 4, 7])
    torch.testing.assert_close(restored.encode(x, tok), restored.encoder(x - restored.token_bias(tok)))
    torch.testing.assert_close(restored(x, tok).x_hat, ae(x, tok).x_hat)


@pytest.mark.parametrize("saved,current,legacy", [
    ("bypass", "decoder_only", False),
    ("decoder_only", "bypass", False),
    ("bypass", "decoder_only", True),
])
def test_resume_mode_mismatch_rejected_before_model_or_optimizer_mutates(tmp_path, saved, current, legacy):
    source = _model(saved)
    cfg = _cfg(mode=saved)
    source_opt = make_optimizer(source, cfg)
    source(torch.randn(3, D), torch.tensor([2, 4, 7])).x_hat.square().mean().backward()
    source_opt.step()
    path = tmp_path / "source.pt"
    _checkpoint(path, source, cfg, opt=source_opt, legacy=legacy)
    target = _model(current)
    target_opt = make_optimizer(target, _cfg(mode=current))
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(ValueError, match="mode"):
        load_resume(path, target, target_opt, np.zeros(D, np.float32),
                    np.ones(D, np.float32), torch.device("cpu"))
    for name, value in before.items():
        torch.testing.assert_close(target.state_dict()[name], value)
    assert not target_opt.state


def test_resume_matching_decoder_only_mode_restores_state(tmp_path):
    source, cfg = _model(), _cfg()
    source_opt = make_optimizer(source, cfg)
    path = tmp_path / "source.pt"
    _checkpoint(path, source, cfg, opt=source_opt)
    target = _model()
    target_opt = make_optimizer(target, cfg)
    assert load_resume(path, target, target_opt, np.zeros(D, np.float32),
                       np.ones(D, np.float32), torch.device("cpu")) == (4, 19)
    tok = torch.tensor([2, 4, 7])
    x = torch.randn(3, D)
    torch.testing.assert_close(target(x, tok).x_hat, source(x, tok).x_hat)


@pytest.mark.parametrize("mode", ["decoder", "subtract", "", None])
def test_invalid_mode_rejected_by_config_and_constructor(mode):
    cfg = _cfg(mode=mode)
    with pytest.raises(ValueError, match="token_bias_mode"):
        cfg.validate()
    with pytest.raises(ValueError, match="token_bias_mode"):
        GeoAE(hidden_size=D, latent_dim=L, n_clusters=K,
              token_bias_rows=2, vocab_size=VOCAB, token_bias_mode=mode)


def test_decoder_only_requires_bias_table_in_config_and_model():
    cfg = _cfg(table_path="")
    with pytest.raises(ValueError, match="decoder_only"):
        cfg.validate()
    with pytest.raises(ValueError, match="decoder_only"):
        GeoAE(hidden_size=D, latent_dim=L, n_clusters=K, token_bias_mode="decoder_only")
