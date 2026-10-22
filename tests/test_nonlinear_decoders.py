"""Decoder experiments preserve old models and implement state-dependent edits."""
import argparse

import pytest
import torch
from torch import nn

from geoae.checkpoint import load_ae_checkpoint
from geoae.config import Config
from geoae.model import GeoAE
from geoae.train_common import add_override_args, apply_overrides, build_model, load_resume, make_optimizer


D, L, K, H = 8, 16, 3, 5


def _model(decoder_type="linear", **kwargs):
    return GeoAE(hidden_size=D, latent_dim=L, n_clusters=K, nonlinearity="gelu",
                 decoder_type=decoder_type, **kwargs)


def _cfg(decoder_type="linear", **kwargs):
    cfg = Config()
    cfg.model.hidden_size, cfg.model.latent_dim, cfg.model.n_clusters = D, L, K
    cfg.model.nonlinearity = "gelu"
    cfg.model.decoder_type = decoder_type
    for name, value in kwargs.items():
        setattr(cfg.model, name, value)
    return cfg


def test_legacy_initialization_and_rng_are_exactly_preserved():
    torch.manual_seed(101)
    encoder = nn.Linear(D, L, bias=True)
    decoder = nn.Linear(L, D, bias=False)
    nn.init.xavier_uniform_(encoder.weight)
    nn.init.zeros_(encoder.bias)
    nn.init.xavier_uniform_(decoder.weight)
    with torch.no_grad():
        decoder.weight.div_(decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(101)
    ae = _model()
    assert torch.equal(ae.encoder[0].weight, encoder.weight)
    assert torch.equal(ae.encoder[0].bias, encoder.bias)
    assert torch.equal(ae.decoder.weight, decoder.weight)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("decoder_type", ["mlp", "residual_mlp"])
@pytest.mark.parametrize("width", [0, H])
@pytest.mark.parametrize("output_bias", [False, True])
def test_matched_seed_encoder_and_rng_match_linear_baseline(decoder_type, width, output_bias):
    torch.manual_seed(202)
    linear = _model(latent_norm="batch")
    expected_rng = torch.get_rng_state()
    torch.manual_seed(202)
    ae = _model(decoder_type, decoder_hidden_dim=width,
                decoder_output_bias=output_bias, latent_norm="batch")
    assert ae.decoder_hidden_dim == (width or (L if decoder_type == "mlp" else D // 4))
    for name, value in linear.encoder.state_dict().items():
        assert torch.equal(ae.encoder.state_dict()[name], value)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    if decoder_type == "residual_mlp":
        assert torch.equal(ae.decoder.skip.weight, linear.decoder.weight)
        assert torch.count_nonzero(ae.decoder.output.weight) == 0
        z = torch.randn(4, L)
        assert torch.equal(ae.decoder(z), linear.decoder(z))


def test_mlp_matches_requested_design_and_trains_all_layers():
    torch.manual_seed(3)
    ae = _model("mlp", decoder_output_bias=True)
    assert (ae.decoder[0].in_features, ae.decoder[0].out_features) == (L, L)
    assert isinstance(ae.decoder[1], nn.GELU)
    assert (ae.decoder[2].in_features, ae.decoder[2].out_features) == (L, D)
    assert ae.decoder[0].bias is not None and ae.decoder[2].bias is not None
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-2)
    x = torch.randn(12, D)
    out = ae(x)
    assert out.x_hat.shape == x.shape and out.z.shape == (12, L)
    ((out.x_hat - x).square().mean() + out.z.square().mean()).backward()
    for name, parameter in ae.named_parameters():
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad), name
    before = {name: p.detach().clone() for name, p in ae.decoder.named_parameters()}
    opt.step()
    for name, parameter in ae.decoder.named_parameters():
        assert not torch.equal(parameter, before[name]), name
    before_renorm = {name: p.detach().clone() for name, p in ae.decoder.named_parameters()}
    ae.post_step()
    for name, parameter in ae.decoder.named_parameters():
        assert torch.equal(parameter, before_renorm[name]), name


def test_residual_zero_branch_matches_copied_parent_and_learns():
    torch.manual_seed(4)
    parent = _model(latent_norm="batch").eval()
    ae = _model("residual_mlp", decoder_hidden_dim=H, latent_norm="batch").eval()
    ae.encoder.load_state_dict(parent.encoder.state_dict())
    ae.decoder.skip.load_state_dict(parent.decoder.state_dict())
    x = torch.randn(12, D)
    assert torch.equal(ae(x).x_hat, parent(x).x_hat)
    opt = torch.optim.SGD(ae.parameters(), lr=0.1)
    # The zero output projection gets gradients first; its upstream layer starts
    # receiving gradients after the first optimizer step.
    for step in range(2):
        opt.zero_grad()
        (ae(x).x_hat - x).square().mean().backward()
        assert torch.count_nonzero(ae.decoder.output.weight.grad)
        if step:
            assert torch.count_nonzero(ae.decoder.hidden.weight.grad)
            assert torch.count_nonzero(ae.decoder.hidden.bias.grad)
        opt.step()
        output_weight = ae.decoder.output.weight.detach().clone()
        hidden_weight = ae.decoder.hidden.weight.detach().clone()
        ae.post_step()
        torch.testing.assert_close(ae.decoder.skip.weight.norm(dim=0), torch.ones(L))
        assert torch.equal(output_weight, ae.decoder.output.weight)
        assert torch.equal(hidden_weight, ae.decoder.hidden.weight)


@pytest.mark.parametrize("decoder_type", ["linear", "mlp", "residual_mlp"])
def test_decoded_delta_is_exact_stateful_edit_and_token_bias_cancels(decoder_type):
    torch.manual_seed(5)
    ae = _model(decoder_type, decoder_hidden_dim=H, token_bias_rows=2, vocab_size=6)
    ae.load_token_bias(torch.tensor([1, 3]), torch.randn(2, D))
    if decoder_type == "residual_mlp":
        nn.init.normal_(ae.decoder.output.weight)
    z, delta, tok = torch.randn(4, L), torch.randn(4, L), torch.tensor([1, 3, 5, 1])
    actual = ae.decoded_delta(z, delta, tok)
    torch.testing.assert_close(actual, ae.decode(z + delta, tok) - ae.decode(z, tok))
    torch.testing.assert_close(actual, ae.decoder(z + delta) - ae.decoder(z))
    if decoder_type == "linear":
        assert torch.equal(actual, ae.decoder(delta))
        assert torch.equal(ae.decoded_delta(z, delta), actual)
        assert ae.decoded_delta(z, delta[0]).shape == (4, D)
    else:
        assert not torch.allclose(actual, ae.decoded_delta(z + 3, delta, tok))


def test_affine_control_has_state_independent_edits_and_bias_cancels():
    ae = _model("mlp", decoder_hidden_dim=H, decoder_nonlinearity="linear", decoder_output_bias=True)
    with torch.no_grad():
        ae.decoder[0].bias.fill_(0.7)
        ae.decoder[2].bias.fill_(2)
    z, delta = torch.randn(4, L), torch.randn(4, L)
    torch.testing.assert_close(ae.decoded_delta(z, delta), ae.decoded_delta(z + 3, delta))
    expected = delta @ ae.decoder[0].weight.T @ ae.decoder[2].weight.T
    torch.testing.assert_close(ae.decoded_delta(z, delta), expected)


@pytest.mark.parametrize("decoder_type", ["mlp", "residual_mlp"])
def test_fixed_direction_operations_refuse_new_decoders(decoder_type):
    with pytest.raises(ValueError, match="decoded_delta"):
        _model(decoder_type).require_linear_decoder("Fixed direction steering")
    _model().require_linear_decoder()
    with pytest.raises(ValueError, match="bias-free"):
        _model(decoder_output_bias=True).require_linear_decoder()


@pytest.mark.parametrize("decoder_type", ["linear", "mlp", "residual_mlp"])
def test_checkpoint_and_optimizer_resume_roundtrip(tmp_path, decoder_type):
    cfg = _cfg(decoder_type, decoder_hidden_dim=H, decoder_output_bias=decoder_type == "mlp")
    ae = build_model(cfg, torch.device("cpu"))
    opt = make_optimizer(ae, cfg)
    x = torch.randn(12, D)
    (ae(x).x_hat - x).square().mean().backward()
    opt.step()
    ae.post_step()
    ae.eval()
    saved_config = cfg.to_dict()
    if decoder_type == "linear":
        # Old checkpoints contain none of the new architecture fields.
        for key in ("decoder_type", "decoder_hidden_dim", "decoder_nonlinearity", "decoder_output_bias"):
            saved_config["model"].pop(key)
    path = tmp_path / "ae.pt"
    torch.save({"model_state": ae.state_dict(), "opt_state": opt.state_dict(), "config": saved_config,
                "epoch": 2, "step": 9, "norm_mean": torch.zeros(D), "norm_std": torch.ones(D)}, path)
    loaded, mean, std, _ = load_ae_checkpoint(path, "cpu")
    assert loaded.decoder_type == decoder_type
    assert torch.equal(loaded(x).x_hat, ae(x).x_hat)
    assert torch.equal(mean, torch.zeros(D)) and torch.equal(std, torch.ones(D))
    resumed = build_model(cfg, torch.device("cpu"))
    resumed_opt = make_optimizer(resumed, cfg)
    assert load_resume(path, resumed, resumed_opt, mean, std, "cpu") == (3, 9)
    resumed.eval()
    assert torch.equal(resumed(x).x_hat, ae(x).x_hat)
    for parameter, original in zip(resumed.parameters(), ae.parameters()):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(resumed_opt.state[parameter][key], opt.state[original][key])


def test_resume_refuses_activation_change_with_identical_state_dict_shapes(tmp_path):
    cfg = _cfg("mlp")
    model = build_model(cfg, torch.device("cpu"))
    path = tmp_path / "ae.pt"
    torch.save({"config": cfg.to_dict(), "model_state": model.state_dict()}, path)
    cfg.model.decoder_nonlinearity = "linear"
    changed = build_model(cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="decoder_nonlinearity mismatch"):
        load_resume(path, changed, make_optimizer(changed, cfg), torch.zeros(D), torch.ones(D), "cpu")


@pytest.mark.parametrize("field,value", [("decoder_type", "typo"), ("decoder_hidden_dim", -1),
                          ("decoder_hidden_dim", 1.5), ("decoder_hidden_dim", True),
                          ("decoder_nonlinearity", "typo"), ("decoder_output_bias", "false"),
                          ("hidden_size", 0), ("latent_dim", -2)])
def test_invalid_decoder_config_rejected(field, value):
    cfg = _cfg(**{field: value})
    with pytest.raises(ValueError, match=field):
        cfg.validate()
    kwargs = {"hidden_size": D, "latent_dim": L, "n_clusters": K, field: value}
    with pytest.raises(ValueError, match=field):
        GeoAE(**kwargs)


def test_decoder_yaml_and_cli_roundtrip(tmp_path):
    import yaml
    cfg = _cfg("mlp", decoder_hidden_dim=H, decoder_output_bias=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()))
    loaded = Config.from_yaml(path)
    assert loaded.to_dict()["model"] == cfg.to_dict()["model"]
    parser = argparse.ArgumentParser()
    add_override_args(parser)
    args = parser.parse_args(["--decoder_type", "residual_mlp", "--decoder_hidden_dim", "7",
                              "--decoder_nonlinearity", "tanh", "--no-decoder_output_bias"])
    apply_overrides(loaded, args)
    assert loaded.model.decoder_type == "residual_mlp"
    assert loaded.model.decoder_hidden_dim == 7
    assert loaded.model.decoder_nonlinearity == "tanh"
    assert loaded.model.decoder_output_bias is False
    ae = build_model(loaded, torch.device("cpu"))
    assert ae.decoder.hidden.out_features == 7 and isinstance(ae.decoder.activation, nn.Tanh)

