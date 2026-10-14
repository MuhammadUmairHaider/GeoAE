"""Native checkpoint math, raw-unit interventions, and loader compatibility guards."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file

from geoae.interp.gemma_scope import GemmaScopeSAE, load_gemma_scope


def config():
    return {
        "hf_hook_point_in": "model.layers.33.output",
        "hf_hook_point_out": "model.layers.33.output", "width": 3,
        "model_name": "google/gemma-3-4b-pt", "architecture": "jump_relu",
        "l0": 20, "affine_connection": False, "type": "sae",
    }


def params():
    # Decoder rows have deliberately unequal norms; b_dec is deliberately large.
    return {
        "w_enc": torch.tensor([[1., 0., -1.], [0., 1., 1.]]),
        "w_dec": torch.tensor([[2., 0.], [0., 5.], [-3., 4.]]),
        "b_enc": torch.tensor([0., 1., -1.]),
        "b_dec": torch.tensor([100., -300.]), "threshold": torch.tensor([2., 1., 0.]),
    }


def test_native_encoder_strict_threshold_no_decoder_bias_centering():
    sae = GemmaScopeSAE(params(), config())
    h = torch.tensor([[2., 0.], [3., 2.], [-1., 3.]])
    # First row hits thresholds exactly, including a negative third preactivation.
    expected = torch.tensor([[0., 0., 0.], [3., 3., 0.], [0., 4., 3.]])
    torch.testing.assert_close(sae.encode(h), expected)
    torch.testing.assert_close(sae(h), expected @ params()["w_dec"] + params()["b_dec"])
    assert sae.metadata["encoder_centers_on_b_dec"] is False
    assert list(sae.parameters()) == []


def test_raw_decoder_directions_and_error_preserving_ablation():
    sae = GemmaScopeSAE(params(), config())
    z = torch.tensor([[3., 5., 7.], [0., 2., 4.]])
    h = sae.decode(z) + torch.tensor([[11., 13.], [-2., 17.]])
    altered = z.clone()
    altered[:, [0, 2]] *= 0.25
    delta = sae.feature_delta(z, [0, 2], alpha=0.75)
    torch.testing.assert_close(delta, sae.decode(z) - sae.decode(altered))
    torch.testing.assert_close(h - delta - sae.decode(altered), h - sae.decode(z))
    torch.testing.assert_close(sae.decoder_directions, params()["w_dec"])
    assert sae.decoder_directions.norm(dim=-1).tolist() == [2., 5., 5.]


def test_batch_dimensions_empty_selection_and_per_token_gain():
    sae = GemmaScopeSAE(params(), config())
    z = torch.ones(2, 4, 3)
    assert sae.decode(z).shape == (2, 4, 2)
    torch.testing.assert_close(sae.feature_delta(z, []), torch.zeros(2, 4, 2))
    gains = torch.arange(8.).reshape(2, 4)
    expected = torch.tensor([2., 0.]).expand(2, 4, 2) * gains[..., None]
    torch.testing.assert_close(sae.feature_delta(z, [0], gains), expected)


def test_inference_remains_fp32_inside_autocast():
    sae = GemmaScopeSAE(params(), config())
    h = torch.tensor([[2.0001, 0.]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = sae.encode(h)
        decoded = sae.decode(z)
    assert z.dtype == decoded.dtype == torch.float32
    assert z[0, 0] > 2.0


@pytest.mark.parametrize("ids", [[0, 0], [-1], [3], [1.5], [[0]]])
def test_bad_feature_selection_rejected(ids):
    sae = GemmaScopeSAE(params(), config())
    with pytest.raises(ValueError):
        sae.feature_delta(torch.ones(1, 3), ids)


def test_nonfinite_or_shape_changing_ablation_gain_rejected():
    sae = GemmaScopeSAE(params(), config())
    with pytest.raises(ValueError, match="finite"):
        sae.feature_delta(torch.ones(1, 3), [0], float("nan"))
    with pytest.raises(ValueError, match="shape"):
        sae.feature_delta(torch.ones(1, 3), [0], torch.ones(3, 1))


@pytest.mark.parametrize("name,value", [
    ("w_dec", torch.zeros(2, 3)), ("threshold", torch.tensor([-1., 0., 0.])),
    ("b_dec", torch.tensor([float("nan"), 0.])),
])
def test_invalid_weights_fail_clearly(name, value):
    weights = params()
    weights[name] = value
    with pytest.raises(ValueError):
        GemmaScopeSAE(weights, config())


def test_checkpoint_key_case_is_not_guessed():
    weights = params()
    weights["W_enc"] = weights.pop("w_enc")
    with pytest.raises(ValueError, match="checkpoint keys"):
        GemmaScopeSAE(weights, config())


@pytest.mark.parametrize("update", [
    {"type": "transcoder"}, {"affine_connection": True},
    {"hf_hook_point_out": "model.layers.33.post_feedforward_layernorm.output"},
    {"width": 4}, {"apply_b_dec_to_input": True}, {"input_scale": 2.},
    {"normalize_activations": "expected_average_only_in"},
])
def test_unsupported_checkpoint_math_is_rejected(update):
    cfg = config()
    cfg.update(update)
    with pytest.raises(ValueError):
        GemmaScopeSAE(params(), cfg)


def write_checkpoint(directory, cfg=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(cfg or config()))
    save_file(params(), str(directory / "params.safetensors"))


def test_local_loader_works_offline_in_direct_or_nested_directory(tmp_path, monkeypatch):
    import huggingface_hub

    def forbid_download(*args, **kwargs):
        pytest.fail("Complete local checkpoints must not contact the Hub")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbid_download)
    nested = tmp_path / "resid_post_all/layer_33_width_16k_l0_small"
    write_checkpoint(nested)
    for root in (tmp_path, nested):
        sae = load_gemma_scope(local_dir=root)
        torch.testing.assert_close(sae.w_enc, params()["w_enc"])
        assert sae.metadata["source"] == "local_files"
        assert sae.metadata["revision"] is None
        assert sae.metadata["revision_verified"] is False
        assert sae.parameter_count == 20
        assert sae.width == 3


@pytest.mark.parametrize("update,message", [
    ({"model_name": "google/gemma-3-4b-it"}, "model"),
    ({"hf_hook_point_in": "model.layers.32.output", "hf_hook_point_out": "model.layers.32.output"}, "layer"),
])
def test_local_loader_rejects_wrong_model_and_layer(tmp_path, update, message):
    cfg = copy.deepcopy(config())
    cfg.update(update)
    write_checkpoint(tmp_path, cfg)
    with pytest.raises(ValueError, match=message):
        load_gemma_scope(local_dir=tmp_path)


def test_unpinned_hub_loader_uses_same_resolved_commit_for_config_and_weights(tmp_path, monkeypatch):
    import huggingface_hub

    checkpoint = tmp_path / "snapshots/verified_commit/resid_post_all/layer_33_width_16k_l0_small"
    write_checkpoint(checkpoint)
    calls = []

    def fake_download(*, repo_id, filename, revision, local_dir):
        calls.append((filename, revision))
        return str(checkpoint / filename.rsplit("/", 1)[-1])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    sae = load_gemma_scope(repo_id="test/native-scope")
    assert calls[0][1] is None
    assert calls[1][1] == "verified_commit"
    assert sae.metadata["revision"] == "verified_commit"


@pytest.mark.parametrize("path", ["../params", "/absolute", ""])
def test_loader_rejects_path_traversal(path):
    with pytest.raises(ValueError, match="relative"):
        load_gemma_scope(sae_path=path)
