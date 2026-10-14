"""Plain GeoAE evaluations retain the exact full-LM splice and external controls."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

from geoae.evaluate import ce_per_token, experiment_loss_recovered
from geoae.hooks import SplicingHook
from geoae.interp.neuronlens import make_z_gate
from geoae.interp.range_number_control import make_last_patch
from geoae.model import GeoAE


@pytest.fixture(params=["llama", "gemma3"])
def native_case(request):
    from transformers import LlamaConfig, LlamaForCausalLM, Gemma3Config, Gemma3ForConditionalGeneration

    torch.manual_seed(42)
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    if request.param == "llama":
        lm = LlamaForCausalLM(LlamaConfig(**text)).eval()
    else:
        lm = Gemma3ForConditionalGeneration(Gemma3Config(
            text_config={**text, "sliding_window": 4, "final_logit_softcapping": 1.0},
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=16, patch_size=8),
        )).eval()
    ae = GeoAE(hidden_size=16, latent_dim=32, n_clusters=2).eval()
    mean, std = torch.randn(16), torch.rand(16) + .5
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    for model in (lm, ae):
        model.requires_grad_(False)
    return lm, ae, mean, std, ids


@pytest.mark.parametrize("layer", [0, 2], ids=["middle", "last"])
@pytest.mark.parametrize("metric", ["mmlu", "number", "loss_recovered"])
@torch.no_grad()
def test_plain_reconstruction_uses_selected_native_layer(native_case, layer, metric):
    lm, ae, mean, std, ids = native_case
    before = [{k: v.clone() for k, v in model.state_dict().items()} for model in (lm, ae)]
    captured = []
    hook = SplicingHook(lm, layer)
    hook.activate(lambda hs: (captured.append(hs.clone()) or hs))
    clean = lm(input_ids=ids, use_cache=False).logits
    hook.deactivate()
    hs = captured[0]
    normalized = (hs.reshape(-1, 16) - mean) / std
    reconstructed = (ae.decoder(ae.encoder(normalized)) * std + mean).reshape_as(hs)
    if metric == "number":
        expected_residual = hs.clone()
        expected_residual[:, -1] = reconstructed[:, -1]
    else:
        expected_residual = reconstructed
    hook.activate(lambda _: expected_residual)
    expected = lm(input_ids=ids, use_cache=False).logits
    hook.deactivate()
    assert not torch.allclose(clean, expected)
    if metric == "loss_recovered":
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return {"input_ids": ids.clone()}

        result = experiment_loss_recovered(lm, ae, {"mean": mean, "std": std},
                                          Tokenizer(), ["fixture"], layer, torch.device("cpu"))
        assert result["per_text"][0]["ce_spliced"] == pytest.approx(
            ce_per_token(expected, ids).mean().item(), abs=1e-6)
    else:
        if metric == "mmlu":
            patch = make_z_gate(np.zeros(1), np.zeros(1), np.zeros(1), ae, mean, std,
                                "cpu", gate=False)
        else:
            patch, _ = make_last_patch(ae, mean, std)
        hook.activate(patch)
        actual = lm(input_ids=ids, use_cache=False).logits
        hook.deactivate()
        torch.testing.assert_close(actual, expected)
    for model, state in zip((lm, ae), before):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def dispersion_module():
    path = Path(__file__).resolve().parents[1] / "eval_out/paper_dispersion.py"
    spec = importlib.util.spec_from_file_location("plain_dispersion_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dispersion_case(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dump = tmp_path / "dump"
    dump.mkdir()
    raw = rng.normal(size=(24, 4)).astype(np.float32) + 2
    tokens = np.arange(24, dtype=np.int64) % 3
    for name, values in {"layer_1": raw, "rows_tok": tokens,
                         "rows_doc": np.arange(24, dtype=np.int64) // 2,
                         "rows_pos": np.arange(24, dtype=np.int64) % 2}.items():
        np.save(dump / f"{name}.npy", values)
    mean, std = np.zeros(4, np.float32), np.ones(4, np.float32)
    table_values = rng.normal(size=(3, 4)).astype(np.float32) * .2
    table = tmp_path / "table.npz"
    np.savez(table, token_ids=np.arange(3), table=table_values, vocab_size=5,
             norm_mean=mean, norm_std=std)
    baseline = dict(centroids=raw[12:15], norm_mean=mean, norm_std=std, layer=1)
    km, kmt = tmp_path / "base.npz", tmp_path / "tokmean.npz"
    np.savez(km, **baseline)
    np.savez(kmt, **baseline, token_bias=str(table))
    return tmp_path, dump, raw, tokens, mean, std, table, table_values, km, kmt


@pytest.mark.parametrize("mode", ["plain", "bypass"])
def test_dispersion_accepts_plain_and_keeps_independent_tokenmean_control(dispersion_case, mode):
    tmp, dump, raw, tokens, mean, std, table, values, km, kmt = dispersion_case
    ae = GeoAE(hidden_size=4, latent_dim=6, n_clusters=3,
               token_bias_rows=3 if mode == "bypass" else 0,
               vocab_size=5 if mode == "bypass" else 0).eval()
    if mode == "bypass":
        ae.load_token_bias(torch.arange(3), torch.from_numpy(values))
    with torch.no_grad():
        ae.centroids.copy_(ae.encode(torch.from_numpy(raw[12:15]), torch.from_numpy(tokens[12:15])))
    checkpoint = tmp / "checkpoint.pt"
    torch.save({"model_state": ae.state_dict(), "norm_mean": mean, "norm_std": std,
                "config": {"model": {"hidden_size": 4, "latent_dim": 6, "n_clusters": 3},
                           "data": {"target_layer": 1, "val_frac": .5}}}, checkpoint)
    original = checkpoint.read_bytes()
    module = dispersion_module()
    args = module.parse_args(["--dump", str(dump), "--layer", "1", "--ckpt", str(checkpoint),
                              "--km", str(km), "--km_tok", str(kmt), "--val_frac", ".5",
                              "--device", "cpu", "--batch", "3", "--folds", "2",
                              "--out", str(tmp / "dispersion.json")])
    result = module.run(args)
    assert result["meta"]["ae_mode"] == mode
    assert result["meta"]["token_bias"] == (mode == "bypass")
    with torch.no_grad():
        latent = ae.encode(torch.from_numpy(raw[12:]), torch.from_numpy(tokens[12:])).numpy()
    mask = ~np.eye(12, dtype=bool)
    for name, vectors in (("geoae", latent), ("token_centred", raw[12:] - values[tokens[12:]])):
        expected = cdist(vectors, vectors, metric="cosine")[mask].mean()
        assert result["token_level"][name]["raw"]["dispersion"] == pytest.approx(expected, abs=1e-7)
    assert checkpoint.read_bytes() == original
    json.dumps(result, allow_nan=False)


def test_dispersion_rejects_tokenmean_control_with_wrong_normalization(dispersion_case):
    tmp, dump, _, _, mean, std, table, values, km, kmt = dispersion_case
    ae = GeoAE(hidden_size=4, latent_dim=6, n_clusters=3)
    checkpoint = tmp / "checkpoint.pt"
    torch.save({"model_state": ae.state_dict(), "norm_mean": mean, "norm_std": std,
                "config": {"model": {"hidden_size": 4, "latent_dim": 6, "n_clusters": 3}}}, checkpoint)
    np.savez(table, token_ids=np.arange(3), table=values, vocab_size=5,
             norm_mean=mean + 1, norm_std=std)
    module = dispersion_module()
    args = module.parse_args(["--dump", str(dump), "--layer", "1", "--ckpt", str(checkpoint),
                              "--km", str(km), "--km_tok", str(kmt), "--val_frac", ".5",
                              "--device", "cpu", "--out", str(tmp / "dispersion.json")])
    with pytest.raises(ValueError, match="different norm stats"):
        module.run(args)
