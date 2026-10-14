"""Probe caches preserve finite Gemma residuals above the FP16 maximum."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from geoae.bias import extract as bias
from geoae.dbpedia import extract as dbpedia


class FakeLM(nn.Module):
    def __init__(self, model_type="gemma3", value=70000):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(model_type=model_type)
        self.model = nn.Module()
        self.model.config = SimpleNamespace(hidden_size=2)
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.value = value

    def forward(self, input_ids):
        # BF16, unlike FP16, can represent the finite residuals seen in Gemma.
        hs = torch.full((*input_ids.shape, 2), self.value, dtype=torch.bfloat16)
        return self.model.layers[0](hs)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))


def extract(module, model, tmp_path, **kwargs):
    dirs = {p: tmp_path / p for p in ("last", "mean")}
    if module is bias:
        examples = [{"hard_text": "test biography", "profession": 0, "gender": 1}]
    else:
        examples = [{"content": "test article", "label": 0}]
        kwargs["mode"] = "unprompted"
    hashes = module.extract_split(model, FakeTokenizer(), examples, layer=0,
                                  max_length=32, poolings=list(dirs), out_dirs=dirs,
                                  split="train", **kwargs)
    return dirs, hashes


@pytest.mark.parametrize("module", [bias, dbpedia])
def test_gemma_default_fp32_preserves_large_bf16_activations(module, tmp_path):
    model = FakeLM()
    dirs, hashes = extract(module, model, tmp_path)
    expected = float(torch.tensor(model.value, dtype=torch.bfloat16))
    assert expected > float(np.finfo(np.float16).max)
    for directory in dirs.values():
        stored = np.load(directory / "layer_0.npy")
        assert stored.dtype == np.float32
        assert np.isfinite(stored).all()
        np.testing.assert_array_equal(stored, np.full((1, 2), expected, np.float32))
        np.testing.assert_array_equal(np.load(directory / "labels_train.npy"), [0])
    np.testing.assert_array_equal(np.load(dirs["last"] / "tokens_train.npy"), [2])
    assert hashes.shape == (1,)


@pytest.mark.parametrize("module", [bias, dbpedia])
def test_llama_default_still_uses_fp16(module, tmp_path):
    dirs, _ = extract(module, FakeLM(model_type="llama", value=12), tmp_path)
    for directory in dirs.values():
        stored = np.load(directory / "layer_0.npy")
        assert stored.dtype == np.float16
        np.testing.assert_array_equal(stored, np.full((1, 2), 12, np.float16))


@pytest.mark.parametrize("module", [bias, dbpedia])
def test_explicit_float32_override_works_for_other_models(module, tmp_path):
    dirs, _ = extract(module, FakeLM(model_type="llama"), tmp_path, dtype="float32")
    for directory in dirs.values():
        stored = np.load(directory / "layer_0.npy")
        assert stored.dtype == np.float32
        assert np.isfinite(stored).all()


@pytest.mark.parametrize("module", [bias, dbpedia])
def test_unsafe_explicit_fp16_fails_before_writing_inf_cache(module, tmp_path):
    model = FakeLM()
    with pytest.raises(ValueError, match="overflow.*--dtype float32"):
        extract(module, model, tmp_path, dtype="float16")
    assert not list(tmp_path.rglob("*.npy"))
    assert not model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("module", [bias, dbpedia])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_actual_nonfinite_model_output_is_rejected_without_clipping(module, tmp_path, value):
    model = FakeLM(value=value)
    with pytest.raises(ValueError, match="Non-finite model activations"):
        extract(module, model, tmp_path)
    assert not list(tmp_path.rglob("*.npy"))
    assert not model.model.layers[0]._forward_hooks
