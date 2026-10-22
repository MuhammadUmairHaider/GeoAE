"""Text-only Gemma loading across direct and read-only forwarding wrappers."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from geoae.checkpoint import _drop_vision_tower, load_lm


@pytest.fixture(autouse=True)
def _isolate_snapshot_environment(monkeypatch):
    monkeypatch.delenv("GEOAE_LM_SNAPSHOTS_JSON", raising=False)


class _DirectWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.Linear(4, 4)
        self.vision_tower = nn.Linear(3, 4)
        self.multi_modal_projector = nn.Linear(4, 4)
        self.config = SimpleNamespace(model_type="gemma3")

    def forward(self, hidden):
        return self.text(hidden)


class _ForwardingWrapper(nn.Module):
    """Match the read-only compatibility properties in Transformers 4.57."""
    def __init__(self):
        super().__init__()
        self.model = _DirectWrapper()
        self.config = self.model.config

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def forward(self, hidden):
        return self.model(hidden)


@pytest.mark.parametrize("wrapper", [_DirectWrapper, _ForwardingWrapper])
def test_drop_vision_preserves_text_weights_and_output(wrapper):
    lm = wrapper().eval()
    hidden = torch.randn(2, 3, 4)
    expected = lm(hidden).detach().clone()
    text = lm.model.text if isinstance(lm, _ForwardingWrapper) else lm.text
    weights = {name: value.detach().clone() for name, value in text.state_dict().items()}

    assert _drop_vision_tower(lm)
    assert lm.vision_tower is None
    assert lm.multi_modal_projector is None
    assert not any("vision_tower" in name or "multi_modal_projector" in name
                   for name, _ in lm.named_parameters())
    for name, value in text.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    torch.testing.assert_close(lm(hidden), expected, rtol=0, atol=0)
    assert not _drop_vision_tower(lm), "dropping twice is harmless"
    if isinstance(lm, _ForwardingWrapper):
        assert "vision_tower" not in lm._modules
        assert "multi_modal_projector" not in lm._modules


def test_text_only_model_has_nothing_to_drop():
    lm = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    assert not _drop_vision_tower(lm)


def test_actual_module_owner_errors_propagate():
    class BrokenOwner(_DirectWrapper):
        def __setattr__(self, name, value):
            if name == "vision_tower" and value is None:
                raise RuntimeError("cannot remove owned tower")
            super().__setattr__(name, value)

    with pytest.raises(RuntimeError, match="cannot remove owned tower"):
        _drop_vision_tower(BrokenOwner())


@pytest.mark.parametrize("text_only", [True, False])
def test_load_lm_freezes_and_evaluates_forwarding_wrapper(monkeypatch, text_only):
    from transformers import AutoModelForCausalLM
    from geoae import hf_auth

    lm = _ForwardingWrapper().train()
    hidden = torch.randn(2, 3, 4)
    expected = lm(hidden).detach().clone()
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return lm

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(hf_auth, "ensure_hf_login", lambda: None)
    loaded = load_lm("offline-tiny-gemma", device="cpu", text_only=text_only)

    assert loaded is lm
    assert calls == [("offline-tiny-gemma", {"dtype": torch.bfloat16})]
    assert all(not module.training for module in loaded.modules())
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert (lm.vision_tower is None) == text_only
    assert (lm.multi_modal_projector is None) == text_only
    torch.testing.assert_close(loaded(hidden), expected, rtol=0, atol=0)


def test_real_tiny_gemma_text_logits_survive_vision_removal():
    # Construct from random weights locally; no gated assets or downloads.
    from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

    text_config = Gemma3TextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64, query_pre_attn_scalar=8,
        sliding_window=8, layer_types=["sliding_attention", "full_attention"],
    )
    vision_config = SiglipVisionConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, image_size=8, patch_size=4,
    )
    config = Gemma3Config(
        text_config=text_config, vision_config=vision_config,
        mm_tokens_per_image=4, image_token_index=63,
        boi_token_index=61, eoi_token_index=62,
    )
    config._attn_implementation = "eager"
    lm = Gemma3ForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[2, 3, 4, 5]])
    with torch.inference_mode():
        expected = lm(input_ids=input_ids, use_cache=False).logits.clone()
        assert _drop_vision_tower(lm)
        actual = lm(input_ids=input_ids, use_cache=False).logits
    assert lm.model.vision_tower is None
    assert lm.model.multi_modal_projector is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_load_lm_local_snapshot_is_offline_and_keeps_semantic_name(tmp_path, monkeypatch, capsys):
    from transformers import AutoModelForCausalLM
    from geoae import hf_auth

    snapshot = tmp_path / "verified-model"
    snapshot.mkdir()
    monkeypatch.setenv("GEOAE_LM_SNAPSHOTS_JSON", json.dumps({"google/gemma-3-4b-pt": str(snapshot)}))
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return _ForwardingWrapper()

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(hf_auth, "ensure_hf_login", lambda: None)
    lm = load_lm("google/gemma-3-4b-pt", device_map="auto")
    assert calls == [(str(snapshot), {
        "dtype": torch.bfloat16, "local_files_only": True, "device_map": "auto",
    })]
    assert "[load_lm] google/gemma-3-4b-pt:" in capsys.readouterr().out
    assert lm.vision_tower is None
    assert not any(parameter.requires_grad for parameter in lm.parameters())


def test_load_lm_unlisted_id_uses_original_loading_behavior(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM
    from geoae import hf_auth

    monkeypatch.setenv("GEOAE_LM_SNAPSHOTS_JSON", json.dumps({"other/model": str(tmp_path)}))
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return _ForwardingWrapper()

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(hf_auth, "ensure_hf_login", lambda: None)
    load_lm("google/gemma-3-4b-pt")
    assert calls == [("google/gemma-3-4b-pt", {"dtype": torch.bfloat16})]


@pytest.mark.parametrize("raw", ["{broken", "[]", "null", '{"model": 7}',
                                 '{"model": "relative/path"}', '{"": "/"}'])
def test_load_lm_rejects_bad_snapshot_mapping_before_loading(raw, monkeypatch):
    from transformers import AutoModelForCausalLM

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid overrides must fail before model loading")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", unexpected_load)
    monkeypatch.setenv("GEOAE_LM_SNAPSHOTS_JSON", raw)
    with pytest.raises(ValueError, match="GEOAE_LM_SNAPSHOTS_JSON"):
        load_lm("google/gemma-3-4b-pt")


def test_load_lm_rejects_missing_snapshot_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOAE_LM_SNAPSHOTS_JSON", json.dumps({
        "google/gemma-3-4b-pt": str(tmp_path / "missing"),
    }))
    with pytest.raises(ValueError, match="absolute existing directory"):
        load_lm("google/gemma-3-4b-pt")
