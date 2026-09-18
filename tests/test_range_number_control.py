"""Validate the control's invariants, holdouts, scoring and LM hook placement."""
from collections import Counter

import numpy as np
import pytest
import torch

from geoae.interp.range_number_control import (
    OrthogonalMix, build_examples, fit_edits, forward_prompts,
    make_edit, make_last_patch, outcome, summarize,
)
from geoae.model import GeoAE


@pytest.mark.parametrize("dim", [8, 11, 32])
def test_rotation_preserves_geometry_and_is_invertible(dim):
    g = torch.Generator().manual_seed(2)
    x, c = torch.randn(30, dim, generator=g), torch.randn(7, dim, generator=g)
    rotation = OrthogonalMix(dim, seed=7)
    y, rc = rotation.forward(x), rotation.forward(c)
    torch.testing.assert_close(rotation.inverse(y), x, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(torch.cdist(y, rc), torch.cdist(x, c), atol=5e-6, rtol=3e-6)
    assert torch.equal(torch.cdist(y, rc).argmin(1), torch.cdist(x, c).argmin(1))
    torch.testing.assert_close(OrthogonalMix(dim, seed=7).forward(x), y)
    assert not torch.allclose(OrthogonalMix(dim, seed=8).forward(x), y)


def test_refit_global_edit_is_rotation_invariant():
    rng = np.random.RandomState(3)
    x = rng.randn(80, 12).astype(np.float32)
    labels = np.arange(80) % 2
    x[labels == 1, :3] += 2
    rotation = OrthogonalMix(12, seed=5)
    tensor = torch.from_numpy(x)
    rotated = rotation.forward(tensor)
    regular_stats = fit_edits(x, labels, 1, 2, 0.3)
    rotated_stats = fit_edits(rotated.numpy(), labels, 1, 2, 0.3)
    expected = make_edit(regular_stats, "global", 1.5, "cpu")(tensor)
    actual = rotation.inverse(make_edit(rotated_stats, "global", 1.5, "cpu")(rotated))
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)


def test_rotation_reconstruction_and_position_control():
    torch.manual_seed(4)
    ae = GeoAE(hidden_size=8, latent_dim=16, n_clusters=4,
               nonlinearity="gelu", latent_norm="batch").eval()
    mean, std = torch.randn(8), torch.rand(8) + 0.5
    hs = torch.randn(3, 7, 8)
    original, _ = make_last_patch(ae, mean, std)
    rotated, _ = make_last_patch(ae, mean, std, rotation=OrthogonalMix(16))
    torch.testing.assert_close(original(hs), rotated(hs), atol=3e-6, rtol=3e-6)
    assert torch.equal(original(hs)[:, :-1], hs[:, :-1])


def test_nouns_and_templates_are_held_out_and_distractors_balanced():
    rows = build_examples()
    fit = [r for r in rows if r["split"] == "fit"]
    test = [r for r in rows if r["split"] == "test"]
    assert len(fit) == 192 and len(test) == 144
    for key in ("subject_lemma", "template", "prompt", "id"):
        assert set(r[key] for r in fit).isdisjoint(r[key] for r in test)
    assert len({r["id"] for r in rows}) == len(rows)
    for split in (fit, test):
        counts = Counter((r["subject_number"], r["attractor_number"]) for r in split)
        assert len(counts) == 4 and len(set(counts.values())) == 1
    assert rows == build_examples()
    assert rows != build_examples(seed=43)


@pytest.mark.parametrize("source", [0, 1])
def test_selective_flip_scores_both_directions(source):
    labels = np.array([0, 0, 1, 1])
    baseline = torch.zeros(4, 5)
    baseline[torch.arange(4), torch.from_numpy(labels) + 2] = 4
    edited = baseline.clone()
    for i in np.flatnonzero(labels == source):
        edited[i, source + 2] = 0
        edited[i, 3 - source] = 4
    values = outcome(edited, baseline, labels, [2, 3])
    score = summarize(values, labels, source, np.ones(4, dtype=bool))
    assert score["target_drop"] == 1
    assert score["complement_drop"] == 0
    assert score["selectivity"] == 1
    assert score["target_counterpart_top1"] == 1
    assert score["complement_kl"] == 0
    empty = summarize(values, labels, source, labels != source)
    assert empty["selectivity"] is None


def test_real_transformer_capture_and_last_token_patch():
    # Instantiate a tiny random LM locally; no model downloads or GPU needed.
    from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    lm = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2)).eval()

    class Tokenizer:
        def __call__(self, texts, padding, return_tensors):
            assert padding and return_tensors == "pt"
            sequences = [[1] + [2 + ord(c) % 25 for c in text] for text in texts]
            width = max(map(len, sequences))
            return BatchEncoding({
                "input_ids": torch.tensor([[0] * (width - len(s)) + s for s in sequences]),
                "attention_mask": torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences]),
            })

    tokenizer = Tokenizer()
    base, captured = forward_prompts(lm, tokenizer, ["a", "abcd"], "cpu", 1, 2, capture=True)
    assert captured.shape == (2, 16)
    replay = lambda hs: torch.cat((hs[:, :-1], captured[:, None].to(hs.dtype)), dim=1)
    same, _ = forward_prompts(lm, tokenizer, ["a", "abcd"], "cpu", 1, 2, patch=replay)
    torch.testing.assert_close(same, base)
    zero_last = lambda hs: torch.cat((hs[:, :-1], torch.zeros_like(hs[:, -1:])), dim=1)
    edited, _ = forward_prompts(lm, tokenizer, ["a", "abcd"], "cpu", 1, 2, patch=zero_last)
    assert not torch.allclose(edited, base)
    restored, _ = forward_prompts(lm, tokenizer, ["a", "abcd"], "cpu", 1, 2)
    torch.testing.assert_close(restored, base)


def test_complete_harness_with_local_tiny_models(tmp_path, monkeypatch):
    """Exercise fitting, every smoke arm, baselines and JSON serialization."""
    import json
    import sys
    from transformers import AutoTokenizer, BatchEncoding, LlamaConfig, LlamaForCausalLM
    from geoae import checkpoint
    from geoae.interp.range_number_control import main

    torch.manual_seed(11)
    lm = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2)).eval()
    ae = GeoAE(hidden_size=16, latent_dim=32, n_clusters=4, nonlinearity="gelu").eval()

    class Tokenizer:
        eos_token = "</s>"

        def encode(self, text, add_special_tokens=False):
            return [{" is": 2, " are": 3}[text]]

        def __call__(self, texts, padding, return_tensors):
            sequences = [[1] + [4 + sum(map(ord, w)) % 28 for w in t.split()] for t in texts]
            width = max(map(len, sequences))
            return BatchEncoding({
                "input_ids": torch.tensor([[0] * (width - len(s)) + s for s in sequences]),
                "attention_mask": torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences]),
            })

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(checkpoint, "load_lm", lambda *a, **kw: lm)
    monkeypatch.setattr(checkpoint, "load_ae_checkpoint", lambda *a, **kw: (
        ae, torch.zeros(16), torch.ones(16),
        {"epoch": 1, "config": {"data": {"target_layer": 1},
                               "extraction": {"model_name": "local-tiny"}}}))
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["range-number", "--checkpoint", "local-tiny",
        "--out", str(output), "--smoke", "--shuffle_labels", "--device", "cpu"])
    main()
    result = json.loads(output.read_text())
    assert result["status"] == "complete"
    assert len(result["arms"]) == 24
    assert result["baselines"]["test_base_pair_accuracy"] >= 0
    for arm in result["arms"].values():
        assert arm["all"]["n_target"] == 12 and arm["all"]["n_complement"] == 12
        assert len(arm["rows"]["correct"]) == 24
        assert len(arm["decoded_edit_norm"]) == 40
        assert np.isfinite(arm["neutral"]["kl"])
    for source in ("singular", "plural"):
        plain = result["arms"][f"z:suppress_{source}:global:a1.0"]
        rotated = result["arms"][f"z_rot0:suppress_{source}:global:a1.0"]
        np.testing.assert_allclose(plain["decoded_edit_norm"], rotated["decoded_edit_norm"], atol=1e-5)
        assert plain["rows"]["correct"] == rotated["rows"]["correct"]
