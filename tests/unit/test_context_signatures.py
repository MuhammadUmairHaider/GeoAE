"""Middle-block cluster profiles must measure the full downstream computation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from geoae.evaluation.context_logits import capture_context_residuals
from geoae.evaluation.context_logits import context_signature_batches
from geoae.evaluation.context_logits import final_residual_logits
from geoae.evaluation.context_logits import prepare_final_readout
from geoae.evaluation.context_logits import validate_layer
from geoae.analysis.cluster_steer_generate import collect_signature_statistics


class Block(nn.Module):
    def __init__(self, contextual=False):
        super().__init__()
        self.contextual = contextual
        self.calls = 0

    def forward(self, hidden, attention_mask):
        self.calls += 1
        if self.contextual:
            hidden = hidden + .3 * (hidden * attention_mask[..., None]).cumsum(1)
        return (hidden,)


class TinyContextLM(nn.Module):
    """The same token has the same L0 vector, but different contextual predictions."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embedding = nn.Embedding(9, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(), Block(contextual=True)])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 9, bias=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(final_logit_softcapping=.8))

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        hidden = self.embedding(input_ids)
        for block in self.model.layers:
            hidden = block(hidden, mask)[0]
        logits = self.lm_head(self.model.norm(hidden))
        cap = self.config.text_config.final_logit_softcapping
        return SimpleNamespace(logits=torch.tanh(logits / cap) * cap)


@torch.no_grad()
def test_middle_block_profiles_execute_remaining_attention_with_full_context():
    lm = TinyContextLM().eval()
    documents = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6, 3]])]
    expected = [lm(ids).logits[0, -1].softmax(-1) for ids in documents]
    lm.model.layers[1].calls = 0
    stream = [(ids, "source", np.array([ids.shape[-1] - 1])) for ids in documents]
    rows = list(context_signature_batches(lm, 0, stream, "cpu", 0, batch_docs=2))
    assert lm.model.layers[1].calls >= 1
    torch.testing.assert_close(rows[0][0], rows[1][0])  # identical L0 residuals
    assert not torch.allclose(expected[0], expected[1])  # actual predictions depend on prefix
    for i, (raw, tok, doc, probabilities, domain) in enumerate(rows):
        torch.testing.assert_close(probabilities[0], expected[i])
        assert tok.tolist() == [3] and doc.tolist() == [i] and domain == "source"
        assert not torch.allclose(final_residual_logits(lm, raw).softmax(-1)[0], expected[i])
        assert not probabilities.requires_grad
    assert lm._geoae_signature_readout["softcap"] == .8
    assert not lm.model.layers[0]._forward_hooks
    assert not lm.model.norm._forward_pre_hooks


@torch.no_grad()
def test_sampled_positions_and_softcapped_readout_match_native_logits():
    lm = TinyContextLM().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    reference = lm(ids).logits[0]
    rows = list(context_signature_batches(lm, 0, [(ids, "s", [1, 3, 4])], "cpu", 0, chunk=2))
    assert [len(row[0]) for row in rows] == [2, 1]
    probabilities = torch.cat([row[3] for row in rows])
    torch.testing.assert_close(probabilities, reference[[1, 3, 4]].softmax(-1))
    assert torch.cat([row[1] for row in rows]).tolist() == [2, 4, 5]


def test_capture_hooks_are_removed_on_model_failure():
    lm = TinyContextLM()
    original = lm.model.layers[1].forward

    def fail(*args):
        raise RuntimeError("broken tail")

    lm.model.layers[1].forward = fail
    ids = torch.tensor([[1, 2]])
    with pytest.raises(RuntimeError, match="broken tail"):
        capture_context_residuals(lm, 0, ids, torch.ones_like(ids))
    assert not lm.model.layers[0]._forward_hooks
    assert not lm.model.norm._forward_pre_hooks
    lm.model.layers[1].forward = original
    assert lm(ids).logits.shape == (1, 2, 9)


def test_unknown_native_readout_is_rejected_instead_of_guessing():
    lm = TinyContextLM()
    original = lm.forward

    def scaled(*args, **kwargs):
        result = original(*args, **kwargs)
        result.logits = result.logits * 2
        return result

    lm.forward = scaled
    with pytest.raises(RuntimeError, match="unsupported native final-logit transform"):
        prepare_final_readout(lm, torch.tensor([[1, 2, 3]]))
    assert not lm.model.norm._forward_pre_hooks
    assert not hasattr(lm, "_geoae_signature_readout")


@pytest.mark.parametrize("layer", [-1, 2])
def test_invalid_layer_rejected_before_registering_hooks(layer):
    lm = TinyContextLM()
    with pytest.raises(ValueError, match="out of range"):
        validate_layer(lm, layer)
    assert not lm.model.layers[0]._forward_hooks


def test_reference_statistics_use_identical_context_rows_for_both_codebooks():
    lm = TinyContextLM().eval()
    documents = [(torch.tensor([[1, 2, 3]]), "a", [1, 2]),
                 (torch.tensor([[4, 5, 6, 3]]), "b", [0, 3])]
    batches = list(context_signature_batches(lm, 0, documents, "cpu", 0))

    class Codebook:
        C = torch.zeros(2, 4)

        def assign(self, x, tok):
            return tok % 2

    arms = {name: Codebook() for name in ["base", "bypass"]}
    mean, std = torch.tensor([.5, -.3, .1, -.2]), torch.tensor([1., 2., .5, 1.5])
    sums, pbar, tok, doc, sources = collect_signature_statistics(
        arms, iter(batches), mean, std, 9)
    normalized = (torch.cat([row[0] for row in batches]) - mean) / std
    probs = torch.cat([row[3] for row in batches])
    torch.testing.assert_close(pbar, probs.mean(0))
    assert sources == {"a": 2, "b": 2} and doc.tolist() == [0, 0, 1, 1]
    for name in arms:
        torch.testing.assert_close(sums[name]["labels"], tok % 2)
        for k in range(2):
            torch.testing.assert_close(sums[name]["M"][k], normalized[tok % 2 == k].sum(0))
            torch.testing.assert_close(sums[name]["P"][k], probs[tok % 2 == k].sum(0))


@pytest.mark.parametrize("architecture", ["llama", "gemma3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
@torch.no_grad()
def test_hf_middle_signatures_match_complete_forward_with_right_padding(architecture, dtype):
    """Pin native rotary/sliding masks and Gemma's multimodal decoder layout offline."""
    from transformers import LlamaConfig, LlamaForCausalLM, Gemma3Config, Gemma3ForConditionalGeneration
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    if architecture == "llama":
        lm = LlamaForCausalLM(LlamaConfig(**text)).eval()
    else:
        lm = Gemma3ForConditionalGeneration(Gemma3Config(
            text_config={**text, "sliding_window": 4, "final_logit_softcapping": 1.0},
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=16, patch_size=8),
        )).eval()
    lm.to(dtype=dtype)
    documents = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6, 7, 8, 9]])]
    expected = [lm(ids, use_cache=False).logits[0, [1, ids.shape[-1] - 1]].float().softmax(-1)
                for ids in documents]
    stream = [(ids, "source", [1, ids.shape[-1] - 1]) for ids in documents]
    rows = list(context_signature_batches(lm, 0, stream, "cpu", 0, batch_docs=2))
    for row, reference in zip(rows, expected):
        torch.testing.assert_close(row[3], reference, atol=1e-6, rtol=1e-5)
    if architecture == "gemma3":
        # The supported native readout is measured, since HF wrapper versions
        # differ on whether they actually apply the configured text softcap.
        assert lm._geoae_signature_readout["max_validation_error"] < 1e-5
