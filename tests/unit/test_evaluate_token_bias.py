"""Causal evaluation must preserve the current-token bypass on a GPT-2 splice."""
import pytest
import torch

from geoae.evaluation.faithfulness import ce_per_token
from geoae.evaluation.faithfulness import experiment_loss_recovered
from geoae.evaluation.faithfulness import experiment_per_cluster_ablation
from geoae.hooks import SplicingHook
from geoae.model import GeoAE


class FixedTokenizer:
    ids = torch.tensor([[3, 4, 5, 9]])

    def __call__(self, text, **kwargs):
        return {"input_ids": self.ids.clone()}

    def decode(self, ids):
        return str(ids[0])


def setup_case(mode):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(42)
    lm = GPT2LMHeadModel(GPT2Config(
        vocab_size=16, n_embd=8, n_layer=2, n_head=2, n_positions=16,
        bos_token_id=0, eos_token_id=1,
    )).eval()
    ae = GeoAE(hidden_size=8, latent_dim=12, n_clusters=2,
               token_bias_rows=3 if mode else 0, vocab_size=16 if mode else 0,
               token_bias_mode=mode or "bypass").eval()
    if mode:
        ae.load_token_bias(torch.tensor([3, 4, 5]), torch.randn(3, 8) * 3)
    ae.centroids.copy_(torch.randn(2, 12) * 0.1)
    norms = {"mean": torch.randn(8), "std": torch.rand(8) + 0.5}
    return lm, ae, norms


@torch.no_grad()
def reference_reconstruction(lm, ae, norms):
    ids = FixedTokenizer.ids
    captured = []
    hook = SplicingHook(lm, 1)
    hook.activate(lambda hs: (captured.append(hs.clone()) or hs))
    lm(input_ids=ids)
    hook.deactivate()
    hs = captured[0]
    flat = (hs.reshape(-1, 8) - norms["mean"]) / norms["std"]
    bias = ae.token_bias(ids.reshape(-1))
    encoder_input = flat
    if bias is not None and ae.token_bias_mode == "bypass":
        encoder_input = flat - bias
    z = ae.encoder(encoder_input)
    reconstruction = ae.decoder(z)
    if bias is not None:
        reconstruction = reconstruction + bias
    raw = (reconstruction * norms["std"] + norms["mean"]).reshape_as(hs)
    hook.activate(lambda hs: raw)
    logits = lm(input_ids=ids).logits
    hook.deactivate()
    return ce_per_token(logits, ids).mean().item(), hs, z, bias


@pytest.mark.parametrize("mode", [None, "bypass", "decoder_only"])
def test_loss_recovered_splice_preserves_token_bias(mode):
    lm, ae, norms = setup_case(mode)
    expected, *_ = reference_reconstruction(lm, ae, norms)
    result = experiment_loss_recovered(
        lm, ae, norms, FixedTokenizer(), ["fixed"], 1, torch.device("cpu"),
    )
    assert result["per_text"][0]["ce_spliced"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("mode", [None, "bypass", "decoder_only"])
@torch.no_grad()
def test_cluster_ablation_readds_each_current_tokens_bias(mode):
    lm, ae, norms = setup_case(mode)
    _, hs, z, bias = reference_reconstruction(lm, ae, norms)
    ids = FixedTokenizer.ids
    original_ce = ce_per_token(lm(input_ids=ids).logits, ids)
    flat = (hs.reshape(-1, 8) - norms["mean"]) / norms["std"]
    Q = ae(flat, ids.reshape(-1)).Q
    expected = {}
    hook = SplicingHook(lm, 1)
    for k in range(ae.n_clusters):
        ablated = ae.decoder(z - Q[:, k:k + 1] * ae.centroids[k:k + 1])
        if bias is not None:
            ablated = ablated + bias
        raw = (ablated * norms["std"] + norms["mean"]).reshape_as(hs)
        hook.activate(lambda hs: raw)
        ce = ce_per_token(lm(input_ids=ids).logits, ids)
        hook.deactivate()
        expected[k] = float((ce - original_ce).mean())

    result = experiment_per_cluster_ablation(
        lm, ae, norms, FixedTokenizer(), ["fixed"], 1, torch.device("cpu"),
    )
    for k, delta in expected.items():
        assert result[k]["mean_ce_change"] == pytest.approx(delta, abs=1e-6)
