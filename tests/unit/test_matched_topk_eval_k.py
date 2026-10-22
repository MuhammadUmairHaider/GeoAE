"""Evaluation arm identity and restored-raw-residual fidelity checks."""
import json

import pytest
import torch
import torch.nn.functional as F

from geoae.baselines.matched_topk.common import BypassTransform
from geoae.baselines.matched_topk.common import MatchedSAE
from geoae.baselines.matched_topk.common import model_name
from geoae.baselines.matched_topk.evaluate import checkpoint_identity
from geoae.baselines.matched_topk.evaluate import raw_variance_normalization
from geoae.baselines.matched_topk.evaluate import sanity_metrics


def test_k100_checkpoint_identity_preserves_original_arm_names():
    assert model_name("batch_topk",20480,20) == "batch_topk_20480"
    assert model_name("batch_topk",20480,100) == "batch_topk_20480_k100"
    recipe = {"method":"batch_topk", "width":20480, "k":100, "visits":500_000_000}
    identity = checkpoint_identity({"recipe":recipe},"batch_topk",20480,100)
    assert identity["name"] == "batch_topk_20480_k100"
    assert identity["recipe"]["visits"] == 500_000_000
    json.dumps(identity,allow_nan=False)


@pytest.mark.parametrize("changed",[{"k":20},{"width":10240},{"method":"matryoshka"}])
def test_evaluation_rejects_wrong_checkpoint_arm(changed):
    recipe = {"method":"batch_topk", "width":20480, "k":100, **changed}
    with pytest.raises(ValueError,match="checkpoint recipe mismatch"):
        checkpoint_identity({"recipe":recipe},"batch_topk",20480,100)


def test_fve_and_next_token_splice_use_restored_nonzero_token_mean():
    class IdentityDecoder(torch.nn.Module):
        def decode(self, features):
            return features

    transform = BypassTransform(torch.tensor([10.,-5.]),torch.tensor([2.,3.]),
        torch.tensor([[8.,-3.],[-8.,3.]]),torch.tensor([0,1]))
    model = MatchedSAE(IdentityDecoder(),transform)
    tok = torch.tensor([0,1,0,1,0,1])
    bypass = torch.tensor([[-1.,-.5],[1.,.5],[-.5,1.],[.5,-1.],[1.,1.],[-1.,-1.]])
    features = bypass+torch.tensor([.2,-.1])
    raw = transform.restore(bypass,tok)
    reconstruction = model.decode(features,tok)
    table = transform.token_bias(tok)*transform.norm_std
    torch.testing.assert_close(reconstruction,features*transform.norm_std+transform.norm_mean+table)
    assert bool((table != 0).all())
    head = torch.nn.Linear(2,3,bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[.03,.02],[-.02,.01],[.01,-.03]]))
    head.requires_grad_(False)
    nxt = torch.tensor([0,1,2,0,1,2])
    result = sanity_metrics(raw,reconstruction,features,nxt,torch.nn.Identity(),head)
    raw_error = (raw-reconstruction).double().square().sum()
    raw_variance = (raw-raw.mean(0)).double().square().sum()
    assert result["raw_fve"] == pytest.approx(float(1-raw_error/raw_variance))
    bypass_fve = float(1-(bypass-features).double().square().sum()/
                       (bypass-bypass.mean(0)).double().square().sum())
    assert result["raw_fve"] > bypass_fve+.01
    expected_clean = F.cross_entropy(head(raw),nxt,reduction="none")
    expected_spliced = F.cross_entropy(head(reconstruction),nxt,reduction="none")
    assert result["clean_nll_nats"] == pytest.approx(float(expected_clean.mean()))
    assert result["spliced_nll_nats"] == pytest.approx(float(expected_spliced.mean()))
    assert result["spliced_delta_nll_nats"] == pytest.approx(float((expected_spliced-expected_clean).mean()))
    missing_mean_loss = float(F.cross_entropy(head(reconstruction-table),nxt))
    assert result["spliced_nll_nats"] != pytest.approx(missing_mean_loss)
    assert result["reconstruction_space"] == "raw residual"
    report = {"normalization":{"raw_variance_per_component":1.},
              "full_sae":{"all":{"raw_mse":result["raw_mse"]}}}
    raw_variance_normalization(report,raw)
    assert report["full_sae"]["all"]["raw_fve"] == pytest.approx(result["raw_fve"])
