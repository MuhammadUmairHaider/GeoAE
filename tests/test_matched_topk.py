"""Tests for transform equality, exposure/resume, and reference recipe behavior."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoae.model import GeoAE
from geoae.matched_topk.common import (BypassTransform, MatchedSAE, decoder_rows, load_matched,
                                      make_trainer, reference_provenance)
from geoae.matched_topk.train import RowStream, save
from geoae.matched_topk.stage import stage_dump
from geoae.matched_topk.evaluate import CenteredAffineView, raw_variance_normalization, sanity_metrics
from geoae.interp.sae_coding import reconstruction_report
from geoae.matched_topk.report import compare_on_common_support


@pytest.fixture
def transform():
    return BypassTransform(torch.arange(8).float(),torch.arange(1,9).float(),
                           torch.randn(3,8).half(),torch.tensor([0,-1,1,2,-1]))


def trainer(method):
    return make_trainer(method,64,activation_dim=8,k=2,device="cpu",visits=500_000_000)


@pytest.mark.parametrize("method",["batch_topk","matryoshka"])
def test_reference_recipe_and_initialization(method):
    model = trainer(method)
    assert model.steps == 244141 and model.lr == 5e-5
    assert model.threshold_start_step == 1000 and model.threshold_beta == .999
    assert model.dead_feature_threshold == 10_000_000 and model.top_k_aux == 4
    assert model.decay_start == 195312 and model.k_anneal_steps == 2441
    rows = decoder_rows(model.ae)
    torch.testing.assert_close(rows.norm(dim=1),torch.ones(64),rtol=1e-5,atol=1e-5)
    enc = model.ae.encoder.weight if method == "batch_topk" else model.ae.W_enc.T
    torch.testing.assert_close(enc,rows,rtol=0,atol=0)
    if method == "matryoshka":
        assert model.group_sizes == [2,4,8,16,34]
        assert model.group_weights == [.2]*5
    assert model.optimizer.param_groups[0]["betas"] == (.9,.999)


def test_reference_hashes():
    assert reference_provenance()["sha"] == "60ec6bf5264944d64a4ca271f45a29ebfb9d4946"


def test_stream_full_epochs_and_exact_resume():
    stream = RowStream(17,seed=42)
    first = stream.take(11); epoch,offset = stream.epoch,stream.offset
    next_rows = stream.take(29)
    restored = RowStream(17,seed=42,epoch=epoch,offset=offset)
    np.testing.assert_array_equal(next_rows,restored.take(29))
    np.testing.assert_array_equal(np.sort(np.concatenate([first,next_rows])[:17]),np.arange(17))
    np.testing.assert_array_equal(np.concatenate([first,next_rows]),RowStream(17,42).take(40))
    assert 244140*2048+1280 == 500_000_000


def test_bypass_matches_geoae_encoder_and_decode(transform):
    geo = GeoAE(hidden_size=8,latent_dim=16,n_clusters=2,token_bias_rows=3,vocab_size=5).eval()
    geo.tb_table.copy_(transform.tb_table); geo.tb_index.copy_(transform.tb_index)
    raw,tok = torch.randn(5,8),torch.arange(5)
    assert transform.assert_geoae_equivalence(geo,raw,tok)["encoder_input_max_abs_error"] == 0
    assert torch.equal(transform.token_bias(tok)[[1,4]],torch.zeros(2,8))
    torch.testing.assert_close(transform.restore(transform.preprocess(raw,tok),tok),raw,rtol=1e-5,atol=1e-5)


@pytest.mark.parametrize("method",["batch_topk","matryoshka"])
def test_reference_threshold_state_does_not_depend_on_inference_batch(method):
    t = trainer(method); x = torch.randn(19,8)
    t.loss(x,step=1000); assert float(t.ae.threshold) == -1
    t.loss(x,step=1001); assert float(t.ae.threshold) >= 0
    whole = t.ae.encode(x)
    chunks = torch.cat([t.ae.encode(chunk) for chunk in x.split(4)])
    torch.testing.assert_close(chunks,whole)
    old = t.num_tokens_since_fired.clone(); t.ae.encode(x)
    assert torch.equal(old,t.num_tokens_since_fired)


@pytest.mark.parametrize("method",["batch_topk","matryoshka"])
def test_optimizer_checkpoint_resume_and_raw_delta(method,transform,tmp_path):
    original = trainer(method); x = torch.randn(16,8)
    for step in (0,1,1001): original.update(step,x)
    assert math.isfinite(original.update(1002,x))
    rows = decoder_rows(original.ae)
    torch.testing.assert_close(rows.norm(dim=1),torch.ones(64),rtol=1e-5,atol=1e-5)
    recipe = {"method":method,"width":64,"activation_dim":8,"k":2}
    path = tmp_path/"checkpoint.pt"
    save(path,original,transform,recipe,{"test":True},resume=True)
    frozen,state = load_matched(path,"cpu")
    restored = trainer(method); restored.ae.load_state_dict(state["model_state"])
    restored.optimizer.load_state_dict(state["optimizer_state"]); restored.scheduler.load_state_dict(state["scheduler_state"])
    restored.num_tokens_since_fired.copy_(state["num_tokens_since_fired"])
    original.update(1003,x); restored.update(1003,x)
    for a,b in zip(original.ae.parameters(),restored.ae.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    raw,tok = torch.randn(5,8),torch.arange(5)
    features = frozen.encode(raw,tok); modified = features.clone(); modified[:,0] = 0
    torch.testing.assert_close(frozen.decode(features,tok)-frozen.decode(modified,tok),
                               features[:,0,None]*frozen.decoder_directions[0],rtol=2e-4,atol=2e-5)
    torch.testing.assert_close(frozen.transform.tb_table,transform.tb_table,rtol=0,atol=0)


@pytest.mark.parametrize("method",["batch_topk","matryoshka"])
def test_affine_budget_pipeline_preserves_token_table_and_mse(method,transform):
    t = trainer(method); t.ae.threshold.fill_(.1)
    model = MatchedSAE(t.ae,transform)
    raw,tok = torch.randn(20,8),torch.arange(20)%5
    f = model.encode(raw,tok); table = transform.token_bias(tok)*transform.norm_std
    report = reconstruction_report(raw-table,f,CenteredAffineView(model),budgets=(1,2,4))
    report = raw_variance_normalization(report,raw)
    direct_mse = float((model.decode(f,tok)-raw).detach().double().square().mean())
    assert report["full_sae"]["all"]["raw_mse"] == pytest.approx(direct_mse,rel=1e-5)
    expected = float((raw-raw.mean(0)).double().square().mean())
    assert report["full_sae"]["all"]["normalized_mse"] == pytest.approx(direct_mse/expected,rel=1e-5)


def test_sanity_identity_splice_has_zero_damage():
    raw = torch.randn(20,8); norm = torch.nn.Identity(); head = torch.nn.Linear(8,11)
    report = sanity_metrics(raw,raw,torch.ones(20,4),torch.arange(20)%11,norm,head)
    assert report["raw_fve"] == 1 and report["spliced_delta_nll_nats"] == 0
    assert report["mean_l0"] == 4 and report["finite_gate_passed"]


def test_width_comparison_uses_same_targets_and_family_weights():
    def record(family,sibling):
        return {"family":family,"evaluation":{"sibling":{"delta_nll":sibling},"random":{"delta_nll":0}}}
    left = {(0,"a"):record("f1",1),(0,"b"):record("f1",3),(1,"a"):record("f2",10),
            (2,"a"):record("excluded",1000)}
    right = {(0,"a"):record("f1",0),(0,"b"):record("f1",0),(1,"a"):record("f2",0)}
    report = compare_on_common_support(left,right)
    assert report["n_common_targets"] == 3 and report["n_left_only"] == 1
    assert report["difference_sibling"]["n_families"] == 2
    assert report["difference_sibling"]["mean"] == 6 # average of family means 2 and 10


def test_stage_reuses_immutable_files_and_refuses_source_change(tmp_path,monkeypatch):
    src,dst = tmp_path/"src",tmp_path/"dst"; src.mkdir()
    for name in ("layer_33.npy","rows_tok.npy","norm_params_layer33.npz","meta.json"):
        (src/name).write_bytes(name.encode())
    monkeypatch.setattr("geoae.matched_topk.stage.shutil.disk_usage",lambda _:SimpleNamespace(free=10**12))
    stage_dump(src,dst); first = (dst/"layer_33.npy").stat().st_mtime_ns
    stage_dump(src,dst); assert (dst/"layer_33.npy").stat().st_mtime_ns == first
    (src/"layer_33.npy").write_bytes(b"changed")
    with pytest.raises(ValueError,match="mismatch"): stage_dump(src,dst)
