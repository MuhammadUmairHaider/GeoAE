"""Exercise the additional sparsity arm and raw-coordinate validation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoae.matched_topk.common import BypassTransform, load_matched, make_trainer, model_name
from geoae.matched_topk.train import run, validate
from geoae.model import GeoAE


def test_k100_reference_schedule_and_checkpoint_identity(tmp_path):
    from geoae.matched_topk.train import save

    trainer = make_trainer("batch_topk", 256, activation_dim=128, k=100, device="cpu")
    x = torch.randn(8, 128)
    trainer.update(0, x)
    trainer.update(3000, x)
    assert int(trainer.ae.k) == 100 and trainer.k == 100
    assert trainer.lr == 5e-5 and trainer.steps == 244141
    transform = BypassTransform(torch.zeros(128), torch.ones(128),
                                torch.zeros(1, 128), torch.tensor([0]))
    path = tmp_path / "k100.pt"
    save(path, trainer, transform, {"method": "batch_topk", "width": 256,
                                   "activation_dim": 128, "k": 100}, {})
    restored, state = load_matched(path, "cpu")
    assert state["recipe"]["k"] == 100 and int(restored.core.k) == 100
    assert model_name("batch_topk", 20480, 100) == "batch_topk_20480_k100"
    assert model_name("batch_topk", 20480) == "batch_topk_20480"


def test_validation_uses_raw_residual_sst_and_restores_token_means():
    transform = BypassTransform(torch.zeros(2), torch.tensor([2., 3.]),
                                torch.tensor([[8., -5.], [-8., 5.]]), torch.tensor([0, 1]))
    tokens = np.array([0, 1, 0, 1])
    bypass = torch.tensor([[1., 2.], [2., 3.], [3., 1.], [4., 4.]])
    raw = transform.restore(bypass, torch.from_numpy(tokens))
    core = SimpleNamespace(encode=lambda x, use_threshold: x * .5,
                           decode=lambda f: f, threshold=torch.tensor(0.), k=torch.tensor(100))
    report = validate({"arm": SimpleNamespace(ae=core)}, raw.numpy(), tokens, 0,
                      transform, "cpu", batch_size=2)["arm"]
    reconstructed = transform.restore(bypass * .5, torch.from_numpy(tokens))
    sse = (raw - reconstructed).double().square().sum()
    sst = (raw.double() - raw.double().mean(0)).square().sum()
    assert report["raw_fve"] == pytest.approx(float(1 - sse / sst))
    assert report["raw_mse"] == pytest.approx(float(sse / raw.numel()))
    assert report["raw_fve"] > report["bypass_fve"] + .1


def test_real_cpu_smoke_selects_only_k100_and_preserves_budget(tmp_path, monkeypatch):
    """Run the actual runner, normalization assertion, optimizer and checkpoint."""
    torch.set_num_threads(2)
    dump = tmp_path / "dump"; dump.mkdir()
    out = tmp_path / "out"; out.mkdir()
    rng = np.random.default_rng(42)
    np.save(dump / "layer_33.npy", rng.standard_normal((20, 2560)).astype(np.float32))
    np.save(dump / "rows_tok.npy", np.arange(20, dtype=np.int64) % 2)
    np.savez(dump / "norm_params_layer33.npz", mean=np.zeros(2560, np.float32),
             std=np.ones(2560, np.float32))
    (out / "analysis_plan.md").write_text("Prospective k100 test plan")
    checkpoint = tmp_path / "geo.pt"; checkpoint.write_bytes(b"fixture")
    geo = GeoAE(hidden_size=2560, latent_dim=4, n_clusters=2, token_bias_rows=2, vocab_size=2)
    geo.tb_table.copy_(torch.randn_like(geo.tb_table) * .1)
    geo.tb_index.copy_(torch.tensor([0, 1]))
    monkeypatch.setattr("geoae.matched_topk.train.D.load_geoae", lambda *args:
                        (geo, torch.zeros(2560), torch.ones(2560),
                         {"config": {"data": {"val_frac": .05}, "train": {"seed": 42}}}))
    args = SimpleNamespace(dump=str(dump), out=str(out), geoae_checkpoint=str(checkpoint),
                           analysis_plan=None, widths=[2560], methods=["batch_topk"], k=100,
                           visits=500_000_000, batch_size=4, seed=42, device="cpu", cpu_threads=2,
                           log_every=1, smoke_steps=2, resume=False)
    run(args)
    path = out / "smoke" / "batch_topk_2560_k100" / "smoke.pt"
    model, saved = load_matched(path, "cpu")
    assert saved["recipe"]["k"] == 100 and int(model.core.k) == 100
    assert not (out / "smoke" / "matryoshka_2560_k100").exists()
    assert not (out / "smoke" / "batch_topk_2560").exists()
    import json
    receipt = json.loads((out / "smoke" / "smoke_k100.json").read_text())
    # The small CPU batch still retains the full 500M-visit schedule.
    assert receipt["full_schedule_steps"] == 125_000_000
    assert receipt["preprocessing"]["activation_buffer_input_max_abs_error"] == 0
