"""Exercise nonlinear decoders through the actual MSE training/checkpoint path."""

import numpy as np
import pytest
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.config import Config
from geoae.train import train


@pytest.mark.parametrize("decoder_type", ["mlp", "residual_mlp"])
def test_nonlinear_bypass_training_saves_usable_clustered_checkpoint(tmp_path, monkeypatch, decoder_type):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("GEOAE_TRAIN_WORKERS", "0")
    monkeypatch.setenv("GEOAE_VAL_WORKERS", "0")
    monkeypatch.delenv("GEOAE_TRAIN_ACTIVATIONS_DIR", raising=False)
    monkeypatch.delenv("GEOAE_DATASET_PROVENANCE", raising=False)
    rng = np.random.default_rng(103)
    raw = rng.normal(size=(128, 8)).astype(np.float32)
    tokens = np.arange(len(raw), dtype=np.int32) % 4
    np.save(tmp_path / "layer_35.npy", raw)
    np.save(tmp_path / "rows_tok.npy", tokens)
    mean = raw[:96].mean(0)
    std = np.sqrt(((raw[:96] - mean) ** 2).mean(0)) + 1e-8
    np.savez(tmp_path / "norm_params_layer35.npz", mean=mean, std=std)
    table_path = tmp_path / "token_bias.npz"
    np.savez(table_path, token_ids=np.arange(4), table=rng.normal(size=(4, 8)).astype(np.float16),
             vocab_size=4, norm_mean=mean, norm_std=std, shrink_k=30., min_count=10)

    cfg = Config()
    cfg.extraction.model_name = "tiny-qwen"
    cfg.data.activations_dir = str(tmp_path)
    cfg.data.target_layer = 35
    cfg.data.val_frac = .25
    cfg.data.batch_size = 16
    cfg.model.hidden_size = 8
    cfg.model.latent_dim = 16
    cfg.model.n_clusters = 4
    cfg.model.nonlinearity = "gelu"
    cfg.model.latent_norm = "batch"
    cfg.model.token_bias = str(table_path)
    cfg.model.decoder_type = decoder_type
    cfg.model.decoder_hidden_dim = 6
    cfg.model.decoder_output_bias = decoder_type == "mlp"
    cfg.loss.lambda_cluster = .1
    cfg.loss.lambda_sep = .01
    cfg.loss.lambda_var = .05
    cfg.loss.lambda_cov = .01
    cfg.train.n_epochs = 2
    cfg.train.recon_only_epochs = 1
    cfg.train.clustering_start_epoch = 2
    cfg.train.full_loss_start_epoch = 2
    cfg.train.centroid_init = "kmeans++"
    cfg.train.checkpoints_dir = str(tmp_path / "checkpoints")
    cfg.train.diag_every = 1000
    cfg.train.reinit_every = 0
    cfg.train.save_every = 1
    cfg.validate()

    train(cfg, use_wandb=False)
    checkpoint = tmp_path / "checkpoints" / "step_0000012.pt"
    ae, saved_mean, saved_std, saved = load_ae_checkpoint(checkpoint, "cpu", allow_token_bias=True)
    assert saved["epoch"] == 2
    assert saved["step"] == 12
    assert np.isfinite(saved["val_mse"]) and saved["val_mse"] > 0
    assert ae.centroids_initialized.item()
    assert torch.isfinite(ae.centroids).all()
    assert ae.decoder_type == decoder_type
    assert saved["opt_state"]["state"]
    x = (torch.from_numpy(raw[-16:]) - saved_mean) / saved_std
    output = ae(x, torch.from_numpy(tokens[-16:]))
    assert torch.isfinite(output.x_hat).all()
    assert output.x_hat.shape == x.shape
