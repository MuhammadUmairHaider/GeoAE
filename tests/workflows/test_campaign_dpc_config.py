"""The middle-layer DPC campaign does not inherit other experiments' controls."""
from __future__ import annotations

from dataclasses import fields
import importlib

import numpy as np
import pytest
import torch
import yaml

from experiments.depth.config import CampaignConfig
from geoae.config import Config
from geoae.model import GeoAE


EXCLUDED = ("lloyd_rows", "lloyd_iters", "freeze_centroids")


def test_campaign_config_omits_other_experiment_controls():
    cfg = CampaignConfig()
    serialized = cfg.to_dict()["train"]
    campaign_fields = {field.name for field in fields(cfg.train)}
    shared = Config().to_dict()["train"]
    for key in EXCLUDED:
        assert key in shared
        assert key not in serialized
        assert key not in campaign_fields
        assert not hasattr(cfg.train, key)
    assert set(serialized) == set(shared) - set(EXCLUDED)


@pytest.mark.parametrize("key", EXCLUDED)
def test_campaign_rejects_excluded_yaml_keys_even_at_default(tmp_path, key):
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump({"train": {
        "centroid_init": "dpc", key: getattr(Config().train, key),
    }}))
    with pytest.raises(ValueError, match="Unknown config key"):
        CampaignConfig.from_yaml(config_path)


@pytest.mark.parametrize("mode", ["kmeans++", "seeded", "dpc_lloyd"])
def test_campaign_requires_original_dpc(mode):
    cfg = CampaignConfig()
    cfg.train.centroid_init = mode
    with pytest.raises(ValueError, match="(?i)dpc"):
        cfg.validate()


def test_campaign_dpc_trains_with_moving_ema_and_clean_checkpoint(tmp_path, monkeypatch):
    train_module = importlib.import_module("geoae.training.mse")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("GEOAE_TRAIN_WORKERS", "0")
    monkeypatch.setenv("GEOAE_VAL_WORKERS", "0")
    monkeypatch.delenv("GEOAE_TRAIN_ACTIVATIONS_DIR", raising=False)
    monkeypatch.delenv("GEOAE_DATASET_PROVENANCE", raising=False)

    activations = tmp_path / "activations"
    activations.mkdir()
    np.save(activations / "layer_0.npy", np.random.default_rng(7).normal(size=(16, 4)).astype(np.float32))
    cfg = CampaignConfig()
    cfg.data.activations_dir = str(activations)
    cfg.data.target_layer = 0
    cfg.data.batch_size = 4
    cfg.data.val_frac = 0.25
    cfg.model.hidden_size = 4
    cfg.model.latent_dim = 4
    cfg.model.n_clusters = 2
    cfg.train.centroid_init = "dpc"
    cfg.train.n_epochs = 1
    cfg.train.recon_only_epochs = 0
    cfg.train.clustering_start_epoch = 1
    cfg.train.full_loss_start_epoch = 1
    cfg.train.density_pool = 12
    cfg.train.density_knn = 2
    cfg.train.peak_refine_k = 2
    cfg.train.reinit_every = 0
    cfg.train.diag_every = 100
    cfg.train.checkpoints_dir = str(tmp_path / "checkpoints")

    original_ema = GeoAE.update_centroids_ema
    movements = []

    # A strict signature guards the original EMA call: no freeze/move control
    # can enter this campaign path, even through an otherwise benign default.
    def moving_ema(model, z, q):
        before = model.centroids.clone()
        original_ema(model, z, q)
        movements.append(not torch.equal(before, model.centroids))

    monkeypatch.setattr(GeoAE, "update_centroids_ema", moving_ema)
    train_module.train(cfg, use_wandb=False, campaign_dpc=True)

    assert len(movements) == 3
    assert any(movements)
    checkpoint = torch.load(tmp_path / "checkpoints" / "step_0000003.pt", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["step"] == 3
    assert bool(checkpoint["model_state"]["centroids_initialized"])
    assert checkpoint["config"] == cfg.to_dict()
    assert not set(EXCLUDED).intersection(checkpoint["config"]["train"])
