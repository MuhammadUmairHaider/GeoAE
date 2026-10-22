"""Unit tests for geoae.train_common and config/path plumbing — CPU only."""
import argparse

import torch

from geoae.config import Config
from geoae.model import GeoAE
from geoae.paths import PACKAGE_ROOT, resolve_path
from geoae.train_common import (
    CheckpointTracker,
    apply_overrides,
    add_override_args,
    build_model,
    lambda_schedule,
    reinit_dead_clusters,
    rotate_checkpoints,
    tau_schedule,
)


def make_cfg(**train_overrides) -> Config:
    cfg = Config()
    cfg.train.n_epochs = 10
    cfg.train.recon_only_epochs = 2
    cfg.train.clustering_start_epoch = 3
    cfg.train.full_loss_start_epoch = 6
    for k, v in train_overrides.items():
        setattr(cfg.train, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def test_tau_schedule_flat_before_full_loss_phase():
    cfg = make_cfg()
    for epoch in range(1, cfg.train.full_loss_start_epoch):
        assert tau_schedule(epoch, cfg) == cfg.loss.tau_start


def test_tau_schedule_anneals_to_tau_end():
    cfg = make_cfg()
    taus = [tau_schedule(e, cfg) for e in range(cfg.train.full_loss_start_epoch,
                                                cfg.train.n_epochs + 1)]
    assert taus[0] == cfg.loss.tau_start
    assert abs(taus[-1] - cfg.loss.tau_end) < 1e-9
    # monotone toward tau_end
    diffs = [b - a for a, b in zip(taus, taus[1:])]
    assert all(d <= 0 for d in diffs) or all(d >= 0 for d in diffs)


def test_lambda_schedule_phases():
    cfg = make_cfg()
    # recon-only phase: both zero
    assert lambda_schedule(1, cfg) == (0.0, 0.0)
    assert lambda_schedule(cfg.train.recon_only_epochs, cfg) == (0.0, 0.0)
    # clustering phase: cluster on, sep off
    lam_c, lam_s = lambda_schedule(cfg.train.clustering_start_epoch, cfg)
    assert lam_c == cfg.loss.lambda_cluster and lam_s == 0.0
    # full phase: sep ramps to its configured value by the last epoch
    _, lam_s_final = lambda_schedule(cfg.train.n_epochs, cfg)
    assert abs(lam_s_final - cfg.loss.lambda_sep) < 1e-9


# ---------------------------------------------------------------------------
# Checkpoint rotation / best tracking
# ---------------------------------------------------------------------------

def _touch_ckpt(d, step):
    p = d / f"step_{step:07d}.pt"
    p.write_bytes(b"x")
    return p


def test_rotate_checkpoints_keeps_last_n_and_protected(tmp_path):
    paths = [_touch_ckpt(tmp_path, s) for s in range(1, 6)]
    protected = {paths[0]}
    rotate_checkpoints(tmp_path, keep=2, protected=protected)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert paths[0].name in remaining          # protected survives rotation
    assert paths[-1].name in remaining and paths[-2].name in remaining
    assert paths[1].name not in remaining and paths[2].name not in remaining


def test_checkpoint_tracker_promotes_best_and_rotates(tmp_path):
    tracker = CheckpointTracker(tmp_path, keep=2)
    p1 = _touch_ckpt(tmp_path, 1)
    assert tracker.finish_epoch(p1, score=1.0) is True
    assert tracker.best_target.exists()
    assert tracker.best_score == 1.0

    p2 = _touch_ckpt(tmp_path, 2)
    assert tracker.finish_epoch(p2, score=2.0) is False   # worse → not promoted
    assert tracker.best_score == 1.0

    p3 = _touch_ckpt(tmp_path, 3)
    p4 = _touch_ckpt(tmp_path, 4)
    assert tracker.finish_epoch(p4, score=0.5) is True
    assert tracker.best_score == 0.5
    remaining = {p.name for p in tmp_path.glob("step_*.pt")}
    assert p4.name in remaining and p3.name in remaining
    assert p1.name not in remaining                       # rotated away


def test_checkpoint_tracker_is_better_record_best(tmp_path):
    tracker = CheckpointTracker(tmp_path, keep=3)
    assert tracker.is_better(5.0)
    tracker.record_best(5.0)
    assert not tracker.is_better(5.0)
    assert tracker.is_better(4.9)
    assert tracker.best_path == tracker.best_target


# ---------------------------------------------------------------------------
# Dead-cluster reinit
# ---------------------------------------------------------------------------

def test_reinit_dead_clusters_revives_unused():
    torch.manual_seed(0)
    K, L, D, B = 8, 16, 32, 64
    model = GeoAE(hidden_size=D, latent_dim=L, n_clusters=K)
    model.init_centroids_from_data(torch.randn(K * 2, L))
    # Mark half the clusters dead
    model.ema_cluster_size.fill_(1.0)
    model.ema_cluster_size[:4] = 1e-8

    z = torch.randn(B, L)
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D)
    n = reinit_dead_clusters(model, z, x, x_hat)
    assert n == 4
    assert (model.ema_cluster_size[:4] == 1.0).all()


def test_reinit_dead_clusters_noop_when_healthy():
    K, L, D, B = 8, 16, 32, 64
    model = GeoAE(hidden_size=D, latent_dim=L, n_clusters=K)
    model.ema_cluster_size.fill_(1.0)
    n = reinit_dead_clusters(model, torch.randn(B, L), torch.randn(B, D), torch.randn(B, D))
    assert n == 0


# ---------------------------------------------------------------------------
# build_model / CLI overrides
# ---------------------------------------------------------------------------

def test_build_model_honours_config():
    cfg = Config()
    cfg.model.hidden_size = 32
    cfg.model.latent_dim = 16
    cfg.model.n_clusters = 4
    cfg.model.nonlinearity = "gelu"
    cfg.model.metric = "cosine"
    model = build_model(cfg, torch.device("cpu"), no_sinkhorn=True)
    assert model.latent_dim == 16 and model.n_clusters == 4
    assert model.metric == "cosine"
    assert model.use_sinkhorn is False


def test_apply_overrides_shared_flags():
    ap = argparse.ArgumentParser()
    add_override_args(ap)
    args = ap.parse_args(["--n_clusters", "77", "--latent_dim", "99",
                          "--nonlinearity", "relu", "--checkpoints_dir", "ck",
                          "--no_cluster", "--lambda_sep", "0.25"])
    cfg = Config()
    apply_overrides(cfg, args)
    assert cfg.model.n_clusters == 77
    assert cfg.model.latent_dim == 99
    assert cfg.model.nonlinearity == "relu"
    assert cfg.train.checkpoints_dir == "ck"
    assert cfg.loss.lambda_cluster == 0.0
    assert cfg.loss.lambda_sep == 0.25


def test_apply_overrides_no_sep_only_without_lambda():
    ap = argparse.ArgumentParser()
    add_override_args(ap)
    cfg = Config()
    apply_overrides(cfg, ap.parse_args(["--no_sep"]))
    assert cfg.loss.lambda_sep == 0.0


def test_apply_overrides_tolerates_missing_flags():
    cfg = Config()
    before = cfg.to_dict()
    apply_overrides(cfg, argparse.Namespace())   # no flags at all
    assert cfg.to_dict() == before


# ---------------------------------------------------------------------------
# Config round-trip / path resolution
# ---------------------------------------------------------------------------

def test_config_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.model.n_clusters = 123
    cfg.train.lr = 3e-4
    cfg.data.activations_dir = "somewhere"
    p = tmp_path / "cfg.yaml"
    import yaml
    with open(p, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f)
    cfg2 = Config.from_yaml(p)
    assert cfg2.model.n_clusters == 123
    assert cfg2.train.lr == 3e-4
    assert cfg2.data.activations_dir == "somewhere"


def test_resolve_path_relative_anchors_to_package_root():
    assert resolve_path("activations_diverse") == PACKAGE_ROOT / "activations_diverse"


def test_resolve_path_absolute_passthrough(tmp_path):
    assert resolve_path(tmp_path) == tmp_path
