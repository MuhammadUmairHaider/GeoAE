"""Resume semantics for the base (MSE) trainer. CPU only, no activations.

Guards what a naive resume gets wrong on clustering runs, where val_mse rises once
the cluster terms engage and best_val.pt ends up being a pre-clustering epoch:
  * `--resume latest` must pick the newest step, not best_val.pt;
  * resuming from an older checkpoint in the same directory would re-save step
    names that already exist and overwrite every later epoch — refused;
  * best_val.pt must not be replaced by the first (worse) post-resume epoch;
  * the base trainer's norm stats are numpy, not tensors.
"""
import numpy as np
import pytest
import torch

from geoae.config import Config
from geoae.train import save_checkpoint
from geoae.train_common import (
    CheckpointTracker, build_model, load_resume, make_optimizer, resolve_resume,
)

D, K = 16, 8


def make_cfg():
    cfg = Config()
    cfg.model.hidden_size = D
    cfg.model.latent_dim = D
    cfg.model.n_clusters = K
    cfg.model.nonlinearity = "tanh"
    return cfg


def write(path, cfg, epoch, step, val_mse, centroids=True):
    model = build_model(cfg, torch.device("cpu"), no_sinkhorn=True)
    model.centroids_initialized.fill_(centroids)
    opt = make_optimizer(model, cfg)
    model(torch.randn(4, D)).x_hat.sum().backward()
    opt.step()
    save_checkpoint(path, model, opt, epoch, step, 0.6,
                    np.zeros(D, np.float32), np.ones(D, np.float32), val_mse, cfg)


@pytest.fixture
def run_dir(tmp_path):
    """A stopped run: best_val is epoch 5 (pre-clustering), newest step is epoch 28."""
    cfg = make_cfg()
    write(tmp_path / "step_0001420.pt", cfg, 5, 1420, 0.01382, centroids=False)
    write(tmp_path / "best_val.pt", cfg, 5, 1420, 0.01382, centroids=False)
    write(tmp_path / "step_0007668.pt", cfg, 27, 7668, 0.03273)
    write(tmp_path / "step_0007952.pt", cfg, 28, 7952, 0.03369)
    return cfg, tmp_path


def test_latest_picks_highest_step(run_dir):
    _, d = run_dir
    assert resolve_resume("latest", d).name == "step_0007952.pt"


def test_refuses_resume_that_would_overwrite_later_epochs(run_dir):
    _, d = run_dir
    with pytest.raises(ValueError, match="would overwrite"):
        resolve_resume(str(d / "best_val.pt"), d)


def test_allows_branching_from_an_older_checkpoint_into_a_new_dir(run_dir, tmp_path_factory):
    _, d = run_dir
    new_dir = tmp_path_factory.mktemp("branch")
    assert resolve_resume(str(d / "best_val.pt"), new_dir) == d / "best_val.pt"


def test_load_resume_with_numpy_norm_stats(run_dir):
    cfg, d = run_dir
    model = build_model(cfg, torch.device("cpu"), no_sinkhorn=True)
    opt = make_optimizer(model, cfg)
    start, step = load_resume(resolve_resume("latest", d), model, opt,
                              np.zeros(D, np.float32), np.ones(D, np.float32), torch.device("cpu"))
    assert (start, step) == (29, 7952)
    assert bool(model.centroids_initialized.item()), "centroid init must not rerun"
    assert len(opt.state) > 0
    with pytest.raises(ValueError, match="does not match"):
        load_resume(d / "step_0007952.pt", model, opt,
                    np.full(D, 3.0, np.float32), np.ones(D, np.float32), torch.device("cpu"))


def test_restore_best_guards_best_val(run_dir):
    _, d = run_dir
    tracker = CheckpointTracker(d, keep=40)
    assert tracker.restore_best("val_mse")
    assert tracker.best_score == pytest.approx(0.01382)
    assert not tracker.is_better(0.03400), "the first post-resume epoch must not replace best_val"


def test_restore_best_without_best_val(tmp_path):
    assert not CheckpointTracker(tmp_path, keep=3).restore_best("val_mse")
