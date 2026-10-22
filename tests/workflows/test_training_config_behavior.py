"""Small CPU regressions for configuration reaching training and checkpoint reloads."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from geoae.checkpoint import load_ae_checkpoint
from geoae.config import Config
from geoae.training.kl_losses import total_loss_e2e
from geoae.losses import sep_loss, variance_loss
from geoae.training.mse import save_checkpoint
from geoae.training.common import add_override_args
from geoae.training.common import apply_overrides
from geoae.training.common import build_model
from geoae.training.common import load_resume
from geoae.training.common import make_optimizer


def tiny_config(tmp_path):
    cfg = Config()
    cfg.model.hidden_size = cfg.model.latent_dim = 4
    cfg.model.n_clusters = 2
    cfg.data.target_layer = 0
    cfg.data.activations_dir = str(tmp_path)
    cfg.train.checkpoints_dir = str(tmp_path / "checkpoints")
    cfg.train.n_epochs = 1
    cfg.train.reinit_every = 0
    cfg.train.teacher_mode = "onfly"
    cfg.loss.sep_mode, cfg.loss.sep_margin, cfg.loss.var_gamma = "hinge", 1.7, 0.25
    cfg.loss.lambda_var = 0.3
    return cfg


def test_kl_geometry_uses_nondefault_separation_and_variance_target():
    z = torch.tensor([[0., 0.], [0.2, 0.], [0.4, 0.], [1., 0.], [1.2, 0.], [1.4, 0.]],
                     requires_grad=True)
    q = nn.functional.one_hot(torch.tensor([0, 0, 0, 1, 1, 1]), 2).float()
    logits = torch.zeros(6, 3)
    kwargs = dict(teacher_logits=logits, student_logits=logits, x=z.detach(), x_hat=z,
                  z=z, centroids=torch.zeros(2, 2), Q=q, lambda_cluster=0., lambda_sep=0.4,
                  lambda_var=0.3, sep_mode="hinge", sep_margin=10., var_gamma=0.25)
    losses = total_loss_e2e(**kwargs)
    expected_sep = sep_loss(z, q, mode="hinge", margin=10.)
    expected_var = variance_loss(z, gamma=0.25)
    torch.testing.assert_close(losses["loss"], 0.4 * expected_sep + 0.3 * expected_var)
    losses["loss"].backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert expected_sep > sep_loss(z, q, mode="median")
    assert expected_var < variance_loss(z)


@pytest.mark.parametrize("configured,cli", [(False, True), (True, False), (False, False)])
def test_assignment_mode_survives_yaml_cli_checkpoint_and_resume(tmp_path, configured, cli):
    cfg = tiny_config(tmp_path)
    cfg.loss.no_sinkhorn = configured
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(f"loss:\n  no_sinkhorn: {str(configured).lower()}\n")
    assert Config.from_yaml(recipe).loss.no_sinkhorn == configured
    parser = ArgumentParser()
    add_override_args(parser)
    apply_overrides(cfg, parser.parse_args(["--no_sinkhorn"] if cli else []))
    model = build_model(cfg, torch.device("cpu"))
    expected = not (configured or cli)
    assert model.use_sinkhorn == expected
    model.centroids.copy_(torch.arange(8).reshape(2, 4) / 10)
    model.eval()
    opt = make_optimizer(model, cfg)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, opt, 1, 1, model.tau, np.zeros(4), np.ones(4), 0., cfg)
    loaded, _, _, saved = load_ae_checkpoint(path, "cpu")
    assert saved["config"]["loss"]["no_sinkhorn"] == (configured or cli)
    assert loaded.use_sinkhorn == expected
    x = torch.arange(16).reshape(4, 4).float() / 10
    assert torch.equal(model(x).Q, loaded(x).Q)
    resumed = build_model(cfg, torch.device("cpu"))
    assert load_resume(path, resumed, make_optimizer(resumed, cfg), torch.zeros(4), torch.ones(4), "cpu") == (2, 1)
    resumed.use_sinkhorn = not expected
    with pytest.raises(ValueError, match="assignment mismatch"):
        load_resume(path, resumed, make_optimizer(resumed, cfg), torch.zeros(4), torch.ones(4), "cpu")


def test_direct_no_sinkhorn_override_is_persisted_and_legacy_load_is_unchanged(tmp_path):
    cfg = tiny_config(tmp_path)
    model = build_model(cfg, torch.device("cpu"), no_sinkhorn=True)
    assert cfg.loss.no_sinkhorn and not model.use_sinkhorn
    config = cfg.to_dict()
    config["loss"].pop("no_sinkhorn")
    path = tmp_path / "legacy.pt"
    torch.save(dict(model_state=model.state_dict(), config=config, norm_mean=np.zeros(4),
                    norm_std=np.ones(4), epoch=1), path)
    loaded, *_ = load_ae_checkpoint(path, "cpu")
    assert loaded.use_sinkhorn  # Historical checkpoints did not record the CLI flag.


class TinyFrozenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(4, 5, bias=False)
        self.requires_grad_(False)


@pytest.mark.parametrize("trainer", ["cached", "stream"])
def test_kl_trainers_forward_geometry_and_allow_disabled_reinitialization(tmp_path, monkeypatch, trainer):
    import geoae.training.kl_cached as cached
    import geoae.training.kl_stream as stream

    module = cached if trainer == "cached" else stream
    cfg = tiny_config(tmp_path)
    lm = TinyFrozenLM()
    monkeypatch.setattr(module, "load_frozen_lm", lambda *args: lm)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []

    def checked_loss(**kwargs):
        calls.append((kwargs["sep_mode"], kwargs["sep_margin"], kwargs["var_gamma"]))
        return total_loss_e2e(**kwargs)

    def forbidden_reinit(*args, **kwargs):
        raise AssertionError("reinit_every=0 must disable reseeding")

    monkeypatch.setattr(module, "total_loss_e2e", checked_loss)
    monkeypatch.setattr(module, "reinit_dead_clusters", forbidden_reinit)
    if trainer == "cached":
        batch = {"x": torch.arange(16).reshape(4, 4).float() / 10}
        buffer = SimpleNamespace(mean=np.zeros(4, np.float32), std=np.ones(4, np.float32))

        class Buffer:
            mean, std = buffer.mean, buffer.std

            def __len__(self):
                return 4

        monkeypatch.setattr(module, "E2EBuffer", lambda *args, **kwargs: Buffer())
        monkeypatch.setattr(module, "make_loader", lambda *args, **kwargs: [batch])
        cached.train(cfg, use_wandb=False, no_sinkhorn=True)
    else:
        ids = torch.arange(5).reshape(1, 5)
        np.save(tmp_path / "layer_0.npy", np.zeros((5, 4), np.float32))

        class Logits:
            n_layers, is_last = 1, True

            def __init__(self, *args):
                pass

            def teacher_and_residual(self, input_ids):
                h = torch.arange(input_ids.numel() * 4).reshape(-1, 4).float() / 10
                return h, lm.lm_head(h).detach()

            def student(self, input_ids, h):
                return lm.lm_head(h)

        monkeypatch.setattr(module, "StreamLogits", Logits)
        from transformers import AutoTokenizer
        monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args: object())
        monkeypatch.setattr(module, "pull_val_docs", lambda *args: [ids])
        monkeypatch.setattr(module, "open_train_stream", lambda *args: iter([(ids, "fixture")]))
        args = Namespace(skip_leading=0, norm_sample=5, no_sinkhorn=True, n_val_docs=1,
                         max_doc_tokens=5, min_doc_len=1, resume=None, tokens_per_epoch=5,
                         resume_replay_data=False, accum=1, no_renorm=False)
        # Real stats need nonzero std; supply known normalization for this fixture.
        monkeypatch.setattr(module, "compute_norm", lambda *args: (np.zeros(4, np.float32), np.ones(4, np.float32)))
        stream.train(cfg, args)
    assert calls == [("hinge", 1.7, 0.25)]
    saved = torch.load(Path(cfg.train.checkpoints_dir) / "best_val.pt", weights_only=False)
    assert saved["step"] == 1 and saved["config"]["loss"]["no_sinkhorn"]
