"""Resume semantics for the cached e2e trainer (CPU, no LM, no activations).

Guards three things a naive resume gets wrong:
  * the epoch counter continues instead of restarting at 1;
  * the global step counter continues (it names step_<n>.pt and is the wandb
    x-axis, which rejects steps that move backwards);
  * best_val.pt is not overwritten by a worse post-resume epoch — the tracker
    starts at +inf, so without seeding it the first epoch always "wins".
"""
import pytest
import torch

from geoae.config import Config
from geoae.e2e.train import load_resume, save_checkpoint
from geoae.train_common import CheckpointTracker, build_model, make_optimizer

D, K = 16, 8


def make_cfg():
    cfg = Config()
    cfg.model.hidden_size = D
    cfg.model.latent_dim = D
    cfg.model.n_clusters = K
    cfg.model.nonlinearity = "gelu"
    cfg.train.keep_checkpoints = 3
    return cfg


def write_ckpt(path, cfg, epoch, step, val_kl, mean, std):
    model = build_model(cfg, torch.device("cpu"), no_sinkhorn=True)
    model.centroids_initialized.fill_(1)          # as if k-means++ already ran
    opt = make_optimizer(model, cfg)
    # take one step so the optimizer has non-empty state to restore
    model(torch.randn(4, D)).x_hat.sum().backward()
    opt.step()
    save_checkpoint(path, model, opt, epoch, step, 0.802,
                    mean.numpy(), std.numpy(), val_kl, 0.0, cfg)


@pytest.fixture
def scratch(tmp_path):
    cfg = make_cfg()
    mean, std = torch.zeros(D), torch.ones(D)
    ckpt = tmp_path / "best_val.pt"
    write_ckpt(ckpt, cfg, epoch=18, step=328158, val_kl=0.04581, mean=mean, std=std)
    return cfg, ckpt, mean, std


def test_resume_continues_epoch_and_step(scratch):
    cfg, ckpt, mean, std = scratch
    model = build_model(cfg, torch.device("cpu"), no_sinkhorn=True)
    opt = make_optimizer(model, cfg)
    assert not bool(model.centroids_initialized.item())

    start_epoch, step = load_resume(ckpt, model, opt, mean, std, torch.device("cpu"))

    assert start_epoch == 19, "must continue at epoch+1, not restart at 1"
    assert step == 328158, "global step must continue (names checkpoints, wandb x-axis)"
    assert bool(model.centroids_initialized.item()), "centroids must survive, else k-means++ reruns"
    assert len(opt.state) > 0, "Adam moments must be restored"


def test_resume_refuses_mismatched_norm_stats(scratch):
    """The encoder was trained against these exact stats — resuming under
    different ones would corrupt the model with no visible error."""
    cfg, ckpt, mean, std = scratch
    model = build_model(cfg, torch.device("cpu"), no_sinkhorn=True)
    opt = make_optimizer(model, cfg)

    with pytest.raises(ValueError, match="does not match"):
        load_resume(ckpt, model, opt, mean + 5.0, std, torch.device("cpu"))


# --------------------------------------------------------------------------- #
# Streaming trainer: the persistent doc stream must be skipped forward on resume
# --------------------------------------------------------------------------- #

def fake_stream(n_docs, doc_len=40, start=0):
    """Mimics geoae.extract.stream_docs: yields (ids (1, T), domain)."""
    for i in range(start, start + n_docs):
        yield torch.full((1, doc_len), float(i)).long(), "web"


def test_fast_forward_skips_consumed_tokens():
    """A run resumed at epoch N must not refeed the tokens epochs 1..N-1 used."""
    from geoae.e2e.train_stream import fast_forward_stream

    gen = fake_stream(100, doc_len=40)
    skipped = fast_forward_stream(gen, n_tokens=400)     # 10 docs of 40

    assert skipped == 400
    # The next doc must be NEW, not a replay of the skipped prefix.
    nxt, _ = next(gen)
    assert int(nxt[0, 0].item()) == 10


def test_fast_forward_stops_at_exhaustion():
    """A short corpus must not hang or raise — it warns and returns what it got."""
    from geoae.e2e.train_stream import fast_forward_stream

    gen = fake_stream(3, doc_len=40)                     # only 120 tokens exist
    skipped = fast_forward_stream(gen, n_tokens=10_000)

    assert skipped == 120
    assert next(gen, None) is None


def test_fast_forward_is_skipped_for_fresh_runs():
    """start_epoch == 1 means nothing was consumed; the stream must be untouched."""
    from geoae.e2e.train_stream import fast_forward_stream

    gen = fake_stream(5, doc_len=40)
    assert fast_forward_stream(gen, n_tokens=0) == 0
    nxt, _ = next(gen)
    assert int(nxt[0, 0].item()) == 0


def test_best_val_not_clobbered_by_worse_epoch(scratch, tmp_path):
    cfg, ckpt, mean, std = scratch
    tracker = CheckpointTracker(tmp_path, cfg.train.keep_checkpoints)

    # Unseeded, ANY score beats +inf — this is the bug.
    assert tracker.is_better(0.9), "sanity: fresh tracker accepts anything"

    prev = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    tracker.record_best(float(prev["val_kl"]))

    assert not tracker.is_better(0.0470), "a worse epoch must NOT replace best_val.pt"
    assert tracker.is_better(0.0450), "a better epoch must still replace it"
