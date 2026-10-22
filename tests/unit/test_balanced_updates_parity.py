"""Exact CPU parity with the two balanced-fit loops before helper extraction."""
import sys

import numpy as np
import pytest
import torch

from geoae.baselines._balanced import balanced_updates
import geoae.baselines.balanced_kmeans as cli
from geoae.baselines.sae_controls import balanced_fit
from geoae.losses import sinkhorn_log
from geoae.training.initialization import density_peaks_select


@torch.no_grad()
def legacy_updates(X, C, *, epochs, batch_size, tau_start=1., tau_end=.1, sinkhorn_iters=3,
                   ema_decay=.9, reinit_every=1000, reinit="peaks", density_power=1., density_knn=2,
                   peak_min_sep_frac=.25, peak_refine_k=0, peak_reinit_pool=8):
    """Frozen pre-refactor update order, including both original reseed branches."""
    K = len(C)
    ema_size = torch.zeros(K, device=X.device)
    ema_sum = C.clone()
    steps_per_epoch = len(X) // batch_size
    total = epochs * steps_per_epoch
    step = n_reinit = 0
    history = []
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=X.device)
        for s in range(steps_per_epoch):
            b = X[perm[s * batch_size:(s + 1) * batch_size]]
            tau = tau_start + (tau_end - tau_start) * (step / max(total - 1, 1))
            Q = sinkhorn_log(torch.cdist(b, C) ** 2, tau, sinkhorn_iters)
            hard = torch.nn.functional.one_hot(Q.argmax(1), K).to(b.dtype)
            cnt = hard.sum(0)
            ema_size.mul_(ema_decay).add_(cnt * (1 - ema_decay))
            ema_sum.mul_(ema_decay).add_((hard.T @ b) * (1 - ema_decay))
            visited = cnt > 0
            C[visited] = ema_sum[visited] / ema_size[visited].unsqueeze(1).clamp_min(1e-6)
            if reinit_every and step and step % reinit_every == 0:
                dead = ema_size < (0.1 * ema_size.mean())
                if int(dead.sum()):
                    if reinit == "peaks":
                        p = b[:peak_reinit_pool]
                        new, _ = density_peaks_select(
                            p, min(int(dead.sum()), len(p)), C_init=C[~dead], seed=0,
                            density_power=density_power, min_sep_frac=peak_min_sep_frac,
                            knn_k=density_knn, refine_k=peak_refine_k)
                        dead = dead.nonzero(as_tuple=True)[0][:len(new)]
                    else:
                        new = b[torch.cdist(b, C).min(1).values.topk(int(dead.sum())).indices]
                    n_reinit += len(new)
                    C[dead] = new
                    ema_size[dead] = ema_size.mean()
                    ema_sum[dead] = new * ema_size.mean()
            step += 1
        history.append((ep + 1, C.clone(), tau, n_reinit))
    return C, history


@pytest.mark.parametrize("reinit,reinit_every", [("peaks", 2), ("farthest", 2), ("peaks", 0)])
def test_updates_match_old_loop_and_rng_even_with_dead_clusters(reinit, reinit_every):
    X = torch.zeros(33, 2)  # One unconsumed tail row; all hard labels pick one centroid.
    initial = torch.tensor([[0., 0.], [5., 5.], [10., 10.]])
    settings = dict(epochs=2, batch_size=8, reinit=reinit, reinit_every=reinit_every,
                    density_knn=2, peak_refine_k=0, peak_reinit_pool=8)
    torch.manual_seed(17)
    expected, expected_history = legacy_updates(X, initial.clone(), **settings)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(17)
    history, progress = [], []

    def report(epoch, C, tau, n_reinit):
        history.append((epoch, C.clone(), tau, n_reinit))

    def steps(sequence, epoch):
        progress.append((epoch, list(sequence)))
        return sequence

    actual = balanced_updates(X, initial.clone(), **settings, on_epoch=report, progress=steps)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert progress == [(1, [0, 1, 2, 3]), (2, [0, 1, 2, 3])]
    for observed, old in zip(history, expected_history):
        assert observed[0] == old[0] and observed[2:] == old[2:]
        assert torch.equal(observed[1], old[1])
    if reinit_every:
        assert history[-1][-1] > 0, "the fixture must exercise reseeding"
    else:
        assert (actual == initial).all(1).sum() >= 2, "unvisited centroids must stay fixed"


def test_sae_wrapper_preserves_dpc_initialization_updates_and_log():
    X = torch.zeros(32, 2)
    seed, K = 19, 3
    torch.manual_seed(seed)
    pool = X[torch.randperm(len(X))[:16]]
    C, _ = density_peaks_select(pool, K, seed=seed, knn_k=2, refine_k=0)
    expected, history = legacy_updates(X, C.clone(), epochs=2, batch_size=8, reinit_every=2)
    expected_rng = torch.get_rng_state()
    logs = []
    actual = balanced_fit(X, K, seed=seed, epochs=2, batch_size=8, density_pool=16,
                          density_knn=2, peak_refine_k=0, peak_reinit_pool=8, reinit_every=2,
                          log=logs.append)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert logs == [f"  epoch {ep}/2: tau {tau:.3f} reinit so far {n}" for ep, _, tau, n in history]


def test_cli_preserves_kmeanspp_initialization_and_saved_codebook(tmp_path, monkeypatch, capsys):
    raw = np.random.RandomState(3).normal(size=(32, 3)).astype(np.float32)
    acts, norm, out = (tmp_path / name for name in ("acts.npy", "norm.npz", "codebook.npz"))
    np.save(acts, raw)
    np.savez(norm, mean=np.zeros(3, np.float32), std=np.ones(3, np.float32))
    seed, K = 17, 3
    idx = np.sort(np.random.RandomState(seed).choice(len(raw), 25, replace=False))
    X = torch.from_numpy(raw[idx])
    torch.manual_seed(seed)
    C = cli.kmeanspp_init(X[torch.randperm(len(X))[:20]], K, seed)
    expected, _ = legacy_updates(X, C, epochs=3, batch_size=6, reinit_every=0, reinit="farthest")
    expected_rng = torch.get_rng_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "balanced-fit", "--norm_params", str(norm), "--model_name", "fixture", "--layer", "0",
        "--activations", str(acts), "--out", str(out), "--n_clusters", str(K),
        "--n_sample", "25", "--init_sample", "20", "--batch_size", "6", "--epochs", "3",
        "--reinit_every", "0", "--seed", str(seed),
    ])
    cli.main()
    with np.load(out) as data:
        np.testing.assert_array_equal(data["centroids"], expected.numpy())
        assert data["init"] == "kmeans++" and data["reinit"] == "farthest"
        assert data["space"] == "raw" and data["model_name"] == "fixture"
    assert torch.equal(torch.get_rng_state(), expected_rng)
    stdout = capsys.readouterr().out
    assert "epoch 1:" in stdout and "epoch 3:" in stdout and "Final on a 25-token probe" in stdout
