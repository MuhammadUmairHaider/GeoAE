"""Token erasure: scatter recovers planted token directions, the noise correction
zeroes a label-free scatter, erased codebooks assign in the projected space, and
tools that do not apply the projection refuse them. CPU only."""
import numpy as np
import pytest
import torch

from geoae.interp.concept_probe import assign
from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.interp.token_erasure import class_scatter, project_out, r2_from_means, top_eigvecs


def _planted(n=20000, D=16, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    key = torch.randint(T, (n,), generator=g).numpy()
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
    planted = Q[:, :2]                                   # token identity lives here only
    offs = torch.randn(T, 2, generator=g) * 3.0
    Z = torch.randn(n, D, generator=g) + offs[key] @ planted.T
    return Z, key, planted


def test_scatter_recovers_planted_token_subspace():
    Z, key, planted = _planted()
    S_B, cls, means, cnt = class_scatter(Z, key, min_count=30)
    U, ev = top_eigvecs(S_B, 2)
    overlap = float((planted.T @ U).pow(2).sum() / 2)
    assert overlap > 0.99
    assert ev[0] > 10 * float(torch.linalg.eigvalsh(S_B)[-3])   # nothing else carries token signal
    # erasing the 2 directions removes the token signal
    r2_before, _ = r2_from_means(Z, key, cls, means)
    r2_after, _ = r2_from_means(project_out(Z, U), key, cls, project_out(means, U))
    assert r2_before > 0.5 and abs(r2_after) < 0.02


def test_noise_correction_zeroes_label_free_scatter():
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(20000, 8, generator=g)
    key = torch.randint(200, (20000,), generator=g).numpy()   # labels carry nothing
    S_B, *_ = class_scatter(Z, key, min_count=30)
    raw_trace = 199 / 20000 * 8                                # what the uncorrected scatter would hold
    assert abs(float(S_B.trace())) < 0.2 * raw_trace


def test_probe_assigns_in_erased_space():
    C = torch.tensor([[0.0, 1.0], [5.0, -1.0]])
    zero, one = torch.zeros(2), torch.ones(2)
    h = np.array([[5.0, 0.9]], dtype=np.float32)
    U = torch.tensor([[1.0], [0.0]])                           # erase dim 0
    assert assign(h, ("km", (C, zero, one, None)), "cpu")[0] == 1
    assert assign(h, ("km", (C, zero, one, U)), "cpu")[0] == 0
    assert assign(h, ("km", (C, zero, one)), "cpu")[0] == 1    # legacy 3-tuple still works


def test_unaware_tools_refuse_erased_codebook(tmp_path):
    p = tmp_path / "km.npz"
    np.savez(p, centroids=np.zeros((2, 3), np.float32), norm_mean=np.zeros(3, np.float32),
             norm_std=np.ones(3, np.float32), layer=27, erase_U=np.eye(3, 1, dtype=np.float32),
             erase_basis="token", erase_rank=1)
    with pytest.raises(SystemExit, match="does not apply the projection"):
        load_baseline_kmeans(p, torch.device("cpu"))
    C, *_ = load_baseline_kmeans(p, torch.device("cpu"), allow_erasure=True)
    assert C.shape == (2, 3)
