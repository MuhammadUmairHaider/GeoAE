"""Unit tests for the geometry regularisers (variance/covariance/uniformity)
and the hard-EMA centroid update. All tests use CPU tensors."""
import math

import torch

from geoae.losses import variance_loss, covariance_loss, uniformity_loss, l2_normalize
from geoae.model import GeoAE

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# variance_loss
# ---------------------------------------------------------------------------

def test_variance_loss_zero_when_spread():
    z = torch.randn(512, 16) * 3.0          # std ~3 per dim, hinge inactive
    assert variance_loss(z).item() < 1e-6


def test_variance_loss_charges_contraction():
    z = torch.randn(512, 16)
    small = variance_loss(z * 0.1)           # std ~0.1 → hinge ~0.9
    assert small.item() > 0.8
    assert small.item() > variance_loss(z).item()


def test_variance_loss_gradient_flows_to_z():
    z = (torch.randn(64, 8) * 0.1).requires_grad_(True)
    variance_loss(z).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# covariance_loss
# ---------------------------------------------------------------------------

def test_covariance_loss_near_zero_for_independent_dims():
    z = torch.randn(20000, 8)
    assert covariance_loss(z).item() < 0.01


def test_covariance_loss_positive_for_correlated_dims():
    base = torch.randn(1024, 1)
    z = base.repeat(1, 8) + 0.01 * torch.randn(1024, 8)   # rank-1 latents
    assert covariance_loss(z).item() > 0.5


def test_covariance_loss_prefers_decorrelated():
    base = torch.randn(1024, 4)
    correlated = base @ torch.ones(4, 4)
    assert covariance_loss(correlated).item() > covariance_loss(base).item()


# ---------------------------------------------------------------------------
# uniformity_loss
# ---------------------------------------------------------------------------

def test_uniformity_loss_lower_for_spread_directions():
    clumped = torch.randn(256, 16) * 0.01 + torch.ones(16)   # one direction
    spread = torch.randn(256, 16)                             # ~uniform on sphere
    assert uniformity_loss(spread).item() < uniformity_loss(clumped).item()


def test_uniformity_loss_scale_invariant():
    z = torch.randn(128, 16)
    a, b = uniformity_loss(z).item(), uniformity_loss(z * 7.3).item()
    assert math.isclose(a, b, rel_tol=1e-5)


def test_uniformity_loss_subsamples_large_batch():
    z = torch.randn(5000, 8)
    val = uniformity_loss(z, max_samples=256)
    assert torch.isfinite(val)


def test_uniformity_loss_gradient_flows_to_z():
    z = torch.randn(64, 8, requires_grad=True)
    uniformity_loss(z).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# hard-EMA centroid update
# ---------------------------------------------------------------------------

def _tiny_model(ema_hard, metric="euclidean", K=4, L=8, decay=0.5):
    m = GeoAE(hidden_size=8, latent_dim=L, n_clusters=K, ema_decay=decay,
              nonlinearity="gelu", metric=metric, ema_hard=ema_hard)
    m.centroids.copy_(torch.randn(K, L))
    m.centroids_initialized.fill_(True)
    return m


def test_hard_ema_uses_argmax_assignment():
    m = _tiny_model(ema_hard=True)
    z = torch.randn(16, 8)
    Q = torch.full((16, 4), 0.25)
    Q[:, 0] = 0.4                              # argmax → all rows in cluster 0
    before = m.centroids.clone()
    m.update_centroids_ema(z, Q)
    # only cluster 0 moves; others untouched (no mass under hard assignment)
    assert not torch.allclose(m.centroids[0], before[0])
    assert torch.allclose(m.centroids[1:], before[1:])
    expected0 = 0.5 * before[0] + 0.5 * z.mean(0)
    assert torch.allclose(m.centroids[0], expected0, atol=1e-5)


def test_hard_ema_unvisited_centroids_do_not_decay():
    m = _tiny_model(ema_hard=True)
    z = torch.randn(16, 8)
    dist2 = torch.cdist(z, m.centroids).pow(2)
    Q = torch.softmax(-dist2, dim=1)
    before = m.centroids.clone()
    for _ in range(50):
        m.update_centroids_ema(z, Q)
    visited = torch.zeros(4, dtype=torch.bool)
    visited[Q.argmax(1).unique()] = True
    # unvisited centroids keep their exact position instead of shrinking to 0
    assert torch.allclose(m.centroids[~visited], before[~visited])
    assert m.centroids[visited].norm(dim=1).min() > 1e-3


def test_soft_ema_unchanged_by_flag_default():
    zs, Qs = torch.randn(16, 8), torch.softmax(torch.randn(16, 4), dim=1)
    m_soft, m_ref = _tiny_model(ema_hard=False), _tiny_model(ema_hard=False)
    m_ref.centroids.copy_(m_soft.centroids)
    m_ref.ema_cluster_size.copy_(m_soft.ema_cluster_size)
    m_soft.update_centroids_ema(zs, Qs)
    # reference implementation of the original soft update
    cs, cc = Qs.T @ zs, Qs.sum(0)
    expected = 0.5 * m_ref.centroids + 0.5 * (cs / cc.unsqueeze(1).clamp(min=1e-6))
    assert torch.allclose(m_soft.centroids, expected, atol=1e-5)


def test_hard_ema_cosine_renorms():
    m = _tiny_model(ema_hard=True, metric="cosine")
    m.centroids.copy_(l2_normalize(m.centroids))
    z = torch.randn(16, 8)
    dist2 = torch.cdist(l2_normalize(z), m.centroids).pow(2)
    m.update_centroids_ema(z, torch.softmax(-dist2, dim=1))
    norms = m.centroids.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)
