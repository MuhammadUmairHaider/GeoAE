"""Unit tests for losses.py. All tests use CPU tensors."""
import math


import torch
from geoae.losses import (
    sinkhorn_log,
    recon_loss,
    cluster_loss,
    sep_loss,
    usage_loss,
    total_loss,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# sinkhorn_log
# ---------------------------------------------------------------------------

def test_sinkhorn_row_sums_to_one():
    B, K = 16, 8
    cost = torch.rand(B, K)
    Q = sinkhorn_log(cost, tau=1.0, n_iter=10)
    assert Q.shape == (B, K)
    row_sums = Q.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), f"row sums: {row_sums}"


def test_sinkhorn_col_sums_to_B_over_K():
    B, K = 16, 8
    cost = torch.rand(B, K)
    Q = sinkhorn_log(cost, tau=1.0, n_iter=10)
    col_sums = Q.sum(dim=0)
    expected = torch.full((K,), B / K)
    assert torch.allclose(col_sums, expected, atol=1e-4), f"col sums: {col_sums}"


def test_sinkhorn_uniform_cost_gives_uniform_Q():
    """Uniform cost → uniform assignment B/K per cell."""
    B, K = 20, 5
    cost = torch.zeros(B, K)
    Q = sinkhorn_log(cost, tau=1.0, n_iter=20)
    expected_cell = B / K / B  # = 1/K per row, B/K per col → cell = 1/K
    assert torch.allclose(Q, torch.full((B, K), 1.0 / K), atol=1e-4)


def test_sinkhorn_all_nonneg():
    B, K = 32, 16
    cost = torch.rand(B, K) * 10
    Q = sinkhorn_log(cost, tau=0.5, n_iter=5)
    assert (Q >= 0).all()


# ---------------------------------------------------------------------------
# recon_loss
# ---------------------------------------------------------------------------

def test_recon_loss_zero_when_identical():
    x = torch.randn(64, 32)
    mse, fve = recon_loss(x, x)
    assert mse.item() < 1e-10
    assert abs(fve.item() - 1.0) < 1e-5


def test_recon_loss_fve_range():
    x = torch.randn(64, 32)
    x_hat = torch.zeros_like(x)   # predicting mean=0, but x has nonzero mean
    mse, fve = recon_loss(x, x_hat)
    assert mse.item() > 0
    # fve can be <= 1; predicting zeros for zero-mean data gives fve ≈ 0
    # (var(x) ≈ MSE since x_hat=0 and x is zero-mean)


def test_recon_loss_worse_than_perfect():
    x = torch.ones(32, 16)
    x_hat_bad = torch.zeros_like(x)
    mse_bad, _ = recon_loss(x, x_hat_bad)
    mse_good, _ = recon_loss(x, x)
    assert mse_bad > mse_good


# ---------------------------------------------------------------------------
# cluster_loss
# ---------------------------------------------------------------------------

def test_cluster_loss_zero_when_z_equals_centroids():
    """If every latent sits exactly on its assigned centroid, loss = 0."""
    B, K, L = 8, 4, 16
    centroids = torch.randn(K, L)
    # Each sample assigned to one centroid exactly
    idx = torch.arange(B) % K
    z = centroids[idx]
    # Build one-hot Q
    Q = torch.zeros(B, K)
    Q.scatter_(1, idx.unsqueeze(1), 1.0)
    loss = cluster_loss(z, centroids.detach(), Q)
    assert loss.item() < 1e-10


def test_cluster_loss_positive_when_z_far_from_centroids():
    B, K, L = 8, 4, 16
    centroids = torch.zeros(K, L)
    z = torch.ones(B, L) * 10
    Q = torch.full((B, K), 1.0 / K)
    loss = cluster_loss(z, centroids.detach(), Q)
    assert loss.item() > 0


def test_cluster_loss_centroids_no_grad():
    """centroids must not accumulate gradients during backward."""
    B, K, L = 8, 4, 16
    centroids_param = torch.randn(K, L, requires_grad=True)
    z = torch.randn(B, L, requires_grad=True)
    Q = torch.full((B, K), 1.0 / K)

    loss = cluster_loss(z, centroids_param.detach(), Q)
    loss.backward()

    # centroids_param was detached before being passed → no grad
    assert centroids_param.grad is None, "centroids must not receive gradients"
    assert z.grad is not None, "z should receive gradients"


# ---------------------------------------------------------------------------
# sep_loss
# ---------------------------------------------------------------------------

def test_sep_loss_positive():
    B, K, L = 32, 8, 16
    z = torch.randn(B, L)
    Q = torch.softmax(torch.randn(B, K), dim=1)
    loss = sep_loss(z, Q)
    assert loss.item() > 0


def test_sep_loss_constant_when_pairs_equidistant():
    """Adaptive σ means equidistant pairs → loss = exp(-1) regardless of scale."""
    # Construct K points equidistant from each other (a regular simplex).
    # Simplest: K=3 points at vertices of an equilateral triangle.
    K = 3
    z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3)/2]])
    Q = torch.eye(K)
    loss = sep_loss(z, Q)
    assert abs(loss.item() - math.exp(-1.0)) < 1e-5


def test_sep_loss_two_points_gives_exp_minus_one():
    """For K=2, only one pair → σ² = its dist² → loss = exp(-1) for any scale."""
    Q = torch.eye(2)
    for scale in (0.1, 1.0, 100.0):
        z = torch.tensor([[0.0, 0.0], [scale, 0.0]])
        loss = sep_loss(z, Q)
        assert abs(loss.item() - math.exp(-1.0)) < 1e-4


def test_sep_loss_gradient_flows_to_z():
    """Gradient must reach z (encoder output)."""
    B, K, L = 16, 4, 8
    z = torch.randn(B, L, requires_grad=True)
    Q = torch.softmax(torch.randn(B, K), dim=1)
    loss = sep_loss(z, Q)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# usage_loss
# ---------------------------------------------------------------------------

def test_usage_loss_zero_when_uniform():
    B, K = 32, 8
    Q = torch.full((B, K), 1.0 / K)
    loss = usage_loss(Q)
    assert loss.item() < 1e-6


def test_usage_loss_positive_when_skewed():
    B, K = 32, 8
    # All mass on cluster 0
    Q = torch.zeros(B, K)
    Q[:, 0] = 1.0
    loss = usage_loss(Q)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# total_loss integration
# ---------------------------------------------------------------------------

def test_total_loss_smoke():
    B, D, L, K = 16, 32, 24, 8
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D)
    z = torch.randn(B, L)
    centroids = torch.randn(K, L)
    Q = sinkhorn_log(torch.cdist(z, centroids).pow(2), tau=1.0)

    out = total_loss(x, x_hat, z, centroids.detach(), Q,
                     lambda_cluster=0.1, lambda_sep=0.01)

    assert "loss" in out
    assert out["loss"].item() > 0
    for key in ("recon", "fve", "cluster", "sep"):
        assert key in out


def test_total_loss_backward():
    B, D, L, K = 16, 32, 24, 8
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D, requires_grad=True)
    z = torch.randn(B, L, requires_grad=True)
    centroids = torch.randn(K, L)
    Q = sinkhorn_log(torch.cdist(z.detach(), centroids).pow(2), tau=1.0)

    out = total_loss(x, x_hat, z, centroids.detach(), Q,
                     lambda_cluster=0.1, lambda_sep=0.01)
    out["loss"].backward()
    assert x_hat.grad is not None
    assert z.grad is not None


# ---------------------------------------------------------------------------
# VICReg terms in the MSE pipeline (mirrors total_loss_e2e)
# ---------------------------------------------------------------------------

def _fixture(seed=0):
    torch.manual_seed(seed)
    B, D, L, K = 16, 32, 24, 8
    x, x_hat = torch.randn(B, D), torch.randn(B, D)
    z = torch.randn(B, L)
    centroids = torch.randn(K, L)
    Q = sinkhorn_log(torch.cdist(z, centroids).pow(2), tau=1.0)
    return x, x_hat, z, centroids.detach(), Q


def test_geometry_lambdas_default_off_reproduce_old_loss():
    """Configs written before the terms existed must be bit-identical."""
    x, x_hat, z, c, Q = _fixture()
    base = total_loss(x, x_hat, z, c, Q, lambda_cluster=0.1, lambda_sep=0.01)
    explicit = total_loss(x, x_hat, z, c, Q, lambda_cluster=0.1, lambda_sep=0.01,
                          lambda_var=0.0, lambda_cov=0.0, lambda_unif=0.0)
    assert torch.equal(base["loss"], explicit["loss"])
    for key in ("var", "cov", "unif"):
        assert key not in base, f"{key} must be absent when its lambda is 0"


def test_var_and_cov_add_to_loss_and_report():
    x, x_hat, z, c, Q = _fixture()
    base = total_loss(x, x_hat, z, c, Q, lambda_cluster=0.1, lambda_sep=0.01)
    geo = total_loss(x, x_hat, z, c, Q, lambda_cluster=0.1, lambda_sep=0.01,
                     lambda_var=0.3, lambda_cov=0.001)

    assert "var" in geo and "cov" in geo
    assert geo["loss"].item() > base["loss"].item(), "geometry terms must add cost"
    expected = (base["loss"] + 0.3 * geo["var"] + 0.001 * geo["cov"])
    assert torch.allclose(geo["loss"], expected, atol=1e-6)
    # the faithfulness/cluster components are untouched
    for key in ("recon", "cluster", "sep"):
        assert torch.equal(base[key], geo[key])


def test_var_term_matches_e2e_definition():
    """MSE and KL runs must share geometry, so the terms must be the same fns."""
    from geoae.e2e.losses import total_loss_e2e

    x, x_hat, z, c, Q = _fixture()
    mse = total_loss(x, x_hat, z, c, Q, lambda_cluster=0.1, lambda_sep=0.01,
                     lambda_var=0.3, lambda_cov=0.001)
    logits_t = torch.randn(z.shape[0], 12)
    kl = total_loss_e2e(teacher_logits=logits_t, student_logits=logits_t.clone(),
                        x=x, x_hat=x_hat, z=z, centroids=c, Q=Q,
                        lambda_cluster=0.1, lambda_sep=0.01,
                        lambda_var=0.3, lambda_cov=0.001)
    assert torch.allclose(mse["var"], kl["var"])
    assert torch.allclose(mse["cov"], kl["cov"])


def test_geometry_terms_reach_the_latent():
    x, x_hat, z, c, Q = _fixture()
    z = z.clone().requires_grad_(True)
    out = total_loss(x, x_hat.clone().requires_grad_(True), z, c, Q,
                     lambda_cluster=0.0, lambda_sep=0.0,
                     lambda_var=0.3, lambda_cov=0.001)
    out["loss"].backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


if __name__ == "__main__":
    # Run all test_ functions manually when pytest isn't available
    import traceback
    passed = failed = 0
    g = dict(globals())
    for name, fn in g.items():
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
