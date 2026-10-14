"""Unit tests for model.py — CPU only."""


import torch
from geoae.model import GeoAE

torch.manual_seed(42)

B, D, L, K = 32, 64, 48, 16  # small dims for fast CPU tests


def make_model(**kwargs) -> GeoAE:
    defaults = dict(hidden_size=D, latent_dim=L, n_clusters=K, sinkhorn_iters=3, tau=1.0)
    defaults.update(kwargs)
    return GeoAE(**defaults)


def test_forward_shapes():
    model = make_model()
    x = torch.randn(B, D)
    out = model(x)
    assert out.x_hat.shape == (B, D)
    assert out.z.shape == (B, L)
    assert out.Q.shape == (B, K)
    assert out.dist2.shape == (B, K)


def test_Q_row_sums_approx_one():
    model = make_model()
    x = torch.randn(B, D)
    out = model(x)
    row_sums = out.Q.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), f"row sums: {row_sums}"


def test_decoder_columns_unit_norm():
    model = make_model()
    col_norms = model.decoder.weight.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones(L), atol=1e-6), f"col norms: {col_norms}"


def test_post_step_renorms_decoder():
    model = make_model()
    # Corrupt decoder weights
    with torch.no_grad():
        model.decoder.weight.mul_(5.0)
    model.post_step()
    col_norms = model.decoder.weight.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones(L), atol=1e-6)


def test_init_centroids_from_data():
    model = make_model()
    assert not model.centroids_initialized.item()
    z_sample = torch.randn(K * 4, L)
    model.init_centroids_from_data(z_sample)
    assert model.centroids_initialized.item()
    assert model.centroids.shape == (K, L)


def test_centroids_no_grad_after_backward():
    """Centroids must not accumulate gradients during a training step."""
    model = make_model()
    x = torch.randn(B, D)
    out = model(x)

    # Simulate a loss that touches z and x_hat
    loss = out.x_hat.pow(2).mean() + out.z.pow(2).mean()
    loss.backward()

    # centroids is a buffer → grad must be None
    assert model.centroids.grad is None, "centroids must not receive gradients"


def test_ema_update_changes_centroids():
    model = make_model()
    model.init_centroids_from_data(torch.randn(K * 2, L))
    before = model.centroids.clone()

    x = torch.randn(B, D)
    out = model(x)
    model.update_centroids_ema(out.z.detach(), out.Q.detach())

    assert not torch.allclose(model.centroids, before), "EMA should have changed centroids"


def test_no_decoder_bias():
    model = make_model()
    assert model.decoder.bias is None


def test_full_forward_backward():
    """Smoke test: forward → loss → backward → step → post_step."""
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(B, D)
    out = model(x)
    loss = (out.x_hat - x).pow(2).mean()
    loss.backward()
    opt.step()
    model.post_step()

    # Decoder columns still unit norm after an optimizer step + renorm
    col_norms = model.decoder.weight.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones(L), atol=1e-6)


if __name__ == "__main__":
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
