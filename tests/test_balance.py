"""Generalised balancing + phase gating. Every default must reproduce prior behaviour."""
import torch

from geoae.config import Config
from geoae.losses import sinkhorn_log, sinkhorn_log_dual, cluster_log_prior
from geoae.model import GeoAE
from geoae.train_common import (build_model, geometry_schedule, zipf_alpha_schedule,
                                lambda_schedule, reinit_dead_clusters)


# --- sinkhorn_log_dual is a strict superset of sinkhorn_log ------------------

def test_dual_defaults_reproduce_sinkhorn_log():
    torch.manual_seed(0)
    cost = torch.rand(64, 40) * 5
    q_new, _ = sinkhorn_log_dual(cost, tau=1.0, n_iter=3)
    assert torch.allclose(q_new, sinkhorn_log(cost, tau=1.0, n_iter=3), atol=1e-6)


def test_rho_zero_is_plain_softmax():
    torch.manual_seed(1)
    cost = torch.rand(32, 20) * 3
    q, _ = sinkhorn_log_dual(cost, tau=0.7, n_iter=3, rho=0.0)
    assert torch.allclose(q, torch.softmax(-cost / 0.7, dim=1), atol=1e-6)


def test_rows_sum_to_one_under_every_setting():
    torch.manual_seed(2)
    cost = torch.rand(16, 9) * 4
    lp = cluster_log_prior(9, 0.5)
    for rho in (0.0, 0.3, 1.0):
        for eta in (0.05, 1.0):
            q, _ = sinkhorn_log_dual(cost, 1.0, 3, log_prior=lp, rho=rho, eta=eta)
            assert torch.allclose(q.sum(1), torch.ones(16), atol=1e-5)


def test_zipf_prior_alpha_zero_is_uniform_and_normalised():
    lp = cluster_log_prior(50, 0.0)
    assert torch.allclose(lp.exp(), torch.full((50,), 1 / 50), atol=1e-6)
    for a in (0.3, 0.5, 1.0):
        assert abs(cluster_log_prior(50, a).exp().sum().item() - 1.0) < 1e-5


def test_columns_hit_the_zipf_target_at_high_iteration():
    torch.manual_seed(3)
    cost = torch.rand(400, 8) * 2
    lp = cluster_log_prior(8, 0.5)
    q, _ = sinkhorn_log_dual(cost, 1.0, n_iter=200, log_prior=lp, rho=1.0)
    assert torch.allclose(q.sum(0) / 400, lp.exp(), atol=2e-3)


def test_eta_carries_the_dual_across_batches():
    """With eta<1 the marginal is enforced over a run of batches, not within one."""
    torch.manual_seed(4)
    g = None
    for _ in range(60):
        cost = torch.rand(24, 10) * 3
        q, g = sinkhorn_log_dual(cost, 1.0, 3, rho=1.0, eta=0.05, g=g)
    assert g.abs().sum() > 0                       # the dual actually accumulated
    assert torch.allclose(q.sum(1), torch.ones(24), atol=1e-5)


# --- model plumbing ---------------------------------------------------------

def test_model_defaults_take_the_legacy_path():
    m = GeoAE(hidden_size=6, latent_dim=4, n_clusters=5)
    assert m._default_balancing()
    assert torch.allclose(m.sinkhorn_g, torch.zeros(5))


def test_zipf_model_forward_and_target_usage():
    m = GeoAE(hidden_size=6, latent_dim=4, n_clusters=5,
              balance="zipf", zipf_alpha=0.5, balance_rho=0.6, balance_eta=0.05)
    assert not m._default_balancing()
    out = m(torch.randn(12, 6))
    assert torch.allclose(out.Q.sum(1), torch.ones(12), atol=1e-5)
    u = m.target_usage()
    assert abs(u.sum().item() - 1.0) < 1e-5 and u.max() > u.min()


def test_sinkhorn_g_updates_in_train_mode_only():
    m = GeoAE(hidden_size=6, latent_dim=4, n_clusters=5,
              balance="zipf", zipf_alpha=0.5, balance_rho=1.0, balance_eta=0.1)
    m.eval(); m(torch.randn(10, 6))
    assert torch.allclose(m.sinkhorn_g, torch.zeros(5))
    m.train(); m(torch.randn(10, 6))
    assert m.sinkhorn_g.abs().sum() > 0


def test_uniform_target_usage_is_flat():
    m = GeoAE(hidden_size=4, latent_dim=3, n_clusters=7)
    assert torch.allclose(m.target_usage(), torch.full((7,), 1 / 7), atol=1e-6)


# --- schedules --------------------------------------------------------------

def test_geometry_schedule_defaults_to_always_on():
    cfg = Config(); cfg.loss.lambda_var, cfg.loss.lambda_cov = 0.3, 0.001
    assert cfg.train.geometry_start_epoch == 0
    assert geometry_schedule(1, cfg) == (0.3, 0.001, 0.0)


def test_geometry_schedule_gates_before_start_epoch():
    cfg = Config(); cfg.loss.lambda_var, cfg.loss.lambda_cov = 0.3, 0.001
    cfg.train.geometry_start_epoch = 6
    assert geometry_schedule(5, cfg) == (0.0, 0.0, 0.0)
    assert geometry_schedule(6, cfg) == (0.3, 0.001, 0.0)


def test_cluster_and_geometry_gates_are_independent():
    """epochs 1-5 recon, 6-10 recon+vicreg, 11+ everything."""
    cfg = Config()
    cfg.loss.lambda_cluster, cfg.loss.lambda_var = 0.05, 0.3
    cfg.train.recon_only_epochs = 10
    cfg.train.geometry_start_epoch = 6
    cfg.train.full_loss_start_epoch = 11
    assert lambda_schedule(3, cfg)[0] == 0.0 and geometry_schedule(3, cfg)[0] == 0.0
    assert lambda_schedule(8, cfg)[0] == 0.0 and geometry_schedule(8, cfg)[0] == 0.3
    assert lambda_schedule(11, cfg)[0] == 0.05 and geometry_schedule(11, cfg)[0] == 0.3


def test_zipf_alpha_schedule_zero_when_uniform():
    cfg = Config()
    assert all(zipf_alpha_schedule(e, cfg) == 0.0 for e in (1, 10, 50))


def test_zipf_alpha_anneals_from_zero():
    cfg = Config(); cfg.loss.balance = "zipf"; cfg.loss.zipf_alpha_end = 0.5
    cfg.train.n_epochs, cfg.train.full_loss_start_epoch = 50, 11
    assert zipf_alpha_schedule(10, cfg) == 0.0
    assert zipf_alpha_schedule(11, cfg) == 0.0
    assert abs(zipf_alpha_schedule(50, cfg) - 0.5) < 1e-9
    seq = [zipf_alpha_schedule(e, cfg) for e in range(11, 51)]
    assert all(b >= a for a, b in zip(seq, seq[1:]))


def test_build_model_passes_balance_fields():
    cfg = Config()
    cfg.loss.balance, cfg.loss.balance_rho = "zipf", 0.6
    cfg.loss.balance_eta, cfg.loss.zipf_alpha_start = 0.05, 0.0
    m = build_model(cfg, torch.device("cpu"))
    assert (m.balance, m.balance_rho, m.balance_eta) == ("zipf", 0.6, 0.05)


# --- reinit threshold is relative to the target, not a flat 1/K -------------

def test_reinit_threshold_matches_legacy_under_uniform():
    m = GeoAE(hidden_size=4, latent_dim=3, n_clusters=8)
    with torch.no_grad():
        m.ema_cluster_size.copy_(torch.tensor([1., 1., 1., 1., 1., 1., 1., 1e-6]))
    z, x = torch.randn(5, 3), torch.randn(5, 4)
    assert reinit_dead_clusters(m, z, x, x + 0.1, threshold_factor=0.1) == 1


def test_reinit_respects_a_zipf_target():
    """A tail cluster at its Zipf share is alive; the same share under uniform is not."""
    m = GeoAE(hidden_size=4, latent_dim=3, n_clusters=8,
              balance="zipf", zipf_alpha=1.0, balance_rho=1.0)
    tgt = m.target_usage()
    with torch.no_grad():
        m.ema_cluster_size.copy_(tgt * 1000.0)     # exactly on target everywhere
    z, x = torch.randn(5, 3), torch.randn(5, 4)
    assert reinit_dead_clusters(m, z, x, x + 0.1, threshold_factor=0.1) == 0
