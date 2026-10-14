"""Unit tests for the density-peaks selector in geoae.seeded_init — CPU only."""
import pytest
import torch

from geoae.model import GeoAE
from geoae.seeded_init import (
    density_peaks_select,
    local_density,
    nearest_higher_density,
)
from geoae.train_common import reinit_dead_clusters


def brute_force_delta(z: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Reference delta, matching the rank-order tie-break used by the real one."""
    order = torch.argsort(rho, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    d = torch.cdist(z, z)
    delta = torch.empty(len(z))
    for i in range(len(z)):
        denser = (rank < rank[i]).nonzero(as_tuple=True)[0]
        delta[i] = d[i, denser].min() if len(denser) else float("nan")
    top = order[0]
    delta[top] = delta[torch.arange(len(z)) != top].max()
    return delta


def blobs_with_outliers(seed=0, per_blob=120, n_out=25, dim=6):
    """Three tight Gaussian blobs plus scattered uniform outliers."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[-8.0], [0.0], [8.0]]).repeat(1, dim)
    blobs = torch.cat([c + 0.35 * torch.randn(per_blob, dim, generator=g) for c in centres])
    out = (torch.rand(n_out, dim, generator=g) - 0.5) * 90.0
    return blobs, out, centres


# ---------------------------------------------------------------------------
# delta
# ---------------------------------------------------------------------------

def test_nearest_higher_density_matches_brute_force():
    torch.manual_seed(0)
    z = torch.randn(200, 5)
    rho = local_density(z, n_ref=200, k=8, seed=0)
    for chunk in (17, 64, 512):        # smaller than, and larger than, N
        got = nearest_higher_density(z, rho, chunk=chunk)
        assert torch.allclose(got, brute_force_delta(z, rho), atol=1e-4)


def test_nearest_higher_density_handles_exact_ties():
    # Two points at identical density must not both count as "not higher" than
    # the other; rank order breaks the tie so exactly one gets the large delta.
    z = torch.tensor([[0.0], [0.1], [5.0], [5.1], [50.0]])
    rho = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.01])   # four-way tie
    delta = nearest_higher_density(z, rho, chunk=2)
    assert torch.isfinite(delta).all()
    # The tied pair (0.0, 0.1): whichever ranks second sits 0.1 from the first.
    assert delta.min() == pytest.approx(0.1, abs=1e-5)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def test_peaks_find_modes_where_kmeanspp_finds_outliers():
    """The falsifiable claim: peaks land on blobs, k-means++ lands on outliers."""
    blobs, out, centres = blobs_with_outliers()
    z = torch.cat([blobs, out])
    n_blob = len(blobs)

    picks, diag = density_peaks_select(z, 3, seed=0, refine_k=8, min_sep_frac=0.25)
    assert picks.shape == (3, z.shape[1])

    # One pick per blob, none on an outlier.
    nearest_centre = torch.cdist(picks, centres).argmin(1)
    assert sorted(nearest_centre.tolist()) == [0, 1, 2]
    assert torch.cdist(picks, centres).min(1).values.max() < 1.0

    # Picks sit in the dense half of the pool; that is the whole point.
    assert diag["density_pct"] > 50.0
    assert diag["n_peak"] == 3
    assert diag["n_forced"] == 0

    # k-means++ on the same data grabs at least one outlier.
    model = GeoAE(hidden_size=4, latent_dim=z.shape[1], n_clusters=3)
    model.init_centroids_kmeans_plus_plus(z, seed=0)
    picked_rows = torch.cdist(model.centroids, z).argmin(1)
    assert (picked_rows >= n_blob).any()


def test_peaks_beat_coverage_fill_on_density():
    from geoae.seeded_init import density_coverage_fill

    blobs, out, _ = blobs_with_outliers()
    z = torch.cat([blobs, out])
    dens = local_density(z, n_ref=len(z), k=16, seed=0)

    peaks, diag = density_peaks_select(z, 8, dens=dens, seed=0, refine_k=0)
    cover = density_coverage_fill(z, None, 8, dens=dens.clone(), seed=0, density_power=1.0)

    def mean_density(picks):
        rows = torch.cdist(picks, z).argmin(1)
        return dens[rows].mean()

    assert mean_density(peaks) > mean_density(cover)
    assert diag["density_pct"] > 50.0


def test_returns_exactly_n_needed_when_radius_exhausts_pool():
    torch.manual_seed(0)
    z = torch.randn(150, 4)
    # A radius far larger than the data means the greedy pass accepts one point
    # and then rejects everything; the top-up must still deliver n_needed.
    picks, diag = density_peaks_select(z, 20, seed=0, min_sep_frac=50.0, refine_k=0)
    assert picks.shape == (20, 4)
    assert diag["n_forced"] > 0
    assert len(torch.unique(picks, dim=0)) == 20      # no duplicated rows


def test_c_init_suppresses_nearby_picks():
    blobs, out, centres = blobs_with_outliers()
    z = torch.cat([blobs, out])
    # Anchor the first two blob centres; peaks must avoid them.
    anchors = centres[:2]
    picks, _ = density_peaks_select(z, 4, C_init=anchors, seed=0,
                                    min_sep_frac=0.25, refine_k=0)
    from geoae.seeded_init import _neighbourhood_scale
    radius = 0.25 * _neighbourhood_scale(local_density(z, n_ref=len(z), k=32, seed=0))
    assert torch.cdist(picks, anchors).min() > radius


def test_refine_places_centroid_off_the_data_points():
    blobs, out, _ = blobs_with_outliers()
    z = torch.cat([blobs, out])
    exact, _ = density_peaks_select(z, 3, seed=0, refine_k=0)
    refined, _ = density_peaks_select(z, 3, seed=0, refine_k=8)
    # refine_k=0 returns literal rows of z; refine_k>0 returns neighbourhood
    # means. Tolerance is loose because cdist goes through a matmul in float32.
    assert torch.cdist(exact, z).min(1).values.max() < 1e-3
    assert torch.cdist(refined, z).min(1).values.max() > 1e-2


# ---------------------------------------------------------------------------
# reinit
# ---------------------------------------------------------------------------

def test_reinit_peaks_mode_revives_dead_clusters():
    torch.manual_seed(0)
    K, L, D, B = 8, 6, 12, 200
    model = GeoAE(hidden_size=D, latent_dim=L, n_clusters=K)
    model.init_centroids_from_data(torch.randn(K * 2, L))
    model.ema_cluster_size.fill_(1.0)
    model.ema_cluster_size[:3] = 1e-8

    blobs, out, _ = blobs_with_outliers(per_blob=60, n_out=20, dim=L)
    z = torch.cat([blobs, out])[:B]
    x, x_hat = torch.randn(len(z), D), torch.randn(len(z), D)

    before = model.centroids[:3].clone()
    n = reinit_dead_clusters(model, z, x, x_hat, mode="peaks", peak_pool=B)
    assert n == 3
    assert (model.ema_cluster_size[:3] == 1.0).all()
    assert not torch.allclose(model.centroids[:3], before)


def test_reinit_loss_mode_unchanged():
    """The default path must be untouched by the peaks branch."""
    torch.manual_seed(0)
    K, L, D, B = 8, 16, 32, 64
    model = GeoAE(hidden_size=D, latent_dim=L, n_clusters=K)
    model.init_centroids_from_data(torch.randn(K * 2, L))
    model.ema_cluster_size.fill_(1.0)
    model.ema_cluster_size[:4] = 1e-8

    z, x, x_hat = torch.randn(B, L), torch.randn(B, D), torch.randn(B, D)
    expected = z[(x_hat - x).pow(2).mean(dim=1).topk(4).indices]
    assert reinit_dead_clusters(model, z, x, x_hat) == 4
    assert torch.allclose(model.centroids[:4], expected)
