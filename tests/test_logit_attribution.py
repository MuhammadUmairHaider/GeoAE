"""GPT-2 LayerNorm's hidden-dimension centering is part of its logit lens."""
import pytest
import torch

from geoae.interp.logit_attribution import compute_dla
from geoae.model import GeoAE


@pytest.mark.parametrize("center,apply", [(True, True), (False, True), (True, False)])
def test_norm_centering_precedes_gain_and_follows_raw_rescaling(center, apply):
    ae = GeoAE(hidden_size=3, latent_dim=3, n_clusters=2)
    ae.decoder.weight.data.copy_(torch.eye(3))
    ae.centroids.copy_(torch.tensor([[1., 2., 4.], [2., 3., 6.]]))
    std = torch.tensor([2., 1., 0.5])
    gain = torch.tensor([1., 2., 3.])
    W_U = torch.tensor([[1., 2., 3.], [3., -1., 2.], [0., 1., -1.], [1., 1., 1.]])

    raw = ae.centroids * std
    if apply:
        if center:
            raw = raw - raw.mean(-1, keepdim=True)
        raw = raw * gain
    expected = raw @ W_U.T
    expected -= expected.mean(-1, keepdim=True)
    got = compute_dla(ae, std, W_U, gain, apply, center_final_norm=center)
    torch.testing.assert_close(got, expected)
    if center and apply:
        # The first raw direction is constant; LayerNorm removes it entirely.
        assert torch.count_nonzero(got[0]) == 0
