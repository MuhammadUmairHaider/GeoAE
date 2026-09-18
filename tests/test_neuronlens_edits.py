"""The shared range-edit operators must reproduce the existing NeuronLens hooks exactly. CPU only."""
import numpy as np
import torch

from geoae.interp import neuronlens as nl
from geoae.model import GeoAE

DEV = torch.device("cpu")


def _hs(B=2, T=5, D=8, seed=0):
    return torch.randn(B, T, D, generator=torch.Generator().manual_seed(seed))


def test_global_shift_equals_make_h_steer():
    hs, r = _hs(), np.random.RandomState(1).randn(8).astype(np.float32)
    inf = np.full(8, np.inf, dtype=np.float32)
    got = nl.make_h_edit(nl.shift_edit(r, -inf, inf, 1.5, DEV), DEV)(hs)
    assert torch.equal(got, nl.make_h_steer(r, 1.5, DEV)(hs))


def test_gate_edit_equals_make_h_gate():
    hs = _hs()
    lo = np.full(8, np.inf, dtype=np.float32); hi = np.full(8, -np.inf, dtype=np.float32)
    lo[:3], hi[:3] = -0.5, 0.5                       # three salient dims, a real range
    rep = np.random.RandomState(2).randn(8).astype(np.float32)
    got = nl.make_h_edit(nl.gate_edit(lo, hi, rep, DEV), DEV)(hs)
    assert torch.equal(got, nl.make_h_gate(lo, hi, rep, DEV)(hs))
    assert torch.equal(got[..., 3:], hs[..., 3:])    # non-salient dims untouched


def test_z_edit_none_equals_recon_and_global_shift_equals_make_z_steer():
    torch.manual_seed(0)
    ae = GeoAE(hidden_size=8, latent_dim=16, n_clusters=4).eval()
    mean, std = torch.randn(8), torch.rand(8) + 0.5
    hs = _hs()
    inf16 = np.full(16, np.inf, dtype=np.float32)
    recon = nl.make_z_gate(inf16, -inf16, np.zeros(16, np.float32), ae, mean, std, DEV, gate=False)(hs)
    assert torch.allclose(nl.make_z_edit(None, ae, mean, std, DEV)(hs), recon)
    r = np.random.RandomState(3).randn(16).astype(np.float32)
    got = nl.make_z_edit(nl.shift_edit(r, -inf16, inf16, 2.0, DEV), ae, mean, std, DEV)(hs)
    assert torch.allclose(got, nl.make_z_steer(r, 2.0, ae, mean, std, DEV)(hs))


def test_range_shift_only_moves_in_range_values():
    a = torch.tensor([[0.0, 5.0], [1.0, -5.0]])
    edit = nl.shift_edit(np.array([1.0, 1.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1.0, DEV)
    assert torch.equal(edit(a), torch.tensor([[-1.0, 5.0], [0.0, -5.0]]))


def test_transport_maps_concept_range_onto_complement_range():
    mu_c, sd_c = np.array([2.0]), np.array([0.5])
    mu_o, sd_o = np.array([-1.0]), np.array([2.0])
    lo, hi = mu_c - 2 * sd_c, mu_c + 2 * sd_c
    edit = nl.transport_edit(mu_c, sd_c, mu_o, sd_o, lo, hi, 1.0, DEV)
    a = torch.tensor([[2.0], [2.5], [9.0]])           # centre, +1 sd, out of range
    out = edit(a)
    assert torch.allclose(out[0], torch.tensor([-1.0]))          # mu_c -> mu_o
    assert torch.allclose(out[1], torch.tensor([1.0]))           # +1 sd_c -> +1 sd_o
    assert torch.equal(out[2], a[2])                             # outside the range: untouched
    half = nl.transport_edit(mu_c, sd_c, mu_o, sd_o, lo, hi, 0.5, DEV)(a)
    assert torch.allclose(half[0], torch.tensor([0.5]))          # alpha interpolates


def test_transport_zero_sd_is_finite():
    edit = nl.transport_edit(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]),
                             np.array([1.0]), np.array([1.0]), 1.0, DEV)
    out = edit(torch.tensor([[1.0], [3.0]]))
    assert torch.isfinite(out).all() and out[0].item() == 0.0 and out[1].item() == 3.0


def test_dprime_ignores_shared_offset():
    rng = np.random.RandomState(0)
    acts = rng.randn(200, 3).astype(np.float32)
    is_c = np.arange(200) < 100
    acts[is_c, 1] += 3.0                               # dim 1 separates the concept
    s = nl.dprime_saliency(acts, is_c)
    shifted = nl.dprime_saliency(acts + 50.0, is_c)    # a shared offset on every dim
    assert s.argmax() == 1 and np.allclose(s, shifted, atol=1e-3)
