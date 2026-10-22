"""Nonlinear latent interventions use the current state and preserve the residual baseline."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from geoae.interp.concept_remove_generate import make_edit_splice
from geoae.interp.concept_steer_generate import decode_direction
from geoae.interp.faithfulness_ablate import AEEnc
from geoae.interp.logit_attribution import compute_dla
from geoae.interp.range_number_control import make_last_patch
from geoae.model import GeoAE


def known_decoder(kind, *, token_bias=True):
    """A tiny GELU decoder with known coordinates and a large constant offset.

    The residual variant also adds z[:2]. Its nonlinear branch is nonzero, as
    it would be after training, rather than the zero correction at initialization.
    """
    ae = GeoAE(hidden_size=2, latent_dim=4, n_clusters=2, nonlinearity="linear",
               decoder_type=kind, decoder_hidden_dim=4, decoder_output_bias=True,
               token_bias_rows=2 if token_bias else 0, vocab_size=3 if token_bias else 0).eval()
    with torch.no_grad():
        ae.encoder.weight.zero_()
        ae.encoder.weight[:2].copy_(torch.eye(2))
        ae.encoder.bias.zero_()
        hidden = ae.decoder[0] if kind == "mlp" else ae.decoder.hidden
        output = ae.decoder[2] if kind == "mlp" else ae.decoder.output
        hidden.weight.copy_(torch.eye(4))
        hidden.bias.copy_(torch.tensor([0.2, -0.1, 0.0, 0.0]))
        output.weight.zero_()
        output.weight[:, :2].copy_(torch.eye(2))
        offset = torch.tensor([7.0, -11.0])
        if kind == "mlp":
            output.bias.copy_(offset)
        else:
            ae.decoder.skip.weight.zero_()
            ae.decoder.skip.weight[:, :2].copy_(torch.eye(2))
            ae.decoder.skip.bias.copy_(offset)
        if token_bias:
            ae.load_token_bias(torch.tensor([0, 1]), torch.tensor([[2.0, -3.0], [-1.0, 4.0]]))
    return ae


def expected_change(kind, z2, delta2):
    shift = torch.tensor([0.2, -0.1])
    change = F.gelu(z2 + shift + delta2) - F.gelu(z2 + shift)
    return change + delta2 if kind == "residual_mlp" else change


@pytest.mark.parametrize("kind", ["mlp", "residual_mlp"])
@torch.no_grad()
def test_same_latent_edit_depends_on_state_and_cancels_decoder_and_token_offsets(kind):
    ae = known_decoder(kind)
    z = torch.tensor([[-2.0, 0.5, 0.0, 0.0], [2.0, -0.5, 0.0, 0.0]])
    delta = torch.tensor([0.5, 0.25, 0.0, 0.0])
    expected = expected_change(kind, z[:, :2], delta[:2])
    tok = torch.tensor([0, 1])
    actual = ae.decoded_delta(z, delta, tok)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    # Different bypass offsets cannot become part of the edit itself.
    torch.testing.assert_close(ae.decoded_delta(z, delta, tok.flip(0)), expected,
                               atol=2e-6, rtol=2e-6)
    assert not torch.allclose(actual[0], actual[1])
    assert not torch.allclose(actual, ae.decoder(delta).expand_as(actual))


@pytest.mark.parametrize("kind", ["mlp", "residual_mlp"])
@pytest.mark.parametrize("edit_mode", ["edit", "identity", "gate_off"])
@torch.no_grad()
def test_concept_removal_changes_only_the_decoded_edit_without_reconstruction_error(kind, edit_mode):
    ae = known_decoder(kind)
    z2 = torch.tensor([[-2.0, 0.5], [2.0, -0.5], [0.0, 1.0], [-1.0, -0.25]])
    tok = torch.tensor([0, 1, 0, 1])
    mean, std = torch.tensor([3.0, -1.0]), torch.tensor([0.5, 2.5])
    x = z2 + ae.token_bias(tok)
    hs = (x * std + mean).reshape(2, 2, 2)
    tap = SimpleNamespace(ids_for=lambda hs: tok)
    delta = torch.tensor([0.5, 0.25, 0.0, 0.0])
    edit_fn = (lambda z: z) if edit_mode == "identity" else (lambda z: z + delta)
    log = {k: [] for k in ("fire", "edit_norm", "pos_fire_tgt", "pos_fire_comp")}
    gate_options = {}
    if edit_mode == "gate_off":
        gate_options = dict(pos_gate="llr", llr_thresh=float("inf"),
                            mu_c=np.zeros(4), sd_c=np.ones(4),
                            mu_rest=np.ones(4), sd_rest=np.ones(4))
    else:
        gate_options = dict(pos_gate="none")
    patch = make_edit_splice("z", "delta", edit_fn, np.full(4, -10.0), np.full(4, 10.0),
                             mean, std, ae, tap, "cpu", log, n_tgt=1, **gate_options)
    actual = patch(hs)
    change = expected_change(kind, z2, delta[:2]) if edit_mode == "edit" else torch.zeros_like(z2)
    expected = hs + (change * std).reshape_as(hs)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-6)
    # The model's deliberately large reconstruction offset must never enter a delta edit.
    recon = ae.decode(ae.encode(x, tok), tok) * std + mean
    assert not torch.allclose(actual.reshape_as(recon), recon)


@pytest.mark.parametrize("kind", ["mlp", "residual_mlp"])
@torch.no_grad()
def test_last_position_splice_decodes_the_edited_state_with_its_current_token(kind):
    ae = known_decoder(kind)
    hs = torch.tensor([[[8.0, -2.0], [4.0, 1.0]], [[-3.0, 5.0], [2.0, -1.0]]])
    mean, std = torch.tensor([3.0, -1.0]), torch.tensor([0.5, 2.5])
    all_tok = torch.tensor([1, 0, 0, 1])
    tok = all_tok.reshape(2, 2)[:, -1]
    tap = SimpleNamespace(ids_for=lambda hs: all_tok)
    delta = torch.tensor([0.5, 0.25, 0.0, 0.0])
    baseline, _ = make_last_patch(ae, mean, std, tap=tap)
    edited, _ = make_last_patch(ae, mean, std, edit=lambda z: z + delta, tap=tap)
    original_z2 = (hs[:, -1] - mean) / std - ae.token_bias(tok)
    change = expected_change(kind, original_z2, delta[:2]) * std
    torch.testing.assert_close(edited(hs)[:, -1] - baseline(hs)[:, -1], change,
                               atol=6e-6, rtol=2e-6)
    assert torch.equal(edited(hs)[:, :-1], hs[:, :-1])


@pytest.mark.parametrize("kind,bias", [("mlp", False), ("residual_mlp", False), ("linear", True)])
@pytest.mark.parametrize("tool", ["label_z", "dla", "faithfulness"])
def test_fixed_global_direction_tools_reject_models_without_bias_free_linear_decoders(kind, bias, tool):
    ae = GeoAE(hidden_size=2, latent_dim=4, n_clusters=2, decoder_type=kind,
               decoder_output_bias=bias).eval()
    with pytest.raises(ValueError, match="requires a bias-free linear decoder"):
        if tool == "label_z":
            decode_direction(ae, torch.ones(4))
        elif tool == "dla":
            compute_dla(ae, torch.ones(2), torch.ones(5, 2), None, False)
        else:
            wrapper = AEEnc.__new__(AEEnc)
            wrapper.ae, wrapper.std = ae, torch.ones(2)
            wrapper.to_raw_dir(torch.ones(4))


def test_dla_still_accepts_the_identity_decoder_kmeans_baseline(tmp_path):
    from geoae.interp.dla_inventory import KMeansShim

    path = tmp_path / "kmeans.npz"
    centroids = np.array([[1.0, 2.0], [-1.0, 3.0]], dtype=np.float32)
    np.savez(path, centroids=centroids, norm_mean=np.zeros(2), norm_std=np.ones(2))
    baseline = KMeansShim(path, "cpu")
    unembed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    actual = compute_dla(baseline, torch.ones(2), unembed, None, False)
    expected = torch.from_numpy(centroids) @ unembed.T
    expected -= expected.mean(1, keepdim=True)
    torch.testing.assert_close(actual, expected)
