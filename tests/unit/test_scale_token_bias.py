"""Partial subtraction must survive table loading and checkpoint reconstruction."""
import numpy as np
import pytest
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.config import Config
from geoae.model import GeoAE
from geoae.token_bias import TokenBiasLookup, load_table
from geoae.extraction.token_bias_scale import scale_table


def test_half_bias_round_trip(tmp_path):
    source, out = tmp_path / "base.npz", tmp_path / "half.npz"
    table = np.array([[4., -2., 1.], [-6., 8., 2.]], dtype=np.float16)
    np.savez(source, token_ids=np.array([2, 4]), table=table, counts=np.array([50, 80]),
             vocab_size=8, norm_mean=np.zeros(3), norm_std=np.ones(3),
             report='{"var_removed_shrunk": 0.4}')
    scale_table(source, out, 0.5)
    half = load_table(out)
    assert float(half["bias_scale"]) == 0.5
    assert "report" not in half and "source_report" in half
    np.testing.assert_array_equal(half["counts"], [50, 80])
    np.testing.assert_array_equal(half["norm_std"], np.ones(3))
    lookup = TokenBiasLookup(out, "cpu")
    tokens = torch.tensor([2, 4, 7])
    expected = torch.tensor([[2., -1., .5], [-3., 4., 1.], [0., 0., 0.]])
    torch.testing.assert_close(lookup(tokens), expected)

    cfg = Config()
    cfg.model.hidden_size = 3
    cfg.model.latent_dim = 4
    cfg.model.n_clusters = 2
    cfg.model.token_bias = str(out)
    model = GeoAE(hidden_size=3, latent_dim=4, n_clusters=2, token_bias_rows=2, vocab_size=8)
    model.load_token_bias(torch.tensor(half["token_ids"]), torch.tensor(half["table"]))
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(), "tau": 1.,
                "epoch": 1, "step": 1, "norm_mean": np.zeros(3), "norm_std": np.ones(3)}, path)
    # Evaluation must get the scale from the embedded table, not reload an unscaled table.
    out.unlink()
    restored, *_ = load_ae_checkpoint(path, "cpu", allow_token_bias=True)
    torch.testing.assert_close(restored.token_bias(tokens), expected)
    x = torch.randn(3, 3)
    torch.testing.assert_close(restored.encode(x, tokens), restored.encoder(x - expected))
    z = torch.randn(3, 4)
    torch.testing.assert_close(restored.decode(z, tokens), restored.decoder(z) + expected)


@pytest.mark.parametrize("alpha", [0, 1, -0.5, float("nan"), float("inf")])
def test_invalid_alpha_does_not_create_table(tmp_path, alpha):
    out = tmp_path / "half.npz"
    with pytest.raises(ValueError, match="alpha"):
        scale_table(tmp_path / "base.npz", out, alpha)
    assert not out.exists()
