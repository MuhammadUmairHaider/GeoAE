"""SAE adapters used by the representation-level evals: matched TopK (token bypass) and Gemma Scope JumpReLU."""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from geoae.baselines.gemma_scope import GemmaScopeSAE
from geoae.baselines.sae_space import MatchedSpace
from geoae.baselines.sae_space import ScopeSpace
from geoae.baselines.sae_space import chunk_rows
from geoae.baselines.sae_space import load_sae_space
from geoae.baselines.sae_space import parse_saes
from geoae.baselines.matched_topk._reference.trainers.batch_top_k import BatchTopKSAE
from geoae.baselines.matched_topk.common import BypassTransform
from geoae.baselines.matched_topk.common import MatchedSAE

D, F, V = 16, 40, 7


def scope_cfg():
    return {"hf_hook_point_in": "model.layers.33.output", "hf_hook_point_out": "model.layers.33.output",
            "width": F, "model_name": "google/gemma-3-4b-pt", "architecture": "jump_relu",
            "l0": 20, "affine_connection": False, "type": "sae"}


def scope_params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w_enc": torch.randn(D, F, generator=g), "w_dec": torch.randn(F, D, generator=g),
            "b_enc": torch.randn(F, generator=g) * .1, "b_dec": torch.randn(D, generator=g),
            "threshold": torch.rand(F, generator=g) * 2}


def matched_parts(seed=0):
    g = torch.Generator().manual_seed(seed)
    core = BatchTopKSAE(D, F, 3)
    core.load_state_dict({k: (torch.randn_like(v, generator=g) if v.ndim else torch.tensor(.5))
                          for k, v in core.state_dict().items()})
    tf = BypassTransform(torch.randn(D, generator=g), torch.rand(D, generator=g) + .5,
                         torch.randn(4, D, generator=g).half(), torch.tensor([0, 1, -1, 2, 3, -1, 0]))
    return core, tf


def data(n=50, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g) * 3, torch.randint(0, V, (n,), generator=g)


def test_matched_formula_and_bypass_required():
    core, tf = matched_parts()
    sp = MatchedSpace(MatchedSAE(core, tf), "m")
    h, tok = data()
    codes = sp.encode_raw(h, tok)
    assert codes.shape == (50, F) and sp.n_features == F and sp.needs_tokens
    tb = tf.token_bias(tok)
    x = (h - tf.norm_mean) / tf.norm_std - tb
    torch.testing.assert_close(codes, core.encode(x, use_threshold=True))
    rec = sp.decode_raw(codes, tok)
    torch.testing.assert_close(rec, (core.decode(codes) + tb) * tf.norm_std + tf.norm_mean)
    # Small/chunked GEMMs use different fp32 accumulation orders. Check both
    # against an independent double-precision formula, including normalization,
    # thresholding and restoration, with an explicit fp32 error tolerance.
    x64 = (h.double() - tf.norm_mean.double()) / tf.norm_std.double() - tb.double()
    pre64 = torch.relu((x64 - core.b_dec.double()) @ core.encoder.weight.double().T
                       + core.encoder.bias.double())
    codes64 = pre64 * (pre64 > core.threshold.double())
    decoded64 = codes64 @ core.decoder.weight.double().T + core.b_dec.double()
    expected64 = (decoded64 + tb.double()) * tf.norm_std.double() + tf.norm_mean.double()
    torch.testing.assert_close(rec.double(), expected64, rtol=1e-5, atol=2e-5)
    chunked = sp.reconstruct_raw(h, tok, chunk=7)
    torch.testing.assert_close(chunked.double(), expected64, rtol=1e-5, atol=2e-5)
    with pytest.raises(ValueError):
        sp.encode_raw(h, None)
    with pytest.raises(ValueError):
        sp.top1_labels(h, None)
    with pytest.raises(ValueError):
        sp.decode_raw(codes, None)


@pytest.mark.parametrize("kind", ["matched", "scope"])
def test_top1_chunking_and_unassigned(kind):
    h, tok = data(60)
    if kind == "matched":
        core, tf = matched_parts()
        core.threshold.fill_(-1.0)
        sp = MatchedSpace(MatchedSAE(core, tf), "m")
        core.threshold.fill_(0.0)
        top = sp.encode_raw(h, tok).max(1).values
        core.threshold.fill_(float(top.median()))   # half the rows have nothing above threshold
    else:
        p = scope_params()
        sp = ScopeSpace(GemmaScopeSAE(p, scope_cfg()), "s")
        top = sp.encode_raw(h).max(1).values
        p["threshold"] = torch.full((F,), float(top.median()))
        sp = ScopeSpace(GemmaScopeSAE(p, scope_cfg()), "s")
    tok = tok if sp.needs_tokens else None
    full = sp.top1_labels(h, tok, chunk=1000)
    np.testing.assert_array_equal(sp.top1_labels(h, tok, chunk=7).numpy(), full.numpy())
    codes = sp.encode_raw(h, tok)
    none = (codes != 0).sum(1) == 0
    assert none.any() and (~none).any()
    assert (full[none] == -1).all()
    assert (full[~none] == codes[~none].argmax(1)).all()


def test_scope_jumprelu_threshold_semantics():
    p = {"w_enc": torch.eye(2, 3)[:, :3].clone(), "w_dec": torch.zeros(3, 2), "b_enc": torch.zeros(3),
         "b_dec": torch.zeros(2), "threshold": torch.tensor([1., 1., 0.])}
    p["w_enc"] = torch.tensor([[1., 0., -1.], [0., 1., -1.]])
    cfg = scope_cfg(); cfg["width"] = 3
    sp = ScopeSpace(GemmaScopeSAE(p, cfg))
    h = torch.tensor([[1., 2.], [.5, 1.5], [-1., -1.]])
    z = sp.encode_raw(h)
    # exactly at threshold stays 0 (strict >); pre-act kept (not threshold-subtracted); negative never fires
    torch.testing.assert_close(z, torch.tensor([[0., 2., 0.], [0., 1.5, 0.], [0., 0., 2.]]))
    assert sp.top1_labels(h).tolist() == [1, 1, 2]
    assert sp.top1_labels(torch.tensor([[0., 0.]])).tolist() == [-1]


def test_scope_shapes_fp32_chunked_recon():
    sp = ScopeSpace(GemmaScopeSAE(scope_params(), scope_cfg()))
    h, _ = data()
    z = sp.encode_raw(h.double(), None)
    assert z.dtype == torch.float32 and z.shape == (50, F)
    rec = sp.decode_raw(z)
    assert rec.shape == h.shape
    torch.testing.assert_close(sp.reconstruct_raw(h, None, chunk=9), sp.decode_raw(sp.encode_raw(h)))


def test_loaders_and_helpers(tmp_path):
    save_file(scope_params(), str(tmp_path / "params.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(scope_cfg()))
    sp = load_sae_space(f"scope:{tmp_path}", "cpu", "gs")
    assert isinstance(sp, ScopeSpace) and sp.n_features == F and sp.name == "gs"

    core, tf = matched_parts()
    ck = {"recipe": {"method": "batch_topk", "activation_dim": D, "width": F, "k": 3},
          "model_state": core.state_dict(),
          "transform_state": {"norm_mean": tf.norm_mean, "norm_std": tf.norm_std,
                              "tb_table": tf.tb_table, "tb_index": tf.tb_index},
          "provenance": {}}
    torch.save(ck, tmp_path / "last.pt")
    m = load_sae_space(f"matched:{tmp_path / 'last.pt'}", "cpu")
    h, tok = data()
    torch.testing.assert_close(m.encode_raw(h, tok), MatchedSpace(MatchedSAE(core, tf)).encode_raw(h, tok))
    with pytest.raises(ValueError):
        load_sae_space("bogus:x", "cpu")
    assert parse_saes("a=matched:/x/last.pt,b=scope:262k") == {"a": "matched:/x/last.pt", "b": "scope:262k"}
    assert chunk_rows(262144) == 1024 and chunk_rows(16384) == 4096
