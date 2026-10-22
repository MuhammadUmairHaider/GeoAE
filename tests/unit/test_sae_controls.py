"""SAE controls: random-SAE null, frequent-feature top-1, code-space k-means, balanced fit, non-sibling null pairs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geoae.baselines.sae_controls import balanced_fit
from geoae.baselines.sae_controls import sample_pairs_nonsibling
from geoae.baselines.sae_controls import summarize_pairs
from geoae.baselines.sae_controls import top_features
from geoae.baselines.sae_coding import analyze_coding
from geoae.baselines.sae_space import CodeKMeans
from geoae.baselines.sae_space import RandomSpace
from geoae.baselines.sae_space import RestrictedSpace
from geoae.baselines.sae_space import load_sae_kmeans
from geoae.baselines.sae_space import load_sae_space
from tests.unit.test_sae_space import D, F, data, matched_parts, scope_cfg, scope_params

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval_out"))


def write_matched(tmp_path):
    core, tf = matched_parts()
    torch.save({"recipe": {"method": "batch_topk", "activation_dim": D, "width": F, "k": 3},
                "model_state": core.state_dict(),
                "transform_state": {"norm_mean": tf.norm_mean, "norm_std": tf.norm_std,
                                    "tb_table": tf.tb_table, "tb_index": tf.tb_index}, "provenance": {}},
               tmp_path / "last.pt")
    return tmp_path / "last.pt", tf


def write_scope(tmp_path):
    save_file(scope_params(), str(tmp_path / "params.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(scope_cfg()))
    return tmp_path


def test_random_space(tmp_path):
    path, tf = write_matched(tmp_path)
    sp = load_sae_space(f"random:64:3:{path}", "cpu")
    again = load_sae_space(f"random:64:3:{path}", "cpu")
    other = load_sae_space(f"random:64:4:{path}", "cpu")
    assert isinstance(sp, RandomSpace) and sp.n_features == 64 and sp.needs_tokens and not sp.has_decoder
    torch.testing.assert_close(sp.W.norm(dim=1), torch.ones(64))
    torch.testing.assert_close(sp.W, again.W)
    assert not torch.allclose(sp.W, other.W)
    h, tok = data()
    x = (h - tf.norm_mean) / tf.norm_std - tf.token_bias(tok)
    torch.testing.assert_close(sp.encode_raw(h, tok), x @ sp.W.T)
    np.testing.assert_array_equal(sp.top1_labels(h, tok, chunk=7).numpy(), (x @ sp.W.T).argmax(1).numpy())
    with pytest.raises(ValueError):
        sp.top1_labels(h, None)
    with pytest.raises(NotImplementedError):
        sp.decode_raw(sp.encode_raw(h, tok), tok)


def test_top_features_and_restricted(tmp_path):
    counts = np.array([5, 9, 9, 0, 7])
    assert top_features(counts, 3).tolist() == [1, 2, 4]
    root = write_scope(tmp_path)
    h, _ = data()
    full = load_sae_space(f"scope:{root}", "cpu")
    c = np.zeros(F, int); c[[3, 11, 20, 30]] = [4, 9, 1, 7]
    np.savez(tmp_path / "counts.npz", counts=c)
    r = load_sae_space(f"scope:{root}#top3={tmp_path / 'counts.npz'}", "cpu")
    assert isinstance(r, RestrictedSpace) and r.n_features == 3 and not r.has_decoder
    sel = [11, 30, 3]
    codes = full.encode_raw(h)
    torch.testing.assert_close(r.encode_raw(h), codes[:, sel])
    expect = torch.where(codes[:, sel].max(1).values > 0, codes[:, sel].argmax(1), torch.full((len(h),), -1))
    np.testing.assert_array_equal(r.top1_labels(h, chunk=7).numpy(), expect.numpy())
    assert r.top1_labels(h).max() < 3


def test_code_kmeans(tmp_path):
    root = write_scope(tmp_path)
    full = load_sae_space(f"scope:{root}", "cpu")
    h, _ = data(60)
    codes = full.encode_raw(h)
    C = codes[[0, 5, 9, 20]].clone()
    np.savez(tmp_path / "km.npz", centroids=C.numpy())
    km = load_sae_kmeans(f"scope:{root}@{tmp_path / 'km.npz'}", "cpu", "x")
    assert isinstance(km, CodeKMeans) and km.n_labels == 4 and not km.needs_tokens
    lab = km.assign_labels(h, None, chunk=1000)
    np.testing.assert_array_equal(km.assign_labels(h, None, chunk=7).numpy(), lab.numpy())
    np.testing.assert_array_equal(lab.numpy(), torch.cdist(codes, C).argmin(1).numpy())
    assert lab[0] == 0 and lab[5] == 1


def test_balanced_fit_tiny():
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(6, 8, generator=g) * 6
    X = centers[torch.randint(0, 6, (600,), generator=g)] + torch.randn(600, 8, generator=g) * .3
    C = balanced_fit(X, 6, epochs=3, batch_size=60, density_pool=300, reinit_every=0, density_knn=8, log=lambda *_: None)
    assert C.shape == (6, 8) and torch.isfinite(C).all()
    assert len(torch.cdist(X, C).argmin(1).unique()) >= 4


def test_nonsibling_pairs_and_summary():
    fam = {"a1": "f1", "b1": "f1", "a2": "f2", "b2": "f2", "c": "f3"}
    pairs, n = sample_pairs_nonsibling(fam.keys(), fam, 5, 0)
    assert n == 8 and len(pairs) == 5 and all(fam[a] != fam[b] for a, b in pairs)
    assert pairs == sample_pairs_nonsibling(fam.keys(), fam, 5, 0)[0]
    assert len(sample_pairs_nonsibling(fam.keys(), fam, 100, 0)[0]) == 8
    s = summarize_pairs([{"decoder_cosine": 1.0, "interpretation": "x", "feature_geometry": "same_feature"},
                         {"decoder_cosine": .95, "interpretation": "y", "feature_geometry": "distinct_parallel"},
                         {"decoder_cosine": .1, "interpretation": "y", "feature_geometry": "distinct_unrelated"},
                         {"decoder_cosine": None, "interpretation": "unresolved", "feature_geometry": "unassigned"}])
    assert s["n_pairs"] == 4 and s["same_feature_rate"] == .25
    assert s["rate_top1_cos_ge_thr"] == .5 and abs(s["rate_top1_cos_ge_thr_among_assigned"] - 2 / 3) < 1e-9


def test_coding_null_script_path():
    import experiments.paper.analysis.sae_coding_null as N
    rng = np.random.default_rng(0)
    n, width = 400, 12
    feats = torch.relu(torch.randn(n, width, generator=torch.Generator().manual_seed(1)))
    labels = {f"L{i}": rng.random(n) < .2 for i in range(6)}
    real = [{"a": "L0", "b": "L1", "family": "fam0"}, {"a": "L2", "b": "L3", "family": "fam1"}]
    d = {"pairs": real + [{"a": "L4", "b": "L5", "family": "fam2"}]}
    nulls, n_cand = N.null_pairs(d, 7, 0)
    fam = {p[s]: p["family"] for p in d["pairs"] for s in "ab"}
    assert nulls and all(fam[p["a"]] != fam[p["b"]] for p in nulls)
    fit = np.arange(n) % 2 == 0
    res = analyze_coding(feats, labels, real + nulls, torch.randn(width, 5), fit, ~fit,
                         controls=~np.any(list(labels.values()), 0), budgets=(1, 2))
    s = summarize_pairs(res["pairs"][len(real):])
    assert s["n_pairs"] == len(nulls)
