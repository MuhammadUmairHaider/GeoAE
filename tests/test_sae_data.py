"""Document isolation, exclusive Atlas labels, cache integrity, and safe resumption."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoae.interp import sae_compare as C
from geoae.interp import sae_data as D


def _write_range(directory, *, dtype=np.float32):
    docs = np.repeat(np.arange(36), 2)
    n = len(docs)
    cluster = np.asarray([7, 9, 99])[docs % 3]
    h = np.zeros((n, 2560), dtype=dtype)
    h[:, 0], h[:, 1] = np.arange(n), docs
    arrays = {
        "h": [h], "tok": [np.arange(n, dtype=np.int64)],
        "nxt": [np.arange(n, dtype=np.int64) + 1], "doc": [docs],
        "cluster": [cluster], "control": [cluster == 99],
        "fit": [D.AS.split_half(docs, 0)],
    }
    meta = {
        "kind": "range", "model": D.MODEL, "layer": D.LAYER, "seed": 0,
        "pairs": [{"a": 7, "b": 9, "family": "range_7", "geoae_direction_count": 1}],
        "checkpoint_sha256": "checkpoint_a",
    }
    D.write_rows(directory, arrays, meta)
    return arrays, meta


def _rewrite_meta(directory, update):
    path = directory / "meta.json"
    meta = json.loads(path.read_text())
    meta.update(update)
    path.write_text(json.dumps(meta))


def test_range_sampling_preserves_document_split_and_is_reproducible(tmp_path):
    _write_range(tmp_path)
    a = D.load_rows(tmp_path, rows_per_label=6, control_rows=5, seed=19)
    b = D.load_rows(tmp_path, rows_per_label=6, control_rows=5, seed=19)
    for key in ("h", "tok", "nxt", "doc", "fit", "controls"):
        np.testing.assert_array_equal(a[key], b[key])
    fit_docs, eval_docs = set(a["doc"][a["fit"]]), set(a["doc"][~a["fit"]])
    assert fit_docs and eval_docs and fit_docs.isdisjoint(eval_docs)
    np.testing.assert_array_equal(a["fit"], D.AS.split_half(a["doc"], 0))
    for mask in a["memberships"].values():
        assert np.count_nonzero(mask & a["fit"]) == 6
        assert np.count_nonzero(mask & ~a["fit"]) == 6
    assert not np.any(a["memberships"]["cluster_7"] & a["memberships"]["cluster_9"])
    assert a["pairs"][0]["a"] == "cluster_7"
    assert a["pairs"][0]["b"] == "cluster_9"


def test_atlas_merged_labels_exclude_both_sides_and_split_whole_documents(tmp_path):
    row_labels = [[11], [13, 13], [22], [11, 22], [13, 22], []]
    docs = np.repeat(np.arange(20), len(row_labels))
    labels = row_labels * 20
    n = len(docs)
    arrays = {
        "h": [np.zeros((n, 2560), np.float32)], "tok": [np.arange(n)],
        "nxt": [np.arange(n) + 1], "doc": [docs],
        "control": [np.tile([False, False, False, False, False, True], 20)],
        "fit": [D.AS.split_half(docs, 2)], "labels": labels,
    }
    meta = {
        "kind": "atlas", "model": D.MODEL, "layer": D.LAYER, "seed": 2,
        "pairs": [{"a": "A", "b": "B", "a_ids": [11, 13], "b_ids": [22], "lcc": "Q"}],
    }
    D.write_rows(tmp_path, arrays, meta)
    data = D.load_rows(tmp_path, rows_per_label=1000, control_rows=1000)
    assert not set(data["doc"][data["fit"]]) & set(data["doc"][~data["fit"]])
    positions = data["tok"] % 6
    np.testing.assert_array_equal(data["memberships"]["pair_0:a"], np.isin(positions, [0, 1]))
    np.testing.assert_array_equal(data["memberships"]["pair_0:b"], positions == 2)
    assert not np.isin(positions, [3, 4]).any()  # Both-label chunks are neither alternative nor controls.
    assert np.count_nonzero(data["memberships"]["pair_0:a"]) == 40
    assert np.count_nonzero(data["memberships"]["pair_0:b"]) == 20


def test_atlas_overlap_does_not_consume_exclusive_collection_quotas(tmp_path, monkeypatch):
    labels = [[11, 22], [11], [22], []] * 2
    docs = np.repeat([0, 1], 4)
    block = {"chunk_text": ["text"] * 8, "doc_int_id": docs,
             "content_ids": labels, "entity_ids": [[]] * 8,
             "tone_ids": [[]] * 8, "document_ids": [[]] * 8}
    monkeypatch.setattr(D.AS, "iter_atlas_chunks", lambda *a, **k: iter([block]))
    captured = {}
    removed = []

    class Handle:
        def remove(self):
            removed.append(True)

    monkeypatch.setattr(D, "register_capture", lambda *a: (captured, [Handle()]))

    def forward(_model, seqs, _pad_id, _device):
        captured[D.LAYER] = torch.zeros(len(seqs), max(map(len, seqs)), 2560)

    monkeypatch.setattr(D, "forward_batch", forward)

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(10))}

    pair_doc = {"pairs": [{"a": "A", "b": "B", "a_ids": [11], "b_ids": [22], "lcc": "Q"}]}
    D.collect_atlas(tmp_path, "unused", pair_doc, torch.nn.Linear(1, 1), Tokenizer(),
                    rows_per_label=1, controls_per_split=1, positions=1, batch=3)
    data = D.load_rows(tmp_path, rows_per_label=100, control_rows=100)
    assert removed == [True]
    assert np.count_nonzero(data["memberships"]["pair_0:a"]) == 2
    assert np.count_nonzero(data["memberships"]["pair_0:b"]) == 2
    assert np.count_nonzero(data["controls"]) == 2
    assert len(data["h"]) == 6


def test_atlas_pairs_sharing_a_label_form_one_bootstrap_family(tmp_path):
    docs = np.repeat(np.arange(12), 5)
    labels = [[11], [22], [33], [44], [55]] * 12
    n = len(docs)
    parts = {"h": [np.zeros((n, 2560), np.float32)], "tok": [np.arange(n)],
             "nxt": [np.arange(n) + 1], "doc": [docs],
             "control": [np.zeros(n, bool)], "fit": [D.AS.split_half(docs)], "labels": labels}
    pairs = [{"a": "A", "b": "B", "a_ids": [11], "b_ids": [22], "lcc": "Q"},
             {"a": "B", "b": "C", "a_ids": [22], "b_ids": [33], "lcc": "Q"},
             {"a": "D", "b": "E", "a_ids": [44], "b_ids": [55], "lcc": "Q"}]
    D.write_rows(tmp_path, parts, {"kind": "atlas", "model": D.MODEL, "layer": D.LAYER, "pairs": pairs})
    data = D.load_rows(tmp_path, rows_per_label=100, control_rows=100)
    family_a, family_b, family_c = [pair["family"] for pair in data["pairs"]]
    assert family_a == family_b
    assert family_a != family_c


def test_fingerprint_detects_changed_content_at_the_same_path(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint version a")
    first = D.fingerprint(path)
    assert first == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"checkpoint version b")
    assert D.fingerprint(path) != first


@pytest.mark.parametrize("dtype", [np.float16, np.float64])
def test_writer_refuses_non_fp32_residuals(tmp_path, dtype):
    with pytest.raises(ValueError, match="float32"):
        _write_range(tmp_path, dtype=dtype)


def test_writer_refuses_fp32_overflow(tmp_path):
    parts = {"h": [np.asarray([[np.inf, 1.]], np.float32)]}
    with pytest.raises(ValueError, match="finite"):
        D.write_rows(tmp_path, parts, {"kind": "range"})


def test_loader_refuses_non_fp32_cache(tmp_path):
    _write_range(tmp_path)
    path = tmp_path / "h.npy"
    np.save(path, np.load(path).astype(np.float16))
    _rewrite_meta(tmp_path, {"h_sha256": D.fingerprint(path)})
    with pytest.raises(ValueError):
        D.load_rows(tmp_path)


def test_loader_detects_tampered_residual_fingerprint(tmp_path):
    _write_range(tmp_path)
    path = tmp_path / "h.npy"
    h = np.load(path)
    h[0, 0] += 1
    np.save(path, h)
    with pytest.raises(ValueError):
        D.load_rows(tmp_path)


def test_loader_refuses_nonfinite_fp32_even_with_matching_hash(tmp_path):
    _write_range(tmp_path)
    path = tmp_path / "h.npy"
    h = np.load(path)
    h[0, 0] = np.nan
    np.save(path, h)
    _rewrite_meta(tmp_path, {"h_sha256": D.fingerprint(path)})
    with pytest.raises(ValueError):
        D.load_rows(tmp_path)


def test_loader_refuses_fit_and_eval_rows_from_the_same_document(tmp_path):
    arrays, _ = _write_range(tmp_path)
    fit = arrays["fit"][0].copy()
    fit[0] = ~fit[1]  # The first two rows belong to the same document.
    np.save(tmp_path / "fit.npy", fit)
    with pytest.raises(ValueError):
        D.load_rows(tmp_path)


def test_loader_refuses_row_count_mismatch(tmp_path):
    _write_range(tmp_path)
    tokens = np.load(tmp_path / "tok.npy")
    np.save(tmp_path / "tok.npy", np.append(tokens, 999))
    with pytest.raises(ValueError):
        D.load_rows(tmp_path)


@pytest.mark.parametrize("update", [{"model": "google/gemma-3-4b-it"}, {"layer": 32}])
def test_loader_refuses_model_or_layer_mismatch(tmp_path, update):
    _write_range(tmp_path)
    _rewrite_meta(tmp_path, update)
    with pytest.raises(ValueError, match="mismatch"):
        D.load_rows(tmp_path)


def test_prepare_refuses_cache_from_another_checkpoint_before_loading_lm(tmp_path, monkeypatch):
    cache = tmp_path / "cache/range"
    cache.mkdir(parents=True)
    (cache / "meta.json").write_text(json.dumps({"checkpoint_sha256": "old_checkpoint"}))
    monkeypatch.setattr(D, "load_geoae", lambda *a: (object(), torch.zeros(2), torch.ones(2), {}))
    monkeypatch.setattr(D, "fingerprint", lambda *a: "new_checkpoint")

    def forbid_lm(*args, **kwargs):
        pytest.fail("Checkpoint mismatch must be caught before LM loading")

    monkeypatch.setattr(C, "load_lm", forbid_lm)
    args = SimpleNamespace(checkpoint="checkpoint.pt", device="cpu", cache_dir=tmp_path / "cache",
                           out_dir=tmp_path / "out", kinds=["range"])
    with pytest.raises(ValueError, match="different checkpoint"):
        C.prepare(args)


def test_analysis_does_not_skip_removal_after_coding_only_completion(tmp_path, monkeypatch):
    """A --skip_removal smoke must never silently complete a later causal run."""
    from geoae.interp import gemma_scope

    class IdentityAE:
        def encode(self, h, tok):
            return h

        def decode(self, z, tok):
            return z

    class ReachedSAELoad(Exception):
        pass

    meta = {"checkpoint_sha256": "current", "h_sha256": "raw_rows", "model": D.MODEL,
            "layer": D.LAYER, "kind": "range", "seed": 0}
    data = {"h": np.ones((4, 2560), np.float32), "tok": np.arange(4), "nxt": np.arange(4),
            "doc": np.arange(4), "fit": np.asarray([True, True, False, False]),
            "controls": np.asarray([False, True, False, True]), "pairs": [],
            "memberships": {}, "meta": meta}
    monkeypatch.setattr(D, "load_geoae", lambda *a: (IdentityAE(), torch.zeros(2560), torch.ones(2560), {}))
    monkeypatch.setattr(D, "fingerprint", lambda *a: "current")
    monkeypatch.setattr(D, "load_rows", lambda *a, **k: data)
    monkeypatch.setattr(C, "load_lm", lambda *a, **k: object())
    monkeypatch.setattr(D, "fp32_head", lambda *a: (torch.nn.Identity(), torch.nn.Identity()))

    def stop_at_sae(**kwargs):
        raise ReachedSAELoad

    monkeypatch.setattr(gemma_scope, "load_gemma_scope", stop_at_sae)
    sae_path = "resid_post_all/layer_33_width_16k_l0_small"
    out = tmp_path / "out"
    dest = out / "range" / sae_path.split("/")[-1]
    dest.mkdir(parents=True)
    args = SimpleNamespace(
        checkpoint="checkpoint.pt", device="cpu", out_dir=out, cache_dir=tmp_path / "cache",
        kinds=["range"], rows_per_label=128, control_rows=512, seed=0, encode_batch=2,
        sae_paths=[sae_path], sae_repo="google/gemma-scope-2-4b-pt", sae_revision=C.REVISION,
        feature_budgets=[1, 2], reconstruction_budgets=[1, 2], strengths=[0, 1],
        target_damage=[.05], min_rows=8, score_rows=128, logit_chunk=32,
        bootstrap_samples=0, skip_removal=False,
    )
    (dest / "coding.json").write_text("{}")
    (dest / "reconstruction.json").write_text("{}")
    (dest / "complete.json").write_text(json.dumps({
        "removal_ran": False, "provenance": {"checkpoint_sha256": "current", "data": meta},
        "signature": C.analysis_signature(args, "range", meta, "current", sae_path),
    }))
    # Either resume the missing work or explicitly reject the incompatible
    # completion; returning successfully without removal is a correctness bug.
    with pytest.raises((ReachedSAELoad, ValueError)):
        C.analyze(args)
