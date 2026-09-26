"""closest_tokens --sampled_from: held-out doc selection and position sampling (CPU)."""
import gzip
import json

import numpy as np
import pytest
import torch

from geoae.interp.closest_tokens import sampled_sources, stream_sampled_docs


class _WordTok:
    """One token per whitespace word; ids are the word's integer value."""

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [int(w) for w in text.split()]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids])}


def _write_dump(tmp_path, docs_per_src, consumed, weights, skipped=None):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    man = {"sources": []}
    for dom, docs in docs_per_src.items():
        with gzip.open(corpus / f"{dom}.jsonl.gz", "wt", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"text": d}) + "\n")
        man["sources"].append({"domain": dom, "file": f"{dom}.jsonl.gz"})
    json.dump(man, open(corpus / "manifest.json", "w"))
    acts = tmp_path / "acts"
    acts.mkdir()
    meta = {"mode": "sampled", "corpus_dir": str(corpus), "context_len": 50,
            "positions_per_doc": 8, "skip_leading": 2,
            "sources": [{"domain": d, "weight": weights[d], "docs": consumed[d],
                         "skipped_short": (skipped or {}).get(d, 0)} for d in docs_per_src]}
    json.dump(meta, open(acts / "meta.json", "w"))
    return acts


def _doc(tag, n):  # n tokens, every token = tag, so a yielded doc names its source doc
    return " ".join([str(tag)] * n)


def test_held_out_docs_skip_consumed_and_skipped_short(tmp_path):
    acts = _write_dump(tmp_path,
                       {"a": [_doc(i, 30) for i in range(10)], "b": [_doc(100 + i, 30) for i in range(10)]},
                       consumed={"a": 4, "b": 6}, weights={"a": 1, "b": 1}, skipped={"b": 1})
    meta, srcs = sampled_sources(acts)
    assert [s["offset"] for s in srcs] == [4, 7]
    seen = {"a": [], "b": []}
    for ids, dom, pos in stream_sampled_docs(srcs, meta, 40, _WordTok(), min_len=3, seed=0):
        seen[dom].append(int(ids[0, 0]))
    assert seen["a"] and min(seen["a"]) == 4          # first held-out doc of a
    assert seen["b"] and min(seen["b"]) == 107
    assert seen["a"] == sorted(seen["a"]) and seen["b"] == sorted(seen["b"])


def test_positions_budget_and_weights(tmp_path):
    acts = _write_dump(tmp_path,
                       {"a": [_doc(1, 40)] * 200, "b": [_doc(2, 40)] * 200},
                       consumed={"a": 0, "b": 0}, weights={"a": 3, "b": 1})
    meta, srcs = sampled_sources(acts)
    rows = {"a": 0, "b": 0}
    total = 0
    for ids, dom, pos in stream_sampled_docs(srcs, meta, 800, _WordTok(), min_len=3, seed=0):
        assert pos.min() >= meta["skip_leading"] and pos.max() < ids.shape[1]
        assert len(pos) <= meta["positions_per_doc"] and (np.diff(pos) > 0).all()
        rows[dom] += len(pos)
        total += len(pos)
    assert total == 800
    assert abs(rows["a"] / total - 0.75) < 0.03


def test_exhausted_source_is_killed_not_fatal(tmp_path, capsys):
    acts = _write_dump(tmp_path,
                       {"a": [_doc(1, 40)] * 3, "b": [_doc(2, 40)] * 100},
                       consumed={"a": 1, "b": 0}, weights={"a": 1, "b": 1})
    meta, srcs = sampled_sources(acts)
    out = list(stream_sampled_docs(srcs, meta, 200, _WordTok(), min_len=3, seed=0))
    assert sum(len(p) for _, _, p in out) == 200
    assert sum(1 for _, d, _ in out if d == "a") == 2
    assert "held-out docs exhausted" in capsys.readouterr().out


def test_rejects_prefix_dump_and_missing_corpus(tmp_path):
    (tmp_path / "prefix").mkdir()
    json.dump({"mode": "prefix"}, open(tmp_path / "prefix" / "meta.json", "w"))
    with pytest.raises(ValueError, match="not a sampled-position dump"):
        sampled_sources(tmp_path / "prefix")
    acts = _write_dump(tmp_path, {"a": [_doc(1, 20)]}, consumed={"a": 0}, weights={"a": 1})
    with pytest.raises(FileNotFoundError, match="--corpus_dir"):
        sampled_sources(acts, corpus_dir=tmp_path / "nowhere")
