"""Sampled-position extractor + structured corpus guard (CPU, no downloads)."""
import json

import numpy as np
import pytest
import torch

from geoae.extract_sampled import QuotaScheduler, extract_sampled, sample_positions
from geoae.structured_corpus import (LeakageGuard, counterfact_fact, filter_items, nq_item,
                                     pack_docs)


# --------------------------------------------------------------------------- positions

def test_sample_positions_bounds_sorted_unique():
    rng = np.random.default_rng(0)
    p = sample_positions(2000, 4, 64, rng)
    assert len(p) == 64 and p.min() >= 4 and p.max() < 2000
    assert (np.diff(p) > 0).all()


def test_sample_positions_short_doc_keeps_all_and_empty():
    rng = np.random.default_rng(0)
    assert sample_positions(20, 4, 64, rng).tolist() == list(range(4, 20))
    assert len(sample_positions(4, 4, 64, rng)) == 0


def test_sample_positions_roughly_uniform():
    rng = np.random.default_rng(0)
    p = np.concatenate([sample_positions(1004, 4, 64, rng) for _ in range(2000)])
    h, _ = np.histogram(p, bins=10, range=(4, 1004))
    assert h.min() / h.max() > 0.9


# --------------------------------------------------------------------------- scheduler

def test_quota_scheduler_tracks_weights_and_kill():
    s = QuotaScheduler([3, 1, 0], 4000)
    for _ in range(400):
        i = s.next()
        assert i != 2
        s.add(i, 10)
    assert abs(s.got[0] / s.got.sum() - 0.75) < 0.02
    s.kill(0)
    assert all(s.next() == 1 for _ in range(3))
    s.kill(1)
    assert s.next() is None


# --------------------------------------------------------------------------- guard

def test_guard_entities_whole_word_and_multiword():
    g = LeakageGuard(entities=["Paris", "Kuala Lumpur"])
    assert g.entity_hit("The capital is Paris.")
    assert not g.entity_hit("A Parisian cafe.")
    assert g.entity_hit("Flights to Kuala Lumpur leave daily.")
    assert not g.entity_hit("Kuala is a word.")


def test_guard_ioi_template():
    g = LeakageGuard(templates=["As {name_A} and {name_B} left the {place}, {name_C} gave a {object} to"])
    assert g.template_hit("As Carl and Maria left the consulate, Carl gave a fridge to Maria")
    assert not g.template_hit("Carl gave Maria a fridge.")


def test_filter_items_counts():
    g = LeakageGuard(entities=["Biu"])
    st = {}
    kept = filter_items(["Biu is a town.", "Rome is old.", None], g, st, "x")
    assert kept == ["Rome is old."]
    assert st["x"] == {"raw": 2, "dropped_entity": 1, "dropped_template": 0, "kept": 1}


def test_item_parsers_accept_dict_and_repr():
    rec = {"requested_rewrite": {"prompt": "The mother tongue of {} is", "subject": "Léon Blum",
                                 "target_true": {"str": "French", "id": "Q150"}}}
    assert counterfact_fact(rec) == "The mother tongue of Léon Blum is French."
    rec_s = {"requested_rewrite": repr(rec["requested_rewrite"])}
    assert counterfact_fact(rec_s) == "The mother tongue of Léon Blum is French."
    assert nq_item({"question": "where is rome", "answer": ["Italy"]}) == \
        "Question: Where is rome?\nAnswer: Italy"
    assert nq_item({"question": "q", "answer": "['A']"}).endswith("Answer: A")


def test_pack_docs_drops_ragged_tail():
    assert pack_docs(list("abcdefg"), 3, "\n") == ["a\nb\nc", "d\ne\nf"]


# --------------------------------------------------------------------------- end to end

class FakeTok:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, truncation=True, max_length=None):
        ids = [1] + [3 + (ord(c) % 60) for c in text]
        return {"input_ids": ids[:max_length] if truncation else ids}

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


def make_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=3, num_attention_heads=4,
                                     num_key_value_heads=2, head_dim=8))
    return m.eval()


def test_end_to_end_rows_match_unpadded_forward(tmp_path):
    rng = np.random.default_rng(1)
    texts = {0: ["x" * int(n) + "y" * 7 for n in rng.integers(5, 300, 200)],
             1: ["ab" * int(n) for n in rng.integers(3, 80, 400)]}
    iters = [iter(texts[0]), iter(texts[1])]
    tok, model = FakeTok(), make_llama()
    n = 3000
    meta = extract_sampled(
        model_name="tiny", layers=[1], target_layer=1, n_tokens=n, hidden_size=32,
        out_dir=tmp_path, sources=[{"domain": "a", "weight": 3}, {"domain": "b", "weight": 1}],
        context_len=128, positions_per_doc=16, skip_leading=4, min_doc_len=6, batch_docs=4,
        shuffle_buffer=0, outlier_norm_mult=0.0, dtype="float32", seed=0, log_every=10**9,
        model=model, tokenizer=tok, doc_iters=iters)

    X = np.load(tmp_path / "layer_1.npy")
    pos, doc, src = (np.load(tmp_path / f"rows_{k}.npy") for k in ("pos", "doc", "src"))
    docs = np.load(tmp_path / "docs.npy")
    assert X.shape == (n, 32) and meta["n_tokens"] == n
    assert pos.min() >= 4 and pos.max() < 128
    assert (np.diff(doc) >= 0).all()                        # docs stay contiguous
    assert docs["rows"].sum() == n
    assert abs(meta["source_share"]["a"] - 0.75) < 0.03

    # Reconstruct each doc's ids (docs were length-sorted inside pools, so match
    # by re-tokenising every text and checking a sampled doc's rows exactly).
    all_ids = [tok(t, max_length=128)["input_ids"] for s in (0, 1) for t in texts[s]]
    by_len = {}
    for ids in all_ids:
        by_len.setdefault(len(ids), []).append(ids)
    checked = 0
    for d in range(0, len(docs), max(1, len(docs) // 15)):
        rows = np.flatnonzero(doc == d)
        cands = [c for c in by_len[int(docs["length"][d])]
                 if all(c[p] == t for p, t in zip(pos[rows], np.load(tmp_path / "rows_tok.npy")[rows]))]
        ids = cands[0]
        with torch.no_grad():
            hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states[2][0]
        np.testing.assert_allclose(X[rows], hs[pos[rows]].numpy(), atol=1e-4, rtol=1e-4)
        checked += 1
    assert checked >= 10
    json.dumps(meta)                                         # meta is serialisable


def test_outlier_filter_drops_and_still_fills(tmp_path):
    texts = ["ab" * 50] * 500
    model, tok = make_llama(), FakeTok()
    meta = extract_sampled(
        model_name="tiny", layers=[1], target_layer=1, n_tokens=1000, hidden_size=32,
        out_dir=tmp_path, sources=[{"domain": "a", "weight": 1}], context_len=128,
        positions_per_doc=16, skip_leading=4, min_doc_len=6, batch_docs=4, shuffle_buffer=0,
        outlier_norm_mult=1.0, dtype="float32", seed=0, log_every=10**9,
        model=model, tokenizer=tok, doc_iters=[iter(texts)])
    assert meta["outliers"]["dropped_rows"] > 0
    assert meta["n_tokens"] == 1000
    norms = np.load(tmp_path / "rows_norm.npy")
    assert (norms <= meta["outliers"]["calibrated_median"] * 1.0 + 1e-5).all()


# --------------------------------------------------------------------------- frozen corpus

def _write_corpus(d, sources, texts, n_tokens, ctx=128, ppd=16, skip=4):
    import gzip
    d.mkdir()
    recs = []
    for s, ts in zip(sources, texts):
        with gzip.open(d / f"{s['domain']}.jsonl.gz", "wt") as f:
            for t in ts:
                f.write(json.dumps({"text": t}) + "\n")
        recs.append({"domain": s["domain"], "file": f"{s['domain']}.jsonl.gz", "docs": len(ts),
                     "est_rows": len(ts) * ppd, "spec": s})
    json.dump({"context_len": ctx, "positions_per_doc": ppd, "skip_leading": skip,
               "n_tokens": n_tokens, "sources": recs}, open(d / "manifest.json", "w"))


def _run(tmp_path, corpus, sources, n, **kw):
    return extract_sampled(
        model_name="tiny", layers=[1], target_layer=1, n_tokens=n, hidden_size=32,
        out_dir=tmp_path / "out", sources=sources, context_len=128, positions_per_doc=16,
        skip_leading=4, min_doc_len=6, batch_docs=4, shuffle_buffer=0, outlier_norm_mult=0.0,
        dtype="float32", seed=0, log_every=10**9, corpus_dir=corpus,
        model=make_llama(), tokenizer=FakeTok(), **kw)


def test_reads_frozen_corpus_and_weight_change_is_allowed(tmp_path):
    built = [{"domain": "a", "weight": 1, "name": "x"}, {"domain": "b", "weight": 1, "name": "y"}]
    _write_corpus(tmp_path / "c", built, [["ab" * 40] * 300, ["cd" * 40] * 300], 4000)
    use = [{"domain": "a", "weight": 3, "name": "x"}, {"domain": "b", "weight": 1, "name": "y"}]
    meta = _run(tmp_path, tmp_path / "c", use, 2000)
    assert abs(meta["source_share"]["a"] - 0.75) < 0.03
    assert meta["corpus_manifest"]["n_tokens"] == 4000


def test_spec_mismatch_rejected(tmp_path):
    built = [{"domain": "a", "weight": 1, "name": "x"}]
    _write_corpus(tmp_path / "c", built, [["ab" * 40] * 300], 4000)
    with pytest.raises(ValueError, match="different spec"):
        _run(tmp_path, tmp_path / "c", [{"domain": "a", "weight": 1, "name": "OTHER"}], 1000)


def test_exhausted_source_is_fatal(tmp_path):
    built = [{"domain": "a", "weight": 1, "name": "x"}, {"domain": "b", "weight": 1, "name": "y"}]
    _write_corpus(tmp_path / "c", built, [["ab" * 40] * 300, ["cd" * 40] * 5], 1000)
    with pytest.raises(RuntimeError, match="ran out"):
        # est_rows in the hand-written manifest is optimistic on purpose (5*16 < quota
        # is caught later, when the stream actually runs dry).
        man = json.load(open(tmp_path / "c" / "manifest.json"))
        man["sources"][1]["est_rows"] = 10**6
        json.dump(man, open(tmp_path / "c" / "manifest.json", "w"))
        _run(tmp_path, tmp_path / "c", built, 1000)


def test_spec_key_ignores_weight_only():
    from geoae.build_corpus import spec_key
    assert spec_key({"domain": "a", "weight": 1, "name": "x"}) == \
        spec_key({"name": "x", "weight": 9, "domain": "a"})
    assert spec_key({"domain": "a", "name": "x"}) != spec_key({"domain": "a", "name": "y"})
