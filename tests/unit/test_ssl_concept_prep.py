"""Freshness and causal text contracts for the focused SSL experiment."""
import json

import numpy as np
import pytest

import experiments.supervised.analysis.ssl_concept_prep as prep
import geoae.analysis.atlas_siblings as AS


PAIR = {"pair_id": "cooking", "a_ids": [2401], "b_ids": [2404, 2405],
        "a_name": "Cooking implements", "b_name": "Cooking techniques", "kind": "atlas"}


def row(doc, label=None, text=None, chunk=0):
    return {"doc_int_id": doc, "chunk_id": chunk,
            "chunk_text": text or f"Document {doc} has some independently sourced words for concept evaluation.",
            "content_ids": [] if label is None else [label]}


def discovery(source):
    docs = np.array([r["doc_int_id"] for r in source], dtype=np.int64)
    chunks = np.array([r["chunk_id"] for r in source], dtype=np.int64)
    return prep.Discovery(docs, chunks, AS.stream_fingerprint(docs, chunks))


def test_document_split_is_stable_and_shared_across_pairs():
    assert prep.document_split("guidelabs/fineweb-atlas:doc_int_id:19") == prep.document_split(
        "guidelabs/fineweb-atlas:doc_int_id:19")
    splits = [prep.document_split(f"guidelabs/fineweb-atlas:doc_int_id:{i}") for i in range(200)]
    assert set(splits) == set(prep.SPLITS)
    assert 70 < splits.count("fit") < 130


def test_whole_discovery_document_and_duplicate_text_are_excluded():
    prefix = [row(10, 2401, "A sufficiently long source passage containing discovery-only text.")]
    source = prefix + [row(10, 2404, chunk=1),
                       row(11, 2401, " a sufficiently LONG source passage containing discovery-only text. "),
                       row(12, 2401), row(13, 2404)]
    kept, stats = prep.collect_records(source, discovery(prefix), [PAIR], start_index=1,
                                      caps={s: 20 for s in prep.SPLITS}, control_cap=0)
    assert {r["doc_int_id"] for r in kept} == {12, 13}
    assert stats["rejections"]["prefix_document"] == 1
    assert stats["rejections"]["discovery_or_skipped_prefix_duplicate_text"] == 1
    assert stats["validated_discovery_n_chunks"] == 1


def test_prefix_beyond_cache_also_excludes_all_its_documents():
    prefix = [row(10, 2401)]
    source = prefix + [row(11, 2401), row(11, 2404, chunk=1), row(12, 2401)]
    kept, stats = prep.collect_records(source, discovery(prefix), [PAIR], start_index=2,
                                      caps={s: 20 for s in prep.SPLITS}, control_cap=0)
    assert {r["doc_int_id"] for r in kept} == {12}
    assert stats["excluded_prefix_documents"] == 2


def test_discovery_order_mismatch_and_short_source_fail_closed():
    prefix = [row(10), row(11)]
    with pytest.raises(ValueError, match="differs from discovery"):
        prep.collect_records([row(11), row(10)], discovery(prefix), [PAIR], start_index=2)
    with pytest.raises(ValueError, match="before freshness boundary"):
        prep.collect_records(prefix[:1], discovery(prefix), [PAIR], start_index=2)


def test_exclusive_labels_document_caps_and_shared_random_pool():
    prefix = [row(0)]
    ambiguous = row(100)
    ambiguous["content_ids"] = [2401, 2404]
    source = prefix + [ambiguous, row(101, 2401), row(101, 2404, chunk=1), row(102, 2405)]
    source += [row(i) for i in range(200, 260)]
    second = dict(PAIR, pair_id="repeat_cooking")
    kept, stats = prep.collect_records(source, discovery(prefix), [PAIR, second], start_index=1,
                                      caps={s: 1 for s in prep.SPLITS}, control_cap=2)
    target = [r for r in kept if r["pair_id"] == "cooking" and r["label"] >= 0]
    assert [r["doc_int_id"] for r in target] == [101, 102]
    assert len([r for r in target if r["doc_int_id"] == 101]) == 1
    assert all(r["doc_int_id"] != 100 for r in kept)
    controls = {p: [r for r in kept if r["pair_id"] == p and r["label"] == -1]
                for p in ("cooking", "repeat_cooking")}
    assert len(controls["cooking"]) == 6
    assert [(r["source_doc_id"], r["split"]) for r in controls["cooking"]] == [
        (r["source_doc_id"], r["split"]) for r in controls["repeat_cooking"]]
    assert stats["rejections"]["both_labels:cooking"] == 1


def test_duplicate_fresh_text_cannot_cross_document_splits():
    prefix = [row(0)]
    source = prefix + [row(10, 2401, "A completely new and sufficiently long cooking implements passage."),
                       row(11, 2404, "a completely NEW and sufficiently long cooking implements passage.")]
    kept, stats = prep.collect_records(source, discovery(prefix), [PAIR], start_index=1,
                                      caps={s: 20 for s in prep.SPLITS}, control_cap=0)
    assert len(kept) == 1
    assert stats["rejections"]["fresh_duplicate_text"] == 1


def test_native_preserves_full_source_suffix_and_actual_anchor():
    native = prep.normalize_pairs({"pairs": [prep.NATIVE_PAIR]})[0]
    prefix = [row(0)]
    text = "A real source paragraph with sufficient preceding context.\nReferences\nCitation one follows."
    external = "Another actual source paragraph with useful prior context.\nExternal links\nOfficial site follows."
    source = prefix + [row(10, text=text), row(11, text=external)]
    kept, _ = prep.collect_records(source, discovery(prefix), [native], start_index=1,
                                   caps={s: 20 for s in prep.SPLITS}, control_cap=0)
    assert len(kept) == 2
    assert kept[0]["text"] == text
    assert kept[0]["causal_prefix_text"].endswith("References\n")
    assert text[:kept[0]["anchor_char_end"]] == kept[0]["causal_prefix_text"]
    assert text[kept[0]["anchor_char_end"]:] == "Citation one follows."
    assert [r["label"] for r in kept] == [0, 1]


def test_native_rejects_both_headers_no_suffix_and_repeated_prefix():
    native = prep.normalize_pairs({"pairs": [prep.NATIVE_PAIR]})[0]
    prefix = [row(0)]
    context = "An actual paragraph with sufficiently long shared preceding context.\n"
    source = prefix + [row(10, text=context + "References\nCitation follows."),
                       row(11, text=context + "References\nOther citation follows."),
                       row(12, text=context + "References\n"),
                       row(13, text=context + "References\nX\nExternal links\nY")]
    kept, stats = prep.collect_records(source, discovery(prefix), [native], start_index=1,
                                      caps={s: 20 for s in prep.SPLITS}, control_cap=0)
    assert len(kept) == 1
    assert stats["rejections"]["native_duplicate_prefix"] == 1
    assert stats["rejections"]["native_anchor_without_true_next_text"] == 1


def test_cache_self_fingerprint_is_verified(tmp_path):
    path = tmp_path / "chunks.npz"
    np.savez(path, doc_int_id=np.array([1]), chunk_id=np.array([0]), fingerprint="wrong")
    with pytest.raises(ValueError, match="own document/chunk arrays"):
        prep.Discovery.load(path)
    with pytest.raises(FileNotFoundError, match="freshness cannot be verified"):
        prep.Discovery.load(tmp_path / "missing")


def test_jsonl_source_filters_status_and_obeys_bound(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in [dict(row(1), chunk_status="bad"), row(2), row(3)]))
    assert [r["doc_int_id"] for r in prep.source_records(tmp_path, 1, source_jsonl=source)] == [2]


def test_parquet_list_columns_use_top_level_arrow_schema(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Parquet's leaf schema calls nested list entries "element"; the Atlas
    # producer must validate top-level content_ids instead.
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist([
        {**row(1, 2401), "chunk_status": "ok"},
        {**row(2, 2405), "chunk_status": "ok"},
    ]), path)
    monkeypatch.setattr(AS, "shard_file", lambda _directory, _index: path)
    records = list(prep.source_records(tmp_path, 2))
    assert [r["content_ids"] for r in records] == [[2401], [2405]]
