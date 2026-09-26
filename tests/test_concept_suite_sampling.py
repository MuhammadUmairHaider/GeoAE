"""Sequence sampling regressions; no Hub access or language-model loading."""
import pytest
from datasets import Dataset

from geoae.interp import concept_suite


def test_class_sorted_split_samples_across_all_blocks(monkeypatch):
    # The former 20k-buffer stream could not leave the first 40k-row block.
    ds = Dataset.from_dict({
        "text": [f"row {i}" for i in range(120_000)],
        "label": [i // 40_000 for i in range(120_000)],
    })
    monkeypatch.setattr(concept_suite, "load_dataset", lambda *a, **kw: ds)
    rows = concept_suite.sample_sequence_rows("fixture", None, "train", "text", "label", 300)
    assert {r["label"] for r in rows} == {0, 1, 2}
    assert len({r["text"] for r in rows}) == 300
    assert rows == concept_suite.sample_sequence_rows("fixture", None, "train", "text", "label", 300)


def test_empty_texts_are_skipped_and_single_class_is_rejected(monkeypatch):
    ds = Dataset.from_dict({"text": ["", None, "a", "b"], "label": [0, 1, 0, 1]})
    monkeypatch.setattr(concept_suite, "load_dataset", lambda *a, **kw: ds)
    rows = concept_suite.sample_sequence_rows("fixture", None, "train", "text", "label", 2)
    assert {r["text"] for r in rows} == {"a", "b"}
    with pytest.raises(ValueError, match="expected 3"):
        concept_suite.sample_sequence_rows("fixture", None, "train", "text", "label", 3)
    with pytest.raises(ValueError, match="fewer than two classes"):
        concept_suite.sample_sequence_rows("fixture", None, "train", "text", "label", 1)
