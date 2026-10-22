import hashlib
import pytest

from experiments.supervised.analysis.ssl_concept_capture import choose_positions
from experiments.supervised.analysis.ssl_concept_capture import validate_records


def record(text, **kw):
    return {"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_doc_id": "doc1", "pair_id": "p", "label": 0, "split": "fit", **kw}


def test_tail_capture_never_scores_missing_next_token():
    ids = list(range(20))
    positions, error = choose_positions(record("text"), ids, [(i, i + 1) for i in ids],
                                        newline_id=99, tail_tokens=8)
    assert error is None and positions == [15, 16, 17, 18]
    assert all(p + 1 < len(ids) for p in positions)


def test_reject_truncation_instead_of_using_unseen_passage_labels():
    positions, reason = choose_positions(record("text"), list(range(257)), [],
                                         newline_id=99, max_length=256)
    assert not positions and reason == "too_long_without_truncation"


def test_native_identity_uses_actual_token_and_observed_anchor():
    text = "context\nReferences\nactual cited work"
    end = text.index("actual")
    offsets = [(0, 0), (0, 7), (7, 8), (8, 18), (18, end), (end, end + 6), (end + 6, len(text))]
    r = record(text, kind="native", anchor_char_end=end)
    ids = [1, 2, 99, 3, 99, 4, 5]
    assert choose_positions(r, ids, offsets, newline_id=99) == ([4], None)
    ids[4] = 100  # a fused/different newline token must not be silently accepted
    assert choose_positions(r, ids, offsets, newline_id=99)[0] == []


def test_record_hash_and_split_leakage_are_rejected():
    r = record("a passage")
    validate_records([r])
    with pytest.raises(ValueError, match="multiple splits"):
        validate_records([r, {**r, "split": "test"}])
    with pytest.raises(ValueError, match="hash"):
        validate_records([{**r, "text": "changed"}])


def test_native_anchor_keeps_source_suffix_for_a_true_next_token():
    r = record("References\n", kind="native", anchor_char_end=11)
    with pytest.raises(ValueError, match="following text"):
        validate_records([r])
