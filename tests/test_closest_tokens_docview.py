"""closest_tokens --doc_view helpers: per-document cap, document spread, titles. CPU only."""
import numpy as np

from geoae.interp.closest_tokens import doc_stats, doc_title, pick_top


def test_pick_top_caps_per_document_and_dedups():
    docs = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    tok_index = [(0, 1), (0, 2), (0, 3), (1, 1), (0, 4), (1, 2)]    # doc 0 is closest
    assert pick_top(range(6), tok_index, docs, top_n=4) == [0, 1, 2, 3]            # no cap: old behaviour
    assert pick_top(range(6), tok_index, docs, top_n=4, per_doc=2) == [0, 1, 3, 5]  # doc 0 capped at 2
    dup_docs = [[5, 5, 5, 5]]                                          # same token + same left context
    assert pick_top(range(2), [(0, 2), (0, 2)], dup_docs, top_n=5) == [0]


def test_doc_stats():
    s = doc_stats([3, 3, 3, 7])
    assert s["n_docs"] == 2 and s["top_doc_share"] == 0.75
    assert 1.0 < s["eff_docs"] < 2.0
    assert doc_stats(np.array([], dtype=int))["n_docs"] == 0


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i == 0))


def test_doc_title_first_nonempty_line_truncated():
    text = "\n\n  Village in Turkey  \nbody"
    assert doc_title([ord(c) for c in text], _Tok()) == "Village in Turkey"
    assert doc_title([0] + [ord(c) for c in "Title"], _Tok()) == "Title"     # special tokens dropped
    assert doc_title([ord("a")] * 200, _Tok(), n_tokens=200, width=10) == "a" * 10 + "…"
