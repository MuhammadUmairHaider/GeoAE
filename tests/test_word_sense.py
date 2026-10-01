"""Word-sense tool: parsing, target-token selection and metrics (CPU, no model, no network)."""
import json

import numpy as np
import pytest
import torch

from geoae.interp import word_sense as ws


def test_coarsewsd_span_and_parse():
    idx, sent = ws.parse_coarsewsd_line("9\tduring the year 1990 , ronnie leibowitz robbed 21 bank branches .")
    assert idx == 9
    start, end, surface = ws.coarsewsd_span(sent, idx)
    assert surface == "bank" and sent[start:end] == "bank"
    assert ws.coarsewsd_span("bank of the river", 0)[:2] == (0, 4)
    assert ws.coarsewsd_span(", banks .", 1)[2] == "banks"
    assert ws.coarsewsd_span("two words", 5) is None
    with pytest.raises(ValueError):
        ws.parse_coarsewsd_line("no tab here")


def test_load_coarsewsd_drops_wrong_index(tmp_path):
    d = tmp_path / "CoarseWSD-20" / "bank"
    d.mkdir(parents=True)
    (d / "classes_map.txt").write_text(json.dumps({"0": "bank_bank", "1": "bank_bank_(geography)"}))
    (d / "train.data.txt").write_text("1\tthe bank lent money\n0\tthe banks flooded\n9\tout of range\n")
    (d / "train.gold.txt").write_text("0\n1\n0\n")
    (d / "test.data.txt").write_text("2\tsat on the bank\n")   # index 2 -> 'the': wrong word
    (d / "test.gold.txt").write_text("1\n")
    rows, st = ws.load_coarsewsd(tmp_path, ["bank"])
    assert [r["surface"] for r in rows] == ["bank"]             # 'the' (index 0) and 'the' dropped
    assert st["bank"]["out_of_range"] == 1 and st["bank"]["form_mismatch"] == 2


def test_last_overlapping_token():
    # BOS (0,0), "He", " sat", " on", " the", " bank", "s", "."   for text "He sat on the banks."
    offs = [(0, 0), (0, 2), (2, 6), (6, 9), (9, 13), (13, 18), (18, 19), (19, 20)]
    assert ws.last_overlapping_token(offs, 14, 19) == 6           # multi-token word -> last piece
    assert ws.last_overlapping_token(offs, 0, 2) == 1             # BOS never counts
    assert ws.last_overlapping_token(offs, 20, 20) is None
    assert ws.last_overlapping_token(offs, 3, 6) == 2


class _FakeTok:
    """Whitespace tokenizer with a BOS, offsets like a fast tokenizer's (leading space kept)."""
    def __call__(self, text, return_offsets_mapping=False):
        ids, offs, i = [1], [(0, 0)], 0
        for w in text.split(" "):
            if w:
                s = text.index(w, i)
                offs.append((max(s - 1, 0) if s else 0, s + len(w)))
                ids.append(100 + len(ids))
                i = s + len(w)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


def test_locate_targets_prefix_and_skip():
    rows = [dict(text="Run fast now", start=0, end=3), dict(text="a b c d e run", start=10, end=13)]
    ws.locate_targets(rows, _FakeTok(), skip_leading=4, prefix="")
    assert rows[0]["keep"] is False and rows[0]["pos"] == 1       # too early without a prefix
    assert rows[1]["keep"] is True and rows[1]["pos"] == 6 and rows[1]["ctx"] == 5
    ws.locate_targets(rows, _FakeTok(), skip_leading=4, prefix="Here is a sentence: ")
    assert rows[0]["keep"] is True and rows[0]["ctx"] == 0        # prefix moves it past skip


def test_cnmi_perfect_and_random():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 200)
    assert ws.cnmi(y, y * 7 + 3, 10, rng) == pytest.approx(1.0, abs=1e-6)
    c = rng.integers(0, 40, len(y))
    assert abs(ws.cnmi(y, c, 20, rng)) < 0.1


def test_partition_stats_and_knn():
    y = np.array([0, 0, 0, 1, 1, 1])
    c = np.array([5, 5, 6, 6, 6, 6])
    st = ws.partition_stats(y, c)
    assert st["n_clusters"] == 2 and st["top_cluster_share"] == pytest.approx(4 / 6)
    assert st["purity"] == pytest.approx(5 / 6)
    R = torch.tensor([[1.0, 0]] * 12 + [[0, 1.0]] * 12) + 0.01 * torch.randn(24, 2)
    assert ws.knn_acc(R, np.repeat([0, 1], 12), k=5) == 1.0


def test_wic_metrics():
    y = np.array([1, 1, 0, 0])
    m = ws.comembership(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5]), y)
    assert m["p_same_cluster_same_sense"] == 1.0 and m["p_same_cluster_diff_sense"] == 0.5
    assert m["gap"] == 0.5
    assert ws.auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 1.0
    assert ws.best_threshold(np.array([0.9, 0.8, 0.2, 0.1]), y) == pytest.approx(0.8)


def test_basic_ci_contains_estimate():
    rng = np.random.default_rng(0)
    lo, hi = ws.basic_ci(0.3, 0.3 + rng.normal(0, 0.05, 2000))
    assert lo < 0.3 < hi and hi - lo == pytest.approx(0.196, abs=0.03)


def test_cap_per_sense():
    rows = [dict(word="w", sense=s) for s in (0,) * 10 + (1,) * 3]
    out = ws.cap_per_sense(rows, 4, np.random.default_rng(0))
    assert sum(r["sense"] == 0 for r in out) == 4 and sum(r["sense"] == 1 for r in out) == 3
