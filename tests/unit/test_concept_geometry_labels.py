"""Exercise geometry's real kNN diagnostic with cached categorical labels."""
import numpy as np
import pytest
import torch

import geoae.evaluation.concept_geometry as geometry


@pytest.mark.parametrize("labels,display_labels", [
    (np.array([2] * 40 + [9] * 40, dtype=object), ["Artist"] * 40 + ["Animal"] * 40),
    (np.array([2] * 40 + [9] * 40, dtype=np.int64), ["Artist"] * 40 + ["Animal"] * 40),
    (np.array(["company"] * 40 + ["artist"] * 40, dtype=object), ["company"] * 40 + ["artist"] * 40),
])
def test_knn_accepts_cache_labels_and_preserves_display_values(labels, display_labels, monkeypatch, tmp_path):
    hidden = np.random.RandomState(0).normal(size=(80, 8)).astype(np.float32)
    hidden[40:, 0] += 100
    monkeypatch.setattr(geometry, "load_rung", lambda *args: (hidden, labels))
    # This fixture supplies activations directly, with no on-disk token-id cache.
    monkeypatch.setattr(geometry, "load_rung_ids", lambda *args: None)
    baseline = ("km", (torch.zeros(2, 8), torch.zeros(8), torch.ones(8)))
    reps, returned_labels, classes = geometry.do_rung(
        "topic14", tmp_path, {"base_a": baseline, "base_b": baseline},
        torch.device("cpu"), tmp_path, 80, 30, 8, set(),
    )
    assert all("kNN 1.000" in name for name, _ in reps)
    np.testing.assert_array_equal(returned_labels, display_labels)
    assert set(classes) == set(display_labels)
    assert not list(tmp_path.iterdir())
