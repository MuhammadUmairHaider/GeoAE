"""Unit tests for data.py using temporary dummy .npy files."""
import tempfile
from pathlib import Path


import numpy as np
import torch
from geoae.data import ActivationBuffer, ShuffledActivationLoader

HIDDEN = 32
N_TOKENS = 1000
LAYER = 27
VAL_FRAC = 0.1


def make_dummy_activations(tmpdir: Path) -> np.ndarray:
    """Write a deterministic fake activation file and return the array."""
    rng = np.random.RandomState(0)
    data = rng.randn(N_TOKENS, HIDDEN).astype(np.float16)
    npy_path = tmpdir / f"layer_{LAYER}.npy"
    np.save(str(npy_path), data)
    return data


def test_split_sizes():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        make_dummy_activations(tmpdir)

        train_buf = ActivationBuffer(tmpdir, LAYER, val_frac=VAL_FRAC, split="train")
        val_buf = ActivationBuffer(tmpdir, LAYER, val_frac=VAL_FRAC, split="val",
                                   norm_cache=tmpdir / f"norm_params_layer{LAYER}.npz")

        expected_val = int(N_TOKENS * VAL_FRAC)
        expected_train = N_TOKENS - expected_val
        assert len(train_buf) == expected_train, f"train: {len(train_buf)} != {expected_train}"
        assert len(val_buf) == expected_val, f"val: {len(val_buf)} != {expected_val}"


def test_getitem_shape_and_dtype():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        make_dummy_activations(tmpdir)
        buf = ActivationBuffer(tmpdir, LAYER, val_frac=VAL_FRAC, split="train")

        item = buf[0]
        assert item.shape == (HIDDEN,)
        assert item.dtype == torch.float32


def test_normalisation_applied():
    """Verify that returned tokens are approximately zero-mean unit-std."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # Create data with a large mean and std so raw ≠ normalised
        rng = np.random.RandomState(1)
        data = (rng.randn(N_TOKENS, HIDDEN) * 10 + 50).astype(np.float16)
        npy_path = tmpdir / f"layer_{LAYER}.npy"
        np.save(str(npy_path), data)

        buf = ActivationBuffer(tmpdir, LAYER, val_frac=VAL_FRAC, split="train")

        # Gather all train tokens
        all_items = torch.stack([buf[i] for i in range(len(buf))])
        # After normalisation the per-dim mean should be ~0 and std ~1
        assert all_items.mean().abs().item() < 0.5, "mean after normalisation should be near 0"
        assert abs(all_items.std().item() - 1.0) < 0.5, "std after normalisation should be near 1"


def test_norm_cache_is_reused(tmp_path):
    """Second instantiation should load cached norm params without recomputing."""
    make_dummy_activations(tmp_path)

    buf1 = ActivationBuffer(tmp_path, LAYER, val_frac=VAL_FRAC, split="train")
    cache_path = tmp_path / f"norm_params_layer{LAYER}.npz"
    assert cache_path.exists(), "norm cache should be written after first instantiation"
    mtime_before = cache_path.stat().st_mtime

    buf2 = ActivationBuffer(tmp_path, LAYER, val_frac=VAL_FRAC, split="train")
    mtime_after = cache_path.stat().st_mtime
    assert mtime_before == mtime_after, "norm cache should not be rewritten on second load"


def test_denormalize_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        make_dummy_activations(tmpdir)
        buf = ActivationBuffer(tmpdir, LAYER, val_frac=VAL_FRAC, split="train")

        raw_np = buf._mmap[buf._indices[0]].astype(np.float32)
        normed = buf[0]
        recovered = buf.denormalize(normed).numpy()

        # Allow for fp16 quantisation error in the stored file
        np.testing.assert_allclose(recovered, raw_np, rtol=0, atol=0.05)


def test_no_train_val_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        make_dummy_activations(tmpdir)

        train_buf = ActivationBuffer(tmpdir, LAYER, val_frac=VAL_FRAC, split="train")
        val_buf = ActivationBuffer(tmpdir, LAYER, val_frac=VAL_FRAC, split="val",
                                   norm_cache=tmpdir / f"norm_params_layer{LAYER}.npz")

        train_set = set(train_buf._indices.tolist())
        val_set = set(val_buf._indices.tolist())
        assert len(train_set & val_set) == 0, "train and val indices must not overlap"
        assert len(train_set | val_set) == N_TOKENS, "train + val should cover all tokens"


def test_shuffled_loader_batch_shape():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        make_dummy_activations(tmpdir)
        buf = ActivationBuffer(tmpdir, LAYER, val_frac=VAL_FRAC, split="train")
        loader = ShuffledActivationLoader(buf, batch_size=64, num_workers=0, pin_memory=False)

        batch = next(iter(loader))
        assert batch.shape == (64, HIDDEN)
        assert batch.dtype == torch.float32


if __name__ == "__main__":
    import traceback
    passed = failed = 0
    g = dict(globals())
    for name, fn in g.items():
        if name.startswith("test_") and callable(fn):
            try:
                fn(Path(tempfile.mkdtemp())) if "tmp_path" in fn.__code__.co_varnames else fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
