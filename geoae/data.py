"""
Activation buffer and shuffled dataloader.

Reads memory-mapped .npy files produced by geoae.extract.
Normalises activations per-dimension using mean/std computed on the train split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class ActivationBuffer(Dataset):
    """
    Random-access dataset over a memory-mapped activation file.

    File layout expected:
        activations/layer_<L>.npy  — shape (N, hidden_size), dtype float16
        activations/meta.json      — extraction metadata

    Normalisation params are computed once on the train split and cached to disk.
    The buffer always returns float32 normalised tensors.
    """

    def __init__(
        self,
        activations_dir: str | Path,
        layer: int,
        val_frac: float = 0.05,
        split: str = "train",      # "train" or "val"
        norm_cache: str | Path | None = None,
        max_train_rows: int | None = None,
    ):
        self.activations_dir = Path(activations_dir)
        self.layer = layer
        self.split = split

        # Load memory-mapped file
        npy_path = self.activations_dir / f"layer_{layer}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Activation file not found: {npy_path}")

        self._mmap = np.load(str(npy_path), mmap_mode="r")
        N = self._mmap.shape[0]

        # Train/val split by index (no shuffle across split boundary)
        val_start = int(N * (1.0 - val_frac))
        if split == "train":
            self._indices = np.arange(0, val_start)
            # Escape hatch, NOT normally needed: MADV_RANDOM above already
            # removes the readahead amplification that made random access slow.
            # This only helps if the residual ~15 KB/row of real I/O still
            # saturates the device — capping the split shrinks the working set so
            # it stays resident in page cache. Costs training data, so prefer
            # leaving it unset.
            if max_train_rows is not None and max_train_rows >= len(self._indices):
                gb = len(self._indices) * self._mmap.shape[1] * self._mmap.dtype.itemsize / 1e9
                print(f"[data] ! --max_train_rows={max_train_rows:,} >= train split "
                      f"({len(self._indices):,}) — NO CAP APPLIED, working set stays "
                      f"{gb:.0f} GB. Pass a value BELOW the split size to shrink it.")
            if max_train_rows is not None and max_train_rows < len(self._indices):
                gb = max_train_rows * self._mmap.shape[1] * self._mmap.dtype.itemsize / 1e9
                print(f"[data] train split capped {len(self._indices):,} -> "
                      f"{max_train_rows:,} rows ({gb:.0f} GB working set)")
                self._indices = self._indices[:max_train_rows]
        elif split == "val":
            self._indices = np.arange(val_start, N)
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        # Normalisation: compute on train split, cache to disk
        if norm_cache is None:
            norm_cache = self.activations_dir / f"norm_params_layer{layer}.npz"
        self._norm_cache = Path(norm_cache)
        # The norm pass (cache miss only) is a SEQUENTIAL scan and is hinted as
        # such inside; MADV_RANDOM is applied after it, for the training reads.
        self.mean, self.std = self._load_or_compute_norm(val_start)
        self._advise("random")

    def _advise(self, kind: str) -> None:
        """
        Tell the kernel how this mapping will be read. Advisory only; failures
        are ignored (non-Linux, or numpy no longer exposing the raw mapping).

        "random"     — training reads one scattered 15 KB row per sample, but the
                       default readahead fetches ~2 MB around every touched page.
                       Measured on the Gemma layer-47 dump: 139-192x read
                       amplification on cold regions, which pins the GPU at 0%
                       waiting on I/O. MADV_RANDOM drops it to ~0.1x
                       (83 -> 75,232 rows/s on that file).
        "sequential" — the norm pass streams the WHOLE file in order. Under
                       MADV_RANDOM that same readahead suppression is a disaster:
                       it degrades to a page fault per page, measured at 6.3 MB/s
                       vs 913 MB/s, turning a ~5 min job into ~12 h. The two
                       access patterns need opposite hints, so each is set around
                       the code that uses it.
        """
        try:
            import mmap as _mmap
            flag = _mmap.MADV_RANDOM if kind == "random" else _mmap.MADV_SEQUENTIAL
            self._mmap._mmap.madvise(flag)
        except (AttributeError, OSError, ValueError):
            pass

    def _load_or_compute_norm(self, val_start: int) -> tuple[np.ndarray, np.ndarray]:
        if self._norm_cache.exists():
            d = np.load(str(self._norm_cache))
            return d["mean"].astype(np.float32), d["std"].astype(np.float32)

        self._advise("sequential")
        print(f"[data] Computing normalisation stats over {val_start} train tokens …")
        from tqdm import tqdm
        D = self._mmap.shape[1]
        chunk = 50_000  # rows per pass — keeps RAM under ~600 MB

        # Pass 1: mean
        acc = np.zeros(D, dtype=np.float64)
        for s in tqdm(range(0, val_start, chunk), desc="[data] Pass 1/2 (Mean)"):
            acc += self._mmap[s:min(s + chunk, val_start)].astype(np.float64).sum(axis=0)
        mean = (acc / val_start).astype(np.float32)

        # Pass 2: variance
        acc[:] = 0.0
        for s in tqdm(range(0, val_start, chunk), desc="[data] Pass 2/2 (Std)"):
            blk = self._mmap[s:min(s + chunk, val_start)].astype(np.float64)
            acc += ((blk - mean) ** 2).sum(axis=0)
        std = (np.sqrt(acc / val_start) + 1e-8).astype(np.float32)

        np.savez(str(self._norm_cache), mean=mean, std=std)
        print(f"[data] Norm stats saved to {self._norm_cache}")
        return mean, std

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Tensor:
        raw = self._mmap[self._indices[i]].astype(np.float32)
        normed = (raw - self.mean) / self.std
        return torch.from_numpy(normed)

    def denormalize(self, x: Tensor) -> Tensor:
        """Convert normalised tensor back to raw activation space."""
        mean = torch.from_numpy(self.mean).to(x.device)
        std = torch.from_numpy(self.std).to(x.device)
        return x * std + mean


class ShuffledActivationLoader:
    """
    Yields shuffled batches by sampling random indices from an ActivationBuffer.

    Decorrelates batches — consecutive tokens from one document are highly
    correlated; random sampling fixes this (standard SAE practice).
    """

    def __init__(
        self,
        buffer: ActivationBuffer,
        batch_size: int = 4096,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self._loader = DataLoader(
            buffer,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)
