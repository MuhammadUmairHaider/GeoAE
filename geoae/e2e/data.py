"""
Data loading for end-to-end KL training.

Wraps the base `ActivationBuffer` (which provides the normalised last-token AE
input plus the per-dimension mean/std and the train/val split) and aligns it,
index-for-index, with the precomputed teacher logits produced by
`extract_e2e.py`. For intermediate-layer targets it also serves the LEFT-padded
`input_ids` / `attention_mask` needed for the on-the-fly spliced forward.

Alignment contract
-------------------
`extract_e2e.py` writes its caches in the SAME row order as the existing
`layer_<L>.npy` activations (which `ActivationBuffer` indexes). The train/val
split is therefore identical: a contiguous tail of size `val_frac` is held out.
Every item i maps to the same global document for x, teacher, and input_ids.
"""
from __future__ import annotations

from pathlib import Path


import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from geoae.data import ActivationBuffer


def teacher_logits_path(activations_dir: str | Path, layer: int) -> Path:
    return Path(activations_dir) / f"teacher_logits_layer{layer}.npy"


def input_ids_path(activations_dir: str | Path, layer: int) -> Path:
    return Path(activations_dir) / f"input_ids_layer{layer}.npy"


def attn_mask_path(activations_dir: str | Path, layer: int) -> Path:
    return Path(activations_dir) / f"attention_mask_layer{layer}.npy"


class E2EBuffer(Dataset):
    """
    Returns a dict per item:
        {"x": (D,) float32 normalised last-token activation,
         "teacher": (V,) float32 original last-token logits,
         # only when needs_input_ids:
         "input_ids": (T,) int64 LEFT-padded,
         "attn": (T,) int64 attention mask}

    The teacher logits are stored fp16 on disk and upcast to float32 here.
    """

    def __init__(
        self,
        activations_dir: str | Path,
        layer: int,
        val_frac: float = 0.05,
        split: str = "train",
        needs_input_ids: bool = False,
        norm_cache: str | Path | None = None,
        load_teacher: bool = True,
        max_train_rows: int | None = None,
    ):
        self.activations_dir = Path(activations_dir)
        self.layer = layer
        self.needs_input_ids = needs_input_ids
        # When False (teacher_mode="onfly"), the teacher logits are recomputed in
        # the training loop as head(norm(x)) — no multi-TB cache on disk.
        self.load_teacher = load_teacher

        # Base buffer: normalised AE input + mean/std + split indices.
        self.act = ActivationBuffer(
            activations_dir, layer, val_frac=val_frac, split=split,
            norm_cache=norm_cache, max_train_rows=max_train_rows,
        )
        self._indices = self.act._indices  # global row indices for this split
        self.mean = self.act.mean
        self.std = self.act.std

        self._teacher = None
        if load_teacher:
            tl_path = teacher_logits_path(activations_dir, layer)
            if not tl_path.exists():
                raise FileNotFoundError(
                    f"Teacher logits not found: {tl_path}\n"
                    f"Run: python e2e/extract_e2e.py --config <e2e config>\n"
                    f"(or set train.teacher_mode: onfly to skip caching — last layer only)"
                )
            self._teacher = np.load(str(tl_path), mmap_mode="r")  # (N, V) fp16
            if self._teacher.shape[0] != self.act._mmap.shape[0]:
                raise ValueError(
                    f"Teacher rows ({self._teacher.shape[0]}) != activation rows "
                    f"({self.act._mmap.shape[0]}); caches are misaligned."
                )

        if needs_input_ids:
            ids_path = input_ids_path(activations_dir, layer)
            am_path = attn_mask_path(activations_dir, layer)
            for p in (ids_path, am_path):
                if not p.exists():
                    raise FileNotFoundError(
                        f"Intermediate-layer target needs {p.name}; re-run "
                        f"extract_e2e.py (it caches input_ids only for "
                        f"non-final layers)."
                    )
            self._input_ids = np.load(str(ids_path), mmap_mode="r")     # (N, T) int32
            self._attn = np.load(str(am_path), mmap_mode="r")           # (N, T) int8

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        gi = int(self._indices[i])
        item = {"x": self.act[i]}  # already normalised float32 (D,)
        if self.load_teacher:
            item["teacher"] = torch.from_numpy(self._teacher[gi].astype(np.float32))
        if self.needs_input_ids:
            item["input_ids"] = torch.from_numpy(self._input_ids[gi].astype(np.int64))
            item["attn"] = torch.from_numpy(self._attn[gi].astype(np.int64))
        return item


def make_loader(
    buffer: E2EBuffer,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Default-collate handles the dict of fixed-shape tensors. A seeded
    generator + worker_init_fn make the shuffle order and worker RNG
    reproducible across runs."""
    from geoae.seeding import seed_worker, make_generator
    return DataLoader(
        buffer,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
