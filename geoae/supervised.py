"""
In-batch supervision for a short fine-tune of an already-trained GeoAE.

WHY THIS IS SEPARATE FROM `centroid_init: seeded`. Seeding places centroids on
labelled class means once, at epoch 11, and `reinit_mode: anchor` spends the
labelled pool on dead clusters in the first few hundred steps. After that the
labels are gone. This module keeps a supervised term in the loss for the whole
(short) fine-tune, so the labels shape the ENCODER rather than only the initial
codebook.

WHAT IT OPTIMISES. Supervised contrastive loss on the latent: rows sharing a
label are pulled together, rows with different labels pushed apart. It does NOT
bind classes to particular centroids — with ~500 classes over K=2000 that
allocation would fight Sinkhorn balancing, and the question here is whether the
REPRESENTATION can be improved, not whether the codebook can be re-indexed.
(The codebook-only question is answered without training, by re-fitting balanced
k-means on the frozen latent: `fit_balanced_kmeans --space latent`.)

TWO DESIGN CHOICES WORTH KNOWING.

1. Supervised rows are ADDED to the step, not swapped into the unlabelled batch.
   The unsupervised stream keeps its full batch, so the recon/cluster/sep/var
   terms and the Sinkhorn marginals see exactly what they saw before the
   fine-tune. `sup_frac` is therefore the size of the extra rows relative to the
   batch, and the Sinkhorn balancing is never computed over labelled rows —
   they are a different corpus and would bias the codebook's mass toward it.

2. The labelled forward runs with BatchNorm in EVAL mode. The labelled pool is
   not iid with the 10M activation dump; letting it update the BN running
   statistics would shift the normalisation the whole model depends on. The
   supervised gradient still reaches the encoder weights.

HOLDOUTS. `holdout_classes` removes whole classes from training so that probe
NMI on them measures generalisation rather than memorisation, and
`holdout_rows` removes a fraction of rows inside the trained classes. The
manifest of what was held out is written next to the checkpoints. The cleanest
transfer measure needs no holdout at all: supervise on a subset of rungs and
read the untouched rungs off the normal probe run.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def bn_eval(module: nn.Module):
    """Run a forward with every BatchNorm in eval mode, then restore."""
    flipped = [m for m in module.modules()
               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and m.training]
    for m in flipped:
        m.eval()
    try:
        yield
    finally:
        for m in flipped:
            m.train()


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised contrastive loss (Khosla et al. 2020), single view.

    Every row with at least one same-label partner in the batch contributes;
    rows whose class happens to be alone are dropped rather than counted as a
    zero, which would silently scale the loss down by the singleton fraction.
    """
    z = F.normalize(z.float(), dim=1)
    sim = (z @ z.T) / temperature
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))            # no self-similarity
    pos = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # The diagonal is -inf by construction and `pos` is False there, so the
    # masked product would evaluate 0 * -inf = NaN. Zero it before multiplying.
    log_prob = log_prob.masked_fill(eye, 0.0)
    n_pos = pos.sum(1)
    keep = n_pos > 0
    if not bool(keep.any()):
        return z.new_zeros(())
    per_row = -(pos * log_prob).sum(1)[keep] / n_pos[keep]
    return per_row.mean()


class LabelledPool:
    """
    Labelled activations from the concept cache, class-balanced sampling.

    Rows are held as RAW activations (normalisation happens per batch with the
    trainer's own mean/std) so this pool is valid for any checkpoint trained on
    the same dump.
    """

    def __init__(self, cache_dir: str, rungs: list[str] | None, per_class: int,
                 holdout_classes: float, holdout_rows: float, seed: int,
                 min_examples: int = 10, verbose: bool = True):
        from geoae.interp.concept_probe import LADDER, load_rung

        rng = np.random.default_rng(seed)
        wanted = set(rungs) if rungs else None
        chunks, labels, names, held = [], [], [], {"classes": [], "rows": {}}
        next_id = 0
        for rung, fname, key, *_ in LADDER:
            if wanted is not None and rung not in wanted:
                continue
            if not (Path(cache_dir) / fname).exists():
                continue
            H, y = load_rung(cache_dir, fname, key, 0)
            classes = [c for c in np.unique(y) if int((y == c).sum()) >= min_examples]
            if len(classes) < 2:
                del H, y
                continue
            order = rng.permutation(len(classes))
            n_hold = int(round(holdout_classes * len(classes)))
            hold_cls = {classes[i] for i in order[:n_hold]}
            for c in classes:
                if c in hold_cls:
                    held["classes"].append(f"{rung}::{c}")
                    continue
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                n_hold_rows = int(round(holdout_rows * len(idx)))
                hold_idx, use_idx = idx[:n_hold_rows], idx[n_hold_rows:][:per_class]
                if len(use_idx) < 2:
                    continue
                held["rows"][f"{rung}::{c}"] = hold_idx.tolist()
                chunks.append(np.asarray(H[use_idx], dtype=np.float32))
                labels.append(np.full(len(use_idx), next_id, dtype=np.int64))
                names.append(f"{rung}::{c}")
                next_id += 1
            del H, y

        if not chunks:
            raise SystemExit("[sup] no labelled classes found — check sup_rungs / sup_cache")
        self.H = torch.from_numpy(np.concatenate(chunks))
        self.y = torch.from_numpy(np.concatenate(labels))
        self.names = names
        self.held = held
        self.by_class = [torch.nonzero(self.y == k, as_tuple=True)[0] for k in range(next_id)]
        self._g = torch.Generator().manual_seed(seed)
        if verbose:
            gb = self.H.numel() * 4 / 2**30
            print(f"[sup] labelled pool: {len(self.H):,} rows, {next_id} classes "
                  f"({gb:.2f} GB) from {len(set(n.split('::')[0] for n in names))} rungs; "
                  f"held out {len(held['classes'])} whole classes")

    def __len__(self) -> int:
        return len(self.H)

    def save_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"n_rows": len(self.H), "classes_trained": self.names,
             "classes_held_out": self.held["classes"],
             "rows_held_out": {k: len(v) for k, v in self.held["rows"].items()}}, indent=1))

    def batch(self, n_rows: int, m_per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Class-balanced draw: n_rows // m_per_class classes, m_per_class rows each,
        so every row has same-class partners for the contrastive term.
        """
        m = max(2, m_per_class)
        n_cls = max(2, n_rows // m)
        pick = torch.randperm(len(self.by_class), generator=self._g)[:n_cls]
        rows = []
        for k in pick.tolist():
            idx = self.by_class[k]
            sel = idx[torch.randint(len(idx), (m,), generator=self._g)] if len(idx) < m \
                else idx[torch.randperm(len(idx), generator=self._g)[:m]]
            rows.append(sel)
        rows = torch.cat(rows)
        return self.H[rows], self.y[rows]
