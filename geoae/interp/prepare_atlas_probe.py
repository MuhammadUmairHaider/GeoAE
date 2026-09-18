"""Prepare a held-out FineWeb-Atlas multi-label concept-probe cache.

The seeded-Atlas checkpoint consumed a deterministic subset of chunk rows as
centroid anchors.  This utility reproduces that exact selection: anchor rows are
the probe-training set, while rows never selected as an anchor are divided into
validation and test sets.  Labels are restricted to the document/tone/content
concepts used by the seeded checkpoint and must have adequate support in both
the training and held-out pools.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from geoae.seeded_init import load_atlas_anchors


FIELDS = ("document_ids", "tone_ids", "content_ids")


def _row_sets(rows: np.ndarray) -> list[set[int]]:
    return [set(np.asarray(row).tolist()) for row in rows]


def _multihot(
    row_indices: np.ndarray,
    concepts: list[tuple[str, int]],
    memberships: dict[str, list[set[int]]],
) -> np.ndarray:
    labels = np.zeros((len(row_indices), len(concepts)), dtype=np.uint8)
    for out_row, source_row in enumerate(row_indices):
        for col, (field, concept_id) in enumerate(concepts):
            labels[out_row, col] = concept_id in memberships[field][source_row]
    return labels


def _balanced_holdout_split(
    labels: np.ndarray,
    seed: int,
    attempts: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Find a deterministic half split with good per-label positive coverage."""
    n_val = len(labels) // 2
    best = None
    for offset in range(attempts):
        rng = np.random.RandomState(seed + offset)
        perm = rng.permutation(len(labels))
        val, test = perm[:n_val], perm[n_val:]
        val_n, test_n = labels[val].sum(0), labels[test].sum(0)
        min_support = int(np.minimum(val_n, test_n).min())
        imbalance = int(np.abs(val_n.astype(int) - test_n.astype(int)).sum())
        score = (min_support, -imbalance)
        if best is None or score > best[0]:
            best = (score, val, test, seed + offset, val_n, test_n)
    assert best is not None
    _, val, test, chosen_seed, val_n, test_n = best
    return val, test, {
        "search_seed": chosen_seed,
        "minimum_validation_positives": int(val_n.min()),
        "minimum_test_positives": int(test_n.min()),
        "maximum_validation_positives": int(val_n.max()),
        "maximum_test_positives": int(test_n.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", default="cache/atlas8k_last.npz")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--anchor_per_class", type=int, default=25)
    parser.add_argument("--atlas_min_examples", type=int, default=25)
    parser.add_argument("--anchor_seed", type=int, default=0)
    parser.add_argument("--min_train_support", type=int, default=25)
    parser.add_argument("--min_holdout_support", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_attempts", type=int, default=1000)
    args = parser.parse_args()

    source = np.load(args.atlas, allow_pickle=True)
    hidden = np.asarray(source["H_last"], dtype=np.float32)
    field_rows = {field: source[field] for field in FIELDS}
    memberships = {field: _row_sets(rows) for field, rows in field_rows.items()}

    _, anchor_keys, anchor_rows = load_atlas_anchors(
        args.atlas,
        per_class=args.anchor_per_class,
        min_examples=args.atlas_min_examples,
        fields=FIELDS,
        seed=args.anchor_seed,
    )
    if anchor_rows is None:
        raise RuntimeError("Atlas anchor selection returned no rows")
    anchor_rows = np.asarray(anchor_rows, dtype=np.int64)
    anchor_mask = np.zeros(len(hidden), dtype=bool)
    anchor_mask[anchor_rows] = True
    holdout_rows = np.flatnonzero(~anchor_mask)

    candidates: list[tuple[str, int]] = []
    for field in FIELDS:
        counts = Counter(
            concept_id
            for row in field_rows[field]
            for concept_id in np.asarray(row).tolist()
        )
        for concept_id, total_support in counts.items():
            # This is the same eligibility threshold used by Atlas seeding.
            if total_support < args.atlas_min_examples:
                continue
            train_support = sum(
                concept_id in memberships[field][row] for row in anchor_rows
            )
            holdout_support = sum(
                concept_id in memberships[field][row] for row in holdout_rows
            )
            if (
                train_support >= args.min_train_support
                and holdout_support >= args.min_holdout_support
            ):
                candidates.append((field, int(concept_id)))
    candidates.sort()

    all_holdout_labels = _multihot(holdout_rows, candidates, memberships)
    val_local, test_local, split_stats = _balanced_holdout_split(
        all_holdout_labels, args.split_seed, args.split_attempts
    )
    val_rows, test_rows = holdout_rows[val_local], holdout_rows[test_local]
    split_rows = {
        "train": anchor_rows,
        "validation": val_rows,
        "test": test_rows,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        suffix = "" if split == "train" else f"_{split}"
        np.save(out_dir / f"layer_{args.layer}{suffix}.npy", hidden[rows])
        np.save(
            out_dir / f"labels_{split}.npy",
            _multihot(rows, candidates, memberships),
        )
        np.save(out_dir / f"source_rows_{split}.npy", rows)

    by_field = Counter(field.replace("_ids", "") for field, _ in candidates)
    meta = {
        "source": args.atlas,
        "layer": args.layer,
        "representation": "chunk last-token residual",
        "fields": list(FIELDS),
        "multi_label": True,
        "concepts": [
            {"field": field.replace("_ids", ""), "concept_id": concept_id}
            for field, concept_id in candidates
        ],
        "n_concepts": len(candidates),
        "concepts_by_field": dict(by_field),
        "split_sizes": {split: len(rows) for split, rows in split_rows.items()},
        "split_policy": (
            "exact seeded-anchor chunks for train; exact non-anchor chunks split "
            "equally into validation/test"
        ),
        "anchor_selection": {
            "per_class": args.anchor_per_class,
            "minimum_examples": args.atlas_min_examples,
            "seed": args.anchor_seed,
            "n_anchor_assignments": len(anchor_keys),
            "n_unique_anchor_chunks": len(anchor_rows),
            "n_non_anchor_chunks": len(holdout_rows),
        },
        "label_filter": {
            "minimum_train_support": args.min_train_support,
            "minimum_combined_holdout_support": args.min_holdout_support,
        },
        "holdout_split": split_stats,
    }
    with (out_dir / "meta.json").open("w") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
