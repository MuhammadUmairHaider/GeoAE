"""Verify a legacy BiasBios activation cache against local source Arrow files.

Reconstruct the extraction shuffle, check label alignment and finite activations,
and export text hashes so train/validation/test can be deduplicated. No LM runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset


def audit(act_dir, source_cache):
    act_dir, source_cache = Path(act_dir), Path(source_cache)
    meta = json.loads((act_dir / "meta.json").read_text())
    texts, report = {}, {"activation_metadata": meta, "source_cache": str(source_cache)}
    for split in ["train", "test"]:
        n = meta[f"n_{split}"]
        dataset = Dataset.from_file(str(source_cache / f"bias_in_bios-{split}.arrow"))
        dataset = dataset.shuffle(seed=meta["seed"]).select(range(n))
        for field in ["profession", "gender"]:
            labels = np.load(act_dir / f"labels_{field}_{split}.npy")
            if not np.array_equal(labels, np.asarray(dataset[field])):
                raise ValueError(f"{split} {field} labels do not match reconstructed extraction order")
        if not np.array_equal(np.load(act_dir / f"labels_{split}.npy"), np.asarray(dataset["profession"])):
            raise ValueError("profession alias labels mismatch")
        suffix = "" if split == "train" else "_test"
        X = np.load(act_dir / f"layer_{meta['layer']}{suffix}.npy", mmap_mode="r")
        if X.ndim != 2 or len(X) != n or not np.isfinite(X).all():
            raise ValueError(f"{split}: non-finite or misaligned activations")
        texts[split] = np.asarray([hashlib.sha256(t.strip().encode()).hexdigest() for t in dataset["hard_text"]])
        report[split] = {"n": n, "shape": list(X.shape), "finite": True,
                         "labels_match_cached_source": True, "unique_texts": len(set(texts[split]))}
    report["cross_split_shared_texts"] = len(set(texts["train"]) & set(texts["test"]))
    return report, texts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act_dir", required=True)
    ap.add_argument("--source_cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hashes", required=True)
    args = ap.parse_args()
    out, hashes = Path(args.out), Path(args.hashes)
    if out.exists() or hashes.exists(): ap.error("choose new output paths; audit outputs already exist")
    if hashes.suffix != ".npz": ap.error("--hashes must end in .npz")
    report, texts = audit(args.act_dir, args.source_cache)
    for path in [out, hashes]: path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(hashes, **texts)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
