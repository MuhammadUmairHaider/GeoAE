"""
Extract per-document residual stream activations from BiasBios biographies.

Dataset: LabHC/bias_in_bios  (De-Arteaga et al. 2019)
  - 28 professions, binary gender (0=male, 1=female)
  - Text field: hard_text (biography, still contains gendered pronouns)

Saves dual label arrays (profession + gender) alongside activations:
  activations/{model}/{pooling}/layer_{L}.npy         (N_train, D) float16/float32
  activations/{model}/{pooling}/layer_{L}_test.npy    (N_test,  D) float16/float32
  activations/{model}/{pooling}/labels_profession_train.npy
  activations/{model}/{pooling}/labels_profession_test.npy
  activations/{model}/{pooling}/labels_gender_train.npy
  activations/{model}/{pooling}/labels_gender_test.npy
  activations/{model}/{pooling}/labels_train.npy      (profession alias)
  activations/{model}/{pooling}/labels_test.npy        (profession alias)
  activations/{model}/{pooling}/meta.json
  activations/{model}/last/tokens_{train,test}.npy   (N,) int64  id of the pooled (final) token
  activations/{model}/last/source_hashes.npz         train/test = SHA256(hard_text) hex strings
The last two are additive (token-bypass AEs need the current token id; geoae.analysis.bias_probe
and probe_perturbation read source_hashes.npz via --source_hashes for exact-text dedup).

Usage:
    python -u -m geoae.extraction.biasbios --layer 27 --n_train 50000 --n_test 20000
    python -u -m geoae.extraction.biasbios --layer 27 --pooling both
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from geoae.checkpoint import load_tokenizer

from geoae.extraction.common import output_dtype, pool, register_hook
from geoae.lm_arch import hidden_size as lm_hidden_size

PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]
GENDERS = ["male", "female"]


def extract_split(
    model,
    tokenizer,
    examples: list[dict],
    layer: int,
    max_length: int,
    poolings: list[str],
    out_dirs: dict[str, Path],
    split: str,
    log_every: int = 100,
    dtype="auto",
) -> np.ndarray:
    N = len(examples)
    D = lm_hidden_size(model)   # text decoder width (model.config is multimodal on Gemma 3)
    dtype = output_dtype(model, dtype)
    acts = {p: np.zeros((N, D), dtype=dtype) for p in poolings}
    labels_prof = np.zeros(N, dtype=np.int32)
    labels_gender = np.zeros(N, dtype=np.int32)
    last_tok = np.zeros(N, dtype=np.int64)   # id of the final (pooled, for 'last') token
    hashes = []

    captured, handle = register_hook(model, layer)
    device = next(model.parameters()).device
    t0 = time.time()

    for i, ex in enumerate(examples):
        text = ex["hard_text"]

        ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).input_ids.to(device)

        last_tok[i] = int(ids[0, -1])
        hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())

        captured.clear()
        with torch.no_grad():
            model(input_ids=ids)

        hs = captured[0][0]            # (T, D)
        for p in poolings:
            try:
                acts[p][i] = pool(hs, p, dtype=dtype)
            except ValueError as exc:
                handle.remove()
                raise ValueError(f"[{split}] example {i}, layer {layer}: {exc}") from exc
        labels_prof[i] = ex["profession"]
        labels_gender[i] = ex["gender"]

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / max(rate, 1)
            print(f"  [{split}] {i+1:>6}/{N}  "
                  f"{rate:.1f} ex/s  ETA {eta/60:.1f} min")

    handle.remove()

    fname = f"layer_{layer}.npy" if split == "train" else f"layer_{layer}_test.npy"
    for p in poolings:
        out_dir = out_dirs[p]
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / fname), acts[p])
        # Dual labels
        np.save(str(out_dir / f"labels_profession_{split}.npy"), labels_prof)
        np.save(str(out_dir / f"labels_gender_{split}.npy"), labels_gender)
        # Compatibility aliases (profession labels under standard names)
        np.save(str(out_dir / f"labels_{split}.npy"), labels_prof)
        if p == "last":   # the id is only meaningful for last-token pooling
            np.save(str(out_dir / f"tokens_{split}.npy"), last_tok)
        print(f"  Saved {out_dir / fname}  shape={acts[p].shape}  (pooling={p})")
    return np.array(hashes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract BiasBios activations for bias probing"
    )
    parser.add_argument("--layer",      type=int, default=27)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--n_train",    type=int, default=50_000)
    parser.add_argument("--n_test",     type=int, default=20_000)
    parser.add_argument("--pooling",    default="last",
                        choices=["last", "mean", "both"])
    parser.add_argument("--out_base",   default="biasbios/activations")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"],
                        help="Activation storage: auto uses float32 for Gemma, float16 otherwise")
    parser.add_argument("--dir_suffix", default="",
                        help="appended to the pooling directory name (e.g. _ids -> .../last_ids)")
    args = parser.parse_args()

    print("[extract] Loading dataset: LabHC/bias_in_bios …")
    from datasets import load_dataset
    ds = load_dataset("LabHC/bias_in_bios")
    train = list(ds["train"].shuffle(seed=args.seed).select(
        range(min(args.n_train, len(ds["train"])))))
    test = list(ds["test"].shuffle(seed=args.seed).select(
        range(min(args.n_test, len(ds["test"])))))
    print(f"[extract] Train: {len(train):,}  Test: {len(test):,}")

    print(f"[extract] Loading model: {args.model_name} …")
    tokenizer = load_tokenizer(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from geoae.checkpoint import load_lm
    model = load_lm(args.model_name, device_map="auto")
    dtype = output_dtype(model, args.dtype)
    print(f"[extract] Activation storage dtype: {dtype}")

    model_short = args.model_name.split("/")[-1]
    model_short = model_short.replace("Llama-", "llama")

    poolings = ["last", "mean"] if args.pooling == "both" else [args.pooling]
    out_dirs = {p: Path(args.out_base) / model_short / (p + args.dir_suffix) for p in poolings}

    src_hashes = {}
    for split, data in [("train", train), ("test", test)]:
        print(f"\n[extract] {split}  ({len(data):,} examples)")
        src_hashes[split] = extract_split(model, tokenizer, data, args.layer,
                      args.max_length, poolings=poolings,
                      out_dirs=out_dirs, split=split, dtype=dtype)

    for p in poolings:
        np.savez(out_dirs[p] / "source_hashes.npz", train=src_hashes["train"], test=src_hashes["test"])
        meta = {
            "pooling": p,
            "model": args.model_name,
            "layer": args.layer,
            "dtype": dtype.name,
            "n_train": len(train),
            "n_test": len(test),
            "max_length": args.max_length,
            "professions": PROFESSIONS,
            "genders": GENDERS,
            "classes": PROFESSIONS,   # compat alias for probe_perturbation
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_dirs[p] / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[extract] meta.json saved → {out_dirs[p] / 'meta.json'}")

    print("\n[extract] Done.")


if __name__ == "__main__":
    main()
