"""
Extract per-document residual stream activations from BiasBios biographies.

Dataset: LabHC/bias_in_bios  (De-Arteaga et al. 2019)
  - 28 professions, binary gender (0=male, 1=female)
  - Text field: hard_text (biography, still contains gendered pronouns)

Saves dual label arrays (profession + gender) alongside activations:
  activations/{model}/{pooling}/layer_{L}.npy         (N_train, D) float16
  activations/{model}/{pooling}/layer_{L}_test.npy    (N_test,  D) float16
  activations/{model}/{pooling}/labels_profession_train.npy
  activations/{model}/{pooling}/labels_profession_test.npy
  activations/{model}/{pooling}/labels_gender_train.npy
  activations/{model}/{pooling}/labels_gender_test.npy
  activations/{model}/{pooling}/labels_train.npy      (profession alias)
  activations/{model}/{pooling}/labels_test.npy        (profession alias)
  activations/{model}/{pooling}/meta.json

Usage:
    python -u -m geoae.bias.extract --layer 27 --n_train 50000 --n_test 20000
    python -u -m geoae.bias.extract --layer 27 --pooling both
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from geoae.lm_arch import decoder_layers, hidden_size as lm_hidden_size

PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]
GENDERS = ["male", "female"]


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

def register_hook(model, layer: int) -> tuple[list, object]:
    captured = []
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured.append(hs.detach().cpu())   # (1, T, D)
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    return captured, handle


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def pool(hs: torch.Tensor, pooling: str) -> np.ndarray:
    """Reduce a (T, D) sequence to a single (D,) document vector."""
    if pooling == "last":
        vec = hs[-1]
    elif pooling == "mean":
        vec = hs.float().mean(dim=0)
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    return vec.float().numpy().astype(np.float16)


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
) -> None:
    N = len(examples)
    D = lm_hidden_size(model)   # text decoder width (model.config is multimodal on Gemma 3)
    acts = {p: np.zeros((N, D), dtype=np.float16) for p in poolings}
    labels_prof = np.zeros(N, dtype=np.int32)
    labels_gender = np.zeros(N, dtype=np.int32)

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

        captured.clear()
        with torch.no_grad():
            model(input_ids=ids)

        hs = captured[0][0]            # (T, D)
        for p in poolings:
            acts[p][i] = pool(hs, p)
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
        print(f"  Saved {out_dir / fname}  shape={acts[p].shape}  (pooling={p})")


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from geoae.checkpoint import load_lm
    model = load_lm(args.model_name, device_map="auto")

    model_short = args.model_name.split("/")[-1]
    model_short = model_short.replace("Llama-", "llama")

    poolings = ["last", "mean"] if args.pooling == "both" else [args.pooling]
    out_dirs = {p: Path(args.out_base) / model_short / p for p in poolings}

    for split, data in [("train", train), ("test", test)]:
        print(f"\n[extract] {split}  ({len(data):,} examples)")
        extract_split(model, tokenizer, data, args.layer,
                      args.max_length, poolings=poolings,
                      out_dirs=out_dirs, split=split)

    for p in poolings:
        meta = {
            "pooling": p,
            "model": args.model_name,
            "layer": args.layer,
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
