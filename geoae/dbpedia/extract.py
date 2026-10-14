"""
Extract per-document residual stream activations from DBpedia-14 texts.

Two modes (how the text is fed to the model):
  unprompted  raw text through Llama
  prompted    text wrapped in a classification prompt before Llama
              (the final ':' token is the last token, so it sees full context)

Two poolings (how the (T, D) sequence is reduced to one (D,) document vector):
  last  last-token hidden state (sees full causal context) — the original default
  mean  mean over all token positions (avg-token pooling)
  both  emit last AND mean from a single forward pass (no extra compute)

Note: mean pooling is intended for `unprompted` mode; on `prompted` mode it
would average in the instruction-template tokens, diluting the signal.

Saves (one tree per pooling):
  activations/{model}/{pooling}/{mode}/layer_{L}.npy        (N_train, D) float16/float32
  activations/{model}/{pooling}/{mode}/layer_{L}_test.npy   (N_test,  D) float16/float32
  activations/{model}/{pooling}/{mode}/labels_train.npy     (N_train,)   int32
  activations/{model}/{pooling}/{mode}/labels_test.npy      (N_test,)    int32
  activations/{model}/{pooling}/{mode}/meta.json
  activations/{model}/last/{mode}/tokens_{train,test}.npy   (N,) int64  id of the pooled (final) token
  activations/{model}/last/{mode}/source_hashes.npz         train/test = SHA256(content) hex strings
The last two are additive (token-bypass AEs need the current token id; probe_perturbation
reads source_hashes.npz via --source_hashes). --dir_suffix _ids writes a parallel tree
(.../last_ids/<mode>) so an existing extraction is never overwritten.

DBpedia-14 class names (0-indexed):
  0  Company                7  NaturalPlace
  1  EducationalInstitution 8  Village
  2  Artist                 9  Animal
  3  Athlete               10  Plant
  4  OfficeHolder          11  Album
  5  MeanOfTransportation  12  Film
  6  Building              13  WrittenWork

Usage:
    python -m geoae.dbpedia.extract --mode unprompted --layer 27 --pooling mean
    python -m geoae.dbpedia.extract --mode prompted   --layer 27 --pooling last
    python -m geoae.dbpedia.extract --mode unprompted --layer 27 --pooling both
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
from transformers import AutoTokenizer

from geoae.lm_arch import decoder_layers, hidden_size as lm_hidden_size

CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
]

PROMPT_TEMPLATE = (
    "Classify the following text into exactly one of these 14 categories: "
    "Company, EducationalInstitution, Artist, Athlete, OfficeHolder, "
    "MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, "
    "Album, Film, WrittenWork.\n\n"
    "Text: {text}\n\n"
    "Category:"
)


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

def output_dtype(model, dtype="auto") -> np.dtype:
    """Gemma residuals can exceed FP16 range; preserve them without clipping."""
    if dtype == "auto":
        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        is_gemma = "gemma" in model_type or "gemma" in type(model).__name__.lower()
        dtype = "float32" if is_gemma else "float16"
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("activation dtype must be auto, float16, or float32")
    return resolved


def pool(hs: torch.Tensor, pooling: str, dtype=np.float16) -> np.ndarray:
    """Reduce a (T, D) sequence to a single (D,) document vector."""
    if pooling == "last":
        vec = hs[-1]                 # last token — sees full causal context
    elif pooling == "mean":
        vec = hs.float().mean(dim=0)  # avg-token pooling
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    values = vec.float().numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite model activations in {pooling} pooling")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = values.astype(dtype)
    if not np.isfinite(stored).all():
        raise ValueError(f"Activation overflow storing {pooling} pooling as {np.dtype(dtype)}; "
                         "use --dtype float32")
    return stored


def extract_split(
    model,
    tokenizer,
    examples: list[dict],        # [{"content": ..., "label": ...}]
    layer: int,
    mode: str,
    max_length: int,
    poolings: list[str],         # e.g. ["last"], ["mean"], or ["last", "mean"]
    out_dirs: dict[str, Path],   # pooling -> output directory
    split: str,
    log_every: int = 100,
    dtype="auto",
) -> np.ndarray:
    N = len(examples)
    D = lm_hidden_size(model)   # text decoder width (model.config is multimodal on Gemma 3)
    # One activation buffer per requested pooling; filled from the SAME forward.
    dtype = output_dtype(model, dtype)
    acts = {p: np.zeros((N, D), dtype=dtype) for p in poolings}
    labels = np.zeros(N, dtype=np.int32)
    last_tok = np.zeros(N, dtype=np.int64)   # id of the final (pooled, for 'last') token
    hashes = []

    captured, handle = register_hook(model, layer)
    device = next(model.parameters()).device
    t0 = time.time()

    for i, ex in enumerate(examples):
        text  = ex["content"]
        label = ex["label"]

        if mode == "prompted":
            # Truncate text first so total prompt fits in max_length
            raw_ids = tokenizer.encode(text, add_special_tokens=False)
            # Reserve ~60 tokens for the prompt wrapper
            raw_ids = raw_ids[:max_length - 60]
            text_short = tokenizer.decode(raw_ids)
            full_text = PROMPT_TEMPLATE.format(text=text_short)
        else:
            full_text = text

        ids = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).input_ids.to(device)

        last_tok[i] = int(ids[0, -1])
        hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())   # source text, not the prompt

        captured.clear()
        with torch.no_grad():
            model(input_ids=ids)

        hs = captured[0][0]            # (T, D)
        for p in poolings:
            try:
                acts[p][i] = pool(hs, p, dtype=dtype)
            except ValueError as exc:
                handle.remove()
                raise ValueError(f"[{mode}/{split}] example {i}, layer {layer}: {exc}") from exc
        labels[i] = label

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (N - i - 1) / max(rate, 1)
            print(f"  [{mode}/{split}] {i+1:>6}/{N}  "
                  f"{rate:.1f} ex/s  ETA {eta/60:.1f} min")

    handle.remove()

    # ActivationBuffer expects layer_<L>.npy for the train split;
    # evaluate_dbpedia.py expects layer_<L>_test.npy for the test split.
    fname = f"layer_{layer}.npy" if split == "train" else f"layer_{layer}_test.npy"
    for p in poolings:
        out_dir = out_dirs[p]
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / fname), acts[p])
        np.save(str(out_dir / f"labels_{split}.npy"), labels)
        if p == "last":   # the id is only meaningful for last-token pooling
            np.save(str(out_dir / f"tokens_{split}.npy"), last_tok)
        print(f"  Saved {out_dir / fname}  shape={acts[p].shape}  (pooling={p})")
    return np.array(hashes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        default="both",
                        choices=["unprompted", "prompted", "both"])
    parser.add_argument("--pooling",     default="last",
                        choices=["last", "mean", "both"],
                        help="Token reduction: last token, mean over tokens, or both")
    parser.add_argument("--layer",       type=int, default=27)
    parser.add_argument("--model_name",  default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--max_length",  type=int, default=256)
    parser.add_argument("--n_train",     type=int, default=50_000,
                        help="Max training examples to extract")
    parser.add_argument("--n_test",      type=int, default=10_000,
                        help="Max test examples to extract")
    parser.add_argument("--out_base",    default="dbpedia/activations")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"],
                        help="Activation storage: auto uses float32 for Gemma, float16 otherwise")
    parser.add_argument("--dir_suffix",  default="",
                        help="appended to the pooling directory name (e.g. _ids -> .../last_ids/<mode>)")
    args = parser.parse_args()

    print("[extract] Loading dataset: dbpedia_14 …")
    from datasets import load_dataset
    ds     = load_dataset("fancyzhx/dbpedia_14")
    train  = list(ds["train"].shuffle(seed=42).select(range(min(args.n_train,  len(ds["train"])))))
    test   = list(ds["test"].shuffle(seed=42).select( range(min(args.n_test,   len(ds["test"])))))
    print(f"[extract] Train: {len(train):,}  Test: {len(test):,}")

    print(f"[extract] Loading model: {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from geoae.checkpoint import load_lm
    model = load_lm(args.model_name, device_map="auto")
    dtype = output_dtype(model, args.dtype)
    print(f"[extract] Activation storage dtype: {dtype}")

    # Model short name for directory structure
    # "meta-llama/Llama-3.2-3B" → "llama3.2-3B"
    # "Qwen/Qwen3.5-9B" → "Qwen3.5-9B"
    model_short = args.model_name.split("/")[-1]
    model_short = model_short.replace("Llama-", "llama")

    modes    = ["unprompted", "prompted"] if args.mode == "both" else [args.mode]
    poolings = ["last", "mean"] if args.pooling == "both" else [args.pooling]

    for mode in modes:
        print(f"\n{'='*55}\n  Mode: {mode}  |  pooling: {', '.join(poolings)}\n{'='*55}")
        # One output tree per pooling: activations/<model>/<pooling>/<mode>/
        out_dirs = {p: Path(args.out_base) / model_short / (p + args.dir_suffix) / mode for p in poolings}

        src_hashes = {}
        for split, data in [("train", train), ("test", test)]:
            print(f"\n[extract] {mode} / {split}  ({len(data):,} examples)")
            src_hashes[split] = extract_split(model, tokenizer, data, args.layer,
                          mode, args.max_length, poolings=poolings,
                          out_dirs=out_dirs, split=split, dtype=dtype)

        for p in poolings:
            np.savez(out_dirs[p] / "source_hashes.npz", train=src_hashes["train"], test=src_hashes["test"])
            meta = {
                "mode": mode,
                "pooling": p,
                "model": args.model_name,
                "layer": args.layer,
                "dtype": dtype.name,
                "n_train": len(train),
                "n_test": len(test),
                "max_length": args.max_length,
                "classes": CLASSES,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(out_dirs[p] / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[extract] meta.json saved → {out_dirs[p] / 'meta.json'}")

    print("\n[extract] Done.")


if __name__ == "__main__":
    main()
