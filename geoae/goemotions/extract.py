"""Extract Llama residual-stream representations for GoEmotions.

The simplified GoEmotions configuration contains 28 non-exclusive labels
(27 emotions plus neutral).  This extractor preserves that multi-label target
as a multi-hot matrix and writes the official train/validation/test splits in
the format consumed by :mod:`geoae.interp.linear_probe_compare`.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.lm_arch import decoder_layers, hidden_size as lm_hidden_size


def _extract_split(
    model,
    tokenizer,
    split,
    *,
    layer: int,
    max_length: int,
    batch_size: int,
    out_dir: Path,
    split_name: str,
    n_labels: int,
) -> None:
    n_rows = len(split)
    width = lm_hidden_size(model)
    suffix = "" if split_name == "train" else f"_{split_name}"
    act_path = out_dir / f"layer_{layer}{suffix}.npy"
    label_path = out_dir / f"labels_{split_name}.npy"
    acts = np.lib.format.open_memmap(
        act_path, mode="w+", dtype=np.float16, shape=(n_rows, width)
    )
    labels = np.zeros((n_rows, n_labels), dtype=np.uint8)

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    device = next(model.parameters()).device
    started = time.time()
    try:
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            batch = split[start:stop]
            encoded = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            captured.clear()
            with torch.inference_mode():
                model(**encoded)
            hidden = captured[0]
            last = encoded.attention_mask.sum(dim=1) - 1
            rows = torch.arange(stop - start, device=hidden.device)
            acts[start:stop] = hidden[rows, last].float().cpu().numpy().astype(np.float16)
            for local_row, ids in enumerate(batch["labels"]):
                labels[start + local_row, ids] = 1
            if stop % 1024 < batch_size or stop == n_rows:
                rate = stop / (time.time() - started)
                eta = (n_rows - stop) / max(rate, 1e-9)
                print(
                    f"  [{split_name}] {stop:>6}/{n_rows}  "
                    f"{rate:.1f} examples/s  ETA {eta / 60:.1f} min",
                    flush=True,
                )
    finally:
        handle.remove()
    acts.flush()
    np.save(label_path, labels)
    print(f"  saved {act_path} {acts.shape} and {label_path} {labels.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_base", default="goemotions/activations")
    args = parser.parse_args()

    print("[extract] loading google-research-datasets/go_emotions (simplified)")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    label_feature = dataset["train"].features["labels"].feature
    label_names = label_feature.names

    print(f"[extract] loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_lm(args.model_name, device_map="auto")
    model.eval()

    model_short = args.model_name.split("/")[-1].replace("Llama-", "llama")
    out_dir = Path(args.out_base) / model_short / "last" / "unprompted"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        _extract_split(
            model,
            tokenizer,
            dataset[split_name],
            layer=args.layer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            out_dir=out_dir,
            split_name=split_name,
            n_labels=len(label_names),
        )

    meta = {
        "dataset": "google-research-datasets/go_emotions",
        "config": "simplified",
        "model": args.model_name,
        "layer": args.layer,
        "pooling": "last",
        "mode": "unprompted",
        "max_length": args.max_length,
        "split_sizes": {name: len(dataset[name]) for name in dataset},
        "labels": label_names,
        "multi_label": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (out_dir / "meta.json").open("w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[extract] wrote {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
