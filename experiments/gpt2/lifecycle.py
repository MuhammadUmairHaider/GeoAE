#!/usr/bin/env python3
"""Check campaign artifacts and publish verified final-checkpoint markers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ACT = ROOT / "activations_gpt2_sampled_struct5_10M"
RUN_NAME = "k2000_bnh_b32k_lam1_d1536_dpc_sampled_struct5_tokbias"


def checkpoint_dir(layer: int) -> Path:
    return ROOT / "checkpoints" / "gpt2" / f"layer{layer}" / RUN_NAME


def check_extraction(config: str | Path, layer: int | None = None) -> dict:
    import numpy as np
    from geoae.config import Config
    ex = Config.from_yaml(config).extraction
    act = ROOT / ex.activations_dir
    meta = json.loads((act / "meta.json").read_text())
    expected = dict(model=ex.model_name, layers=ex.layers, mode="sampled",
                    n_tokens=ex.n_tokens, hidden_size=ex.hidden_size, dtype=ex.dtype,
                    context_len=ex.context_len, positions_per_doc=ex.positions_per_doc,
                    skip_leading=ex.skip_leading, seed=42)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{act}/meta.json: {key}={meta.get(key)!r}, expected {value!r}")
    if [(s["domain"], s["weight"]) for s in meta["sources"]] != [
            (s["domain"], s["weight"]) for s in ex.sources]:
        raise ValueError("Extraction source domains/weights do not match campaign")
    for number in ex.layers if layer is None else [layer]:
        x = np.load(act / f"layer_{number}.npy", mmap_mode="r")
        if x.shape != (ex.n_tokens, ex.hidden_size) or x.dtype != np.dtype(ex.dtype):
            raise ValueError(f"Layer {number} has shape/dtype {x.shape}/{x.dtype}")
    for name in ("rows_doc", "rows_pos", "rows_tok", "rows_src", "rows_norm"):
        if np.load(act / f"{name}.npy", mmap_mode="r").shape != (ex.n_tokens,):
            raise ValueError(f"Wrong row count: {name}.npy")
    np.load(act / "docs.npy", mmap_mode="r")
    return meta


def check_table(layer: int) -> Path:
    import numpy as np
    table_path = ROOT / "e2e/checkpoints/general/gpt2" / f"layer{layer}" / "token_bias_sampled_struct5.npz"
    with np.load(table_path, allow_pickle=False) as table, np.load(ACT / f"norm_params_layer{layer}.npz") as norm:
        if int(table["layer"]) != layer or str(table["model_name"]) != "gpt2":
            raise ValueError(f"Wrong model/layer token table: {table_path}")
        if table["table"].shape != (len(table["token_ids"]), 768):
            raise ValueError(f"Wrong token table shape: {table_path}")
        if len(table["token_ids"]) == 0 or not np.isfinite(table["table"]).all():
            raise ValueError(f"Empty or nonfinite token table: {table_path}")
        if int(table["vocab_size"]) != 50257 or float(table["shrink_k"]) != 30.0 or int(table["min_count"]) != 10:
            raise ValueError(f"Wrong token table recipe: {table_path}")
        for key in ("mean", "std"):
            if not np.array_equal(table[f"norm_{key}"], norm[key]):
                raise ValueError(f"Token table and dump {key} disagree: {table_path}")
    return table_path


def write_completion(layer: int) -> dict:
    directory = checkpoint_dir(layer)
    files = list(directory.glob("step_*.pt"))
    if not files:
        raise ValueError(f"No checkpoints in {directory}")
    import torch
    path = max(files, key=lambda p: int(p.stem.split("_")[1]))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    epoch = int(checkpoint["epoch"])
    expected = [("model", "hidden_size", 768), ("model", "latent_dim", 1536),
                ("model", "n_clusters", 2000), ("model", "token_bias_mode", "bypass"),
                ("data", "target_layer", layer), ("train", "seed", 42),
                ("train", "n_epochs", 50), ("extraction", "model_name", "gpt2")]
    for section, key, value in expected:
        actual = cfg[section].get(key, "bypass" if key == "token_bias_mode" else None)
        if actual != value:
            raise ValueError(f"Checkpoint {section}.{key}={actual!r}, expected {value!r}")
    if epoch < 50:
        raise ValueError(f"Latest checkpoint is epoch {epoch}/50: {path.name}")
    expected_table = f"e2e/checkpoints/general/gpt2/layer{layer}/token_bias_sampled_struct5.npz"
    if cfg["model"].get("token_bias") != expected_table:
        raise ValueError("Final checkpoint does not use this layer's campaign token bypass")
    marker = dict(checkpoint=str(path.relative_to(ROOT)), checkpoint_absolute=str(path),
                  layer=layer, epoch=epoch, step=int(checkpoint["step"]),
                  config=f"experiments/gpt2/configs/layer{layer}_2x.yaml", model="gpt2",
                  latent_dim=1536, n_clusters=2000, token_bias_mode="bypass",
                  written_at=datetime.now(timezone.utc).isoformat())
    target = directory / "training_complete.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, indent=2) + "\n")
    os.replace(temporary, target)
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("extraction", "table", "complete"))
    parser.add_argument("--layer", type=int, choices=range(12))
    parser.add_argument("--config", default=str(ROOT / "experiments/gpt2/configs/extract_all.yaml"))
    args = parser.parse_args()
    if args.operation != "extraction" and args.layer is None:
        parser.error("table/complete require --layer")
    try:
        if args.operation == "extraction":
            result = check_extraction(args.config, args.layer)
            print(f"[verified] {result['n_tokens']:,} sampled rows; layers {result['layers']}")
        elif args.operation == "table":
            print(f"[verified] {check_table(args.layer)}")
        else:
            result = write_completion(args.layer)
            print(f"[complete] layer {args.layer}, epoch {result['epoch']}, {result['checkpoint']}")
    except (OSError, ValueError, KeyError, EOFError, RuntimeError) as error:
        print(f"[incomplete] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
