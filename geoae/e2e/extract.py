"""
Precompute and cache the teacher (original-model) last-token logits used as the
KL target during end-to-end training.

The caches are written into the SAME activations directory as the existing
`layer_<L>.npy`, in the SAME row order, so `E2EBuffer` can align them
index-for-index (see data_e2e.py).

Two regimes (auto-detected from the target layer vs. number of decoder layers):

  last layer
      teacher = lm_head(norm(act))  where `act` is the cached raw last-token
      residual at the final layer (== the input to the final norm). No dataset
      re-tokenisation and no full forward are needed: we stream the existing
      `layer_<L>.npy` through the head. Writes:
          teacher_logits_layer<L>.npy   (N, V) float16

  intermediate layer
      teacher = real model logits at the last token. Re-tokenises DBpedia in the
      identical order (shuffle(seed=42).select(range(N))) and runs the full
      forward, also caching the LEFT-padded input_ids / attention_mask that the
      train-time spliced forward consumes. Writes additionally:
          input_ids_layer<L>.npy        (N, T) int32   (left-padded)
          attention_mask_layer<L>.npy   (N, T) int8

Usage:
    python e2e/extract_e2e.py --config e2e/configs/Qwen3.5-9B/layer31/unprompted_last_gelu.yaml
    python e2e/extract_e2e.py --config <cfg> --split both
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path



import numpy as np
import torch
from transformers import AutoTokenizer

from geoae.config import Config
from geoae.e2e.logits import LogitsComputer

PROMPT_TEXT_KEY = "content"  # DBpedia document body column


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_frozen_lm(model_name: str, device: torch.device):
    from geoae.checkpoint import load_lm
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # last real token at index -1 for batched forward
    return tok, load_lm(model_name, device=device)


# --------------------------------------------------------------------------- #
# Last-layer: stream cached activations through the head.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_last_layer(
    lc: LogitsComputer,
    activations_dir: Path,
    layer: int,
    split: str,
    device: torch.device,
    chunk: int = 2048,
) -> None:
    fname = f"layer_{layer}.npy" if split == "train" else f"layer_{layer}_test.npy"
    acts_path = activations_dir / fname
    if not acts_path.exists():
        raise FileNotFoundError(f"Activation file not found: {acts_path}")

    acts = np.load(str(acts_path), mmap_mode="r")  # (N, D) raw fp16
    N, D = acts.shape
    V = lc.lm_head.weight.shape[0]

    out_name = (
        f"teacher_logits_layer{layer}.npy"
        if split == "train"
        else f"teacher_logits_layer{layer}_test.npy"
    )
    out_path = activations_dir / out_name
    teacher = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float16, shape=(N, V)
    )

    print(f"[extract-e2e] last-layer head pass: {acts_path.name} -> {out_name} "
          f"(N={N}, V={V})")
    t0 = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        a = torch.from_numpy(np.asarray(acts[s:e]).astype(np.float32)).to(device)
        logits = lc.head_logits(a).float().cpu().numpy().astype(np.float16)
        teacher[s:e] = logits
        if s % (chunk * 10) == 0:
            done = e / N
            print(f"  {e:>7}/{N}  ({done*100:5.1f}%)  "
                  f"{(time.time()-t0)/max(done,1e-9)/60:.1f} min est total")
    teacher.flush()
    print(f"[extract-e2e] wrote {out_path}  shape={teacher.shape}  dtype=float16")


# --------------------------------------------------------------------------- #
# Intermediate layer: full forward, cache logits + left-padded ids.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_intermediate(
    tok,
    lm,
    lc: LogitsComputer,
    activations_dir: Path,
    layer: int,
    split: str,
    device: torch.device,
    n_examples: int,
    max_length: int = 256,
    log_every: int = 200,
) -> None:
    from datasets import load_dataset

    ds = load_dataset("dbpedia_14")
    key = "train" if split == "train" else "test"
    data = list(
        ds[key].shuffle(seed=42).select(range(min(n_examples, len(ds[key]))))
    )
    N = len(data)
    V = lc.lm_head.weight.shape[0]

    suffix = "" if split == "train" else "_test"
    teacher = np.lib.format.open_memmap(
        str(activations_dir / f"teacher_logits_layer{layer}{suffix}.npy"),
        mode="w+", dtype=np.float16, shape=(N, V),
    )
    ids_arr = np.lib.format.open_memmap(
        str(activations_dir / f"input_ids_layer{layer}{suffix}.npy"),
        mode="w+", dtype=np.int32, shape=(N, max_length),
    )
    attn_arr = np.lib.format.open_memmap(
        str(activations_dir / f"attention_mask_layer{layer}{suffix}.npy"),
        mode="w+", dtype=np.int8, shape=(N, max_length),
    )

    pad_id = tok.pad_token_id
    print(f"[extract-e2e] intermediate full-forward: layer {layer}, split {split}, "
          f"N={N}, V={V}, max_length={max_length}")
    t0 = time.time()
    for i, ex in enumerate(data):
        enc = tok(
            ex[PROMPT_TEXT_KEY], return_tensors="pt",
            truncation=True, max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)          # (1, t)
        attn = enc["attention_mask"].to(device)          # (1, t)
        out = lm(input_ids=input_ids, attention_mask=attn)
        teacher[i] = out.logits[0, -1].float().cpu().numpy().astype(np.float16)

        # Store LEFT-padded to max_length so the train-time batch is rectangular.
        t = input_ids.shape[1]
        ids_row = np.full(max_length, pad_id, dtype=np.int32)
        am_row = np.zeros(max_length, dtype=np.int8)
        ids_row[max_length - t:] = input_ids[0].cpu().numpy()
        am_row[max_length - t:] = 1
        ids_arr[i] = ids_row
        attn_arr[i] = am_row

        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1:>7}/{N}  {rate:.1f} ex/s  "
                  f"ETA {(N-i-1)/max(rate,1e-9)/60:.1f} min")

    for arr in (teacher, ids_arr, attn_arr):
        arr.flush()
    print(f"[extract-e2e] wrote teacher/input_ids/attention_mask for layer {layer} "
          f"({split})")


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sanity_check_last_layer(
    lc: LogitsComputer, activations_dir: Path, layer: int, device: torch.device,
    n: int = 8,
) -> None:
    """Recompute a few teacher rows and confirm they match the cache exactly."""
    acts = np.load(str(activations_dir / f"layer_{layer}.npy"), mmap_mode="r")
    teacher = np.load(
        str(activations_dir / f"teacher_logits_layer{layer}.npy"), mmap_mode="r"
    )
    a = torch.from_numpy(np.asarray(acts[:n]).astype(np.float32)).to(device)
    recomputed = lc.head_logits(a).float().cpu().numpy().astype(np.float16)
    cached = np.asarray(teacher[:n])
    max_diff = np.abs(recomputed.astype(np.float32) - cached.astype(np.float32)).max()
    print(f"[extract-e2e] sanity: max |recomputed - cached| logit diff = {max_diff:.3e}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to an e2e YAML config")
    ap.add_argument("--split", default="train", choices=["train", "test", "both"])
    ap.add_argument("--n_examples", type=int, default=50_000,
                    help="Used only for intermediate-layer re-tokenisation")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    model_name = cfg.extraction.model_name
    layer = cfg.data.target_layer
    from geoae.paths import resolve_path
    act_dir = resolve_path(cfg.data.activations_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract-e2e] model={model_name}  layer={layer}  device={device}")
    print(f"[extract-e2e] activations_dir={act_dir}")

    tok, lm = load_frozen_lm(model_name, device)
    lc = LogitsComputer(lm, layer)
    print(f"[extract-e2e] n_layers={lc.n_layers}  is_last={lc.is_last}")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        if lc.is_last:
            extract_last_layer(lc, act_dir, layer, split, device)
        else:
            extract_intermediate(
                tok, lm, lc, act_dir, layer, split, device,
                n_examples=args.n_examples, max_length=args.max_length,
            )

    if lc.is_last and "train" in splits:
        sanity_check_last_layer(lc, act_dir, layer, device)

    print("[extract-e2e] done.")


if __name__ == "__main__":
    main()
