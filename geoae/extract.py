"""
Phase 1: Extract residual-stream activations from frozen Llama 3.2 3B.

Writes one memory-mapped .npy per layer, shape (n_tokens, hidden_size), float16,
plus meta.json. This is the corpus the AE is trained on, so it uses the same
interpretability hygiene as the max-activating-example collector
(geoae.interp.closest_tokens):

  * diversified data : round-robins over web / wiki / code / math with the
                       CURRENT working dataset names (the old names — `wikipedia`,
                       `bookcorpus`, `cc_news`, `codeparrot/github-code`,
                       `RedPajama` — silently fail to load and leave you training
                       on C4 only).
  * per-doc forward  : one document per forward pass (truncated), so attention
                       never crosses document boundaries — no packing
                       contamination, and every row is a clean activation.
  * domain balance   : each doc is truncated to `max_doc_tokens`, so long
                       sources (wiki) don't dominate the token budget.
  * leading skip     : the first `skip_leading` tokens of each doc (BOS /
                       document-start) are NOT written — they otherwise flood the
                       boundary clusters.

Usage:
    python -m geoae.extract --n_tokens 5000000
    python -m geoae.extract --config configs/base/full_run_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from geoae.lm_arch import decoder_layers, describe, hidden_size as lm_hidden_size


# ---------------------------------------------------------------------------
# Diverse corpus (current, working dataset names). Each is probed on open; any
# that fails (auth / network / schema) is skipped, and the budget is spread over
# whatever loads — so one broken source can't silently collapse you to C4-only.
# ---------------------------------------------------------------------------

DEFAULT_SOURCES = [
    dict(domain="web",  name="allenai/c4",                        config="en",          split="train", field="text"),
    dict(domain="wiki", name="wikimedia/wikipedia",               config="20231101.en", split="train", field="text"),
    dict(domain="code", name="codeparrot/codeparrot-clean-valid", config=None,          split="train", field="content"),
    dict(domain="math", name="open-web-math/open-web-math",       config=None,          split="train", field="text"),
    dict(domain="pile", name="monology/pile-uncopyrighted",       config=None,          split="train", field="text"),
]


def open_sources(sources: list[dict]) -> list[dict]:
    from datasets import load_dataset
    live = []
    for s in sources:
        try:
            ds = load_dataset(s["name"], name=s["config"], split=s["split"], streaming=True)
            it = iter(ds)
            first = next(it)                       # probe so a bad schema fails here
            if s["field"] not in first:
                print(f"[extract]   ! {s['domain']}: no field '{s['field']}' — skipping")
                continue
            live.append({**s, "iter": it, "primed": first})
            print(f"[extract]   + {s['domain']:<5} {s['name']}")
        except Exception as e:
            print(f"[extract]   ! {s['domain']:<5} {s['name']}: {type(e).__name__} — skipping")
    if not live:
        raise RuntimeError("All data sources failed to load. Cannot proceed.")
    return live


def stream_docs(sources: list[dict], n_tokens: int, tokenizer, min_len: int, max_len: int):
    """Round-robin over sources, yielding (input_ids[1,T], domain). Each doc is
    truncated to max_len so domains stay balanced and contexts stay diverse."""
    seen = 0
    exhausted = set()
    while seen < n_tokens and len(exhausted) < len(sources):
        for si, s in enumerate(sources):
            if si in exhausted or seen >= n_tokens:
                continue
            try:
                ex = s.pop("primed", None) or next(s["iter"])
            except StopIteration:
                exhausted.add(si)
                continue
            text = ex.get(s["field"]) or ""
            if not text:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]
            if ids.shape[1] < min_len:
                continue
            yield ids, s["domain"]
            seen += ids.shape[1]


# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------

def trim_npy(path: Path, n_rows: int) -> None:
    """Shrink a preallocated .npy to n_rows in place (edit header + truncate), so
    a run that exhausts its sources before n_tokens leaves no trailing zero rows
    (those would otherwise land in the val tail and corrupt validation)."""
    import numpy.lib.format as npf
    with open(path, "r+b") as f:
        npf.read_magic(f)
        shape, _fortran, dtype = npf.read_array_header_1_0(f)
        data_off = f.tell()
        if n_rows >= shape[0]:
            return
        f.seek(0)
        head = bytearray(f.read(data_off))
    old_shape = b"(%s)" % b", ".join(b"%d" % s for s in shape)
    new_shape = b"(%s)" % b", ".join(b"%d" % s for s in (n_rows,) + tuple(shape[1:]))
    k = head.find(old_shape)
    assert k != -1, f"shape marker not found in {path}"
    head[k:k + len(old_shape)] = new_shape
    # Re-pad the header to its original byte length (data offset must not move).
    body = bytes(head[:10]) + bytes(head[10:]).rstrip(b" \n")
    head = body + b" " * (data_off - len(body) - 1) + b"\n"
    assert len(head) == data_off
    rowbytes = dtype.itemsize
    for s in shape[1:]:
        rowbytes *= s
    with open(path, "r+b") as f:
        f.seek(0); f.write(head)
    os.truncate(path, data_off + n_rows * rowbytes)


def register_hooks(model, layers: list[int], store_dtype=torch.float16):
    """Capture the residual stream after each requested decoder block.

    `nonfinite` counts elements lost to overflow when storing in float16 — a
    real risk on Gemma-family models, whose residual stream is far larger than
    Llama's (embeddings scaled by sqrt(d)) and can exceed the fp16 max of 65504.
    Counting happens on-device (no per-doc sync); read it after the run.
    """
    captured: dict[int, torch.Tensor | None] = {l: None for l in layers}
    nonfinite: dict[int, torch.Tensor | int] = {l: 0 for l in layers}
    blocks = decoder_layers(model)
    bad = [l for l in layers if not 0 <= l < len(blocks)]
    if bad:
        raise ValueError(
            f"Layer(s) {bad} out of range: {type(model).__name__} has "
            f"{len(blocks)} decoder layers (valid 0..{len(blocks) - 1})."
        )
    handles = []
    for layer_idx in layers:
        def make_hook(idx):
            def hook(module, inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                h = hs.detach().to(store_dtype)
                if store_dtype == torch.float16:
                    nonfinite[idx] = nonfinite[idx] + (~torch.isfinite(h)).sum()
                captured[idx] = h.cpu()
            return hook
        handles.append(blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))
    return captured, handles, nonfinite


# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------

def extract(model_name, layers, n_tokens, hidden_size, out_dir: Path,
            max_doc_tokens, skip_leading, min_doc_len, log_every, dtype="float16"):
    out_dir.mkdir(parents=True, exist_ok=True)

    if dtype not in ("float16", "float32"):
        raise ValueError(f"extraction.dtype must be float16 or float32, got {dtype!r} "
                         "(numpy memmaps cannot hold bfloat16).")
    np_dtype = np.dtype(dtype)
    store_dtype = getattr(torch, dtype)

    # Preflight disk check. open_memmap creates SPARSE files, so an oversized
    # request allocates instantly and then dies with ENOSPC deep into the run,
    # after the GPU time is already spent. Fail here instead, before the model
    # is even loaded.
    need = n_tokens * hidden_size * np_dtype.itemsize * len(layers)
    free = shutil.disk_usage(out_dir).free
    if need > free * 0.98:
        per_layer_m = hidden_size * np_dtype.itemsize * 1e6 / 1e9   # GB per 1M tokens
        fits = int(free * 0.98 / (hidden_size * np_dtype.itemsize * len(layers)))
        raise RuntimeError(
            f"Need {need/1e9:.0f} GB for {len(layers)} layers x {n_tokens:,} tokens "
            f"({dtype}, d={hidden_size}) but only {free/1e9:.0f} GB is free in {out_dir}.\n"
            f"  One layer costs {per_layer_m:.1f} GB per 1M tokens.\n"
            f"  Fits as-is: n_tokens <= {fits:,} across these {len(layers)} layers, "
            f"or {int(free * 0.98 / (hidden_size * np_dtype.itemsize)):,} tokens for a "
            f"single layer (--layers L).\n"
            f"  Or free space / point --out_dir at another volume."
        )

    print(f"[extract] Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from geoae.checkpoint import load_lm
    model = load_lm(model_name, device_map="auto")
    device = next(model.parameters()).device
    print(f"[extract] {describe(model)}")
    print(f"[extract] Model on {device}")

    # The memmaps are preallocated at `hidden_size`; a config that disagrees with
    # the checkpoint would write a silently misshaped multi-GB file. Fail here.
    true_d = lm_hidden_size(model)
    if hidden_size != true_d:
        raise ValueError(
            f"Config hidden_size={hidden_size} but {model_name} has d={true_d}. "
            "Fix `extraction.hidden_size` (and `model.hidden_size`) in the config."
        )

    captured, handles, nonfinite = register_hooks(model, layers, store_dtype)

    gb = n_tokens * hidden_size * np_dtype.itemsize * len(layers) / 1e9
    print(f"[extract] Allocating {len(layers)} × ({n_tokens:,}, {hidden_size}) "
          f"{dtype} = {gb:.1f} GB in {out_dir}")
    mmaps = {l: np.lib.format.open_memmap(str(out_dir / f"layer_{l}.npy"), mode="w+",
                                          dtype=np_dtype, shape=(n_tokens, hidden_size))
             for l in layers}

    print("[extract] Opening data sources:")
    sources = open_sources(DEFAULT_SOURCES)

    skip = max(0, skip_leading)
    write_ptr = 0
    n_docs = 0
    dom_tokens: dict[str, int] = {}
    t0 = time.time()

    with torch.no_grad():
        for ids, domain in stream_docs(sources, n_tokens, tokenizer, min_doc_len, max_doc_tokens):
            T = ids.shape[1]
            if T <= skip:
                continue
            model(input_ids=ids.to(device))

            n_eff = T - skip
            if write_ptr + n_eff > n_tokens:
                n_eff = n_tokens - write_ptr
            if n_eff <= 0:
                break
            for l in layers:
                acts = captured[l][0]                       # (T, H)
                mmaps[l][write_ptr:write_ptr + n_eff] = acts[skip:skip + n_eff].numpy()

            write_ptr += n_eff
            n_docs += 1
            dom_tokens[domain] = dom_tokens.get(domain, 0) + n_eff

            # Abort on the FIRST sign of float16 overflow. An overflowing model
            # (Gemma 3 12B: dim 2339 exceeds 65504 on ~98% of tokens at late
            # layers) writes inf, which makes mean/std inf/nan and silently
            # poisons every downstream consumer. Reporting this only at the end
            # costs hours of GPU and hundreds of GB, so check each log interval
            # — a per-token overflow trips within the first few docs.
            if store_dtype == torch.float16 and n_docs % log_every == 0:
                hit = {l: int(v.item()) for l, v in nonfinite.items()
                       if torch.is_tensor(v) and int(v.item()) > 0}
                if hit:
                    for h in handles:
                        h.remove()
                    raise RuntimeError(
                        f"float16 OVERFLOW after {n_docs} docs / {write_ptr:,} tokens: "
                        f"non-finite elements per layer {hit}.\n"
                        f"  This model's residual stream exceeds the float16 max of "
                        f"65504, so the dump would be poisoned with inf.\n"
                        f"  Fix: set `extraction.dtype: float32` in the config (2x "
                        f"disk — reduce n_tokens to match) and re-run.\n"
                        f"  Partial files in {out_dir} are sparse and will be "
                        f"overwritten by the next run."
                    )

            if n_docs % log_every == 0:
                rate = write_ptr / max(time.time() - t0, 1e-6)
                eta = (n_tokens - write_ptr) / max(rate, 1)
                print(f"[extract] {write_ptr:>9,}/{n_tokens:,} ({100*write_ptr/n_tokens:.1f}%) "
                      f"| {rate:.0f} tok/s | ETA {eta/60:.1f} min | "
                      f"domains={ {d: f'{100*c/write_ptr:.0f}%' for d, c in dom_tokens.items()} }")
            if write_ptr >= n_tokens:
                break

    for l in layers:
        mmaps[l].flush()
        del mmaps[l]   # release the mmap so the file can be trimmed
    for h in handles:
        h.remove()

    # If sources ran out before n_tokens, the preallocated tail is zeros — and the
    # val split is the contiguous tail, so those zeros would corrupt validation.
    # Trim every layer file down to what was actually written.
    if write_ptr < n_tokens:
        print(f"[extract] Sources exhausted at {write_ptr:,}/{n_tokens:,}; "
              f"trimming .npy files to drop trailing zero rows.")
        for l in layers:
            trim_npy(out_dir / f"layer_{l}.npy", write_ptr)

    # The activations just changed, so any cached normalisation stats are stale.
    # Training recomputes train mean/std but the val buffer trusts this cache —
    # leaving it would normalise val with the OLD corpus's stats. Delete it.
    for stale in out_dir.glob("norm_params_layer*.npz"):
        stale.unlink()
        print(f"[extract] Removed stale norm cache: {stale.name}")

    # fp16 overflow report: any non-finite element means the residual exceeded
    # 65504 and was stored as inf, which would poison norm stats and training.
    overflow = {l: int(v.item()) if torch.is_tensor(v) else int(v)
                for l, v in nonfinite.items()}
    if any(overflow.values()):
        print(f"[extract] !! float16 OVERFLOW — non-finite elements per layer: {overflow}")
        print("[extract] !! Re-run with extraction.dtype: float32 (2x disk) for these layers.")

    meta = {
        "model": model_name, "layers": layers, "n_tokens": write_ptr,
        "hidden_size": hidden_size, "n_docs": n_docs,
        "max_doc_tokens": max_doc_tokens, "skip_leading": skip, "min_doc_len": min_doc_len,
        "data_sources": [{"domain": s["domain"], "name": s["name"]} for s in sources],
        "domain_tokens": dom_tokens, "dtype": dtype, "per_doc_forward": True,
        "nonfinite_elements": overflow,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[extract] Done. {write_ptr:,} tokens / {n_docs:,} docs -> {out_dir}/")
    print(f"[extract] Domain mix: { {d: f'{100*c/max(write_ptr,1):.0f}%' for d, c in dom_tokens.items()} }")

    # Sanity check. |max| matters as much as mean/std here: it is the headroom
    # left before float16 storage overflows.
    for l in layers:
        sample = np.load(str(out_dir / f"layer_{l}.npy"), mmap_mode="r")[:min(10_000, write_ptr)].astype(np.float32)
        print(f"[extract] layer {l} sanity: |mean|={abs(sample.mean()):.4f} "
              f"std={sample.std():.4f} |max|={np.abs(sample).max():.1f} "
              f"median_token_norm={np.median(np.linalg.norm(sample, axis=1)):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Extract Llama residual-stream activations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n_tokens", type=int, default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_doc_tokens", type=int, default=256,
                        help="Truncate each doc (balances domains, diversifies contexts)")
    parser.add_argument("--skip_leading", type=int, default=4,
                        help="Skip first N tokens of each doc (BOS / document-start flood)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Override which layers to extract (e.g. --layers 27)")
    parser.add_argument("--min_doc_len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from geoae.seeding import seed_everything
    seed_everything(args.seed)
    from geoae.config import Config
    from geoae.paths import default_config, resolve_path
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        d = default_config()
        cfg = Config.from_yaml(d) if d.exists() else Config()
    ex = cfg.extraction
    if args.n_tokens is not None:
        ex.n_tokens = args.n_tokens
    if args.layers is not None:
        ex.layers = args.layers

    out_dir = resolve_path(args.out_dir if args.out_dir else ex.activations_dir)

    extract(
        model_name=ex.model_name, layers=ex.layers, n_tokens=ex.n_tokens,
        hidden_size=ex.hidden_size, out_dir=out_dir,
        max_doc_tokens=getattr(ex, "max_doc_tokens", args.max_doc_tokens),
        skip_leading=getattr(ex, "skip_leading", args.skip_leading),
        min_doc_len=args.min_doc_len, log_every=getattr(ex, "log_every", 200),
        dtype=getattr(ex, "dtype", "float16"),
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # HF streaming threads don't join cleanly; all work is saved above
