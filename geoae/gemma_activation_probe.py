"""Small, non-destructive activation-range audit before a Gemma extraction run.

Downloads/loads the frozen LM and samples public source documents, but does not
write an activation dump, start AE training, or clip any activations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from geoae.config import Config
from geoae.gemma_dpc import DEFAULT_CONFIG, MODEL_NAME, write_json


def range_metrics(hidden: np.ndarray) -> dict:
    if hidden.ndim != 2 or len(hidden) == 0:
        raise ValueError("Expected a nonempty token-by-channel activation matrix")
    finite = np.isfinite(hidden)
    if not finite.all():
        return {"rows": len(hidden), "hidden_size": hidden.shape[1],
                "nonfinite_elements": int((~finite).sum()), "safe_for_training": False}
    abs_h = np.abs(hidden)
    fp16 = float(torch.finfo(torch.float16).max)
    bf16 = float(torch.finfo(torch.bfloat16).max)
    over = abs_h > fp16
    counts = over.sum(axis=0)
    maxima = abs_h.max(axis=0)
    medians = np.median(abs_h, axis=0)
    strongest = np.argsort(maxima)[-10:][::-1]
    # Counterfactual only: original hidden is never changed.
    delta = hidden - np.clip(hidden, -fp16, fp16)
    denom = float(np.square(hidden, dtype=np.float64).sum())
    return {
        "rows": len(hidden), "hidden_size": hidden.shape[1], "nonfinite_elements": 0,
        "safe_for_training": True, "float16_max": fp16, "bfloat16_max": bf16,
        "max_abs": float(maxima.max()),
        "absolute_value_quantiles": {str(q): float(np.quantile(abs_h, q)) for q in (.5, .9, .99, .999, .9999)},
        "tokens_over_float16_max": int(over.any(axis=1).sum()),
        "fraction_tokens_over_float16_max": float(over.any(axis=1).mean()),
        "elements_over_float16_max": int(over.sum()),
        "fraction_elements_over_float16_max": float(over.mean()),
        "channels_over_float16_max": np.flatnonzero(counts).tolist(),
        "elements_over_bfloat16_max": int((abs_h > bf16).sum()),
        "counterfactual_fp16_clip_relative_squared_change": float(np.square(delta, dtype=np.float64).sum() / max(denom, 1e-30)),
        "largest_channels": [{"channel": int(j), "max_abs": float(maxima[j]),
                              "median_abs": float(medians[j]),
                              "fraction_over_float16_max": float(counts[j] / len(hidden))}
                             for j in strongest],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-docs-per-domain", type=int, default=20)
    ap.add_argument("--max-doc-tokens", type=int, default=256)
    ap.add_argument("--skip-leading", type=int, default=4)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("Output already exists; choose a new path to preserve the previous audit")
    if args.n_docs_per_domain < 1 or args.max_doc_tokens <= args.skip_leading:
        ap.error("Invalid sampling limits")
    from geoae.hf_auth import ensure_hf_login
    from geoae.checkpoint import load_lm
    from geoae.lm_arch import decoder_layers
    from geoae.extract import DEFAULT_SOURCES, open_sources
    from transformers import AutoTokenizer
    ensure_hf_login(verbose=False)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    lm = load_lm(MODEL_NAME, device="cuda")
    cap = {}
    handle = decoder_layers(lm)[25].register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("hidden", (o[0] if isinstance(o, tuple) else o).detach()))
    sources = open_sources(DEFAULT_SOURCES)
    blocks, domains = [], {}
    try:
        with torch.inference_mode():
            for source in sources:
                n_docs = n_tokens = attempts = 0
                while n_docs < args.n_docs_per_domain and attempts < 10 * args.n_docs_per_domain:
                    attempts += 1
                    row = source.pop("primed", None)
                    if row is None:
                        try:
                            row = next(source["iter"])
                        except StopIteration:
                            break
                    text = row.get(source["field"]) or ""
                    ids = tok(text, return_tensors="pt", truncation=True,
                              max_length=args.max_doc_tokens)["input_ids"]
                    if ids.shape[1] < max(10, args.skip_leading + 1):
                        continue
                    lm(input_ids=ids.to("cuda"), use_cache=False)
                    h = cap.pop("hidden")[0, args.skip_leading:].float().cpu().numpy()
                    blocks.append(h)
                    n_tokens += len(h)
                    n_docs += 1
                domains[source["domain"]] = {"source": source["name"], "documents": n_docs, "tokens": n_tokens}
                print(f"[probe] {source['domain']}: {n_docs} documents, {n_tokens} tokens", flush=True)
    finally:
        handle.remove()
    if not blocks:
        raise RuntimeError("No probe activations collected")
    hidden = np.concatenate(blocks)
    stats = range_metrics(hidden)
    existing = args.out.parent
    while not existing.exists():
        existing = existing.parent
    extraction = Config.from_yaml(DEFAULT_CONFIG).extraction
    result = {"model": MODEL_NAME, "layer": 25, "compute_dtype": "bfloat16",
              "timestamp": datetime.now(timezone.utc).isoformat(), "sampling": domains,
              "max_doc_tokens": args.max_doc_tokens, "skip_leading": args.skip_leading,
              "activation_stats": stats, "free_disk_bytes_after_probe": shutil.disk_usage(existing).free,
              "planned_extraction_tokens": extraction.n_tokens,
              "planned_extraction_float32_bytes": extraction.n_tokens * extraction.hidden_size * 4,
              "caveat": "Small deterministic public-corpus sample, not a worst-case bound. No activations were clipped; no full extraction or AE training was launched."}
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
