"""
Rebuild the FineWeb-Atlas activation cache consumed by concept_separability.

`guidelabs/fineweb-atlas` labels ~95-token CHUNKS with multi-label concepts
(document-type 31, tone 587, content 12,786, entity 3,386). This streams the
chunks, runs the LM once, and stores per-LM-token layer-L residuals plus an
`owner` array mapping each token to its chunk, so the consumer can pool however
it likes (last / mean / max / token).

Determinism: streaming the first `--n_stream` chunks and keeping
`chunk_status == "ok"` is reproducible, so re-running yields the same chunk set
(7,742 out of the first 8,000) and results stay comparable to earlier caches.

float32 ONLY. A previous gemma cache was written as fp16 and 70% of its tokens
were non-finite, which makes cdist return inf and argmin return 0 — it produced
a plausible-looking "cluster collapse" that cost real time. This asserts
finiteness before writing.

    python -m geoae.interp.atlas_cache --checkpoint <ckpt.pt> \
        --n_stream 8000 --out_acts cache/atlas8k.npz --out_labels cache/atlas8k_labels.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="supplies model_name + target_layer")
    ap.add_argument("--n_stream", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--out_acts", default="cache/atlas8k.npz")
    ap.add_argument("--out_labels", default="cache/atlas8k_labels.parquet")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = ck["config"]["extraction"]["model_name"]
    layer = ck["config"]["data"]["target_layer"]
    print(f"[atlas] {model_name} layer {layer}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from geoae.hooks import SplicingHook
    from geoae.lm_arch import decoder_layers
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(dev).eval()

    cap = {}
    h = decoder_layers(lm)[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    from datasets import load_dataset
    ds = load_dataset("guidelabs/fineweb-atlas", "chunks", split="train", streaming=True)

    texts, rows = [], []
    for i, r in enumerate(ds):
        if i >= args.n_stream:
            break
        if r.get("chunk_status") != "ok":
            continue
        texts.append(r["chunk_text"] if "chunk_text" in r else r["text"])
        rows.append({k: list(r.get(k) or []) for k in
                     ("document_ids", "tone_ids", "content_ids", "entity_ids")})
    print(f"[atlas] {len(texts):,} chunks kept of the first {args.n_stream:,} streamed")
    # Release the streaming dataset before the LM loop. Its teardown races with
    # torch's at interpreter exit (PyGILState_Release fatal error / coredump);
    # dropping it here means a crash cannot happen after the activations are
    # computed but before they are written.
    del ds
    import gc; gc.collect()

    Hs, owner = [], []
    with torch.no_grad():
        for b in range(0, len(texts), args.batch_size):
            bt = texts[b:b + args.batch_size]
            enc = tok(bt, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len).to(dev)
            lm(**enc)
            hs = cap["h"]                                   # (B, T, D)
            m = enc["attention_mask"].bool()
            for j in range(len(bt)):
                v = hs[j][m[j]]
                Hs.append(v.cpu().numpy().astype(np.float32))
                owner.append(np.full(len(v), b + j, dtype=np.int32))
            if b % (args.batch_size * 40) == 0:
                print(f"  {b:>6}/{len(texts)}", flush=True)
    h.remove()

    H = np.concatenate(Hs)
    own = np.concatenate(owner)
    if not np.isfinite(H).all():
        raise SystemExit("[atlas] non-finite activations — abort (see docstring)")
    print(f"[atlas] {H.shape[0]:,} tokens x {H.shape[1]}  max|x| {np.abs(H).max():.1f}")
    np.savez(args.out_acts, H=H, owner=own)
    pd.DataFrame(rows).to_parquet(args.out_labels)
    print(f"[atlas] wrote {args.out_acts} ({H.nbytes/2**30:.1f} GB) and {args.out_labels}")


if __name__ == "__main__":
    main()
