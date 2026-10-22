"""Step 2 of the Atlas sibling test: layer-27 residuals at sampled token positions of Atlas chunks.

Streams the FineWeb Atlas `chunks` config in its fixed order (shard 0000, row order, chunk_status == "ok", first
N_CHUNKS chunks: the SAME stream geoae/extraction/atlas_siblings_pairs.py counted labels on), runs Llama-3.2-3B in bfloat16
on each chunk on its own (BOS + chunk text, right-padded batches: exact for a causal LM) and captures the residual
stream at the output of decoder layer 27 with the hook geoae.extraction.sampled uses (the forward is stopped there, so
no logits are computed). Per chunk it keeps up to POSITIONS token positions drawn uniformly from [SKIP, L-1) with a
generator seeded by (seed, chunk index), so any prefix of a longer run is identical to a shorter run.

Rows are stored float16 exactly like the training dump (finiteness asserted). Layout of OUT_DIR:
  rows.npy (R, 3072) f16   rows_tok.npy / rows_nxt.npy (R,) i32 current / next token id   rows_chunk.npy, rows_pos.npy
  chunks.npz  doc_int_id, chunk_id, n_tok, row_start, label_ptr + label_ids (concept ids of ALL types),
              tok_ptr + tok_flat (the chunk's token ids, BOS first), fingerprint (sha256 of the (doc, chunk) stream)
  meta.json   arguments, timings, sanity check of the stored rows against the model's own logits
  pair_counts.json  (with --pairs) exclusive chunk counts of every sibling and null pair and how many pass MIN_EXCL.
Phase 2 (next-token damage) needs only rows + next-token ids because layer 27 is the LAST decoder layer: an edit
there only changes the same position's logits. The chunk token ids are kept anyway for any later full-forward work.

--last_word replaces the sampled positions with ONE row per chunk: the last LM token of the chunk's last word, i.e. the
position that has read the whole chunk up to its final word. Words are NLTK word tokens (NLTKWordTokenizer spans, no
data files needed); trailing tokens without a letter or digit ('.', '"', newlines, ')' ...) are dropped, and the row is
the last LM token whose character span starts before the end of the last remaining word (tokenizer offsets). nltk is
imported from $GEOAE_NLTK_PATH appended to sys.path (after site-packages, so it never shadows the venv). --ref_cache
asserts the chunk stream and token ids equal an existing cache's.

    scripts/py geoae/extraction/atlas_siblings_extract.py --n_chunks 300000 --out_dir /scratch/dogar/geoae/atlas_l27
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geoae.analysis.atlas_siblings as AS  # noqa: E402

LM_NAME = "meta-llama/Llama-3.2-3B"
LAYER = 27


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sanity_check(lm, tokenizer_ids, positions, rows16, dev, layer):
    """Stored fp16 rows -> fp32 final norm + lm_head vs the model's own logits at the same positions."""
    from geoae.evaluation.atlas_geometry import lm_head_fp32
    from geoae.lm_arch import decoder_layers
    norm32, head32 = lm_head_fp32(lm)
    norm32, head32 = norm32.to(dev), head32.to(dev)
    T = max(len(s) for s in tokenizer_ids)
    ids = torch.zeros(len(tokenizer_ids), T, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for b, s in enumerate(tokenizer_ids):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, :len(s)] = 1
    cap = {}
    h = decoder_layers(lm)[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    with torch.no_grad():
        out = lm(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
    h.remove()
    diffs, diffs_stored, same_h = [], [], []
    k = 0
    for b, pos in enumerate(positions):
        for p in pos:
            if p + 1 >= len(tokenizer_ids[b]):               # --include_last row: no next token to score
                k += 1
                continue
            tgt = int(tokenizer_ids[b][p + 1])
            lp_model = torch.log_softmax(out.logits[b, p].float(), -1)[tgt]
            h_same = cap["h"][b, p].float()                           # from the very pass that produced the logits
            h_stored = torch.as_tensor(rows16[k], device=dev).float()  # from the (differently batched) extraction pass
            lp_same = torch.log_softmax(head32(norm32(h_same[None])), -1)[0, tgt]
            lp_stored = torch.log_softmax(head32(norm32(h_stored[None])), -1)[0, tgt]
            diffs.append(abs(float(lp_model - lp_same)))
            diffs_stored.append(abs(float(lp_model - lp_stored)))
            same_h.append(float((h_same - h_stored).abs().max()))
            k += 1
    # max_abs_logprob_diff = fp32 head on the same-pass h vs the model's bf16 head (the check of
    # paper_same_direction_removal.py, expect < 0.05); *_stored adds the bf16 noise between batch compositions.
    return {"max_abs_logprob_diff": float(max(diffs)), "mean_abs_logprob_diff": float(np.mean(diffs)),
            "max_abs_logprob_diff_stored_rows": float(max(diffs_stored)), "mean_abs_logprob_diff_stored_rows": float(np.mean(diffs_stored)),
            "n_positions": len(diffs), "max_abs_h_diff_stored_vs_unstopped_forward": float(max(same_h))}


def last_word_end(text, word_tok):
    """Character end of the text's last word: NLTK word-token spans with trailing tokens that hold no letter or digit
    dropped. Falls back to the last letter/digit character when NLTK cannot align its tokens to the text. -> (end, how)"""
    try:
        spans = list(word_tok.span_tokenize(text))
        how = "nltk"
    except ValueError:
        spans = [(i, i + 1) for i, ch in enumerate(text) if ch.isalnum()][-1:]
        how = "fallback"
    for s, e in reversed(spans):
        if any(ch.isalnum() for ch in text[s:e]):
            return e, how
    return None, "no_word"


def last_word_position(text, offsets, word_tok):
    """LM token position of the last word's last token (see --last_word) -> (position, how). offsets: the tokenizer's
    (start, end) character spans, special tokens (BOS) as empty spans. how: nltk | fallback | no_word (no letter or digit:
    final token) | truncated (the last word lies past --max_len: final kept token)."""
    end, how = last_word_end(text, word_tok)
    n = len(offsets)
    if end is None:
        return n - 1, how
    cand = [i for i, (s, e) in enumerate(offsets) if e > s and s < end]
    if not cand:
        return n - 1, "no_word"
    p = cand[-1]
    if p == n - 1 and offsets[p][1] < end:
        return p, "truncated"
    return p, how


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_chunks", type=int, default=300_000)
    ap.add_argument("--out_dir", default="/scratch/dogar/geoae/atlas_l27")
    ap.add_argument("--atlas_dir", default="/scratch/dogar/geoae/atlas")
    ap.add_argument("--model_name", default=LM_NAME)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--positions", type=int, default=16, help="token positions kept per chunk")
    ap.add_argument("--skip", type=int, default=4, help="leading tokens (BOS + 3) never kept, as in the training dump")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128, help="chunks per forward pass")
    ap.add_argument("--max_len", type=int, default=256, help="tokens per chunk (truncation; ok chunks are <= ~130)")
    ap.add_argument("--pairs", default=None, help="pairs.json: write pair_counts.json for its sibling and null pairs")
    ap.add_argument("--min_excl", type=int, default=100, help="exclusive chunks per member that a pair needs")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="storage dtype of rows.npy. Gemma-3 residuals (mean |h| ~52k, max ~208k) overflow float16: use float32")
    ap.add_argument("--model_dtype", default="bfloat16", choices=["bfloat16", "float32"], help="dtype the LM runs in")
    ap.add_argument("--include_last", action="store_true",
                    help="always keep each chunk's FINAL token position (in addition to --positions sampled ones); "
                         "its rows_nxt is -1. Needed for a sequence-grain view. Default off.")
    ap.add_argument("--last_word", action="store_true",
                    help="ONE row per chunk, at the last token of its last NLTK word (trailing punctuation dropped); "
                         "replaces --positions / --include_last")
    ap.add_argument("--ref_cache", default=None, help="cache dir whose chunk stream and token ids must equal this run's")
    ap.add_argument("--sanity_warn_only", action="store_true", help="a failed logit sanity check warns instead of aborting")
    ap.add_argument("--report_only", action="store_true", help="skip extraction; only (re)write pair_counts.json")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        extract(args, out)
    if args.pairs:
        report(args, out)


def report(args, out):
    doc = json.load(open(args.pairs))
    cache = AS.AtlasCache(out)
    fp = doc["meta"]["stream"]["fingerprint_sha256"]
    if cache.n_chunks == doc["meta"]["stream"]["n_chunks"] and cache.fingerprint != fp:
        raise SystemExit(f"stream fingerprint of the cache ({cache.fingerprint[:12]}) differs from pairs.json ({fp[:12]})")
    d_, c_ = [], []
    for b in AS.iter_atlas_chunks(args.atlas_dir, cache.n_chunks, ["doc_int_id", "chunk_id"]):
        d_ += b["doc_int_id"]
        c_ += b["chunk_id"]
    if AS.stream_fingerprint(d_, c_) != cache.fingerprint:
        raise SystemExit("the cache is not a prefix of the Atlas chunk stream pairs.json was built on")
    np.save(out / "concept_chunk_counts.npy", np.bincount(cache.label_ids, minlength=16790))   # chunks per concept id (all types)
    recs = {"pairs": doc["pairs"], "null_same": [x for x in doc["null_same"] if x], "null_cross": [x for x in doc["null_cross"] if x]}
    res = {"n_chunks": cache.n_chunks, "min_excl": args.min_excl, "pairs_json_sha256": doc["pair_list_sha256"]}
    for name, rs in recs.items():
        rep, _, _ = AS.pair_exclusive_report(cache, rs, args.min_excl)
        res[name] = {"n": len(rs), "n_pass": int(sum(r["passes"] for r in rep)), "records": rep}
        if name == "pairs":
            for t in ("E", "C", "both"):
                idx = [i for i, p in enumerate(rs) if p["tag"] == t]
                res[name][f"n_pass_{t}"] = int(sum(rep[i]["passes"] for i in idx))
                res[name][f"n_{t}"] = len(idx)
    (out / "pair_counts.json").write_text(json.dumps(res))
    log(f"pair counts @ {cache.n_chunks:,} chunks, >= {args.min_excl} exclusive per member: "
        + ", ".join(f"{k} {res[k]['n_pass']}/{res[k]['n']}" for k in recs)
        + "  | by tag " + ", ".join(f"{t} {res['pairs'][f'n_pass_{t}']}/{res['pairs'][f'n_{t}']}" for t in ("E", "C", "both")))


def extract(args, out):
    from geoae.checkpoint import load_tokenizer

    from geoae.checkpoint import load_lm
    from geoae.extraction.sampled import forward_batch
    from geoae.extraction.sampled import register_capture
    from geoae.extraction.sampled import sample_positions
    from geoae.lm_arch import decoder_layers, hidden_size

    dev = torch.device("cuda")
    t_start = time.time()
    tokenizer = load_tokenizer(args.model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    lm = load_lm(args.model_name, device=dev)
    if args.model_name == LM_NAME:
        assert len(decoder_layers(lm)) == 28 and 0 <= args.layer < 28, "expected the 28-layer Llama-3.2-3B"
    else:
        assert 0 <= args.layer < len(decoder_layers(lm)), f"layer {args.layer} out of range for {args.model_name}"
    if args.model_dtype == "float32":
        lm.float()
    isz = np.dtype(args.dtype).itemsize
    D = int(hidden_size(lm))
    log(f"loaded {args.model_name} (d={D}) in {time.time() - t_start:.0f}s; GPU {torch.cuda.get_device_name(0)}")

    n_max = args.n_chunks
    cap_rows = n_max if args.last_word else n_max * (args.positions + int(args.include_last))
    word_tok, lw_how = None, {}
    if args.last_word:
        import os
        if os.environ.get("GEOAE_NLTK_PATH"):
            sys.path.append(os.environ["GEOAE_NLTK_PATH"])
        from nltk.tokenize import NLTKWordTokenizer
        word_tok = NLTKWordTokenizer()
        assert tokenizer.is_fast, "--last_word needs a fast tokenizer (offset mapping)"
    # In RAM and written once at the end: sequential writes are what GPFS is good at (N=300k -> 29.5 GB).
    rows = np.empty((cap_rows, D), dtype=np.dtype(args.dtype))
    r_tok = np.zeros(cap_rows, np.int32)
    r_nxt = np.zeros(cap_rows, np.int32)
    r_chunk = np.zeros(cap_rows, np.int32)
    r_pos = np.zeros(cap_rows, np.int16)

    doc_l, cid_l, ntok_l, lab_l, tok_l, row_start = [], [], [], [], [], [0]
    n_done, n_rows = 0, 0
    captured = handles = None
    sanity = None
    max_abs = 0.0
    t0 = time.time()
    n_fwd = 0
    cols = ["doc_int_id", "chunk_id", "chunk_text", "content_ids", "entity_ids", "tone_ids", "document_ids"]
    for blk in AS.iter_atlas_chunks(args.atlas_dir, n_max, cols):
        texts = blk["chunk_text"]
        tk = tokenizer(texts, truncation=True, max_length=args.max_len, return_offsets_mapping=args.last_word)
        enc = tk["input_ids"]
        base = n_done
        pos_l = []
        for j, ids in enumerate(enc):
            if args.last_word:
                p, how = last_word_position(texts[j], tk["offset_mapping"][j], word_tok)
                lw_how[how] = lw_how.get(how, 0) + 1
                pos_l.append(np.asarray([p], dtype=np.int64))
                continue
            rng = np.random.default_rng([args.seed, base + j])
            pos = sample_positions(len(ids) - 1, args.skip, args.positions, rng)       # next token must exist
            if args.include_last:
                pos = np.append(pos, len(ids) - 1).astype(pos.dtype)    # sampled positions are all < len-1: no duplicate
            pos_l.append(pos)
        slots = []
        r0 = n_rows
        for pos in pos_l:
            slots.append(np.arange(r0, r0 + len(pos)))
            r0 += len(pos)
        # bookkeeping in chunk order
        for j, ids in enumerate(enc):
            doc_l.append(blk["doc_int_id"][j])
            cid_l.append(blk["chunk_id"][j])
            ntok_l.append(len(ids))
            lab_l.append(list(blk["content_ids"][j]) + list(blk["entity_ids"][j]) + list(blk["tone_ids"][j]) + list(blk["document_ids"][j]))
            tok_l.append(np.asarray(ids, dtype=np.int32))
            row_start.append(row_start[-1] + len(pos_l[j]))
        # forward in length-sorted sub-batches (less padding); results land in their own slots
        order = np.argsort([len(x) for x in enc], kind="stable")
        if sanity is None:                                    # before the capture hook that stops the forward is installed
            first = [int(i) for i in order[len(order) // 2:len(order) // 2 + 8]]
        for s in range(0, len(order), args.batch):
            sub = [int(i) for i in order[s:s + args.batch]]
            if captured is None:
                captured, handles = register_capture(lm, [args.layer])
            with torch.no_grad():
                forward_batch(lm, [enc[i] for i in sub], pad_id, dev)
                H = captured[args.layer]                                      # (B, T, D) bf16
                bi = torch.cat([torch.full((len(pos_l[i]),), b, dtype=torch.long) for b, i in enumerate(sub)]).to(dev)
                pi = torch.cat([torch.as_tensor(pos_l[i], dtype=torch.long) for i in sub]).to(dev)
                sel = H[bi, pi].float()
                x16 = sel.to(getattr(torch, args.dtype))
                if not torch.isfinite(x16).all():
                    raise SystemExit(f"non-finite {args.dtype} row: a layer-{args.layer} value exceeds the {args.dtype} range")
                max_abs = max(max_abs, float(x16.abs().max()))
            captured.clear()
            sl = np.concatenate([slots[i] for i in sub])
            rows[sl] = x16.cpu().numpy()
            for i in sub:
                ids = np.asarray(enc[i])
                p = pos_l[i]
                r_tok[slots[i]] = ids[p]
                r_nxt[slots[i]] = np.where(p + 1 < len(ids), ids[np.minimum(p + 1, len(ids) - 1)], -1)
                r_chunk[slots[i]] = base + i
                r_pos[slots[i]] = p
            n_fwd += 1
        if sanity is None:
            for h in handles:
                h.remove()
            captured = handles = None
            sub_ids = [enc[i] for i in first]
            rows16 = np.concatenate([np.asarray(rows[slots[i]]) for i in first])
            sanity = sanity_check(lm, sub_ids, [pos_l[i] for i in first], rows16, dev, args.layer)
            log(f"sanity ({sanity['n_positions']} positions): fp32 head on same-pass h: max |dlogp| = {sanity['max_abs_logprob_diff']:.4f}, "
                f"mean = {sanity['mean_abs_logprob_diff']:.4f} (expect < 0.05; abort > 0.5); on the stored rows: max "
                f"{sanity['max_abs_logprob_diff_stored_rows']:.4f}, mean {sanity['mean_abs_logprob_diff_stored_rows']:.4f}; "
                f"max |h_stored - h_unstopped| = {sanity['max_abs_h_diff_stored_vs_unstopped_forward']:.3g}")
            if sanity["max_abs_logprob_diff"] > 0.5 or sanity["max_abs_logprob_diff_stored_rows"] > 0.5:
                if not args.sanity_warn_only:
                    raise SystemExit("sanity check failed: stored rows do not reproduce the model's logits")
                log("WARNING: sanity check failed (--sanity_warn_only): stored rows do not reproduce the model's logits")
        n_done += len(enc)
        n_rows = r0
        el = time.time() - t0
        log(f"{n_done:,}/{n_max:,} chunks | {n_rows:,} rows | {n_done / el:.0f} chunks/s | ETA {(n_max - n_done) / max(n_done / el, 1e-9) / 60:.1f} min")
    if handles:
        for h in handles:
            h.remove()
    assert n_done == n_max, f"stream ended at {n_done} chunks"
    t_save = time.time()
    np.save(out / "rows.npy", rows[:n_rows])
    log(f"saved rows.npy ({n_rows * D * isz / 1e9:.1f} GB) in {time.time() - t_save:.0f}s")
    del rows
    W = slice(0, n_rows)
    for name, arr in (("rows_tok", r_tok), ("rows_nxt", r_nxt), ("rows_chunk", r_chunk), ("rows_pos", r_pos)):
        np.save(out / f"{name}.npy", arr[W])
    lab_ptr, lab_ids = AS.ragged_to_csr(lab_l)
    tok_ptr = np.zeros(len(tok_l) + 1, dtype=np.int64)
    np.cumsum([len(t) for t in tok_l], out=tok_ptr[1:])
    fp = AS.stream_fingerprint(doc_l, cid_l)
    np.savez(out / "chunks.npz", doc_int_id=np.asarray(doc_l, dtype=np.int32), chunk_id=np.asarray(cid_l, dtype=np.int32),
             n_tok=np.asarray(ntok_l, dtype=np.int32), row_start=np.asarray(row_start, dtype=np.int64),
             label_ptr=lab_ptr, label_ids=lab_ids.astype(np.int32), tok_ptr=tok_ptr, tok_flat=np.concatenate(tok_l),
             fingerprint=np.asarray(fp))
    ref_check = None
    if args.ref_cache:
        z = np.load(Path(args.ref_cache) / "chunks.npz")       # this run must equal the ref's first n_done chunks
        n_ref = len(z["n_tok"])
        n = n_done
        same = (n_ref >= n and (n_ref > n or str(z["fingerprint"]) == fp)
                and np.array_equal(z["doc_int_id"][:n], np.asarray(doc_l)) and np.array_equal(z["chunk_id"][:n], np.asarray(cid_l))
                and np.array_equal(z["n_tok"][:n], np.asarray(ntok_l))
                and np.array_equal(z["tok_flat"][:z["tok_ptr"][n]], np.concatenate(tok_l))
                and np.array_equal(z["label_ids"][:z["label_ptr"][n]], lab_ids))
        ref_check = {"ref_cache": args.ref_cache, "identical_stream_tokens_labels": bool(same)}
        log(f"ref cache {args.ref_cache}: chunk stream, token ids and labels identical = {same}")
        if not same:
            raise SystemExit("this run's chunks differ from --ref_cache")
    el = time.time() - t0
    meta = {"args": vars(args), "n_chunks": n_done, "n_rows": int(n_rows), "dim": D, "dtype": args.dtype, "include_last": bool(args.include_last), "layer": args.layer, "model_name": args.model_name,
            "max_abs_value": max_abs, "stream_fingerprint_sha256": fp, "sanity": sanity,
            "mean_rows_per_chunk": n_rows / n_done, "seconds_extraction": el, "seconds_total": time.time() - t_start,
            "chunks_per_second": n_done / el, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "forward_passes": n_fwd, "model_dtype": args.model_dtype,
            "last_word": bool(args.last_word), "last_word_how": lw_how, "ref_check": ref_check}
    if args.last_word:
        log(f"last-word positions: {lw_how}")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {out} | {n_rows:,} rows ({n_rows * D * isz / 1e9:.2f} GB) | {el:.0f}s extraction ({n_done / el:.0f} chunks/s)")


if __name__ == "__main__":
    main()
