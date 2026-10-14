"""
Sampled-position extraction (`extraction.mode: sampled`).

The legacy extractor (geoae/extract.py, mode "prefix") keeps EVERY position
4..255 of each document: 10M tokens = ~43k documents, nothing ever seen past
position 255, and a hardcoded 5-way equal mix (43% code+math). Topic- and fact-
bearing state accumulates later in a document than syntax does, so that dump
under-represents exactly what the RAVEL/topic probes measure.

This mode instead:
  * FULL DOCUMENTS: runs each document up to `context_len` tokens (2048 default)
    and keeps a uniform random sample of `positions_per_doc` positions from
    [skip_leading, L). Same token budget, ~3.5x more documents, positions up to
    context_len. Short docs contribute all their positions, so the natural
    length distribution is preserved.
  * TOKEN-QUOTA MIX from `extraction.sources` (weights), not round-robin. Each
    next document comes from the source furthest below its quota.
  * FAIL CLOSED: a source that does not load aborts the run (the legacy
    extractor skipped it and silently changed the mix).
  * SHUFFLED STREAMS: every HF stream goes through `.shuffle(seed, buffer)`, so
    e.g. Wikipedia is not "the first N articles by page id".
  * ROW SIDECARS: rows_doc / rows_pos / rows_tok / rows_src / rows_norm .npy
    beside layer_L.npy, plus docs.npy (per-doc source, length, rows kept) for
    position/outlier analysis and document-level bookkeeping.
  * OUTLIER REPORT (always) and optional FILTER (`outlier_norm_mult` > 0 drops
    rows whose target-layer norm exceeds mult x the calibrated median norm).
  * BATCHED forward with right padding (exact for causal LMs), and the forward
    is STOPPED after the last requested layer, so the LM head / vocab logits
    (B x 2048 x 128k) are never computed.

Rows from one document stay contiguous, so the train.py contiguous-tail val
split is document-disjoint except for at most one straddling document (<=
positions_per_doc rows); the meta reports that document.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from geoae.lm_arch import decoder_layers, describe, hidden_size as lm_hidden_size

POS_BINS = [0, 16, 64, 128, 256, 512, 1024, 2048, 4096, 1 << 30]


class _StopForward(Exception):
    pass


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def sample_positions(L: int, skip: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted positions: all of [skip, L) if it has <= k, else a uniform k-subset."""
    n = L - skip
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= k:
        return np.arange(skip, L, dtype=np.int64)
    return np.sort(rng.choice(n, size=k, replace=False)) + skip


class QuotaScheduler:
    """Picks the live source with the largest remaining fraction of its token quota."""

    def __init__(self, weights: list[float], total: int):
        w = np.asarray(weights, dtype=np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"source weights must be >= 0 with a positive sum, got {weights}")
        self.target = w / w.sum() * total
        self.got = np.zeros(len(w))
        self.dead = self.target <= 0

    def next(self) -> int | None:
        rem = np.where(self.dead, -np.inf, (self.target - self.got) / np.maximum(self.target, 1))
        i = int(np.argmax(rem))
        return None if not np.isfinite(rem[i]) else i

    def add(self, i: int, n: int) -> None:
        self.got[i] += n

    def kill(self, i: int) -> None:
        self.dead[i] = True


def _get_path(record: dict, dotted: str):
    from geoae.structured_corpus import _as_obj
    cur = record
    for k in dotted.split("."):
        cur = _as_obj(cur)
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def open_source(src: dict, seed: int, shuffle_buffer: int, stats: dict):
    """-> iterator of document strings. Raises if the source cannot be read."""
    kind = src.get("kind", "hf")
    if kind == "structured":
        from geoae.structured_corpus import iter_structured_docs
        it = iter_structured_docs(seed=seed, items_per_doc=src.get("items_per_doc", 8),
                                  stats=stats)
    elif kind == "hf":
        from datasets import load_dataset
        ds = load_dataset(src["name"], name=src.get("config"), split=src.get("split", "train"),
                          streaming=True)
        if shuffle_buffer > 0:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        field = src.get("field", "text")
        excl = src.get("exclude")          # {"key": "meta.pile_set_name", "values": [...]}
        excl_vals = set(excl["values"]) if excl else set()

        def gen():
            for r in ds:
                if excl and _get_path(r, excl["key"]) in excl_vals:
                    stats["excluded"] = stats.get("excluded", 0) + 1
                    continue
                t = r.get(field)
                if t:
                    yield t
        it = gen()
    else:
        raise ValueError(f"unknown source kind {kind!r}")
    first = next(it, None)                 # fail here, not hours in
    if first is None:
        raise RuntimeError(f"source {src} yielded no documents")

    def chained():
        yield first
        yield from it
    return chained()


class Prefetcher:
    """One daemon thread per source: pulls documents and tokenises them ahead of
    the GPU, into a bounded queue. Streaming + tokenising otherwise runs serially
    with the forward and costs ~1/3 of wall time. Errors re-raise in the caller."""

    _END = object()

    def __init__(self, doc_iters, tokenize, depth: int = 64):
        import queue
        import threading
        self.queues = [queue.Queue(maxsize=depth) for _ in doc_iters]
        for it, q in zip(doc_iters, self.queues):
            threading.Thread(target=self._run, args=(it, q, tokenize), daemon=True).start()

    def _run(self, it, q, tokenize):
        try:
            for text in it:
                q.put(tokenize(text))
            q.put(self._END)
        except BaseException as e:           # surfaced on the next get()
            q.put(e)

    def get(self, i: int):
        """-> token id list, or None when source i is exhausted."""
        x = self.queues[i].get()
        if x is self._END:
            self.queues[i].put(self._END)      # stay exhausted on repeat calls
            return None
        if isinstance(x, BaseException):
            raise RuntimeError(f"source {i} failed while streaming") from x
        return x


# ---------------------------------------------------------------------------
# Forward with early stop
# ---------------------------------------------------------------------------

def register_capture(model, layers: list[int]):
    captured: dict[int, torch.Tensor] = {}
    blocks = decoder_layers(model)
    bad = [l for l in layers if not 0 <= l < len(blocks)]
    if bad:
        raise ValueError(f"Layer(s) {bad} out of range (0..{len(blocks) - 1})")
    last = max(layers)
    handles = []
    for l in layers:
        def make(idx):
            def hook(module, inp, out):
                captured[idx] = (out[0] if isinstance(out, tuple) else out).detach()
                if idx == last:
                    raise _StopForward
            return hook
        handles.append(blocks[l].register_forward_hook(make(l)))
    return captured, handles


def forward_batch(model, seqs: list[list[int]], pad_id: int, device) -> None:
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, :len(s)] = 1
    try:
        model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    except _StopForward:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def extract_sampled(model_name: str, layers: list[int], target_layer: int, n_tokens: int,
                    hidden_size: int, out_dir: Path, sources: list[dict], context_len: int,
                    positions_per_doc: int, skip_leading: int, min_doc_len: int,
                    batch_docs: int, shuffle_buffer: int, outlier_norm_mult: float,
                    dtype: str, seed: int, log_every: int, corpus_dir=None,
                    val_frac_report: float = 0.05, allow_exhaust: bool = False,
                    model=None, tokenizer=None, doc_iters=None):
    """`model`/`tokenizer`/`doc_iters` are injectable for tests; normally loaded here."""
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    if dtype not in ("float16", "float32"):
        raise ValueError(f"dtype must be float16 or float32, got {dtype!r}")
    if not sources:
        raise ValueError("extraction.mode=sampled needs a non-empty extraction.sources list")
    np_dtype = np.dtype(dtype)
    tl = target_layer if target_layer in layers else layers[-1]

    need = n_tokens * hidden_size * np_dtype.itemsize * len(layers)
    free = shutil.disk_usage(out_dir).free
    if need > free * 0.98:
        raise RuntimeError(f"Need {need/1e9:.0f} GB but only {free/1e9:.0f} GB free in {out_dir}")

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if model is None:
        from geoae.checkpoint import load_lm
        model = load_lm(model_name, device_map="auto")
        print(f"[extract] {describe(model)}")
    device = next(model.parameters()).device
    if hidden_size != lm_hidden_size(model):
        raise ValueError(f"hidden_size={hidden_size} but model has d={lm_hidden_size(model)}")

    # ---- sources: the frozen corpus from geoae.build_corpus (fail closed) ----
    src_stats = [dict() for _ in sources]
    corpus_manifest = None
    if doc_iters is None:
        from geoae.build_corpus import iter_corpus_file
        if not corpus_dir:
            raise ValueError("extraction.corpus_dir is not set; build it first with "
                             "`python -m geoae.build_corpus --config <same config>`")
        corpus_dir = Path(corpus_dir)
        man = corpus_dir / "manifest.json"
        if not man.exists():
            raise FileNotFoundError(f"{man} missing — run geoae.build_corpus with this config first")
        corpus_manifest = json.load(open(man))
        built = {r["domain"]: r for r in corpus_manifest["sources"]}
        from geoae.build_corpus import spec_key
        wsum = sum(x["weight"] for x in sources)
        doc_iters = []
        for s in sources:
            r = built.get(s["domain"])
            if r is None or spec_key(r["spec"]) != spec_key(s):
                raise ValueError(f"corpus {corpus_dir} has no / a different spec for source "
                                 f"{s['domain']!r}; run geoae.build_corpus with this config")
            if r["est_rows"] < s["weight"] / wsum * n_tokens:
                raise ValueError(f"corpus source {s['domain']!r} holds ~{r['est_rows']:,} rows, "
                                 f"below this config's quota; rebuild with a larger margin")
            doc_iters.append(iter_corpus_file(corpus_dir / r["file"]))
            print(f"[extract]   + {s['domain']:<10} w={s['weight']:<5} {r['docs']:>8,} docs")
        if corpus_manifest["context_len"] != context_len or \
                corpus_manifest["positions_per_doc"] != positions_per_doc or \
                corpus_manifest["skip_leading"] != skip_leading or \
                corpus_manifest["n_tokens"] < n_tokens:
            raise ValueError("corpus was built for different context_len / positions_per_doc / "
                             "skip_leading / smaller n_tokens; rebuild it")
    sched = QuotaScheduler([s["weight"] for s in sources], n_tokens)
    rng = np.random.default_rng(seed)

    # Cut the TEXT before tokenising: a PG-19 book is ~1M chars, and tokenising it
    # in full just to keep 2048 tokens costs seconds and hundreds of MB. No
    # tokenizer averages under 1 char/token on real text, so 16 chars/token of
    # headroom never shortens a doc below context_len.
    max_chars = context_len * 16
    prefetch = Prefetcher(doc_iters, lambda t: tokenizer(t[:max_chars], truncation=True,
                                                         max_length=context_len)["input_ids"],
                          depth=32)
    captured, handles = register_capture(model, layers)
    mmaps = {l: np.lib.format.open_memmap(str(out_dir / f"layer_{l}.npy"), mode="w+",
                                          dtype=np_dtype, shape=(n_tokens, hidden_size))
             for l in layers}
    r_doc = np.empty(n_tokens, np.int32); r_pos = np.empty(n_tokens, np.int32)
    r_tok = np.empty(n_tokens, np.int32); r_src = np.empty(n_tokens, np.uint8)
    r_norm = np.empty(n_tokens, np.float32)
    docs: list[tuple[int, int, int]] = []          # (src, L, rows_kept)

    planned = 0; written = 0; dropped = 0; median = None
    skipped_short = Counter()
    t0 = time.time(); last_log = 0
    pool_size = max(1, batch_docs) * 16

    def pull_pool():
        nonlocal planned
        pool = []
        while len(pool) < pool_size and planned < n_tokens:
            i = sched.next()
            if i is None:
                break
            ids = prefetch.get(i)
            if ids is None:
                if not allow_exhaust:
                    raise RuntimeError(
                        f"source {sources[i].get('domain')!r} ran out at {sched.got[i]:,.0f} of "
                        f"{sched.target[i]:,.0f} rows — the mix would drift. Rebuild the corpus "
                        f"with a larger corpus_margin.")
                print(f"[extract]   ! source {i} ({sources[i].get('domain')}) exhausted")
                sched.kill(i)
                continue
            L = len(ids)
            if L < max(min_doc_len, skip_leading + 1):
                skipped_short[i] += 1
                continue
            pos = sample_positions(L, skip_leading, positions_per_doc, rng)
            pos = pos[:max(0, n_tokens - planned)]
            if len(pos) == 0:
                break
            planned += len(pos); sched.add(i, len(pos))
            pool.append((i, ids, pos))
        return pool

    with torch.no_grad():
        while written < n_tokens:
            pool = pull_pool()
            if not pool:
                break
            pool.sort(key=lambda d: len(d[1]))
            for b0 in range(0, len(pool), batch_docs):
                chunk = pool[b0:b0 + batch_docs]
                forward_batch(model, [c[1] for c in chunk], pad_id, device)
                if median is None:                     # calibrate on the first batch
                    median = float(np.median(np.concatenate([
                        captured[tl][bi].index_select(0, torch.as_tensor(p, device=device))
                        .float().norm(dim=-1).cpu().numpy() for bi, (_, _, p) in enumerate(chunk)])))
                for bi, (si, ids, pos) in enumerate(chunk):
                    pt = torch.as_tensor(pos, device=device)
                    rows = {l: captured[l][bi].index_select(0, pt).float() for l in layers}
                    norms = rows[tl].norm(dim=-1).cpu().numpy()
                    keep = np.ones(len(pos), bool)
                    if outlier_norm_mult > 0 and median:
                        keep = norms <= outlier_norm_mult * median
                        nd = int((~keep).sum())
                        if nd:
                            dropped += nd; sched.add(si, -nd); planned -= nd
                    n = int(keep.sum())
                    n = min(n, n_tokens - written)
                    if n <= 0:
                        continue
                    kidx = np.flatnonzero(keep)[:n]
                    kt = torch.as_tensor(kidx, device=device)
                    for l in layers:
                        blk = rows[l].index_select(0, kt).to(getattr(torch, dtype)).cpu()
                        if dtype == "float16" and not torch.isfinite(blk).all():
                            raise RuntimeError(f"float16 overflow at layer {l}; use dtype float32")
                        mmaps[l][written:written + n] = blk.numpy()
                    sl = slice(written, written + n)
                    r_doc[sl] = len(docs); r_pos[sl] = pos[kidx]
                    r_tok[sl] = np.asarray(ids)[pos[kidx]]; r_src[sl] = si
                    r_norm[sl] = norms[kidx]
                    docs.append((si, len(ids), n))
                    written += n
                captured.clear()
            if len(docs) - last_log >= log_every:
                last_log = len(docs)
                rate = written / max(time.time() - t0, 1e-6)
                mix = {sources[i].get("domain", i): f"{100*g/max(written,1):.0f}%"
                       for i, g in enumerate(sched.got) if g}
                print(f"[extract] {written:>10,}/{n_tokens:,} ({100*written/n_tokens:.1f}%) "
                      f"| {len(docs):,} docs | {rate:.0f} rows/s "
                      f"| ETA {(n_tokens-written)/max(rate,1)/60:.0f} min | {mix}", flush=True)

    for h in handles:
        h.remove()
    for l in layers:
        mmaps[l].flush()
    del mmaps
    if written < n_tokens:
        from geoae.extract import trim_npy
        print(f"[extract] Sources exhausted at {written:,}/{n_tokens:,}; trimming.")
        for l in layers:
            trim_npy(out_dir / f"layer_{l}.npy", written)
    for stale in out_dir.glob("norm_params_layer*.npz"):
        stale.unlink()

    W = slice(0, written)
    for name, arr in (("rows_doc", r_doc), ("rows_pos", r_pos), ("rows_tok", r_tok),
                      ("rows_src", r_src), ("rows_norm", r_norm)):
        np.save(out_dir / f"{name}.npy", arr[W])
    docs_arr = np.array(docs, dtype=[("src", np.uint8), ("length", np.int32), ("rows", np.int32)])
    np.save(out_dir / "docs.npy", docs_arr)

    meta = _meta(model_name, layers, tl, written, hidden_size, sources, src_stats, docs_arr,
                 r_pos[W], r_tok[W], r_src[W], r_norm[W], r_doc[W], median, dropped,
                 skipped_short, tokenizer, locals_cfg=dict(
                     mode="sampled", context_len=context_len, positions_per_doc=positions_per_doc,
                     skip_leading=skip_leading, min_doc_len=min_doc_len, batch_docs=batch_docs,
                     shuffle_buffer=shuffle_buffer, outlier_norm_mult=outlier_norm_mult,
                     dtype=dtype, seed=seed, corpus_dir=str(corpus_dir) if corpus_dir else None,
                     corpus_manifest=corpus_manifest), val_frac=val_frac_report,
                 elapsed=time.time() - t0)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[extract] Done. {written:,} rows / {len(docs):,} docs -> {out_dir}/")
    print(f"[extract] Source mix: {meta['source_share']}")
    print(f"[extract] Positions: {meta['position_hist']}")
    print(f"[extract] Norm outliers: {meta['outliers']['frac_over']}")
    return meta


def _meta(model_name, layers, tl, written, hidden_size, sources, src_stats, docs_arr,
          pos, tok, src, norm, doc, median, dropped, skipped_short, tokenizer,
          locals_cfg, val_frac, elapsed):
    names = [s.get("domain", str(i)) for i, s in enumerate(sources)]
    share = Counter(src.tolist())
    hist, _ = np.histogram(pos, bins=POS_BINS)
    pos_hist = {f"[{a},{b})": int(h) for a, b, h in zip(POS_BINS[:-1], POS_BINS[1:], hist) if h}
    med = float(np.median(norm)) if written else 0.0
    outl = {"median_norm": med, "calibrated_median": median, "dropped_rows": int(dropped),
            "frac_over": {f">{m}x": float((norm > m * med).mean()) if written else 0.0
                          for m in (3, 5, 10, 20)}}
    big = norm > 10 * med if written else np.zeros(0, bool)
    if big.any():
        top = Counter(tok[big].tolist()).most_common(20)
        outl["top_tokens_over_10x"] = [(tokenizer.decode([t]), c) for t, c in top]
        h, _ = np.histogram(pos[big], bins=POS_BINS)
        outl["positions_over_10x"] = {f"[{a},{b})": int(x)
                                      for a, b, x in zip(POS_BINS[:-1], POS_BINS[1:], h) if x}
    vs = int(written * (1.0 - val_frac))
    straddle = None
    if 0 < vs < written:
        d = int(doc[vs])
        rows_d = np.flatnonzero(doc == d)
        straddle = {"doc": d, "rows_train": int((rows_d < vs).sum()),
                    "rows_val": int((rows_d >= vs).sum())}
    return {
        "model": model_name, "layers": layers, "target_layer": tl, "n_tokens": int(written),
        "hidden_size": hidden_size, "n_docs": int(len(docs_arr)), **locals_cfg,
        "per_doc_forward": True,
        "sources": [{**{k: v for k, v in s.items()}, "stats": st,
                     "docs": int((docs_arr["src"] == i).sum()),
                     "rows": int(share.get(i, 0)),
                     "skipped_short": int(skipped_short.get(i, 0))}
                    for i, (s, st) in enumerate(zip(sources, src_stats))],
        "source_share": {names[i]: round(share.get(i, 0) / max(written, 1), 4)
                         for i in range(len(sources))},
        "doc_length": {"mean": float(docs_arr["length"].mean()) if len(docs_arr) else 0,
                       "median": float(np.median(docs_arr["length"])) if len(docs_arr) else 0,
                       "frac_at_context_len": float((docs_arr["length"] >= locals_cfg["context_len"]).mean())
                       if len(docs_arr) else 0},
        "position_hist": pos_hist,
        "outliers": outl,
        "val_split_report": {"val_frac": val_frac, "val_start": vs, "straddling_doc": straddle},
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
    }
