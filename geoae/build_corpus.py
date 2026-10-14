"""
Freeze the document corpus for sampled extraction (step 1 of 2).

    python -u -m geoae.build_corpus --config configs/base/llama3.2-3b_extract_sampled.yaml
    python -u -m geoae.extract      --config configs/base/llama3.2-3b_extract_sampled.yaml

WHY A SEPARATE STEP. Holding 9 Hugging Face streams open in one process costs
~60 GB RSS before a single forward pass (Parquet/Arrow read buffers: fineweb
~8 GB, pile ~15 GB, each Wikipedia 6-8 GB), and grows. So each source is
streamed ONCE, in its own subprocess (memory is returned to the OS when it
exits), into  <corpus_dir>/<domain>.jsonl.gz  — one {"text": ...} per line, in
the stream's shuffled order, text cut to context_len * 16 chars. Extraction
then reads local files at ~no memory cost.

Side benefits: the exact documents behind an activation dump are on disk and
inspectable; the same corpus can be re-extracted for another model (Gemma) so
the data is held fixed across models; rebuilds are deterministic in `seed`.

HOW MUCH PER SOURCE. Each source must supply  weight/sum(w) * n_tokens  rows.
While streaming, every doc is tokenised with the extraction model's tokenizer
and its row yield  min(positions_per_doc, L - skip_leading)  is summed; the
source stops once that sum reaches  corpus_margin x  its quota (default 1.3,
headroom for the outlier filter). The yield estimate uses the same tokenizer,
context_len and skip as extraction, so it is exact up to that margin.

A source that fails to open or runs dry before its quota is an ERROR — the
mix is never silently changed. `manifest.json` records per-source counts,
estimated rows, the source spec and the structured-corpus guard stats.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path


def _build_one(args) -> dict:
    """Runs in a fresh subprocess: stream one source into its jsonl.gz."""
    (i, src, quota, cfg) = args
    from transformers import AutoTokenizer
    from geoae.extract_sampled import open_source

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    max_chars = cfg["context_len"] * 16
    stats: dict = {}
    t0 = time.time()
    it = open_source(src, cfg["seed"] + i, src.get("shuffle_buffer", cfg["shuffle_buffer"]),
                     stats)
    out = Path(cfg["corpus_dir"]) / f"{src['domain']}.jsonl.gz"
    tmp = out.with_suffix(".tmp")
    need = quota * cfg["margin"]
    rows = docs = short = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for text in it:
            text = text[:max_chars]
            L = len(tok(text, truncation=True, max_length=cfg["context_len"])["input_ids"])
            if L < max(cfg["min_doc_len"], cfg["skip_leading"] + 1):
                short += 1
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            docs += 1
            rows += min(cfg["positions_per_doc"], L - cfg["skip_leading"])
            if docs % 5000 == 0:
                print(f"[corpus]   {src['domain']:<10} {docs:>7,} docs  {rows:>10,}/{int(need):,} rows"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            if rows >= need:
                break
    if rows < need:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"source {src['domain']} ran dry: {rows:,} of {int(need):,} rows")
    tmp.rename(out)
    print(f"[corpus] + {src['domain']:<10} {docs:,} docs, ~{rows:,} rows "
          f"({time.time()-t0:.0f}s) -> {out.name}", flush=True)
    return {"domain": src["domain"], "file": out.name, "docs": docs, "est_rows": rows,
            "quota_rows": int(quota), "skipped_short": short, "stats": stats,
            "spec": src, "seconds": round(time.time() - t0, 1)}


def spec_key(src: dict) -> str:
    """A source's identity minus its weight: two configs that differ only in
    weights (e.g. the struct5 twin) share one corpus file per source."""
    return json.dumps({k: v for k, v in src.items() if k != "weight"}, sort_keys=True)


def build_corpus(ex, seed: int, workers: int = 3) -> dict:
    if not ex.sources:
        raise ValueError("extraction.sources is empty")
    if not ex.corpus_dir:
        raise ValueError("extraction.corpus_dir is not set")
    domains = [s["domain"] for s in ex.sources]
    if len(set(domains)) != len(domains):
        raise ValueError(f"source domains must be unique (they name files): {domains}")
    from geoae.paths import resolve_path
    corpus_dir = resolve_path(ex.corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    wsum = sum(s["weight"] for s in ex.sources)
    cfg = dict(model_name=ex.model_name, context_len=ex.context_len, seed=seed,
               positions_per_doc=ex.positions_per_doc, skip_leading=ex.skip_leading,
               min_doc_len=ex.min_doc_len, shuffle_buffer=ex.shuffle_buffer,
               margin=ex.corpus_margin, corpus_dir=str(corpus_dir))
    # INCREMENTAL: keep any source already built with the same spec, seed and
    # geometry and enough rows for this config's quota; build only the rest.
    man_path = corpus_dir / "manifest.json"
    old = json.load(open(man_path)) if man_path.exists() else None
    geom = ("model_tokenizer", "seed", "context_len", "positions_per_doc", "skip_leading")
    cur = dict(model_tokenizer=ex.model_name, seed=seed, context_len=ex.context_len,
               positions_per_doc=ex.positions_per_doc, skip_leading=ex.skip_leading)
    kept = {}
    if old and all(old.get(k) == cur[k] for k in geom):
        for r in old["sources"]:
            if (corpus_dir / r["file"]).exists():
                kept[r["domain"]] = r
    elif old:
        raise ValueError(f"{man_path} was built with different tokenizer/seed/geometry; "
                         f"use a new corpus_dir")
    jobs = []
    for i, s in enumerate(ex.sources):
        quota = s["weight"] / wsum * ex.n_tokens
        r = kept.get(s["domain"])
        if s["weight"] <= 0:
            continue
        if r and spec_key(r["spec"]) == spec_key(s) and r["est_rows"] >= quota:
            print(f"[corpus] = {s['domain']:<10} reuse ({r['docs']:,} docs, ~{r['est_rows']:,} rows)")
            continue
        kept.pop(s["domain"], None)
        jobs.append((i, s, quota, cfg))
    print(f"[corpus] {len(jobs)} sources -> {corpus_dir}  (workers={workers}, "
          f"margin={ex.corpus_margin}, n_tokens={ex.n_tokens:,})", flush=True)
    results = []
    if jobs:
        ctx = mp.get_context("spawn")
        # maxtasksperchild=1: each source gets a fresh process, so Arrow memory is freed.
        with ctx.Pool(processes=min(workers, len(jobs)), maxtasksperchild=1) as pool:
            results = pool.map(_build_one, jobs, chunksize=1)
    results = list(kept.values()) + results
    manifest = {
        "model_tokenizer": ex.model_name, "seed": seed,
        "n_tokens": max(ex.n_tokens, old["n_tokens"]) if old else ex.n_tokens,
        "context_len": ex.context_len, "positions_per_doc": ex.positions_per_doc,
        "skip_leading": ex.skip_leading, "margin": ex.corpus_margin,
        "sources": results, "built": datetime.now(timezone.utc).isoformat(),
    }
    with open(corpus_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[corpus] Done: {sum(r['docs'] for r in results):,} docs -> {corpus_dir}/manifest.json")
    return manifest


def iter_corpus_file(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3,
                    help="sources streamed concurrently (each peaks at up to ~15 GB RSS)")
    ap.add_argument("--n_tokens", type=int, default=None)
    ap.add_argument("--corpus_dir", default=None)
    a = ap.parse_args()
    from geoae.config import Config
    ex = Config.from_yaml(a.config).extraction
    if a.n_tokens is not None:
        ex.n_tokens = a.n_tokens
    if a.corpus_dir is not None:
        ex.corpus_dir = a.corpus_dir
    build_corpus(ex, a.seed, a.workers)


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)     # HF streaming threads don't join cleanly
