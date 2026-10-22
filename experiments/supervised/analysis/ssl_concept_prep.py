"""Freeze fresh Atlas passages for the focused SSL separation/control experiment.

This is a CPU-only preparation step. It verifies the discovery cache against the
original ordered Atlas stream, excludes every document in the skipped prefix,
and never decodes/reuses the discovery activations as a fresh evaluation set.
The output JSONL is consumed by the activation extractor, independently of any
model or intervention outcome. Document identity fixes fit/calibration/test.

    python experiments/supervised/analysis/ssl_concept_prep.py --manifest results/.../manifest.json \
        --discovery-cache /scratch/.../atlas_l27 --atlas-dir /scratch/.../atlas \
        --out-dir results/.../fresh --max-source-chunks 2000000

Missing shards are an explicit failure unless --download-missing is supplied.
The JSONL includes one passage per document and pair. Random-control
passages have label -1 and contain none of the selected Atlas concepts or native
headers. Their source document and split are shared across pairs.
"""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import geoae.analysis.atlas_siblings as AS  # noqa: E402

SPLITS = ("fit", "calibration", "test")
DEFAULT_PAIRS = [
    {"pair_id": "cooking", "a_ids": [2401], "b_ids": [2404, 2405],
     "a_name": "Cooking implements", "b_name": "Cooking techniques"},
    {"pair_id": "graphics", "a_ids": [5034], "b_ids": [5038],
     "a_name": "Graphics hardware", "b_name": "Graphics rendering"},
    {"pair_id": "construction", "a_ids": [2276], "b_ids": [5811],
     "a_name": "Construction project management", "b_name": "Infrastructure management"},
]
NATIVE_PAIR = {"pair_id": "native_newline", "kind": "native", "a_name": "References",
               "b_name": "External links", "a_header": "References", "b_header": "External links"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_text_hash(text: str) -> str:
    """Suppress repeated passages across documents, including formatting variants."""
    return sha256_text(" ".join(unicodedata.normalize("NFKC", text).casefold().split()))


def document_split(source_doc_id: str, seed: int = 0) -> str:
    """A globally fixed document split: 50% fit, 25% calibration, 25% test."""
    value = int.from_bytes(hashlib.sha256(f"ssl-concepts:{seed}:{source_doc_id}".encode()).digest()[:8], "big")
    return SPLITS[0 if value % 4 < 2 else value % 4 - 1]


def normalize_pairs(manifest: dict, include_native: bool = False) -> list[dict]:
    pairs = manifest.get("pairs", manifest.get("candidates", DEFAULT_PAIRS))
    if isinstance(pairs, dict):
        pairs = [dict(p, pair_id=k) for k, p in pairs.items()]
    result = []
    for i, pair in enumerate(pairs):
        p = dict(pair)
        p.setdefault("pair_id", p.get("id", f"pair_{i}"))
        p.setdefault("kind", "atlas")
        if p["kind"] == "atlas":
            for side in ("a", "b"):
                p[f"{side}_ids"] = sorted({int(x) for x in p.get(f"{side}_ids", [p[side]] if side in p else [])})
                if not p[f"{side}_ids"]:
                    raise ValueError(f"{p['pair_id']}: {side}_ids is empty")
                p.setdefault(f"{side}_name", side.upper())
            if set(p["a_ids"]) & set(p["b_ids"]):
                raise ValueError(f"{p['pair_id']}: concept memberships overlap")
        elif p["kind"] == "native":
            for side in ("a", "b"):
                p.setdefault(f"{side}_header", p.get(f"{side}_name", NATIVE_PAIR[f"{side}_header"]))
                p.setdefault(f"{side}_name", p[f"{side}_header"])
        else:
            raise ValueError(f"Unsupported pair kind {p['kind']}")
        result.append(p)
    if include_native and not any(p["kind"] == "native" for p in result):
        result.append(dict(NATIVE_PAIR))
    if not result or len({p["pair_id"] for p in result}) != len(result):
        raise ValueError("Pair IDs must be nonempty and unique")
    return result


def native_prefix(text: str, pair: dict, min_chars: int = 24) -> tuple[int, str] | None:
    """Exact isolated header followed by a newline, with real preceding context.

    The prefix retains its original characters. A tokenizer must subsequently
    verify that its final token is the same newline ID in both groups; this CPU
    step makes no assumption that character identity implies token identity.
    Reject chunks containing both headings so labels are exclusive.
    """
    matches = []
    for label, side in enumerate(("a", "b")):
        header = re.escape(pair[f"{side}_header"])
        found = list(re.finditer(rf"(?mi)^[ \t]*{header}[ \t]*\r?\n", text))
        if found:
            matches.append((label, found[0]))
    if len(matches) != 1:
        return None
    label, match = matches[0]
    prefix = text[:match.end()]
    return (label, prefix) if len(prefix) >= min_chars else None


@dataclass
class Discovery:
    doc: np.ndarray
    chunk: np.ndarray
    fingerprint: str

    @property
    def n_chunks(self) -> int:
        return len(self.doc)

    @classmethod
    def load(cls, path: str | Path) -> "Discovery":
        path = Path(path)
        source = path / "chunks.npz" if resolve_input_path(path).is_dir() else path
        if not resolve_input_path(source).is_file():
            raise FileNotFoundError(f"Discovery chunks.npz missing: {source}; freshness cannot be verified")
        with np.load(resolve_input_path(source), allow_pickle=False) as z:
            d = cls(z["doc_int_id"].astype(np.int64), z["chunk_id"].astype(np.int64), str(z["fingerprint"]))
        actual = AS.stream_fingerprint(d.doc, d.chunk)
        if actual != d.fingerprint:
            raise ValueError(f"Discovery cache fingerprint fails its own document/chunk arrays: {actual}")
        return d


def source_records(atlas_dir: str | Path, max_records: int, download_missing: bool = False,
                   source_jsonl: str | Path | None = None, batch_rows: int = 8192) -> Iterable[dict]:
    """Same retained-row order as AS.iter_atlas_chunks, with bounded shard reads."""
    if source_jsonl is not None:
        count = 0
        with resolve_input_path(Path(source_jsonl)).open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("chunk_status", "ok") != "ok":
                    continue
                yield row
                count += 1
                if count >= max_records:
                    return
        return
    import pyarrow.parquet as pq

    columns = ["chunk_status", "doc_int_id", "chunk_id", "chunk_text", "content_ids", "entity_ids",
               "tone_ids", "document_ids"]
    count = 0
    for shard_index in range(57):
        path = AS.shard_file(atlas_dir, shard_index)
        if not resolve_input_path(path).is_file():
            if not download_missing:
                raise FileNotFoundError(f"Fresh source needs missing shard {path}; use --download-missing to fetch it")
            path = AS.ensure_shard(atlas_dir, shard_index)
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        essential = {"chunk_status", "doc_int_id", "chunk_id", "chunk_text", "content_ids"}
        if not essential <= available:
            raise ValueError(f"Atlas shard lacks columns {sorted(essential - available)}: {path}")
        cols = [c for c in columns if c in available]
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=cols):
            block = batch.to_pydict()
            for i, status in enumerate(block["chunk_status"]):
                if status != "ok":
                    continue
                yield {c: block[c][i] for c in cols}
                count += 1
                if count >= max_records:
                    return


def collect_records(source: Iterable[dict], discovery: Discovery, pairs: list[dict], *,
                    start_index: int = 300_000, max_source_chunks: int = 2_000_000,
                    caps: dict[str, int] | None = None, control_cap: int = 32,
                    seed: int = 0, min_text_chars: int = 24) -> tuple[list[dict], dict]:
    """Select in source order before any activation or intervention is measured.

    Prefix equality verifies doc_int_id identity and ordering directly. All
    prefix documents and normalized texts are excluded, including documents
    whose later chunks cross the prefix boundary. One source may legitimately
    enter several different pairs, always under the same split.
    """
    caps = dict(caps or {"fit": 50, "calibration": 30, "test": 50})
    if set(caps) != set(SPLITS) or min(caps.values()) < 1 or control_cap < 0 or max_source_chunks < 1:
        raise ValueError("All three split caps must be positive, control cap nonnegative, scan bound positive")
    start_index = max(start_index, discovery.n_chunks)
    excluded_docs = set(int(d) for d in discovery.doc)
    excluded_hashes: set[str] = set()
    seen_hashes: set[str] = set()
    selected_prefix_hashes: set[str] = set()
    counts: Counter = Counter()
    rejected: Counter = Counter()
    selected_docs: set[tuple[str, str]] = set()
    control_docs: set[str] = set()
    rows, controls = [], []
    discovery_seen = 0
    fresh_seen = 0
    prefix_seen = 0
    atlas_ids = {int(c) for p in pairs if p["kind"] == "atlas" for side in ("a", "b") for c in p[f"{side}_ids"]}
    native_pairs = [p for p in pairs if p["kind"] == "native"]

    def all_caps_filled() -> bool:
        return all(counts[(p["pair_id"], label, split)] >= caps[split]
                   for p in pairs for label in (0, 1) for split in SPLITS) and all(
                       counts[("_control", -1, s)] >= control_cap for s in SPLITS)

    def record(row: dict, index: int, pair: dict, label: int, text: str, split: str, source_id: str) -> dict:
        ids = sorted({int(c) for key in ("content_ids", "entity_ids", "tone_ids", "document_ids")
                      for c in row.get(key, []) or []})
        side = "a" if label == 0 else "b"
        return {"pair_id": pair["pair_id"], "label": label,
                "label_name": pair[f"{side}_name"] if label >= 0 else "Random control",
                "split": split, "source_doc_id": source_id, "doc_int_id": int(row["doc_int_id"]),
                "chunk_id": int(row["chunk_id"]), "stream_index": index, "text": text,
                "text_sha256": sha256_text(text), "normalized_text_sha256": normalized_text_hash(text),
                "concept_ids": ids, "kind": pair["kind"] if label >= 0 else "atlas",
                "anchor_char_end": None}

    for index, raw in enumerate(source):
        if index >= start_index + max_source_chunks:
            break
        if index and index % 100000 == 0:
            progress = {p["pair_id"]: {
                str(label): {s: counts[(p["pair_id"], label, s)] for s in SPLITS}
                for label in (0, 1)} for p in pairs}
            print(json.dumps({"retained_chunks_scanned": index, "counts": progress}), flush=True)
        doc_id, chunk_id = int(raw["doc_int_id"]), int(raw["chunk_id"])
        text = str(raw["chunk_text"])
        text_hash = normalized_text_hash(text)
        if index < discovery.n_chunks:
            if doc_id != int(discovery.doc[index]) or chunk_id != int(discovery.chunk[index]):
                raise ValueError(f"Atlas stream differs from discovery at retained chunk {index}: "
                                 f"source {(doc_id, chunk_id)} versus cache "
                                 f"{(int(discovery.doc[index]), int(discovery.chunk[index]))}")
            discovery_seen += 1
        if index < start_index:
            prefix_seen += 1
            excluded_docs.add(doc_id)
            excluded_hashes.add(text_hash)
            for p in native_pairs:
                hit = native_prefix(text, p, min_text_chars)
                if hit is not None:
                    excluded_hashes.add(normalized_text_hash(hit[1]))
            continue
        fresh_seen += 1
        if doc_id in excluded_docs:
            rejected["prefix_document"] += 1
            continue
        if text_hash in excluded_hashes:
            rejected["discovery_or_skipped_prefix_duplicate_text"] += 1
            continue
        if text_hash in seen_hashes:
            rejected["fresh_duplicate_text"] += 1
            continue
        seen_hashes.add(text_hash)
        if len(text) < min_text_chars:
            rejected["short_text"] += 1
            continue
        source_id = f"guidelabs/fineweb-atlas:doc_int_id:{doc_id}"
        split = document_split(source_id, seed)
        labels = {int(c) for key in ("content_ids", "entity_ids", "tone_ids", "document_ids")
                  for c in raw.get(key, []) or []}
        contains_native_header = False
        for p in pairs:
            if p["kind"] == "atlas":
                a, b = bool(labels & set(p["a_ids"])), bool(labels & set(p["b_ids"]))
                if a and b:
                    rejected[f"both_labels:{p['pair_id']}"] += 1
                if a == b:
                    continue
                label, selected_text = (0 if a else 1), text
            else:
                contains_native_header |= any(re.search(rf"(?mi)^[ \t]*{re.escape(p[f'{s}_header'])}[ \t]*\r?\n", text)
                                              for s in ("a", "b"))
                hit = native_prefix(text, p, min_text_chars)
                if hit is None:
                    continue
                label, prefix_text = hit
                if len(prefix_text) >= len(text):
                    rejected["native_anchor_without_true_next_text"] += 1
                    continue
                selected_text = text
                prefix_hash = normalized_text_hash(prefix_text)
                if prefix_hash in excluded_hashes or prefix_hash in selected_prefix_hashes:
                    rejected["native_duplicate_prefix"] += 1
                    continue
            key = (p["pair_id"], label, split)
            doc_key = (p["pair_id"], source_id)
            if counts[key] >= caps[split] or doc_key in selected_docs:
                continue
            selected_record = record(raw, index, p, label, selected_text, split, source_id)
            if p["kind"] == "native":
                selected_record["anchor_char_end"] = len(prefix_text)
                selected_record["causal_prefix_text"] = prefix_text
                selected_record["causal_prefix_sha256"] = sha256_text(prefix_text)
            rows.append(selected_record)
            counts[key] += 1
            selected_docs.add(doc_key)
            if p["kind"] == "native":
                selected_prefix_hashes.add(prefix_hash)
        if not labels & atlas_ids and not contains_native_header and source_id not in control_docs:
            if counts[("_control", -1, split)] < control_cap:
                controls.append(record(raw, index, pairs[0], -1, text, split, source_id))
                control_docs.add(source_id)
                counts[("_control", -1, split)] += 1
        if all_caps_filled():
            break
    if discovery_seen != discovery.n_chunks or prefix_seen != start_index:
        raise ValueError(f"Source ended before freshness boundary: discovery {discovery_seen}/{discovery.n_chunks}, "
                         f"skipped prefix {prefix_seen}/{start_index}; no fresh corpus can be claimed")
    for p in pairs:
        rows.extend(dict(r, pair_id=p["pair_id"]) for r in controls)
    return rows, {"discovery_n_chunks": discovery.n_chunks, "discovery_fingerprint": discovery.fingerprint,
                  "validated_discovery_n_chunks": discovery_seen, "start_index": start_index,
                  "fresh_source_chunks_scanned": fresh_seen, "max_source_chunks": max_source_chunks,
                  "excluded_prefix_documents": len(excluded_docs), "excluded_prefix_texts": len(excluded_hashes),
                  "counts": {p["pair_id"]: {str(label): {s: counts[(p["pair_id"], label, s)] for s in SPLITS}
                                            for label in (0, 1)} for p in pairs},
                  "random_control_unique_documents": {s: counts[("_control", -1, s)] for s in SPLITS},
                  "rejections": dict(rejected), "caps_per_label": caps, "control_cap_per_split": control_cap,
                  "document_split": {"method": "SHA256(seed, immutable Atlas source_doc_id), modulo 4",
                                     "seed": seed, "fractions": {"fit": 0.5, "calibration": 0.25, "test": 0.25}},
                  "source_doc_id_definition": "guidelabs/fineweb-atlas:doc_int_id:<original Atlas integer>",
                  "freshness": "Discovery stream verified; all skipped-prefix documents and normalized texts excluded"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--discovery-cache", required=True)
    parser.add_argument("--atlas-dir", default="/scratch/dogar/geoae/atlas")
    parser.add_argument("--source-jsonl", help="Original ordered ok-chunk source JSONL; useful for portable mirrors")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-index", type=int, default=300_000)
    parser.add_argument("--max-source-chunks", type=int, default=2_000_000)
    parser.add_argument("--fit-cap", type=int, default=50)
    parser.add_argument("--calibration-cap", type=int, default=30)
    parser.add_argument("--test-cap", type=int, default=50)
    parser.add_argument("--control-cap", type=int, default=32)
    parser.add_argument("--min-per-label-split", type=int, default=24)
    parser.add_argument("--min-control-split", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.jsonl", "prep.json", "examples.json"):
        if (out / name).exists():
            raise FileExistsError(f"Frozen preparation artifact already exists: {out / name}; choose a fresh output directory")
    t0 = time.monotonic()
    report = {"args": vars(args), "status": "failed", "dataset_kind": "fresh_confirmation"}
    try:
        manifest = json.loads(resolve_input_path(Path(args.manifest)).read_text())
        pairs = normalize_pairs(manifest, args.include_native)
        discovery = Discovery.load(args.discovery_cache)
        start = max(args.start_index, discovery.n_chunks)
        source = source_records(args.atlas_dir, start + args.max_source_chunks,
                                args.download_missing, args.source_jsonl)
        rows, stats = collect_records(source, discovery, pairs, start_index=start,
                                     max_source_chunks=args.max_source_chunks,
                                     caps=dict(zip(SPLITS, (args.fit_cap, args.calibration_cap, args.test_cap))),
                                     control_cap=args.control_cap, seed=args.seed)
        report.update(stats)
        report["pairs"] = pairs
        report["manifest_sha256"] = AS.file_sha256(args.manifest)
        insufficient = [f"{p}/{label}/{split}={n}" for p, groups in stats["counts"].items()
                        for label, group in groups.items() for split, n in group.items()
                        if n < args.min_per_label_split]
        if args.control_cap:
            insufficient.extend(f"random/{split}={n}" for split, n in stats["random_control_unique_documents"].items()
                                if n < args.min_control_split)
        report["insufficient_groups"] = insufficient
        report["status"] = "complete" if not insufficient else "insufficient_fresh_documents"
        with (out / "dataset.jsonl").open("x") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report["dataset_sha256"] = AS.file_sha256(out / "dataset.jsonl")
        report["n_records"] = len(rows)
        report["n_unique_documents"] = len({r["source_doc_id"] for r in rows})
        examples = defaultdict(list)
        for row in rows:
            key = f"{row['pair_id']}/{row['label']}/{row['split']}"
            if len(examples[key]) < 2:
                examples[key].append(row)
        (out / "examples.json").write_text(json.dumps(dict(examples), ensure_ascii=False, indent=2) + "\n")
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    report["seconds"] = time.monotonic() - t0
    (out / "prep.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(out / "prep.json"),
                      "n_records": report.get("n_records"), "counts": report.get("counts"),
                      "error": report.get("error"), "insufficient_groups": report.get("insufficient_groups")}))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
