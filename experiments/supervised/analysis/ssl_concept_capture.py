"""Capture fresh passages for the focused last-layer control experiment."""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with resolve_input_path(Path(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_positions(record, ids, offsets, *, newline_id, tail_tokens=8, max_length=256):
    """Keep true next tokens and enforce the native contrast's token identity."""
    if len(ids) > max_length:
        return [], "too_long_without_truncation"
    if len(ids) < 6:
        return [], "too_short"
    if record.get("kind") == "native" and int(record["label"]) >= 0:
        end = int(record["anchor_char_end"])
        positions = [i for i, (start, stop) in enumerate(offsets)
                     if stop == end and stop > start and ids[i] == newline_id and i + 1 < len(ids)]
        return (positions[-1:], None) if positions else ([], "native_anchor_not_exact_single_newline")
    start = max(4, int(np.ceil(.75 * len(ids))))
    positions = list(range(start, len(ids) - 1))[-tail_tokens:]
    return (positions, None) if positions else ([], "no_true_next_token_in_tail")


def validate_records(records):
    """Reject producer mistakes before model loading."""
    document_parts, memberships, hashes = {}, {}, {}
    for record in records:
        text = record["text"]
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != record["text_sha256"]:
            raise ValueError("source text hash differs from frozen record")
        doc, part = str(record["source_doc_id"]), record["split"]
        if part not in ("fit", "calibration", "test"):
            raise ValueError(f"unknown split {part}")
        previous = document_parts.setdefault(doc, part)
        if previous != part:
            raise ValueError(f"source document appears in multiple splits: {doc}")
        old = hashes.setdefault(actual, part)
        if old != part:
            raise ValueError("duplicate source text appears in different splits")
        if int(record["label"]) >= 0:
            key = record["pair_id"], doc
            old = memberships.setdefault(key, int(record["label"]))
            if old != int(record["label"]):
                raise ValueError(f"ambiguous class membership: {key}")
        if record.get("anchor_char_end") is not None:
            end = int(record["anchor_char_end"])
            if not 0 < end < len(text) or text[end - 1] != "\n":
                raise ValueError("native anchor must be an observed newline with real following text")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prep", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tail-tokens", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "activations.npz").exists() or (out / "capture.json").exists():
        raise FileExistsError("Capture artifacts are frozen; use a new output directory")
    manifest = json.loads(resolve_input_path(Path(args.manifest)).read_text())
    prep = json.loads(resolve_input_path(Path(args.prep)).read_text())
    if prep.get("dataset_sha256") != file_hash(args.dataset):
        raise ValueError("dataset hash differs from preparation report")
    if prep.get("manifest_sha256") != file_hash(args.manifest):
        raise ValueError("manifest changed after data preparation")
    if prep.get("validated_discovery_n_chunks") != manifest["protocol"]["discovery_ok_chunks"]:
        raise ValueError("discovery exclusion has not been validated")
    records = [json.loads(line) for line in resolve_input_path(Path(args.dataset)).read_text().splitlines() if line.strip()]
    validate_records(records)
    import torch
    from geoae.checkpoint import load_tokenizer
    from geoae.checkpoint import load_lm
    from geoae.extraction.sampled import forward_batch
    from geoae.extraction.sampled import register_capture
    from geoae.evaluation.atlas_geometry import lm_head_fp32
    from geoae.lm_arch import decoder_layers

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = load_tokenizer(args.model)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required to verify native anchor offsets")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise ValueError("single newline is not one token in this tokenizer")
    newline_id = newline_ids[0]
    groups, record_groups, rejected = [], [], Counter()
    group_lookup = {}
    for index, record in enumerate(records):
        key = (record["text_sha256"], record.get("anchor_char_end"), record.get("kind"))
        if key in group_lookup:
            record_groups.append((index, group_lookup[key]))
            continue
        encoded = tokenizer(record["text"], return_offsets_mapping=True, truncation=False)
        positions, reason = choose_positions(record, encoded["input_ids"], encoded["offset_mapping"],
                                             newline_id=newline_id, tail_tokens=args.tail_tokens,
                                             max_length=args.max_length)
        if reason:
            rejected[f"{record['pair_id']}/{record['label']}/{record['split']}/{reason}"] += 1
            continue
        group_lookup[key] = len(groups)
        record_groups.append((index, len(groups)))
        groups.append({"ids": encoded["input_ids"], "positions": positions, "source_record": index})
    if not groups:
        raise ValueError("no eligible source passages remain")
    start = time.monotonic()
    lm = load_lm(args.model, device=args.device)
    if args.layer != len(decoder_layers(lm)) - 1:
        raise ValueError("head-only control scoring requires the final decoder layer")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Verify a full forward and the exact cached-layer readout before extraction.
    norm, head = lm_head_fp32(lm)
    samples = groups[:2]
    length = max(len(g["ids"]) for g in samples)
    ids = torch.full((len(samples), length), pad, dtype=torch.long, device=args.device)
    mask = torch.zeros_like(ids)
    for i, sample in enumerate(samples):
        ids[i, :len(sample["ids"])] = torch.tensor(sample["ids"], device=args.device)
        mask[i, :len(sample["ids"])] = 1
    held = {}
    handle = decoder_layers(lm)[args.layer].register_forward_hook(
        lambda _module, _inp, value, _held=held: _held.update(
            h=(value[0] if isinstance(value, tuple) else value).detach()))
    try:
        output = lm(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()
    differences = []
    for i, sample in enumerate(samples):
        for position in sample["positions"]:
            nxt = sample["ids"][position + 1]
            actual = output.logits[i, position].float().log_softmax(-1)[nxt]
            offline = head(norm(held["h"][i, position].float().unsqueeze(0))).log_softmax(-1)[0, nxt]
            differences.append(abs(float(actual - offline)))
    sanity = {"n": len(differences), "max_abs_logprob_error": max(differences),
              "mean_abs_logprob_error": float(np.mean(differences)), "same_forward": True}
    if sanity["max_abs_logprob_error"] > .15 or sanity["mean_abs_logprob_error"] > .04:
        raise RuntimeError(f"cached-layer scoring does not reproduce model logits: {sanity}")
    del output, held, norm, head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    features = [None] * len(groups)
    captured, handles = register_capture(lm, [args.layer])
    try:
        order = sorted(range(len(groups)), key=lambda i: len(groups[i]["ids"]))
        for offset in range(0, len(order), args.batch_size):
            sub = order[offset:offset + args.batch_size]
            forward_batch(lm, [groups[i]["ids"] for i in sub], pad, args.device)
            values = captured[args.layer]
            for local, index in enumerate(sub):
                features[index] = values[local, groups[index]["positions"]].float().cpu().numpy()
                if not np.isfinite(features[index]).all():
                    raise ValueError("nonfinite captured activation")
            captured.clear()
            if offset % (args.batch_size * 10) == 0:
                print(f"Captured {min(offset + len(sub), len(groups))}/{len(groups)} unique passages", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    chunks = {k: [] for k in ("h", "tok", "nxt", "doc", "pair", "label", "split", "record", "position")}
    count_report = Counter()
    for record_index, group_index in record_groups:
        record, group = records[record_index], groups[group_index]
        positions = group["positions"]
        n = len(positions)
        chunks["h"].append(features[group_index])
        chunks["tok"].extend(group["ids"][p] for p in positions)
        chunks["nxt"].extend(group["ids"][p + 1] for p in positions)
        for key, value in (("doc", record["source_doc_id"]), ("pair", record["pair_id"]),
                           ("label", record["label"]), ("split", record["split"]), ("record", record_index)):
            chunks[key].extend([value] * n)
        chunks["position"].extend(positions)
        count_report[f"{record['pair_id']}/{record['label']}/{record['split']}"] += 1
    arrays = {k: (np.concatenate(v) if k == "h" else np.asarray(v)) for k, v in chunks.items()}
    with (out / "activations.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    metadata = {"model": args.model, "layer": args.layer, "args": vars(args), "manifest": manifest,
                "manifest_sha256": file_hash(args.manifest), "dataset_sha256": file_hash(args.dataset),
                "input_prep": prep, "counts_documents": dict(count_report), "rejected": dict(rejected),
                "native_current_token_id": newline_id, "same_native_token_enforced": True,
                "n_rows": len(arrays["h"]), "n_unique_passages": len(groups), "sanity": sanity,
                "npz_sha256": file_hash(out / "activations.npz"),
                "tokenizer_name": tokenizer.name_or_path, "seconds": time.monotonic() - start}
    (out / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"n_rows": metadata["n_rows"], "counts": dict(count_report),
                      "rejected": dict(rejected), "sanity": sanity}), flush=True)


if __name__ == "__main__":
    main()
