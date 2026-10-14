"""SugarCrepe++: unit-normalized Euclidean geometry of base and bypass AE.

The primary rule is d(caption, caption2) <= d(caption, negative_caption).
Last real token, post-decoder-block/pre-final-norm; no prompt or truncation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CATEGORIES = ("replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj")
RULES = ("pass", "reverse_pass", "both_pass", "strict_pass", "strict_both_pass", "ties")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "provenance/manifest.json").read_text())
    rows = []
    for category in CATEGORIES:
        source = directory / f"{category}.json"
        info = manifest["files"][category]
        if sha256(source) != info["sha256"]:
            raise ValueError(f"dataset checksum mismatch: {source}")
        data = json.loads(source.read_text())
        if len(data) != info["records"]:
            raise ValueError(f"dataset row-count mismatch: {source}")
        rows.extend(dict(row, category=category) for row in data)
    if len(rows) != manifest["total_records"]:
        raise ValueError("dataset total mismatch")
    return rows, manifest


def build_text_index(rows):
    texts, lookup, indices = [], {}, []
    for row in rows:
        triplet = []
        for key in ("caption", "caption2", "negative_caption"):
            value = row[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid blank/non-string text: {key}")
            if value not in lookup:
                lookup[value] = len(texts)
                texts.append(value)
            triplet.append(lookup[value])
        indices.append(triplet)
    return texts, np.asarray(indices, dtype=np.int64).reshape(-1, 3)


def normalized_euclidean_pairs(embeddings, triplet_indices):
    return euclidean_pairs(embeddings, triplet_indices, normalize=True)


def raw_euclidean_pairs(embeddings, triplet_indices):
    return euclidean_pairs(embeddings, triplet_indices, normalize=False)


def euclidean_pairs(embeddings, triplet_indices, *, normalize):
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite matrix")
    if normalize:
        norms = np.linalg.norm(values, axis=1)
        if not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("zero or nonfinite embedding norm")
        values = values / norms[:, None]
    indices = np.asarray(triplet_indices)
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu":
        raise ValueError("triplet indices must be an integer N x 3 matrix")
    if indices.size and (indices.min() < 0 or indices.max() >= len(values)):
        raise ValueError("triplet index out of range")
    distances = {}
    for key, left, right in (("d_paraphrase", 0, 1), ("d_distractor", 0, 2),
                             ("d_paraphrase_distractor", 1, 2)):
        chunks = []
        for start in range(0, len(indices), 256):
            sub = indices[start:start + 256]
            chunks.append(np.linalg.norm(values[sub[:, left]] - values[sub[:, right]], axis=1))
        distances[key] = np.concatenate(chunks) if chunks else np.empty(0)
    positive, negative, reverse = (distances[key] for key in
        ("d_paraphrase", "d_distractor", "d_paraphrase_distractor"))
    distances.update({"pass": positive <= negative, "reverse_pass": positive <= reverse,
        "both_pass": (positive <= negative) & (positive <= reverse),
        "strict_pass": positive < negative,
        "strict_both_pass": (positive < negative) & (positive < reverse),
        "ties": positive == negative})
    return distances


def summarize(scores, mask):
    n = int(mask.sum())
    result = {"n": n}
    for arm, values in scores.items():
        result[arm] = {rule: {"count": int(values[rule][mask].sum()),
                             "percent": float(100 * values[rule][mask].mean())}
                       for rule in RULES}
        result[arm]["mean_distances"] = {key: float(values[key][mask].mean())
            for key in ("d_paraphrase", "d_distractor", "d_paraphrase_distractor")}
        result[arm]["mean_margin"] = float((values["d_distractor"] - values["d_paraphrase"])[mask].mean())
    base, ae = scores["base"]["pass"][mask], scores["bypass_ae"]["pass"][mask]
    result["paired"] = {"both_pass": int((base & ae).sum()),
        "base_only_pass": int((base & ~ae).sum()), "ae_only_pass": int((~base & ae).sum()),
        "both_fail": int((~base & ~ae).sum()),
        "ae_minus_base_percentage_points": float(100 * (ae.astype(float) - base).mean())}
    return result


def paired_image_bootstrap(scores, rows, repetitions, seed):
    """Resample images together: some images occur in multiple triplets/categories."""
    groups = {}
    group_ids = []
    for row in rows:
        key = row["filename"]
        if key not in groups:
            groups[key] = len(groups)
        group_ids.append(groups[key])
    ids = np.asarray(group_ids)
    sizes = np.bincount(ids)
    wins = {arm: np.bincount(ids, weights=values["pass"].astype(float)) for arm, values in scores.items()}
    rng = np.random.default_rng(seed)
    estimates = {arm: [] for arm in scores}
    differences = []
    for _ in range(repetitions):
        sample = rng.integers(len(groups), size=len(groups))
        denominator = sizes[sample].sum()
        pair = {arm: 100 * totals[sample].sum() / denominator for arm, totals in wins.items()}
        for arm in estimates:
            estimates[arm].append(pair[arm])
        differences.append(pair["bypass_ae"] - pair["base"])
    return {"unit": "image filename (all its triplets resampled together)", "n_images": len(groups),
        "repetitions": repetitions, "seed": seed,
        "percent_ci95": {arm: np.quantile(values, [.025, .975]).tolist() for arm, values in estimates.items()},
        "ae_minus_base_percentage_points_ci95": np.quantile(differences, [.025, .975]).tolist()}


def capture_embeddings(args, texts, out):
    import torch
    from transformers import AutoTokenizer
    from geoae.checkpoint import load_ae_checkpoint, load_lm
    from geoae.extract_sampled import forward_batch, register_capture
    from geoae.lm_arch import decoder_layers

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    sequences = [tokenizer.encode(t, add_special_tokens=args.add_special_tokens) for t in texts]
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    if (lengths == 0).any() or lengths.max() > args.max_length:
        raise ValueError(f"empty or overlength caption; truncation forbidden, max={lengths.max()}")
    ae, mean, std, checkpoint = load_ae_checkpoint(args.checkpoint, args.device, allow_token_bias=True)
    if not ae.has_token_bias or ae.token_bias_mode != "bypass":
        raise ValueError("checkpoint must be a token-bypass AE")
    trained_model = checkpoint["config"]["extraction"]["model_name"]
    if trained_model != args.model:
        raise ValueError(f"checkpoint model {trained_model} differs from {args.model}")
    trained_layer = checkpoint["config"]["data"]["target_layer"]
    if trained_layer != args.layer:
        raise ValueError(f"checkpoint layer {trained_layer} differs from {args.layer}")
    metadata = {"checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_step": checkpoint.get("step"),
        "checkpoint_model_config": checkpoint["config"]["model"],
        "tokenization": {"add_special_tokens": args.add_special_tokens,
            "bos_token_id": tokenizer.bos_token_id,
            "all_sequences_start_with_bos": tokenizer.bos_token_id is not None and
                all(s[0] == tokenizer.bos_token_id for s in sequences)},
        "training_normalization": {"mean_shape": list(mean.shape), "std_shape": list(std.shape)},
        "token_count": {"min": int(lengths.min()), "max": int(lengths.max()), "mean": float(lengths.mean())}}
    del checkpoint
    lm = load_lm(args.model, device=args.device)
    # BF16 forward arithmetic can alter residual directions across batch shapes.
    # Float32 forward arithmetic makes close pairwise comparisons reproducible.
    if args.lm_precision == "float32":
        lm.float()
    metadata.update({"model_commit": getattr(lm.config, "_commit_hash", None),
        "n_layers": len(decoder_layers(lm)), "lm_dtype": str(next(lm.parameters()).dtype),
        "ae_dtype": str(next(ae.parameters()).dtype)})
    captured, handles = register_capture(lm, [args.layer])
    base = np.empty((len(texts), ae.hidden_size), np.float32)
    latent = np.empty((len(texts), ae.latent_dim), np.float32)
    token_ids = np.asarray([s[-1] for s in sequences], dtype=np.int64)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    start = time.monotonic()
    try:
        order = sorted(range(len(texts)), key=lambda i: len(sequences[i]))
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset:offset + args.batch_size]
            forward_batch(lm, [sequences[i] for i in indices], pad_id, args.device)
            h = captured[args.layer]
            positions = torch.as_tensor(lengths[indices] - 1, device=args.device)
            last = h[torch.arange(len(indices), device=args.device), positions].float()
            ids = torch.as_tensor(token_ids[indices], device=args.device)
            z = ae.encode((last - mean) / std, ids)
            base[indices] = last.cpu().numpy()
            latent[indices] = z.cpu().numpy()
            captured.clear()
            if offset % (args.batch_size * 20) == 0:
                print(f"Captured {offset + len(indices)}/{len(texts)} unique captions ({time.monotonic()-start:.1f}s)", flush=True)
        # Verify sorted, padded batches against unpadded singleton captures.
        checks = []
        for index in (order[0], order[len(order) // 2], order[-1]):
            forward_batch(lm, [sequences[index]], pad_id, args.device)
            single = captured[args.layer][0, -1].float().cpu().numpy()
            error = float(np.linalg.norm(single - base[index]) / np.linalg.norm(single))
            checks.append({"text_index": index, "relative_l2_error": error})
            captured.clear()
        metadata["singleton_batch_checks"] = checks
        tolerance = 2e-4 if args.lm_precision == "float32" else .03
        metadata["singleton_batch_tolerance"] = tolerance
        if max(c["relative_l2_error"] for c in checks) > tolerance:
            raise RuntimeError(f"batched capture differs from singleton: {checks}")
    finally:
        for handle in handles:
            handle.remove()
    if not np.isfinite(base).all() or not np.isfinite(latent).all():
        raise ValueError("nonfinite model output")
    np.save(out / "base_last.npy", base)
    np.save(out / "bypass_ae_last.npy", latent)
    np.save(out / "last_token_ids.npy", token_ids)
    np.save(out / "token_lengths.npy", lengths)
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False) + "\n")
    metadata["capture_seconds"] = time.monotonic() - start
    (out / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"base": base, "bypass_ae": latent}, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/sugarcrepe_plus_plus")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--checkpoint", default="checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lm-precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True,
                        help="Use native special tokens (Gemma adds BOS; Qwen adds none).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows, manifest = load_rows(args.dataset)
    texts, indices = build_text_index(rows)
    print(f"SugarCrepe++: {len(rows)} triplets; {len(texts)} unique exact captions", flush=True)
    (out / "triplets.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    np.save(out / "triplet_indices.npy", indices)
    source_dir = out / "source"
    source_dir.mkdir()
    for source in (Path(__file__), ROOT / "geoae/checkpoint.py", ROOT / "geoae/model.py",
                   ROOT / "geoae/extract_sampled.py", ROOT / "geoae/lm_arch.py", ROOT / "geoae/hooks.py"):
        shutil.copy2(source, source_dir / source.name)
    source_hashes = {p.name: sha256(p) for p in source_dir.iterdir()}
    checkpoint_hash = sha256(args.checkpoint)
    embeddings, capture = capture_embeddings(args, texts, out)
    scores = {arm: normalized_euclidean_pairs(value, indices) for arm, value in embeddings.items()}
    totals = summarize(scores, np.ones(len(rows), dtype=bool))
    categories = {category: summarize(scores, np.asarray([r["category"] == category for r in rows]))
                  for category in CATEGORIES}
    bootstrap = paired_image_bootstrap(scores, rows, args.bootstrap, args.seed)
    with (out / "per_triplet.jsonl").open("w") as stream:
        for i, row in enumerate(rows):
            record = dict(row, row_index=i)
            record["arms"] = {arm: {key: value[i].item() for key, value in arm_scores.items()}
                              for arm, arm_scores in scores.items()}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.savez(out / "distances.npz", **{f"{arm}_{key}": value for arm, vals in scores.items() for key, value in vals.items()})
    result = {"dataset": "SugarCrepe++", "model": args.model, "layer": args.layer,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args),
        "protocol": {"primary": "d(caption, caption2) <= d(caption, negative_caption)",
            "distance": "Euclidean after independent per-vector unit L2 normalization; float64 scoring",
            "base": "native post-block residual h; no training centering before unit normalization",
            "bypass_ae": "z = ae.encode((h - checkpoint_mean) / checkpoint_std, current_token_id)",
            "pooling": "last real token", "layer_point": "post-block, before final RMSNorm",
            "tokenization": f"add_special_tokens={args.add_special_tokens}; original exact text; "
                "native special tokens; no prompt, manually appended EOS, or truncation",
            "ties": "exact distance equality counts as pass; no epsilon tolerance",
            "secondary": "reverse anchor; both anchors inclusive; official strict both-anchor TOT"},
        "dataset_manifest": manifest, "n_unique_texts": len(texts), "capture": capture,
        "hashes": {"checkpoint_sha256": checkpoint_hash, "sources": source_hashes,
                   "base_last_sha256": sha256(out / "base_last.npy"),
                   "bypass_ae_last_sha256": sha256(out / "bypass_ae_last.npy")},
        "overall": totals, "categories": categories, "paired_image_bootstrap": bootstrap}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# SugarCrepe++ — {args.model} base versus bypass AE", "",
        f"All {len(rows):,} official triplets; layer {args.layer}; last real token.", "",
        "Each vector is normalized to unit L2 length. Pass: caption–paraphrase distance ≤ caption–distractor distance. Exact ties count as passes.", "",
        "| Category | Triplets | Base passes | Bypass AE passes | AE − base (points) |",
        "|---|---:|---:|---:|---:|"]
    for label, group in [("Overall", totals), *categories.items()]:
        cells = [f"{group[a]['pass']['count']}/{group['n']} ({group[a]['pass']['percent']:.2f}%)" for a in scores]
        lines.append(f"| {label} | {group['n']} | {cells[0]} | {cells[1]} | {group['paired']['ae_minus_base_percentage_points']:+.2f} |")
    ci = bootstrap["ae_minus_base_percentage_points_ci95"]
    lines += ["", f"Paired image bootstrap 95% interval for AE − base: [{ci[0]:+.2f}, {ci[1]:+.2f}] percentage points ({args.bootstrap:,} resamples; {bootstrap['n_images']:,} images).", "",
        f"Exact primary ties: base {totals['base']['ties']['count']}, bypass AE {totals['bypass_ae']['ties']['count']}.", "",
        "Secondary checks (different rules from the requested primary test):", "",
        "| Rule | Base | Bypass AE |", "|---|---:|---:|"]
    for rule, label in (("reverse_pass", "Paraphrase used as anchor, inclusive"),
                        ("both_pass", "Both anchors must pass, inclusive"),
                        ("strict_both_pass", "Official TOT, both anchors strict")):
        lines.append(f"| {label} | {totals['base'][rule]['percent']:.2f}% | {totals['bypass_ae'][rule]['percent']:.2f}% |")
    lines += ["", f"Model forward arithmetic: {capture['lm_dtype']}; maximum singleton/batch relative L2 error: {max(c['relative_l2_error'] for c in capture['singleton_batch_checks']):.3g}.", "",
        "Base uses the native residual before final RMSNorm. The bypass uses the saved encoder with its token-specific bias subtraction and training normalization. No models were trained or fitted on this dataset.", "",
        "Official dataset: https://github.com/Sri-Harsha/scpp", "",
        "`per_triplet.jsonl` contains original texts, all distances, and pass/fail flags. Saved `.npy` vectors and `triplet_indices.npy` allow scoring again without inference. `summary.json` includes model/dataset/source checksums and protocol details.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(json.dumps({"overall": totals, "paired_image_bootstrap": bootstrap}, indent=2), flush=True)


if __name__ == "__main__":
    main()
