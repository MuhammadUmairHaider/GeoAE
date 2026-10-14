"""Rescore frozen SugarCrepe++ vectors using raw or unit-L2 Euclidean distance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_out.sugarcrepe_geometry import (  # noqa: E402
    CATEGORIES, build_text_index, load_rows, normalized_euclidean_pairs,
    paired_image_bootstrap, raw_euclidean_pairs, sha256, summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--distance", choices=("raw", "unit_l2"), default="raw")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    source = Path(args.capture_dir).resolve()
    prior = json.loads((source / "summary.json").read_text())
    rows = json.loads((source / "triplets.json").read_text())
    actual_rows, manifest = load_rows(ROOT / prior["args"]["dataset"])
    if rows != actual_rows or manifest != prior["dataset_manifest"]:
        raise ValueError("saved triplets differ from pinned official dataset")
    texts, expected_indices = build_text_index(rows)
    indices = np.load(source / "triplet_indices.npy", allow_pickle=False)
    if not np.array_equal(indices, expected_indices) or texts != json.loads((source / "texts.json").read_text()):
        raise ValueError("saved text/index mapping differs from original triplets")
    if len(rows) != prior["overall"]["n"]:
        raise ValueError("prior summary row count differs")
    scorer = raw_euclidean_pairs if args.distance == "raw" else normalized_euclidean_pairs
    scores, vector_hashes = {}, {}
    for arm, filename, width in (
        ("base", "base_last.npy", prior["capture"]["checkpoint_model_config"]["hidden_size"]),
        ("bypass_ae", "bypass_ae_last.npy", prior["capture"]["checkpoint_model_config"]["latent_dim"]),
    ):
        digest = sha256(source / filename)
        if digest != prior["hashes"][filename.replace(".npy", "_sha256")]:
            raise ValueError(f"frozen vector checksum differs: {filename}")
        vector_hashes[filename] = digest
        vectors = np.load(source / filename, mmap_mode="r", allow_pickle=False)
        if vectors.shape != (len(texts), width) or vectors.dtype != np.float32:
            raise ValueError(f"frozen vector shape/dtype differs: {filename}")
        scores[arm] = scorer(vectors, indices)
        del vectors
    totals = summarize(scores, np.ones(len(rows), dtype=bool))
    categories = {category: summarize(scores, np.asarray([r["category"] == category for r in rows]))
                  for category in CATEGORIES}
    bootstrap = paired_image_bootstrap(scores, rows, args.bootstrap, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    source_dir = out / "source"
    source_dir.mkdir()
    for path in (Path(__file__), ROOT / "eval_out/sugarcrepe_geometry.py"):
        shutil.copy2(path, source_dir / path.name)
    # Reference the original capture rather than duplicate hundreds of MB.
    for filename in ("base_last.npy", "bypass_ae_last.npy", "triplet_indices.npy", "triplets.json",
                     "texts.json", "last_token_ids.npy", "token_lengths.npy", "capture.json"):
        (out / filename).symlink_to(source / filename)
    np.savez(out / "distances.npz", **{f"{arm}_{key}": value for arm, vals in scores.items()
                                      for key, value in vals.items()})
    with (out / "per_triplet.jsonl").open("w") as stream:
        for i, row in enumerate(rows):
            record = dict(row, row_index=i, arms={
                arm: {key: value[i].item() for key, value in vals.items()} for arm, vals in scores.items()})
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    description = ("Raw Euclidean on saved native base h and AE latent z; "
                   "no per-vector unit normalization, centering, or rescaling at scoring; float64 calculation"
                   if args.distance == "raw" else
                   "Euclidean after independent per-vector unit L2 normalization; float64 scoring")
    protocol = dict(prior["protocol"], distance=description,
                    base="native post-block residual h; no training centering at scoring",
                    secondary="Reverse anchor; both anchors inclusive; both anchors strict, using the same distance")
    result = {"dataset": prior["dataset"], "model": prior["model"], "layer": prior["layer"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args),
        "protocol": protocol, "dataset_manifest": manifest, "n_unique_texts": len(texts),
        "capture": prior["capture"], "source_capture": {"directory": str(source),
            "summary_sha256": sha256(source / "summary.json"), "vector_sha256": vector_hashes,
            "inference_reused": True, "trained_ae_input_preprocessing_preserved": True},
        "hashes": {"checkpoint_sha256": prior["hashes"]["checkpoint_sha256"],
            "sources": {p.name: sha256(p) for p in source_dir.iterdir()}},
        "overall": totals, "categories": categories, "paired_image_bootstrap": bootstrap,
        "prior_score": {"distance": prior["protocol"]["distance"], "overall": prior["overall"]}}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    label = "raw Euclidean" if args.distance == "raw" else "unit-L2 Euclidean"
    lines = [f"# SugarCrepe++ — {prior['model']}, K={prior['capture']['checkpoint_model_config']['n_clusters']:,}, {label}", "",
        f"All {len(rows):,} triplets; layer {prior['layer']}; last real token; saved float32 inference.", "",
        "Pass: distance(original, paraphrase) ≤ distance(original, distractor). Exact ties count as passes.", "",
        description + ".", "",
        "| Category | Triplets | Base passes | Bypass AE passes | AE − base (points) |",
        "|---|---:|---:|---:|---:|"]
    for category, group in [("Overall", totals), *categories.items()]:
        cells = [f"{group[a]['pass']['count']}/{group['n']} ({group[a]['pass']['percent']:.2f}%)" for a in scores]
        lines.append(f"| {category} | {group['n']} | {cells[0]} | {cells[1]} | "
                     f"{group['paired']['ae_minus_base_percentage_points']:+.2f} |")
    ci = bootstrap["ae_minus_base_percentage_points_ci95"]
    lines += ["", f"AE − base 95% paired image bootstrap interval: [{ci[0]:+.2f}, {ci[1]:+.2f}] points "
              f"({args.bootstrap:,} resamples, {bootstrap['n_images']:,} images, seed {args.seed}).", "",
        "The frozen model vectors and the AE's trained input preprocessing are preserved. Only distance scoring was changed; no inference or training was repeated.", "",
        f"Original capture: `{source}`. Vector symlinks reference the original capture, and `summary.json` records source checksums.", "",
        "`per_triplet.jsonl` contains all three distances and the inclusive, reverse, both-anchor, strict, and tie flags. The primary score is the inclusive original-anchor comparison above.", "",
        "Rerun with a new output directory:", "", "```bash",
        f"scripts/delta/py eval_out/sugarcrepe_rescore.py --capture-dir {args.capture_dir} "
        f"--distance {args.distance} --out-dir results/sugarcrepe_rescore_NEW", "```", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(json.dumps({"overall": totals, "paired_image_bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
