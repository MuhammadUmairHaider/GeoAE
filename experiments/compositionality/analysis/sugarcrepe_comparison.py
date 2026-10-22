"""Compare saved Qwen and Gemma SugarCrepe++ runs with paired image intervals."""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("base", "bypass_ae")
COMMON_PROTOCOL = {
    "primary": "d(caption, caption2) <= d(caption, negative_caption)",
    "distance": "Euclidean after independent per-vector unit L2 normalization; float64 scoring",
    "pooling": "last real token",
    "layer_point": "post-block, before final RMSNorm",
    "ties": "exact distance equality counts as pass; no epsilon tolerance",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(resolve_input_path(Path(path)).read_bytes()).hexdigest()


def load_run(directory):
    directory = resolve_input_path(directory)
    summary = json.loads((resolve_input_path(directory / "summary.json")).read_text())
    rows = json.loads((resolve_input_path(directory / "triplets.json")).read_text())
    for key, expected in COMMON_PROTOCOL.items():
        require(summary["protocol"][key] == expected, f"{directory}: incompatible {key}")
    require(summary["args"]["lm_precision"] == "float32", f"{directory}: LM precision")
    for key in ("lm_dtype", "ae_dtype"):
        require(summary["capture"][key] == "torch.float32", f"{directory}: {key}")
    require(summary["capture"]["checkpoint_model_config"]["token_bias_mode"] == "bypass",
            f"{directory}: checkpoint is not bypass")
    require(len(rows) == summary["overall"]["n"] == summary["dataset_manifest"]["total_records"],
            f"{directory}: inconsistent row count")
    records = [json.loads(line) for line in (resolve_input_path(directory / "per_triplet.jsonl")).read_text().splitlines()]
    require(len(records) == len(rows), f"{directory}: incomplete per-triplet records")
    passes = {arm: [] for arm in ARMS}
    for index, (row, record) in enumerate(zip(rows, records)):
        require(record["row_index"] == index and
                {key: value for key, value in record.items() if key not in ("row_index", "arms")} == row,
                f"{directory}: triplet identity/order at row {index}")
        for arm in ARMS:
            score = record["arms"][arm]
            distances = (score["d_paraphrase"], score["d_distractor"])
            require(np.isfinite(distances).all(), f"{directory}: nonfinite distance")
            require(type(score["pass"]) is bool and score["pass"] == (distances[0] <= distances[1]),
                    f"{directory}: inclusive pass mismatch at row {index}, {arm}")
            passes[arm].append(score["pass"])
    passes = {arm: np.asarray(values, dtype=bool) for arm, values in passes.items()}
    for arm, flags in passes.items():
        total = summary["overall"][arm]["pass"]
        require(int(flags.sum()) == total["count"] and
                np.isclose(100 * flags.mean(), total["percent"], rtol=0, atol=1e-10),
                f"{directory}: summary disagrees with per-triplet scores")
    return summary, rows, passes


def paired_bootstrap(rows, qwen, gemma, repetitions=5000, seed=0):
    groups, ids = {}, []
    for row in rows:
        ids.append(groups.setdefault(row["filename"], len(groups)))
    ids = np.asarray(ids)
    sizes = np.bincount(ids)
    # All four spaces use the same resampled images and triplet denominator.
    flags = np.column_stack((qwen["base"], qwen["bypass_ae"], gemma["base"], gemma["bypass_ae"]))
    totals = np.column_stack([np.bincount(ids, weights=flags[:, column]) for column in range(4)])
    draws = np.empty((repetitions, 4))
    rng = np.random.default_rng(seed)
    for index in range(repetitions):
        sample = rng.integers(len(groups), size=len(groups))
        draws[index] = 100 * totals[sample].sum(axis=0) / sizes[sample].sum()
    qwen_gain, gemma_gain = draws[:, 1] - draws[:, 0], draws[:, 3] - draws[:, 2]
    return {
        "unit": "image filename (all its triplets resampled together)",
        "n_images": len(groups), "repetitions": repetitions, "seed": seed,
        "qwen_ae_minus_base_ci95": np.quantile(qwen_gain, [.025, .975]).tolist(),
        "gemma_ae_minus_base_ci95": np.quantile(gemma_gain, [.025, .975]).tolist(),
        "gemma_bypass_minus_qwen_bypass_ci95": np.quantile(draws[:, 3] - draws[:, 1], [.025, .975]).tolist(),
        "gemma_gain_minus_qwen_gain_ci95": np.quantile(gemma_gain - qwen_gain, [.025, .975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", default="results/sugarcrepe_qwen25_3b_l35_20261010_fp32")
    parser.add_argument("--gemma", default="results/sugarcrepe_gemma3_4b_l33_k8000_20261010_fp32")
    parser.add_argument("--out-dir", default="results/sugarcrepe_comparison_20261010")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    require(args.bootstrap > 0, "bootstrap repetitions must be positive")
    qs, rows, qwen = load_run(args.qwen)
    gs, gemma_rows, gemma = load_run(args.gemma)
    require(rows == gemma_rows, "runs have different exact triplets or ordering")
    require(qs["dataset_manifest"] == gs["dataset_manifest"], "runs have different dataset source manifests")
    require(qs["model"] == "Qwen/Qwen2.5-3B" and gs["model"] == "google/gemma-3-4b-pt",
            "expected Qwen2.5-3B and Gemma 3 4B")
    # Qwen's native tokenizer adds no special tokens; its earlier run disabled
    # them explicitly. Gemma must retain its native BOS for training parity.
    require("add_special_tokens=False" in qs["protocol"]["tokenization"] or
            qs["capture"].get("tokenization", {}).get("add_special_tokens") is True,
            "unexpected Qwen tokenization protocol")
    gt = gs["capture"]["tokenization"]
    require(gt["add_special_tokens"] is True and gt["all_sequences_start_with_bos"] is True,
            "Gemma must use native BOS")
    bootstrap = paired_bootstrap(rows, qwen, gemma, args.bootstrap, args.seed)
    models = {}
    for name, summary, passes in (("qwen", qs, qwen), ("gemma", gs, gemma)):
        gain = 100 * (passes["bypass_ae"].astype(float) - passes["base"]).mean()
        models[name] = {
            "model": summary["model"], "layer": summary["layer"],
            "k": summary["capture"]["checkpoint_model_config"]["n_clusters"],
            "base": summary["overall"]["base"]["pass"],
            "bypass_ae": summary["overall"]["bypass_ae"]["pass"],
            "ae_minus_base_percentage_points": float(gain),
            "ae_minus_base_ci95": bootstrap[f"{name}_ae_minus_base_ci95"],
        }
        original = summary["paired_image_bootstrap"]
        if original["repetitions"] == args.bootstrap and original["seed"] == args.seed:
            require(np.allclose(models[name]["ae_minus_base_ci95"],
                                original["ae_minus_base_percentage_points_ci95"], rtol=0, atol=1e-10),
                    f"{name}: bootstrap disagrees with original run")
    cross = {
        "gemma_bypass_minus_qwen_bypass_percentage_points":
            float(100 * (gemma["bypass_ae"].astype(float) - qwen["bypass_ae"]).mean()),
        "gemma_bypass_minus_qwen_bypass_ci95": bootstrap["gemma_bypass_minus_qwen_bypass_ci95"],
        "gemma_gain_minus_qwen_gain_percentage_points":
            models["gemma"]["ae_minus_base_percentage_points"] - models["qwen"]["ae_minus_base_percentage_points"],
        "gemma_gain_minus_qwen_gain_ci95": bootstrap["gemma_gain_minus_qwen_gain_ci95"],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "n_triplets": len(rows),
        "protocol": COMMON_PROTOCOL, "dataset_manifest": qs["dataset_manifest"],
        "tokenization_difference": "Qwen adds no native special tokens; Gemma uses its standard BOS.",
        "models": models, "cross_model": cross, "paired_image_bootstrap": bootstrap,
        "inputs": {name: {"directory": str(resolve_input_path(directory).resolve()),
                    "summary_sha256": sha256(Path(directory) / "summary.json"),
                    "triplets_sha256": sha256(Path(directory) / "triplets.json"),
                    "checkpoint_sha256": summary["hashes"]["checkpoint_sha256"]}
                   for name, directory, summary in (("qwen", args.qwen, qs), ("gemma", args.gemma, gs))},
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# SugarCrepe++ — Qwen versus Gemma K={models['gemma']['k']:,}", "",
             f"Same {len(rows):,} exact triplets and dataset source manifest; last real token, float32 inference.", "",
             "Pass: distance(original, paraphrase) ≤ distance(original, distractor), after independently normalizing each vector to unit L2 length. Exact ties pass.", "",
             "| Model | Layer / K | Base passes | Bypass AE passes | AE − base (points; 95% CI) |",
             "|---|---:|---:|---:|---:|"]
    for name, label in (("qwen", "Qwen2.5-3B"), ("gemma", "Gemma 3 4B")):
        model = models[name]
        ci = model["ae_minus_base_ci95"]
        cells = [f"{model[arm]['count']:,}/{len(rows):,} ({model[arm]['percent']:.2f}%)" for arm in ARMS]
        lines.append(f"| {label} | {model['layer']} / {model['k']:,} | {cells[0]} | {cells[1]} | "
                     f"{model['ae_minus_base_percentage_points']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines += ["", "| Paired comparison | Difference (points; 95% CI) |", "|---|---:|"]
    for key, label in (("gemma_bypass_minus_qwen_bypass", "Gemma bypass − Qwen bypass"),
                       ("gemma_gain_minus_qwen_gain", "Gemma AE gain − Qwen AE gain")):
        ci = cross[f"{key}_ci95"]
        lines.append(f"| {label} | {cross[f'{key}_percentage_points']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines += ["", f"All intervals use {args.bootstrap:,} paired image bootstrap resamples "
              f"({bootstrap['n_images']:,} images; seed {args.seed}); each image's triplets stay together.", "",
              "Qwen adds no native special tokens; Gemma uses its standard BOS.", "",
              "These are different language models and trained AEs, so this comparison does not isolate the effect of K.", "",
              f"Input runs: `{args.qwen}` and `{args.gemma}`. `summary.json` records input hashes and protocol checks.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(json.dumps({"models": models, "cross_model": cross}, indent=2))


if __name__ == "__main__":
    main()
