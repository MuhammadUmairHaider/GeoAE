"""Aggregate only the isolated matched-data comparison after all evals finish."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from geoae.evaluation.sae_removal import family_bootstrap
from .common import METHODS, ROOT, model_name, write_json


DEFAULT_WIDTHS = (10240, 20480)
DEFAULT_EXTRA_MODELS = ("batch_topk_20480_k100",)
REQUIRED_BASELINES = ("geoae", "gemma_scope_16k")
SANITY_METRICS = ("mean_l0", "raw_fve", "clean_nll_nats", "spliced_nll_nats",
                  "spliced_delta_nll_nats")


def model_spec(name):
    match = re.fullmatch(r"(batch_topk|matryoshka)_(\d+)(?:_k(\d+))?", name)
    if match is None:
        raise ValueError(f"Invalid matched SAE name: {name}")
    method, width, k = match.groups()
    width, k = int(width), int(k or 20)
    if width < 1 or not 1 <= k <= width:
        raise ValueError(f"Invalid width or k in SAE name: {name}")
    return method, width, k


def expected_models(widths=DEFAULT_WIDTHS, extra_models=DEFAULT_EXTRA_MODELS):
    widths = sorted(set(widths))
    if not widths or any(width < 20 for width in widths):
        raise ValueError("Expected widths must support k=20")
    names = [model_name(method, width) for method in METHODS for width in widths]
    for name in extra_models:
        model_spec(name)
        if name not in names:
            names.append(name)
    return names


def requested_comparisons(names):
    """Vary one factor at a time; use the same reachable targets for each pair."""
    specs = {name: model_spec(name) for name in names}
    comparisons = []
    for left_index, left in enumerate(names):
        lm, lw, lk = specs[left]
        for right in names[left_index + 1:]:
            rm, rw, rk = specs[right]
            if lm == rm and lk == rk and lw != rw:
                comparisons.append((left, right, "width"))
            elif lw == rw and lk == rk and lm != rm:
                # Preserve the original Matryoshka minus BatchTopK direction.
                a, b = (left, right) if lm == "matryoshka" else (right, left)
                comparisons.append((a, b, "trainer"))
            elif lm == rm and lw == rw and lk != rk:
                a, b = (left, right) if lk > rk else (right, left)
                comparisons.append((a, b, "k"))
    return comparisons


def reachable_targets(analysis, level=.05, budget=1):
    rows = analysis["matches"]
    geo = {(r["pair_id"],r["target"]):r for r in rows if r["arm"] == "geoae_range"
           and r["level"] == level and r["status"] == "observed_interpolated"}
    return {(r["pair_id"],r["target"]):r for r in rows if r["arm"] == "sae"
            and r["level"] == level and r["feature_budget"] == budget
            and r["status"] == "observed_interpolated" and (r["pair_id"],r["target"]) in geo}


def compare_on_common_support(left, right):
    common = sorted(set(left)&set(right))
    groups = {name:defaultdict(list) for name in ("left_sibling","right_sibling","difference_sibling",
                                                "left_random","right_random","difference_random")}
    for key in common:
        a,b = left[key],right[key]
        if a["family"] != b["family"]: raise ValueError("Mismatched family IDs")
        for group in ("sibling","random"):
            av,bv = a["evaluation"][group]["delta_nll"],b["evaluation"][group]["delta_nll"]
            groups[f"left_{group}"][a["family"]].append(av)
            groups[f"right_{group}"][a["family"]].append(bv)
            groups[f"difference_{group}"][a["family"]].append(av-bv)
    return {"n_common_targets":len(common),"n_left_only":len(set(left)-set(right)),
            "n_right_only":len(set(right)-set(left)),"common_targets":[list(k) for k in common],
            **{key:family_bootstrap(values,500,0) for key,values in groups.items()}}


def number(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def read_sanity(path, missing, failed_sanity):
    if not path.exists():
        missing.append(str(path))
        return None
    value = json.loads(path.read_text())
    absent = [key for key in SANITY_METRICS if key not in value]
    if absent:
        missing.append(f"{path}: metrics {', '.join(absent)}")
        return None
    invalid = [key for key in SANITY_METRICS
               if not isinstance(value[key], (int, float)) or not math.isfinite(value[key])]
    if invalid or not value.get("finite_gate_passed", False):
        failed_sanity.append({"path": str(path), "invalid_metrics": invalid,
                              "finite_gate_passed": value.get("finite_gate_passed", False)})
        for key in invalid:
            value[key] = None
    return value


def sanity_row(kind, name, value):
    return f"| {kind} | {name} | " + " | ".join(number(value[key]) for key in SANITY_METRICS) + " |"


def run(root, widths=DEFAULT_WIDTHS, extra_models=DEFAULT_EXTRA_MODELS):
    root = Path(root)
    names = expected_models(widths, extra_models)
    summary = {"models":{},"baselines":{},"comparisons":[],"status":"complete",
               "expected_models":names,
               "primary_checkpoint":"last.pt at exactly 500M training row visits",
               "observed_matching":"secondary conditional interpolation; no extrapolation",
               "raw_fve_definition":"1 - sum((raw - restored_reconstruction)^2) / sum((raw - mean(raw, axis=0))^2)",
               "raw_reconstruction":"token mean added back, then per-dimension normalization inverted",
               "sanity_sample":"same held-out random controls within each dataset",
               "nll_units":"nats per next token; delta = spliced minus clean"}
    lines = ["# Matched-data TopK comparison", "", f"All {len(names)} matched SAEs use the same GeoAE dump, split, normalization and frozen token table.",
             "Primary checkpoints have exactly 500M training visits. See [preregistration and amendment](analysis_plan.md).", "",
             "Reconstruction sanity below is measured on the same held-out random controls within each dataset.",
             "RAW residual FVE = 1 − SSE / SST, where SSE compares the original residual with the full reconstruction",
             "after the frozen token mean is added back and normalization is inverted; SST centers the original RAW residual per dimension.",
             "Clean and spliced next-token loss are absolute NLL in nats/token; ΔNLL = spliced − clean.",
             "Reference values are read from this evaluation's sanity files, not copied from earlier approximate results.", "",
             "| Dataset | Model | L0 | RAW residual FVE | Clean next-token NLL | Spliced next-token NLL | Spliced ΔNLL |",
             "|---|---|---:|---:|---:|---:|---:|"]
    semantic_lines = ["", "Semantic coding and observed equal-damage removal:", "",
             "| Dataset | SAE | Intensity pairs / all | L0 | Raw FVE | Spliced dNLL | Observed-match sibling dNLL (SAE / GeoAE) | Targets / families |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    missing = []
    failed_sanity = []
    for kind in ("range","atlas"):
        reachable = {}
        for name in (*REQUIRED_BASELINES, "gemma_scope_262k"):
            path = root/"evaluation"/kind/(name+"_sanity.json")
            if name not in REQUIRED_BASELINES and not path.exists():
                continue
            sanity = read_sanity(path, missing, failed_sanity)
            if sanity is not None:
                summary["baselines"][f"{kind}/{name}"] = sanity
                lines.append(sanity_row(kind, name, sanity))
        for name in names:
            dest = root/"evaluation"/kind/name
            required = ("complete.json", "coding.json", "observed_sae_matching.json", "removal.json", "sanity.json")
            absent = [filename for filename in required if not (dest/filename).exists()]
            if absent:
                missing.append(f"{kind}/{name}: {', '.join(absent)}")
                continue
            coding_document = json.loads((dest/"coding.json").read_text())
            coding = coding_document["analysis"]
            matching = json.loads((dest/"observed_sae_matching.json").read_text())["analysis"]
            sanity = read_sanity(dest/"sanity.json", missing, failed_sanity)
            if sanity is None:
                continue
            lines.append(sanity_row(kind, name, sanity))
            primary = json.loads((dest/"removal.json").read_text())["analysis"]
            completion = json.loads((dest/"complete.json").read_text())
            rate = coding["summary"]["interpretation_counts"].get("same_feature_intensity_separation",0)
            matched = next((r for r in matching["comparisons"] if r["level"] == .05 and r["feature_budget"] == 1),
                {"sae":{"sibling_delta_nll":{"mean":None}},"geoae_range":{"sibling_delta_nll":{"mean":None}},
                 "n_paired_targets":0,"n_families":0})
            summary["models"][f"{kind}/{name}"] = {"sanity":sanity,"coding_summary":coding["summary"],
                "intensity_pair_fraction":rate/len(coding["pairs"]) if coding["pairs"] else None,
                "all_pairs":len(coding["pairs"]),
                "training_provenance":coding_document.get("provenance", {}),
                "evaluation_checkpoint":{key:completion.get(key) for key in ("checkpoint", "checkpoint_sha256", "smoke")},
                "primary_fit_calibrated":primary["paired_comparisons"],
                "observed_equal_damage":matching["comparisons"],"reachability":matching["status_counts"]}
            reachable[name] = reachable_targets(matching)
            sae = matched["sae"]["sibling_delta_nll"]["mean"]
            geo = matched["geoae_range"]["sibling_delta_nll"]["mean"]
            semantic_lines.append(f"| {kind} | {name} | {rate}/{len(coding['pairs'])} | {number(sanity['mean_l0'])} | {number(sanity['raw_fve'])} | {number(sanity['spliced_delta_nll_nats'])} | {number(sae)} / {number(geo)} | {matched['n_paired_targets']} / {matched['n_families']} |")
        for left,right,factor in requested_comparisons(names):
            if left in reachable and right in reachable:
                comparison = compare_on_common_support(reachable[left],reachable[right])
                summary["comparisons"].append({"kind":kind,"left":left,"right":right,"factor":factor,"level":.05,"feature_budget":1,**comparison})
    lines += semantic_lines
    lines += ["", "Observed matching conditions on measured held-out damage; it does not independently calibrate strengths on a new held-out split.",
              "The fit-calibrated records, all budgets and levels, unreachable targets, and range-operator diagnostics remain in each evaluation directory.",
              "Width, trainer, and k differences below use identical reachable targets and decision-family bootstrap intervals.", "",
              "| Dataset | Factor | Left / right | Common targets | Sibling dNLL difference (left minus right) | 95% family interval |",
              "|---|---|---|---:|---:|---|"]
    for comparison in summary["comparisons"]:
        diff = comparison["difference_sibling"]
        lines.append(f"| {comparison['kind']} | {comparison['factor']} | {comparison['left']} / {comparison['right']} | {comparison['n_common_targets']} | {number(diff['mean'])} | {diff['ci95']} |")
    lines += ["", "Training validation curves use normalized bypass input with the frozen token mean removed.",
              "Their bypass validation FVE is a different metric from RAW residual FVE above and must not be substituted for it.",
              "![Bypass validation curves](figures/validation_curves.png)"]
    summary["missing"] = missing
    summary["failed_sanity"] = failed_sanity
    if missing or failed_sanity: summary["status"] = "incomplete"
    lines += ["", f"Status: {summary['status']}. Missing: {len(missing)} required evaluation artifacts or metrics. Failed sanity gates: {len(failed_sanity)}."]
    if missing:
        lines += ["", "Missing required artifacts:", ""] + [f"- {entry}" for entry in missing]
    if failed_sanity:
        lines += ["", "Failed reconstruction sanity gates:", ""] + [f"- {entry['path']}" for entry in failed_sanity]
    write_json(root/"summary.json",summary)
    (root/"report.md").write_text("\n".join(lines)+"\n")
    plot_validation(root,names)


def plot_validation(root,names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1,3,figsize=(12,3.4))
    for name in names:
        path = root/"training"/name/"validation.jsonl"
        if not path.exists(): continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        visits = [r["visits"]/1e6 for r in rows]
        for axis,key in zip(axes,("mse","fve","mean_l0")):
            axis.plot(visits,[r[key] for r in rows],label=name)
    for axis,label in zip(axes,("Bypass validation MSE","Bypass validation FVE","Validation mean L0")):
        axis.set(xlabel="Training row visits (millions)",ylabel=label); axis.grid(alpha=.2)
    for k in sorted({model_spec(name)[2] for name in names}):
        axes[2].axhline(k,color="black",linestyle=":",alpha=.5)
    axes[0].legend(fontsize=6)
    figure.tight_layout(); dest = root/"figures"; dest.mkdir(exist_ok=True)
    for suffix in ("png","pdf","svg"): figure.savefig(dest/f"validation_curves.{suffix}",dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out",default=str(ROOT))
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--extra-models", nargs="*", default=list(DEFAULT_EXTRA_MODELS))
    args = parser.parse_args()
    run(args.out, args.widths, args.extra_models)


if __name__ == "__main__": main()
