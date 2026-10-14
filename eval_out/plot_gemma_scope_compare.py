"""Export Gemma Scope sibling coding, reconstruction and observed-matched removal.

python eval_out/plot_gemma_scope_compare.py results/.../gemma_scope_compare --pdf
The input is frozen analysis JSON; no model, dataset download or GPU is used.
Missing observations are omitted. Removal interpolation is observational and
uses the same reachable target intersection for both arms at the declared level.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np


DATASET_NAMES = {"atlas": "Atlas siblings", "range": "GeoAE range pairs"}
CATEGORIES = [
    ("same_feature_intensity_separation", "Shared atom, separated strengths", "#326879"),
    ("shared_feature_without_heldout_separation", "Shared atom, unseparated", "#79AAA0"),
    ("distinct_parallel", "Separate parallel atoms", "#C99C46"),
    ("distinct_unrelated", "Separate unrelated atoms", "#756A99"),
    ("unresolved", "Association unresolved", "#DDDDDD"),
]
ARM_COLORS = {"sae": "#C4533C", "geoae": "#26709C"}


def _load(path):
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    return document.get("analysis", document)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _source(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_runs(root, *, level, bootstrap_samples, seed):
    """Load only available artifacts and delegate observed matching to its audited API."""
    runs = []
    module_spec = importlib.util.find_spec("geoae.interp.sae_matched")
    observed_analysis = None
    if module_spec is not None:
        from geoae.interp.sae_matched import analyze_observed_matching
        observed_analysis = analyze_observed_matching
    for coding_path in sorted(root.rglob("coding.json")):
        directory = coding_path.parent
        coding = _load(coding_path)
        kind = directory.parent.name
        match = re.search(r"width_([^_]+)", directory.name)
        width = match[1] if match else f"{coding['n_features']:,} atoms"
        run = {"directory": str(directory.resolve()), "kind": kind, "width": width,
               "dataset": DATASET_NAMES.get(kind, kind), "coding": coding,
               "reconstruction": _load(directory / "reconstruction.json"),
               "removal": _load(directory / "removal.json"),
               "complete": (directory / "complete.json").exists(),
               "sources": {"coding": _source(coding_path)}}
        for name in ("reconstruction", "removal"):
            path = directory / f"{name}.json"
            if path.exists():
                run["sources"][name] = _source(path)
        persisted = directory / "observed_matching.json"
        run["observed_matching"] = _load(persisted)
        if run["observed_matching"] is not None:
            run["sources"]["observed_matching"] = _source(persisted)
        saved_levels = (run["observed_matching"] or {}).get("metadata", {}).get("levels", [])
        level_available = any(math.isclose(value, level) for value in saved_levels)
        if not level_available and run["removal"] is not None and observed_analysis is not None:
            run["observed_matching"] = observed_analysis(
                run["removal"], levels=(level,), bootstrap_samples=bootstrap_samples, seed=seed)
        runs.append(run)
    # Numeric width order remains stable for 16k and 262k.
    return sorted(runs, key=lambda run: (run["kind"], run["coding"]["n_features"]))


def _style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def _coding_panel(ax, runs):
    x = np.arange(len(runs))
    bottom = np.zeros(len(runs), dtype=float)
    rows = []
    known = {item[0] for item in CATEGORIES}
    counts = []
    for run in runs:
        actual = Counter(pair.get("interpretation", "unresolved") for pair in run["coding"]["pairs"])
        unknown = set(actual) - known
        if unknown:
            raise ValueError(f"Unrecognized coding interpretations: {sorted(unknown)}")
        counts.append(actual)
    for category, label, color in CATEGORIES:
        heights = np.array([count[category] for count in counts])
        ax.bar(x, heights, bottom=bottom, width=0.66, color=color, label=label,
               edgecolor="white", linewidth=0.5)
        for index, height in enumerate(heights):
            if height:
                ax.text(index, bottom[index] + height / 2, str(height), ha="center", va="center",
                        fontsize=8, color="white" if category in ("same_feature_intensity_separation", "distinct_unrelated") else "#242424")
        bottom += heights
    for index, run in enumerate(runs):
        summary = run["coding"]["summary"]
        n_pairs = len(run["coding"]["pairs"])
        ax.text(index, n_pairs + max(0.4, max(bottom, default=1) * 0.02),
                f"{summary['n_heldout_supported']}/{summary['n_labels']} labels", ha="center", va="bottom", fontsize=7)
        rows.append({"dataset": run["kind"], "width": run["width"], "n_pairs": n_pairs,
                     "n_labels": summary["n_labels"], "heldout_supported_labels": summary["n_heldout_supported"],
                     "category_counts": dict(counts[index])})
    ax.set_xticks(x, [f"{'Atlas' if r['kind'] == 'atlas' else 'Ranges'}\n{r['width']}" for r in runs])
    ax.set_ylabel("Sibling pairs (count)")
    ax.set_ylim(0, max(max(bottom, default=1) * 1.2, 1))
    ax.set_title("a  Sibling feature coding", loc="left", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False, fontsize=7, ncol=1)
    return rows


def _reconstruction_panel(ax, runs):
    source_rows, seen_geo = [], set()
    all_budgets = set()
    for run in runs:
        reconstruction = run["reconstruction"]
        if reconstruction is None:
            continue
        matched = reconstruction.get("matched_active_budget_reconstructions") or []
        style = "-" if run["kind"] == "atlas" else "--"
        dataset = "Atlas" if run["kind"] == "atlas" else "Ranges"
        for arm in ("sae", "geoae"):
            points = []
            for row in matched:
                metric = row[arm]["metrics"]["controls"]
                value = metric["normalized_mse"]
                if _finite(value) and metric["n_rows"] > 0:
                    points.append((row["requested_budget"], value, metric["n_rows"]))
            if not points:
                continue
            all_budgets.update(point[0] for point in points)
            source_rows.append({"dataset": run["kind"], "width": run["width"], "arm": arm,
                                "points": [{"active_cap": x, "control_nmse": y, "n_controls": n}
                                           for x, y, n in points],
                                "normalization": reconstruction["normalization"]})
            geo_key = (run["kind"], tuple(points))
            if arm == "geoae" and geo_key in seen_geo:
                continue
            if arm == "geoae":
                seen_geo.add(geo_key)
            marker = "o" if arm == "geoae" or run["width"] == "16k" else "^"
            label = f"GeoAE · {dataset}" if arm == "geoae" else f"SAE {run['width']} · {dataset}"
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    color=ARM_COLORS[arm], linestyle=style, marker=marker,
                    markersize=4, linewidth=1.5, label=label)
    ax.set_title("b  Matched active-contribution budget", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Maximum active decoder contributions per token")
    ax.set_ylabel("Control reconstruction NMSE")
    if source_rows:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(sorted(all_budgets)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        counts = sorted({point["n_controls"] for row in source_rows for point in row["points"]})
        ax.text(0.01, 0.02, "Control rows per curve: " + ", ".join(str(n) for n in counts),
                transform=ax.transAxes, fontsize=7, va="bottom")
    else:
        ax.text(0.5, 0.5, "Matched reconstruction unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#666666")
    return source_rows


def _removal_panel(ax, runs, *, level, feature_budget):
    rows = []
    for run in runs:
        observed = run["observed_matching"]
        if observed is None:
            continue
        comparisons = [row for row in observed.get("comparisons", [])
                       if row["feature_budget"] == feature_budget and math.isclose(row["level"], level)
                       and row["n_paired_targets"] > 0]
        if len(comparisons) > 1:
            raise ValueError("Observed-matching API returned duplicate budget/level comparisons")
        if comparisons:
            row = comparisons[0]
            if all(_finite(row[arm]["sibling_delta_nll"]["mean"]) for arm in ("sae", "geoae_range")):
                rows.append({"dataset": run["kind"], "width": run["width"], "comparison": row,
                             "status_counts": observed.get("status_counts", {})})
    x = np.arange(len(rows))
    for arm, label, offset in (("sae", "SAE", -0.18), ("geoae_range", "GeoAE range edit", 0.18)):
        means = [row["comparison"][arm]["sibling_delta_nll"]["mean"] for row in rows]
        if not means:
            continue
        color = ARM_COLORS["sae" if arm == "sae" else "geoae"]
        ax.scatter(x + offset, means, color=color, label=label, s=27, zorder=3)
        for index, (row, value) in enumerate(zip(rows, means)):
            ci = row["comparison"][arm]["sibling_delta_nll"].get("ci95")
            if ci is not None and len(ci) == 2 and all(_finite(v) for v in ci):
                # Draw the reported interval directly, even if a percentile
                # interval does not bracket its bootstrap point estimate.
                ax.vlines(index + offset, ci[0], ci[1], color=color, linewidth=1.2)
                ax.hlines(ci, index + offset - 0.035, index + offset + 0.035, color=color, linewidth=1)
        ax.set_xticks(x, [f"{'Atlas' if r['dataset'] == 'atlas' else 'Ranges'} {r['width']}\n"
                         f"n={r['comparison']['n_paired_targets']}, f={r['comparison']['n_families']}" for r in rows])
    ax.set_title(f"c  Observed target ΔNLL = {level:g} nats", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sibling ΔNLL (nats)")
    ax.axhline(0, color="#666666", linewidth=0.7, zorder=1)
    if rows:
        ax.legend(frameon=False, fontsize=7)
        ax.text(0.01, 0.98, f"SAE handle cap {feature_budget}; family-weighted means\n"
                "n: paired targets; f: decision families", transform=ax.transAxes, fontsize=7, va="top")
        ax.margins(x=0.22, y=0.18)
    else:
        ax.text(0.5, 0.5, "No observed reachable paired matches\nat this level and feature cap",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#666666")
        ax.set_xticks([])
    return rows


def render_figures(result_root, *, out_dir=None, level=0.05, feature_budget=1,
                   bootstrap_samples=500, seed=0, dpi=300, pdf=False):
    root = Path(result_root).resolve()
    out = Path(out_dir).resolve() if out_dir else root / "figures"
    runs = load_runs(root, level=level, bootstrap_samples=bootstrap_samples, seed=seed)
    if not runs:
        raise ValueError(f"No coding.json artifacts found under {root}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), gridspec_kw={"width_ratios": [1.1, 1.05, 1]})
    for ax in axes:
        _style_axis(ax)
    coding = _coding_panel(axes[0], runs)
    reconstruction = _reconstruction_panel(axes[1], runs)
    removal = _removal_panel(axes[2], runs, level=level, feature_budget=feature_budget)
    fig.suptitle("Gemma Scope 2 versus GeoAE · Gemma 3 4B, layer 33", fontsize=13, y=0.97)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.85, bottom=0.33, wspace=0.34)
    fig.text(0.06, 0.015, "Counts and missing cases are preserved in the source JSON. Removal uses conditional interpolation of observed held-out curves;\n"
             "the paired reachable intersection is shared by both arms. Active-contribution caps match; dictionary sizes and retained biases differ.",
             fontsize=7, color="#555555", va="bottom")
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "gemma_scope_comparison"
    paths = []
    for extension in ("svg", "png") + (("pdf",) if pdf else ()):
        path = prefix.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    sources = {
        "result_root": str(root), "observed_target_delta_nll": level, "sae_handle_cap": feature_budget,
        "coding": coding, "reconstruction": reconstruction, "removal": removal,
        "input_runs": [{"dataset": run["kind"], "width": run["width"], "complete": run["complete"],
                        "sources": run["sources"],
                        "removal_status_counts": (run["observed_matching"] or {}).get("status_counts"),
                        "removal_matching_metadata": (run["observed_matching"] or {}).get("metadata")}
                       for run in runs],
    }
    source_path = prefix.with_name(prefix.name + "_source.json")
    source_path.write_text(json.dumps(sources, indent=2, allow_nan=False) + "\n")
    paths.append(str(source_path))
    caption = (
        "Gemma Scope 2 versus GeoAE at Gemma 3 4B layer 33. "
        "(a) Sibling pair coding, with fit-only SAE handle selection and held-out label-association support; "
        "segment labels are raw pair counts and annotations give supported labels / all labels. "
        "(b) Reconstruction NMSE on held-out random controls at equal caps on raw decoder contributions per token, "
        "selected by coefficient magnitude times decoder norm. SAE decoder bias and GeoAE token bypass are retained. "
        "Dictionary widths and total parameters are not matched. "
        f"(c) Sibling collateral at observed target ΔNLL = {level:g} nats, SAE handle cap {feature_budget}. "
        "Values conditionally interpolate observed held-out intervention curves, with no extrapolation and the same "
        "reachable target intersection for both arms. Points are means over decision families; intervals are the "
        "reported family-bootstrap 95% confidence intervals. Counts give paired targets (n) and families (f). "
        "Missing or unreachable observations are omitted. The source JSON records every input artifact hash, "
        "plotted value, row count and matching status.\n"
    )
    caption_path = prefix.with_suffix(".md")
    caption_path.write_text(caption)
    paths.append(str(caption_path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_root")
    parser.add_argument("--out-dir")
    parser.add_argument("--level", type=float, default=0.05)
    parser.add_argument("--feature-budget", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    # Standalone scripts launched by path must find the checkout's pure helpers.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for path in render_figures(args.result_root, out_dir=args.out_dir, level=args.level,
                               feature_budget=args.feature_budget, bootstrap_samples=args.bootstrap_samples,
                               seed=args.seed, dpi=args.dpi, pdf=args.pdf):
        print(path)


if __name__ == "__main__":
    main()
