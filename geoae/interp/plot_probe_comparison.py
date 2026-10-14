"""Plot linear/nonlinear probe convergence and held-out comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


COLORS = {"raw_base": "#2878B5", "ae": "#E07A2D"}
LABELS = {"raw_base": "Raw base", "ae": "GeoAE d3072 PD"}


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _ae_name(result: dict, requested: str | None) -> str:
    if requested:
        if requested not in result["spaces"]:
            raise KeyError(f"{requested!r} not found in {list(result['spaces'])}")
        return requested
    names = [name for name in result["spaces"] if name != "raw_base"]
    if len(names) != 1:
        raise ValueError("pass --ae_name when the result contains multiple AE spaces")
    return names[0]


def _curve(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Plot only epochs shared by every seed, so every band always summarizes
    # the same number of runs.
    n = min(len(run["history"]) for run in runs)
    values = np.asarray([[run["history"][i][key] for i in range(n)] for run in runs])
    epochs = np.arange(1, n + 1)
    return epochs, values.mean(0), values.std(0, ddof=1)


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "font.size": 10,
        "axes.titleweight": "bold",
    })


def _convergence_axis(
    ax: plt.Axes,
    result: dict,
    ae_name: str,
    metric: str,
    title: str,
) -> None:
    for name, alias in [("raw_base", "raw_base"), (ae_name, "ae")]:
        space = result["spaces"][name]
        color = COLORS[alias]
        for split, linestyle, alpha in [("train", "--", 0.10), ("val", "-", 0.16)]:
            key = f"{split}_{metric}"
            epochs, mean, std = _curve(space["runs"], key)
            ax.plot(
                epochs, mean, color=color, linestyle=linestyle, linewidth=2,
                label=f"{LABELS[alias]} — {split}",
            )
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=alpha)

        best_epoch = space["summary"]["best_epoch_mean"]
        summary_key = "cross_entropy" if metric == "cross_entropy" else "accuracy"
        best_value = space["summary"]["validation"][summary_key]["mean"]
        ax.scatter(
            [best_epoch], [best_value], color=color, marker="*", s=115,
            edgecolor="white", linewidth=0.7, zorder=5,
        )

    ax.set_title(title)
    ax.set_xlabel("Probe epoch")
    if metric == "cross_entropy":
        ax.set_ylabel("Cross-entropy (lower is better)")
    else:
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, ncol=2)


def _save_single_convergence(
    result: dict, ae_name: str, metric: str, probe_name: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    metric_label = "cross-entropy" if metric == "cross_entropy" else "accuracy"
    _convergence_axis(
        ax, result, ae_name, metric,
        f"{probe_name.title()} probe {metric_label} convergence",
    )
    fig.suptitle("Mean ± 1 SD across seeds; stars mark selected checkpoints", fontsize=9, y=0.96)
    fig.savefig(out)
    plt.close(fig)


def _dashboard(linear: dict, nonlinear: dict, ae_name: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _convergence_axis(axes[0, 0], linear, ae_name, "cross_entropy", "Linear: loss")
    _convergence_axis(axes[0, 1], linear, ae_name, "accuracy", "Linear: accuracy")
    _convergence_axis(axes[1, 0], nonlinear, ae_name, "cross_entropy", "Nonlinear: loss")
    _convergence_axis(axes[1, 1], nonlinear, ae_name, "accuracy", "Nonlinear: accuracy")
    fig.suptitle(
        "BiasBios profession probe convergence — raw base vs GeoAE d3072 PD\n"
        "Mean ± 1 SD over seeds 42/43/44; stars are validation-loss selections",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def _test_metrics(linear: dict, nonlinear: dict, ae_name: str, out: Path) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    x = np.arange(2)
    width = 0.34
    for ax, (metric, label) in zip(axes, metrics):
        for offset, (space_name, alias) in zip(
            [-width / 2, width / 2], [("raw_base", "raw_base"), (ae_name, "ae")]
        ):
            means, stds = [], []
            for result in [linear, nonlinear]:
                item = result["spaces"][space_name]["summary"]["test"][metric]
                means.append(item["mean"])
                stds.append(item["std"])
            ax.bar(
                x + offset, means, width, yerr=stds, capsize=4,
                color=COLORS[alias], label=LABELS[alias], alpha=0.9,
            )
        ax.set_title(label)
        ax.set_xticks(x, ["Linear", "Nonlinear"])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.70, 0.86)
    axes[0].set_ylabel("Held-out test score")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("Held-out BiasBios profession performance", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def _split_accuracy(linear: dict, nonlinear: dict, ae_name: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    x = np.arange(2)
    width = 0.24
    for ax, (probe_name, result) in zip(axes, [("Linear", linear), ("Nonlinear", nonlinear)]):
        for i, split in enumerate(["train", "validation", "test"]):
            means, stds = [], []
            for space in ["raw_base", ae_name]:
                item = result["spaces"][space]["summary"][split]["accuracy"]
                means.append(item["mean"])
                stds.append(item["std"])
            ax.bar(
                x + (i - 1) * width, means, width, yerr=stds, capsize=3,
                label=split.title(), alpha=0.88,
            )
        ax.set_title(f"{probe_name} probe")
        ax.set_xticks(x, [LABELS["raw_base"], LABELS["ae"]])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.78, 0.94)
    axes[0].set_ylabel("Accuracy at selected checkpoint")
    axes[-1].legend(frameon=False)
    fig.suptitle("Generalization at the validation-selected checkpoint", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def _selection_epochs(linear: dict, nonlinear: dict, ae_name: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    entries = [("Linear", linear), ("Nonlinear", nonlinear)]
    x = np.arange(len(entries))
    width = 0.34
    for offset, (space_name, alias) in zip(
        [-width / 2, width / 2], [("raw_base", "raw_base"), (ae_name, "ae")]
    ):
        best = [r["spaces"][space_name]["summary"]["best_epoch_mean"] for _, r in entries]
        stopped = [
            np.mean([run["stopped_epoch"] for run in r["spaces"][space_name]["runs"]])
            for _, r in entries
        ]
        bars = ax.bar(x + offset, stopped, width, color=COLORS[alias], alpha=0.34)
        ax.scatter(x + offset, best, color=COLORS[alias], marker="*", s=150, zorder=4)
        for bar, value in zip(bars, stopped):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, [name for name, _ in entries])
    ax.set_ylabel("Epoch")
    ax.set_title("Best validation epoch (star) and early-stop epoch (bar)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=COLORS["raw_base"], lw=8, alpha=0.5, label=LABELS["raw_base"]),
            plt.Line2D([0], [0], color=COLORS["ae"], lw=8, alpha=0.5, label=LABELS["ae"]),
        ],
        frameon=False,
    )
    fig.savefig(out)
    plt.close(fig)


def _test_deltas(linear: dict, nonlinear: dict, ae_name: str, out: Path) -> None:
    score_metrics = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."), ("macro_f1", "Macro-F1")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    x = np.arange(len(score_metrics))
    width = 0.34
    for offset, (probe_name, result) in zip([-width / 2, width / 2], [("Linear", linear), ("Nonlinear", nonlinear)]):
        values = [
            100 * (
                result["spaces"][ae_name]["summary"]["test"][metric]["mean"]
                - result["spaces"]["raw_base"]["summary"]["test"][metric]["mean"]
            )
            for metric, _ in score_metrics
        ]
        axes[0].bar(x + offset, values, width, label=probe_name, alpha=0.9)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x, [label for _, label in score_metrics])
    axes[0].set_ylabel("GeoAE − base (percentage points)")
    axes[0].set_title("Test score delta")
    ce = []
    for _, result in [("Linear", linear), ("Nonlinear", nonlinear)]:
        ce.append(
            result["spaces"][ae_name]["summary"]["test"]["cross_entropy"]["mean"]
            - result["spaces"]["raw_base"]["summary"]["test"]["cross_entropy"]["mean"]
        )
    axes[1].bar([0, 1], ce, color=["#5B8FF9", "#61DDAA"], width=0.58)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks([0, 1], ["Linear", "Nonlinear"])
    axes[1].set_ylabel("GeoAE − base cross-entropy")
    axes[1].set_title("Test loss delta (lower is better)")
    axes[0].legend(frameon=False)
    fig.suptitle("GeoAE relative to the raw base", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--linear", type=Path, required=True)
    ap.add_argument("--nonlinear", type=Path, required=True)
    ap.add_argument("--ae_name")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    linear = _load(args.linear)
    nonlinear = _load(args.nonlinear)
    ae_name = _ae_name(linear, args.ae_name)
    if ae_name not in nonlinear["spaces"]:
        raise KeyError(f"{ae_name!r} is missing from nonlinear results")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    outputs = []
    for probe_name, result in [("linear", linear), ("nonlinear", nonlinear)]:
        for metric, suffix in [("cross_entropy", "loss"), ("accuracy", "accuracy")]:
            path = args.out_dir / f"{probe_name}_{suffix}.png"
            _save_single_convergence(result, ae_name, metric, probe_name, path)
            outputs.append(path)

    jobs = [
        ("convergence_dashboard.png", lambda p: _dashboard(linear, nonlinear, ae_name, p)),
        ("test_metrics.png", lambda p: _test_metrics(linear, nonlinear, ae_name, p)),
        ("split_accuracy.png", lambda p: _split_accuracy(linear, nonlinear, ae_name, p)),
        ("selection_epochs.png", lambda p: _selection_epochs(linear, nonlinear, ae_name, p)),
        ("test_deltas.png", lambda p: _test_deltas(linear, nonlinear, ae_name, p)),
    ]
    for filename, make in jobs:
        path = args.out_dir / filename
        make(path)
        outputs.append(path)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
