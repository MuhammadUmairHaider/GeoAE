"""Audit and compare completed absolute-activation and d-prime number runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

paths = {"abs": "results/range_number_dpc_control.json",
         "dprime": "results/range_number_dpc_dprime.json"}
runs = {name: json.loads(Path(path).read_text()) for name, path in paths.items()}
old, new = runs["abs"], runs["dprime"]
assert {k for k in new["meta"] if new["meta"][k] != old["meta"][k]} == {"out", "saliency"}
for field in ["fit_examples", "test_examples", "neutral_prompts", "baselines"]:
    assert old[field] == new[field], field
labels = np.array([r["subject_number"] for r in new["test_examples"]])
lemmas = np.array([r["subject_lemma"] for r in new["test_examples"]])
nouns = sorted(set(lemmas))
for field in ["subject_lemma", "template", "prompt", "id"]:
    assert set(r[field] for r in new["fit_examples"]).isdisjoint(
        r[field] for r in new["test_examples"])


def target_rows(run, space, mode, alpha, field):
    values = np.zeros(len(labels))
    for source, direction in enumerate(["singular", "plural"]):
        arm = run["arms"][f"{space}:suppress_{direction}:{mode}:a{alpha:.1f}"]
        mask = labels == source
        values[mask] = np.asarray(arm["rows"][field])[mask]
    return values


summary = {}
for selector, run in runs.items():
    assert run["status"] == "complete" and len(run["arms"]) == 144
    for key, arm in run["arms"].items():
        source = int("suppress_plural" in key)
        rows = arm["rows"]
        drop = np.asarray(rows["ref_correct"], float) - np.asarray(rows["correct"], float)
        for name, mask in [("target", labels == source), ("complement", labels != source)]:
            assert np.isclose(arm["all"][f"{name}_drop"], drop[mask].mean())
        if ":global:" in key:
            # Gate diagnostics describe the fitted selector even for ungated arms.
            for field in arm:
                if field not in {"gate_fire_target", "gate_fire_complement"}:
                    assert old["arms"][key][field] == new["arms"][key][field], (key, field)
    for space in ["h", "z", "z_rot0", "z_rot1", "z_rot2", "z_shuffled"]:
        for mode in ["global", "salient", "range", "transport"]:
            for alpha in [.5, 1., 2.]:
                arms = [run["arms"][f"{space}:suppress_{d}:{mode}:a{alpha:.1f}"]
                        for d in ["singular", "plural"]]
                probs = target_rows(run, space, mode, alpha, "gold_pair_probability")
                row = {"strict_flip": float(np.mean(probs < .5)),
                       "tie_rate": float(np.mean(probs == .5))}
                for field in ["target_counterpart_top1", "complement_drop",
                              "complement_kl", "complement_top1_change"]:
                    row[field] = float(np.mean([a["all"][field] for a in arms]))
                for field in ["kl", "top1_change"]:
                    row[f"neutral_{field}"] = float(np.mean([a["neutral"][field] for a in arms]))
                summary[f"{selector}:{space}:{mode}:a{alpha:.1f}"] = row

rng = np.random.RandomState(20260917)
indices = rng.randint(len(nouns), size=(20000, len(nouns)))
contrasts = {}
for mode in ["range", "transport"]:
    z = target_rows(new, "z", mode, 1., "gold_pair_probability") < .5
    h = target_rows(new, "h", mode, 1., "gold_pair_probability") < .5
    delta = z.astype(float) - h.astype(float)
    grouped = np.array([delta[lemmas == noun].mean() for noun in nouns])
    contrasts[mode] = {"ae_minus_base_strict_flip": float(grouped.mean()),
                       "noun_bootstrap_95": np.quantile(grouped[indices].mean(1), [.025, .975]).tolist(),
                       "noun_differences": dict(zip(nouns, grouped.tolist()))}
report = {"sources": paths, "checked_arms": 288,
          "audit": "Only output path and selector differ; datasets, baselines and all global editing outcomes are identical. Accuracy drops recomputed from rows.",
          "summary": summary, "dprime_alpha1_contrasts": contrasts,
          "caveats": ["Strict flips exclude ties and are a forced-choice endpoint.",
                      "KL uses each representation's own baseline; AE reconstruction cost is separate.",
                      "16 neutral prompts; one checkpoint; noun bootstrap excludes training-seed and template-family uncertainty.",
                      "Equal alpha and coordinate fraction do not match perturbation norms or coordinate counts."]}
Path("eval_out/number_selector_review.json").write_text(json.dumps(report, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5))
categories = [("h", "range"), ("z", "range"), ("h", "transport"), ("z", "transport")]
x = np.arange(len(categories))
for offset, selector, color in [(-.18, "abs", "#868e96"), (.18, "dprime", "#4263eb")]:
    vals = [100 * summary[f"{selector}:{s}:{m}:a1.0"]["strict_flip"] for s, m in categories]
    bars = ax.bar(x + offset, vals, width=.36, label=selector, color=color)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
ax.set_xticks(x, ["Base\nrange", "AE\nrange", "Base\ntransport", "AE\ntransport"])
ax.set_ylim(0, 116)
ax.set_ylabel("Strict is/are reversals (%)")
ax.set_title("Discriminative selection closes most of the gap (alpha = 1)")
ax.legend(loc="upper left", ncol=2, fontsize=9)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=.2)
for selector, space, mode, label, color, style in [
    ("dprime", "h", "range", "Base d-prime range", "#f08c00", "--"),
    ("dprime", "h", "transport", "Base d-prime transport", "#e03131", "--"),
    ("dprime", "z", "range", "AE d-prime range", "#4263eb", "-"),
    ("dprime", "z", "transport", "AE d-prime transport", "#0b9b83", "-"),
    ("abs", "z", "transport", "AE abs transport (previous)", "#862e9c", ":"),
]:
    vals = [summary[f"{selector}:{space}:{mode}:a{a:.1f}"] for a in [.5, 1., 2.]]
    bx.plot([r["neutral_kl"] for r in vals], [100*r["strict_flip"] for r in vals],
            marker="o", color=color, linestyle=style, label=label)
bx.set_xscale("log")
bx.set_ylim(0, 106)
bx.set_xlabel("Neutral KL from own baseline (lower is better)")
bx.set_ylabel("Strict is/are reversals (%)")
bx.set_title("Collateral tradeoff depends on the selector")
bx.grid(alpha=.2)
bx.legend(loc="lower right", fontsize=8)
fig.text(.5, .015, "144 held-out prompts; 16 neutral prompts. Curves: alpha 0.5, 1, 2. AE KL excludes reconstruction cost.", ha="center", fontsize=9)
fig.tight_layout(rect=[0, .045, 1, 1])
out = Path("figures/number_control")
out.mkdir(parents=True, exist_ok=True)
for suffix in ["png", "pdf"]:
    fig.savefig(out / f"selector_comparison.{suffix}", dpi=180)
plt.close(fig)
print(report["audit"])
print(json.dumps(contrasts, indent=2))
print("Saved eval_out/number_selector_review.json and figures/number_control/selector_comparison.{png,pdf}")
