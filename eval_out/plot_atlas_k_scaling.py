"""K scaling of FineWeb-Atlas topic homogeneity (14 topics, mean of the 3 draws, last word of each chunk) on Gemma 3 4B
layer 33: plain k-means on the residual stream, plain k-means on the token-removed residual stream, plain k-means on the
Gemma Scope 16k codes and the bypass GeoAE, each with K = 2,000 / 4,000 / 8,000 clusters. Every value is read from the
nmi_filtered*_k{K}.json files of scripts/argo/atlas_lastword_gs16k_plain_kscale.sbatch (rows kept iff their cluster holds
more than 5 rows in every plotted space at that K); nothing is typed by hand.

    python eval_out/plot_atlas_k_scaling.py [--variant 4|3sae] [--out <stem>]
Writes <stem>.png, <stem>.pdf and <stem>.md (the plotted values, per draw, with rows kept).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/kscale")
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="4", choices=["4", "3sae"], help="4 = with the token-removed control in the filter")
ap.add_argument("--dir", default=str(D), help="directory of the nmi_filtered*_k{K}.json files")
ap.add_argument("--out", default="")
args = ap.parse_args()
D = Path(args.dir)
out = Path(args.out or D / f"atlas_k_scaling_{args.variant}")

KS = [(2000, ""), (4000, "4k"), (8000, "8k")]
SETS = ["atlas14_s0", "atlas14_s1", "atlas14_s2"]
# (arm stem, label, color, linestyle, marker): GeoAE accent blue, SAE orange, baselines neutral gray (paper_figures.py)
SERIES = [("bypass", "GeoAE (bypass)", "#2a78d6", "-", "o"),
          ("gs16k_plain", "Gemma Scope SAE, k-means on codes", "#eb6834", "-", "s"),
          ("plain_tokmean", "Token-removed residual, k-means", "#898781", (0, (4, 2)), "^"),
          ("plain", "Residual stream, k-means", "#52514e", "-", "D")]
if args.variant == "3sae":
    SERIES = [s for s in SERIES if s[0] != "plain_tokmean"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

R = {K: json.loads((D / f"nmi_filtered{args.variant}_k{K}.json").read_text())["sets"] for K, _ in KS}
H = {a: np.array([[R[K][s]["point"][a + sf]["homogeneity"] for s in SETS] for K, sf in KS]) for a, *_ in SERIES}  # (K, draw)
mean = {a: v.mean(1) for a, v in H.items()}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
x = np.arange(len(KS))
dodge = np.linspace(-0.075, 0.075, len(SERIES))
for i, ((a, label, col, ls, mk), dx) in enumerate(zip(SERIES, dodge)):
    ax.vlines(x + dx, H[a].min(1), H[a].max(1), color=col, lw=1.2, alpha=0.45, zorder=2)      # range of the 3 draws
    ax.plot(x + dx, mean[a], color=col, lw=2, ls=ls, marker=mk, ms=9 if mk == "^" else 7.5, mec=SURFACE, mew=1.5, label=label,
            zorder=6 - i)                                                                    # GeoAE and the SAE on top
ends = sorted(((mean[a][-1], a, label) for a, label, *_ in SERIES), reverse=True)
ypos, gap = [], 0.017
for v, a, label in ends:                                                                       # no overlapping end labels
    ypos.append(v if not ypos else min(v, ypos[-1] - gap))
for (v, a, label), y in zip(ends, ypos):
    ax.annotate(f"{v:.3f}".lstrip("0"), (x[-1] + 0.15, y), color=INK, va="center", ha="left", fontsize=9,
                fontweight="bold" if a == "bypass" else "normal", annotation_clip=False)
for a, *_ in SERIES:
    ax.annotate(f"{mean[a][0]:.3f}".lstrip("0"), (x[0] - 0.14, mean[a][0]), color=INK2, va="center", ha="right",
                fontsize=8.5, annotation_clip=False)

ax.set_xticks(x, [f"{K:,}" for K, _ in KS])
ax.set_xlim(-0.36, len(KS) - 1 + 0.36)
lo = min(v.min() for v in H.values())
hi = max(v.max() for v in H.values())
ax.set_ylim(np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20 + 0.0)
ax.set_xlabel("Number of clusters K", color=INK2)
ax.set_ylabel("Topic homogeneity", color=INK2)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}".lstrip("0"))
gain = {a: mean["bypass"][-1] - mean[a][-1] for a in mean if a != "bypass"}
gain0 = {a: mean["bypass"][0] - mean[a][0] for a in mean if a != "bypass"}
fig.suptitle("GeoAE keeps the highest topic homogeneity as K grows", x=0.065, y=0.975, ha="left", fontsize=11.5,
             fontweight="bold", color=INK)
fig.text(0.065, 0.905, "Gemma 3 4B, layer 33 · FineWeb-Atlas, 14 topics, last word of each chunk · line = mean of 3 topic "
         "draws, bars = their range", ha="left", fontsize=7.6, color=INK2)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.2, handlelength=2.6,
                columnspacing=2.0, labelcolor=INK2)
fig.subplots_adjust(left=0.105, right=0.925, top=0.86, bottom=0.25)
for ext in ("png", "pdf"):
    fig.savefig(out.with_suffix(f".{ext}"), facecolor=SURFACE)

lines = [f"# Atlas 14-topic homogeneity vs K (Gemma 3 4B L33, last word; filter variant `{args.variant}`)", "",
         "Rows kept iff their cluster holds more than 5 rows in every listed space at that K. Mean of the 3 draws "
         "(draw 0 / 1 / 2).", "", "| Space | " + " | ".join(f"K = {K:,}" for K, _ in KS) + " |", "|---|" + "---|" * len(KS)]
for a, label, *_ in SERIES:
    lines.append(f"| {label} | " + " | ".join(f"**{mean[a][i]:.3f}** ({' / '.join(f'{v:.3f}' for v in H[a][i])})"
                                              for i in range(len(KS))) + " |")
for a, label, *_ in SERIES[1:]:
    cells = []
    for (K, sf) in KS:
        d = [R[K][s]["diff_ref_minus_arm"][a + sf]["homogeneity"] for s in SETS]
        sig = sum(1 for v in d if v["ci"][0] > 0 or v["ci"][1] < 0)
        cells.append(f"{np.mean([v['point'] for v in d]):+.3f} ({sig}/3 draws with CI excluding 0)")
    lines.append(f"| GeoAE − {label} | " + " | ".join(cells) + " |")
lines.append("| Rows kept (draw 0 / 1 / 2) | " + " | ".join(
    " / ".join(f"{R[K][s]['n_kept']:,} ({R[K][s]['kept_share']:.0%})" for s in SETS) for K, _ in KS) + " |")
lines.append("| Clusters used, draw 1 (" + ", ".join(lab.split(",")[0] for _, lab, *_ in SERIES) + ") | " + " | ".join(
    " / ".join(str(R[K]["atlas14_s1"]["point"][a + sf]["clusters"]) for a, *_ in SERIES) for K, sf in KS) + " |")
out.with_suffix(".md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote", out.with_suffix(".png"), out.with_suffix(".pdf"), out.with_suffix(".md"))
