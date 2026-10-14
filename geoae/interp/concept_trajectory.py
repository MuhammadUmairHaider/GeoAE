"""
The concept ladder, measured at EVERY epoch of a run.

concept_probe answers "what does this clustering know" for one checkpoint. This
answers "when during training did it learn it, and what did it trade away" — the
question that matters now that the ladder has shown the AE's advantage is narrow
and token-shaped: it beats an encoder-free baseline on surface / POS / NER and
loses on DBpedia topic. Whether that is a trade the encoder makes as clustering
strengthens, or a difference present from initialisation, is only visible across
epochs.

Requires the trajectory to exist: train with save_every=1 and keep_checkpoints
above n_epochs. The default keep_checkpoints=3 deletes it.

Emits:
  <out>.json                     every rung x every epoch, NMI and F1
  figures/concept_trajectory/    one line chart per rung family, token rungs and
                                 sequence rungs on separate axes because they
                                 move in opposite directions

Usage:
    python -m geoae.interp.concept_trajectory \
      --checkpoints checkpoints/llama3.2-3B/layer27/k2000_sq3072_bnhinge_bigbatch_mse \
      --out trajectory_bigbatch.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import normalized_mutual_info_score as nmi_score

from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.interp.concept_probe import LADDER, load_rung, assign, best_f1

TOKEN_RUNGS = {"surface", "pos_coarse", "pos_fine", "ner_coarse", "ner_fine"}
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
       "#56B4E9", "#7B3294", "#1B7837", "#B2182B", "#35978F"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--baseline",
                    default="e2e/checkpoints/general/llama3.2-3B/layer27/balanced_kmeans_k2000.npz",
                    help="encoder-free reference drawn as a dashed line on every chart")
    ap.add_argument("--n_token", type=int, default=60000)
    ap.add_argument("--every", type=int, default=1, help="score every Nth checkpoint")
    ap.add_argument("--figdir", default="figures/concept_trajectory")
    ap.add_argument("--out", default="trajectory.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(Path(args.checkpoints).glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))[:: args.every]
    if not ckpts:
        raise SystemExit(f"no step_*.pt in {args.checkpoints} — keep_checkpoints too small?")
    print(f"[traj] {len(ckpts)} checkpoints")

    rungs = [r for r in LADDER if (Path(args.cache) / r[1]).exists()]
    data = {}
    for rung, fname, key, grain, desc in rungs:
        H, y = load_rung(args.cache, fname, key, args.n_token if grain == "token" else 0)
        data[rung] = dict(grain=grain, desc=desc, n=len(H), epochs=[], nmi=[], f1=[])
        data[rung]["_H"], data[rung]["_y"] = H, y
    print(f"[traj] {len(rungs)} rungs cached in memory")

    base = {}
    if args.baseline and Path(args.baseline).exists():
        C, m, s, K, *_ = load_baseline_kmeans(Path(args.baseline), dev)
        for rung in data:
            lab = assign(data[rung]["_H"], ("km", (C, m, s)), dev)
            base[rung] = dict(nmi=float(nmi_score(data[rung]["_y"], lab)),
                              f1=float(best_f1(lab, data[rung]["_y"])))
        print("[traj] baseline scored")

    for cp in ckpts:
        ae, mean, std, ck = load_ae_checkpoint(cp, dev)
        ae.eval()
        ep = int(ck["epoch"])
        for rung in data:
            lab = assign(data[rung]["_H"], ("ae", (ae, mean, std)), dev)
            data[rung]["epochs"].append(ep)
            data[rung]["nmi"].append(round(float(nmi_score(data[rung]["_y"], lab)), 4))
            data[rung]["f1"].append(round(best_f1(lab, data[rung]["_y"]), 4))
        print(f"  epoch {ep:>3}  " + "  ".join(
            f"{r}={data[r]['nmi'][-1]:.3f}" for r in ("pos_fine", "topic14") if r in data), flush=True)

    for r in data:
        data[r].pop("_H", None); data[r].pop("_y", None)
    json.dump({"rungs": data, "baseline": base}, open(args.out, "w"), indent=1)
    print(f"[traj] wrote {args.out}")

    # ---- charts: token and sequence on separate axes, they move oppositely ---
    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    for metric in ("nmi", "f1"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, grain, title in ((axes[0], "token", "TOKEN-level concepts"),
                                 (axes[1], "sequence", "SEQUENCE-level concepts")):
            i = 0
            for r, d in data.items():
                if d["grain"] != grain:
                    continue
                ax.plot(d["epochs"], d[metric], lw=1.6, color=PAL[i % len(PAL)], label=r)
                if r in base:
                    ax.axhline(base[r][metric], color=PAL[i % len(PAL)], ls=":", lw=1, alpha=.65)
                i += 1
            ax.set_title(f"{title}   ({metric.upper()})", fontsize=10)
            ax.set_xlabel("epoch"); ax.legend(frameon=False, fontsize=7.5, ncol=2)
        fig.text(0.005, 1.0, "dotted = balanced k-means with NO encoder, the bar the encoder must clear",
                 fontsize=8, color="#5A646D", va="top")
        fig.savefig(fig_dir / f"trajectory_{metric}.png"); plt.close(fig)
    print(f"[traj] figures -> {fig_dir}/")


if __name__ == "__main__":
    main()
