"""
PCA plots for DBpedia-14: raw activations vs AE latent space.

Both panels use the same test-split points coloured by ground-truth class,
so you can directly compare how well the 14 categories are separated before
and after the AE encoding.

Usage:
    python -m geoae.dbpedia.pca_plot \\
        --checkpoint dbpedia/checkpoints/unprompted_gelu/best_val.pt \\
        --mode unprompted \\
        --out dbpedia/pca_dbpedia.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
]

BASE = Path("dbpedia")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test(act_dir: Path, layer: int):
    """Returns (acts_norm, labels) for test split."""
    acts   = np.load(str(act_dir / f"layer_{layer}_test.npy")).astype(np.float32)
    labels = np.load(str(act_dir / "labels_test.npy")).astype(np.int32)
    mean   = acts.mean(axis=0)
    std    = acts.std(axis=0) + 1e-8
    return (acts - mean) / std, labels, mean, std


def encode_test(checkpoint: Path, acts_norm: np.ndarray, mean: np.ndarray,
                std: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, str]:
    """Encode normalised test acts through the AE; return (z, cluster_pred, label_str)."""
    import torch
    from geoae.checkpoint import load_ae_checkpoint

    ae, _, _, ckpt = load_ae_checkpoint(checkpoint, device)
    mc = ckpt["config"]["model"]
    nl = mc.get("nonlinearity", "linear")

    # Re-normalise using AE's own norm stats (may differ from test-split stats)
    ae_mean = ckpt["norm_mean"]
    ae_std  = ckpt["norm_std"]

    # Raw acts = acts_norm * std + mean  →  re-normalise with AE stats
    raw = acts_norm * std + mean
    x_ae = (raw - ae_mean) / ae_std

    batch, z_list, q_list = 512, [], []
    with torch.no_grad():
        for s in range(0, len(x_ae), batch):
            x_t = torch.from_numpy(x_ae[s:s+batch]).to(device)
            out = ae(x_t)
            z_list.append(out.z.cpu().numpy())
            q_list.append(out.Q.cpu().numpy())

    z      = np.concatenate(z_list)                 # (N, latent_dim)
    labels = np.concatenate(q_list).argmax(axis=1)  # (N,) cluster assignments
    title  = f"{nl}  L={mc['latent_dim']}  K={mc['n_clusters']}"
    return z, labels, title


# ---------------------------------------------------------------------------
# Plot panel
# ---------------------------------------------------------------------------

def plot_panel(ax, coords: np.ndarray, class_labels: np.ndarray,
               cmap, title: str, n_classes: int = 14,
               pca: PCA | None = None) -> PCA:
    if pca is None:
        pca = PCA(n_components=2, random_state=0)
        xy  = pca.fit_transform(coords)
    else:
        xy  = pca.transform(coords)

    var = pca.explained_variance_ratio_

    for c in range(n_classes):
        mask = class_labels == c
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   color=cmap(c / n_classes), s=6, alpha=0.45,
                   linewidths=0, rasterized=True, label=CLASSES[c])

    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}%)", fontsize=7)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal", adjustable="datalim")
    return pca


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Single checkpoint (original 3-panel mode)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Multiple checkpoints for side-by-side comparison")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Panel labels for each checkpoint (must match --checkpoints length)")
    parser.add_argument("--mode",       default="unprompted",
                        choices=["unprompted", "prompted"])
    parser.add_argument("--pooling",    default="last",
                        choices=["last", "mean"],
                        help="Token reduction tree to plot (default: last)")
    parser.add_argument("--model",     default="llama3.2-3B",
                        help="Model name subdirectory (default: llama3.2-3B)")
    parser.add_argument("--layer",      type=int, default=27)
    parser.add_argument("--out",        default=None,
                        help="Output path (default: auto-generated under results/)")
    args = parser.parse_args()

    import torch
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_dir = BASE / "activations" / args.model / args.pooling / args.mode

    # Default output path under results
    if args.out is None:
        res_dir = BASE / "results" / args.model / f"layer{args.layer}"
        res_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(res_dir / f"pca_{args.mode}_{args.pooling}.png")

    print("[pca] Loading test activations …")
    acts_norm, true_labels, mean, std = load_test(act_dir, args.layer)
    N = len(true_labels)
    print(f"[pca] {N:,} test examples, {len(np.unique(true_labels))} classes")

    cmap = plt.get_cmap("tab20")

    # ------------------------------------------------------------------
    # Multi-checkpoint comparison mode
    # ------------------------------------------------------------------
    if args.checkpoints:
        ckpts = [Path(c) for c in args.checkpoints]
        labels = args.labels or [c.parent.name for c in ckpts]
        n_panels = 1 + len(ckpts)  # raw + one per checkpoint

        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(6 * n_panels, 5.5),
                                 constrained_layout=True)

        print("[pca] Plotting raw activation panel …")
        plot_panel(axes[0], acts_norm, true_labels, cmap,
                   title=f"Raw activations  (D=3072)\n{args.mode}")

        for i, (ckpt, label) in enumerate(zip(ckpts, labels)):
            print(f"[pca] Encoding through {label} …")
            z, cluster_pred, ae_title = encode_test(ckpt, acts_norm, mean, std, device)
            plot_panel(axes[i + 1], z, true_labels, cmap,
                       title=f"{label}\n{ae_title}")

    # ------------------------------------------------------------------
    # Original single-checkpoint 3-panel mode
    # ------------------------------------------------------------------
    else:
        if not args.checkpoint:
            parser.error("Provide --checkpoint or --checkpoints")

        print("[pca] Encoding through AE …")
        z, cluster_pred, ae_title = encode_test(
            Path(args.checkpoint), acts_norm, mean, std, device
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

        print("[pca] Plotting raw activation panel …")
        plot_panel(axes[0], acts_norm, true_labels, cmap,
                   title=f"Raw activations  (D=3072)\n{args.mode}")

        print("[pca] Plotting AE latent panel …")
        plot_panel(axes[1], z, true_labels, cmap,
                   title=f"AE latent space\n{ae_title}  ·  {args.mode}")

        print("[pca] Plotting AE cluster assignment panel …")
        K     = int(cluster_pred.max()) + 1
        pca_z = PCA(n_components=2, random_state=0).fit(z)
        xy_z  = pca_z.transform(z)
        var_z  = pca_z.explained_variance_ratio_
        cmap_k = plt.get_cmap("tab20")
        for k in range(K):
            mask = cluster_pred == k
            if mask.sum() == 0:
                continue
            axes[2].scatter(xy_z[mask, 0], xy_z[mask, 1],
                            color=cmap_k(k / K), s=6, alpha=0.45,
                            linewidths=0, rasterized=True)
        axes[2].set_title(f"AE cluster assignments  (K={K})\n{ae_title}  ·  {args.mode}",
                          fontsize=9, pad=5)
        axes[2].set_xlabel(f"PC1  ({var_z[0]*100:.1f}%)", fontsize=7)
        axes[2].set_ylabel(f"PC2  ({var_z[1]*100:.1f}%)", fontsize=7)
        axes[2].tick_params(labelsize=6)
        axes[2].set_aspect("equal", adjustable="datalim")

    # Shared legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(c / 14), markersize=7, label=CLASSES[c])
        for c in range(14)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        f"DBpedia-14  ·  Layer {args.layer}  ·  {N:,} test points  "
        f"·  coloured by true class",
        fontsize=11,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"[pca] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
