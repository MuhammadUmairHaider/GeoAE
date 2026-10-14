"""
PCA plots (n=2) with cross-method cluster correspondence.

Same sample of points is passed through both the baseline k-means and each AE.
Cluster assignments are matched via the Hungarian algorithm on the co-occurrence
matrix — baseline cluster i gets the same colour as the AE cluster it overlaps
most with.  This lets you visually track what happened to each cluster when the
representation changes.

Usage:
    python -m geoae.interp.pca_plot \\
        --baseline    checkpoints/baseline_layer27_k128.npz \\
        --checkpoint  checkpoints/layer27_k128_d3072_linear/best_val.pt \\
        --layer 27 --n_sample 5000 --top_k 10 \\
        --out pca_clusters.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment



# ---------------------------------------------------------------------------
# Sample points and get assignments from both methods
# ---------------------------------------------------------------------------

def sample_activations(act_dir: Path, layer: int, n_sample: int,
                        seed: int, norm_mean: np.ndarray,
                        norm_std: np.ndarray) -> np.ndarray:
    mmap = np.load(str(act_dir / f"layer_{layer}.npy"), mmap_mode="r")
    rng  = np.random.RandomState(seed)
    idx  = np.sort(rng.choice(mmap.shape[0], min(n_sample, mmap.shape[0]), replace=False))
    raw  = mmap[idx].astype(np.float32)
    return (raw - norm_mean) / norm_std          # (N, D) normalised


def baseline_assignments(x_norm: np.ndarray,
                          baseline_npz: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (labels, centroids_in_norm_space, K)."""
    from sklearn.metrics import pairwise_distances_argmin
    data      = np.load(str(baseline_npz))
    centroids = data["centroids"]               # (K, D) in norm space
    labels    = pairwise_distances_argmin(x_norm, centroids, metric="euclidean")
    return labels, centroids, len(centroids)


def ae_assignments(x_norm: np.ndarray, ckpt_path: Path,
                   device) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """
    Encode x_norm through the AE.
    Returns (latents, labels, centroids_in_latent_space, K, label_str).
    """
    import torch
    from geoae.checkpoint import load_ae_checkpoint

    ae, _, _, ckpt = load_ae_checkpoint(ckpt_path, device)
    mc = ckpt["config"]["model"]
    nl = mc.get("nonlinearity", "linear")

    batch  = 1024
    z_list, q_list = [], []
    with torch.no_grad():
        for s in range(0, len(x_norm), batch):
            x_t  = torch.from_numpy(x_norm[s:s+batch]).to(device)
            out  = ae(x_t)
            z_list.append(out.z.cpu().numpy())
            q_list.append(out.Q.cpu().numpy())

    z      = np.concatenate(z_list)
    Q      = np.concatenate(q_list)
    labels = Q.argmax(axis=1)
    K      = mc["n_clusters"]
    label  = (f"AE {mc['hidden_size']}→{mc['latent_dim']}→{mc['hidden_size']}\n"
              f"({nl}, K={K})")
    return z, labels, ae.centroids.cpu().numpy(), K, label


# ---------------------------------------------------------------------------
# Cluster matching via Hungarian algorithm
# ---------------------------------------------------------------------------

def match_clusters(base_labels: np.ndarray, ae_labels: np.ndarray,
                   K_base: int, K_ae: int) -> dict[int, int]:
    """
    For each baseline cluster i, find the AE cluster j that maximises overlap.
    Returns mapping  baseline_cluster_id → ae_cluster_id.
    Uses the Hungarian algorithm so the matching is a bijection.
    """
    overlap = np.zeros((K_base, K_ae), dtype=np.int64)
    for b, a in zip(base_labels, ae_labels):
        overlap[b, a] += 1
    row_ind, col_ind = linear_sum_assignment(-overlap)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def top_k_clusters(labels: np.ndarray, K: int, top_k: int) -> list[int]:
    """Return the IDs of the top_k largest clusters, sorted by descending size."""
    counts = np.bincount(labels, minlength=K)
    return list(np.argsort(counts)[-top_k:][::-1])


# ---------------------------------------------------------------------------
# Plot one panel
# ---------------------------------------------------------------------------

def plot_panel(ax, z: np.ndarray, labels: np.ndarray,
               show_clusters: list[int],       # cluster IDs to show, in colour order
               colour_offset: dict[int, int],  # cluster_id → colour index (0..top_k-1)
               cmap, title: str,
               pca: PCA | None = None) -> PCA:
    keep = np.isin(labels, show_clusters)
    z_f  = z[keep]
    l_f  = labels[keep]

    if pca is None:
        pca = PCA(n_components=2, random_state=0)
        z2  = pca.fit_transform(z_f)
    else:
        z2  = pca.transform(z_f)

    var = pca.explained_variance_ratio_

    for cid in show_clusters:
        mask  = l_f == cid
        cidx  = colour_offset[cid]
        color = cmap(cidx)
        ax.scatter(z2[mask, 0], z2[mask, 1],
                   c=[color], s=10, alpha=0.55, linewidths=0,
                   rasterized=True, label=f"C{cidx}")

    # Centroids
    cents = np.stack([z2[l_f == cid].mean(axis=0) for cid in show_clusters])
    for i, cid in enumerate(show_clusters):
        ax.scatter(cents[i, 0], cents[i, 1],
                   c=[cmap(colour_offset[cid])],
                   s=120, marker="*", edgecolors="black",
                   linewidths=0.6, zorder=6)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=7)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, markerscale=1.2, framealpha=0.6,
              loc="upper right", ncol=2, title="cluster", title_fontsize=6)
    ax.set_aspect("equal", adjustable="datalim")
    return pca


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",    required=True,
                        help="Baseline k-means .npz from cluster_baseline.py fit")
    parser.add_argument("--checkpoint",  default=None,
                        help="Single AE checkpoint (alias for --checkpoints)")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="One or more AE checkpoint .pt files")
    parser.add_argument("--layer",      type=int, default=27)
    parser.add_argument("--n_sample",   type=int, default=5_000)
    parser.add_argument("--top_k",      type=int, default=10,
                        help="Number of baseline clusters to show (matched in AE panel)")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--activations_dir", default="activations")
    parser.add_argument("--out",        default="pca_clusters.png")
    args = parser.parse_args()

    import torch
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_dir = Path(args.activations_dir)

    # Merge --checkpoint and --checkpoints
    ckpt_list = list(args.checkpoints)
    if args.checkpoint:
        ckpt_list.insert(0, args.checkpoint)
    if not ckpt_list:
        print("ERROR: provide at least one --checkpoint or --checkpoints path")
        return

    # Load norm params from baseline
    bdata     = np.load(str(args.baseline))
    norm_mean = bdata["norm_mean"]
    norm_std  = bdata["norm_std"]

    # Sample the SAME points for all methods
    print(f"[pca] Sampling {args.n_sample:,} points …")
    x_norm = sample_activations(act_dir, args.layer, args.n_sample,
                                args.seed, norm_mean, norm_std)

    # Baseline assignments
    print("[pca] Baseline k-means assignments …")
    base_labels, _, K_base = baseline_assignments(x_norm, Path(args.baseline))
    top_base    = top_k_clusters(base_labels, K_base, args.top_k)
    base_colour = {cid: i for i, cid in enumerate(top_base)}
    cmap        = plt.get_cmap("tab10")

    # One AE per checkpoint
    ae_data = []   # list of (ae_z, ae_labels, top_ae, ae_colour, ae_title)
    for ckpt in ckpt_list:
        print(f"[pca] AE assignments: {ckpt} …")
        ae_z, ae_labels, _, K_ae, ae_title = ae_assignments(x_norm, Path(ckpt), device)
        matching  = match_clusters(base_labels, ae_labels, K_base, K_ae)
        top_ae    = [matching[b] for b in top_base]
        ae_colour = {cid: i for i, cid in enumerate(top_ae)}

        # Print matching table
        print(f"\n  {ae_title.replace(chr(10),' ')}")
        print(f"  {'Colour':>6}  {'Base C':>8}  {'AE C':>8}  {'Overlap':>9}")
        print("  " + "-"*38)
        base_counts = np.bincount(base_labels, minlength=K_base)
        for i, (b, a) in enumerate(zip(top_base, top_ae)):
            ov  = int(((base_labels == b) & (ae_labels == a)).sum())
            pct = ov / max(base_counts[b], 1) * 100
            print(f"  {i:>6}  {b:>8}  {a:>8}  {ov:>6} ({pct:.0f}%)")

        ae_data.append((ae_z, ae_labels, top_ae, ae_colour, ae_title))

    # Layout: 1 baseline panel + N AE panels
    n_panels = 1 + len(ae_data)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.5 * n_panels, 5.5),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    print("\n[pca] Plotting baseline panel …")
    plot_panel(axes[0], x_norm, base_labels,
               show_clusters=top_base,
               colour_offset=base_colour,
               cmap=cmap,
               title=f"Raw k-means  (D=3072, K={K_base})\ntop {args.top_k} clusters")

    for i, (ae_z, ae_labels, top_ae, ae_colour, ae_title) in enumerate(ae_data):
        print(f"[pca] Plotting AE panel {i+1} …")
        plot_panel(axes[i + 1], ae_z, ae_labels,
                   show_clusters=top_ae,
                   colour_offset=ae_colour,
                   cmap=cmap,
                   title=ae_title + f"\ntop {args.top_k} matched clusters")

    fig.suptitle(
        f"Same {args.n_sample:,} points — same colour = Hungarian-matched cluster pair"
        f"  ·  Layer {args.layer}  ·  ★ = centroid",
        fontsize=10,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"[pca] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
