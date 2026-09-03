"""
Geometric views of what a clustering does to concept structure: PCA and t-SNE of
the latent, coloured by ground-truth concept labels, for several models side by
side.

Why both projections, and why side by side. PCA is linear and faithful to global
variance — if two concept classes are linearly separable in the latent, PCA shows
it, and distances between the blobs mean something. t-SNE is non-linear and only
preserves neighbourhoods — it will happily invent well-separated islands out of a
continuum, so it answers "is there local structure" and NOT "are these far apart".
Reading them together is the point: structure visible in PCA is real and linear;
structure visible only in t-SNE is local and possibly an artefact of perplexity.

Every panel in a row shares the SAME points and the SAME colouring; only the
representation changes. The baselines have no encoder, so their "latent" is the
normalised raw activation — which is exactly the comparison we want, since
balanced k-means on raw activations has matched the AE on most concept rungs.

Figures written to figures/<subdir>/:
  tsne/<rung>.png       t-SNE, neighbourhoods only
  pca/<rung>.png        PCA, linear, distances meaningful
  umap_<rung>.png       UMAP — keeps more global structure than t-SNE, so
                        between-cluster distance is *somewhat* interpretable
  umap/<rung>.png       UMAP — more global structure survives than in t-SNE
  centroids/<rung>.png  cluster centroids in PCA, sized by usage, coloured by
                        their dominant concept — the partition itself, not the data
  variance/<rung>.png   scree + cumulative variance, all models on one axis

One subfolder per PROJECTION, not per rung, so a single method reads straight
down the list of concepts: does the structure that appears for POS also appear
for topic?

Usage:
    python -m geoae.interp.concept_geometry --rungs pos_coarse,topic14,domain
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.interp.concept_probe import LADDER, load_rung, assign

MODELS = [
    ("AE mse λ2",    "ae", "checkpoints/llama3.2-3B/layer27/k2000_sq3072_strong_mse/step_0113900.pt"),
    ("AE BN+hinge",  "ae", "checkpoints/llama3.2-3B/layer27/k2000_sq3072_bnhinge_mse/step_0113900.pt"),
    ("AE e2e-KL",    "ae", "e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt"),
    ("balanced km",  "km", "e2e/checkpoints/general/llama3.2-3B/layer27/balanced_kmeans_k2000.npz"),
    ("plain km",     "km", "e2e/checkpoints/general/llama3.2-3B/layer27/baseline_kmeans_k2000_refit.npz"),
]
# Colour-blind-safe qualitative set (Okabe-Ito, extended). Concept classes are
# nominal, so a categorical palette — never a sequential ramp.
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9",
       "#F0E442", "#7B3294", "#8C8C8C", "#1B7837", "#762A83", "#B2182B",
       "#2166AC", "#F4A582", "#4D9221", "#C51B7D", "#35978F", "#BF812D",
       "#5AAE61", "#9970AB"]


def style(dark=False):
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8A939B", "axes.labelcolor": "#20272D",
        "text.color": "#20272D", "xtick.color": "#5A646D", "ytick.color": "#5A646D",
        "axes.titlesize": 10, "axes.titleweight": "semibold",
        "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def repr_of(H, kind, obj, dev, chunk=16384):
    """The space each model actually clusters in: AE latent, or normalised raw."""
    out = []
    with torch.no_grad():
        for i in range(0, len(H), chunk):
            h = torch.from_numpy(H[i:i + chunk]).to(dev).float()
            if kind == "ae":
                ae, mean, std = obj
                out.append(ae.encoder((h - mean) / std).cpu().numpy())
            else:
                _C, mean, std = obj
                out.append(((h - mean) / std).cpu().numpy())
    return np.concatenate(out)


def sub(outdir: Path, name: str) -> Path:
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def scatter_row(fig, axes, embs, y, classes, title_fmt, note):
    handles = None
    for ax, (name, E) in zip(axes, embs):
        for ci, cls in enumerate(classes):
            m = y == cls
            ax.scatter(E[m, 0], E[m, 1], s=3.2, alpha=.55, linewidths=0,
                       color=PAL[ci % len(PAL)], rasterized=True)
        ax.set_title(title_fmt.format(name))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", ms=5, color=PAL[i % len(PAL)],
                      label=str(c)[:22]) for i, c in enumerate(classes)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(classes), 7),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.005, 0.985, note, fontsize=8, color="#5A646D", va="top")


def do_rung(rung, cache, models, dev, outdir, n_points, perplexity, max_classes, want):
    src = {r[0]: r for r in LADDER}[rung]
    _n, fname, key, grain, desc = src
    H, y = load_rung(cache, fname, key, 0)
    keep = [c for c, _ in Counter(y.tolist()).most_common(max_classes)]
    m = np.isin(y, keep)
    H, y = H[m], y[m]
    if len(H) > n_points:
        idx = np.random.RandomState(0).choice(len(H), n_points, replace=False)
        H, y = H[idx], y[idx]
    classes = sorted(set(y.tolist()), key=lambda c: -Counter(y.tolist())[c])
    print(f"[geo] {rung}: {len(H):,} points, {len(classes)} classes ({grain})")

    reps = [(nm, repr_of(H, k, o, dev)) for nm, (k, o) in models.items()]

    # kNN label agreement, computed in the REPRESENTATION (not the 2-D embedding).
    # t-SNE panels cannot be compared to each other by eye — each has its own
    # arbitrary rotation, scale and layout — so every panel is labelled with a
    # number that IS comparable. PCA-50 first so the score is measured in the same
    # way for a 3072-d baseline and a 6144-d latent.
    knn = {}
    for nm, R in reps:
        Rp = PCA(n_components=min(50, R.shape[1]), random_state=0).fit_transform(R)
        tr, te = train_test_split(np.arange(len(Rp)), test_size=0.4,
                                  random_state=0, stratify=y)
        km_ = KNeighborsClassifier(n_neighbors=10).fit(Rp[tr], y[tr])
        knn[nm] = float(km_.score(Rp[te], y[te]))
    order = sorted(knn, key=lambda k: -knn[k])
    print("[geo]   kNN-10 label agreement: " +
          "  ".join(f"{k} {knn[k]:.3f}" for k in order))
    reps = [(f"{nm}\nkNN {knn[nm]:.3f}", R) for nm, R in reps]

    # ---- PCA -------------------------------------------------------------
    pcas, varexp = [], {}
    for nm, R in reps:
        p = PCA(n_components=min(10, R.shape[1]), random_state=0).fit(R)
        pcas.append((nm, p.transform(R)[:, :2]))
        varexp[nm] = p.explained_variance_ratio_
    if "pca" in want:
        fig, axes = plt.subplots(1, len(reps), figsize=(3.05 * len(reps), 3.5))
        scatter_row(fig, axes, pcas, y, classes,
                    "{}", f"PCA (linear, distances meaningful) — {rung}: {desc}, {grain}-level")
        fig.savefig(sub(outdir, "pca") / f"{rung}.png"); plt.close(fig)

    # ---- t-SNE -----------------------------------------------------------
    tsnes = [] if "tsne" in want else None
    for nm, R in (reps if tsnes is not None else []):
        pre = PCA(n_components=min(50, R.shape[1]), random_state=0).fit_transform(R)
        tsnes.append((nm, TSNE(n_components=2, perplexity=perplexity, init="pca",
                               random_state=0, max_iter=750).fit_transform(pre)))
    if tsnes is not None:
        fig, axes = plt.subplots(1, len(reps), figsize=(3.05 * len(reps), 3.5))
        scatter_row(fig, axes, tsnes, y, classes, "{}",
                    f"t-SNE (neighbourhoods only — blob DISTANCE is not meaningful) — {rung}, perplexity {perplexity}")
        fig.savefig(sub(outdir, "tsne") / f"{rung}.png"); plt.close(fig)

    # ---- UMAP ------------------------------------------------------------
    # t-SNE destroys global geometry; UMAP keeps more of it. Running both and
    # agreeing is much stronger evidence than either alone.
    if "umap" in want:
     try:
        import umap
        ums = []
        for nm, R in reps:
            pre = PCA(n_components=min(50, R.shape[1]), random_state=0).fit_transform(R)
            ums.append((nm, umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=0,
                                      verbose=False).fit_transform(pre)))
        fig, axes = plt.subplots(1, len(reps), figsize=(3.05 * len(reps), 3.5))
        scatter_row(fig, axes, ums, y, classes, "{}",
                    f"UMAP (local + some global structure) — {rung}, n_neighbors 25")
        fig.savefig(sub(outdir, "umap") / f"{rung}.png"); plt.close(fig)
     except ImportError:
        pass

    # ---- variance spectrum ----------------------------------------------
    if "variance" not in want:
        return reps, y, classes
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for nm, _ in reps:
        v = varexp[nm]
        a1.plot(range(1, len(v) + 1), v, marker="o", ms=3, lw=1.3, label=nm)
        a2.plot(range(1, len(v) + 1), np.cumsum(v), marker="o", ms=3, lw=1.3, label=nm)
    a1.set_title("variance per component"); a1.set_xlabel("component")
    a2.set_title("cumulative"); a2.set_xlabel("component"); a2.set_ylim(0, 1)
    a2.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"How concentrated is each space?  —  {rung}", fontsize=10, y=1.02)
    fig.savefig(sub(outdir, "variance") / f"{rung}.png"); plt.close(fig)
    return reps, y, classes


def do_centroids(rung, cache, models, dev, outdir, max_classes):
    """The PARTITION itself: centroids in PCA, sized by usage, coloured by the
    concept that dominates them. Shows whether clusters specialise at all."""
    src = {r[0]: r for r in LADDER}[rung]
    H, y = load_rung(cache, src[1], src[2], 60000)
    keep = [c for c, _ in Counter(y.tolist()).most_common(max_classes)]
    m = np.isin(y, keep); H, y = H[m], y[m]
    classes = sorted(set(y.tolist()))
    fig, axes = plt.subplots(1, len(models), figsize=(3.05 * len(models), 3.4))
    for ax, (nm, (kind, obj)) in zip(np.atleast_1d(axes), models.items()):
        lab = assign(H, (kind, obj), dev)
        C = (obj[0].centroids if kind == "ae" else obj[0]).cpu().numpy()
        used = sorted(set(lab.tolist()))
        dom, size, pure = [], [], []
        for k in used:
            cc = Counter(y[lab == k].tolist())
            tot = sum(cc.values())
            best, n = cc.most_common(1)[0]
            dom.append(classes.index(best)); size.append(tot); pure.append(n / tot)
        E = PCA(n_components=2, random_state=0).fit_transform(C[used])
        s = 4 + 44 * (np.array(size) / max(size))
        ax.scatter(E[:, 0], E[:, 1], s=s, c=[PAL[d % len(PAL)] for d in dom],
                   alpha=.7, linewidths=0, rasterized=True)
        ax.set_title(f"{nm}\n{len(used)} live · purity {np.mean(pure):.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("left", "bottom"): ax.spines[sp].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=PAL[i % len(PAL)],
                              label=str(c)[:22]) for i, c in enumerate(classes)],
               loc="lower center", ncol=min(len(classes), 7), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.005, .985, f"Cluster CENTROIDS in PCA — {rung}. Dot size = tokens won, "
             f"colour = dominant class, purity = mean share of that class.",
             fontsize=8, color="#5A646D", va="top")
    fig.savefig(sub(outdir, "centroids") / f"{rung}.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--rungs", default="pos_coarse,ner_coarse,topic14,domain,language")
    ap.add_argument("--outdir", default="figures/concept_geometry")
    ap.add_argument("--n_points", type=int, default=6000)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--max_classes", type=int, default=8)
    ap.add_argument("--projections", default="tsne,pca,umap,variance,centroids")
    ap.add_argument("--models", default="auto",
                    help="'auto', or name=path.pt,name2=path2.pt — AE checkpoints, "
                         "in the order you want the panels drawn")
    ap.add_argument("--baselines", default="auto",
                    help="'auto', or name=path.npz,... — encoder-free controls")
    args = ap.parse_args()

    style()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    spec = []
    if args.models == "auto" and args.baselines == "auto":
        spec = list(MODELS)
    else:
        if args.models != "auto":
            spec += [(s.split("=", 1)[0], "ae", s.split("=", 1)[1])
                     for s in args.models.split(",") if s]
        if args.baselines == "auto":
            spec += [(n, k, pth) for n, k, pth in MODELS if k == "km"]
        else:
            spec += [(s.split("=", 1)[0], "km", s.split("=", 1)[1])
                     for s in args.baselines.split(",") if s]
    models = {}
    for nm, kind, path in spec:
        if not Path(path).exists():
            print(f"[geo] skip {nm} (missing)"); continue
        if kind == "ae":
            ae, mean, std, _ = load_ae_checkpoint(path, dev); ae.eval()
            models[nm] = ("ae", (ae, mean, std))
        else:
            C, mean, std, K, *_ = load_baseline_kmeans(Path(path), dev)
            models[nm] = ("km", (C, mean, std))
    print(f"[geo] {len(models)} models -> {out}/")

    for rung in args.rungs.split(","):
        want = set(args.projections.split(","))
        do_rung(rung, args.cache, models, dev, out, args.n_points,
                args.perplexity, args.max_classes, want)
        if "centroids" in want:
            do_centroids(rung, args.cache, models, dev, out, args.max_classes)
        print(f"[geo]   wrote {sorted(want)} for {rung}")
    print(f"[geo] done -> {out}/")


if __name__ == "__main__":
    main()
