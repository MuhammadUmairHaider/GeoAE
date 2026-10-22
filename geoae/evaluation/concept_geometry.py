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

Token-bypass AEs: pass `--models name=ckpt`; the rung cache must carry token ids
(token_id, or last_token_id beside H_last — experiments/paper/scripts/run_delta_prep.sh's cache_ids/),
and the latent is ae.encode(x, tok). A baseline whose .npz carries `token_bias`
(the token-mean control) is drawn as normalised x - b[tok]. A model that needs ids
is skipped, with a printed note, on any rung whose cache has none.

Usage:
    python -m geoae.evaluation.concept_geometry --rungs pos_coarse,topic14,domain
    scripts/py -m geoae.evaluation.concept_geometry --cache cache_ids \\
        --models bypass=<ckpt> --baselines base=<km.npz>,base_tokmean=<kmt.npz> \\
        --order base,base_tokmean,bypass --rungs topic14,pos_coarse --max_classes 14
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
from geoae.analysis.closest_tokens import load_baseline_kmeans
from geoae.analysis.concept_probe import LADDER
from geoae.analysis.concept_probe import load_rung
from geoae.analysis.concept_probe import load_rung_ids
from geoae.analysis.concept_probe import assign

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


def needs_ids(kind, obj):
    """True when this model's representation depends on the current token id."""
    if kind == "ae":
        return bool(obj[0].has_token_bias)
    return len(obj) > 4 and obj[4] is not None      # k-means baseline with a token_bias table


def repr_of(H, kind, obj, dev, chunk=16384, ids=None):
    """The space each model actually clusters in: AE latent, or normalised raw.

    Token-bypass AE: ae.encode(x, tok). Baseline carrying `token_bias` (the
    token-mean control): normalised x minus b[tok]. Returns None when the model
    needs token ids and the rung has none (caller skips it, never drops the bias).
    """
    if needs_ids(kind, obj) and ids is None:
        return None
    out = []
    with torch.no_grad():
        for i in range(0, len(H), chunk):
            h = torch.from_numpy(H[i:i + chunk]).to(dev).float()
            tok = None if ids is None else torch.from_numpy(ids[i:i + chunk]).to(dev)
            if kind == "ae":
                ae, mean, std = obj
                out.append(ae.encode((h - mean) / std,
                                     tok if ae.has_token_bias else None).cpu().numpy())
            else:
                _C, mean, std, *rest = obj
                v = (h - mean) / std
                if len(rest) > 1 and rest[1] is not None:
                    v = v - rest[1](tok)
                if rest and rest[0] is not None:
                    v = v - (v @ rest[0]) @ rest[0].T
                out.append(v.cpu().numpy())
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
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.005, 0.995, note, fontsize=8, color="#5A646D", va="top")
    # Reserve the strips the note and the legend live in. Without this the axes keep
    # matplotlib's default top of 0.88, the two-line panel titles are drawn straight
    # through the note, and the legend is anchored below the canvas edge.
    fig.subplots_adjust(top=0.79, bottom=0.20, left=0.015, right=0.985, wspace=0.08)


# Rungs whose caches store integer class ids; order = the dataset's label index.
LABEL_NAMES = {
    "topic14": ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
                "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
                "Plant", "Album", "Film", "WrittenWork"],          # fancyzhx/dbpedia_14
    "topic4": ["World", "Sports", "Business", "Sci/Tech"],        # fancyzhx/ag_news
    "subjectivity": ["objective", "subjective"],                  # SetFit/subj
}


def do_rung(rung, cache, models, dev, outdir, n_points, perplexity, max_classes, want):
    src = {r[0]: r for r in LADDER}[rung]
    _n, fname, key, grain, desc = src
    H, y = load_rung(cache, fname, key, 0)
    ids = load_rung_ids(cache, fname)      # None on caches without token ids
    if ids is not None and len(ids) != len(H):
        ids = None
    keep = [c for c, _ in Counter(y.tolist()).most_common(max_classes)]
    m = np.isin(y, keep)
    H, y = H[m], y[m]
    ids = None if ids is None else ids[m]
    if len(H) > n_points:
        idx = np.random.RandomState(0).choice(len(H), n_points, replace=False)
        H, y = H[idx], y[idx]
        ids = None if ids is None else ids[idx]
    if rung in LABEL_NAMES:                 # integer class ids -> readable legend names
        names = LABEL_NAMES[rung]
        y = np.array([names[int(v)] if str(v).lstrip("-").isdigit() and 0 <= int(v) < len(names)
                      else v for v in y.tolist()], dtype=object)
    classes = sorted(set(y.tolist()), key=lambda c: -Counter(y.tolist())[c])
    print(f"[geo] {rung}: {len(H):,} points, {len(classes)} classes ({grain})")

    reps = []
    for nm, (k, o) in models.items():
        R = repr_of(H, k, o, dev, ids=ids)
        if R is None:
            print(f"[geo]   note: {rung}: no token ids in {cache}/{fname} — skipping "
                  f"'{nm}' on this rung (a token-bias model is never encoded without them)")
            continue
        reps.append((nm, R))
    if not reps:
        print(f"[geo]   {rung}: no model can be drawn on this rung; nothing written")
        return [], y, classes

    # kNN label agreement, computed in the REPRESENTATION (not the 2-D embedding).
    # t-SNE panels cannot be compared to each other by eye — each has its own
    # arbitrary rotation, scale and layout — so every panel is labelled with a
    # number that IS comparable. PCA-50 first so the score is measured in the same
    # way for a 3072-d baseline and a 6144-d latent.
    # Sequence caches store integer labels as object arrays, which sklearn
    # classifies as an unknown target type. Encode nominal IDs for the diagnostic
    # while retaining the original labels for plot legends and return values.
    class_index = {label: i for i, label in enumerate(classes)}
    target = np.array([class_index[label] for label in y], dtype=np.int64)
    knn = {}
    for nm, R in reps:
        Rp = PCA(n_components=min(50, R.shape[1]), random_state=0).fit_transform(R)
        tr, te = train_test_split(np.arange(len(Rp)), test_size=0.4,
                                  random_state=0, stratify=target)
        km_ = KNeighborsClassifier(n_neighbors=10).fit(Rp[tr], target[tr])
        knn[nm] = float(km_.score(Rp[te], target[te]))
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
    ids = load_rung_ids(cache, src[1])
    if ids is not None:
        n_full = len(ids)
        if n_full > 60000:   # same subsample load_rung just drew
            ids = ids[np.random.RandomState(0).choice(n_full, 60000, replace=False)]
        if len(ids) != len(H):
            ids = None
    keep = [c for c, _ in Counter(y.tolist()).most_common(max_classes)]
    m = np.isin(y, keep); H, y = H[m], y[m]
    ids = None if ids is None else ids[m]
    models = {nm: mo for nm, mo in models.items()
              if not (needs_ids(*mo) and ids is None)}
    if not models:
        print(f"[geo]   centroids {rung}: no drawable model (token ids missing)"); return
    classes = sorted(set(y.tolist()))
    fig, axes = plt.subplots(1, len(models), figsize=(3.05 * len(models), 3.4))
    for ax, (nm, (kind, obj)) in zip(np.atleast_1d(axes), models.items()):
        lab = assign(H, (kind, obj), dev, ids=ids)
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
    ap.add_argument("--order", default="",
                    help="optional comma list of model names giving the panel order "
                         "(e.g. base,base_tokmean,bypass); default = models then baselines")
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
            # allow_token_bias=True asserts every use below goes through ae.encode(x, tok)
            ae, mean, std, _ = load_ae_checkpoint(path, dev, allow_token_bias=True); ae.eval()
            models[nm] = ("ae", (ae, mean, std))
        else:
            C, mean, std, K, *_ = load_baseline_kmeans(Path(path), dev, allow_erasure=True)
            meta = np.load(path, allow_pickle=True)
            U = (torch.as_tensor(meta["erase_U"], dtype=torch.float32, device=dev)
                 if "erase_U" in meta.files else None)
            tbl = None
            if "token_bias" in meta.files:      # token-mean control: x - b[tok]
                from geoae.token_bias import TokenBiasLookup
                tbl = TokenBiasLookup(str(meta["token_bias"]), dev)
            models[nm] = ("km", (C, mean, std, U, tbl) if (U is not None or tbl is not None)
                          else (C, mean, std))
    if args.order:
        names = [n for n in args.order.split(",") if n in models]
        models = {n: models[n] for n in names + [n for n in models if n not in names]}
    print(f"[geo] {len(models)} models ({', '.join(models)}) -> {out}/")

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
