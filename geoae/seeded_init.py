"""
Semi-supervised, density-first centroid initialisation and reinit.

MOTIVATION.  Two separate problems with the k-means++ default, both observed:

1. On DBpedia-14 at K=14, seeding each centroid from 5-10 LABELLED examples per
   class reached ~93% purity where k-means++ reproduced ~55%. k-means++ samples
   the next centroid with probability proportional to D^2 (squared distance to
   the nearest existing centroid), which deliberately chases the points FURTHEST
   from what it already has -- i.e. outliers. That is the right objective for
   minimising worst-case k-means cost and the wrong one for landing a centroid
   on a semantic mode.

2. `reinit_dead_clusters` inherits the same bias in a sharper form: it replaces a
   dead centroid with the latent of the HIGHEST-RECONSTRUCTION-LOSS sample in the
   batch, which is close to a definition of an outlier. On the b32k run this
   churned ~20-40 clusters per cycle without ever settling.

WHAT THIS DOES INSTEAD.
  anchors  Each labelled class from the concept caches contributes one centroid,
           the mean of `per_class` encoded examples (5-100; few-shot on purpose).
           Anchors closer than `min_sep` to an existing anchor are dropped, since
           the rungs overlap (a token can be both a POS tag and an NER type).
  fill     The remaining K - n_anchor centroids, by one of two rules:
           "coverage"  density(x) * d2_to_nearest_selected(x). Keeps k-means++'s
                       coverage property, with a density term meant to stop it
                       landing on outliers. In practice the d2 term still wins
                       (see `density_peaks_select`).
           "peaks"     DENSITY PEAKS: rho^a * delta, where delta is the distance
                       to the nearest DENSER point, not to the nearest selected
                       one. No coverage term, so nothing outgrows the density.
           Density is 1 / (mean distance to the k nearest reference points),
           estimated against a fixed reference subsample.
  reinit   Dead centroids are replaced from the same density-ranked pool rather
           than from high-loss samples.

Density is computed in LATENT space, at the moment of initialisation, so it
reflects the encoder as trained rather than the raw activation geometry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------- anchors
def anchor_row_indices(cache_dir: str | Path, rungs=None, per_class: int = 25,
                       min_examples: int = 5, seed: int = 0,
                       atlas_last: str = "", atlas_min_examples: int = 25):
    """
    {rung: np.array of row indices} that seeding consumes, WITHOUT loading H.

    Exists so evaluation can exclude exactly the rows used as anchors. Seeding
    draws from the same concept caches the probe scores on, so without this the
    anchored rungs report a fake win: on the L14 seeded run, 37% of
    ravel_country and 27% of ravel_language eval rows were also anchors, and
    those were precisely the rungs that "improved".

    Selection must stay deterministic in (cache_dir, rungs, per_class,
    min_examples, seed) or the exclusion will not match what training used.
    """
    from geoae.interp.concept_probe import LADDER, load_rung

    rng = np.random.RandomState(seed)
    out = {}
    for rung, fname, key, grain, _ in LADDER:
        if rungs and rung not in rungs:
            continue
        if not (Path(cache_dir) / fname).exists():
            continue
        try:
            _, y = load_rung(str(cache_dir), fname, key, 0)
        except Exception:
            continue
        picks = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) < min_examples:
                continue
            picks.append(idx if len(idx) <= per_class
                         else rng.choice(idx, per_class, replace=False))
        if picks:
            out[rung] = np.concatenate(picks)

    # Atlas anchors are selected CHUNK-wise by load_atlas_anchors, not by the
    # per-class token rule above, so the ladder loop cannot find them. Map the
    # anchor chunk ids onto each atlas rung's rows via its stored `chunk` array.
    # Without this the atlas rungs are ~78% contaminated and the seeded arm
    # reports a large fake win on exactly the concepts it was seeded from.
    if atlas_last:
        _, _, anchor_chunks = load_atlas_anchors(atlas_last, per_class,
                                                 atlas_min_examples, seed=seed)
        if anchor_chunks is not None:
            A = set(anchor_chunks.tolist())
            for rung, fname, key, grain, _ in LADDER:
                if not rung.startswith("atlas_"):
                    continue
                fp = Path(cache_dir) / fname
                if not fp.exists():
                    continue
                ch = np.load(str(fp), allow_pickle=True).get("chunk")
                if ch is None:
                    continue
                hit = np.array([i for i, c in enumerate(ch.tolist()) if c in A],
                               dtype=np.int64)
                if len(hit):
                    out[rung] = hit
    return out


def load_atlas_anchors(path: str | Path, per_class: int = 25, min_examples: int = 25,
                       fields=("document_ids", "tone_ids", "content_ids"), seed: int = 0):
    """
    FineWeb-Atlas concepts as anchors, from CHUNK LAST-TOKEN activations.

    Atlas labels ~95-token chunks, not tokens, so the chunk's final token is the
    representative: at a late layer it has attended over the whole chunk, and it
    is still a single token of the kind the AE was trained on (a mean-pooled
    chunk vector is off-distribution for the encoder).

    Multi-label by construction — a chunk carries ~1.6 document, ~6.9 tone and
    ~5.7 content labels — so one chunk's last token seeds several concepts. The
    dedup in `anchor_centroids` merges anchors that end up in the same place.

    BUDGET: at min_examples=25 this yields ~448 classes (23 doc + 89 tone + 336
    content) on the 7,742-chunk cache. At min_examples=5 it is ~2,081, which
    exceeds K=2000 on its own — anchors would fill the entire codebook and the
    density fill would never run. Keep the default unless K is much larger.

    CONTAMINATION: these chunks are the same ones `concept_separability` scores.
    Returns the chunk indices used so an evaluation can hold them out.
    """
    d = np.load(str(path), allow_pickle=True)
    H = d["H_last"]
    rng = np.random.RandomState(seed)
    Hs, keys, used = [], [], []
    for f in fields:
        if f not in d.files:
            continue
        rows = d[f]
        pos = {}
        for i, r in enumerate(rows):
            for c in np.asarray(r).tolist():
                pos.setdefault(c, []).append(i)
        for c, idx in pos.items():
            if len(idx) < min_examples:
                continue
            idx = np.asarray(idx)
            take = idx if len(idx) <= per_class else rng.choice(idx, per_class, replace=False)
            Hs.append(H[take])
            keys += [f"atlas_{f.replace('_ids', '')}::{c}"] * len(take)
            used.append(take)
    if not Hs:
        return None, None, None
    return (np.concatenate(Hs).astype(np.float32), np.asarray(keys),
            np.unique(np.concatenate(used)))


def load_anchor_pool(cache_dir: str | Path, rungs=None, per_class: int = 25,
                     min_examples: int = 5, seed: int = 0,
                     atlas_last: str = "", atlas_min_examples: int = 25):
    """
    Returns (H, keys): raw activations and a "rung::class" key per row, taking at
    most `per_class` examples from every class that has at least `min_examples`.
    """
    from geoae.interp.concept_probe import LADDER, load_rung

    sel = anchor_row_indices(cache_dir, rungs, per_class, min_examples, seed)
    byname = {r[0]: (r[1], r[2]) for r in LADDER}
    Hs, keys = [], []
    for rung, idx in sel.items():
        fname, key = byname[rung]
        H, y = load_rung(str(cache_dir), fname, key, 0)
        Hs.append(H[idx])
        keys += [f"{rung}::{c}" for c in y[idx]]
    if atlas_last:
        aH, aK, aidx = load_atlas_anchors(atlas_last, per_class, atlas_min_examples, seed=seed)
        if aH is not None:
            Hs.append(aH)
            keys += aK.tolist()
            print(f"[seeded] atlas: +{len(set(aK.tolist()))} concept classes "
                  f"({len(aH):,} last-token rows) from {len(aidx):,} chunks")
    if not Hs:
        raise ValueError(f"no anchor classes found under {cache_dir}")
    return np.concatenate(Hs).astype(np.float32), np.asarray(keys)


def _as_t(v, device):
    """
    Coerce a normalisation stat to a float tensor on `device`.

    ActivationBuffer stores `mean`/`std` as NUMPY (see data.py:147, which converts
    at the use-site), while callers that build them from a checkpoint pass torch
    tensors. Accepting only one of those silently killed two 10-epoch runs at the
    epoch-11 init, and the traceback went to stderr where `| tee` did not capture
    it. Normalise the input here instead of trusting the caller.
    """
    return torch.as_tensor(v, device=device, dtype=torch.float32)


@torch.no_grad()
def anchor_centroids(model, H, keys, mean, std, device, min_sep_frac: float = 0.25,
                     encode_batch: int = 8192):
    """Per-class mean latent for each anchor class, near-duplicates removed."""
    mean, std = _as_t(mean, device), _as_t(std, device)
    z = []
    for i in range(0, len(H), encode_batch):
        h = torch.from_numpy(H[i:i + encode_batch]).to(device).float()
        z.append(model.encoder((h - mean) / std))
    z = torch.cat(z)

    uniq = sorted(set(keys.tolist()))
    C, names = [], []
    for k in uniq:
        m = torch.from_numpy(keys == k).to(device)
        C.append(z[m].mean(0))
        names.append(k)
    C = torch.stack(C)

    # Drop near-duplicate anchors: the rungs overlap, so the same region can be
    # claimed by several labels (a token may be a POS tag AND an entity type).
    D = torch.cdist(C, C)
    typical = D[~torch.eye(len(C), dtype=torch.bool, device=device)].median()
    keep, kept_idx = [], []
    for i in range(len(C)):
        if not keep or torch.cdist(C[i:i + 1], torch.stack(keep)).min() > min_sep_frac * typical:
            keep.append(C[i])
            kept_idx.append(i)
    return torch.stack(keep), [names[i] for i in kept_idx]


# --------------------------------------------------------------------- density
@torch.no_grad()
def local_density(z: torch.Tensor, n_ref: int = 8192, k: int = 32, seed: int = 0,
                  chunk: int = 8192) -> torch.Tensor:
    """1 / (mean distance to the k nearest reference points). Higher = denser."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ref = z[torch.randperm(len(z), generator=g)[:min(n_ref, len(z))].to(z.device)]
    out = []
    for i in range(0, len(z), chunk):
        d = torch.cdist(z[i:i + chunk], ref)
        out.append(d.topk(min(k + 1, d.shape[1]), largest=False).values[:, 1:].mean(1))
    d = torch.cat(out)
    return 1.0 / d.clamp_min(1e-6)


@torch.no_grad()
def nearest_higher_density(z: torch.Tensor, rho: torch.Tensor,
                           chunk: int = 1024) -> torch.Tensor:
    """
    delta_i = distance from i to the nearest point of STRICTLY HIGHER density.

    This is the quantity that makes density-peaks different from farthest-first.
    A point in the tail of a mode always has a denser neighbour just uphill, so
    its delta is tiny however far it sits from the centroids chosen so far; only
    a local maximum of rho can have a large delta.

    Computed in rho-descending order, which makes it a triangular scan (half the
    distance work) and breaks exact rho ties by rank. Comparing with a literal
    `>` instead would let two equal-density points each be "not higher" than the
    other, giving both an inflated delta -- duplicate peaks on one mode.
    """
    order = torch.argsort(rho, descending=True)
    zs = z[order]
    n = len(zs)
    delta = torch.full((n,), float("inf"), device=z.device)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        blk = zs[i:j]
        if i > 0:
            delta[i:j] = torch.cdist(blk, zs[:i]).min(1).values
        # Denser points inside this chunk: strictly-preceding rows only.
        dd = torch.cdist(blk, blk)
        dd.masked_fill_(torch.triu(torch.ones_like(dd, dtype=torch.bool)), float("inf"))
        delta[i:j] = torch.minimum(delta[i:j], dd.min(1).values)
    # The global density maximum has no denser point; convention is the largest
    # delta in the set, so it always ranks first rather than inf-poisoning gamma.
    delta[0] = delta[1:].max() if n > 1 else torch.zeros((), device=z.device)
    out = torch.empty_like(delta)
    out[order] = delta
    return out


def _neighbourhood_scale(dens: torch.Tensor) -> torch.Tensor:
    """
    Median kNN radius — the length `min_sep_frac` is a fraction of.

    Deliberately LOCAL. `anchor_centroids` dedups against the median PAIRWISE
    distance, which is right for ~500 class means spread over the whole space
    but wrong here: that statistic is set by how far apart the extremes are, so
    on a pool with any outlier spread the radius swallows whole modes and the
    greedy pass is left choosing between points no centroid should ever sit on.
    Measured on 3 tight blobs plus uniform outliers, the pairwise scale accepted
    one pick per blob and then forced every further pick out onto an outlier.
    rho is 1/(mean kNN distance), so this costs nothing.
    """
    return 1.0 / dens.median().clamp_min(1e-9)


@torch.no_grad()
def density_peaks_select(z: torch.Tensor, n_needed: int, C_init: torch.Tensor | None = None,
                         dens: torch.Tensor | None = None, seed: int = 0,
                         density_power: float = 1.0, min_sep_frac: float = 0.25,
                         knn_k: int = 32, refine_k: int = 8, chunk: int = 1024):
    """
    Density-peaks selection (Rodriguez & Laio, Science 2014).

    Ranks by gamma = rho^density_power * delta, where delta is the distance to
    the nearest DENSER point (see `nearest_higher_density`). Contrast with
    `density_coverage_fill`, whose d2 term measures distance to the nearest
    SELECTED point: that term grows without bound as the selected set spreads
    out, while the density weight is a bounded median-normalised ratio, so
    coverage always wins in the end. Measured on the L27 parent checkpoint, the
    coverage fill at density_power=1.0 still selected points at density 0.0105
    against a pool median of 0.0173 — below-median density, i.e. the density
    term never actually took over. Here there is no coverage term to lose to:
    an outlier has a large delta but near-zero rho, and a tail point has decent
    rho but near-zero delta, so only mode centres score.

    Peaks also self-separate: two points on the same mode cannot both have a
    large delta, since one is uphill of the other. `min_sep_frac` is a guard
    against near-duplicates, not the mechanism — it is a fraction of the median
    kNN radius (see `_neighbourhood_scale`), so it rejects coincident picks
    without ever excluding a whole mode.

    Returns (picks (n_needed, L), diag dict). Always returns exactly n_needed
    rows — `init_centroids_from_class_means` asserts an exact (K, L) shape.
    """
    if dens is None:
        dens = local_density(z, k=knn_k, seed=seed)
    dens_n = dens / dens.median().clamp_min(1e-9)
    delta = nearest_higher_density(z, dens, chunk=chunk)
    gamma = dens_n.pow(density_power) * delta

    scale = _neighbourhood_scale(dens)
    radius = min_sep_frac * scale

    zsq = (z * z).sum(1)

    def d2_to(c):
        return (zsq - 2.0 * (z @ c) + (c * c).sum()).clamp_min_(0)

    # Anchors (when given) act as suppression centres so peaks never land on top
    # of a centroid that is already placed.
    dmin2 = torch.full((len(z),), float("inf"), device=z.device)
    if C_init is not None and len(C_init):
        for c in C_init:
            dmin2 = torch.minimum(dmin2, d2_to(c))

    order = torch.argsort(gamma, descending=True)
    r2 = float(radius) ** 2
    picked: list[int] = []
    taken = torch.zeros(len(z), dtype=torch.bool, device=z.device)
    for idx in order.tolist():
        if len(picked) >= n_needed:
            break
        if dmin2[idx] <= r2:
            continue
        picked.append(idx)
        taken[idx] = True
        dmin2 = torch.minimum(dmin2, d2_to(z[idx]))

    # The radius pass can exhaust the pool before n_needed. Top up by gamma
    # regardless of separation rather than returning a short tensor.
    n_forced = 0
    if len(picked) < n_needed:
        for idx in order.tolist():
            if len(picked) >= n_needed:
                break
            if taken[idx]:
                continue
            picked.append(idx)
            taken[idx] = True
            n_forced += 1

    sel = torch.tensor(picked, device=z.device)
    picks = z[sel]

    if refine_k > 0:
        # One mean-shift step onto the local mode: the same "mean of the most
        # typical members, not a single medoid" rule AnchorPool.take uses, so
        # one unusual point cannot define a centroid.
        out = []
        for i in range(0, len(picks), chunk):
            d = torch.cdist(picks[i:i + chunk], z)
            nn = d.topk(min(refine_k, len(z)), largest=False).indices    # (c, refine_k)
            out.append(z[nn].mean(1))
        picks = torch.cat(out)

    # density_pct is THE number to check: percentile of the picks' density
    # within the pool. Below 50 means the selection is still outlier-seeking.
    # Measured at the SELECTED PEAK POINTS, before refine_k moves the centroid
    # to a neighbourhood mean, so it understates the placement the trainer ends
    # up with: on the L27 parent it read 71st here against 97th for the refined
    # centroids. Conservative on purpose — it scores the selection, not the
    # smoothing.
    pct = (dens[sel].unsqueeze(1) > dens.unsqueeze(0)).float().mean(1).mean() * 100
    diag = {
        "density_pct": float(pct),
        # A pick is a genuine peak when the nearest denser point lies OUTSIDE
        # its own kNN radius (1/rho is that mean radius) — a strict-local-max
        # proxy that costs nothing, since both terms are already computed.
        "n_peak": int((delta[sel] > (1.0 / dens[sel].clamp_min(1e-9))).sum()),
        "n_sel": len(sel),
        "n_forced": n_forced,
        # As a multiple of the median kNN radius, so <1 means two centroids are
        # closer together than a typical neighbourhood is wide.
        "min_sep_frac": (float(torch.cdist(picks, picks).masked_fill(
            torch.eye(len(picks), dtype=torch.bool, device=z.device), float("inf")).min()
            / scale) if len(picks) > 1 else float("nan")),
    }
    return picks, diag


def format_peak_diag(d: dict) -> str:
    """
    One-line density-peaks init summary.

    `density pct` is the number that decides whether the selection worked: the
    mean percentile of the picks' density within the pool. The coverage fill it
    replaces measured BELOW the 50th (density 0.0105 vs a pool median of 0.0173
    on the L27 parent), i.e. it was still seeding outliers. Above 50 means the
    density term actually took over. `peaks` counts picks whose nearest denser
    point lies outside their own kNN radius, so it doubles as a count of how
    many real modes the latent space has.
    """
    s = (f"density pct of picks: {d['density_pct']:.0f}th (pool median = 50th) "
         f"| peaks: {d['n_peak']:,}/{d['n_sel']:,} "
         f"| min sep: {d['min_sep_frac']:.2f}x typical")
    if d["n_forced"]:
        s += f" | {d['n_forced']:,} forced past min_sep (pool exhausted)"
    return s


@torch.no_grad()
def density_coverage_fill(z: torch.Tensor, C_init: torch.Tensor | None, n_needed: int,
                          dens: torch.Tensor | None = None, seed: int = 0,
                          density_power: float = 1.0) -> torch.Tensor:
    """
    Greedy selection maximising density(x) * d2(x, nearest selected).

    density_power = 0 reduces this to greedy k-means++ (coverage only); 1.0 is
    the balanced default. The d2 factor is what stops every centroid collapsing
    into the single densest mode.
    """
    if dens is None:
        dens = local_density(z, seed=seed)
    dens = dens / dens.median().clamp_min(1e-9)
    w = dens.pow(density_power)

    zsq = (z * z).sum(1)

    def d2_to(c):
        return (zsq - 2.0 * (z @ c) + (c * c).sum()).clamp_min_(0)

    if C_init is not None and len(C_init):
        d2 = torch.full((len(z),), float("inf"), device=z.device)
        for c in C_init:
            d2 = torch.minimum(d2, d2_to(c))
    else:
        start = int(torch.multinomial(w, 1).item())
        d2 = d2_to(z[start])
        C_init = z[start:start + 1]

    picked = []
    for _ in range(n_needed):
        i = int((w * d2).argmax().item())
        picked.append(z[i])
        d2 = torch.minimum(d2, d2_to(z[i]))
    return torch.stack(picked)


@torch.no_grad()
def seeded_init(model, cache_dir, mean, std, device, z_pool: torch.Tensor,
                per_class: int = 25, min_examples: int = 5, rungs=None,
                density_power: float = 1.0, seed: int = 0, min_sep_frac: float = 0.25,
                atlas_last: str = "", atlas_min_examples: int = 25,
                fill_mode: str = "coverage", knn_k: int = 32, refine_k: int = 8):
    """
    Anchors from labels, then a density fill for the remaining K - n_anchor.

    `fill_mode` picks the fill rule, with the anchors held constant so the two
    can be compared directly:
      "coverage"  density x distance-to-nearest-selected (`density_coverage_fill`)
      "peaks"     density peaks (`density_peaks_select`), anchors passed as
                  suppression centres so no peak lands on an existing anchor

    Returns (C, names, pool, diag) where `pool` is an AnchorPool over the
    INDIVIDUAL labelled examples (not the class means). reinit draws from it so
    the supervised signal is exhausted before any unsupervised heuristic is
    used. `diag` is the fill diagnostic dict, or None for "coverage".
    """
    H, keys = load_anchor_pool(cache_dir, rungs, per_class, min_examples, seed,
                               atlas_last, atlas_min_examples)
    A, names = anchor_centroids(model, H, keys, mean, std, device, min_sep_frac)
    pool = AnchorPool(H, keys, mean, std, device)
    K = model.n_clusters
    if len(A) >= K:
        return A[:K], names[:K], pool, None
    diag = None
    if fill_mode == "peaks":
        fill, diag = density_peaks_select(z_pool, K - len(A), C_init=A, seed=seed,
                                          density_power=density_power,
                                          min_sep_frac=min_sep_frac,
                                          knn_k=knn_k, refine_k=refine_k)
    else:
        fill = density_coverage_fill(z_pool, A, K - len(A), seed=seed,
                                     density_power=density_power)
    return (torch.cat([A, fill]), names + [f"fill::{i}" for i in range(len(fill))],
            pool, diag)


class AnchorPool:
    """
    The individual labelled examples, held as RAW activations and re-encoded on
    demand.

    Re-encoding rather than caching latents matters: the encoder keeps training
    after the centroids are initialised, so latents captured at init go stale
    within an epoch. The pool is small (~11k rows at per_class=25) so a forward
    pass costs nothing next to a reinit cycle.

    `used` tracks which examples have already been spent as a centroid seed, so
    successive reinits exhaust the supervised pool instead of re-proposing the
    same few points.
    """

    def __init__(self, H, keys, mean, std, device):
        self.H = torch.from_numpy(H).to(device)
        self.keys = np.asarray(keys)
        self.mean, self.std = _as_t(mean, device), _as_t(std, device)
        self.used = torch.zeros(len(H), dtype=torch.bool, device=device)

    def __len__(self):
        return int((~self.used).sum().item())

    @torch.no_grad()
    def take(self, model, n: int, live_centroids: torch.Tensor, mean_k: int = 5):
        """
        Up to `n` seeds, selected PER CLASS and placed at the class CENTRE.

        An earlier version picked the individual unused example farthest from any
        live centroid. That is farthest-first selection — the same outlier-seeking
        rule as k-means++, merely restricted to a labelled subset — and within a
        class the farthest point is that class's most ATYPICAL example, so it
        seeded every concept on its worst representative.

        Instead:
          priority  per CLASS, distance from the class mean to the nearest live
                    centroid, divided by the class's own radius. High means "this
                    concept has no home relative to how tight it is", which is a
                    coverage question, not a distance question.
          placement the MEAN of the `mean_k` most typical members (those closest
                    to the class mean). Averaging >=5 examples rather than taking a
                    single point keeps one unusual example from defining a concept.

        A class is consumed once seeded (all its remaining examples are marked
        used), so successive reinits move on to the next unserved concept rather
        than re-seeding the same one.
        """
        avail = (~self.used).nonzero(as_tuple=True)[0]
        if len(avail) == 0 or n <= 0:
            return None, []
        z = model.encoder((self.H[avail] - self.mean) / self.std)
        keys = self.keys[avail.cpu().numpy()]

        uniq = sorted(set(keys.tolist()))
        stats = []
        for cls in uniq:
            m = torch.from_numpy(keys == cls).to(z.device)
            zc = z[m]
            mu = zc.mean(0)
            radius = (zc - mu).norm(dim=1).mean().clamp_min(1e-6)
            gap = torch.cdist(mu[None], live_centroids).min()
            # mean of the mean_k most typical members, not a single medoid
            k_use = min(mean_k, len(zc))
            typical = (zc - mu).norm(dim=1).topk(k_use, largest=False).indices
            stats.append(((gap / radius).item(), cls, zc[typical].mean(0)))
        stats.sort(key=lambda s: -s[0])                                    # worst-served first

        picks, names = [], []
        for _, cls, seed_vec in stats[:n]:
            picks.append(seed_vec)
            names.append(cls)
            self.used[avail[torch.from_numpy(keys == cls).to(z.device)]] = True
        if not picks:
            return None, []
        return torch.stack(picks), names
