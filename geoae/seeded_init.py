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
  fill     The remaining K - n_anchor centroids are chosen by DENSITY x COVERAGE:
           score = density(x) * d2_to_nearest_selected(x). The d2 term keeps
           k-means++'s coverage property; the density term stops it landing on
           outliers. Density is 1 / (mean distance to the k nearest reference
           points), estimated against a fixed reference subsample.
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
def load_anchor_pool(cache_dir: str | Path, rungs=None, per_class: int = 25,
                     min_examples: int = 5, seed: int = 0):
    """
    Returns (H, keys): raw activations and a "rung::class" key per row, taking at
    most `per_class` examples from every class that has at least `min_examples`.
    """
    from geoae.interp.concept_probe import LADDER, load_rung

    rng = np.random.RandomState(seed)
    Hs, keys = [], []
    for rung, fname, key, grain, _ in LADDER:
        if rungs and rung not in rungs:
            continue
        if not (Path(cache_dir) / fname).exists():
            continue
        try:
            H, y = load_rung(str(cache_dir), fname, key, 0)
        except Exception:
            continue
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            if len(idx) < min_examples:
                continue
            take = idx if len(idx) <= per_class else rng.choice(idx, per_class, replace=False)
            Hs.append(H[take])
            keys += [f"{rung}::{cls}"] * len(take)
    if not Hs:
        raise ValueError(f"no anchor classes found under {cache_dir}")
    return np.concatenate(Hs).astype(np.float32), np.asarray(keys)


@torch.no_grad()
def anchor_centroids(model, H, keys, mean, std, device, min_sep_frac: float = 0.25,
                     encode_batch: int = 8192):
    """Per-class mean latent for each anchor class, near-duplicates removed."""
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
                density_power: float = 1.0, seed: int = 0, min_sep_frac: float = 0.25):
    """
    Anchors from labels, then a density x coverage fill.

    Returns (C, names, pool) where `pool` is an AnchorPool over the INDIVIDUAL
    labelled examples (not the class means). reinit draws from it so the
    supervised signal is exhausted before any unsupervised heuristic is used.
    """
    H, keys = load_anchor_pool(cache_dir, rungs, per_class, min_examples, seed)
    A, names = anchor_centroids(model, H, keys, mean, std, device, min_sep_frac)
    pool = AnchorPool(H, keys, mean, std, device)
    K = model.n_clusters
    if len(A) >= K:
        return A[:K], names[:K], pool
    fill = density_coverage_fill(z_pool, A, K - len(A), seed=seed,
                                 density_power=density_power)
    return torch.cat([A, fill]), names + [f"fill::{i}" for i in range(len(fill))], pool


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
        self.mean, self.std = mean, std
        self.used = torch.zeros(len(H), dtype=torch.bool, device=device)

    def __len__(self):
        return int((~self.used).sum().item())

    @torch.no_grad()
    def take(self, model, n: int, live_centroids: torch.Tensor):
        """
        Up to `n` unused labelled latents, chosen FARTHEST from the live
        centroids first — those are the labelled regions the partition currently
        has no home for. Returns (latents, keys) and marks them used.
        """
        avail = (~self.used).nonzero(as_tuple=True)[0]
        if len(avail) == 0 or n <= 0:
            return None, []
        z = model.encoder((self.H[avail] - self.mean) / self.std)
        d = torch.cdist(z, live_centroids).min(1).values
        k = min(n, len(avail))
        pick = d.topk(k).indices
        chosen = avail[pick]
        self.used[chosen] = True
        return z[pick], self.keys[chosen.cpu().numpy()].tolist()
