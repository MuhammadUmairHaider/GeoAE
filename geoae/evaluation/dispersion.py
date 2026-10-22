"""Streaming cosine dispersion, including the paper's empirical-centroid metrics.

The within metric averages embedding-to-cluster-centroid cosine distances; the
between metric averages cosine distances between empirical cluster centroids.
We use arithmetic means of the original vectors for those centroids. This is a
documented convention: the paper does not specify whether vectors are normalized
before averaging. The primary within metric weights occupied clusters equally;
``within_sample_weighted`` also reports the mean over samples for unequal groups.

The separately named ``pairwise_*`` metrics describe row pairs, not the paper's
within/between definition. All reductions use float64 and need no pair matrix.
"""
from __future__ import annotations

import math

import torch


def _distance(value: torch.Tensor) -> float:
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("nonfinite cosine-distance reduction")
    # Only absorb floating-point roundoff at the cosine-distance boundaries.
    if result < -1e-10 or result > 2.0 + 1e-10:
        raise ValueError(f"cosine distance outside [0, 2]: {result}")
    return min(2.0, max(0.0, result))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def dispersion_metrics(
    vector_sum: torch.Tensor,
    unit_sum: torch.Tensor,
    counts: torch.Tensor,
    unit_self_norm: torch.Tensor,
) -> dict:
    """Reduce (K, D) sums and (K,) counts for one population or one fold.

    Empty clusters are ignored. Any occupied cluster with a zero empirical
    centroid makes *all* centroid metrics undefined; no cluster subset is
    silently substituted. Row-pair metrics remain defined where pairs exist.
    """
    if vector_sum.ndim != 2 or unit_sum.shape != vector_sum.shape:
        raise ValueError("vector_sum and unit_sum must have matching (K, D) shapes")
    if counts.shape != vector_sum.shape[:1] or unit_self_norm.shape != counts.shape:
        raise ValueError("counts and unit_self_norm must have shape (K,)")
    if not all(bool(torch.isfinite(t).all()) for t in (vector_sum, unit_sum, counts, unit_self_norm)):
        raise ValueError("sufficient statistics must be finite")
    if bool((counts < 0).any()):
        raise ValueError("counts must be nonnegative")
    if not torch.equal(counts, counts.floor()):
        raise ValueError("counts must be integers")
    v = vector_sum.double()
    u = unit_sum.double()
    count = counts.double()
    self_norm = unit_self_norm.double()
    occupied = count > 0
    n = int(count.sum().item())
    k = int(occupied.sum().item())
    total_pairs = n * (n - 1)
    within_pairs = int((count * (count - 1)).sum().item())
    between_pairs = total_pairs - within_pairs
    total_dot = u.sum(0).square().sum() - self_norm.sum()
    within_dot = (u.square().sum(1) - self_norm).sum()
    dispersion = _distance(1 - total_dot / total_pairs) if total_pairs else None
    pairwise_within = _distance(1 - within_dot / within_pairs) if within_pairs else None
    pairwise_between = (
        _distance(1 - (total_dot - within_dot) / between_pairs) if between_pairs else None
    )

    centroids = v[occupied]  # Dividing by count does not change centroid direction.
    centroid_norms = torch.linalg.vector_norm(centroids, dim=1)
    zero_centroids = int((centroid_norms == 0).sum().item())
    within = within_sample_weighted = between = None
    if k and not zero_centroids:
        directions = centroids / centroid_norms[:, None]
        distances = 1 - (u[occupied] * directions).sum(1) / count[occupied]
        within = _distance(distances.mean())
        within_sample_weighted = _distance((distances * count[occupied]).sum() / n)
        if k >= 2:
            dot = directions.sum(0).square().sum() - directions.square().sum()
            between = _distance(1 - dot / (k * (k - 1)))

    return {
        "dispersion": dispersion,
        "within": within,
        "within_sample_weighted": within_sample_weighted,
        "between": between,
        "between_over_within": _ratio(between, within),
        "pairwise_within": pairwise_within,
        "pairwise_between": pairwise_between,
        "pairwise_between_over_within": _ratio(pairwise_between, pairwise_within),
        "rows": n,
        "nonempty_clusters": k,
        "empty_clusters": len(count) - k,
        "singleton_clusters": int((count == 1).sum().item()),
        "zero_centroid_clusters": zero_centroids,
        "centroid_metrics_defined": bool(k and not zero_centroids),
        "within_ordered_pairs": within_pairs,
        "between_ordered_pairs": between_pairs,
    }


def vendi_score(vectors, max_rows: int = 2048) -> float | None:
    """Bounded cosine-kernel Vendi score following the paper's author code.

    Uses eigvalsh(U @ U.T) / N, keeping eigenvalues > 1e-10. The memory
    guard does not implicitly subsample: callers must sample explicitly.
    """
    vectors = torch.as_tensor(vectors).double()
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape (N, D)")
    if len(vectors) > max_rows:
        raise ValueError(f"Vendi is limited to {max_rows} rows; sample explicitly first")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("nonfinite representation rows: cosine distance is undefined")
    if not len(vectors):
        return None
    norms = torch.linalg.vector_norm(vectors, dim=1)
    if bool((norms == 0).any()) or not bool(torch.isfinite(norms).all()):
        raise ValueError("zero or nonfinite representation norms: cosine distance is undefined")
    units = vectors / norms[:, None]
    eigenvalues = torch.linalg.eigvalsh(units @ units.T) / len(units)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return float(torch.exp(-(eigenvalues * eigenvalues.log()).sum()).item())


class ClusterStatistics:
    """Per-fold cluster sums with separately streamed unit-vector variants.

    Use ``add`` for a single raw-vector pass. To obtain both raw and centred
    metrics in two passes, first ``add_vectors``, calculate the global mean,
    then ``add_units`` for each variant. ``metrics(..., centre=mean)`` derives
    centred empirical centroids without retaining another raw-sum tensor.
    The rows and assignments supplied for every variant must match pass one.
    """

    def __init__(self, n_clusters: int, dimension: int, n_folds: int = 1, device="cpu"):
        if min(n_clusters, dimension, n_folds) < 1:
            raise ValueError("n_clusters, dimension and n_folds must be positive")
        self.n_clusters = n_clusters
        self.dimension = dimension
        self.n_folds = n_folds
        self.device = torch.device(device)
        self.vector_sum = torch.zeros(n_folds, n_clusters, dimension, dtype=torch.float64, device=device)
        self.counts = torch.zeros(n_folds, n_clusters, dtype=torch.int64, device=device)
        self.units: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _inputs(self, vectors, labels, folds=None):
        vectors = torch.as_tensor(vectors, device=self.device).double()
        labels = torch.as_tensor(labels, device=self.device)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"vectors must have shape (N, {self.dimension})")
        if labels.shape != vectors.shape[:1] or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("cluster ids must be an integer vector of length N")
        if folds is None:
            folds = torch.zeros(len(vectors), dtype=torch.int64, device=self.device)
        else:
            folds = torch.as_tensor(folds, device=self.device)
        if folds.shape != labels.shape or folds.dtype not in (torch.int32, torch.int64):
            raise ValueError("fold ids must be an integer vector of length N")
        if bool(((labels < 0) | (labels >= self.n_clusters)).any()):
            raise ValueError("cluster id outside this space's cluster range")
        if bool(((folds < 0) | (folds >= self.n_folds)).any()):
            raise ValueError("fold id outside fold range")
        if not bool(torch.isfinite(vectors).all()):
            raise ValueError("nonfinite representation rows: cosine distance is undefined")
        return vectors, (folds.long() * self.n_clusters + labels.long())

    def add_vectors(self, vectors, labels, folds=None):
        vectors, index = self._inputs(vectors, labels, folds)
        self.vector_sum.view(-1, self.dimension).index_add_(0, index, vectors)
        self.counts.view(-1).index_add_(0, index, torch.ones_like(index))

    def add_units(self, vectors, labels, folds=None, variant="raw"):
        vectors, index = self._inputs(vectors, labels, folds)
        norms = torch.linalg.vector_norm(vectors, dim=1)
        if bool((norms == 0).any()):
            count = int((norms == 0).sum().item())
            raise ValueError(f"{count} zero representation rows in {variant}: cosine distance is undefined")
        if not bool(torch.isfinite(norms).all()):
            raise ValueError("nonfinite representation norms: cosine distance is undefined")
        units = vectors / norms[:, None]
        if variant not in self.units:
            self.units[variant] = (
                torch.zeros_like(self.vector_sum),
                torch.zeros_like(self.counts, dtype=torch.float64),
                torch.zeros_like(self.counts),
            )
        sums, self_norms, counts = self.units[variant]
        sums.view(-1, self.dimension).index_add_(0, index, units)
        self_norms.view(-1).index_add_(0, index, units.square().sum(1))
        counts.view(-1).index_add_(0, index, torch.ones_like(index))

    def add(self, vectors, labels, folds=None):
        # Validate units before changing vector statistics when a row is invalid.
        self.add_units(vectors, labels, folds)
        self.add_vectors(vectors, labels, folds)

    def mean(self):
        count = int(self.counts.sum().item())
        if not count:
            raise ValueError("cannot compute a mean of an empty population")
        return self.vector_sum.sum((0, 1)) / count

    def metrics(self, variant="raw", fold: int | None = None, centre=None):
        if variant not in self.units:
            if bool((self.counts > 0).any()):
                raise ValueError(f"no unit-vector statistics for variant {variant!r}")
            sums, self_norms, unit_counts = self.vector_sum, self.counts.double(), self.counts
        else:
            sums, self_norms, unit_counts = self.units[variant]
        if not torch.equal(unit_counts, self.counts):
            raise ValueError("unit-vector rows/cluster counts do not match the original-vector pass")
        if fold is None:
            v, u, c, q = (self.vector_sum.sum(0), sums.sum(0), self.counts.sum(0), self_norms.sum(0))
        else:
            if fold < 0 or fold >= self.n_folds:
                raise ValueError("fold id outside fold range")
            v, u, c, q = (self.vector_sum[fold], sums[fold], self.counts[fold], self_norms[fold])
        if centre is not None:
            centre = torch.as_tensor(centre, dtype=torch.float64, device=self.device)
            if centre.shape != (self.dimension,) or not bool(torch.isfinite(centre).all()):
                raise ValueError("centre must be a finite vector of the representation dimension")
            v = v - c[:, None] * centre
        return dispersion_metrics(v, u, c, q)
