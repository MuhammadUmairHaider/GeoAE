"""
Diagnostic metrics logged every N training steps.

All functions accept plain tensors; caller is responsible for .detach().
Compute-heavy ops (silhouette, pairwise distances) work on fixed-size samples
to keep overhead manageable.
"""
from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def cluster_usage(Q_history: list[Tensor], n_clusters: int) -> Tensor:
    """
    u_k = mean over all batches in Q_history of Q[:, k].  Returns (K,).

    Accepts EITHER form of history entry:
      (B, K) full soft-assignment matrices, or
      (K,)   that batch's column means.

    The second is what the training loops now store, and it is exact rather than
    an approximation: with equal-sized batches (drop_last=True) the mean of the
    per-batch column means equals the mean over all rows. It matters because the
    full form scales with BATCH SIZE — at batch 32768, K=2000, a 200-entry
    history is 52 GB of host RAM and OOM-kills the run, while the means form is
    1.6 MB.
    """
    if Q_history and Q_history[0].dim() == 1:
        return torch.stack(Q_history, dim=0).mean(dim=0)   # (K,)
    stacked = torch.cat(Q_history, dim=0)   # (N_total, K)
    return stacked.mean(dim=0)              # (K,)


@torch.no_grad()
def centroid_distances(centroids: Tensor) -> dict[str, float]:
    """Mean, min, max of pairwise centroid distances."""
    K = centroids.shape[0]
    dists = torch.cdist(centroids, centroids, p=2)
    mask = ~torch.eye(K, dtype=torch.bool, device=centroids.device)
    d = dists[mask]
    return {
        "centroid_dist_mean": d.mean().item(),
        "centroid_dist_min":  d.min().item(),
        "centroid_dist_max":  d.max().item(),
    }


@torch.no_grad()
def intra_cluster_variance(z: Tensor, Q: Tensor, centroids: Tensor) -> float:
    """
    Mean within-cluster variance of latents (assigned via argmax of Q).
    Should be < inter-cluster variance for a good clustering.
    """
    assignments = Q.argmax(dim=1)      # (B,)
    variances = []
    K = centroids.shape[0]
    for k in range(K):
        mask = assignments == k
        if mask.sum() < 2:
            continue
        zk = z[mask]
        variances.append(zk.var(dim=0).mean().item())
    return float(torch.tensor(variances).mean().item()) if variances else 0.0


@torch.no_grad()
def effective_k(u: Tensor, threshold_factor: float = 0.1) -> int:
    """
    Number of clusters with u_k > threshold_factor / K.
    This is the "real" K after dead cluster decay.
    """
    K = u.shape[0]
    threshold = threshold_factor / K
    return int((u > threshold).sum().item())


@torch.no_grad()
def sinkhorn_entropy(Q: Tensor) -> float:
    """Mean row entropy of Q.  Higher = softer, lower = harder."""
    eps = 1e-10
    H = -(Q * (Q + eps).log()).sum(dim=1).mean()
    return H.item()


@torch.no_grad()
def silhouette_sample(
    z: Tensor,
    Q: Tensor,
    max_samples: int = 10_000,
) -> float:
    """
    Crude silhouette score on up to max_samples points.
    Uses sklearn for correctness; falls back to -1.0 if unavailable.
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return -1.0

    N = z.shape[0]
    if N > max_samples:
        idx = torch.randperm(N)[:max_samples]
        z, Q = z[idx], Q[idx]

    labels = Q.argmax(dim=1).cpu().numpy()
    z_np = z.cpu().float().numpy()

    # silhouette_score requires 2 <= n_unique_labels <= n_samples - 1.
    # With K >> batch every sample can land in its own cluster, hitting the
    # upper bound; both degenerate cases make silhouette undefined.
    n_labels = len(set(labels.tolist()))
    if n_labels < 2 or n_labels > len(labels) - 1:
        return -1.0

    return float(silhouette_score(z_np, labels, metric="euclidean",
                                  sample_size=min(5000, len(z_np))))


def compute_all(
    z: Tensor,              # (B, L)  detached latents
    Q: Tensor,              # (B, K)  detached soft assignments
    centroids: Tensor,      # (K, L)  detached centroids
    train_mse: float,
    val_mse: float,
    train_fve: float,
    val_fve: float,
    Q_history: list[Tensor],
    run_silhouette: bool = False,
) -> dict[str, float]:
    """
    Aggregate all diagnostic metrics into a single flat dict for logging.
    """
    u = cluster_usage(Q_history, centroids.shape[0])
    dying_threshold = 0.5 / centroids.shape[0]
    dominant_threshold = 5.0 / centroids.shape[0]

    metrics = {
        "train/mse":          train_mse,
        "val/mse":            val_mse,
        "train/fve":          train_fve,
        "val/fve":            val_fve,
        "cluster/dying":      int((u < dying_threshold).sum().item()),
        "cluster/dominant":   int((u > dominant_threshold).sum().item()),
        "cluster/effective_k": effective_k(u),
        "cluster/entropy":    sinkhorn_entropy(Q),
        "cluster/intra_var":  intra_cluster_variance(z, Q, centroids),
    }
    metrics.update(centroid_distances(centroids))

    if run_silhouette:
        metrics["cluster/silhouette"] = silhouette_sample(z, Q)

    return metrics
