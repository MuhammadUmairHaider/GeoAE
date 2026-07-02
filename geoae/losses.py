"""
Loss functions for GeoAE.

All functions accept float32 tensors (cast before calling if working in bf16/fp16).
Shapes annotated as: B=batch, D=hidden, L=latent, K=n_clusters.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Distance metric helpers
# ---------------------------------------------------------------------------
# Two clustering geometries are supported, selected by `metric`:
#   "euclidean" : magnitude-based — squared L2 distance on the raw latents.
#   "cosine"    : directional     — squared L2 distance on L2-normalised latents,
#                 which is monotone in cosine distance: ||â-ĉ||² = 2(1 - cos).
# Cosine is therefore "spherical k-means": the exact same Euclidean machinery,
# run on unit vectors. Keeping it this way means assignment, cluster_loss,
# sep_loss and the baseline all stay consistent under one switch.

VALID_METRICS = ("euclidean", "cosine")


def l2_normalize(t: Tensor, eps: float = 1e-8) -> Tensor:
    """Row-wise L2 normalisation (last dim)."""
    return t / t.norm(dim=-1, keepdim=True).clamp(min=eps)


def _maybe_normalize(t: Tensor, metric: str) -> Tensor:
    return l2_normalize(t) if metric == "cosine" else t


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp in log space (numerically stable)
# ---------------------------------------------------------------------------

def sinkhorn_log(
    cost: Tensor,      # (B, K)  squared distances
    tau: float,
    n_iter: int = 3,
) -> Tensor:
    """
    Returns soft assignment matrix Q of shape (B, K) where:
      - each row sums to 1  (each latent is fully assigned)
      - each column sums to B/K  (balanced across clusters)

    Works entirely in log space to avoid exp overflow / underflow.
    """
    B, K = cost.shape
    log_Q = -cost / tau  # (B, K)

    log_BK = torch.log(torch.tensor(B / K, dtype=log_Q.dtype, device=log_Q.device))

    for _ in range(n_iter):
        # Row normalisation: each row sums to 1
        log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True)
        # Column normalisation: each column sums to B/K
        log_Q = log_Q - torch.logsumexp(log_Q, dim=0, keepdim=True) + log_BK

    # End on a row normalisation so row sums are exactly 1 at output
    log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True)

    return torch.exp(log_Q)  # (B, K)


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def recon_loss(x: Tensor, x_hat: Tensor) -> tuple[Tensor, Tensor]:
    """
    L_recon = MSE(x_hat, x).
    Also returns fraction of variance explained = 1 - MSE / Var(x).

    Returns: (mse_scalar, fve_scalar)
    """
    mse = F.mse_loss(x_hat, x)
    var_x = x.var(unbiased=False)
    fve = 1.0 - mse / (var_x + 1e-8)
    return mse, fve


def cluster_loss(
    z: Tensor,          # (B, L)
    centroids: Tensor,  # (K, L)  — must be detached before passing in
    Q: Tensor,          # (B, K)  soft assignment from sinkhorn_log
    metric: str = "euclidean",
) -> Tensor:
    """
    L_cluster = mean over batch of sum_k Q[b,k] * ||z[b] - c[k]||^2.

    Pulls each latent toward its Sinkhorn-assigned centroid(s).
    centroids must be detached (no gradient through centroid EMA updates).

    For metric="cosine" the latents and centroids are L2-normalised first, so
    the pull is directional and dist2 ∈ [0, 4] (no per-dim scaling needed).
    """
    if metric == "cosine":
        z = l2_normalize(z)
        centroids = l2_normalize(centroids)
        dist2 = torch.cdist(z, centroids, p=2).pow(2)              # (B, K), O(1) scale
    else:
        # dist2[b, k] = ||z[b] - c[k]||^2 / latent_dim  (per-dim, comparable scale to MSE)
        dist2 = torch.cdist(z, centroids, p=2).pow(2) / z.shape[1]  # (B, K)
    return (Q * dist2).sum(dim=1).mean()



def sep_loss(z: Tensor, Q: Tensor, metric: str = "euclidean") -> Tensor:
    """
    L_sep = mean_{i≠j} exp(-||m_i - m_j||^2 / sigma^2)

    where m_k = Σ_b Q[b,k] * z[b] / Σ_b Q[b,k]  are the soft batch cluster means,
    and sigma^2 = median pairwise dist^2 across (i,j) pairs (computed in no_grad).

    Adaptive sigma means the loss focuses pressure on whichever pairs are
    currently closest: pairs at the median get exp(-1)≈0.37; pairs much further
    decay toward zero; pairs much closer push harder. Once all pairs are roughly
    equidistant the loss is constant → no further forcing.

    Gradients flow back through z to the encoder.

    For metric="cosine" the latents are L2-normalised first, so separation is
    measured between directional cluster means.
    """
    if metric == "cosine":
        z = l2_normalize(z)
    weights = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)  # (B, K)
    cluster_means = weights.T @ z                        # (K, L)

    K = cluster_means.shape[0]
    dist2 = torch.cdist(cluster_means, cluster_means, p=2).pow(2)  # (K, K)
    mask = ~torch.eye(K, dtype=torch.bool, device=cluster_means.device)
    pair_d2 = dist2[mask]

    with torch.no_grad():
        sigma2 = pair_d2.median().clamp(min=1e-8)

    return torch.exp(-pair_d2 / sigma2).mean()


def usage_loss(Q: Tensor) -> Tensor:
    """
    L_usage = L1(u - 1/K) where u_k = mean over batch of Q[:, k].

    Penalises deviation from uniform cluster usage.
    """
    K = Q.shape[1]
    u = Q.mean(dim=0)           # (K,)
    target = torch.full_like(u, 1.0 / K)
    return F.l1_loss(u, target)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def total_loss(
    x: Tensor,
    x_hat: Tensor,
    z: Tensor,
    centroids: Tensor,   # detached
    Q: Tensor,
    lambda_cluster: float,
    lambda_sep: float,
    metric: str = "euclidean",
) -> dict[str, Tensor]:
    """
    Returns a dict with 'loss' (scalar to backprop) and individual components.

    usage_loss is omitted: Sinkhorn already enforces column sums = B/K, so
    Q.mean(0) ≈ 1/K by construction and the term is always ~0.
    """
    l_recon, fve = recon_loss(x, x_hat)
    l_cluster = cluster_loss(z, centroids, Q, metric=metric)
    l_sep = sep_loss(z, Q, metric=metric)

    loss = (
        l_recon
        + lambda_cluster * l_cluster
        + lambda_sep * l_sep
    )

    return {
        "loss": loss,
        "recon": l_recon,
        "fve": fve,
        "cluster": l_cluster,
        "sep": l_sep,
    }


# ---------------------------------------------------------------------------
# Sinkhorn cost matrix helper (used by model and losses)
# ---------------------------------------------------------------------------

def pairwise_sq_dist(z: Tensor, centroids: Tensor, metric: str = "euclidean") -> Tensor:
    """Returns (B, K) squared-distance matrix.

    metric="cosine" normalises both sides first, so the squared L2 distance is
    monotone in cosine distance (directional / spherical k-means).
    """
    z = _maybe_normalize(z, metric)
    centroids = _maybe_normalize(centroids, metric)
    return torch.cdist(z, centroids, p=2).pow(2)
