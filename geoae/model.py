"""
GeoAE: encoder → cluster module → decoder.

Encoder variants (controlled by `nonlinearity`):
  None / "linear" : Linear(D→L)                  — rotation-ambiguous latent space
  "relu"          : Linear(D→L) → ReLU           — non-negative, privileged basis
  "gelu"          : Linear(D→L) → GeLU           — smooth, approximate sparse basis
  "tanh"          : Linear(D→L) → tanh           — signed, bounded in (-1, 1)

Privileged-basis intuition:
  A linear encoder is equivariant to rotations of the latent space — you can rotate
  z without changing reconstruction loss, so there is no canonical coordinate system.
  A ReLU encoder breaks this: rotating z changes which units are non-zero, so the
  individual latent dimensions have meaning.  This is the same reasoning that makes
  SAEs use ReLU.  The clustering module then operates in a space where each axis
  already has a preferred direction.

Shapes throughout:
  B = batch size
  D = hidden_size (3072 for Llama 3.2 3B)
  L = latent_dim  (2048 default)
  K = n_clusters  (128 default)
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from geoae.losses import (sinkhorn_log, sinkhorn_log_dual, cluster_log_prior,
                          pairwise_sq_dist, l2_normalize, VALID_METRICS)

VALID_BALANCES = ("uniform", "zipf")


class AEOutput(NamedTuple):
    x_hat: Tensor      # (B, D)  reconstruction in normalised space
    z: Tensor          # (B, L)  latent
    Q: Tensor          # (B, K)  soft cluster assignment
    dist2: Tensor      # (B, K)  squared distances to centroids


_NONLINEARITIES = {
    None:      None,
    "linear":  None,
    "relu":    nn.ReLU(),
    "gelu":    nn.GELU(),
    "tanh":    nn.Tanh(),
}


class GeoAE(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3072,
        latent_dim: int = 2048,
        n_clusters: int = 128,
        ema_decay: float = 0.99,
        sinkhorn_iters: int = 3,
        balance: str = "uniform",
        balance_rho: float = 1.0,
        balance_eta: float = 1.0,
        zipf_alpha: float = 0.0,
        tau: float = 1.0,
        nonlinearity: str | None = None,
        metric: str = "euclidean",
        ema_hard: bool = False,
        latent_norm: str = "none",
        dist_scale: str = "raw",
    ):
        super().__init__()
        self.dist_scale = dist_scale
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.ema_decay = ema_decay
        self.sinkhorn_iters = sinkhorn_iters
        if balance not in VALID_BALANCES:
            raise ValueError(f"balance must be one of {VALID_BALANCES}, got {balance!r}")
        self.balance = balance
        self.balance_rho = balance_rho
        self.balance_eta = balance_eta
        self.zipf_alpha = zipf_alpha
        self.tau = tau
        self.nonlinearity = nonlinearity
        self.use_sinkhorn = True
        self.ema_hard = ema_hard

        if metric not in VALID_METRICS:
            raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")
        self.metric = metric

        if nonlinearity not in _NONLINEARITIES:
            raise ValueError(f"nonlinearity must be one of {list(_NONLINEARITIES)}, got {nonlinearity!r}")

        if latent_norm not in ("none", "batch"):
            raise ValueError(f"latent_norm must be 'none' or 'batch', got {latent_norm!r}")
        self.latent_norm = latent_norm

        # latent_norm="batch" inserts BatchNorm1d BETWEEN the linear map and the
        # activation (the standard Linear->BN->act placement). Two reasons it is
        # placed there rather than on the post-activation latent:
        #
        #   1. Nothing else constrains the latent's scale. There is no norm layer
        #      anywhere in this model and z is unnormalised, while cluster_loss
        #      actively rewards CONTRACTION (pulling points to centroids is cheap
        #      in a small space). lambda_var is currently the sole counter-
        #      pressure; BN makes it structural instead of a soft penalty.
        #   2. It keeps the pre-activations centred and unit-scaled THROUGHOUT
        #      training, which is what stops the encoder from drifting to a
        #      degenerate scale. (It does NOT buy sparsity — measured at init,
        #      pre-activations are already 50% negative with and without BN, and
        #      GELU leaves only ~2% of the latent below |z|<0.01 either way.
        #      An overcomplete latent still has no sparsity pressure.)
        #
        # Placing BN AFTER the activation would instead recentre the (mostly
        # non-negative) GELU output, which would break the shared mean offset
        # that effective_rank already has to correct for.
        #
        # affine=True is the standard choice and is kept, but note the learnable
        # gamma can itself shrink — BN bounds the contraction, it does not forbid
        # it. affine=False would be the hard version.
        #
        # Default "none" constructs no module at all, so every existing
        # checkpoint's state_dict is unchanged and loads exactly as before.
        act = _NONLINEARITIES[nonlinearity]
        layers: list[nn.Module] = [nn.Linear(hidden_size, latent_dim, bias=True)]
        if latent_norm == "batch":
            layers.append(nn.BatchNorm1d(latent_dim))
        if act is not None:
            layers.append(act)
        self.encoder = layers[0] if len(layers) == 1 else nn.Sequential(*layers)

        # Decoder: no bias; columns renormalised to unit norm after each step
        self.decoder = nn.Linear(latent_dim, hidden_size, bias=False)

        # Centroids stored as a buffer (not a parameter — updated by EMA only)
        self.register_buffer("centroids", torch.zeros(n_clusters, latent_dim))
        self.register_buffer("centroids_initialized", torch.tensor(False))

        # EMA running counts (used to stabilise early EMA updates)
        self.register_buffer("ema_cluster_size", torch.ones(n_clusters))
        # Persistent column dual for online (balance_eta < 1) Sinkhorn. Zero unless used.
        self.register_buffer("sinkhorn_g", torch.zeros(n_clusters))

        self._init_weights()

    def _init_weights(self) -> None:
        # For Sequential encoder the Linear is always the first child
        enc_linear = self.encoder[0] if isinstance(self.encoder, nn.Sequential) else self.encoder
        nn.init.xavier_uniform_(enc_linear.weight)
        nn.init.zeros_(enc_linear.bias)
        # Decoder columns initialised to unit norm
        nn.init.xavier_uniform_(self.decoder.weight)
        self._renorm_decoder()

    def _renorm_decoder(self) -> None:
        """Renormalise decoder weight columns to unit norm (standard SAE practice)."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    @torch.no_grad()
    def _renorm_centroids(self) -> None:
        """For cosine/directional clustering, project centroids onto the unit sphere.

        Centroids live on the sphere, so all distance computations are purely
        directional. No-op for the euclidean (magnitude) metric.
        """
        if self.metric == "cosine":
            self.centroids.copy_(l2_normalize(self.centroids))

    @torch.no_grad()
    def init_centroids_from_data(self, z_sample: Tensor) -> None:
        """
        Initialise centroids by randomly sampling from a batch of latents.
        Kept for callers that don't want k-means++ (e.g. tests).
        """
        K = self.n_clusters
        assert z_sample.shape[0] >= K, (
            f"Need at least {K} samples to initialise centroids, got {z_sample.shape[0]}"
        )
        perm = torch.randperm(z_sample.shape[0], device=z_sample.device)[:K]
        self.centroids.copy_(z_sample[perm])
        self._renorm_centroids()
        self.centroids_initialized.fill_(True)

    @torch.no_grad()
    def init_centroids_kmeans_plus_plus(self, z_sample: Tensor, seed: int = 0) -> None:
        """
        K-means++ initialisation. First centroid is uniform random; each
        subsequent centroid is sampled from z_sample with probability
        proportional to its squared distance to the nearest existing centroid.

        Best run AFTER the recon-only warmup so the encoder has produced
        meaningful latents.
        """
        K = self.n_clusters
        N, _ = z_sample.shape
        assert N >= K, f"Need ≥ K={K} samples, got {N}"

        # Directional metric: select on the unit sphere so distances are cosine.
        if self.metric == "cosine":
            z_sample = l2_normalize(z_sample)

        g = torch.Generator(device=z_sample.device).manual_seed(seed)

        first = torch.randint(0, N, (1,), generator=g, device=z_sample.device).item()
        picked = [first]
        centroids = z_sample[first:first+1].clone()       # (1, L)

        for _ in range(K - 1):
            # Squared distance from each sample to its nearest existing centroid
            d2 = torch.cdist(z_sample, centroids, p=2).pow(2).min(dim=1).values  # (N,)
            d2[picked] = 0.0                               # don't re-pick
            total = d2.sum()
            if total <= 0:
                # All remaining points coincide with existing centroids — fall
                # back to a random pick from the not-yet-chosen set
                remaining = [i for i in range(N) if i not in picked]
                idx = remaining[torch.randint(0, len(remaining), (1,),
                                              generator=g).item()]
            else:
                probs = d2 / total
                idx = torch.multinomial(probs, 1, generator=g).item()
            picked.append(idx)
            centroids = torch.cat([centroids, z_sample[idx:idx+1]], dim=0)

        self.centroids.copy_(centroids)
        self._renorm_centroids()
        self.centroids_initialized.fill_(True)

    @torch.no_grad()
    def init_centroids_from_class_means(self, class_means: Tensor) -> None:
        """
        Semi-supervised init: set each centroid to the mean latent of its class.
        class_means: (K, L) — one mean per class, ordered by class index.
        """
        assert class_means.shape == (self.n_clusters, self.latent_dim), (
            f"Expected ({self.n_clusters}, {self.latent_dim}), got {class_means.shape}"
        )
        self.centroids.copy_(class_means)
        self._renorm_centroids()
        self.centroids_initialized.fill_(True)

    def _default_balancing(self) -> bool:
        """True when the knobs reduce exactly to the original sinkhorn_log call."""
        return (self.balance == "uniform" and self.balance_rho == 1.0
                and self.balance_eta == 1.0)

    @torch.no_grad()
    def _column_log_prior(self) -> Tensor | None:
        """(K,) log target marginal indexed by CLUSTER id, or None for uniform.

        A power law is only defined up to a permutation of clusters, so the
        masses are matched to clusters by descending `ema_cluster_size`: the
        constraint then fixes the SHAPE of the usage histogram without dictating
        which cluster must be frequent.
        """
        if self.balance != "zipf" or self.zipf_alpha == 0.0:
            return None
        ranked = cluster_log_prior(self.n_clusters, self.zipf_alpha,
                                   device=self.centroids.device,
                                   dtype=self.centroids.dtype)
        order = torch.argsort(self.ema_cluster_size, descending=True)
        out = torch.empty_like(ranked)
        out[order] = ranked
        return out

    @torch.no_grad()
    def target_usage(self) -> Tensor:
        """(K,) target usage share per cluster id — uniform, or the ranked Zipf prior."""
        lp = self._column_log_prior()
        if lp is None:
            return torch.full((self.n_clusters,), 1.0 / self.n_clusters,
                              device=self.centroids.device, dtype=self.centroids.dtype)
        return lp.exp()

    def forward(self, x: Tensor) -> AEOutput:
        """
        x: (B, D)  normalised activation
        Returns AEOutput with x_hat, z, Q, dist2.
        """
        z = self.encoder(x)                              # (B, L)
        dist2 = pairwise_sq_dist(z, self.centroids, metric=self.metric)  # (B, K)
        # SCALE OF THE SINKHORN INPUT.
        #
        # cluster_loss divides ||z - c||^2 by latent_dim (losses.py:187) so the
        # cluster term is on the same per-dimension scale as MSE. forward() did
        # NOT, so `tau` operated on distances latent_dim times larger than the
        # loss's own notion of distance. Measured on the trained d6144 L27 run:
        #
        #   raw      nearest-vs-2nd gap 484.2, gap/tau = 4841  -> exp(-gap) = 0
        #            Q row entropy 0.008 of 7.60 nats, max Q 0.996
        #   per_dim  gap 0.0788,             gap/tau = 0.79    -> exp(-gap) = 0.46
        #            Q row entropy 5.02 nats, max Q 0.188
        #
        # i.e. with dist_scale="raw" the assignment is one-hot to four decimals
        # and the whole tau_start -> tau_end anneal is INERT: gap/tau is 484 at
        # tau=1.0 and 4841 at tau=0.1, both effectively infinite. cluster_loss
        # then reduces to plain k-means (one centroid pulls). Sinkhorn's column
        # constraint still binds -- it reassigns ~2.7% of tokens as a discrete
        # swap -- which is why balancing shapes the partition even though tau
        # does nothing.
        #
        # "raw" is kept as the default so every pre-2026-09-04 checkpoint and
        # config reproduces exactly. "per_dim" puts tau on the loss's scale and
        # NEEDS A DIFFERENT TAU RANGE (~0.2 -> 0.01, not 1.0 -> 0.1).
        # Only the Sinkhorn input is rescaled: `dist2` returned in AEOutput stays
        # raw, so every argmin-based eval and the reinit path are untouched.
        sk_dist2 = dist2 / z.shape[1] if self.dist_scale == "per_dim" else dist2
        if self.use_sinkhorn:
            if self._default_balancing():
                # Untouched legacy path — bit-identical to every shipped checkpoint.
                Q = sinkhorn_log(sk_dist2, tau=self.tau, n_iter=self.sinkhorn_iters)  # (B, K)
            else:
                Q, g = sinkhorn_log_dual(
                    sk_dist2, tau=self.tau, n_iter=self.sinkhorn_iters,
                    log_prior=self._column_log_prior(),
                    rho=self.balance_rho, eta=self.balance_eta,
                    g=self.sinkhorn_g if self.balance_eta < 1.0 else None,
                )
                if self.training and self.balance_eta < 1.0:
                    self.sinkhorn_g.copy_(g.detach())
        else:
            Q = torch.softmax(-dist2 / self.tau, dim=1)  # (B, K) unbalanced
        x_hat = self.decoder(z)                          # (B, D)
        return AEOutput(x_hat=x_hat, z=z, Q=Q, dist2=dist2)

    @torch.no_grad()
    def update_centroids_ema(self, z: Tensor, Q: Tensor) -> None:
        """
        EMA update of centroids using the latents and soft assignment from the
        most recent forward pass.  Call AFTER the optimizer step.

        z: (B, L)   latents (from the forward pass — pre-step values are fine)
        Q: (B, K)   soft assignment (detached)

        With ema_hard=True the soft Q is replaced by a one-hot of its argmax, so
        each centroid averages only the latents it actually wins. This breaks
        the soft-EMA "chase" where every centroid mixes broadly similar
        Sinkhorn-weighted batch means and the whole configuration drifts into a
        low-rank slice around the data mean.
        """
        if self.ema_hard:
            Q = torch.nn.functional.one_hot(
                Q.argmax(dim=1), num_classes=self.n_clusters
            ).to(z.dtype)

        # Directional metric: accumulate unit-vector latents so the running mean
        # is a mean direction (spherical k-means update = mean then renormalise).
        if self.metric == "cosine":
            z = l2_normalize(z)

        # Per-cluster sum of assigned latents
        cluster_sum = Q.T @ z               # (K, L)
        cluster_count = Q.sum(dim=0)        # (K,)

        # EMA update of running counts
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_count * (1 - self.ema_decay))

        # Mean of assigned latents, normalised by the RAW batch count (clamped).
        # NOTE: all shipped checkpoints were trained with this raw-count division;
        # switching to the EMA-smoothed count would change training dynamics.
        new_centroids = cluster_sum / cluster_count.unsqueeze(1).clamp(min=1e-6)

        # EMA update of centroid positions. Under hard assignment a centroid can
        # win zero points in a batch (its new_centroids row is 0); it must keep
        # its position rather than decay toward the origin, so only visited rows
        # are updated. Under soft Sinkhorn every column has mass — no-op there.
        if self.ema_hard:
            visited = (cluster_count > 0).unsqueeze(1).to(self.centroids.dtype)  # (K, 1)
            step = (1 - self.ema_decay) * visited
            self.centroids.mul_(1 - step).add_(new_centroids * step)
        else:
            self.centroids.mul_(self.ema_decay).add_(new_centroids * (1 - self.ema_decay))

        # Re-project onto the unit sphere for cosine clustering (no-op otherwise)
        self._renorm_centroids()

    def post_step(self) -> None:
        """Call after optimizer.step(): renormalise decoder columns."""
        self._renorm_decoder()
