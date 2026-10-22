"""Shared Sinkhorn hard-EMA updates for already initialized balanced codebooks."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor

from geoae.losses import sinkhorn_log
from geoae.training.initialization import density_peaks_select


@torch.no_grad()
def balanced_updates(
    X: Tensor, C: Tensor, *, epochs: int, batch_size: int,
    tau_start: float = 1.0, tau_end: float = 0.1, sinkhorn_iters: int = 3,
    ema_decay: float = 0.9, reinit_every: int = 1000, reinit: str = "peaks",
    density_power: float = 1.0, density_knn: int = 32,
    peak_min_sep_frac: float = 0.25, peak_refine_k: int = 8, peak_reinit_pool: int = 8192,
    progress: Callable[[range, int], Iterable[int]] | None = None,
    on_epoch: Callable[[int, Tensor, float, int], None] | None = None,
) -> Tensor:
    """Update C in place and return it, without reseeding or choosing initialization.

    Only full batches are used. Unvisited centroids retain their positions;
    starved centroids are optionally replaced by density peaks or farthest rows.
    Callbacks receive one-based epoch numbers and must not change tensors or RNG.
    """
    K = len(C)
    if batch_size < K:
        raise ValueError("batch_size < K: Sinkhorn column marginal is meaningless")
    if reinit not in ("peaks", "farthest"):
        raise ValueError("reinit must be peaks or farthest")
    ema_size = torch.zeros(K, device=X.device)
    ema_sum = C.clone()
    steps_per_epoch = len(X) // batch_size
    total = epochs * steps_per_epoch
    step = n_reinit = 0
    tau = tau_start
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=X.device)
        steps = range(steps_per_epoch)
        if progress is not None:
            steps = progress(steps, ep + 1)
        for s in steps:
            b = X[perm[s * batch_size:(s + 1) * batch_size]]
            frac = step / max(total - 1, 1)
            tau = tau_start + (tau_end - tau_start) * frac
            Q = sinkhorn_log(torch.cdist(b, C) ** 2, tau, sinkhorn_iters)
            hard = torch.nn.functional.one_hot(Q.argmax(1), K).to(b.dtype)
            cnt = hard.sum(0)
            ema_size.mul_(ema_decay).add_(cnt * (1 - ema_decay))
            ema_sum.mul_(ema_decay).add_((hard.T @ b) * (1 - ema_decay))
            visited = cnt > 0
            C[visited] = ema_sum[visited] / ema_size[visited].unsqueeze(1).clamp_min(1e-6)
            if reinit_every and step and step % reinit_every == 0:
                dead = ema_size < (0.1 * ema_size.mean())
                if int(dead.sum()):
                    if reinit == "peaks":
                        pool = b[:peak_reinit_pool]
                        new, _ = density_peaks_select(
                            pool, min(int(dead.sum()), len(pool)), C_init=C[~dead],
                            seed=0, density_power=density_power, min_sep_frac=peak_min_sep_frac,
                            knn_k=density_knn, refine_k=peak_refine_k,
                        )
                        dead = dead.nonzero(as_tuple=True)[0][:len(new)]
                    else:
                        new = b[torch.cdist(b, C).min(1).values.topk(int(dead.sum())).indices]
                    n_reinit += len(new)
                    C[dead] = new
                    ema_size[dead] = ema_size.mean()
                    ema_sum[dead] = new * ema_size.mean()
            step += 1
        if on_epoch is not None:
            on_epoch(ep + 1, C, tau, n_reinit)
    return C
