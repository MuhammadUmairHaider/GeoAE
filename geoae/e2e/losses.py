"""
End-to-end loss for GeoAE: KL faithfulness + cluster + sep.

KL replaces the MSE reconstruction term. The cluster and separation terms are
imported unchanged from the base `losses` module, so the clustering geometry is
identical to the MSE pipeline — only the faithfulness signal changes.

KL conventions (hardcoded, per design decision):
  * direction   : forward KL( P_teacher || Q_student )  (teacher = original LLM)
  * temperature : 1.0
  * reduction   : mean over batch
  * units       : nats (natural log), computed in float32 for stability
"""
from __future__ import annotations



import torch
import torch.nn.functional as F
from torch import Tensor

from geoae.losses import cluster_loss, sep_loss, recon_loss


def kl_loss(teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
    """
    Forward KL( P_teacher || Q_student ), averaged over the batch.

        KL = Σ_v p_t(v) · ( log p_t(v) − log q_s(v) )

    teacher_logits / student_logits : (B, V). Computed in float32.
    The teacher term is detached (it is a fixed target; no gradient to it).
    """
    log_p_t = F.log_softmax(teacher_logits.float(), dim=-1)
    log_q_s = F.log_softmax(student_logits.float(), dim=-1)
    p_t = log_p_t.exp().detach()
    log_p_t = log_p_t.detach()
    return (p_t * (log_p_t - log_q_s)).sum(dim=-1).mean()


def total_loss_e2e(
    teacher_logits: Tensor,   # (B, V)  original last-token logits (target)
    student_logits: Tensor,   # (B, V)  reconstructed-splice last-token logits
    x: Tensor,                # (B, D)  AE input (normalised) — diagnostics only
    x_hat: Tensor,            # (B, D)  AE reconstruction (normalised) — diag only
    z: Tensor,                # (B, L)  latent
    centroids: Tensor,        # (K, L)  detached
    Q: Tensor,                # (B, K)  soft assignment
    lambda_cluster: float,
    lambda_sep: float,
    lambda_mse: float = 0.0,
    metric: str = "euclidean",
) -> dict[str, Tensor]:
    """
    L = KL + lambda_cluster · L_cluster + lambda_sep · L_sep + lambda_mse · MSE

    With lambda_mse == 0 (default) the MSE / FVE are computed without grad and
    returned as diagnostics only, so the run matches the pure-KL pipeline. A
    small lambda_mse adds a weak reconstruction anchor to the gradient, keeping
    the AE faithful to the residual geometry instead of drifting to a KL-only
    solution that ignores it.
    """
    l_kl = kl_loss(teacher_logits, student_logits)
    l_cluster = cluster_loss(z, centroids, Q, metric=metric)
    l_sep = sep_loss(z, Q, metric=metric)

    loss = l_kl + lambda_cluster * l_cluster + lambda_sep * l_sep

    if lambda_mse > 0:
        l_recon, fve = recon_loss(x, x_hat)
        loss = loss + lambda_mse * l_recon
    else:
        with torch.no_grad():
            l_recon, fve = recon_loss(x, x_hat)

    return {
        "loss": loss,
        "kl": l_kl,
        "recon": l_recon,   # in the gradient only when lambda_mse > 0
        "fve": fve,         # diagnostic only
        "cluster": l_cluster,
        "sep": l_sep,
    }
