"""
Scaffolding shared by all three trainers (geoae.train, geoae.e2e.train,
geoae.e2e.train_stream): schedules, model/optimizer construction, wandb init,
CLI overrides, checkpoint rotation and best-val tracking, dead-cluster reinit.

The training loops themselves stay in their own modules — their loss paths
genuinely differ — but everything that must stay in lockstep lives here so the
pipelines can't drift apart.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch

from geoae.config import Config
from geoae.model import GeoAE


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def tau_schedule(epoch: int, cfg: Config) -> float:
    """
    Linear anneal of Sinkhorn temperature from tau_start to tau_end
    over epochs [full_loss_start_epoch .. n_epochs].
    Returns tau_start before full-loss phase begins.
    """
    tc = cfg.loss
    tr = cfg.train
    if epoch < tr.full_loss_start_epoch:
        return tc.tau_start
    progress = (epoch - tr.full_loss_start_epoch) / max(
        tr.n_epochs - tr.full_loss_start_epoch, 1
    )
    return tc.tau_start + (tc.tau_end - tc.tau_start) * min(progress, 1.0)


def lambda_schedule(epoch: int, cfg: Config) -> tuple[float, float]:
    """
    Returns (lambda_cluster, lambda_sep) for the current epoch.

    Epochs 1..recon_only_epochs:       both zero
    Epochs clustering_start..full-1:   cluster only
    Epochs full_loss_start..:          both active (ramp sep from 0)
    """
    tr = cfg.train
    lc = cfg.loss
    if epoch <= tr.recon_only_epochs:
        return 0.0, 0.0
    if epoch < tr.full_loss_start_epoch:
        return lc.lambda_cluster, 0.0
    progress = (epoch - tr.full_loss_start_epoch) / max(
        tr.n_epochs - tr.full_loss_start_epoch, 1
    )
    progress = min(progress, 1.0)
    return lc.lambda_cluster, lc.lambda_sep * progress


# ---------------------------------------------------------------------------
# Model / optimizer / wandb construction from config
# ---------------------------------------------------------------------------

def build_model(cfg: Config, device: torch.device, no_sinkhorn: bool = False) -> GeoAE:
    """Construct a GeoAE from config on `device` (single source of truth)."""
    model = GeoAE(
        hidden_size=cfg.model.hidden_size,
        latent_dim=cfg.model.latent_dim,
        n_clusters=cfg.model.n_clusters,
        ema_decay=cfg.train.ema_decay,
        sinkhorn_iters=cfg.loss.sinkhorn_iters,
        tau=cfg.loss.tau_start,
        nonlinearity=cfg.model.nonlinearity,
        metric=cfg.model.metric,
        ema_hard=cfg.train.ema_hard,
    ).to(device)
    if no_sinkhorn:
        model.use_sinkhorn = False
    return model


def make_optimizer(model: GeoAE, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )


def init_wandb(cfg: Config, use_wandb: bool, run_name: str, tag: str = "train"):
    """Returns the wandb module as logger, or None (unavailable / disabled)."""
    if not use_wandb:
        return None
    try:
        import wandb
        wandb.init(
            project=cfg.train.wandb_project,
            entity=cfg.train.wandb_entity,
            config=cfg.to_dict(),
            name=run_name,
        )
        print(f"[{tag}] wandb initialised")
        return wandb
    except Exception as e:
        print(f"[{tag}] wandb unavailable ({e}); stdout only")
        return None


# ---------------------------------------------------------------------------
# CLI overrides shared by the trainers
# ---------------------------------------------------------------------------

def add_override_args(ap) -> None:
    """Config-override flags common to all trainers."""
    ap.add_argument("--n_clusters", type=int, default=None, help="Override model.n_clusters")
    ap.add_argument("--latent_dim", type=int, default=None, help="Override model.latent_dim")
    ap.add_argument("--nonlinearity", default=None, choices=["linear", "relu", "gelu"],
                    help="Override encoder nonlinearity")
    ap.add_argument("--checkpoints_dir", default=None, help="Override train.checkpoints_dir")
    ap.add_argument("--no_cluster", action="store_true",
                    help="Disable cluster loss (set lambda_cluster=0)")
    ap.add_argument("--no_sep", action="store_true",
                    help="Disable separation loss (set lambda_sep=0)")
    ap.add_argument("--lambda_sep", type=float, default=None, help="Override lambda_sep")
    ap.add_argument("--no_sinkhorn", action="store_true",
                    help="Disable Sinkhorn balanced assignment (use softmax instead)")
    ap.add_argument("--no_renorm", action="store_true",
                    help="Disable decoder unit-norm renormalisation (diagnostic)")


def apply_overrides(cfg: Config, args) -> None:
    """Apply CLI overrides to cfg. Tolerates flags a given CLI doesn't define."""
    def opt(name):
        return getattr(args, name, None)

    if opt("n_clusters") is not None:
        cfg.model.n_clusters = args.n_clusters
    if opt("latent_dim") is not None:
        cfg.model.latent_dim = args.latent_dim
    if opt("nonlinearity") is not None:
        cfg.model.nonlinearity = args.nonlinearity
    if opt("metric") is not None:
        cfg.model.metric = args.metric
    if opt("batch_size") is not None:
        cfg.data.batch_size = args.batch_size
    if opt("centroid_init") is not None:
        cfg.train.centroid_init = args.centroid_init
    if opt("teacher_mode") is not None:
        cfg.train.teacher_mode = args.teacher_mode
    if opt("checkpoints_dir") is not None:
        cfg.train.checkpoints_dir = args.checkpoints_dir
    if opt("lr") is not None:
        cfg.train.lr = args.lr
    if getattr(args, "no_cluster", False):
        cfg.loss.lambda_cluster = 0.0
    if opt("lambda_sep") is not None:
        cfg.loss.lambda_sep = args.lambda_sep
    elif getattr(args, "no_sep", False):
        cfg.loss.lambda_sep = 0.0
    if opt("lambda_mse") is not None:
        cfg.loss.lambda_mse = args.lambda_mse


# ---------------------------------------------------------------------------
# Checkpoint rotation + best-val tracking
# ---------------------------------------------------------------------------

def rotate_checkpoints(ckpt_dir: Path, keep: int, protected: set[Path]) -> None:
    """Keep the last `keep` checkpoints by step number; never delete protected paths."""
    all_ckpts = sorted(
        [p for p in ckpt_dir.glob("step_*.pt")],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for old in all_ckpts[:-keep]:
        if old not in protected:
            old.unlink(missing_ok=True)


class CheckpointTracker:
    """
    Owns best-val promotion and rotation. The subtle invariant — never rotate
    away the just-saved checkpoint or the current best — lives here once.
    """

    def __init__(self, ckpt_dir: Path, keep: int):
        self.ckpt_dir = ckpt_dir
        self.keep = keep
        self.best_score = float("inf")
        self.best_path: Path | None = None

    @property
    def best_target(self) -> Path:
        return self.ckpt_dir / "best_val.pt"

    def is_better(self, score: float) -> bool:
        return score < self.best_score

    def record_best(self, score: float) -> None:
        """Note that best_val.pt was (re)written externally with this score."""
        self.best_score = score
        self.best_path = self.best_target

    def finish_epoch(self, ckpt_path: Path, score: float) -> bool:
        """Promote ckpt_path to best_val.pt if score improved, then rotate.
        Returns True if this checkpoint became the new best."""
        is_best = self.is_better(score)
        if is_best:
            shutil.copy2(ckpt_path, self.best_target)
            self.record_best(score)
        protected = {ckpt_path}
        if self.best_path is not None:
            protected.add(self.best_path)
        rotate_checkpoints(self.ckpt_dir, self.keep, protected)
        return is_best


# ---------------------------------------------------------------------------
# Dead-cluster reinitialization
# ---------------------------------------------------------------------------

@torch.no_grad()
def reinit_dead_clusters(
    model: GeoAE,
    z_batch: torch.Tensor,
    x_batch: torch.Tensor,
    x_hat_batch: torch.Tensor,
    threshold_factor: float = 0.1,
) -> int:
    """
    Identify dead clusters (usage < threshold_factor/K) and reinitialise
    their centroids to latents from high-reconstruction-loss samples.
    Returns number of clusters reinitialised.
    """
    K = model.n_clusters
    u = model.ema_cluster_size / model.ema_cluster_size.sum()
    dead_mask = u < (threshold_factor / K)
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    # Rank batch samples by reconstruction loss (highest = most informative)
    per_sample_loss = (x_hat_batch - x_batch).pow(2).mean(dim=1)  # (B,)
    _, top_idx = per_sample_loss.topk(min(n_dead, len(per_sample_loss)))
    replacement_latents = z_batch[top_idx]  # (n_dead, L)

    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    for i, k in enumerate(dead_indices[:len(replacement_latents)]):
        model.centroids[k].copy_(replacement_latents[i])
        model.ema_cluster_size[k] = 1.0  # reset EMA count

    return n_dead
