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


def zipf_alpha_schedule(epoch: int, cfg: Config) -> float:
    """
    Zipf exponent for the current epoch, annealed 0 -> zipf_alpha_end over the
    same window as `tau_schedule`.

    Returns 0.0 for balance="uniform", which makes the prior exactly uniform, so
    a run with the defaults is unaffected. Starting at 0 matters: the Zipf masses
    are matched to clusters by current usage, and applying a heavy tail from
    step 1 would just amplify whichever cluster happened to win at init.
    """
    tc, tr = cfg.loss, cfg.train
    if tc.balance != "zipf":
        return 0.0
    if epoch < tr.full_loss_start_epoch:
        return tc.zipf_alpha_start
    progress = (epoch - tr.full_loss_start_epoch) / max(
        tr.n_epochs - tr.full_loss_start_epoch, 1
    )
    return tc.zipf_alpha_start + (tc.zipf_alpha_end - tc.zipf_alpha_start) * min(progress, 1.0)


def geometry_schedule(epoch: int, cfg: Config) -> tuple[float, float, float]:
    """
    Returns (lambda_var, lambda_cov, lambda_unif) for the current epoch.

    Gated by `train.geometry_start_epoch`, which defaults to 0 — meaning the
    geometry terms are live from the first step, exactly as before. Set it above
    1 to get an explicit reconstruction-only warmup before VICReg engages, e.g.
    epochs 1-5 recon only, 6-10 recon + VICReg, 11+ everything.

    Note this is a separate gate from `lambda_schedule`: cluster/sep and the
    geometry terms are phased independently.
    """
    lc, tr = cfg.loss, cfg.train
    if epoch < getattr(tr, "geometry_start_epoch", 0):
        return 0.0, 0.0, 0.0
    return lc.lambda_var, lc.lambda_cov, lc.lambda_unif


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
        balance=cfg.loss.balance,
        balance_rho=cfg.loss.balance_rho,
        balance_eta=cfg.loss.balance_eta,
        zipf_alpha=cfg.loss.zipf_alpha_start,
        tau=cfg.loss.tau_start,
        nonlinearity=cfg.model.nonlinearity,
        metric=cfg.model.metric,
        ema_hard=cfg.train.ema_hard,
        latent_norm=getattr(cfg.model, "latent_norm", "none"),
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
    ap.add_argument("--balance", default=None, choices=["uniform", "zipf"],
                    help="Override loss.balance (column-marginal target shape)")
    ap.add_argument("--balance_rho", type=float, default=None,
                    help="Override loss.balance_rho (1=hard Sinkhorn, 0=softmax)")
    ap.add_argument("--balance_eta", type=float, default=None,
                    help="Override loss.balance_eta (<1 carries the column dual across batches)")
    ap.add_argument("--zipf_alpha", type=float, default=None,
                    help="Override loss.zipf_alpha_end")
    ap.add_argument("--geometry_start_epoch", type=int, default=None,
                    help="Epoch at which var/cov/unif switch on (0 = from step 1)")
    ap.add_argument("--no_sinkhorn", action="store_true",
                    help="Disable Sinkhorn balanced assignment (use softmax instead)")
    ap.add_argument("--no_renorm", action="store_true",
                    help="Disable decoder unit-norm renormalisation (diagnostic)")
    ap.add_argument("--semisup_cap", type=int, default=None,
                    help="semisup init: max labeled samples/class for class-mean centroids")
    ap.add_argument("--activations_dir", default=None, help="Override data.activations_dir")
    ap.add_argument("--n_epochs", type=int, default=None,
                    help="Override train.n_epochs. Also rescales the tau anneal, which "
                         "runs full_loss_start_epoch..n_epochs — lowering it reaches "
                         "tau_end sooner.")


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
    if opt("semisup_cap") is not None:
        cfg.train.semisup_cap = args.semisup_cap
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
    if opt("balance") is not None:
        cfg.loss.balance = args.balance
    if opt("balance_rho") is not None:
        cfg.loss.balance_rho = args.balance_rho
    if opt("balance_eta") is not None:
        cfg.loss.balance_eta = args.balance_eta
    if opt("zipf_alpha") is not None:
        cfg.loss.zipf_alpha_end = args.zipf_alpha
    if opt("geometry_start_epoch") is not None:
        cfg.train.geometry_start_epoch = args.geometry_start_epoch
    if opt("activations_dir") is not None:
        cfg.data.activations_dir = args.activations_dir
    if opt("n_epochs") is not None:
        cfg.train.n_epochs = args.n_epochs


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
    u = model.ema_cluster_size / model.ema_cluster_size.sum()
    # Threshold is relative to each cluster's TARGET share, not a flat 1/K, so it
    # stays correct under a non-uniform (Zipf) prior. Reduces to threshold/K when
    # the target is uniform, i.e. unchanged for every existing config.
    dead_mask = u < (threshold_factor * model.target_usage())
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
