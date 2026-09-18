"""
Scaffolding shared by all three trainers (geoae.train, geoae.e2e.train,
geoae.e2e.train_stream): schedules, model/optimizer construction, wandb init,
CLI overrides, checkpoint rotation and best-val tracking, dead-cluster reinit.

The training loops themselves stay in their own modules — their loss paths
genuinely differ — but everything that must stay in lockstep lives here so the
pipelines can't drift apart.
"""
from __future__ import annotations

import os
import shutil
import zipfile
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
        dist_scale=cfg.model.dist_scale,
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
    ap.add_argument("--nonlinearity", default=None, choices=["linear", "relu", "gelu", "tanh"],
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


def atomic_torch_save(obj, path: Path) -> None:
    """
    torch.save to a temp file beside `path`, then atomically rename over it.

    A plain torch.save that fails mid-write (disk full, killed process) leaves a
    truncated file under the real name. On 2026-09-18 a full disk left an 8 MB
    step_0008804.pt among 251 MB siblings, which `--resume latest` would then pick.
    With this, a failed save leaves the previous files untouched and no partial.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_copy(src: Path, dst: Path) -> None:
    """copy2 to a temp file, then rename — never leaves `dst` half-written."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def checkpoint_is_readable(path: Path) -> bool:
    """Cheap integrity check: torch checkpoints are zip archives, and a truncated
    one has no end-of-central-directory record. Does not load any tensors."""
    try:
        with zipfile.ZipFile(path) as z:
            return len(z.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def ensure_free_space(ckpt_dir: Path, need_bytes: int, what: str) -> None:
    """Stop with an actionable message instead of a cryptic iostream error."""
    free = shutil.disk_usage(ckpt_dir).free
    if free < need_bytes:
        raise OSError(
            f"[checkpoint] only {free / 1e9:.2f} GB free on the disk holding {ckpt_dir}, "
            f"need ~{need_bytes / 1e9:.2f} GB to save {what}. Nothing was written. "
            f"Free space, then continue with --resume latest."
        )


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

    def restore_best(self, score_key: str) -> bool:
        """
        On resume, seed the best score from the best_val.pt already on disk.

        The tracker starts at +inf, so without this the FIRST epoch after a
        resume always counts as "best" and overwrites best_val.pt — usually with
        a worse checkpoint, since the schedule keeps annealing and the val metric
        typically rises for a while. Rotation then ages out the older step_*.pt
        files and the good state is gone. Returns True if a score was restored.
        """
        if not self.best_target.exists():
            return False
        prev = torch.load(str(self.best_target), map_location="cpu", weights_only=False)
        self.record_best(float(prev[score_key]))
        print(f"[resume] best_val.pt guarded at {score_key}={self.best_score:.5f} "
              f"(epoch {prev.get('epoch', '?')}) — only a better score replaces it")
        return True

    def finish_epoch(self, ckpt_path: Path, score: float) -> bool:
        """Promote ckpt_path to best_val.pt if score improved, then rotate.
        Returns True if this checkpoint became the new best."""
        is_best = self.is_better(score)
        if is_best:
            # Copying straight onto best_val.pt would destroy the current best if the
            # disk filled mid-copy — the one file a run can't afford to lose.
            ensure_free_space(self.ckpt_dir, Path(ckpt_path).stat().st_size, "best_val.pt")
            atomic_copy(ckpt_path, self.best_target)
            self.record_best(score)
        protected = {ckpt_path}
        if self.best_path is not None:
            protected.add(self.best_path)
        rotate_checkpoints(self.ckpt_dir, self.keep, protected)
        return is_best


# ---------------------------------------------------------------------------
# Resume (shared by the MSE and e2e trainers)
# ---------------------------------------------------------------------------

def load_resume(path, model, opt, norm_mean, norm_std, device) -> tuple[int, int]:
    """
    Restore model + optimizer from a checkpoint; return (start_epoch, global_step).

    The checkpoint carries everything needed (model_state, opt_state, epoch,
    step, tau, norm stats), so a stopped run continues instead of restarting
    from scratch. The centroids and the `centroids_initialized` flag live in
    model_state, so centroid init is correctly skipped on resume.

    Normalisation is asserted to match: the encoder was trained against these
    exact mean/std, so silently resuming under different stats would corrupt
    the model without any visible error.
    """
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "opt_state" in ckpt:
        opt.load_state_dict(ckpt["opt_state"])
    else:
        print("[resume] ! checkpoint has no opt_state — Adam moments restart cold")

    for name, saved, cur in (("mean", ckpt["norm_mean"], norm_mean),
                             ("std", ckpt["norm_std"], norm_std)):
        s = torch.as_tensor(saved, dtype=torch.float32, device=device)
        c = torch.as_tensor(cur, dtype=torch.float32, device=device)
        if not torch.allclose(s, c, rtol=1e-4, atol=1e-6):
            raise ValueError(
                f"[resume] norm {name} in {path} does not match the current "
                f"activations (max diff {(s - c).abs().max():.3e}). "
                "The checkpoint was trained on a different dump — refusing to resume."
            )

    start_epoch = int(ckpt["epoch"]) + 1
    # Continue the global step counter too: it names checkpoints
    # (step_<n>.pt) and is the wandb x-axis, which rejects steps that go
    # backwards. Restarting it at 0 would interleave new files among the old.
    step = int(ckpt.get("step", 0))
    val = " ".join(f"{k} {ckpt[k]:.5f}" for k in ("val_kl", "val_mse") if ckpt.get(k) is not None)
    print(f"[resume] {path}")
    print(f"[resume] epoch {ckpt['epoch']} step {step} "
          f"tau {ckpt.get('tau', float('nan')):.3f} {val} "
          f"| centroids_initialized={bool(model.centroids_initialized.item())}")
    print(f"[resume] resuming at epoch {start_epoch}, step counter continues from {step}")
    return start_epoch, step


def resolve_resume(resume: str, ckpt_dir: Path) -> Path:
    """
    Resolve --resume to a checkpoint path, refusing resumes that would destroy data.

    "latest" picks the highest-step step_*.pt in ckpt_dir. Any other value is a
    path. Checkpoints are named by global step, so resuming from an EARLIER point
    than the newest file (typically best_val.pt, which on clustering runs is often
    a pre-clustering epoch because val loss rises once the cluster terms engage)
    would re-save step_<n>.pt names that already exist and silently overwrite
    every later epoch of the original run. That is refused; point
    checkpoints_dir elsewhere to branch from an earlier state.
    """
    steps = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if resume == "latest":
        if not steps:
            raise FileNotFoundError(f"[resume] --resume latest: no step_*.pt in {ckpt_dir}")
        # Newest READABLE checkpoint. A save interrupted before atomic writes existed
        # (or by a hard kill) can leave a truncated newest file.
        for cand in reversed(steps):
            if checkpoint_is_readable(cand):
                return cand
            print(f"[resume] ! skipping {cand.name}: truncated/corrupt "
                  f"({cand.stat().st_size / 1e6:.1f} MB), likely an interrupted save")
        raise FileNotFoundError(f"[resume] no readable step_*.pt in {ckpt_dir}")
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"[resume] checkpoint not found: {path}")
    if steps and path.resolve().parent == ckpt_dir.resolve():
        step = int(torch.load(str(path), map_location="cpu", weights_only=False).get("step", 0))
        later = [p.name for p in steps if int(p.stem.split("_")[1]) > step]
        if later:
            raise ValueError(
                f"[resume] {path.name} is at step {step}, but {len(later)} later checkpoints "
                f"exist ({later[0]} .. {later[-1]}); resuming here would overwrite them. "
                f"Use --resume latest, or set checkpoints_dir to a new directory to branch."
            )
    return path


# ---------------------------------------------------------------------------
# Centroid-init pool
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_init_pool(model: GeoAE, train_loader, device, n_rows: int) -> torch.Tensor:
    """
    Encode up to `n_rows` training activations for centroid init.

    A single batch is only 32,768 rows at b32k, i.e. ~16 points per centroid at
    K=2000 — thin for a density estimate, which is why this accumulates rather
    than taking `next(iter(train_loader))`. Same accumulate-until-enough loop as
    the e2e k-means++ path (e2e/train.py), generalised to a row budget.
    """
    zs, have = [], 0
    for batch in train_loader:
        x = batch["x"] if isinstance(batch, dict) else batch
        z = model.encoder(x.to(device))[: n_rows - have]
        zs.append(z)
        have += len(z)
        if have >= n_rows:
            break
    if not zs:
        raise ValueError("train_loader yielded no batches for centroid init")
    return torch.cat(zs, dim=0)


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
    mode: str = "loss",
    anchor_pool=None,
    mean_k: int = 5,
    peak_pool: int = 8192,
    peak_refine_k: int = 8,
    peak_min_sep_frac: float = 0.25,
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
        return (0, 0) if mode == "anchor" else 0

    if mode == "anchor" and anchor_pool is not None:
        # PRIORITY 1: unused LABELLED points. Exhaust the supervised signal
        # before falling back to any heuristic — a labelled example is a known
        # mode, whereas both D^2 sampling and highest-reconstruction-loss pick
        # whatever is furthest out, i.e. outliers.
        live = (~dead_mask).nonzero(as_tuple=True)[0]
        z_a, keys = anchor_pool.take(model, n_dead, model.centroids[live], mean_k=mean_k)
        n_from_anchor = 0 if z_a is None else len(z_a)
        if n_from_anchor < n_dead:
            # PRIORITY 2: pool exhausted -> D^2 sampling over the batch, the
            # k-means++ rule, applied probabilistically rather than greedily.
            need = n_dead - n_from_anchor
            d2 = torch.cdist(z_batch, model.centroids[live]).min(1).values.pow(2)
            p = d2 / d2.sum().clamp_min(1e-12)
            extra = z_batch[torch.multinomial(p, min(need, len(z_batch)))]
            replacement_latents = extra if z_a is None else torch.cat([z_a, extra])
        else:
            replacement_latents = z_a
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        for i, k in enumerate(dead_indices[:len(replacement_latents)]):
            model.centroids[k].copy_(replacement_latents[i])
            model.ema_cluster_size[k] = 1.0
        return n_dead, n_from_anchor

    replacement_latents = None
    if mode == "peaks":
        # Like "density" below, but delta is measured against DENSER points
        # rather than against the live centroids, so a dead cluster cannot be
        # reseeded onto an outlier that merely happens to be far from everything.
        # Pooled down to `peak_pool` rows: the delta scan is O(N^2) and this
        # fires every reinit_every steps, unlike the one-off init.
        from geoae.seeded_init import density_peaks_select
        live = (~dead_mask).nonzero(as_tuple=True)[0]
        pool = z_batch[:peak_pool]
        replacement_latents, _ = density_peaks_select(
            pool, min(n_dead, len(pool)), C_init=model.centroids[live], seed=0,
            min_sep_frac=peak_min_sep_frac, refine_k=peak_refine_k,
        )
    elif mode == "density":
        # Highest-reconstruction-loss samples are close to a DEFINITION of an
        # outlier, so the default rule reseeds dead clusters onto exactly the
        # points least likely to anchor a mode -- which is why churn never
        # settles. Score by local density x distance-to-nearest-live-centroid
        # instead: dense enough to be a real mode, far enough to be a new one.
        from geoae.seeded_init import local_density
        with torch.no_grad():
            dens = local_density(z_batch, seed=0)
            dens = dens / dens.median().clamp_min(1e-9)
            live = (~dead_mask).nonzero(as_tuple=True)[0]
            d2 = torch.cdist(z_batch, model.centroids[live]).min(1).values.pow(2)
            _, top_idx = (dens * d2).topk(min(n_dead, len(z_batch)))
    else:
        # Rank batch samples by reconstruction loss (highest = most informative)
        per_sample_loss = (x_hat_batch - x_batch).pow(2).mean(dim=1)  # (B,)
        _, top_idx = per_sample_loss.topk(min(n_dead, len(per_sample_loss)))
    if replacement_latents is None:
        replacement_latents = z_batch[top_idx]  # (n_dead, L)

    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    for i, k in enumerate(dead_indices[:len(replacement_latents)]):
        model.centroids[k].copy_(replacement_latents[i])
        model.ema_cluster_size[k] = 1.0  # reset EMA count

    return n_dead
