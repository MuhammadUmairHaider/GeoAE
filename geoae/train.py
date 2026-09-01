"""
Training loop for GeoAE.

Usage:
    python train.py                              # uses configs/default.yaml
    python train.py --config configs/small_test.yaml
    python train.py --config configs/small_test.yaml --no_wandb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from geoae.config import Config
from geoae.data import ActivationBuffer, ShuffledActivationLoader
from geoae.model import GeoAE
from geoae.losses import total_loss
from geoae.seeding import seed_everything
from geoae import diagnostics as diag
from geoae.train_common import (   # noqa: F401 — re-exported for compatibility
    tau_schedule, lambda_schedule, geometry_schedule, zipf_alpha_schedule,
    rotate_checkpoints, reinit_dead_clusters,
    build_model, make_optimizer, init_wandb, add_override_args, apply_overrides,
    CheckpointTracker,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: GeoAE,
    opt: torch.optim.Optimizer,
    epoch: int,
    step: int,
    tau: float,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    val_mse: float,
    cfg: Config,
) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "epoch": epoch,
        "step": step,
        "tau": tau,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "val_mse": val_mse,
        "config": cfg.to_dict(),
    }, path)


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model: GeoAE,
    val_loader: ShuffledActivationLoader,
    device: torch.device,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Returns (val_mse, val_fve)."""
    model.eval()
    mse_sum = fve_sum = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch.to(device)
        out = model(x)
        mse = (out.x_hat - x).pow(2).mean().item()
        var_x = x.var(unbiased=False).item()
        fve = 1.0 - mse / (var_x + 1e-8)
        mse_sum += mse
        fve_sum += fve
        n += 1
    model.train()
    return mse_sum / max(n, 1), fve_sum / max(n, 1)


# ---------------------------------------------------------------------------
# Supervised centroid initialisation (per-class mean latent)
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_class_latent_means(
    model: GeoAE,
    act_dir: Path,
    target_layer: int,
    val_frac: float,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    device: torch.device,
    max_per_class: int | None = None,
    encode_batch: int = 4096,
) -> torch.Tensor:
    """
    Per-class mean of encoded latents over the TRAIN split. Returns (K, L).

    For each class k, encode its training activations and average in LATENT
    space (the space clustering operates in). `max_per_class` caps the number
    of samples per class (semisup uses 500); None uses ALL of them (the
    "actual" / exact class mean). Requires labels_train.npy beside the .npy.
    """
    raw_acts = np.load(str(act_dir / f"layer_{target_layer}.npy"), mmap_mode="r")
    labels = np.load(str(act_dir / "labels_train.npy"))
    val_start = int(len(raw_acts) * (1.0 - val_frac))
    train_labels = labels[:val_start]
    K = model.n_clusters

    means = []
    for k in range(K):
        idxs = np.where(train_labels == k)[0]
        if len(idxs) == 0:
            raise ValueError(f"No training samples for class {k}")
        if max_per_class is not None:
            idxs = idxs[:max_per_class]
        # Accumulate the latent sum in chunks so a large class never has all its
        # activations resident at once; the running mean is exact either way.
        z_sum = None
        n = 0
        for s in range(0, len(idxs), encode_batch):
            chunk = raw_acts[idxs[s:s + encode_batch]].astype(np.float32)
            chunk_norm = (chunk - train_mean) / train_std
            z = model.encoder(torch.from_numpy(chunk_norm).to(device))
            z_sum = z.sum(0) if z_sum is None else z_sum + z.sum(0)
            n += z.shape[0]
        means.append(z_sum / n)
    return torch.stack(means)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Config, use_wandb: bool = True, no_renorm: bool = False,
          max_train_rows: int | None = None,
          no_sinkhorn: bool = False) -> None:
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    logger = init_wandb(cfg, use_wandb, run_name=f"geosep_seed{cfg.train.seed}",
                        tag="train")

    # Data
    from geoae.paths import resolve_path
    act_dir = resolve_path(cfg.data.activations_dir)

    train_buf = ActivationBuffer(act_dir, cfg.data.target_layer,
                                 val_frac=cfg.data.val_frac, split="train",
                                 max_train_rows=max_train_rows)
    val_buf   = ActivationBuffer(act_dir, cfg.data.target_layer,
                                 val_frac=cfg.data.val_frac, split="val",
                                 norm_cache=act_dir / f"norm_params_layer{cfg.data.target_layer}.npz")
    train_loader = ShuffledActivationLoader(train_buf, batch_size=cfg.data.batch_size,
                                            num_workers=4, pin_memory=True)
    val_loader   = ShuffledActivationLoader(val_buf, batch_size=cfg.data.batch_size,
                                            num_workers=2, pin_memory=True)
    print(f"[train] Train tokens: {len(train_buf):,}  Val tokens: {len(val_buf):,}")

    # Model
    model = build_model(cfg, device, no_sinkhorn=no_sinkhorn)
    print(f"[train] Encoder: {cfg.model.nonlinearity}  "
          f"latent_dim={cfg.model.latent_dim}  n_clusters={cfg.model.n_clusters}"
          f"  metric={model.metric}"
          f"  sinkhorn={'on' if model.use_sinkhorn else 'off'}")
    print(f"[train] Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Centroids stay at zero until the clustering phase begins; they're then
    # initialised from already-trained latents (see loop below).
    init_mode = cfg.train.centroid_init
    print(f"[train] Centroid init deferred until epoch {cfg.train.clustering_start_epoch}"
          f" ({init_mode} from trained latents)")

    opt = make_optimizer(model, cfg)

    ckpt_dir = resolve_path(cfg.train.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tracker = CheckpointTracker(ckpt_dir, cfg.train.keep_checkpoints)

    global_step = 0
    Q_history: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, cfg.train.n_epochs + 1):
        # Centroid init at the start of the clustering phase
        if (not bool(model.centroids_initialized.item())
                and epoch >= cfg.train.clustering_start_epoch):
            if init_mode in ("semisup", "class_means"):
                # Supervised: each centroid = per-class mean of encoded latents.
                #   semisup     — capped at 500 samples/class (fast approximation)
                #   class_means — ALL training samples/class (exact "actual" mean)
                cap = 500 if init_mode == "semisup" else None
                means = per_class_latent_means(
                    model, act_dir, cfg.data.target_layer, cfg.data.val_frac,
                    train_buf.mean, train_buf.std, device, max_per_class=cap,
                )
                model.init_centroids_from_class_means(means)
                detail = "≤500/class" if cap else "full train split"
                print(f"[train] Epoch {epoch}: {init_mode} init of {model.n_clusters} "
                      f"centroids (per-class mean latent, {detail})")
            else:
                # Default: k-means++ from unlabeled batch
                init_batch = next(iter(train_loader)).to(device)
                with torch.no_grad():
                    init_z = model.encoder(init_batch)
                model.init_centroids_kmeans_plus_plus(init_z, seed=cfg.train.seed)
                print(f"[train] Epoch {epoch}: k-means++ init of {model.n_clusters} "
                      f"centroids from {len(init_batch)} trained latents")

        # Update schedule
        tau = tau_schedule(epoch, cfg)
        model.tau = tau
        lam_c, lam_s = lambda_schedule(epoch, cfg)
        lam_v, lam_cv, lam_u = geometry_schedule(epoch, cfg)
        model.zipf_alpha = zipf_alpha_schedule(epoch, cfg)

        print(f"\n[train] Epoch {epoch}/{cfg.train.n_epochs} | tau={tau:.3f} "
              f"| λ_c={lam_c:.3f} λ_s={lam_s:.4f} "
              f"| λ_var={lam_v:.3f} λ_cov={lam_cv:.4f}")

        epoch_start = time.time()

        for batch in train_loader:
            x = batch.to(device)

            # Forward
            out = model(x)

            # Losses
            losses = total_loss(
                x=x,
                x_hat=out.x_hat,
                z=out.z,
                centroids=model.centroids.detach(),
                Q=out.Q,
                lambda_cluster=lam_c,
                lambda_sep=lam_s,
                metric=model.metric,
                lambda_var=lam_v,
                lambda_cov=lam_cv,
                lambda_unif=lam_u,
                sep_mode=getattr(cfg.loss, "sep_mode", "median"),
                sep_margin=getattr(cfg.loss, "sep_margin", 2.0),
            )

            # Backward
            opt.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

            # Post-step: renorm decoder + EMA centroid update
            if not no_renorm:
                model.post_step()
            if bool(model.centroids_initialized.item()):
                model.update_centroids_ema(out.z.detach(), out.Q.detach())

            # Track Q history for usage diagnostics
            Q_history.append(out.Q.detach().cpu())
            if len(Q_history) > 200:
                Q_history = Q_history[-200:]

            global_step += 1

            # ----------------------------------------------------------
            # Diagnostics
            # ----------------------------------------------------------
            if global_step % cfg.train.diag_every == 0:
                val_mse, val_fve = run_validation(model, val_loader, device)
                run_sil = (global_step % (cfg.train.diag_every * 10) == 0)
                metrics = diag.compute_all(
                    z=out.z.detach(),
                    Q=out.Q.detach(),
                    centroids=model.centroids.detach(),
                    train_mse=losses["recon"].item(),
                    val_mse=val_mse,
                    train_fve=losses["fve"].item(),
                    val_fve=val_fve,
                    Q_history=Q_history,
                    run_silhouette=run_sil,
                )
                metrics["train/loss"] = losses["loss"].item()
                metrics["train/cluster_loss"] = losses["cluster"].item()
                metrics["train/sep_loss"] = losses["sep"].item()
                metrics["train/tau"] = tau
                metrics["step"] = global_step
                # Only present when the corresponding lambda is > 0.
                extra = ""
                for key in ("var", "cov", "unif"):
                    if key in losses:
                        metrics[f"train/{key}_loss"] = losses[key].item()
                        extra += f" | {key} {losses[key].item():.4f}"

                print(
                    f"  step {global_step:6d} | "
                    f"loss {metrics['train/loss']:.4f} | "
                    f"recon {metrics['train/mse']:.4f} | "
                    f"clus {metrics['train/cluster_loss']:.4f} | "
                    f"sep {metrics['train/sep_loss']:.4f}{extra} | "
                    f"val_mse {val_mse:.4f} | "
                    f"fve {metrics['train/fve']:.3f} | "
                    f"eff_K {metrics['cluster/effective_k']}/{model.n_clusters} | "
                    f"dying {metrics['cluster/dying']}"
                )

                if logger is not None:
                    logger.log(metrics, step=global_step)

            # ----------------------------------------------------------
            # Dead cluster reinit
            # ----------------------------------------------------------
            if global_step % cfg.train.reinit_every == 0:
                n_reinit = reinit_dead_clusters(
                    model, out.z.detach(), x.detach(), out.x_hat.detach()
                )
                if n_reinit > 0:
                    print(f"  [reinit] step {global_step}: reinitialised {n_reinit} dead clusters")
                    if logger is not None:
                        logger.log({"cluster/reinit_count": n_reinit}, step=global_step)

        # ------------------------------------------------------------------
        # End-of-epoch checkpoint
        # ------------------------------------------------------------------
        if epoch % cfg.train.save_every == 0:
            val_mse, val_fve = run_validation(model, val_loader, device)
            ckpt_path = ckpt_dir / f"step_{global_step:07d}.pt"
            save_checkpoint(
                ckpt_path, model, opt, epoch, global_step, tau,
                train_buf.mean, train_buf.std, val_mse, cfg,
            )
            print(f"[train] Saved checkpoint: {ckpt_path.name}  val_mse={val_mse:.5f}")

            if tracker.finish_epoch(ckpt_path, val_mse):
                print(f"[train] New best val_mse={tracker.best_score:.5f} → best_val.pt")

        epoch_elapsed = time.time() - epoch_start
        print(f"[train] Epoch {epoch} done in {epoch_elapsed/60:.1f} min")

    if logger is not None:
        logger.finish()
    print(f"\n[train] Training complete. Best val_mse = {tracker.best_score:.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--metric", default=None, choices=["euclidean", "cosine"],
                        help="Clustering geometry: euclidean (magnitude) or cosine (directional)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--max_train_rows", type=int, default=None,
                        help="Cap the train split to the first N rows. Costs training "
                             "data, but shrinks the working set so it stays resident in "
                             "page cache: cold random reads on a >RAM dump run ~300 "
                             "rows/s vs ~108k warm, so an over-RAM split is I/O bound.")
    parser.add_argument("--centroid_init", default=None,
                        choices=["kmeans++", "semisup", "class_means"],
                        help="Centroid init: kmeans++ (unsup), semisup (per-class mean "
                             "latent, ≤500/class), class_means (exact per-class mean, full split)")
    add_override_args(parser)
    args = parser.parse_args()

    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        from geoae.paths import default_config
        default = default_config()
        cfg = Config.from_yaml(default) if default.exists() else Config()

    apply_overrides(cfg, args)

    train(cfg, use_wandb=not args.no_wandb, no_renorm=args.no_renorm,
          no_sinkhorn=args.no_sinkhorn, max_train_rows=args.max_train_rows)


if __name__ == "__main__":
    main()
