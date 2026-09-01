"""
End-to-end KL training loop for GeoAE.

Same architecture, schedule, centroid init, EMA updates and diagnostics as the
base `train.py`, with ONE change: the faithfulness signal is the KL divergence
between the original and reconstructed last-token logits, not MSE in activation
space. The frozen LLM is loaded and kept in the loop so the reconstruction can
be mapped to logits (see logits.LogitsComputer).

Pure KL from epoch 1: KL is active every epoch; the cluster/sep terms still
phase in via the existing schedule (lambda_schedule), so the "recon-only" warmup
epochs are now "KL-only" epochs.

Usage:
    python e2e/geoae.e2e.train --config e2e/configs/Qwen3.5-9B/layer31/unprompted_last_gelu.yaml --no_wandb
"""
from __future__ import annotations

import argparse
import time



import torch
import torch.nn as nn

from geoae.config import Config
from geoae import diagnostics as diag

from geoae.e2e.logits import LogitsComputer
from geoae.e2e.losses import total_loss_e2e, kl_loss
from geoae.e2e.data import E2EBuffer, make_loader

# Reuse the base-pipeline helpers verbatim.
from geoae.seeding import seed_everything
from geoae.train import per_class_latent_means
from geoae.train_common import (
    tau_schedule, lambda_schedule, geometry_schedule, zipf_alpha_schedule,
    reinit_dead_clusters,
    build_model, make_optimizer, init_wandb, add_override_args, apply_overrides,
    CheckpointTracker,
)


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #

def save_checkpoint(path, model, opt, epoch, step, tau,
                    norm_mean, norm_std, val_kl, val_mse, cfg) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "epoch": epoch,
        "step": step,
        "tau": tau,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "val_kl": val_kl,
        "val_mse": val_mse,  # kept for compatibility with evaluate.py loaders
        "config": cfg.to_dict(),
    }, path)


# --------------------------------------------------------------------------- #
# Frozen LLM
# --------------------------------------------------------------------------- #

def load_frozen_lm(model_name: str, device: torch.device):
    print(f"[train-e2e] Loading frozen LLM: {model_name}")
    from geoae.checkpoint import load_lm
    return load_lm(model_name, device=device)


# --------------------------------------------------------------------------- #
# Validation + sanity
# --------------------------------------------------------------------------- #

def compute_teacher(batch, x_raw, lc, onfly, device):
    """
    Teacher last-token logits.
      onfly  : head(norm(x_raw)) recomputed in-loop (last layer; no cache on disk)
      cached : the precomputed teacher logits served by E2EBuffer
    Either way the teacher is a fixed target (kl_loss detaches it internally).
    """
    if onfly:
        with torch.no_grad():
            return lc.head_logits(x_raw)
    return batch["teacher"].to(device)


@torch.no_grad()
def run_validation(model, lc, val_loader, mean_t, std_t, device, onfly,
                   max_batches: int = 20) -> tuple[float, float, float]:
    """Returns (val_kl, val_mse, val_fve)."""
    model.eval()
    kl_sum = mse_sum = fve_sum = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch["x"].to(device)
        out = model(x)
        recon_raw = out.x_hat * std_t + mean_t
        x_raw = (x * std_t + mean_t) if onfly else None
        teacher = compute_teacher(batch, x_raw, lc, onfly, device)
        if lc.needs_input_ids:
            student = lc.student_logits(recon_raw,
                                        batch["input_ids"].to(device),
                                        batch["attn"].to(device))
        else:
            student = lc.head_logits(recon_raw)
        kl_sum += kl_loss(teacher, student).item()
        mse = (out.x_hat - x).pow(2).mean().item()
        var_x = x.var(unbiased=False).item()
        mse_sum += mse
        fve_sum += 1.0 - mse / (var_x + 1e-8)
        n += 1
    model.train()
    n = max(n, 1)
    return kl_sum / n, mse_sum / n, fve_sum / n


@torch.no_grad()
def sanity_check_logits(model, lc, loader, mean_t, std_t, device, onfly) -> float:
    """
    Feed the ORIGINAL activation through the logits computer and confirm it
    reproduces the cached teacher logits (last-layer: exact head path;
    intermediate: identity splice). Validates alignment, padding and dtype
    before any training happens.

    For teacher_mode="onfly" there is no cache to validate (teacher == this same
    head path by construction), so we just confirm the head path runs and shapes.
    """
    batch = next(iter(loader))
    x = batch["x"].to(device)
    raw = x * std_t + mean_t  # denorm(normalised x) == raw activation
    if lc.needs_input_ids:
        student = lc.student_logits(raw,
                                    batch["input_ids"].to(device),
                                    batch["attn"].to(device))
    else:
        student = lc.head_logits(raw)
    if onfly:
        print(f"[train-e2e] logits sanity (onfly): head path OK, "
              f"logits shape={tuple(student.shape)} (teacher = same path, no cache)")
        return 0.0
    teacher = batch["teacher"].to(device)
    max_diff = (student.float() - teacher.float()).abs().max().item()
    kl = kl_loss(teacher, student).item()
    print(f"[train-e2e] logits sanity: max|student(orig) - teacher| = {max_diff:.3e}"
          f" | KL = {kl:.3e}")
    if kl > 1e-2:
        print("[train-e2e] WARNING: identity KL is not ~0 — the logits path or "
              "cache alignment is suspect. Investigate before trusting results.")
    return kl


# --------------------------------------------------------------------------- #
# Centroid init (ported from train.py, adapted to the dict-batch loader)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def init_centroids(model, cfg, train_buf, train_loader, act_dir, device, init_mode):
    if init_mode in ("semisup", "class_means"):
        # Supervised: centroid = per-class mean of encoded latents (shared with
        # the MSE pipeline). semisup caps at 500/class; class_means uses ALL.
        cap = getattr(cfg.train, "semisup_cap", 500) if init_mode == "semisup" else None
        means = per_class_latent_means(
            model, act_dir, cfg.data.target_layer, cfg.data.val_frac,
            train_buf.mean, train_buf.std, device, max_per_class=cap,
        )
        model.init_centroids_from_class_means(means)
        detail = f"≤{cap}/class" if cap else "full train split"
        print(f"[train-e2e] {init_mode} init of {model.n_clusters} centroids "
              f"(per-class mean latent, {detail})")
    else:
        # k-means++ needs ≥ K samples. A single batch can be smaller than
        # n_clusters, so accumulate batches until we have enough latents.
        need = model.n_clusters
        zs = []
        n_have = 0
        for batch in train_loader:
            z = model.encoder(batch["x"].to(device))
            zs.append(z)
            n_have += z.shape[0]
            if n_have >= need:
                break
        if n_have < need:
            raise ValueError(
                f"k-means++ needs ≥ K={need} latents but the train split only "
                f"yielded {n_have}. Reduce --n_clusters or add more data."
            )
        init_z = torch.cat(zs, dim=0)
        model.init_centroids_kmeans_plus_plus(init_z, seed=cfg.train.seed)
        print(f"[train-e2e] k-means++ init of {model.n_clusters} centroids "
              f"from {len(init_z)} latents")


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def load_resume(path, model, opt, norm_mean, norm_std, device) -> int:
    """
    Restore model + optimizer from a checkpoint and return the epoch to start at.

    The checkpoint carries everything needed (model_state, opt_state, epoch,
    step, tau, norm stats), so a stopped run continues instead of restarting
    from scratch. The centroids and the `centroids_initialized` flag live in
    model_state, so k-means++ init is correctly skipped on resume.

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
        if not torch.allclose(s, cur.float(), rtol=1e-4, atol=1e-6):
            raise ValueError(
                f"[resume] norm {name} in {path} does not match the current "
                f"activations (max diff {(s - cur.float()).abs().max():.3e}). "
                "The checkpoint was trained on a different dump — refusing to resume."
            )

    start_epoch = int(ckpt["epoch"]) + 1
    # Continue the global step counter too: it names checkpoints
    # (step_<n>.pt) and is the wandb x-axis, which rejects steps that go
    # backwards. Restarting it at 0 would interleave new files among the old.
    step = int(ckpt.get("step", 0))
    print(f"[resume] {path}")
    print(f"[resume] epoch {ckpt['epoch']} step {step} "
          f"tau {ckpt.get('tau', float('nan')):.3f} val_kl {ckpt.get('val_kl', float('nan')):.5f} "
          f"| centroids_initialized={bool(model.centroids_initialized.item())}")
    print(f"[resume] resuming at epoch {start_epoch}, step counter continues from {step}")
    return start_epoch, step


def train(cfg: Config, use_wandb: bool = True, no_renorm: bool = False,
          no_sinkhorn: bool = False, resume: str | None = None,
          max_train_rows: int | None = None) -> None:
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train-e2e] Device: {device}")

    logger = init_wandb(cfg, use_wandb, run_name=f"geosep_e2e_seed{cfg.train.seed}",
                        tag="train-e2e")

    # ---- frozen LLM + logits computer ----
    lm = load_frozen_lm(cfg.extraction.model_name, device)
    lc = LogitsComputer(lm, cfg.data.target_layer)
    print(f"[train-e2e] target layer {cfg.data.target_layer} of {lc.n_layers} "
          f"(is_last={lc.is_last}); "
          f"{'head-only (lm_head∘norm)' if lc.is_last else 'full-forward + splice'}")

    # ---- teacher mode ----
    onfly = cfg.train.teacher_mode == "onfly"
    if onfly and not lc.is_last:
        raise ValueError(
            "teacher_mode='onfly' is only supported at the last layer (head path). "
            "Intermediate layers need cached per-token teacher logits + input_ids."
        )
    print(f"[train-e2e] teacher_mode="
          f"{'onfly — head(norm(x)) computed in-loop, no cache' if onfly else 'cached'}")

    # ---- data ----
    from geoae.paths import resolve_path
    act_dir = resolve_path(cfg.data.activations_dir)

    train_buf = E2EBuffer(act_dir, cfg.data.target_layer, val_frac=cfg.data.val_frac,
                          split="train", needs_input_ids=lc.needs_input_ids,
                          load_teacher=not onfly, max_train_rows=max_train_rows)
    val_buf = E2EBuffer(act_dir, cfg.data.target_layer, val_frac=cfg.data.val_frac,
                        split="val", needs_input_ids=lc.needs_input_ids,
                        load_teacher=not onfly,
                        norm_cache=act_dir / f"norm_params_layer{cfg.data.target_layer}.npz")
    train_loader = make_loader(train_buf, cfg.data.batch_size, shuffle=True, num_workers=4,
                               seed=cfg.train.seed)
    val_loader = make_loader(val_buf, cfg.data.batch_size, shuffle=True, num_workers=2,
                             drop_last=False, seed=cfg.train.seed)
    print(f"[train-e2e] Train docs: {len(train_buf):,}  Val docs: {len(val_buf):,}")

    mean_t = torch.from_numpy(train_buf.mean).to(device)
    std_t = torch.from_numpy(train_buf.std).to(device)

    # ---- model ----
    model = build_model(cfg, device, no_sinkhorn=no_sinkhorn)
    print(f"[train-e2e] Encoder={cfg.model.nonlinearity} latent_dim={cfg.model.latent_dim} "
          f"K={cfg.model.n_clusters} metric={model.metric} "
          f"sinkhorn={'on' if model.use_sinkhorn else 'off'}")
    print(f"[train-e2e] lr={cfg.train.lr:g} grad_clip={cfg.train.grad_clip:g} "
          f"lambda_mse={cfg.loss.lambda_mse:g} "
          f"({'MSE anchor ON' if cfg.loss.lambda_mse > 0 else 'KL-only, no recon anchor'})")

    # ---- validate the logits path BEFORE training ----
    sanity_check_logits(model, lc, train_loader, mean_t, std_t, device, onfly)

    opt = make_optimizer(model, cfg)

    ckpt_dir = resolve_path(cfg.train.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tracker = CheckpointTracker(ckpt_dir, cfg.train.keep_checkpoints)

    init_mode = cfg.train.centroid_init
    global_step = 0
    Q_history: list[torch.Tensor] = []

    start_epoch = 1
    if resume:
        start_epoch, global_step = load_resume(resume, model, opt, mean_t, std_t, device)
        if start_epoch > cfg.train.n_epochs:
            raise ValueError(
                f"[resume] checkpoint is at epoch {start_epoch - 1} but "
                f"train.n_epochs={cfg.train.n_epochs} — nothing left to run. "
                "Raise n_epochs (or lower it to compress the tau anneal)."
            )
        # Seed the best-val score from the existing best_val.pt. Without this the
        # tracker starts at +inf, so the FIRST epoch after resume always counts as
        # "best" and overwrites best_val.pt — with a worse checkpoint, since tau
        # steps down at resume and val_kl typically rises for a while. Rotation
        # would then age out the older step_*.pt files and the good state is gone.
        best_path = ckpt_dir / "best_val.pt"
        if best_path.exists():
            prev = torch.load(str(best_path), map_location="cpu", weights_only=False)
            tracker.record_best(float(prev["val_kl"]))
            print(f"[resume] best_val.pt guarded at val_kl={tracker.best_score:.5f} "
                  f"(epoch {prev.get('epoch', '?')}) — only a better score replaces it")

    for epoch in range(start_epoch, cfg.train.n_epochs + 1):
        if (not bool(model.centroids_initialized.item())
                and epoch >= cfg.train.clustering_start_epoch):
            init_centroids(model, cfg, train_buf, train_loader, act_dir, device, init_mode)

        tau = tau_schedule(epoch, cfg)
        model.tau = tau
        lam_c, lam_s = lambda_schedule(epoch, cfg)
        lam_v, lam_cv, lam_u = geometry_schedule(epoch, cfg)
        model.zipf_alpha = zipf_alpha_schedule(epoch, cfg)
        print(f"\n[train-e2e] Epoch {epoch}/{cfg.train.n_epochs} | tau={tau:.3f} "
              f"| λ_c={lam_c:.3f} λ_s={lam_s:.4f} "
              f"| λ_var={lam_v:.3f} λ_cov={lam_cv:.4f}")
        epoch_start = time.time()

        for batch in train_loader:
            x = batch["x"].to(device)

            out = model(x)
            recon_raw = out.x_hat * std_t + mean_t
            x_raw = (x * std_t + mean_t) if onfly else None
            teacher = compute_teacher(batch, x_raw, lc, onfly, device)
            if lc.needs_input_ids:
                student = lc.student_logits(recon_raw,
                                            batch["input_ids"].to(device),
                                            batch["attn"].to(device))
            else:
                student = lc.head_logits(recon_raw)

            losses = total_loss_e2e(
                teacher_logits=teacher, student_logits=student,
                x=x, x_hat=out.x_hat, z=out.z,
                centroids=model.centroids.detach(), Q=out.Q,
                lambda_cluster=lam_c, lambda_sep=lam_s,
                lambda_mse=cfg.loss.lambda_mse,
                # These were previously never passed, so lambda_var/cov/unif silently
                # defaulted to 0.0 and VICReg never ran under this trainer.
                lambda_var=lam_v, lambda_cov=lam_cv, lambda_unif=lam_u,
                metric=model.metric,
            )

            opt.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

            if not no_renorm:
                model.post_step()
            if bool(model.centroids_initialized.item()):
                model.update_centroids_ema(out.z.detach(), out.Q.detach())

            Q_history.append(out.Q.detach().cpu())
            if len(Q_history) > 200:
                Q_history = Q_history[-200:]
            global_step += 1

            if global_step % cfg.train.diag_every == 0:
                val_kl, val_mse, val_fve = run_validation(
                    model, lc, val_loader, mean_t, std_t, device, onfly)
                run_sil = (global_step % (cfg.train.diag_every * 10) == 0)
                metrics = diag.compute_all(
                    z=out.z.detach(), Q=out.Q.detach(),
                    centroids=model.centroids.detach(),
                    train_mse=losses["recon"].item(), val_mse=val_mse,
                    train_fve=losses["fve"].item(), val_fve=val_fve,
                    Q_history=Q_history, run_silhouette=run_sil,
                )
                metrics["train/loss"] = losses["loss"].item()
                metrics["train/kl"] = losses["kl"].item()
                metrics["val/kl"] = val_kl
                metrics["train/cluster_loss"] = losses["cluster"].item()
                metrics["train/sep_loss"] = losses["sep"].item()
                metrics["train/tau"] = tau
                metrics["step"] = global_step
                print(
                    f"  step {global_step:6d} | loss {metrics['train/loss']:.4f} | "
                    f"kl {metrics['train/kl']:.4f} | val_kl {val_kl:.4f} | "
                    f"clus {metrics['train/cluster_loss']:.4f} | "
                    f"sep {metrics['train/sep_loss']:.4f} | "
                    f"mse(diag) {metrics['train/mse']:.4f} | "
                    f"eff_K {metrics['cluster/effective_k']}/{model.n_clusters} | "
                    f"dying {metrics['cluster/dying']}"
                )
                if logger is not None:
                    logger.log(metrics, step=global_step)

            if global_step % cfg.train.reinit_every == 0:
                n_reinit = reinit_dead_clusters(
                    model, out.z.detach(), x.detach(), out.x_hat.detach())
                if n_reinit > 0:
                    print(f"  [reinit] step {global_step}: {n_reinit} dead clusters")
                    if logger is not None:
                        logger.log({"cluster/reinit_count": n_reinit}, step=global_step)

        if epoch % cfg.train.save_every == 0:
            val_kl, val_mse, val_fve = run_validation(
                model, lc, val_loader, mean_t, std_t, device, onfly)
            ckpt_path = ckpt_dir / f"step_{global_step:07d}.pt"
            save_checkpoint(ckpt_path, model, opt, epoch, global_step, tau,
                            train_buf.mean, train_buf.std, val_kl, val_mse, cfg)
            print(f"[train-e2e] Saved {ckpt_path.name}  val_kl={val_kl:.5f}  "
                  f"val_mse={val_mse:.5f}")
            if tracker.finish_epoch(ckpt_path, val_kl):
                print(f"[train-e2e] New best val_kl={tracker.best_score:.5f} -> best_val.pt")

        print(f"[train-e2e] Epoch {epoch} done in {(time.time()-epoch_start)/60:.1f} min")

    if logger is not None:
        logger.finish()
    print(f"\n[train-e2e] Done. Best val_kl = {tracker.best_score:.5f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    ap.add_argument("--centroid_init", default=None,
                    choices=["kmeans++", "semisup", "class_means"])
    ap.add_argument("--teacher_mode", default=None, choices=["cached", "onfly"],
                    help="onfly = compute teacher head(norm(x)) in-loop (last layer, no cache)")
    ap.add_argument("--lambda_mse", type=float, default=None,
                    help="Weak normalised-space MSE recon anchor (0 = off, KL-only)")
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint to resume from (e.g. .../best_val.pt). Restores "
                         "model + optimizer + centroids and continues at epoch+1.")
    ap.add_argument("--max_train_rows", type=int, default=None,
                    help="Escape hatch: cap the train split to the first N rows so the "
                         "working set stays in page cache. Normally unnecessary — the "
                         "buffer sets MADV_RANDOM, which is what actually fixes slow "
                         "random reads. Costs training data; prefer leaving it unset. "
                         "Norm stats are still computed over the full split.")
    add_override_args(ap)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    apply_overrides(cfg, args)

    train(cfg, use_wandb=not args.no_wandb, no_renorm=args.no_renorm,
          no_sinkhorn=args.no_sinkhorn, resume=args.resume,
          max_train_rows=args.max_train_rows)


if __name__ == "__main__":
    main()
