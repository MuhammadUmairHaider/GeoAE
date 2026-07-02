"""
Streaming on-the-fly end-to-end KL training — works at ANY layer, no teacher cache.

The cached-teacher e2e path (geoae.e2e.train) only supports onfly at the LAST layer
and otherwise needs a ~1.28 TB teacher-logit cache. This trainer removes that
limit by computing BOTH the teacher and the student logits in-loop, over a
streamed diverse corpus, for ALL token positions of each document:

  per document (input_ids):
    1. teacher forward (no_grad): full LM pass -> teacher logits (T, V), and a
       hook captures the layer-L residual (T, D).
    2. AE: x = (residual - mean)/std -> x_hat, z, Q ; recon = x_hat*std + mean.
    3. student forward (grad): full LM pass with the layer-L residual REPLACED by
       `recon` at every position -> student logits (T, V). (Last layer: the head
       is position-wise, so student = lm_head(norm(recon)) directly — no forward.)
    4. loss = forward-KL(teacher || student) + lambda_cluster*L_cluster
              + lambda_sep*L_sep, over positions [skip_leading:].
    Gradients flow through the frozen tail (layers L+1..last + norm + head) into
    the AE; the LM weights are frozen.

Checkpoints are written in the SAME format as geoae.e2e.train (model_state,
norm_mean/std, config, val_kl), so collect_closest_tokens / logit_attribution /
dla_inventory work on them unchanged.

Usage:
    python e2e/train_e2e_stream.py \
      --config e2e/configs/general/llama3.2-3B/layer16/kl_gelu.yaml \
      --n_clusters 1000 --no_sinkhorn --accum 8 --tokens_per_epoch 1000000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path



import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from geoae.config import Config
from geoae.seeding import seed_everything
from geoae.hooks import SplicingHook
from geoae.e2e.logits import locate_lm_parts
from geoae.e2e.losses import total_loss_e2e, kl_loss
from geoae.train_common import (
    tau_schedule, lambda_schedule, reinit_dead_clusters,
    build_model, make_optimizer, add_override_args, apply_overrides,
    CheckpointTracker,
)
from geoae.e2e.train import save_checkpoint, load_frozen_lm
from geoae.extract import open_sources, stream_docs, DEFAULT_SOURCES


# --------------------------------------------------------------------------- #
# All-position teacher + student logits via the frozen LM
# --------------------------------------------------------------------------- #

class StreamLogits:
    """Teacher (un-spliced) and student (recon-spliced) logits for every position
    of one document, computed in-loop. Generalises logits.LogitsComputer from the
    last-token-only path to all positions."""

    def __init__(self, lm, layer_idx: int):
        self.lm = lm
        _, layers, self.final_norm, self.lm_head = locate_lm_parts(lm)
        self.n_layers = len(layers)
        self.layer_idx = int(layer_idx)
        self.is_last = self.layer_idx == self.n_layers - 1
        self.head_dtype = self.lm_head.weight.dtype
        self.hook = SplicingHook(lm, self.layer_idx)

    @torch.no_grad()
    def teacher_and_residual(self, input_ids):
        """Returns (residual (T, D) fp32, teacher_logits (T, V) fp32), detached."""
        cap = []
        self.hook.activate(lambda hs: (cap.append(hs) or hs))
        out = self.lm(input_ids=input_ids)
        self.hook.deactivate()
        return cap[0][0].float(), out.logits[0].float()

    def student(self, input_ids, recon_raw):
        """recon_raw (T, D) carries grad. Returns student logits (T, V) with grad."""
        if self.is_last:
            return self.lm_head(self.final_norm(recon_raw.to(self.head_dtype)))
        rl = recon_raw.to(self.head_dtype).unsqueeze(0)        # (1, T, D)
        self.hook.activate(lambda hs: rl.to(hs.dtype))
        try:
            out = self.lm(input_ids=input_ids)
        finally:
            self.hook.deactivate()
        return out.logits[0]


# --------------------------------------------------------------------------- #
# Normalisation stats from the cached layer activations (matches the AE pipeline)
# --------------------------------------------------------------------------- #

def compute_norm(act_path: Path, n_sample: int, seed: int):
    mmap = np.load(str(act_path), mmap_mode="r")
    N = mmap.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(N, min(n_sample, N), replace=False))
    sample = mmap[idx].astype(np.float32)
    mean = sample.mean(axis=0)
    std = sample.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


# --------------------------------------------------------------------------- #
# Streaming helpers
# --------------------------------------------------------------------------- #

def pull_val_docs(tokenizer, n_docs, max_len, min_len):
    """Stream a fixed held-out set of documents (input_ids on CPU) for validation."""
    srcs = open_sources([dict(s) for s in DEFAULT_SOURCES])
    val = []
    for ids, _ in stream_docs(srcs, n_docs * max_len * 4, tokenizer, min_len, max_len):
        val.append(ids)
        if len(val) >= n_docs:
            break
    return val


def open_train_stream(tokenizer, args, total_tokens):
    """Persistent training doc stream for the whole run, resumed across epochs.

    Holds out the first n_val_docs — these are exactly the validation docs, because
    pull_val_docs streams the same sources in the same deterministic order. The rest
    are yielded continuously, so consecutive epochs see FRESH data instead of replaying
    the same ~tokens_per_epoch slice every epoch (which overfits).
    """
    srcs = open_sources([dict(s) for s in DEFAULT_SOURCES])
    gen = stream_docs(srcs, total_tokens, tokenizer, args.min_doc_len, args.max_doc_tokens)
    for _ in range(args.n_val_docs):          # hold out the val prefix
        if next(gen, None) is None:
            break
    return gen


@torch.no_grad()
def run_validation(model, sl, val_docs, mean_t, std_t, device, skip):
    model.eval()
    kl_sum = mse_sum = 0.0
    n = 0
    for ids in val_docs:
        ids = ids.to(device)
        residual, teacher = sl.teacher_and_residual(ids)
        x = (residual - mean_t) / std_t
        out = model(x)
        recon_raw = out.x_hat * std_t + mean_t
        student = sl.student(ids, recon_raw)
        v = slice(skip, ids.shape[1])
        if teacher[v].shape[0] == 0:
            continue
        kl_sum += kl_loss(teacher[v], student[v]).item()
        mse_sum += (out.x_hat[v] - x[v]).pow(2).mean().item()
        n += 1
    model.train()
    n = max(n, 1)
    return kl_sum / n, mse_sum / n


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def train(cfg, args):
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.extraction.model_name
    layer = cfg.data.target_layer
    skip = args.skip_leading

    # ---- norm stats from cached activations (<activations_dir>/layer_<L>.npy) ----
    from geoae.paths import resolve_path
    act_dir = resolve_path(cfg.data.activations_dir)
    act_path = act_dir / f"layer_{layer}.npy"
    if not act_path.exists():
        raise FileNotFoundError(f"Need cached activations for norm stats: {act_path}")
    print(f"[stream] Computing norm stats from {act_path.name} …")
    mean_np, std_np = compute_norm(act_path, args.norm_sample, cfg.train.seed)
    mean_t = torch.from_numpy(mean_np).to(device)
    std_t = torch.from_numpy(std_np).to(device)

    # ---- frozen LM + logits ----
    lm = load_frozen_lm(model_name, device)
    sl = StreamLogits(lm, layer)
    print(f"[stream] target layer {layer}/{sl.n_layers} (is_last={sl.is_last}) "
          f"{'head-only' if sl.is_last else 'full-forward + all-position splice'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---- model ----
    model = build_model(cfg, device, no_sinkhorn=args.no_sinkhorn)
    print(f"[stream] AE enc={cfg.model.nonlinearity} L={cfg.model.latent_dim} "
          f"K={cfg.model.n_clusters} sinkhorn={'on' if model.use_sinkhorn else 'off'} "
          f"λ_mse={cfg.loss.lambda_mse}")

    # ---- sanity: splicing the TRUE residual must reproduce the teacher logits ----
    val_docs = pull_val_docs(tokenizer, args.n_val_docs, args.max_doc_tokens, args.min_doc_len)
    if not val_docs:
        raise RuntimeError("No validation docs streamed — check data sources.")
    ids0 = val_docs[0].to(device)
    res0, teach0 = sl.teacher_and_residual(ids0)
    with torch.no_grad():
        stud0 = sl.student(ids0, res0)
    drift = (teach0 - stud0).abs().max().item()
    print(f"[stream] splice sanity: max|teacher-student(true residual)| = {drift:.4f} "
          f"({'OK' if drift < 2.0 else 'WARN — splice path may be wrong'})")

    opt = make_optimizer(model, cfg)

    ckpt_dir = resolve_path(cfg.train.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tracker = CheckpointTracker(ckpt_dir, cfg.train.keep_checkpoints)

    global_step = 0
    z_buffer: list[torch.Tensor] = []     # latents gathered during warmup for k-means++
    z_buffer_rows = 0                     # running row count (rolling cap below)

    # One persistent training stream for the WHOLE run. Each epoch consumes a FRESH
    # ~tokens_per_epoch slice (the generator is resumed across epochs), so n_epochs
    # epochs cover ~n_epochs× more UNIQUE tokens at the same compute — instead of
    # re-opening the same stream every epoch and replaying the same ~tokens_per_epoch
    # docs (which the AE memorises while held-out val_kl plateaus).
    total_budget = args.tokens_per_epoch * cfg.train.n_epochs + args.n_val_docs * args.max_doc_tokens
    doc_iter = open_train_stream(tokenizer, args, total_budget)

    for epoch in range(1, cfg.train.n_epochs + 1):
        tau = tau_schedule(epoch, cfg); model.tau = tau
        lam_c, lam_s = lambda_schedule(epoch, cfg)
        print(f"\n[stream] Epoch {epoch}/{cfg.train.n_epochs} | tau={tau:.3f} "
              f"| λ_c={lam_c:.3f} λ_s={lam_s:.4f} | "
              f"centroids_init={bool(model.centroids_initialized.item())}")
        epoch_start = time.time()

        accum = 0
        ema_z, ema_Q = [], []
        epoch_tokens = 0
        while epoch_tokens < args.tokens_per_epoch:
            try:
                ids, _domain = next(doc_iter)
            except StopIteration:
                print("[stream] corpus exhausted — restarting stream (data repeats)")
                doc_iter = open_train_stream(tokenizer, args, total_budget)
                ids, _domain = next(doc_iter)
            ids = ids.to(device)
            T = ids.shape[1]
            if T <= skip:
                continue
            epoch_tokens += T

            residual, teacher = sl.teacher_and_residual(ids)     # (T,D),(T,V) detached
            x = (residual - mean_t) / std_t
            out = model(x)
            recon_raw = out.x_hat * std_t + mean_t
            student = sl.student(ids, recon_raw)                 # (T,V) grad
            v = slice(skip, T)

            losses = total_loss_e2e(
                teacher_logits=teacher[v], student_logits=student[v],
                x=x[v], x_hat=out.x_hat[v], z=out.z[v],
                centroids=model.centroids.detach(), Q=out.Q[v],
                lambda_cluster=lam_c, lambda_sep=lam_s,
                lambda_mse=cfg.loss.lambda_mse, metric=model.metric,
            )
            (losses["loss"] / args.accum).backward()
            ema_z.append(out.z[v].detach()); ema_Q.append(out.Q[v].detach())
            if not bool(model.centroids_initialized.item()):
                zc = out.z[v].detach().cpu()
                z_buffer.append(zc); z_buffer_rows += zc.shape[0]
                # rolling cap: keep only the most recent latents so the warmup
                # buffer can't grow across epochs and OOM the host at torch.cat.
                while z_buffer_rows > max(20000, model.n_clusters * 8) and len(z_buffer) > 1:
                    z_buffer_rows -= z_buffer.pop(0).shape[0]

            accum += 1
            if accum >= args.accum:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step(); opt.zero_grad()
                if not args.no_renorm:
                    model.post_step()
                if bool(model.centroids_initialized.item()):
                    model.update_centroids_ema(torch.cat(ema_z), torch.cat(ema_Q))
                ema_z, ema_Q = [], []
                accum = 0
                global_step += 1

                # k-means++ init once the warmup is over and we have enough latents
                if (not bool(model.centroids_initialized.item())
                        and epoch >= cfg.train.clustering_start_epoch):
                    zb = torch.cat(z_buffer)[: max(20000, model.n_clusters * 8)]
                    if zb.shape[0] >= model.n_clusters:
                        model.init_centroids_kmeans_plus_plus(zb.to(device), seed=cfg.train.seed)
                        print(f"[stream] k-means++ centroid init from {zb.shape[0]:,} latents")
                        z_buffer = []; z_buffer_rows = 0

                if global_step % cfg.train.diag_every == 0:
                    val_kl, val_mse = run_validation(model, sl, val_docs, mean_t, std_t, device, skip)
                    eff_k = int((model.ema_cluster_size > 0.1 * model.ema_cluster_size.mean()).sum())
                    print(f"  step {global_step:6d} | loss {losses['loss'].item():.4f} | "
                          f"kl {losses['kl'].item():.4f} | val_kl {val_kl:.4f} | "
                          f"clus {losses['cluster'].item():.4f} | sep {losses['sep'].item():.4f} | "
                          f"fve {losses['fve'].item():.3f} | eff_K {eff_k}/{model.n_clusters}")
                    if tracker.is_better(val_kl):
                        save_checkpoint(tracker.best_target, model, opt, epoch, global_step,
                                        tau, mean_np, std_np, val_kl, val_mse, cfg)
                        tracker.record_best(val_kl)
                        print(f"  [stream] new best val_kl={tracker.best_score:.5f} -> best_val.pt")

                if (cfg.train.reinit_every and global_step % cfg.train.reinit_every == 0
                        and bool(model.centroids_initialized.item())):
                    n_re = reinit_dead_clusters(model, out.z[v].detach(), x[v].detach(), out.x_hat[v].detach())
                    if n_re:
                        print(f"  [reinit] step {global_step}: {n_re} dead clusters")

        # end of epoch checkpoint
        val_kl, val_mse = run_validation(model, sl, val_docs, mean_t, std_t, device, skip)
        ckpt_path = ckpt_dir / f"step_{global_step:07d}.pt"
        save_checkpoint(ckpt_path, model, opt, epoch, global_step, tau,
                        mean_np, std_np, val_kl, val_mse, cfg)
        tracker.finish_epoch(ckpt_path, val_kl)
        print(f"[stream] Epoch {epoch} done in {(time.time()-epoch_start)/60:.1f} min "
              f"| val_kl={val_kl:.5f} | best={tracker.best_score:.5f}")

    print(f"\n[stream] Done. Best val_kl = {tracker.best_score:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lambda_mse", type=float, default=None,
                    help="Weak MSE recon anchor added to the KL loss (0 = off, e.g. 0.05)")
    add_override_args(ap)
    # streaming-specific
    ap.add_argument("--accum", type=int, default=8, help="Documents per optimizer step (grad accumulation)")
    ap.add_argument("--tokens_per_epoch", type=int, default=1_000_000)
    ap.add_argument("--max_doc_tokens", type=int, default=256)
    ap.add_argument("--skip_leading", type=int, default=4)
    ap.add_argument("--min_doc_len", type=int, default=10)
    ap.add_argument("--n_val_docs", type=int, default=150)
    ap.add_argument("--norm_sample", type=int, default=500_000)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    apply_overrides(cfg, args)

    train(cfg, args)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(0)   # HF streaming threads don't join cleanly
