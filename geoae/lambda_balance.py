"""
Gradient-balance analysis for the GeoAE loss weights.

WHY THIS EXISTS
---------------
The lambdas in the gemma3 configs were inherited from the Llama-3.2-3B layer-27
tuning and carried forward unchanged to 12B/L47 and 4B/L33. They were never
re-derived per layer, and they cannot transfer as-is: every term in
`geoae.losses.total_loss` has a different dependence on the latent scale, and
the latent scale depends on the layer (4B layer 22 has median token norm 41,411
vs layer 33's 79,870 — a 1.9x difference before z-scoring, and the encoder's
output scale is free to differ afterwards).

WHAT IT MEASURES
----------------
For each loss term T, the gradient it delivers to the ENCODER — the shared trunk
that every term acts through, and therefore the place where the terms actually
compete:

    g_T = || d(T) / d(theta_encoder) ||_2

Balancing on the encoder rather than on all parameters is deliberate: the
geometry terms (cluster/sep/var/cov) touch nothing else, so an all-parameter
norm would dilute them by the decoder's contribution to recon alone and
systematically over-weight them.

The suggested weight for a target gradient share s_T is then

    lambda_T = s_T * g_recon / g_T

so that lambda_T * g_T = s_T * g_recon by construction.

It also reports, per term:
  * cos(grad_T, grad_recon) — GRADIENT CONFLICT. Negative means the term is
    actively pulling against reconstruction; that term's share is a real budget
    being spent, not a free addition. Near zero means it is spending its
    pressure in directions reconstruction does not care about, which is the
    ideal regime for a geometry regulariser.
  * the latent std distribution, which decides whether the VICReg variance
    hinge (relu(1.0 - std_j), gamma hardcoded to 1.0 in losses.variance_loss)
    is active at all. If every std is already >= 1 the hinge is dead and
    lambda_var is irrelevant no matter how it is set; if every std is far below
    1 it is saturated and behaves like a constant pull.
  * effective rank of z (centred before SVD — GELU latents are non-negative and
    share a mean offset that otherwise shows up as one dominant component).

TWO MEASUREMENT POINTS, because the schedule is not uniform
-----------------------------------------------------------
`train_common.lambda_schedule` ramps only cluster and sep (zero until
`recon_only_epochs`, then cluster, then sep ramped from `full_loss_start_epoch`).
lambda_var and lambda_cov are NOT scheduled — they are live from step 1. So:

  phase "init"   — random encoder, no centroids. This is where var/cov actually
                   start competing with recon, so it is the honest place to set
                   lambda_var / lambda_cov.
  phase "warm"   — after a short recon-only warmup and a k-means++ centroid init
                   from trained latents, mirroring what train.py does at
                   clustering_start_epoch. This is where cluster/sep switch on,
                   so it is the honest place to set lambda_cluster / lambda_sep.

Reading the lambdas for all four terms off a single random-init measurement
would be wrong for cluster/sep by however much the encoder moves in the recon-
only phase, which is exactly the phase designed to move it a lot.

USAGE
    python -u -m geoae.lambda_balance --config <cfg.yaml> [--warmup_steps 400]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from geoae.config import Config
from geoae.data import ActivationBuffer, ShuffledActivationLoader
from geoae.losses import (
    cluster_loss,
    covariance_loss,
    recon_loss,
    sep_loss,
    uniformity_loss,
    variance_loss,
)
from geoae.paths import resolve_path
from geoae.seeding import seed_everything
from geoae.train_common import build_model, make_optimizer


# Target gradient shares, relative to the reconstruction gradient.
#
# cluster 0.25 — it must do real work (it is the only term that forms clusters)
#   but must never rival recon, or the latent contracts to satisfy it.
# var 0.25 — deliberately EQUAL to cluster. cluster_loss rewards latent
#   contraction and the adaptive-sigma sep_loss is scale-invariant, so it cannot
#   resist that pull; variance_loss is the only counterforce in the objective.
#   Setting it below cluster's share means signing up for the contraction the
#   term exists to prevent.
# cov 0.10 — anti-low-rank pressure. Cheaper than var because it fights a
#   failure mode (rank collapse) that shows up more slowly.
# sep 0.05 — small, and ramped from 0 by the schedule anyway. It plateaus once
#   pairs are roughly equidistant, so a large weight buys nothing.
DEFAULT_SHARES = {"cluster": 0.25, "sep": 0.05, "var": 0.25, "cov": 0.10, "unif": 0.0}


def _encoder_grad(model: nn.Module, term: torch.Tensor) -> tuple[float, torch.Tensor]:
    """||d term / d theta_enc|| and the flattened gradient vector."""
    params = [p for p in model.encoder.parameters() if p.requires_grad]
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])
    return float(flat.norm()), flat


def _effective_rank(z: torch.Tensor) -> float:
    zc = (z - z.mean(dim=0, keepdim=True)).float()
    sv = torch.linalg.svdvals(zc)
    p = sv / sv.sum().clamp(min=1e-12)
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def probe(model, loader, device, metric, n_batches, phase, use_centroids,
          with_unif=False):
    """Average per-term loss value, encoder-grad norm, and conflict cosine."""
    acc: dict[str, list] = {}
    zstats, eranks = [], []

    it = iter(loader)
    for _ in range(n_batches):
        x = next(it).to(device)
        out = model(x)

        terms = {}
        terms["recon"], _ = recon_loss(x, out.x_hat)
        terms["var"] = variance_loss(out.z)
        terms["cov"] = covariance_loss(out.z)
        if with_unif:
            # Off by default: pdist's backward materialises an (S, S, D) tensor
            # (S=2048 subsample, D=latent_dim), which is 40 GB at D=2560 and OOMs
            # a 40 GB A100. Only measurable at a reduced subsample, and pointless
            # unless lambda_unif is actually going to be nonzero — the cosine
            # metric it exists for is a documented dead end at layer 27.
            terms["unif"] = uniformity_loss(out.z, max_samples=512)
        if use_centroids:
            terms["cluster"] = cluster_loss(
                out.z, model.centroids.detach(), out.Q, metric=metric
            )
            terms["sep"] = sep_loss(out.z, out.Q, metric=metric)

        _, g_recon = _encoder_grad(model, terms["recon"])
        n_recon = float(g_recon.norm())

        for name, t in terms.items():
            n, g = _encoder_grad(model, t)
            cos = float(
                torch.dot(g, g_recon) / (g.norm().clamp(min=1e-12) * g_recon.norm().clamp(min=1e-12))
            )
            acc.setdefault(name, []).append((float(t.detach()), n, n / max(n_recon, 1e-12), cos))

        with torch.no_grad():
            zstats.append(out.z.std(dim=0).cpu())
            if out.z.shape[0] >= 512:
                eranks.append(_effective_rank(out.z[:2048]))

    std = torch.stack(zstats).mean(0)
    summary = {
        name: dict(
            value=float(np.mean([r[0] for r in rows])),
            grad=float(np.mean([r[1] for r in rows])),
            ratio=float(np.mean([r[2] for r in rows])),
            cos=float(np.mean([r[3] for r in rows])),
        )
        for name, rows in acc.items()
    }
    return summary, std, float(np.mean(eranks)) if eranks else float("nan")


def _report(phase, summary, std, erank):
    print(f"\n=== phase: {phase} ===")
    print(f"{'term':<9} {'value':>12} {'|grad_enc|':>12} {'/recon':>9} {'cos(recon)':>11}")
    for name in ("recon", "cluster", "sep", "var", "cov", "unif"):
        if name not in summary:
            continue
        s = summary[name]
        print(f"{name:<9} {s['value']:>12.5g} {s['grad']:>12.5g} "
              f"{s['ratio']:>9.4g} {s['cos']:>+11.3f}")
    below = int((std < 1.0).sum())
    print(f"latent std: min={std.min():.3f} med={std.median():.3f} max={std.max():.3f} "
          f"| dims below the gamma=1.0 hinge: {below}/{len(std)} "
          f"({100*below/len(std):.1f}%)")
    print(f"effective rank of z: {erank:.1f} / {len(std)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--warmup_steps", type=int, default=400,
                    help="Recon-only steps before the 'warm' probe, standing in "
                         "for recon_only_epochs at a fraction of the cost.")
    ap.add_argument("--probe_batches", type=int, default=8)
    ap.add_argument("--max_train_rows", type=int, default=2_000_000,
                    help="Caps the working set only. Norm stats are always "
                         "computed over the FULL train split (data.py passes "
                         "val_start, not the capped indices), so the cached "
                         "norm_params npz is identical to a full run's.")
    ap.add_argument("--probe_unif", action="store_true",
                    help="Also measure uniformity_loss (see the note in probe()).")
    ap.add_argument("--shares", default=None,
                    help='JSON override, e.g. \'{"cluster":0.3,"var":0.3}\'')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shares = dict(DEFAULT_SHARES)
    if args.shares:
        shares.update(json.loads(args.shares))

    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    act_dir = resolve_path(cfg.data.activations_dir)
    buf = ActivationBuffer(act_dir, cfg.data.target_layer, val_frac=cfg.data.val_frac,
                           split="train", max_train_rows=args.max_train_rows)
    loader = ShuffledActivationLoader(buf, batch_size=cfg.data.batch_size,
                                      num_workers=4, pin_memory=True)
    model = build_model(cfg, device)
    print(f"[lambda] layer {cfg.data.target_layer}  latent_dim={cfg.model.latent_dim}  "
          f"K={cfg.model.n_clusters}  nonlinearity={cfg.model.nonlinearity}  "
          f"metric={model.metric}")

    # ---- phase 1: init (where var/cov begin competing with recon) ----
    init_sum, init_std, init_er = probe(model, loader, device, model.metric,
                                        args.probe_batches, "init", use_centroids=False,
                                        with_unif=args.probe_unif)
    _report("init (random encoder, var/cov live from step 1)", init_sum, init_std, init_er)

    # ---- recon-only warmup, mirroring recon_only_epochs ----
    opt = make_optimizer(model, cfg)
    print(f"\n[lambda] recon-only warmup: {args.warmup_steps} steps "
          f"@ batch {cfg.data.batch_size} …")
    it = iter(loader)
    for step in range(args.warmup_steps):
        try:
            x = next(it).to(device)
        except StopIteration:
            it = iter(loader); x = next(it).to(device)
        l, _ = recon_loss(x, model(x).x_hat)
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step(); model.post_step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>5}  mse {float(l):.5f}")

    # k-means++ centroid init from trained latents, exactly as train.py does
    init_batch = next(iter(loader)).to(device)
    with torch.no_grad():
        init_z = model.encoder(init_batch)
    model.init_centroids_kmeans_plus_plus(init_z, seed=cfg.train.seed)
    model.tau = cfg.loss.tau_start
    print(f"[lambda] k-means++ init of {cfg.model.n_clusters} centroids "
          f"from {len(init_batch)} warmed latents")

    warm_sum, warm_std, warm_er = probe(model, loader, device, model.metric,
                                        args.probe_batches, "warm", use_centroids=True,
                                        with_unif=args.probe_unif)
    _report("warm (post-warmup, centroids live — where cluster/sep switch on)",
            warm_sum, warm_std, warm_er)

    # ---- derive every lambda from the WARM measurement ----
    #
    # An earlier version of this tool calibrated var/cov at init, reasoning that
    # they are unscheduled and therefore live from step 1. That is the wrong
    # calibration point for a CONSTANT weight. Measured on 4B layer 22, the
    # ratios move ~30x between the two phases: recon's encoder gradient decays
    # 5x over a 400-step recon-only warmup (0.0631 -> 0.0121) while cov's grows
    # 6x (0.214 -> 1.199), so cov/recon goes 3.4 -> 99.2. A lambda_cov set to a
    # 0.10 share at init delivers a 2.9x share at the warm point and only climbs
    # further as MSE keeps falling.
    #
    # A fixed lambda should be balanced where the bulk of optimisation happens,
    # which is after the recon-only phase, not at a random init the run leaves
    # within a few hundred steps. The init probe is still reported because it
    # exposes the transient conflict (var starts at cos=-0.675 to recon and
    # turns to +0.148 once the encoder has trained), but it no longer sets any
    # weight.
    src = {"var": warm_sum, "cov": warm_sum, "unif": warm_sum,
           "cluster": warm_sum, "sep": warm_sum}
    lam = {}
    print("\n=== suggested lambdas  (lambda_T = share_T * |grad_recon| / |grad_T|) ===")
    print(f"{'term':<9} {'share':>7} {'measured at':>12} {'lambda':>12}")
    for name in ("cluster", "sep", "var", "cov", "unif"):
        s = shares.get(name, 0.0)
        if s <= 0 or name not in src[name]:
            lam[name] = 0.0
            print(f"{name:<9} {s:>7.2f} {'-':>12} {0.0:>12.4g}")
            continue
        phase = "warm"
        lam[name] = s / max(src[name][name]["ratio"], 1e-12)
        print(f"{name:<9} {s:>7.2f} {phase:>12} {lam[name]:>12.4g}")

    print("\n--- paste into the loss block ---")
    for name in ("cluster", "sep", "var", "cov", "unif"):
        print(f"  lambda_{name}: {lam[name]:.4g}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(config=args.config, shares=shares, lambdas=lam,
                           init=init_sum, warm=warm_sum,
                           init_erank=init_er, warm_erank=warm_er,
                           warmup_steps=args.warmup_steps), f, indent=2)
        print(f"\n[lambda] wrote {args.out}")


if __name__ == "__main__":
    main()
