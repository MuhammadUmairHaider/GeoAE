"""
Fit a SINKHORN-BALANCED k-means baseline on residual-stream activations.

Why this exists. Plain MiniBatchKMeans on a residual stream is not merely
"collapsed" — it is catastrophically imbalanced. Measured on llama L27
(refit K=2000, --max_no_improvement 0), TEN clusters absorb 87% of tokens on the
fitter's own training distribution and 96% on FineWeb-Atlas chunks, while the AE
at the same K puts 5.5-6.7% in its top ten. Raising K does not fix it: the live
fraction FALLS with K (36% live at K=256, 28% at K=2000) and the concentration
stays put. Balance is not something plain k-means optimises.

That makes the existing baseline a confounded control. A GeoAE is an encoder
PLUS a Sinkhorn-balanced clustering objective, so beating unbalanced k-means
cannot say which half did the work. This fitter supplies the missing arm:
the same balancing, the same K, the same normalised space, NO encoder.

  AE                    encoder + Sinkhorn-balanced clusters
  fit_balanced_kmeans   identity + Sinkhorn-balanced clusters   <- this file
  fit_baseline_kmeans   identity + plain k-means

If the balanced baseline matches the AE on downstream metrics, the encoder is
contributing nothing and the balancing is the whole story. If the AE still wins,
the learned representation is doing real work. Either way it is decidable, which
the plain baseline alone could not make it.

The algorithm mirrors training (geoae/model.py + losses.sinkhorn_log): an init,
then minibatch steps of Sinkhorn assignment with an annealed temperature and a
hard-EMA centroid update, plus reinit of starved centroids.

INIT AND REINIT MUST MATCH THE AE ARM BEING CONTROLLED FOR. The defaults
(--init kmeans++ --reinit farthest) reproduce the original balanced baseline and
are the control for every k-means++-initialised AE. Against a density-peaks AE
(centroid_init: dpc, reinit_mode: peaks) that control is confounded a second
time, now by init: if the dpc AE beats it, the init alone may be the cause. Use
--init dpc --reinit peaks with the AE config's density settings, so the ONLY
remaining difference is the encoder. Note the old reinit is farthest-first (the
batch points furthest from every centroid), i.e. the outlier-seeking rule the
peaks reinit replaces.

Output .npz is byte-compatible with fit_baseline_kmeans, so closest_tokens
--baseline_kmeans, clustering_quality and load_baseline_kmeans all accept it
unchanged.

Usage:
    python -m geoae.interp.fit_balanced_kmeans \
      --checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt \
      --activations activations_diverse_10M/layer_27.npy \
      --n_clusters 2000 --n_sample 1500000 \
      --out e2e/checkpoints/general/llama3.2-3B/layer27/balanced_kmeans_k2000.npz

    # density-peaks control for a centroid_init: dpc / reinit_mode: peaks AE
    python -m geoae.interp.fit_balanced_kmeans \
      --checkpoint checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt \
      --activations activations_diverse_10M/layer_27.npy \
      --n_clusters 2000 --n_sample 1500000 --init dpc --reinit peaks \
      --out e2e/checkpoints/general/llama3.2-3B/layer27/balanced_kmeans_k2000_dpc.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from geoae.checkpoint import load_ae_checkpoint
from geoae.losses import sinkhorn_log
from geoae.seeded_init import density_peaks_select, format_peak_diag, seeded_init
from geoae.seeding import seed_everything


@torch.no_grad()
def encode_sample(ae, sample: np.ndarray, device, chunk: int = 16384) -> torch.Tensor:
    """Normalised activations -> AE latent, in chunks, straight onto the device.

    The latent is wider than the activation (2x-4x here), so this is the memory
    ceiling of a --space latent fit: n_sample x latent_dim x 4 bytes must fit on
    the card. At latent_dim 12288 that is ~49 GB for 1M rows, so latent fits run
    on a smaller sample than raw ones.
    """
    out = None
    for i in tqdm(range(0, len(sample), chunk), desc="encoding", leave=False):
        blk = torch.from_numpy(sample[i:i + chunk]).to(device)
        z = ae.encoder(blk)
        if out is None:
            out = torch.empty((len(sample), z.shape[1]), dtype=z.dtype, device=device)
        out[i:i + len(z)] = z
    return out


def kmeanspp_init(X: torch.Tensor, K: int, seed: int, chunk: int = 4096) -> torch.Tensor:
    """k-means++ seeding on a GPU sample (same init family as the plain fitter)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = X.shape[0]
    first = int(torch.randint(N, (1,), generator=g).item())
    cent = [X[first]]
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2. The naive ((X - c)**2).sum(1)
    # materialises an (N, D) temporary EVERY iteration -- 2.4 GB at N=200k,
    # D=3072 -- and there are K-1 = 1999 iterations. Expanding it turns each
    # step into a matvec over the precomputed squared norms, which is ~2 orders
    # of magnitude less memory traffic and leaves the result mathematically
    # identical (float rounding aside; this is a seeding step).
    Xsq = (X * X).sum(1)

    def _d2_to(c):
        return (Xsq - 2.0 * (X @ c) + (c * c).sum()).clamp_min_(0)

    d2 = _d2_to(cent[0])
    for _ in tqdm(range(1, K), desc="kmeans++ init", leave=False):
        p = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        nxt = int(torch.multinomial(p, 1, generator=g).item())
        c = X[nxt]
        cent.append(c)
        d2 = torch.minimum(d2, _d2_to(c))
    return torch.stack(cent)


@torch.no_grad()
def occupancy(X: torch.Tensor, C: torch.Tensor, chunk: int = 8192):
    lab = []
    for i in range(0, len(X), chunk):
        lab.append(torch.cdist(X[i:i + chunk], C).argmin(1))
    lab = torch.cat(lab)
    counts = torch.bincount(lab, minlength=C.shape[0]).float()
    share = counts.sort(descending=True).values
    return int((counts > 0).sum()), float(share[:10].sum() / counts.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="AE checkpoint (for norm + model_name + layer)")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--n_clusters", type=int, default=2000)
    ap.add_argument("--n_sample", type=int, default=1_500_000)
    ap.add_argument("--batch_size", type=int, default=4096,
                    help="Sinkhorn batch. Must be >= n_clusters for the column "
                         "marginal B/K to be meaningful; 2*K or more is better.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--tau_start", type=float, default=1.0)
    ap.add_argument("--tau_end", type=float, default=0.1)
    ap.add_argument("--sinkhorn_iters", type=int, default=3)
    ap.add_argument("--ema_decay", type=float, default=0.9)
    ap.add_argument("--reinit_every", type=int, default=1000,
                    help="Steps between reinit of starved centroids (0 disables).")
    ap.add_argument("--init_sample", type=int, default=200_000,
                    help="Points used for k-means++ init (full sample is too slow).")
    ap.add_argument("--init", default="kmeans++", choices=["kmeans++", "dpc", "seeded"],
                    help="Centroid init. dpc = density peaks, matching centroid_init: dpc. "
                         "seeded = labelled class means + density fill (requires --space latent).")
    ap.add_argument("--space", default="raw", choices=["raw", "latent"],
                    help="Where to cluster. raw = normalised activations (the encoder-free "
                         "control). latent = the AE's own latent, which asks whether a better "
                         "partition of the SAME representation recovers what the AE's trained "
                         "centroids miss.")
    # --init seeded: labelled anchors, same knobs as the trainer's centroid_init: seeded
    ap.add_argument("--anchor_cache", default="cache")
    ap.add_argument("--anchor_per_class", type=int, default=25)
    ap.add_argument("--anchor_min_examples", type=int, default=5)
    ap.add_argument("--anchor_rungs", default="", help="comma list; empty = every rung")
    ap.add_argument("--atlas_last", default="", help="atlas last-token cache for doc concepts")
    ap.add_argument("--atlas_min_examples", type=int, default=25)
    ap.add_argument("--fill_mode", default="peaks", choices=["coverage", "peaks"])
    ap.add_argument("--reinit", default="farthest", choices=["farthest", "peaks"],
                    help="Starved-centroid reinit. farthest = batch points furthest from "
                         "every centroid (original); peaks = matching reinit_mode: peaks.")
    # Density-peaks settings. Names and defaults mirror TrainConfig, so a dpc AE
    # config's values can be passed straight across.
    ap.add_argument("--density_pool", type=int, default=32768,
                    help="Points the dpc init selects from; delta is O(N^2) in this.")
    ap.add_argument("--density_knn", type=int, default=32)
    ap.add_argument("--density_power", type=float, default=1.0)
    ap.add_argument("--peak_min_sep_frac", type=float, default=0.25)
    ap.add_argument("--peak_refine_k", type=int, default=8)
    ap.add_argument("--peak_reinit_pool", type=int, default=8192)
    # Token erasure (geoae.interp.token_erasure): cluster the normalised activations
    # with a fixed set of directions projected out. The basis is stored in the
    # output, and only concept_probe applies it; other tools refuse the codebook.
    ap.add_argument("--erase", default="", help="token_erasure .npz; empty = no erasure")
    ap.add_argument("--erase_basis", default="token", choices=["token", "pca"],
                    help="token = between-token directions; pca = same-rank control")
    ap.add_argument("--erase_rank", type=int, default=128)
    # Token-mean subtraction (geoae/token_bias.py): cluster x - b[current token] with the
    # SAME table a token-bypass AE subtracts — the encoder-free control for that AE.
    # Needs rows_tok.npy beside the activations. Only concept_probe-style tools apply it.
    ap.add_argument("--token_bias", default="", help="token_bias .npz; empty = none")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if (args.erase or args.token_bias) and args.space != "raw":
        raise SystemExit("[balanced-fit] --erase / --token_bias act on the normalised activations: need --space raw")
    if args.erase and args.token_bias:
        raise SystemExit("[balanced-fit] pick one of --erase and --token_bias")

    seed_everything(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch_size < args.n_clusters:
        raise SystemExit(f"[balanced-fit] --batch_size {args.batch_size} < K {args.n_clusters}: "
                         f"Sinkhorn's column target B/K < 1 makes balancing meaningless.")

    if args.init == "seeded" and args.space != "latent":
        raise SystemExit("[balanced-fit] --init seeded needs --space latent: the class means "
                         "are computed with the AE encoder.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(ckpt["norm_mean"], dtype=np.float32)
    std = np.asarray(ckpt["norm_std"], dtype=np.float32)
    cfg = ckpt["config"]
    layer, model_name = cfg["data"]["target_layer"], cfg["extraction"]["model_name"]
    print(f"[balanced-fit] AE norm from checkpoint (layer {layer}, {model_name})")

    ae = None
    if args.space == "latent":
        ae, _, _, _ = load_ae_checkpoint(args.checkpoint, dev)
        ae.eval()
        if args.init == "seeded" and ae.n_clusters != args.n_clusters:
            raise SystemExit(f"[balanced-fit] --init seeded uses the AE's own K "
                             f"({ae.n_clusters}) but --n_clusters is {args.n_clusters}.")
        print(f"[balanced-fit] clustering in the AE LATENT (dim {ae.centroids.shape[1]})")

    mmap = np.load(args.activations, mmap_mode="r")
    N, D = mmap.shape
    rng = np.random.RandomState(args.seed)
    idx = np.sort(rng.choice(N, size=min(args.n_sample, N), replace=False))
    print(f"[balanced-fit] Sampling {len(idx):,}/{N:,} activations and normalising …")
    sample = np.empty((len(idx), D), dtype=np.float32)
    for s in tqdm(range(0, len(idx), 100_000), desc="reading", unit="chunk"):
        blk = idx[s:s + 100_000]
        sample[s:s + len(blk)] = (mmap[blk].astype(np.float32) - mean) / std
    if not np.isfinite(sample).all():
        raise SystemExit("[balanced-fit] non-finite values in the sample — check the "
                         "activation dump's dtype (fp16 overflows on large-magnitude layers).")

    if args.space == "latent":
        X = encode_sample(ae, sample, dev)
        del sample
    else:
        X = torch.from_numpy(sample).to(dev)
    erase_U = None
    if args.erase:
        e = np.load(args.erase, allow_pickle=True)
        if not (np.allclose(e["norm_mean"], mean) and np.allclose(e["norm_std"], std)):
            raise SystemExit(f"[balanced-fit] {args.erase} was fit under different norm stats than "
                             f"{args.checkpoint}; pass the checkpoint token_erasure used.")
        erase_U = torch.from_numpy(e[f"U_{args.erase_basis}"][:, :args.erase_rank]).to(dev)
        if erase_U.shape[1] < args.erase_rank:
            raise SystemExit(f"[balanced-fit] basis has only {erase_U.shape[1]} directions")
        for i in range(0, len(X), 65536):
            X[i:i + 65536] -= (X[i:i + 65536] @ erase_U) @ erase_U.T
        print(f"[balanced-fit] erased {args.erase_rank} {args.erase_basis} directions ({args.erase})")
    if args.token_bias:
        from geoae.token_bias import TokenBiasLookup
        tb = TokenBiasLookup(args.token_bias, dev)
        if not (np.allclose(tb.norm_mean, mean) and np.allclose(tb.norm_std, std)):
            raise SystemExit(f"[balanced-fit] {args.token_bias} was built under different norm stats "
                             f"than {args.checkpoint}")
        tok_path = Path(args.activations).parent / "rows_tok.npy"
        rows_tok = np.load(tok_path)[idx]
        for i in range(0, len(X), 65536):
            X[i:i + 65536] -= tb(torch.from_numpy(rows_tok[i:i + 65536]).to(dev))
        print(f"[balanced-fit] subtracted b[current token] ({args.token_bias}, {tok_path})")
    print(f"[balanced-fit] fitting on {tuple(X.shape)} ({X.numel() * 4 / 2**30:.1f} GB)")

    if args.init == "seeded":
        rungs = [r for r in args.anchor_rungs.split(",") if r] or None
        # The fill's density estimate is pairwise over its pool, computed in row
        # chunks against the WHOLE pool, so the pool must be a subsample and not
        # the full sample: at 300k x 12288 one chunk alone asks for ~3.7 GB and
        # the fit OOMs. The trainer sizes this pool at density_pool for the same
        # reason, so use the same knob here.
        pool_n = min(args.density_pool, len(X))
        z_pool = X[torch.randperm(len(X), device=dev)[:pool_n]]
        print(f"[balanced-fit] density fill pool: {pool_n:,} of {len(X):,} latents")
        C, names, _pool, diag = seeded_init(
            ae, args.anchor_cache, torch.from_numpy(mean).to(dev), torch.from_numpy(std).to(dev),
            dev, z_pool, per_class=args.anchor_per_class, min_examples=args.anchor_min_examples,
            rungs=rungs, density_power=args.density_power, seed=args.seed,
            min_sep_frac=args.peak_min_sep_frac, atlas_last=args.atlas_last,
            atlas_min_examples=args.atlas_min_examples, fill_mode=args.fill_mode,
            knn_k=args.density_knn, refine_k=args.peak_refine_k)
        C = C.clone()
        n_anchor = sum(1 for n in names if not n.startswith("fill::"))
        print(f"[balanced-fit] seeded init — {n_anchor} labelled class means + "
              f"{args.n_clusters - n_anchor} {args.fill_mode} fill")
        if diag is not None:
            print(f"[balanced-fit]   {format_peak_diag(diag)}")
        del z_pool
    elif args.init == "dpc":
        pool_n = min(args.density_pool, len(X))
        print(f"[balanced-fit] density-peaks init on {pool_n:,} points "
              f"(density_power={args.density_power}, knn={args.density_knn}) …")
        pool = X[torch.randperm(len(X), device=dev)[:pool_n]]
        C, diag = density_peaks_select(
            pool, args.n_clusters, seed=args.seed, density_power=args.density_power,
            min_sep_frac=args.peak_min_sep_frac, knn_k=args.density_knn,
            refine_k=args.peak_refine_k,
        )
        C = C.clone()
        print(f"[balanced-fit]   {format_peak_diag(diag)}")
        del pool
    else:
        init_n = min(args.init_sample, len(X))
        print(f"[balanced-fit] k-means++ init on {init_n:,} points …")
        C = kmeanspp_init(X[torch.randperm(len(X), device=dev)[:init_n]], args.n_clusters, args.seed)

    ema_size = torch.zeros(args.n_clusters, device=dev)
    ema_sum = C.clone()
    steps_per_epoch = len(X) // args.batch_size
    total = args.epochs * steps_per_epoch
    print(f"[balanced-fit] {args.epochs} epochs x {steps_per_epoch} steps, batch {args.batch_size}")
    step = 0
    n_reinit = 0
    for ep in range(args.epochs):
        perm = torch.randperm(len(X), device=dev)
        for s in tqdm(range(steps_per_epoch), desc=f"epoch {ep+1}/{args.epochs}", leave=False):
            b = X[perm[s * args.batch_size:(s + 1) * args.batch_size]]
            frac = step / max(total - 1, 1)
            tau = args.tau_start + (args.tau_end - args.tau_start) * frac
            cost = torch.cdist(b, C) ** 2
            Q = sinkhorn_log(cost, tau, args.sinkhorn_iters)
            hard = torch.nn.functional.one_hot(Q.argmax(1), args.n_clusters).to(b.dtype)
            cnt = hard.sum(0)
            ema_size.mul_(args.ema_decay).add_(cnt * (1 - args.ema_decay))
            ema_sum.mul_(args.ema_decay).add_((hard.T @ b) * (1 - args.ema_decay))
            visited = cnt > 0                      # never decay unvisited centroids to the origin
            C[visited] = (ema_sum[visited] / ema_size[visited].unsqueeze(1).clamp_min(1e-6))
            if args.reinit_every and step and step % args.reinit_every == 0:
                dead = ema_size < (0.1 * ema_size.mean())
                if int(dead.sum()):
                    if args.reinit == "peaks":
                        # Same rule as reinit_mode "peaks": batch density peaks,
                        # with the live centroids as suppression centres.
                        pool = b[:args.peak_reinit_pool]
                        new, _ = density_peaks_select(
                            pool, min(int(dead.sum()), len(pool)), C_init=C[~dead],
                            seed=0, density_power=args.density_power,
                            min_sep_frac=args.peak_min_sep_frac,
                            knn_k=args.density_knn, refine_k=args.peak_refine_k,
                        )
                        dead = dead.nonzero(as_tuple=True)[0][:len(new)]
                    else:
                        new = b[torch.cdist(b, C).min(1).values.topk(int(dead.sum())).indices]
                    n_reinit += len(new)
                    C[dead] = new
                    ema_size[dead] = ema_size.mean()
                    ema_sum[dead] = new * ema_size.mean()
            step += 1
        live, top10 = occupancy(X[:400_000], C)
        print(f"  epoch {ep+1}: tau {tau:.3f} | live {live}/{args.n_clusters} | "
              f"top-10 share {top10:.1%} | reinit so far {n_reinit}")

    live, top10 = occupancy(X[:1_000_000], C)
    print(f"[balanced-fit] Final on a {min(len(X),1_000_000):,}-token probe: "
          f"{live}/{args.n_clusters} live, top-10 share {top10:.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if erase_U is not None:
        extra = dict(erase_U=erase_U.cpu().numpy(), erase_basis=args.erase_basis,
                     erase_rank=args.erase_rank, erase_source=args.erase)
    if args.token_bias:
        extra = dict(token_bias=args.token_bias)
    np.savez(str(out), centroids=C.cpu().numpy().astype(np.float32),
             norm_mean=mean, norm_std=std, layer=layer,
             n_clusters=args.n_clusters, model_name=model_name,
             init=args.init, reinit=args.reinit, space=args.space,
             ae_checkpoint=(args.checkpoint if args.space == "latent" else ""), **extra)
    print(f"[balanced-fit] Saved -> {out}")


if __name__ == "__main__":
    main()
