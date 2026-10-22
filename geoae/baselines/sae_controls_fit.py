"""Fit step for the SAE controls: on ONE seeded sample of the dump TRAIN split (first 95% of rows),
  * per-feature activation frequency (rows with code > 0) for every SAE  -> counts_<name>.npz   (C3)
  * balanced k-means (K=2000, same procedure as the base/base_tokmean codebooks) in each SAE's CODE space
    -> kmeans_<name>.npz (centroids)                                                          (C1)
Codes are raw SAE codes (no rescaling). A SAE whose dense code matrix would exceed --max_dense_gb is skipped
for k-means with a logged reason in fit_summary.json (gs262k: 200k x 262144 fp32 = 210 GB; counts still computed).

    python -u geoae/baselines/sae_controls_fit.py --saes bt10k=matched:...,gs16k=scope:16k --out_dir <controls dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geoae.baselines.sae_controls import balanced_fit  # noqa: E402
from geoae.baselines.sae_space import chunk_rows
from geoae.baselines.sae_space import load_sae_space
from geoae.baselines.sae_space import parse_saes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saes", required=True)
    ap.add_argument("--activations_dir", default="activations_gemma3_4b_sampled_struct5_10M")
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--n_sample", type=int, default=200_000)
    ap.add_argument("--train_frac", type=float, default=0.95)
    ap.add_argument("--n_clusters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_dense_gb", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--density_pool", type=int, default=32768)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    act = Path(args.activations_dir)
    mm = np.load(act / f"layer_{args.layer}.npy", mmap_mode="r")
    tok_all = np.load(act / "rows_tok.npy", mmap_mode="r")
    n_train = int(len(mm) * args.train_frac)
    idx = np.sort(np.random.RandomState(args.seed).choice(n_train, min(args.n_sample, n_train), replace=False))
    print(f"[ctl-fit] {len(idx):,} rows from the first {n_train:,} (train split) of {len(mm):,}", flush=True)
    H = np.empty((len(idx), mm.shape[1]), np.float32)
    for s in range(0, len(idx), 50_000):
        H[s:s + 50_000] = mm[idx[s:s + 50_000]]
    T = np.asarray(tok_all[idx], np.int64)
    summary = {"n_rows": int(len(idx)), "train_rows": n_train, "seed": args.seed, "arms": {}}

    for name, spec in parse_saes(args.saes).items():
        sae = load_sae_space(spec, dev, name)
        F = sae.n_features
        gb = len(idx) * F * 4 / 2**30
        do_km = gb <= args.max_dense_gb
        rec = {"spec": spec, "n_features": F, "dense_codes_gb": round(gb, 1)}
        X = torch.empty((len(idx), F), device=dev) if do_km else None
        counts = torch.zeros(F, dtype=torch.long, device=dev)
        step = chunk_rows(F)
        with torch.no_grad():
            for i in range(0, len(idx), step):
                h = torch.from_numpy(H[i:i + step]).to(dev)
                t = torch.from_numpy(T[i:i + step]).to(dev) if sae.needs_tokens else None
                codes = sae.encode_raw(h, t)
                counts += (codes > 0).sum(0)
                if do_km:
                    X[i:i + len(codes)] = codes
        c = counts.cpu().numpy()
        np.savez(out / f"counts_{name}.npz", counts=c, n_rows=len(idx), spec=spec)
        rec.update(features_ever_active=int((c > 0).sum()), mean_l0=float(c.sum() / len(idx)))
        if do_km:
            print(f"[ctl-fit] {name}: balanced k-means K={args.n_clusters} on codes {tuple(X.shape)} ({gb:.1f} GB)", flush=True)
            C = balanced_fit(X, args.n_clusters, seed=args.seed, epochs=args.epochs, density_pool=args.density_pool,
                             log=lambda m: print(f"[ctl-fit] {name}{m}", flush=True))
            lab = torch.cat([torch.cdist(X[i:i + 8192], C).argmin(1) for i in range(0, len(X), 8192)])
            use = torch.bincount(lab, minlength=args.n_clusters).float()
            rec.update(live=int((use > 0).sum()), top10_share=float(use.sort(descending=True).values[:10].sum() / use.sum()))
            np.savez(out / f"kmeans_{name}.npz", centroids=C.cpu().numpy().astype(np.float32), spec=spec,
                     n_rows=len(idx), seed=args.seed, n_clusters=args.n_clusters, init="dpc", reinit="peaks",
                     space="sae_codes")
            del X, C
        else:
            rec["kmeans_skipped"] = (f"dense code matrix {gb:.0f} GB > --max_dense_gb {args.max_dense_gb} "
                                     f"(a sparse/subsampled fit would be too few rows for K={args.n_clusters})")
            print(f"[ctl-fit] {name}: k-means SKIPPED: {rec['kmeans_skipped']}", flush=True)
        summary["arms"][name] = rec
        print(f"[ctl-fit] {name}: {rec}", flush=True)
        del sae, counts
        torch.cuda.empty_cache()
        (out / "fit_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
