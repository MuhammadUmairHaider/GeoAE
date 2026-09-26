"""
Token-identity erasure: a fixed linear projection that removes the directions of
the residual stream carrying WHICH TOKEN a row is (and, at the last block, which
token the model is about to predict), so a clustering of what is left has to
organise by context instead.

Why a projection and not per-token mean subtraction. Subtracting E[h | token] is
exact, but it needs the token id at assignment time, and the sequence-level probe
caches (topic, sentiment, domain, language, atlas) store only H_last / H_mean.
A projection needs nothing but h, so every rung can be scored.

Basis. Eigenvectors of the count-weighted between-class scatter of the z-scored
activations,

    S_B = sum_t n_t (mu_t - mu)(mu_t - mu)^T / N ,

summed over the requested keys:
    cur   the current token (rows_tok.npy beside the dump)
    pred  the LM's argmax next token, from final norm + lm_head. Only defined when
          the dump layer is the LAST decoder block (L27 of Llama-3.2-3B), where
          that is exactly the model's own prediction.
Only classes with >= min_count rows enter. The class means carry estimation
noise, which inflates S_B by ~(T-1)/N * S_W for T classes; that term is
subtracted before the eigendecomposition.

Control. The top principal components of the TOTAL covariance are saved as a
second basis ("pca"). Removing r dominant directions changes k-means whether or
not they carry token identity, so a token-erased fit must beat the same-rank pca
fit before the gain can be credited to erasing tokens.

Diagnostics, on the held-out val tail (never in the fit sample): for each basis
and rank, the variance removed and the out-of-sample R^2 of current token,
predicted token (class means from the fit sample) and document (means from the
even positions of each document, scored on the odd ones), all in the projected
space.

Usage (L27, new dump; ~10 min, reads 1M rows and loads the LM for `pred`):
    python -u -m geoae.interp.token_erasure \
      --checkpoint checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt \
      --activations_dir activations_sampled_10M \
      --out e2e/checkpoints/general/llama3.2-3B/layer27/token_erasure_sampled.npz

Then fit_balanced_kmeans --erase <out> --erase_basis token|pca --erase_rank R,
and score with concept_probe (the only consumer that applies the projection;
every other tool refuses an erased codebook).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geoae.interp.clustering_quality import read_rows
from geoae.seeding import seed_everything

RANKS = (16, 32, 64, 128, 256, 512)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

@torch.no_grad()
def class_scatter(Z: torch.Tensor, key: np.ndarray, min_count: int):
    """Noise-corrected between-class scatter of Z over classes with >= min_count rows.

    Returns (S_B (D, D) float64, classes (T,), means (T, D) float32, counts (T,)).
    Means are uncentred class means of Z, so they can be projected and reused.
    """
    classes, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep_cls = counts >= min_count
    if keep_cls.sum() < 2:
        raise ValueError(f"fewer than 2 classes with >= {min_count} rows")
    remap = np.full(len(classes), -1, dtype=np.int64)
    remap[keep_cls] = np.arange(int(keep_cls.sum()))
    lab_t = torch.from_numpy(remap[inv]).to(Z.device)
    T, D = int(keep_cls.sum()), Z.shape[1]
    n = torch.from_numpy(counts[keep_cls]).to(Z.device, torch.float64)
    N = int(n.sum())
    # Chunked: a masked float64 copy of a 1M x 3072 sample alone is ~25 GB.
    sums = torch.zeros(T, D, device=Z.device, dtype=torch.float64)
    M2 = torch.zeros(D, D, device=Z.device, dtype=torch.float64)
    for i in range(0, len(Z), 65536):
        lb = lab_t[i:i + 65536]
        m = lb >= 0
        blk = Z[i:i + 65536][m].double()
        sums.index_add_(0, lb[m], blk)
        M2 += blk.T @ blk
    means = sums / n[:, None]
    mu = sums.sum(0) / N
    dM = means - mu
    S_B = (dM * n[:, None]).T @ dM / N
    S_T = M2 / N - torch.outer(mu, mu)
    S_W = S_T - S_B
    S_B = S_B - (T - 1) / N * S_W
    return S_B, classes[keep_cls], means.float(), counts[keep_cls]


def top_eigvecs(S: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k eigenpairs of a symmetric matrix, descending. Returns (U (D, k), eigvals (k,))."""
    ev, V = torch.linalg.eigh(S)
    order = torch.argsort(ev, descending=True)[:k]
    return V[:, order].float(), ev[order].float()


def project_out(Z: torch.Tensor, U: torch.Tensor | None) -> torch.Tensor:
    """Z - (Z U) U^T; U has orthonormal columns. U None or empty = identity."""
    if U is None or U.shape[1] == 0:
        return Z
    return Z - (Z @ U) @ U.T


@torch.no_grad()
def r2_from_means(Z: torch.Tensor, key: np.ndarray, classes: np.ndarray, means: torch.Tensor):
    """Out-of-sample R^2 of class means fit elsewhere, over rows whose class was fit.

    Z and means must be in the same (possibly projected) space. Baseline is the
    mean of the scored rows. Returns (r2, fraction of rows scored).
    """
    pos = np.searchsorted(classes, key)
    pos = np.clip(pos, 0, len(classes) - 1)
    ok = classes[pos] == key
    if not ok.any():
        return float("nan"), 0.0
    okt = torch.from_numpy(ok).to(Z.device)
    Zs = Z[okt]
    M = means[torch.from_numpy(pos[ok]).to(Z.device)]
    res = (Zs - M).pow(2).sum()
    tot = (Zs - Zs.mean(0)).pow(2).sum()
    return float(1 - res / tot), float(ok.mean())


@torch.no_grad()
def r2_split(Z: torch.Tensor, key: np.ndarray, fit: np.ndarray, min_count: int = 3):
    """R^2 of per-key means fit on rows `fit`, scored on rows ~fit (e.g. doc, by position parity)."""
    classes, inv = np.unique(key, return_inverse=True)
    inv_t = torch.from_numpy(inv).to(Z.device)
    fit_t = torch.from_numpy(fit).to(Z.device)
    S = torch.zeros(len(classes), Z.shape[1], device=Z.device).index_add_(0, inv_t[fit_t], Z[fit_t])
    n = torch.bincount(inv_t[fit_t], minlength=len(classes)).float()
    ok = ~fit_t & (n[inv_t] >= min_count)
    Zs = Z[ok]
    res = (Zs - (S / n.clamp_min(1)[:, None])[inv_t[ok]]).pow(2).sum()
    tot = (Zs - Zs.mean(0)).pow(2).sum()
    return float(1 - res / tot)


# ---------------------------------------------------------------------------

@torch.no_grad()
def predicted_next(lm, Z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                   chunk: int = 8192) -> np.ndarray:
    """argmax next token from z-scored last-block residuals: denormalise, final norm +
    lm_head (as the LM does). Chunked, so the raw residuals are never all resident."""
    norm = getattr(getattr(lm, "model", None), "norm", None)
    if norm is None:
        raise SystemExit("[erasure] this LM has no model.norm; `pred` is only wired for Llama-style LMs")
    head = lm.get_output_embeddings()
    out = []
    for i in range(0, len(Z), chunk):
        x = (Z[i:i + chunk] * std + mean).to(head.weight.device, head.weight.dtype)
        out.append(head(norm(x)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="AE checkpoint: supplies norm stats, layer and model name. Use the SAME one "
                         "passed to fit_balanced_kmeans, or the fit refuses the erasure.")
    ap.add_argument("--activations_dir", required=True, help="dump dir with layer_L.npy + rows_tok.npy")
    ap.add_argument("--keys", default="cur,pred", help="comma list of cur, pred: what DEFINES the basis")
    ap.add_argument("--diag_keys", default="cur,pred",
                    help="keys whose R^2 is REPORTED, whether or not they define the basis, so a "
                         "cur-only basis shows how much predicted-next identity it also removes")
    ap.add_argument("--n_sample", type=int, default=1_000_000, help="train-split rows for the scatter")
    ap.add_argument("--n_eval", type=int, default=200_000, help="val-tail rows for the diagnostics")
    ap.add_argument("--val_frac", type=float, default=0.05, help="must match training's val split")
    ap.add_argument("--min_count", type=int, default=30)
    ap.add_argument("--max_rank", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = [k for k in args.keys.split(",") if k]
    if not set(keys) <= {"cur", "pred"} or not keys:
        raise SystemExit(f"[erasure] --keys must be a subset of cur,pred; got {args.keys}")
    dkeys = list(dict.fromkeys(keys + [k for k in args.diag_keys.split(",") if k in ("cur", "pred")]))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(ckpt["norm_mean"], dtype=np.float32)
    std = np.asarray(ckpt["norm_std"], dtype=np.float32)
    layer, model_name = ckpt["config"]["data"]["target_layer"], ckpt["config"]["extraction"]["model_name"]
    act_dir = Path(args.activations_dir)
    npy = act_dir / f"layer_{layer}.npy"
    tok = np.load(act_dir / "rows_tok.npy") if "cur" in dkeys else None
    N = np.load(str(npy), mmap_mode="r").shape[0]
    val_start = int(N * (1.0 - args.val_frac))
    lm = None
    if "pred" in dkeys:
        from geoae.checkpoint import load_lm
        from geoae.lm_arch import decoder_layers
        lm = load_lm(model_name, device=dev)
        n_blocks = len(decoder_layers(lm))
        if layer != n_blocks - 1:
            if "pred" in keys:
                raise SystemExit(f"[erasure] `pred` needs the last block ({n_blocks - 1}); dump is layer "
                                 f"{layer}. Use --keys cur.")
            print(f"[erasure] ! layer {layer} is not the last block: dropping `pred` from diagnostics")
            dkeys.remove("pred")
            lm = None
    print(f"[erasure] {npy}  N={N:,}  train rows [0, {val_start:,})  basis keys={keys}  reported={dkeys}")

    mean_t, std_t = torch.from_numpy(mean).to(dev), torch.from_numpy(std).to(dev)

    def load_z(idx):
        Z = torch.empty(len(idx), len(mean), device=dev)
        for s in range(0, len(idx), 100_000):
            blk = torch.from_numpy(read_rows(npy, idx[s:s + 100_000]).astype(np.float32)).to(dev)
            Z[s:s + len(blk)] = (blk - mean_t) / std_t
        return Z

    # ---- fit sample: train split only ----------------------------------------
    rng = np.random.RandomState(args.seed)
    idx = np.sort(rng.choice(val_start, size=min(args.n_sample, val_start), replace=False))
    print(f"[erasure] reading {len(idx):,} train rows …")
    Z = load_z(idx)
    key_rows = {}
    if "cur" in dkeys:
        key_rows["cur"] = tok[idx]
    if "pred" in dkeys:
        key_rows["pred"] = predicted_next(lm, Z, mean_t, std_t)

    S_tok = torch.zeros(Z.shape[1], Z.shape[1], device=dev, dtype=torch.float64)
    fitted = {}
    for k, kr in key_rows.items():
        S_B, cls, means, cnt = class_scatter(Z, kr, args.min_count)
        fitted[k] = (cls, means)
        cover = cnt.sum() / len(kr)
        print(f"[erasure]   {k}{' (basis)' if k in keys else ' (reported only)'}: {len(cls):,} classes "
              f">= {args.min_count} rows cover {cover:.0%} of rows; "
              f"between-class share of variance {float(S_B.trace() / Z.var(0).sum()):.3f}")
        if k in keys:
            S_tok += S_B
    U_tok, ev_tok = top_eigvecs(S_tok, args.max_rank)
    Zc = Z - Z.mean(0)
    C_tot = torch.zeros_like(S_tok)
    for i in range(0, len(Zc), 65536):
        blk = Zc[i:i + 65536].double()
        C_tot += blk.T @ blk
    C_tot /= len(Zc)
    U_pca, ev_pca = top_eigvecs(C_tot, args.max_rank)
    total_var = float(C_tot.trace())
    del Z, Zc
    torch.cuda.empty_cache()

    # ---- diagnostics on the val tail (contiguous, so documents stay whole) ---------
    ev_idx = np.arange(val_start, min(N, val_start + args.n_eval))
    Ze = load_z(ev_idx)
    ek = {}
    if "cur" in dkeys:
        ek["cur"] = tok[ev_idx]
    if "pred" in dkeys:
        ek["pred"] = predicted_next(lm, Ze, mean_t, std_t)
    doc = np.load(act_dir / "rows_doc.npy")[ev_idx] if (act_dir / "rows_doc.npy").exists() else None
    pos = np.load(act_dir / "rows_pos.npy")[ev_idx] if (act_dir / "rows_pos.npy").exists() else ev_idx
    tot_e = float((Ze - Ze.mean(0)).pow(2).sum())

    report = []
    hdr = (f"{'basis':>6s} {'rank':>5s} | {'var removed':>11s} | "
           + " | ".join(f"{'R2 ' + k:>8s}" for k in ek) + (f" | {'R2 doc':>7s} | {'doc var kept':>12s}" if doc is not None else ""))
    print(f"\n[erasure] held-out diagnostics on {len(ev_idx):,} val rows (projected space)\n{hdr}")
    doc_between0 = None
    for basis, U_all in (("none", None), ("token", U_tok), ("pca", U_pca)):
        for r in ((0,) if basis == "none" else [r for r in RANKS if r <= args.max_rank]):
            U = None if U_all is None else U_all[:, :r]
            Zp = project_out(Ze, U)
            tot_p = float((Zp - Zp.mean(0)).pow(2).sum())
            row = dict(basis=basis, rank=r, var_removed=1 - tot_p / tot_e)
            for k, kr in ek.items():
                cls, means = fitted[k]
                row[f"r2_{k}"], row[f"cover_{k}"] = r2_from_means(Zp, kr, cls, project_out(means, U))
            if doc is not None:
                row["r2_doc"] = r2_split(Zp, doc, pos % 2 == 0)
                between = row["r2_doc"] * tot_p
                doc_between0 = between if doc_between0 is None else doc_between0
                row["doc_var_kept"] = between / doc_between0
            report.append(row)
            print(f"{basis:>6s} {r:5d} | {row['var_removed']:11.3f} | "
                  + " | ".join(f"{row[f'r2_{k}']:8.3f}" for k in ek)
                  + (f" | {row['r2_doc']:7.3f} | {row['doc_var_kept']:12.3f}" if doc is not None else ""))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), U_token=U_tok.cpu().numpy(), eig_token=ev_tok.cpu().numpy(),
             U_pca=U_pca.cpu().numpy(), eig_pca=ev_pca.cpu().numpy(), total_var=total_var,
             norm_mean=mean, norm_std=std, layer=layer, model_name=model_name,
             keys=",".join(keys), min_count=args.min_count, n_sample=len(idx),
             activations=str(npy), checkpoint=args.checkpoint, report=json.dumps(report))
    print(f"\n[erasure] Saved -> {out}")


if __name__ == "__main__":
    main()
