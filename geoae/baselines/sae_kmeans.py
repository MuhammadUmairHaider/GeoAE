"""Plain Lloyd k-means (K=2000) on an SAE's native code vectors (all features), fitted on exactly the training rows of the
plain residual baselines (experiments/paper/analysis/paper_plain_kmeans_grid.py): every 6th row of the training 95% of the dump, density-peaks
seeds from the same 32,768-row pool draw (torch.randperm, seed 0, as the Gemma plain fits) with the AE's settings, no
balancing, no re-seeding, empty clusters left in place, <= 100 iterations, stop when < 1e-4 of the labels change.

The space is the SAE's raw codes (no standardisation): Gemma Scope reads the raw residual. Dense codes of 1.58M rows x 16,384
features would be ~100 GB, so the fit keeps them sparse on the GPU (CSR: crow / col / val; L0 ~24) and computes distances
as |x|^2 - 2 x.c + |c|^2 with a sparse x dense product. Codes are encoded in full fp32 (TF32 off), as at evaluation time.

The saved codebook has the `centroids` field geoae/interp/sae_space.load_sae_kmeans reads, so the Atlas rows are labelled
by geoae/evaluation/atlas_assign.py --sae_kmeans "<name>=<sae spec>@<codebook>" (nearest centroid of the row's codes).

    python -u geoae/baselines/sae_kmeans.py --activations_dir activations_gemma3_4b_sampled_struct5_10M --layer 33 \
        --sae scope:16k --codebook results/delta/g4b_l33_d5120/evals/sae_suite/controls/plain_kmeans_gs16k.npz \
        --balanced results/delta/g4b_l33_d5120/evals/sae_suite/controls/kmeans_gs16k.npz
Writes --codebook and <codebook stem>_fit.json (fit history, usage of the plain and balanced code codebooks on the
training rows and the held-out 5%).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geoae.baselines.sae_space import load_sae_space  # noqa: E402
from geoae.training.initialization import density_peaks_select  # noqa: E402

TOL = 1e-4
BLOCK = 65536          # rows per slab read (x stride) and per assignment chunk
ENC = 16384            # rows per dense encode chunk (16,384 x 16,384 fp32 = 1 GiB per temporary)


class SparseRows:
    """Rows of non-negative SAE codes as one CSR on the device: crow (n+1,), col / val (nnz,), sq (n,) = |x|^2."""

    def __init__(self, crow, col, val, n_features):
        self.crow, self.col, self.val, self.F = crow, col, val, int(n_features)
        self.n = len(crow) - 1
        self.row = torch.repeat_interleave(torch.arange(self.n, device=crow.device), crow.diff())
        self.sq = torch.zeros(self.n, device=val.device).index_add_(0, self.row, val * val)

    @classmethod
    def from_dense_chunks(cls, chunks, n_features):
        crows, cols, vals, off = [torch.zeros(1, dtype=torch.long, device=chunks[0].device)], [], [], 0
        for x in chunks:
            s = x.to_sparse_csr()
            crows.append(s.crow_indices()[1:].long() + off)
            cols.append(s.col_indices().long())
            vals.append(s.values().float())
            off += int(s.crow_indices()[-1])
        return cls(torch.cat(crows), torch.cat(cols), torch.cat(vals), n_features)

    @classmethod
    def concat(cls, parts, n_features):
        crows, off = [parts[0].crow[:1]], 0
        for p in parts:
            crows.append(p.crow[1:] + off)
            off += int(p.crow[-1])
        return cls(torch.cat(crows), torch.cat([p.col for p in parts]), torch.cat([p.val for p in parts]), n_features)

    def block(self, i, j):
        lo, hi = int(self.crow[i]), int(self.crow[j])
        return torch.sparse_csr_tensor(self.crow[i:j + 1] - lo, self.col[lo:hi], self.val[lo:hi], size=(j - i, self.F))

    def dense(self, idx):
        """Dense rows idx (in the order given)."""
        pos = torch.full((self.n,), -1, dtype=torch.long, device=self.crow.device)
        pos[idx] = torch.arange(len(idx), device=pos.device)
        keep = pos[self.row] >= 0
        out = torch.zeros(len(idx), self.F, device=self.val.device)
        out.index_put_((pos[self.row[keep]], self.col[keep]), self.val[keep])
        return out


def assign(X: SparseRows, C: torch.Tensor, chunk: int = BLOCK):
    """Nearest centroid (squared Euclidean) of every row -> (labels, mean squared distance)."""
    c2 = (C * C).sum(1)
    lab = torch.empty(X.n, dtype=torch.long, device=C.device)
    inertia = 0.0
    for i in range(0, X.n, chunk):
        j = min(i + chunk, X.n)
        m, a = (c2[None, :] - 2.0 * (X.block(i, j) @ C.T)).min(1)
        lab[i:j] = a
        inertia += float((m + X.sq[i:j]).sum())
    return lab, inertia / X.n


def lloyd(X: SparseRows, C: torch.Tensor, max_iter: int):
    C = C.float().clone()
    K, F = C.shape
    prev, hist = None, []
    for it in range(max_iter):
        lab, inertia = assign(X, C)
        cnt = torch.bincount(lab, minlength=K)
        changed = 1.0 if prev is None else float((lab != prev).float().mean())
        hist.append({"iter": it, "inertia": inertia, "changed": changed, "live": int(cnt.gt(0).sum())})
        if prev is not None and changed < TOL:
            break
        prev = lab
        sums = torch.zeros(K * F, device=C.device).index_add_(0, lab[X.row] * F + X.col, X.val).view(K, F)
        live = cnt > 0
        C[live] = sums[live] / cnt[live, None].float()
    return C, hist


def usage(lab, K) -> dict:
    counts = torch.bincount(torch.as_tensor(lab).long(), minlength=K).double()
    p = counts / counts.sum()
    s = counts.sort(descending=True).values
    nz = p[p > 0]
    return {"live_clusters": int((counts > 0).sum()), "empty_clusters": int((counts == 0).sum()),
            "top10_share": float(s[:10].sum() / counts.sum()),
            "eff_k_exp_entropy": float(math.exp(-(nz * nz.log()).sum())),
            "eff_k_inverse_simpson": float(1.0 / (nz * nz).sum()),
            "clusters_for_half_the_tokens": int((s.cumsum(0) < 0.5 * counts.sum()).sum()) + 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sae", required=True, help="SAE spec of geoae/baselines/sae_space.py, e.g. scope:16k")
    ap.add_argument("--codebook", required=True, help="where to save the centroids .npz")
    ap.add_argument("--balanced", default="", help="optional balanced code k-means .npz (sae_controls_fit) for usage")
    ap.add_argument("--n_clusters", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    t0 = time.time()

    def log(*a):
        print(time.strftime("[%H:%M:%S]"), *a, f"({time.time() - t0:.0f}s)", flush=True)

    sae = load_sae_space(args.sae, dev, "sae")
    F, K = sae.n_features, args.n_clusters
    D = Path(args.activations_dir)
    A = np.load(D / f"layer_{args.layer}.npy", mmap_mode="r")
    TOK = np.load(D / "rows_tok.npy", mmap_mode="r")
    N = A.shape[0]
    v0 = int(N * 0.95)

    def encode_range(lo: int, hi: int, stride: int) -> SparseRows:
        """Codes of dump rows lo, lo + stride, ... < hi, read in contiguous slabs (fast on GPFS), kept sparse."""
        assert lo % stride == 0
        parts, pending = [], []
        for s in range(lo, hi, stride * BLOCK):
            h = np.asarray(A[s:min(s + stride * BLOCK, hi)])[::stride]
            t = np.asarray(TOK[s:min(s + stride * BLOCK, hi)])[::stride]
            for i in range(0, len(h), ENC):
                hh = torch.from_numpy(np.ascontiguousarray(h[i:i + ENC], dtype=np.float32)).to(dev)
                tt = torch.from_numpy(np.asarray(t[i:i + ENC], dtype=np.int64)).to(dev) if sae.needs_tokens else None
                pending.append(sae.encode_raw(hh, tt))
                if len(pending) == 8:      # 8 x 1 GiB dense chunks -> CSR
                    parts.append(SparseRows.from_dense_chunks(pending, F))
                    pending = []
        if pending:
            parts.append(SparseRows.from_dense_chunks(pending, F))
        X = SparseRows.concat(parts, F)
        assert X.n == len(range(lo, hi, stride)), (X.n, lo, hi, stride)
        return X

    res = {"meta": {"sae": args.sae, "n_features": F, "activations_dir": str(D), "layer": args.layer, "K": K,
                    "stride": args.stride, "max_iter": args.max_iter, "tol": TOL, "seed": args.seed,
                    "space": "raw SAE codes, all features, no standardisation", "fit": "plain Lloyd",
                    "fit_rows": f"every {args.stride}th row of [0, {v0:,})", "heldout_rows": f"[{v0:,}, {N:,})"}}

    # ---- fit
    X = encode_range(0, v0, args.stride)
    log(f"encoded {X.n:,} training rows: {X.val.numel():,} non-zero codes (mean L0 {X.val.numel() / X.n:.1f}), "
        f"{sum(t.numel() * t.element_size() for t in (X.crow, X.col, X.val, X.row, X.sq)) / 2**30:.2f} GiB")
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    perm = torch.randperm(X.n, generator=g)
    C0, _ = density_peaks_select(X.dense(perm[:32768].to(dev)), K, seed=args.seed, density_power=1.0,
                                 min_sep_frac=0.25, knn_k=32, refine_k=8)
    log("density-peaks seeds", tuple(C0.shape))
    C, hist = lloyd(X, C0, args.max_iter)
    log(f"Lloyd: {len(hist)} iterations, last changed {hist[-1]['changed']:.5f}, live {hist[-1]['live']}")
    Cb = (torch.as_tensor(np.load(args.balanced)["centroids"], dtype=torch.float32, device=dev)
          if args.balanced else None)
    res["fit"] = {"iterations": len(hist), "last_changed": hist[-1]["changed"], "history": hist,
                  "train_usage_plain": usage(assign(X, C)[0], K)}
    if Cb is not None:
        res["fit"]["train_usage_balanced"] = usage(assign(X, Cb)[0], len(Cb))
    del X
    torch.cuda.empty_cache()

    cb = Path(args.codebook)
    cb.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cb), centroids=C.cpu().numpy().astype(np.float32), n_clusters=K, layer=args.layer, sae=args.sae,
             init="dpc", reinit="none", space="sae_codes", fit="plain Lloyd", stride=args.stride, seed=args.seed,
             activations_dir=str(D))
    log("saved", cb)

    # ---- held-out usage
    Xh = encode_range(v0, N, 1)
    res["heldout_usage"] = {"plain": usage(assign(Xh, C)[0], K)}
    if Cb is not None:
        res["heldout_usage"]["balanced"] = usage(assign(Xh, Cb)[0], len(Cb))
    log("held-out usage", json.dumps(res["heldout_usage"]))
    out = cb.with_name(cb.stem + "_fit.json")
    out.write_text(json.dumps(res, indent=1))
    log("wrote", out)


if __name__ == "__main__":
    main()
