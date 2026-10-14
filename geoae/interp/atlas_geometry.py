"""Torch helpers for the Atlas sibling test: the three activation spaces, per-pair range geometry, bootstraps.

Definitions are those of eval_out/paper_ranges_along_directions.py, applied to a PAIR of named concepts instead of
a cluster and its best-cosine partner:

  1. centre rows on the space's global mean; mu_c = mean(member rows) - global mean;
  2. cos = <mu_a, mu_b> / (|mu_a| |mu_b|);  shared direction u = normalise(mu_a/|mu_a| + mu_b/|mu_b|);
  3. d' = |m_a - m_b| / sqrt((s_a^2 + s_b^2) / 2) of the members' projections on u (population variance);
  4. radial share = <gap, u>^2 / |gap|^2 with gap = mu_a - mu_b;
  range pair = cos >= 0.9 and d' >= 2.

Difference from the original: the means and u come from the FIT half of the chunks' documents and d' is measured on
the EVAL half (the original measured both on the same rows). The in-sample numbers are returned as well.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

COS, DPRIME = 0.9, 2.0
SPACES = ("base", "token_centred", "geoae")

CKPT = "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt"
TBFILE = "e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz"
KM = "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled.npz"
KMT = "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled_tokmean.npz"
NORM_PARAMS = "activations_sampled_10M/norm_params_layer27.npz"


# ---------------------------------------------------------------------------------------------
# the three spaces
# ---------------------------------------------------------------------------------------------
class Spaces:
    """base x = (h - m)/s ; token-centred xt = (h - m_t)/s_t - b[tok] ; GeoAE z = ae.encode((h - m_a)/s_a, tok).
    Normalisation constants come from the same files paper_ranges_along_directions.py uses (k-means npz for the two
    base spaces, the AE checkpoint for z); __init__ checks they agree with the dump's norm_params file."""

    def __init__(self, device, ckpt=CKPT, km=KM, kmt=KMT, tbfile=TBFILE, norm_params=NORM_PARAMS):
        from geoae.checkpoint import load_ae_checkpoint
        from geoae.token_bias import TokenBiasLookup
        self.dev = torch.device(device)
        self.ae, self.m_a, self.s_a, _ = load_ae_checkpoint(ckpt, self.dev, allow_token_bias=True)
        self.ae.eval()
        assert self.ae.has_token_bias
        d = np.load(km, allow_pickle=True)
        dt = np.load(kmt, allow_pickle=True)
        self.m_b, self.s_b = (torch.as_tensor(d[k], dtype=torch.float32, device=self.dev) for k in ("norm_mean", "norm_std"))
        self.m_t, self.s_t = (torch.as_tensor(dt[k], dtype=torch.float32, device=self.dev) for k in ("norm_mean", "norm_std"))
        assert str(dt["token_bias"]) == tbfile, f"token-mean k-means was fit with {dt['token_bias']}, not {tbfile}"
        self.tb = TokenBiasLookup(tbfile, self.dev)
        self.dim = int(self.ae.centroids.shape[1])
        self.norm_check = {"km_vs_ckpt_max_abs": float((self.m_b - self.m_a).abs().max()),
                           "kmt_vs_ckpt_max_abs": float((self.m_t - self.m_a).abs().max()),
                           "km_std_vs_ckpt_max_abs": float((self.s_b - self.s_a).abs().max())}
        try:
            npz = np.load(norm_params)
            self.norm_check["dump_norm_params_vs_ckpt_max_abs"] = float(
                max(np.abs(npz["mean"] - self.m_a.cpu().numpy()).max(), np.abs(npz["std"] - self.s_a.cpu().numpy()).max()))
        except FileNotFoundError:
            self.norm_check["dump_norm_params_vs_ckpt_max_abs"] = None

    @torch.no_grad()
    def __call__(self, h16, tok, bs: int = 16384) -> dict:
        """h16 (n, 3072) float16 numpy/tensor, tok (n,) ints -> {"base": (n, 3072), "token_centred": (n, 3072), "geoae": (n, d)}."""
        out = {k: [] for k in SPACES}
        h16 = torch.as_tensor(h16)
        tok = torch.as_tensor(tok, dtype=torch.long)
        for s in range(0, len(h16), bs):
            h = h16[s:s + bs].to(self.dev).float()
            t = tok[s:s + bs].to(self.dev)
            out["base"].append((h - self.m_b) / self.s_b)
            out["token_centred"].append((h - self.m_t) / self.s_t - self.tb(t))
            out["geoae"].append(self.ae.encode((h - self.m_a) / self.s_a, t))
        return {k: (torch.cat(v) if v else torch.zeros(0, self.dim if k == "geoae" else 3072, device=self.dev))
                for k, v in out.items()}


def lm_head_fp32(lm):
    """float32 copies of the final norm and lm_head: edited and clean next-token logits share one exact path."""
    from geoae.lm_arch import locate_lm_parts
    _, _, final_norm, lm_head = locate_lm_parts(lm)
    norm32 = copy.deepcopy(final_norm).float().eval()
    head32 = copy.deepcopy(lm_head).float().eval()
    for p in list(norm32.parameters()) + list(head32.parameters()):
        p.requires_grad_(False)
    return norm32, head32


# ---------------------------------------------------------------------------------------------
# per-pair geometry
# ---------------------------------------------------------------------------------------------
def _unit(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return v / v.norm(dim=dim, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _stats_from_means(mu_a, mu_b):
    ua, ub = _unit(mu_a), _unit(mu_b)
    u = _unit(ua + ub)
    gap = mu_a - mu_b
    radial = (gap @ u) ** 2 / (gap @ gap).clamp_min(1e-24)
    return (ua * ub).sum(-1), u, radial


@torch.no_grad()
def dprime_of(pa: torch.Tensor, pb: torch.Tensor) -> float:
    ma, mb = pa.double().mean(), pb.double().mean()
    va, vb = pa.double().var(unbiased=False), pb.double().var(unbiased=False)
    return float((ma - mb).abs() / ((va + vb) / 2).sqrt().clamp_min(1e-12))


@torch.no_grad()
def pair_stats(fa, fb, ea, eb, mean) -> dict:
    """fa, fb: fit-half rows of A-only / B-only chunks (n, D) in one space; ea, eb: eval-half rows; mean (D,): the
    space's global mean. Direction from the fit half, d' on the eval half; in-sample (pooled rows) numbers added."""
    mu_a, mu_b = fa.mean(0, dtype=torch.float64) - mean.double(), fb.mean(0, dtype=torch.float64) - mean.double()
    cos, u, radial = _stats_from_means(mu_a, mu_b)
    uf = u.float()
    pa, pb = (ea - mean) @ uf, (eb - mean) @ uf
    d = dprime_of(pa, pb)
    # eval-half cosine of the centred means (a check that does not reuse the rows that fixed u)
    cos_eval = float(_stats_from_means(ea.mean(0, dtype=torch.float64) - mean.double(), eb.mean(0, dtype=torch.float64) - mean.double())[0])
    # in-sample, as the original script measured it
    A, Bm = torch.cat([fa, ea]), torch.cat([fb, eb])
    cos_i, u_i, rad_i = _stats_from_means(A.mean(0, dtype=torch.float64) - mean.double(), Bm.mean(0, dtype=torch.float64) - mean.double())
    d_i = dprime_of((A - mean) @ u_i.float(), (Bm - mean) @ u_i.float())
    return {"cos": float(cos), "dprime": d, "radial_share": float(radial), "cos_eval": cos_eval,
            "insample_cos": float(cos_i), "insample_dprime": d_i, "insample_radial_share": float(rad_i),
            "mean_a": float(pa.mean()), "mean_b": float(pb.mean()), "norm_ratio": float(torch.maximum(mu_a.norm(), mu_b.norm()) / torch.minimum(mu_a.norm(), mu_b.norm()).clamp_min(1e-12)),
            "n_rows": [int(len(fa)), int(len(fb)), int(len(ea)), int(len(eb))]}


def _doc_sums(X: torch.Tensor, d: torch.Tensor):
    nd = int(d.max()) + 1 if len(d) else 0
    S = torch.zeros(nd, X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, d, X)
    n = torch.bincount(d, minlength=nd).double()
    return S.double(), n, nd


def _weights(nd: int, B: int, gen: torch.Generator, device) -> torch.Tensor:
    idx = torch.randint(0, nd, (B, nd), generator=gen, device=device)
    return torch.zeros(B, nd, dtype=torch.float64, device=device).scatter_add_(1, idx, torch.ones(B, nd, dtype=torch.float64, device=device))


@torch.no_grad()
def bootstrap_pair(fa, fb, ea, eb, mean, B: int, gen: torch.Generator) -> dict:
    """Document-level bootstrap of the whole pair statistic (B replicates at once). Each of fa/fb/ea/eb is a tuple
    (rows (n, D), doc_index (n,) int64 in [0, n_docs)). Documents are resampled with replacement within each of the
    four sets; the fit replicates fix mu_a, mu_b, u; the eval replicates give d'. Returns numpy arrays of length B."""
    dev = mean.device
    m64 = mean.double()
    mus = []
    for X, d in (fa, fb):
        S, n, nd = _doc_sums(X, d)
        W = _weights(nd, B, gen, dev)
        mus.append((W @ S) / (W @ n).clamp_min(1.0)[:, None] - m64)
    ua, ub = _unit(mus[0]), _unit(mus[1])
    cos = (ua * ub).sum(1)
    U = _unit(ua + ub)
    gap = mus[0] - mus[1]
    radial = (gap * U).sum(1) ** 2 / (gap * gap).sum(1).clamp_min(1e-24)
    Uf = U.float()
    mom = []
    for X, d in (ea, eb):
        P = ((X - mean) @ Uf.T).double()                              # (n, B)
        nd = int(d.max()) + 1 if len(d) else 0
        T = torch.zeros(nd, B, dtype=torch.float64, device=dev).index_add_(0, d, P)
        Q = torch.zeros(nd, B, dtype=torch.float64, device=dev).index_add_(0, d, P * P)
        n = torch.bincount(d, minlength=nd).double()
        W = _weights(nd, B, gen, dev)
        cnt = (W @ n).clamp_min(1.0)
        m = (W * T.T).sum(1) / cnt
        v = (W * Q.T).sum(1) / cnt - m * m
        mom.append((m, v.clamp_min(0)))
    dprime = (mom[0][0] - mom[1][0]).abs() / ((mom[0][1] + mom[1][1]) / 2).sqrt().clamp_min(1e-12)
    return {"cos": cos.cpu().numpy(), "dprime": dprime.cpu().numpy(), "radial_share": radial.cpu().numpy()}


# ---------------------------------------------------------------------------------------------
# rows <-> documents / chunks
# ---------------------------------------------------------------------------------------------
def row_docs(cache, chunks) -> torch.Tensor:
    """Document index (0..n_docs-1) of every row of `chunks`, in the row order cache.gather(chunks) returns."""
    per = cache.row_start[chunks + 1] - cache.row_start[chunks]
    docs = np.repeat(cache.doc[chunks], per)
    return torch.as_tensor(np.unique(docs, return_inverse=True)[1], dtype=torch.long)


def chunk_means(cache, chunks, X: torch.Tensor):
    """Pool each chunk's rows (X in the row order cache.gather(chunks) returns) into one mean vector.
    Returns (means (n_chunks_with_rows, D), document index per chunk-mean (0..n_docs-1))."""
    per = cache.row_start[chunks + 1] - cache.row_start[chunks]
    keep = per > 0
    seg = torch.as_tensor(np.repeat(np.arange(len(chunks)), per), dtype=torch.long, device=X.device)
    S = torch.zeros(len(chunks), X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, seg, X)
    n = torch.as_tensor(per, dtype=X.dtype, device=X.device).clamp_min(1)[:, None]
    docs = np.unique(cache.doc[chunks][keep], return_inverse=True)[1]
    return (S / n)[torch.as_tensor(keep, device=X.device)], torch.as_tensor(docs, dtype=torch.long, device=X.device)


# ---------------------------------------------------------------------------------------------
# aggregation over pairs (numpy)
# ---------------------------------------------------------------------------------------------
def boot_ci(values, stat, B: int = 2000, seed: int = 0):
    """(point, lo, hi): percentile bootstrap of stat(values) resampling the PAIRS (rows of `values`)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (B, len(v)))
    s = stat(v[idx], axis=1)
    return float(stat(v)), float(np.nanquantile(s, 0.025)), float(np.nanquantile(s, 0.975))


def summarise_group(cos, dprime, radial, B: int = 2000, seed: int = 0) -> dict:
    """Share of range pairs and medians (with pair-bootstrap 95% CIs) for one group of pairs in one space."""
    cos, dprime, radial = (np.asarray(x, dtype=np.float64) for x in (cos, dprime, radial))
    ok = np.isfinite(cos) & np.isfinite(dprime) & np.isfinite(radial)
    cos, dprime, radial = cos[ok], dprime[ok], radial[ok]
    rng_flag = ((cos >= COS) & (dprime >= DPRIME)).astype(np.float64)
    par = cos >= COS
    out = {"n": int(ok.sum()), "n_dropped_nonfinite": int((~ok).sum())}
    for nm, v, st in (("range_share", rng_flag, np.mean), ("parallel_share", par.astype(float), np.mean),
                      ("dprime_ge_2_share", (dprime >= DPRIME).astype(float), np.mean),
                      ("cos_median", cos, np.median), ("dprime_median", dprime, np.median),
                      ("radial_share_median", radial, np.median)):
        p, lo, hi = boot_ci(v, st, B, seed)
        out[nm] = {"value": p, "ci": [lo, hi]}
    # exploratory only: the share of pairs under softer thresholds (the registered range pair is cos >= 0.9 and d' >= 2)
    out["threshold_sweep_share"] = {f"cos>={c:g},d'>={d:g}": float(np.mean((cos >= c) & (dprime >= d))) if len(cos) else None
                                    for c in (0.5, 0.7, 0.9) for d in (0.5, 1.0, 2.0)}
    out["range_pairs"] = int(rng_flag.sum())
    out["parallel_pairs"] = int(par.sum())
    for nm, v in (("dprime_median_among_parallel", dprime[par]), ("radial_share_median_among_parallel", radial[par])):
        out[nm] = float(np.median(v)) if len(v) else None
    return out


def paired_diff(a_flags, b_flags, B: int = 2000, seed: int = 0) -> dict:
    """Mean of (a_i - b_i) over matched pairs, with a bootstrap CI over the matched index i."""
    d = np.asarray(a_flags, dtype=np.float64) - np.asarray(b_flags, dtype=np.float64)
    p, lo, hi = boot_ci(d, np.mean, B, seed)
    return {"mean_diff": p, "ci": [lo, hi], "n": int(len(d))}
