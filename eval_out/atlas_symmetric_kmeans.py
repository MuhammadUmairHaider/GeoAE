r"""Symmetric k-means readout on the FineWeb-Atlas cache: every representation space gets the IDENTICAL readout.

Why. The earlier Atlas comparisons read each space out its own way (a codebook trained in the space, or an SAE's top-1
feature, or k-means on SAE codes fitted once at K=2000). Here every space is read out by the same two procedures, fitted
on the same training-dump rows, so a difference between spaces cannot come from the readout.

Spaces (continuous coordinates exactly as atlas_token_detail.embed gives them; NOT the spaces' trained codebooks):
  base          (h - mean) / std                      base_tokmean   (h - mean) / std - b[tok]
  bypass        the token-bypass GeoAE's latent z     mat10k / gs16k raw SAE codes (Matryoshka-10k, Gemma Scope 16k)

Readout 1, balanced k-means (`geoae.interp.sae_controls.balanced_fit`, the procedure behind the base / base_tokmean
codebooks and the SAE code k-means), one codebook per K and space. The schedule is scaled with K so that every K runs
the same number of epochs, steps per epoch and steps (rows_per_centroid rows per centroid, batch = batch_per_k2000 x K / 2000):
      K     rows (train split)   batch   steps/epoch x epochs
    2000      200,000            4096        48 x 40
    4000      400,000            8192        48 x 40
    8000      800,000           16384        48 x 40
The rows of a K are `np.sort(np.random.RandomState(seed).choice(n_train, n_rows, replace=False))` over the first
`train_frac` of the dump, so the K=2000 rows are exactly those of eval_out/sae_controls_fit.py (asserted) and the sets are
nested (the K=2000 rows are a subset of the K=4000 rows, ...).

Readout 2, "unit readout" (the reverse check: treat every coordinate of a space as a unit, as an SAE feature is). On the
K=2000 training rows, usage_j = number of rows whose argmax over the space's coordinates is unit j. Units are ordered by
usage (descending, ties -> lower id). The label of a row with N > 0 is the argmax over its coordinates restricted to the
N most-used units (ties -> lower unit id); N = 0 (or N >= the dimension) uses every unit. For the non-negative SAE codes a
row with no positive coordinate among the allowed units has no active unit and gets label -1 (the same convention as
SAESpace.top1_labels, which atlas_token_detail's SAE labels use; for N = 0 the labels are exactly the SAE's native top-1).
Signed spaces (base, base_tokmean, bypass) never get -1.

Outputs in --out_dir:
  codebooks/km{K}_{space}.npz        centroids (K, dim) float32 + the fit's settings and its stats on the training rows
  codebooks/unit_usage_{space}.npz   usage / order of the units
  assign.npz                         int32 label per Atlas cache row: km{K}_{space}, unit{N}_{space} (unitall_ for N=0),
                                     plus row_chunk, row_pos, row_tok, chunk_doc, chunk_ntok EXACTLY as
                                     atlas_token_detail.py writes them (so atlas_concept_stats & co. read it unchanged)
  meta.json                          args, schedule, fit stats, unit usage, agreement with the existing SAE code
                                     k-means, and per label array on the Atlas SAMPLED rows (row_pos != last position of
                                     its chunk): live clusters, effective K = exp(entropy of the usage), unassigned share
Resume: meta.json is rewritten after every finished piece of work and every codebook is saved atomically the moment it is
fitted. A rerun skips a codebook that exists AND matches the requested fit (K, rows, seed, batch, epochs, spec, ...); a
codebook that exists but does not match is an error (--refit overwrites). Unit usage, the agreement check and the Atlas
labelling are cheap and always recomputed.

    python -u eval_out/atlas_symmetric_kmeans.py --cache C --rows_dir SHM_ATLAS --activations_dir SHM_DUMP \
        --models bypass=<ckpt> --baselines base=<npz>,base_tokmean=<npz> \
        --saes "mat10k=matched:<last.pt>,gs16k=scope:16k" --out_dir OUT
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geoae.interp import atlas_siblings as AS  # noqa: E402
from geoae.interp.sae_controls import balanced_fit  # noqa: E402
from geoae.interp.sae_space import chunk_rows, parse_saes  # noqa: E402
from atlas_token_detail import embed  # noqa: E402

DEFAULT_SPACES = "base,base_tokmean,bypass,mat10k,gs16k"
REF_K = 2000                    # the K that batch_per_k2000 refers to
LABEL_CHUNK = 8192              # Atlas rows embedded + labelled at a time
DUMP_READ_CHUNK = 50_000        # sorted dump rows read per slice (as eval_out/sae_controls_fit.py)
STAT_KEYS = ("live", "top10_share", "eff_k", "seconds", "embed_seconds", "fit_seconds", "gpu_peak_gb")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------------------------------------- pure helpers
def parse_ints(text, what, lo):
    """'2000,4000' -> [2000, 4000] (order kept, duplicates dropped); every value must be >= lo."""
    try:
        vals = [int(v) for v in str(text).split(",") if v.strip()]
    except ValueError:
        raise SystemExit(f"{what}: expected comma-separated integers, got {text!r}")
    if not vals:
        raise SystemExit(f"{what} is empty")
    if min(vals) < lo:
        raise SystemExit(f"{what}: every value must be >= {lo}, got {vals}")
    return list(dict.fromkeys(vals))


class Schedule(NamedTuple):
    K: int
    n_rows: int
    batch_size: int
    steps_per_epoch: int


def k_schedule(K, rows_per_centroid, batch_per_k2000):
    """Rows and batch size of the K-cluster fit: rows_per_centroid x K rows, batch = batch_per_k2000 x K / 2000, so
    that every K runs the same steps per epoch (n_rows // batch) and the same total number of steps."""
    if (batch_per_k2000 * K) % REF_K:
        raise ValueError(f"batch_per_k2000 x K = {batch_per_k2000} x {K} is not a multiple of {REF_K}: "
                         f"the batch size at K={K} would not be an integer")
    batch = batch_per_k2000 * K // REF_K
    n_rows = rows_per_centroid * K
    if batch < K:
        raise ValueError(f"batch size {batch} < K={K}: balanced_fit's Sinkhorn column marginal needs batch >= K")
    if n_rows < batch:
        raise ValueError(f"{n_rows} rows < batch size {batch} at K={K}")
    return Schedule(K, n_rows, batch, n_rows // batch)


def sample_rows(n_train, n_rows, seed):
    """Sorted dump-row indices of one fit. The call of eval_out/sae_controls_fit.py (n_rows = its n_sample = 200,000 at
    K=2000); the schedule must not be silently capped, so too small a train split is an error."""
    if n_rows > n_train:
        raise SystemExit(f"{n_rows:,} training rows requested but the train split has only {n_train:,}")
    return np.sort(np.random.RandomState(seed).choice(n_train, n_rows, replace=False))


def sae_controls_rows(n_train, seed, n_sample=200_000):
    """The row sample of eval_out/sae_controls_fit.py, verbatim (its default n_sample)."""
    return np.sort(np.random.RandomState(seed).choice(n_train, min(n_sample, n_train), replace=False))


def fingerprint(idx):
    """sha1 of the sorted row indices: pins the fit's training rows."""
    return hashlib.sha1(np.ascontiguousarray(idx, dtype="<i8").tobytes()).hexdigest()


def km_key(K, space):
    return f"km{K}_{space}"


def unit_key(N, space):
    return f"unit{'all' if N == 0 else N}_{space}"


def count_stats(counts):
    """Live clusters, share of the 10 largest, and effective K = exp(entropy of the usage) of cluster usage counts."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return dict(live=0, top10_share=0.0, eff_k=0.0)
    nz = counts[counts > 0]
    p = nz / total
    return dict(live=int(len(nz)), top10_share=float(np.sort(counts)[::-1][:10].sum() / total),
                eff_k=float(np.exp(-(p * np.log(p)).sum())))


def label_stats(lab, mask=None):
    """count_stats of the non-negative labels of `lab` (restricted to `mask`) + the share of -1 (unassigned) labels."""
    lab = np.asarray(lab)
    if mask is not None:
        lab = lab[mask]
    ok = lab >= 0
    r = count_stats(np.bincount(lab[ok])) if ok.any() else count_stats([])
    return dict(n_rows=int(len(lab)), **r, unassigned=float(1.0 - ok.mean()) if len(lab) else 0.0)


def usage_order(usage):
    """Units by usage, most used first; ties -> lower unit id (the convention of sae_controls.top_features)."""
    return np.argsort(-np.asarray(usage), kind="stable")


def unit_sets(order, ns, dim):
    """{N: ascending ids of the N most-used units, or None for 'every unit'} (N = 0 or N >= dim -> None)."""
    return {N: (None if N == 0 or N >= dim else np.sort(np.asarray(order)[:N])) for N in ns}


def usage_report(usage, order, ns, n_rows, n_inactive):
    usage = np.asarray(usage, dtype=np.float64)
    total = usage.sum()
    r = dict(dim=int(len(usage)), n_rows=int(n_rows), n_inactive=int(n_inactive), **count_stats(usage))
    r["live_units"], r["eff_units"] = r.pop("live"), r.pop("eff_k")
    r["coverage_of_top_n"] = {str(N): float(usage[np.asarray(order)[:N]].sum() / total) if total > 0 else 0.0
                              for N in ns if N > 0}
    return r


def partition_nmi(a, b):
    """Arithmetic-mean NMI of two integer labelings (label values arbitrary)."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    P = np.bincount(a.astype(np.int64) * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)
    P /= P.sum()
    pa, pb = P.sum(1), P.sum(0)
    nz = P > 0
    mi = (P[nz] * np.log(P[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    denom = (h(pa) + h(pb)) / 2
    return float(mi / denom) if denom > 0 else 1.0


def fit_memory_gb(n_rows, dim, K, batch, pool):
    """Rough peak GPU memory (decimal GB) of one fit: the row matrix X plus balanced_fit's working set (density pool and
    its sorted copy, centroids / EMA / mean-shift copies, one batch, ~10 (batch x K) matrices incl. the int64 one-hot)."""
    x = n_rows * dim * 4
    work = 4 * (2 * min(pool, n_rows) * dim + 4 * K * dim + batch * dim + 10 * batch * K)
    return (x + work) / 1e9 + 1.5


def too_big_fits(total_gb, sched, dims, ks, spaces, margin_gb=2.0):
    """(key, row-matrix GB) of every fit whose row matrix X alone cannot be held by a GPU of `total_gb` GB."""
    out = []
    for K in ks:
        for s in spaces:
            x = sched[K].n_rows * dims[s] * 4 / 1e9
            if x + margin_gb > total_gb:
                out.append((km_key(K, s), round(x, 1)))
    return out


def space_specs(models, baselines, saes):
    """{space: spec string} from the same strings load_arms parses."""
    out = {}
    for text in (models, baselines):
        for item in filter(None, text.split(",")):
            n, p = item.split("=", 1)
            out[n] = p
    out.update(parse_saes(saes))
    return out


def row_metadata(cache, row_pos):
    """The row/chunk arrays of assign.npz, exactly the expressions of eval_out/atlas_token_detail.py."""
    chunk_all = np.asarray(cache.row_chunk, dtype=np.int64)
    tok_all = np.asarray(cache.tok, dtype=np.int64)
    return dict(row_chunk=chunk_all.astype(np.int32), row_pos=np.asarray(row_pos, dtype=np.int32),
                row_tok=tok_all.astype(np.int32), chunk_doc=cache.doc.astype(np.int64),
                chunk_ntok=np.asarray(cache.n_tok, dtype=np.int32))


def sampled_mask(row_chunk, row_pos, n_tok):
    """Atlas SAMPLED rows: every row but each chunk's final-token row."""
    last = np.asarray(n_tok, dtype=np.int64) - 1
    return np.asarray(row_pos) != last[np.asarray(row_chunk, dtype=np.int64)]


def save_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------- codebook files
def codebook_path(out, K, space):
    return Path(out) / "codebooks" / f"{km_key(K, space)}.npz"


def usage_path(out, space):
    return Path(out) / "codebooks" / f"unit_usage_{space}.npz"


def codebook_expect(a, K, space, spec, sch, n_train, sha):
    """Everything that defines a fit; stored in the codebook and compared on resume."""
    return dict(space=space, K=int(K), n_rows=int(sch.n_rows), seed=int(a.seed), batch_size=int(sch.batch_size),
                epochs=int(a.epochs), density_pool=int(a.density_pool), spec=str(spec), n_train=int(n_train),
                rows_sha1=sha)


def check_codebook(path, expect, dim):
    """None if `path` is a valid codebook of exactly the fit `expect` describes; otherwise why not."""
    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            C = z["centroids"] if "centroids" in files else None
            got = {k: z[k].item() for k in expect if k in files and z[k].ndim == 0}
    except Exception as e:                                          # truncated / not an npz / pickled
        return f"unreadable ({type(e).__name__}: {e})"
    why = []
    for k, v in expect.items():
        if k not in got:
            why.append(f"{k} not stored")
        elif got[k] != v:
            why.append(f"{k}: stored {got[k]!r}, requested {v!r}")
    if C is None:
        why.append("no centroids")
    elif C.shape != (expect["K"], dim):
        why.append(f"centroids {C.shape}, expected {(expect['K'], dim)}")
    elif C.dtype != np.float32 or not np.isfinite(C).all():
        why.append("centroids are not finite float32")
    return "; ".join(why) or None


def stored_stats(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].item() for k in STAT_KEYS if k in z.files}


def save_codebook(path, C, expect, stats):
    """Atomic: write a temp .npz beside the target, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, centroids=np.ascontiguousarray(C, dtype=np.float32), init="dpc", reinit="peaks", **expect, **stats)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------- GPU helpers
def clock(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return time.time()


def gpu_peak_gb(dev):
    return torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else -1.0     # -1 = CPU run


def free_gpu(dev):
    """Hand the caching allocator's blocks back between spaces (the caller has already deleted its tensors)."""
    if dev.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


def embed_into(X, H, T, arm, dev, chunk):
    """Fill the preallocated (n, dim) tensor X with the arm's coordinates of the rows H, one chunk at a time (never a
    second full-size copy). Non-finite coordinates abort: they would silently wreck the k-means."""
    for i in range(0, len(H), chunk):
        v = embed(H[i:i + chunk], arm, dev, T[i:i + chunk], chunk=chunk)
        if not bool(torch.isfinite(v).all()):
            raise RuntimeError(f"non-finite coordinates in rows {i}..{i + len(v)}")
        X[i:i + len(v)] = v
    return X


def nearest_centroid(V, C, chunk=8192):
    """Nearest centroid (Euclidean) of every row of V, as eval_out/sae_controls_fit.py labels its codes."""
    return torch.cat([torch.cdist(V[i:i + chunk], C).argmin(1) for i in range(0, len(V), chunk)])


def coord_argmax(V, units, nonneg):
    """Unit readout of the coordinates V (n, dim): argmax over every coordinate (`units` None) or over the coordinates in
    the ascending index tensor `units` (ties -> lower unit id). nonneg (SAE codes): -1 where no allowed coordinate is > 0."""
    if units is None:
        val, idx = V.max(1)
    else:
        val, idx = V[:, units].max(1)
        idx = units[idx]
    if nonneg:
        idx = torch.where(val > 0, idx, torch.full_like(idx, -1))
    return idx


# ---------------------------------------------------------------------------------------------- the training dump
class Dump:
    """layer_<L>.npy (rows, d) + rows_tok.npy of the training dump, read through memmaps (stage the files in /dev/shm
    first on Argo: /scratch is GPFS without a page cache)."""

    def __init__(self, act_dir, layer, train_frac):
        act = Path(act_dir)
        self.mm = np.load(act / f"layer_{layer}.npy", mmap_mode="r")
        self.tok = np.load(act / "rows_tok.npy", mmap_mode="r")
        if len(self.tok) != len(self.mm):
            raise SystemExit(f"{act}: layer_{layer}.npy has {len(self.mm):,} rows but rows_tok.npy {len(self.tok):,}")
        self.d = int(self.mm.shape[1])
        self.n_train = int(len(self.mm) * train_frac)

    def index(self, n_rows, seed):
        return sample_rows(self.n_train, n_rows, seed)

    def read(self, idx):
        """(H float32 (n, d), token ids int64 (n,)) of the sorted rows `idx`, read in slices like sae_controls_fit."""
        H = np.empty((len(idx), self.d), np.float32)
        for s in range(0, len(idx), DUMP_READ_CHUNK):
            H[s:s + DUMP_READ_CHUNK] = self.mm[idx[s:s + DUMP_READ_CHUNK]]
        return H, np.asarray(self.tok[idx], np.int64)


class Ctx:
    """Everything the stages share."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def arm(self, space):
        return self.arm_of[space]

    def nonneg(self, space):
        return self.arm_of[space][0] == "sae"


# ---------------------------------------------------------------------------------------------- stage: unit usage
def usage_stage(ctx):
    """usage_j on the K=ref_k training rows, per space; codebooks/unit_usage_<space>.npz."""
    a, dev, meta = ctx.args, ctx.dev, ctx.meta
    n_rows = a.rows_per_centroid * a.ref_k
    idx = ctx.dump.index(n_rows, a.seed)
    sha = fingerprint(idx)
    H, T = ctx.dump.read(idx)
    log(f"unit usage on {n_rows:,} rows (the K={a.ref_k} training sample)")
    for space in ctx.spaces:
        t0 = clock(dev)
        arm, dim, nonneg = ctx.arm(space), ctx.dims[space], ctx.nonneg(space)
        usage = torch.zeros(dim, dtype=torch.long, device=dev)
        n_inactive = 0
        chunk = chunk_rows(dim)
        for i in range(0, n_rows, chunk):
            V = embed(H[i:i + chunk], arm, dev, T[i:i + chunk], chunk=chunk)
            lab = coord_argmax(V, None, nonneg)
            ok = lab >= 0
            usage += torch.bincount(lab[ok], minlength=dim)
            n_inactive += int((~ok).sum())
        usage = usage.cpu().numpy()
        order = usage_order(usage)
        path = usage_path(ctx.out, space)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, usage=usage.astype(np.int64), order=order.astype(np.int64), space=space, dim=dim,
                 n_rows=n_rows, n_inactive=n_inactive, nonneg=nonneg, spec=ctx.specs[space], n_train=ctx.dump.n_train,
                 rows_sha1=sha)
        os.replace(tmp, path)
        rep = usage_report(usage, order, ctx.unit_ns, n_rows, n_inactive)
        meta["unit_usage"][space] = rep
        meta.save()
        cov = "  ".join(f"top{N} {c:.3f}" for N, c in rep["coverage_of_top_n"].items())
        log(f"  {space:<13} dim {dim:>6}  live units {rep['live_units']:>6}  eff units {rep['eff_units']:>8.1f}  "
            f"inactive rows {n_inactive:,}  coverage {cov}  ({clock(dev) - t0:.0f}s)")
        del V, usage
        free_gpu(dev)


# ---------------------------------------------------------------------------------------------- stage: k-means fits
def fit_stage(ctx):
    a, dev, meta = ctx.args, ctx.dev, ctx.meta
    for K in ctx.ks:
        sch = ctx.sched[K]
        idx = ctx.dump.index(sch.n_rows, a.seed)
        sha = fingerprint(idx)
        if K == REF_K and sch.n_rows == 200_000:
            if not np.array_equal(idx, sae_controls_rows(ctx.dump.n_train, a.seed)):
                raise RuntimeError("the K=2000 training rows differ from eval_out/sae_controls_fit.py's")
            log(f"K={K}: row sample identical to eval_out/sae_controls_fit.py's ({len(idx):,} rows, sha1 {sha[:10]})")
        meta["schedule"][str(K)].update(rows_sha1=sha, first_row=int(idx[0]), last_row=int(idx[-1]))
        todo = []
        for space in ctx.spaces:
            key, path = km_key(K, space), codebook_path(ctx.out, K, space)
            expect = codebook_expect(a, K, space, ctx.specs[space], sch, ctx.dump.n_train, sha)
            if path.exists() and not a.refit:
                why = check_codebook(path, expect, ctx.dims[space])
                if why:
                    raise SystemExit(f"[resume] {path} exists but does not match this run ({why}). "
                                     "Delete it or pass --refit.")
                meta["fits"][key] = dict(K=K, space=space, resumed=True, **stored_stats(path))
                log(f"  {key}: valid codebook on disk, skipped")
                continue
            todo.append((space, key, path, expect))
        meta.save()
        if not todo:
            continue
        H, T = ctx.dump.read(idx)
        for space, key, path, expect in todo:
            arm, dim = ctx.arm(space), ctx.dims[space]
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = clock(dev)
            X = torch.empty((len(H), dim), dtype=torch.float32, device=dev)
            embed_into(X, H, T, arm, dev, chunk_rows(dim))
            t1 = clock(dev)
            epoch_calls = [0]

            def epoch_log(msg, key=key, epochs=a.epochs):
                epoch_calls[0] += 1
                if epoch_calls[0] % 10 == 0 or epoch_calls[0] == epochs:
                    log(f"    {key}{msg}")
            C = balanced_fit(X, K, seed=a.seed, epochs=a.epochs, batch_size=sch.batch_size,
                             density_pool=a.density_pool, log=epoch_log)
            if tuple(C.shape) != (K, dim) or not bool(torch.isfinite(C).all()):
                raise RuntimeError(f"{key}: balanced_fit returned centroids {tuple(C.shape)} (finite: "
                                   f"{bool(torch.isfinite(C).all())}), expected {(K, dim)}")
            counts = torch.bincount(nearest_centroid(X, C), minlength=K).cpu().numpy()
            t2 = clock(dev)
            stats = dict(**count_stats(counts), seconds=t2 - t0, embed_seconds=t1 - t0, fit_seconds=t2 - t1,
                         gpu_peak_gb=gpu_peak_gb(dev))
            save_codebook(path, C.cpu().numpy(), expect, stats)
            meta["fits"][key] = dict(K=K, space=space, resumed=False, **stats)
            meta.save()
            log(f"  {key}: dim {dim}, live {stats['live']}/{K}, top10 share {stats['top10_share']:.4f}, "
                f"effective K {stats['eff_k']:.1f}, embed {stats['embed_seconds']:.0f}s, fit {stats['fit_seconds']:.0f}s, "
                f"peak GPU {stats['gpu_peak_gb']:.1f} GB")
            del X, C
            free_gpu(dev)
        del H, T


# ---------------------------------------------------------------------------------------------- stage: agreement
def agreement_stage(ctx):
    """Label agreement (and NMI, which is blind to a permutation of cluster ids) between the new K=ref_k codebook of a
    space and the existing eval_out/sae_controls_fit.py code k-means kmeans_<space>.npz, on the same training rows."""
    a, dev, meta = ctx.args, ctx.dev, ctx.meta
    cdir = Path(a.controls_dir) if a.controls_dir else None
    cand = [s for s in ctx.spaces if cdir is not None and (cdir / f"kmeans_{s}.npz").is_file()]
    if a.ref_k not in ctx.ks or not cand:
        log(f"agreement: skipped (K={a.ref_k} fitted: {a.ref_k in ctx.ks}; existing codebooks found for: {cand})")
        return
    n_rows = a.rows_per_centroid * a.ref_k
    idx = ctx.dump.index(n_rows, a.seed)
    H, T = ctx.dump.read(idx)
    for space in cand:
        old_path = cdir / f"kmeans_{space}.npz"
        rec = dict(old=str(old_path))
        with np.load(old_path, allow_pickle=False) as z:
            old = z["centroids"]
            rec.update({f"old_{k}": z[k].item() for k in ("spec", "n_rows", "seed", "n_clusters")
                        if k in z.files and z[k].ndim == 0})
        dim = ctx.dims[space]
        if old.shape[1] != dim:
            rec["error"] = f"existing codebook {old.shape} does not live in this space (dim {dim})"
        else:
            with np.load(codebook_path(ctx.out, a.ref_k, space), allow_pickle=False) as z:
                new = z["centroids"]
            X = embed_into(torch.empty((len(H), dim), dtype=torch.float32, device=dev), H, T, ctx.arm(space), dev,
                           chunk_rows(dim))
            lab_new = nearest_centroid(X, torch.from_numpy(new).to(dev)).cpu().numpy()
            lab_old = nearest_centroid(X, torch.from_numpy(old).to(dev)).cpu().numpy()
            rec.update(n_rows=int(len(H)), K_new=int(new.shape[0]), K_old=int(old.shape[0]),
                       label_agreement=float((lab_new == lab_old).mean()), nmi=partition_nmi(lab_new, lab_old),
                       identical_centroids=bool(new.shape == old.shape and np.array_equal(new, old)))
            del X
            free_gpu(dev)
        meta["agreement"][space] = rec
        meta.save()
        log(f"  agreement {space}: " + (rec["error"] if "error" in rec else
            f"label agreement {rec['label_agreement']:.4f}, NMI {rec['nmi']:.4f} over {rec['n_rows']:,} rows "
            f"(old fit: n_rows {rec.get('old_n_rows')}, seed {rec.get('old_seed')})"))
    del H, T


# ---------------------------------------------------------------------------------------------- stage: Atlas labels
def label_stage(ctx):
    """Label every Atlas cache row in every space (km{K}, unit{N}); write assign.npz and the label diagnostics."""
    dev, meta, cache = ctx.dev, ctx.meta, ctx.cache
    n = cache.n_rows
    labels = {}
    for space in ctx.spaces:
        t0 = clock(dev)
        arm, dim, nonneg = ctx.arm(space), ctx.dims[space], ctx.nonneg(space)
        cents = {}
        for K in ctx.ks:
            with np.load(codebook_path(ctx.out, K, space), allow_pickle=False) as z:
                cents[K] = torch.from_numpy(z["centroids"]).to(dev)
        with np.load(usage_path(ctx.out, space), allow_pickle=False) as z:
            order = z["order"]
        units = {N: (None if u is None else torch.from_numpy(u).to(dev))
                 for N, u in unit_sets(order, ctx.unit_ns, dim).items()}
        out = {km_key(K, space): np.empty(n, np.int32) for K in ctx.ks}
        out.update({unit_key(N, space): np.empty(n, np.int32) for N in ctx.unit_ns})
        for i in range(0, n, LABEL_CHUNK):
            V = embed(np.array(cache.rows[i:i + LABEL_CHUNK]), arm, dev, ctx.tok_all[i:i + LABEL_CHUNK],
                      chunk=LABEL_CHUNK)                                  # np.array: a writable copy of the read-only memmap
            j = i + len(V)
            for K, C in cents.items():
                out[km_key(K, space)][i:j] = nearest_centroid(V, C).cpu().numpy()
            for N, u in units.items():
                out[unit_key(N, space)][i:j] = coord_argmax(V, u, nonneg).cpu().numpy()
        labels.update(out)
        for key, lab in out.items():
            meta["labels"][key] = label_stats(lab, ctx.sampled)
        meta["label_seconds"][space] = clock(dev) - t0
        meta.save()
        eff = "  ".join(f"{key.split('_')[0]} {meta['labels'][key]['eff_k']:.0f}" for key in out)
        log(f"  labels {space:<13} {len(out)} arrays  effective K on sampled rows: {eff}  ({clock(dev) - t0:.0f}s)")
        del cents, units, V
        free_gpu(dev)
    path = Path(ctx.out) / "assign.npz"
    tmp = path.with_name("assign.tmp.npz")
    np.savez_compressed(tmp, **labels, **ctx.row_meta)
    os.replace(tmp, path)
    log(f"wrote {path}: {len(labels)} label arrays over {n:,} rows")


# ---------------------------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations_dir", default="activations_gemma3_4b_sampled_struct5_10M",
                    help="training dump holding layer_<layer>.npy and rows_tok.npy (a /dev/shm copy on Argo)")
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--train_frac", type=float, default=0.95, help="the first this share of the dump rows is the train split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="", help="name=AE checkpoint (the token-bypass AE)")
    ap.add_argument("--baselines", default="", help="name=k-means npz: only its normalisation stats / token table are used")
    ap.add_argument("--saes", default="", help="name=matched:<last.pt> | scope:<id>")
    ap.add_argument("--spaces", default=DEFAULT_SPACES, help="spaces to read out, each a loaded arm or SAE name")
    ap.add_argument("--ks", default="2000,4000,8000")
    ap.add_argument("--rows_per_centroid", type=int, default=100)
    ap.add_argument("--batch_per_k2000", type=int, default=4096, help="batch size at K=2000; scaled linearly with K")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--density_pool", type=int, default=32768)
    ap.add_argument("--unit_ns", default="500,1000,1500,2000,3000,0", help="unit-readout sizes; 0 = every unit")
    ap.add_argument("--ref_k", type=int, default=REF_K,
                    help="the K whose training rows define the unit usage and the agreement check")
    ap.add_argument("--controls_dir", default="results/delta/g4b_l33_d5120/evals/sae_suite/controls",
                    help="holds the existing kmeans_<space>.npz of eval_out/sae_controls_fit.py ('' = skip the agreement check)")
    ap.add_argument("--cache", required=True, help="Atlas cache directory")
    ap.add_argument("--rows_dir", default=None, help="staged copy of the Atlas row arrays (/dev/shm)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--refit", action="store_true", help="overwrite codebooks that exist but do not match this run")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    ks = parse_ints(args.ks, "--ks", 2)
    unit_ns = parse_ints(args.unit_ns, "--unit_ns", 0)
    spaces = list(dict.fromkeys(s for s in args.spaces.split(",") if s))
    if not spaces:
        raise SystemExit("--spaces is empty")
    if args.rows_per_centroid < 1 or args.epochs < 1 or args.ref_k < 1 or not 0 < args.train_frac <= 1:
        ap.error("--rows_per_centroid, --epochs, --ref_k must be >= 1 and 0 < --train_frac <= 1")
    try:
        sched = {K: k_schedule(K, args.rows_per_centroid, args.batch_per_k2000) for K in ks}
    except ValueError as e:
        ap.error(str(e))

    import probe_chance_corrected as PCC

    torch.backends.cuda.matmul.allow_tf32 = False       # fp32 distances, as in every other comparison
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    (out / "codebooks").mkdir(parents=True, exist_ok=True)

    cache = AS.AtlasCache(args.cache, args.rows_dir)
    row_pos = np.load(Path(args.rows_dir or args.cache) / "rows_pos.npy", mmap_mode="r")
    row_meta = row_metadata(cache, row_pos)
    sampled = sampled_mask(row_meta["row_chunk"], row_pos, cache.n_tok)
    log(f"Atlas cache {cache.n_rows:,} rows ({int(sampled.sum()):,} sampled, {cache.n_rows - int(sampled.sum()):,} "
        f"final-token), {cache.n_chunks:,} chunks, d={cache.rows.shape[1]}")
    dump = Dump(args.activations_dir, args.layer, args.train_frac)
    if dump.d != cache.rows.shape[1]:
        raise SystemExit(f"dump rows have {dump.d} dims but the Atlas cache rows {cache.rows.shape[1]}")
    log(f"dump {args.activations_dir}: {len(dump.mm):,} rows, train split {dump.n_train:,}, d={dump.d}")

    arms, saes = PCC.load_arms(args.models, args.baselines, args.saes, dev, sae_kmeans="")
    log(f"arms {list(arms)}  saes {list(saes)}")
    missing = [s for s in spaces if s not in arms and s not in saes]
    if missing:
        raise SystemExit(f"--spaces {missing} are neither loaded arms {list(arms)} nor SAEs {list(saes)}")
    arm_of = {s: (arms[s] if s in arms else ("sae", saes[s])) for s in spaces}
    specs = space_specs(args.models, args.baselines, args.saes)
    specs = {s: specs.get(s, "") for s in spaces}
    H0, T0 = dump.read(np.arange(2))
    dims = {s: int(embed(H0, arm_of[s], dev, T0, chunk=2).shape[1]) for s in spaces}

    meta = Meta(out / "meta.json")
    meta.update(args=vars(args), torch=torch.__version__, specs=specs, dims=dims,
                nonneg_spaces=[s for s in spaces if arm_of[s][0] == "sae"],       # their unit labels use -1 = no active unit
                device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                cache=dict(n_rows=int(cache.n_rows), n_chunks=int(cache.n_chunks), n_sampled_rows=int(sampled.sum())),
                dump=dict(n_rows=int(len(dump.mm)), n_train=dump.n_train, d=dump.d),
                schedule={str(K): dict(s._asdict(), total_steps=s.steps_per_epoch * args.epochs) for K, s in sched.items()},
                fits={}, unit_usage={}, agreement={}, labels={}, label_seconds={})
    log("schedule: " + "; ".join(f"K={K}: {s.n_rows:,} rows, batch {s.batch_size}, {s.steps_per_epoch} steps/epoch x "
                                 f"{args.epochs} = {s.steps_per_epoch * args.epochs} steps" for K, s in sched.items()))
    plan = {(K, s): fit_memory_gb(sched[K].n_rows, dims[s], K, sched[K].batch_size, args.density_pool)
            for K in ks for s in spaces}
    meta["plan_gpu_gb"] = {f"{km_key(K, s)}": round(v, 1) for (K, s), v in plan.items()}
    log("planned peak GPU GB: " + "  ".join(f"{km_key(K, s)} {v:.0f}" for (K, s), v in plan.items()))
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(dev).total_memory / 1e9
        too_big = too_big_fits(total, sched, dims, ks, spaces)
        if too_big:
            raise SystemExit(f"the row matrix alone does not fit on this {total:.0f} GB GPU for: {too_big}")
        log(f"GPU {meta['device']}: {total:.0f} GB; largest planned fit {max(plan.values()):.0f} GB")
    meta.save()

    ctx = Ctx(args=args, dev=dev, out=out, dump=dump, cache=cache, ks=ks, unit_ns=unit_ns, spaces=spaces, sched=sched,
              arm_of=arm_of, specs=specs, dims=dims, meta=meta, row_meta=row_meta, sampled=sampled,
              tok_all=np.asarray(cache.tok, dtype=np.int64))
    t0 = time.time()
    usage_stage(ctx)
    fit_stage(ctx)
    agreement_stage(ctx)
    label_stage(ctx)
    meta["total_seconds"] = time.time() - t0
    meta.save()
    log(f"done in {meta['total_seconds'] / 60:.1f} min -> {out}")
    summary(meta, ks, spaces)


class Meta(dict):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def save(self):
        save_json(self.path, self)


def summary(meta, ks, spaces):
    """Effective K of the K-means readouts on the Atlas sampled rows, one row per space."""
    log("effective K on Atlas sampled rows (live clusters in brackets):")
    for s in spaces:
        cells = [f"K={K}: {meta['labels'][km_key(K, s)]['eff_k']:.0f} ({meta['labels'][km_key(K, s)]['live']})" for K in ks]
        log(f"  {s:<13} " + "   ".join(cells))


if __name__ == "__main__":
    main()
