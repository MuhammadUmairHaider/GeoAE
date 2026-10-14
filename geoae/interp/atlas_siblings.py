"""Pure helpers for the Atlas sibling-pair test of the ranges hypothesis.

Hypothesis (the project's founding one): related-but-distinct concepts are REGIONS (ranges) along a SHARED
direction, not one direction per concept. Earlier evidence picked "range pairs" out of the model's own clusters.
This module builds the test around NAMED concepts of the FineWeb Atlas (guidelabs/fineweb-atlas), paired by rules
that never look at any model activation:

  Rule E  sentence-embedding neighbours of "name: description" (external semantics),
  Rule C  second-order similarity of PPMI co-occurrence profiles with LOW first-order co-occurrence (substitutes),

each followed by a synonym/containment guard. Nothing here imports torch at module level except where a function
takes a tensor: every function is plain numpy so it can be unit-tested on a laptop.

The orchestration lives in eval_out/atlas_siblings_pairs.py (Step 1), eval_out/atlas_siblings_extract.py (Step 2),
eval_out/paper_atlas_sibling_ranges.py (phase 1) and eval_out/paper_atlas_sibling_removal.py (phase 2).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

import numpy as np

N_CHUNKS_ATLAS = 95_486_049          # total chunks in the Atlas (README), the PPMI normaliser


# ---------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------
def _singular(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("ses"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def name_tokens(name: str) -> tuple[str, ...]:
    """Lower-cased alphanumeric tokens with a light plural strip ('Horror films' -> ('horror', 'film'))."""
    return tuple(_singular(t) for t in re.findall(r"[a-z0-9]+", str(name).casefold()))


def name_contains(a: str, b: str) -> bool:
    """True when one name's token sequence occurs contiguously inside the other's (either direction)."""
    return _tokens_contain(name_tokens(a), name_tokens(b))


def merge_duplicate_names(names, counts):
    """Group case-insensitive duplicate names. Returns a list of units, each a sorted list of ROW positions into
    `names`; units are ordered by their first member, and the primary member of a unit (the one that supplies
    description / LCC letter) is `max(unit, key=counts)`."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, nm in enumerate(names):
        groups[" ".join(str(nm).casefold().split())].append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def primary_member(unit, counts) -> int:
    return max(unit, key=lambda i: (counts[i], -i))


# ---------------------------------------------------------------------------------------------
# co-occurrence matrix (packed upper triangle, README: idx = i*n - i*(i-1)//2 + (j-i) for i <= j)
# ---------------------------------------------------------------------------------------------
def packed_index(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    i, j = np.minimum(i, j).astype(np.int64), np.maximum(i, j).astype(np.int64)
    return i * n - (i * (i - 1)) // 2 + (j - i)


def packed_rows(packed, n: int, rows) -> np.ndarray:
    """Dense rows of the symmetric co-occurrence matrix for concept ids `rows` -> (len(rows), n) float64."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((len(rows), n), dtype=np.float64)
    cols = np.arange(n, dtype=np.int64)
    for k, i in enumerate(rows):
        out[k] = packed[packed_index(np.full(n, i), cols, n)]
    return out


def ppmi_profiles(counts: np.ndarray, own: np.ndarray, col_marginal: np.ndarray, n_total: float,
                  min_count: float = 20.0, zero_cols=None) -> np.ndarray:
    """PPMI of rows of a co-occurrence matrix: max(0, log(C_ij * N / (C_i * C_j))).

    counts (R, n): C[row, j];  own (R,): chunk count of each row's concept;  col_marginal (n,): chunk count of every
    column concept.  Entries with C_ij < min_count are zeroed (rare co-occurrences give huge, noisy PMI).
    `zero_cols` (R,) optionally names, per row, the column holding the row concept itself, which is zeroed (the
    diagonal C_ii is a count, not a co-occurrence)."""
    counts = np.asarray(counts, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(counts * float(n_total) / (own[:, None] * col_marginal[None, :]))
    pmi[~np.isfinite(pmi) | (counts < min_count)] = 0.0
    np.maximum(pmi, 0.0, out=pmi)
    if zero_cols is not None:
        pmi[np.arange(len(pmi)), np.asarray(zero_cols)] = 0.0
    return pmi


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(nrm, 1e-12)
    return Xn @ Xn.T


# ---------------------------------------------------------------------------------------------
# pair rules
# ---------------------------------------------------------------------------------------------
def top_k_pairs(sim: np.ndarray, allowed: np.ndarray, k: int) -> set[tuple[int, int, bool]]:
    """Directed nearest neighbours: for every row i, its top-k columns by `sim` among `allowed[i]` (a bool matrix).
    Returns {(i, j, mutual)} with i < j, where `mutual` says j also ranks i in ITS top-k. One-way pairs are kept."""
    n = sim.shape[0]
    masked = np.where(allowed, sim, -np.inf)
    kk = min(k, n - 1)
    nn = np.argpartition(-masked, kk - 1, axis=1)[:, :kk]
    picked = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), kk)
    cols = nn.reshape(-1)
    ok = np.isfinite(masked[rows, cols])
    picked[rows[ok], cols[ok]] = True
    out = {}
    for i, j in zip(*np.nonzero(picked | picked.T)):
        if i < j:
            out[(int(i), int(j))] = bool(picked[i, j] and picked[j, i])
    return {(i, j, m) for (i, j), m in out.items()}


def _tokens_contain(ta: tuple, tb: tuple) -> bool:
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    n = len(short)
    return any(long_[i:i + n] == short for i in range(len(long_) - n + 1))


def synonym_guard(names: list[str], jaccard: np.ndarray, max_jaccard: float) -> np.ndarray:
    """(n, n) bool: True where the pair is ALLOWED, i.e. neither name contains the other (token-wise, either way) and
    the chunk Jaccard of the pair in the Atlas co-occurrence is <= max_jaccard."""
    n = len(names)
    ok = jaccard <= max_jaccard
    toks = [name_tokens(nm) for nm in names]
    by_token: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(toks):
        for w in set(t):
            by_token[w].append(i)
    for i, t in enumerate(toks):
        if not t:
            continue
        # a contained name's FIRST token must occur in the container, so only units sharing it can contain / be contained
        for j in by_token[t[0]]:
            if j != i and _tokens_contain(t, toks[j]):
                ok[i, j] = ok[j, i] = False
    np.fill_diagonal(ok, False)
    return ok


def first_order_stats(cij: np.ndarray, ci: np.ndarray):
    """Overlap coefficient C_ij / min(C_ii, C_jj) and Jaccard C_ij / (C_ii + C_jj - C_ij) for all unit pairs."""
    mn = np.minimum(ci[:, None], ci[None, :])
    overlap = cij / np.maximum(mn, 1.0)
    jac = cij / np.maximum(ci[:, None] + ci[None, :] - cij, 1.0)
    return overlap, jac


# ---------------------------------------------------------------------------------------------
# chunk bookkeeping
# ---------------------------------------------------------------------------------------------
def labels_csr(ptr: np.ndarray, ids: np.ndarray, n_chunks: int, units: list[list[int]], n_concepts: int):
    """Sparse chunk x unit incidence (scipy csc, 0/1): chunk c carries unit u when ANY concept id of u is in c's labels.
    (Units may overlap; a chunk carrying two ids of one unit still counts once.)"""
    import scipy.sparse as sp
    chunk_of = np.repeat(np.arange(n_chunks), np.diff(ptr))
    K = sp.csr_matrix((np.ones(len(ids), dtype=np.int32), (chunk_of, np.asarray(ids, dtype=np.int64))),
                      shape=(n_chunks, n_concepts))
    ur = np.concatenate([np.asarray(m, dtype=np.int64) for m in units]) if units else np.empty(0, np.int64)
    uc = np.repeat(np.arange(len(units)), [len(m) for m in units])
    Um = sp.csr_matrix((np.ones(len(ur), dtype=np.int32), (ur, uc)), shape=(n_concepts, len(units)))
    M = (K @ Um).tocsr()
    M.data[:] = 1
    M.eliminate_zeros()
    return M.tocsc()


def chunk_sets(M_csc, a: int, b: int):
    """(a_only, b_only, both) sorted chunk indices for units a and b."""
    ca = M_csc.indices[M_csc.indptr[a]:M_csc.indptr[a + 1]]
    cb = M_csc.indices[M_csc.indptr[b]:M_csc.indptr[b + 1]]
    both = np.intersect1d(ca, cb, assume_unique=True)
    return (np.setdiff1d(ca, both, assume_unique=True), np.setdiff1d(cb, both, assume_unique=True), both)


def exclusive_counts(M_csc, pairs) -> np.ndarray:
    """(P, 3) int array of [A-only chunks, B-only chunks, shared chunks] for pairs [(a, b), ...] of unit indices."""
    out = np.zeros((len(pairs), 3), dtype=np.int64)
    for k, (a, b) in enumerate(pairs):
        ao, bo, bt = chunk_sets(M_csc, a, b)
        out[k] = (len(ao), len(bo), len(bt))
    return out


def split_half(doc_ids: np.ndarray, seed: int = 0) -> np.ndarray:
    """Deterministic document-level fit/eval split: True = FIT half. Hash of (doc id, seed); all chunks of a
    document land on the same side (chunks of one document share topic and labels, so splitting them would leak)."""
    with np.errstate(over="ignore"):                    # uint64 wrap-around is the point of the hash
        x = np.asarray(doc_ids, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x & np.uint64(1)).astype(bool)


# ---------------------------------------------------------------------------------------------
# null pairs
# ---------------------------------------------------------------------------------------------
def sample_null_pairs(sib_pairs, log_count: np.ndarray, lcc: np.ndarray, eligible: np.ndarray,
                      forbidden: set, mode: str, rng: np.random.Generator, tol: float = 0.10,
                      allowed_fn=None, max_tries: int = 4000):
    """One null pair per sibling pair, matched on prevalence and (for x) on LCC letter. mode is "same" or "cross".

    For each sibling (a, b) (same LCC letter L by construction) draw x with |log_count[x] - log_count[a]| <= tol and
    lcc[x] == L, and y with |log_count[y] - log_count[b]| <= tol and lcc[y] == L ("same": a random same-LCC pair)
    or lcc[y] != L ("cross": a random cross-LCC pair), from the `eligible` units, such that x != y, {x, y} is not in
    `forbidden` (the sibling candidate union) nor already used, and allowed_fn(x, y) holds (the synonym guard and data
    filter siblings pass). tol grows by 50% after every 500 failed tries. Returns [(x, y, tol_used)] aligned to
    sib_pairs, None where nothing was found."""
    assert mode in ("same", "cross")
    out = []
    used: set = set()
    elig = np.flatnonzero(eligible)
    lc = log_count[elig]
    lcc_e = lcc[elig]
    for (a, b) in sib_pairs:
        L = lcc[a]
        found = None
        t = tol
        for tries in range(max_tries):
            if tries and tries % 500 == 0:
                t *= 1.5
            cx = elig[(np.abs(lc - log_count[a]) <= t) & (lcc_e == L)]
            cy = elig[(np.abs(lc - log_count[b]) <= t) & ((lcc_e == L) if mode == "same" else (lcc_e != L))]
            if len(cx) == 0 or len(cy) == 0:
                t *= 1.5
                continue
            x, y = int(rng.choice(cx)), int(rng.choice(cy))
            if x == y:
                continue
            key = frozenset((x, y))
            if key in forbidden or key in used:
                continue
            if allowed_fn is not None and not allowed_fn(x, y):
                continue
            used.add(key)
            found = (x, y, float(t))
            break
        out.append(found)
    return out


# ---------------------------------------------------------------------------------------------
# frozen pair list
# ---------------------------------------------------------------------------------------------
def pair_list_sha256(pairs) -> str:
    """sha256 over the canonical form of the pair list: sorted [min_id, max_id, tag] rows (concept ids of the primary
    member of each unit), so two lists with the same pairs and tags hash alike regardless of order."""
    rows = sorted([int(min(p["a"], p["b"])), int(max(p["a"], p["b"])), str(p["tag"])] for p in pairs)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# the Atlas chunk stream (deterministic: shard order, row order, chunk_status == "ok")
# ---------------------------------------------------------------------------------------------
ATLAS_REPO = "guidelabs/fineweb-atlas"


def shard_file(atlas_dir, i: int):
    from pathlib import Path
    return Path(atlas_dir) / "fineweb-atlas-annotated" / f"shard_{i:04d}.parquet"


def ensure_shard(atlas_dir, i: int):
    """Local path of annotated shard i, downloading ONLY that file (~385 MB) if it is missing."""
    p = shard_file(atlas_dir, i)
    if not p.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(ATLAS_REPO, f"fineweb-atlas-annotated/shard_{i:04d}.parquet", repo_type="dataset",
                        local_dir=str(atlas_dir))
    return p


def iter_atlas_chunks(atlas_dir, n_chunks: int, columns, batch_rows: int = 8192, max_shards: int = 57):
    """Yield dicts {column: list} covering the first `n_chunks` chunks with chunk_status == "ok" of the Atlas
    `chunks` config, in shard order then row order. The stream is a pure function of the dataset files, so any two
    runs (and the pair filter vs the activation extractor) see the identical chunk sequence."""
    import pyarrow.parquet as pq
    cols = list(dict.fromkeys(["chunk_status", *columns]))
    got = 0
    for s in range(max_shards):
        f = pq.ParquetFile(ensure_shard(atlas_dir, s))
        for batch in f.iter_batches(batch_size=batch_rows, columns=cols):
            d = batch.to_pydict()
            keep = [i for i, st in enumerate(d["chunk_status"]) if st == "ok"]
            keep = keep[:n_chunks - got]
            if keep:
                yield {c: [d[c][i] for i in keep] for c in columns}
                got += len(keep)
            if got >= n_chunks:
                return
    raise RuntimeError(f"only {got:,} ok chunks in {max_shards} shards, wanted {n_chunks:,}")


def stream_fingerprint(doc_ids, chunk_ids) -> str:
    """sha256 over the (doc_int_id, chunk_id) sequence of a chunk stream; identical streams hash alike."""
    h = hashlib.sha256()
    h.update(np.asarray(doc_ids, dtype=np.int64).tobytes())
    h.update(np.asarray(chunk_ids, dtype=np.int64).tobytes())
    return h.hexdigest()


def ragged_to_csr(lists):
    """list of int lists -> (ptr (n+1,) int64, flat int64)."""
    lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum(lens, out=ptr[1:])
    flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]) if len(lists) else np.empty(0, np.int64)
    return ptr, flat


# ---------------------------------------------------------------------------------------------
# the activation cache written by eval_out/atlas_siblings_extract.py
# ---------------------------------------------------------------------------------------------
def rows_of_chunks(row_start: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    """Row indices (sorted by chunk, in order) of all rows belonging to `chunks`; row_start has n_chunks + 1 entries."""
    chunks = np.asarray(chunks, dtype=np.int64)
    starts = row_start[chunks]
    lens = row_start[chunks + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    first = np.cumsum(lens) - lens
    return np.repeat(starts - first, lens) + np.arange(total, dtype=np.int64)


def cap_chunks(chunks: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """At most `cap` of `chunks`, a uniform subset in sorted order (all of them when there are fewer)."""
    if len(chunks) <= cap:
        return chunks
    return np.sort(rng.choice(chunks, size=cap, replace=False))


class AtlasCache:
    """Read access to an Atlas activation cache directory.

    Layout (all arrays row-aligned): rows.npy (R, d) float16 layer residuals, rows_tok.npy / rows_nxt.npy (R,) current
    and next token ids, rows_chunk.npy (R,) chunk index, rows_pos.npy (R,) token position; chunks.npz with
    doc_int_id / chunk_id / n_tok / row_start (n_chunks + 1) / label_ptr + label_ids (the chunk's concept ids, all
    types) / tok_ptr + tok_flat (the chunk's token ids). `rows_dir` lets the big arrays live elsewhere (a /dev/shm copy)."""

    def __init__(self, cache_dir, rows_dir=None):
        from pathlib import Path
        self.dir = Path(cache_dir)
        rd = Path(rows_dir) if rows_dir else self.dir
        z = np.load(self.dir / "chunks.npz", allow_pickle=False)
        self.doc = z["doc_int_id"].astype(np.int64)
        self.chunk_id = z["chunk_id"]
        self.n_tok = z["n_tok"]
        self.row_start = z["row_start"].astype(np.int64)
        self.label_ptr = z["label_ptr"]
        self.label_ids = z["label_ids"]
        self.tok_ptr = z["tok_ptr"]
        self.tok_flat = z["tok_flat"]
        self.fingerprint = str(z["fingerprint"])
        self.n_chunks = len(self.doc)
        self.rows = np.load(rd / "rows.npy", mmap_mode="r")
        self.tok = np.load(rd / "rows_tok.npy", mmap_mode="r")
        self.nxt = np.load(rd / "rows_nxt.npy", mmap_mode="r")
        self.row_chunk = np.load(rd / "rows_chunk.npy", mmap_mode="r")
        self.n_rows = len(self.rows)
        assert self.row_start[-1] == self.n_rows, "chunks.npz and rows.npy disagree on the row count"

    def membership(self, unit_ids: list[list[int]], n_concepts: int = 16790):
        """chunk x unit incidence (csc) for units given as lists of concept ids."""
        return labels_csr(self.label_ptr, self.label_ids, self.n_chunks, unit_ids, n_concepts)

    def fit_mask(self, seed: int = 0) -> np.ndarray:
        """(n_chunks,) True where the chunk's DOCUMENT is in the fit half."""
        return split_half(self.doc, seed)

    def gather(self, chunks: np.ndarray):
        """(row indices, h float16 (n, d), tok int64, nxt int64) of all rows of `chunks`."""
        idx = rows_of_chunks(self.row_start, chunks)
        return idx, np.asarray(self.rows[idx]), np.asarray(self.tok[idx], dtype=np.int64), np.asarray(self.nxt[idx], dtype=np.int64)


def pair_chunk_sets(M_csc, a_unit: int, b_unit: int, fit: np.ndarray):
    """Exclusive chunk sets of a pair split by fit/eval half: dict(a_fit, a_eval, b_fit, b_eval, both)."""
    ao, bo, both = chunk_sets(M_csc, a_unit, b_unit)
    return {"a_fit": ao[fit[ao]], "a_eval": ao[~fit[ao]], "b_fit": bo[fit[bo]], "b_eval": bo[~fit[bo]], "both": both}


def unit_index(records) -> dict:
    """{tuple(sorted concept ids): column} over every member of every record (a record has a_ids and b_ids)."""
    uid: dict = {}
    for r in records:
        for key in (tuple(r["a_ids"]), tuple(r["b_ids"])):
            uid.setdefault(key, len(uid))
    return uid


def pair_exclusive_report(cache: "AtlasCache", records, min_excl: int, n_concepts: int = 16790):
    """For each pair record: [A-only, B-only, shared] chunk counts in `cache` and whether both members reach
    `min_excl` exclusive chunks. Returns (list of dicts aligned to records, uid, membership csc)."""
    uid = unit_index(records)
    M = cache.membership([list(k) for k in uid], n_concepts)
    out = []
    for r in records:
        a, b = uid[tuple(r["a_ids"])], uid[tuple(r["b_ids"])]
        ao, bo, both = chunk_sets(M, a, b)
        out.append({"a_only": int(len(ao)), "b_only": int(len(bo)), "shared": int(len(both)),
                    "a_total": int(M.indptr[a + 1] - M.indptr[a]), "b_total": int(M.indptr[b + 1] - M.indptr[b]),
                    "passes": bool(len(ao) >= min_excl and len(bo) >= min_excl)})
    return out, uid, M
