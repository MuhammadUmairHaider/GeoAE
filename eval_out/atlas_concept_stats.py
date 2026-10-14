"""Per-concept Atlas statistics: does a representation space group the rows of ONE Atlas concept across documents?

CPU only (numpy; scipy.stats for the two tests). Implements the pre-registered plan
results/delta/g4b_l33_d5120/evals/sae_suite/atlas_concepts/ANALYSIS_PLAN.md. Inputs: the per-space cluster labels of every
Atlas cache row (eval_out/atlas_token_detail.py -> assign.npz), the chunk / label CSR of the cache (chunks.npz), the concept
metadata and, for metric 2, the cross-document kNN graph of eval_out/atlas_knn_graph.py (skipped when it does not exist).

Concepts: counts over ALL chunks; a concept is eligible in a set (--sets name:min_chunks:min_docs:max_prevalence) when it
is carried by >= min_chunks chunks, >= min_docs documents and <= max_prevalence of the chunks. A sampled row (final-token
rows excluded) is POSITIVE for c when its chunk carries c (labels are multi-label; no exclusivity filter).
  metric 1  lift_c   = P(same cluster | two positive rows, different documents) / P(same cluster | two rows, different
            documents), over all sampled rows. With n_ckd = positive rows of c in cluster k from document d:
            A_c = sum_k n_ck^2 - sum_kd n_ckd^2 (same-cluster pairs), B_c = n_c^2 - sum_d n_cd^2 (pairs), G the same on all
            sampled rows (N_k, N_kd, N_d); lift_c = (A_c / B_c) / G.
  metric 2  enrich_c = precision_c / expected_c: precision_c = hits / (k * m_c) with hits = neighbours (of the k, other
            documents only) of a positive graph row whose chunk carries c; expected_c = mean over positive queries of the
            share of positive rows among the graph rows of other documents.
Document bootstrap: w_d ~ Multinomial(D, uniform over the D documents with sampled rows), one weight vector per rep shared by
all spaces and both metrics. --copy_pairs across (default, the plan): a document with weight w is w distinct copies, pairs
inside one copy are excluded, pairs across copies kept (unit weights reproduce the plain definitions exactly); --copy_pairs
exclude: all copies of a document are ONE document, every within-document pair is excluded (the pairs across copies of one
document co-cluster far more than pairs of different documents, so `across` shifts the bootstrap lift upward: see the
`gmr_boot_mean_doc` / `gmr_ci_doc_centered` fields and the `!` in the printout). The graph is held fixed (queries are weighted).
Comparison X vs Y: delta_c = ln(metric_X / metric_Y) over concepts where both are finite and > 0; GMR = exp(mean delta),
median ratio = exp(median delta), win = share of delta > 0; 95% percentile CI + SD over the document bootstrap, a concept
bootstrap CI of the GMR, Wilcoxon signed-rank and sign tests over concepts; strata = concept type and size terciles.

    python -u eval_out/atlas_concept_stats.py --out_dir results/delta/g4b_l33_d5120/evals/sae_suite/atlas_concepts
Writes OUT/concept_table.csv (first set, one row per concept) and OUT/concept_stats.json; prints a plain-text summary.

Inputs and comparisons are configurable; every default reproduces the pre-registered run exactly.
  --assign a.npz,b.npz   label arrays merged by key name; row_chunk / row_pos / chunk_doc / chunk_ntok must be identical in
                         all files; an arm present in two files must have identical labels.
  --graph g1.npz,g2.npz  nbr_<arm> merged; rows / chunk / doc must be identical; an arm present in two files must have
                         identical neighbours unless --graph_prefer first|last picks that file's copy (the number of
                         differing neighbour entries is logged).
  --m1_compare / --m2_compare  name=x:y,...: per concept the ratio of the metric of arm x to that of arm y (metric 1: cluster
                         arms, metric 2: graph arms). Arms they name are added to --cluster_arms / --knn_arms. The decision
                         rule stays tied to the comparison named bypass_vs_base.
  --unassigned_as_cluster  cluster labels of -1 (rows where a top-K readout has no active feature) become one extra cluster
                         per arm; the share of such rows is recorded per arm (JSON meta and the printed per-arm table).
  --out_prefix P         writes P + concept_table.csv and P + concept_stats.json (the printed log goes to stdout).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):    # login-node rule: never multi-thread
    os.environ.setdefault(_v, "1")

import numpy as np

SUITE = "results/delta/g4b_l33_d5120/evals/sae_suite"
DEFAULT_ASSIGN = f"{SUITE}/atlas_token/assign.npz"
DEFAULT_CHUNKS = "/work/hdd/bbyl/mhaider/GeoAE/atlas_g4b_l33/chunks.npz"
DEFAULT_CONCEPTS = "/work/hdd/bbyl/mhaider/GeoAE/atlas_g4b_l33/atlas_concept_meta.json"
DEFAULT_OUT = f"{SUITE}/atlas_concepts"
DEFAULT_GRAPH = f"{DEFAULT_OUT}/knn_graph.npz"
TYPES = ("content", "entity", "tone", "document")
# comparisons name=x:y (ratio of the metric of arm x to that of arm y, per concept): metric 1 compares cluster arms, metric 2
# neighbour-graph arms (the SAE: k-means on its codes for metric 1, its codes for metric 2). The decision rule uses the one
# named DECISION_COMPARISON.
DEFAULT_M1_COMPARE = "bypass_vs_base=bypass:base,bypass_vs_tokrm=bypass:base_tokmean,bypass_vs_sae=bypass:mat10k_km"
DEFAULT_M2_COMPARE = "bypass_vs_base=bypass:base,bypass_vs_tokrm=bypass:base_tokmean,bypass_vs_sae=bypass:mat10k"
DECISION_COMPARISON = "bypass_vs_base"
METRICS = {"m1": ("metric1", "lift", "cross-document cluster lift"),
           "m2": ("metric2", "enrich", "cross-document neighbour enrichment")}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


@contextmanager
def stage(timings, name):
    t0 = time.time()
    yield
    timings[name] = round(time.time() - t0, 2)
    log(f"{name}: {timings[name]:.1f}s")


# ------------------------------------------------------------------------------------------------ concepts and pairs
def sampled_mask(row_chunk, row_pos, chunk_ntok):
    """True for the sampled rows: every row except the final-token row of its chunk (position n_tok - 1)."""
    row_chunk = np.asarray(row_chunk, dtype=np.int64)
    return np.asarray(row_pos, dtype=np.int64) != np.asarray(chunk_ntok, dtype=np.int64)[row_chunk] - 1


def concept_counts(label_ptr, label_ids, chunk_doc, n_all):
    """(n_chunks_c, n_docs_c, pair_chunk, pair_concept): per concept the chunks carrying it and the distinct documents of
    those chunks, over ALL chunks; plus the unique (chunk, concept) label pairs, sorted by chunk then concept."""
    label_ptr = np.asarray(label_ptr, dtype=np.int64)
    n_chunks = len(label_ptr) - 1
    chunk_of = np.repeat(np.arange(n_chunks, dtype=np.int64), np.diff(label_ptr))
    key = np.unique(chunk_of * n_all + np.asarray(label_ids, dtype=np.int64))
    pair_chunk, pair_concept = key // n_all, key % n_all
    n_chunks_c = np.bincount(pair_concept, minlength=n_all)
    chunk_doc = np.asarray(chunk_doc, dtype=np.int64)
    span = int(chunk_doc.max()) + 1
    n_docs_c = np.bincount(np.unique(pair_concept * span + chunk_doc[pair_chunk]) // span, minlength=n_all)
    return n_chunks_c, n_docs_c, pair_chunk, pair_concept


def eligible_mask(n_chunks_c, n_docs_c, n_chunks, min_chunks, min_docs, max_prev):
    return (n_chunks_c >= min_chunks) & (n_docs_c >= min_docs) & (n_chunks_c / n_chunks <= max_prev)


def parse_sets(text):
    """'primary:100:30:0.2,robust:50:20:0.2' -> {name: (min_chunks, min_docs, max_prevalence)}."""
    out = {}
    for item in [s for s in text.split(",") if s.strip()]:
        p = item.split(":")
        if len(p) != 4 or p[0] in out:
            raise ValueError(f"--sets item {item!r}: want unique name:min_chunks:min_docs:max_prevalence")
        out[p[0]] = (int(p[1]), int(p[2]), float(p[3]))
    if not out:
        raise ValueError("--sets is empty")
    return out


def split_list(text):
    """'a, b,,c' -> ['a', 'b', 'c'] (comma-separated command-line list: items stripped, empty items dropped)."""
    return [s.strip() for s in text.split(",") if s.strip()]


def parse_compare(text, flag="--m1_compare"):
    """'bypass_vs_base=bypass:base,sae_vs_base=mat10k:base' -> {name: (x, y)} in the given order: the comparison `name` is
    the per-concept ratio of the metric of arm x to that of arm y. An empty text means no comparisons."""
    out = {}
    for item in split_list(text):
        name, eq, pair = item.partition("=")
        x, colon, y = pair.partition(":")
        name, x, y = name.strip(), x.strip(), y.strip()
        if not (eq and colon and name and x and y) or ":" in y or "=" in pair:
            raise ValueError(f"{flag} item {item!r}: want name=x:y (x and y are arm names)")
        if name in out:
            raise ValueError(f"{flag}: comparison name {name!r} is used twice")
        if x == y:
            raise ValueError(f"{flag} item {item!r}: x and y are the same arm")
        out[name] = (x, y)
    return out


DEFAULT_COMPARE_SPECS = {"m1": parse_compare(DEFAULT_M1_COMPARE, "--m1_compare"),
                         "m2": parse_compare(DEFAULT_M2_COMPARE, "--m2_compare")}


def chunk_csr(pair_chunk, pair_local, n_chunks):
    """CSR over chunks of the concepts (local indices) each chunk carries; pairs must be sorted by chunk."""
    ptr = np.zeros(n_chunks + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(pair_chunk, minlength=n_chunks))
    return ptr, np.asarray(pair_local, dtype=np.int64)


def expand_rows(chunk_of_row, ptr, ids):
    """Positive pairs: every row paired with each concept of its chunk's CSR slice -> (row index, concept)."""
    chunk_of_row = np.asarray(chunk_of_row, dtype=np.int64)
    cnt = ptr[chunk_of_row + 1] - ptr[chunk_of_row]
    total = int(cnt.sum())
    row = np.repeat(np.arange(len(chunk_of_row), dtype=np.int64), cnt)
    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return row, ids[np.repeat(ptr[chunk_of_row], cnt) + within]


def doc_weights(n_docs, n_boot, seed):
    """(n_boot, n_docs) integer multinomial document counts: D draws with replacement from the D documents."""
    return np.random.default_rng(seed).multinomial(n_docs, np.full(n_docs, 1.0 / n_docs), size=n_boot)


# ------------------------------------------------------------------------------------------------ metric 1
class CDGroups:
    """Arm-independent (concept, document) groups of the positive rows: n_cd = positive rows of c in document d."""

    def __init__(self, pair_c, pair_d, n_docs):
        self.key, n = np.unique(np.asarray(pair_c, dtype=np.int64) * n_docs + pair_d, return_counts=True)
        self.c = (self.key // n_docs).astype(np.intp)
        self.d = (self.key % n_docs).astype(np.intp)
        self.n = n.astype(np.float64)


COPY_PAIRS = ("across", "exclude")


def within_weights(w, copy_pairs):
    """Weight of a document's within-document pair terms (n^2): w_d when only pairs inside one copy are within-document
    (`across`: w_d copies x n^2 ordered pairs each), w_d^2 when all copies are one document (`exclude`: (w_d n)^2)."""
    return w if copy_pairs == "across" else w * w


def b_parts(cd, w, n_conc, ww=None):
    """(n_c(w), B_c(w)) = (sum_d w_d n_cd, n_c(w)^2 - sum_d ww_d n_cd^2): ordered positive pairs from different documents
    (ww = w: copies of a document are different documents)."""
    ww = w if ww is None else ww
    n_c = np.bincount(cd.c, weights=w[cd.d] * cd.n, minlength=n_conc)
    return n_c, n_c * n_c - np.bincount(cd.c, weights=ww[cd.d] * (cd.n * cd.n), minlength=n_conc)


class ArmGroups:
    """One cluster arm: unique (concept, cluster, document) triples with counts n_ckd, their (c, k) group index, the
    per-(c, d) sum_k n_ckd^2, and the (cluster, document) counts of ALL sampled rows (for G). Built once; a bootstrap rep
    only gathers w_d per triple and bincounts."""

    def __init__(self, pair_c, pair_k, pair_d, row_k, row_d, cd, n_clusters, n_docs):
        self.n_clusters = n_clusters
        key, n = np.unique((np.asarray(pair_c, dtype=np.int64) * n_clusters + pair_k) * n_docs + pair_d,
                           return_counts=True)
        self.T = len(key)
        ck = key // n_docs
        self.d = (key % n_docs).astype(np.intp)
        self.n = n.astype(np.float64)
        if self.T:
            first = np.empty(self.T, dtype=bool)
            first[0] = True
            np.not_equal(ck[1:], ck[:-1], out=first[1:])
            self.ck = (np.cumsum(first) - 1).astype(np.intp)                   # (c, k) group of each triple
            self.ck_c = (ck[first] // n_clusters).astype(np.intp)               # concept of each (c, k) group
        else:
            self.ck = np.zeros(0, dtype=np.intp)
            self.ck_c = np.zeros(0, dtype=np.intp)
        cd_idx = np.searchsorted(cd.key, (ck // n_clusters) * n_docs + self.d)   # the triple's (c, d) group
        self.q = np.bincount(cd_idx, weights=self.n * self.n, minlength=len(cd.key))   # sum_k n_ckd^2 per (c, d)
        gk, gn = np.unique(np.asarray(row_k, dtype=np.int64) * n_docs + row_d, return_counts=True)
        self.g_k = (gk // n_docs).astype(np.intp)
        self.g_d = (gk % n_docs).astype(np.intp)
        self.g_n = gn.astype(np.float64)
        self.N_d = np.bincount(row_d, minlength=n_docs).astype(np.float64)


def a_parts(ag, cd, w, n_conc, buf=None, ww=None):
    """A_c(w) = sum_k S_ck^2 - sum_kd ww_d n_ckd^2 with S_ck = sum_d w_d n_ckd: ordered same-cluster positive pairs from
    different documents (ww = w: copies of a document are different documents). buf: optional float64 scratch of >= ag.T."""
    ww = w if ww is None else ww
    wn = np.take(w, ag.d, out=None if buf is None else buf[:ag.T], mode="clip")
    wn *= ag.n
    S = np.bincount(ag.ck, weights=wn, minlength=len(ag.ck_c))
    t1 = np.bincount(ag.ck_c, weights=S * S, minlength=n_conc)
    return t1 - np.bincount(cd.c, weights=ww[cd.d] * ag.q, minlength=n_conc)


def g_value(ag, w, ww=None):
    """G(w): P(same cluster) over ordered pairs of ALL sampled rows from different documents (see a_parts for ww)."""
    ww = w if ww is None else ww
    Nk = np.bincount(ag.g_k, weights=w[ag.g_d] * ag.g_n, minlength=ag.n_clusters)
    N = float(w @ ag.N_d)
    den = N * N - float(ww @ (ag.N_d * ag.N_d))
    return (float(Nk @ Nk) - float(ww[ag.g_d] @ (ag.g_n * ag.g_n))) / den if den > 0 else np.nan


def unassigned_to_cluster(L, arm=""):
    """Labels of the sampled rows of one arm, -1 = unassigned (a top-K readout with no active selected feature) -> (labels,
    n_unassigned, cluster_id): every -1 becomes ONE extra cluster id, max label + 1 (0 when no row is assigned); cluster_id
    is None when no row is unassigned (labels returned unchanged). Labels below -1 are not valid."""
    L = np.asarray(L, dtype=np.int64)
    if (L < -1).any():
        raise SystemExit(f"cluster arm {arm!r} has labels below -1 (-1 is the only unassigned label): min {int(L.min())}")
    un = L < 0
    n = int(un.sum())
    if n == 0:
        return L, 0, None
    extra = int(L.max()) + 1
    return np.where(un, extra, L), n, extra


class Metric1:
    """Cross-document cluster lift of every concept, in every cluster arm, for any document weight vector.
    unassigned_as_cluster: labels of -1 become one extra cluster per arm (unassigned_to_cluster; self.unassigned records
    arm -> n, share of the rows and the cluster id); otherwise they are an error."""

    def __init__(self, pair_c, pair_row, row_doc, row_labels, n_conc, n_docs, copy_pairs="across",
                 unassigned_as_cluster=False):
        assert copy_pairs in COPY_PAIRS, copy_pairs
        self.n_conc, self.n_docs, self.copy_pairs = n_conc, n_docs, copy_pairs
        pair_d = row_doc[pair_row]
        self.cd = CDGroups(pair_c, pair_d, n_docs)
        self.arms, self.unassigned = {}, {}
        for arm, L in row_labels.items():
            L = np.asarray(L, dtype=np.int64)
            if L.min() < 0:
                if not unassigned_as_cluster:
                    raise SystemExit(f"cluster arm {arm!r} has negative labels (unassigned rows); use its k-means arm "
                                     f"or pass --unassigned_as_cluster")
                L, n_un, cid = unassigned_to_cluster(L, arm)
                self.unassigned[arm] = dict(n=n_un, share=n_un / len(L), cluster_id=cid)
            else:
                self.unassigned[arm] = dict(n=0, share=0.0, cluster_id=None)
            self.arms[arm] = ArmGroups(pair_c, L[pair_row], pair_d, L, row_doc, self.cd, int(L.max()) + 1, n_docs)
        self.buf = np.empty(max([a.T for a in self.arms.values()] + [0]), dtype=np.float64)

    def parts(self, w):
        """arm -> (A_c, B_c, G) for the document weights w; B_c is arm-independent."""
        ww = within_weights(w, self.copy_pairs)
        _, B = b_parts(self.cd, w, self.n_conc, ww)
        return {arm: (a_parts(ag, self.cd, w, self.n_conc, self.buf, ww), B, g_value(ag, w, ww))
                for arm, ag in self.arms.items()}

    def lift(self, w):
        """arm -> lift_c (NaN where fewer than one positive cross-document pair exists)."""
        out = {}
        for arm, (A, B, G) in self.parts(w).items():
            x = np.full(self.n_conc, np.nan)
            ok = B > 0
            x[ok] = A[ok] / B[ok] / G
            out[arm] = x
        return out


# ------------------------------------------------------------------------------------------------ metric 2
def membership_keys(ptr, ids, union_ids, n_all):
    """Sorted int64 keys chunk * n_all + concept id of every (chunk, eligible concept) label pair."""
    chunk_of = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
    return np.sort(chunk_of * n_all + union_ids[ids])


def knn_hits(g_chunk, nbr, pair_row, pair_c, member, union_ids, n_all, block=400_000):
    """hits[p] for positive pair p = (graph row i, concept c): how many of i's k neighbours have a chunk carrying c
    (np.searchsorted membership of chunk * n_all + c in the sorted label keys)."""
    P = len(pair_row)
    hits = np.zeros(P, dtype=np.int32)
    if P == 0 or len(member) == 0:
        return hits
    nch = np.asarray(g_chunk, dtype=np.int64)[nbr]                         # (R, k) chunk of every neighbour
    for s in range(0, P, block):
        e = min(P, s + block)
        q = nch[pair_row[s:e]] * n_all + union_ids[pair_c[s:e]][:, None]   # (b, k)
        pos = np.searchsorted(member, q)
        np.minimum(pos, len(member) - 1, out=pos)
        hits[s:e] = (member[pos] == q).sum(1)
    return hits


class Metric2:
    """Cross-document neighbour enrichment of every concept for every graph arm and any document weight vector. Hits are
    held fixed; a document weight multiplies its positive queries and enters the expected-share pool."""

    def __init__(self, g_chunk, g_doc, nbrs, chunk_ptr, chunk_ids, union_ids, n_all, n_docs, copy_pairs="across"):
        assert copy_pairs in COPY_PAIRS, copy_pairs
        self.n_conc, self.n_docs, self.copy_pairs = len(union_ids), n_docs, copy_pairs
        g_chunk = np.asarray(g_chunk, dtype=np.int64)
        g_doc = np.asarray(g_doc, dtype=np.int64)
        pair_row, pair_c = expand_rows(g_chunk, chunk_ptr, chunk_ids)          # positive (graph row, concept) pairs
        pair_d = g_doc[pair_row]
        key, inv, m = np.unique(pair_c * n_docs + pair_d, return_inverse=True, return_counts=True)
        self.c = (key // n_docs).astype(np.intp)
        self.d = (key % n_docs).astype(np.intp)
        self.m = m.astype(np.float64)                                          # m_cd
        self.R_d = np.bincount(g_doc, minlength=n_docs).astype(np.float64)
        self.n_pos = np.bincount(pair_c, minlength=self.n_conc)                # positive graph rows per concept (m_c)
        member = membership_keys(chunk_ptr, chunk_ids, union_ids, n_all)
        self.k, self.H = None, {}
        for arm, nbr in nbrs.items():
            if self.k is not None and nbr.shape[1] != self.k:
                raise SystemExit(f"graph arm {arm!r} has k={nbr.shape[1]}, others k={self.k}")
            self.k = nbr.shape[1]
            hits = knn_hits(g_chunk, nbr, pair_row, pair_c, member, union_ids, n_all)
            self.H[arm] = np.bincount(inv.reshape(-1), weights=hits, minlength=len(key))   # hits per (c, d) group

    def common(self, w):
        """(m_c(w), expected_c(w)): arm-independent. expected_c(w) = sum_i w_d(i) (m_c(w) - m_c,d(i)) / (R(w) - R_d(i))
        / m_c(w) over positive queries i, summed per (c, d) group (all its m_cd queries share the same term). The pool of a
        query drops its own copy (`across`) or every copy of its document (`exclude`: m_c,d -> w_d m_c,d, R_d -> w_d R_d)."""
        wd = w[self.d]
        own = wd if self.copy_pairs == "exclude" else 1.0
        m_w = np.bincount(self.c, weights=wd * self.m, minlength=self.n_conc)
        denom = float(w @ self.R_d) - own * self.R_d[self.d]
        use = (wd > 0) & (denom > 0)
        term = np.zeros(len(self.c))
        own_m = (own * self.m)[use] if self.copy_pairs == "exclude" else self.m[use]
        term[use] = wd[use] * self.m[use] * (m_w[self.c[use]] - own_m) / denom[use]
        exp_num = np.bincount(self.c, weights=term, minlength=self.n_conc)
        bad = np.bincount(self.c[(wd > 0) & (denom <= 0)], minlength=self.n_conc) > 0   # all weight on one document
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.where((m_w > 0) & ~bad, exp_num / m_w, np.nan)
        return m_w, expected

    def precision(self, arm, w, m_w):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.bincount(self.c, weights=w[self.d] * self.H[arm], minlength=self.n_conc) / (self.k * m_w)

    def enrich(self, w):
        """arm -> enrich_c = precision_c / expected_c (NaN where undefined)."""
        m_w, expected = self.common(w)
        out = {}
        for arm in self.H:
            with np.errstate(divide="ignore", invalid="ignore"):
                x = self.precision(arm, w, m_w) / expected
            x[~((m_w > 0) & (expected > 0))] = np.nan
            out[arm] = x
        return out


# ------------------------------------------------------------------------------------------------ statistics
def log_ratios(x, y):
    """(delta, ok): ln(x / y) over the concepts where both are finite and > 0, and that mask."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return np.log(x[ok] / y[ok]), ok


def gmr_win(delta):
    """(GMR, median ratio, win share, wins, losses, ties) of log ratios; NaNs when there are none."""
    if len(delta) == 0:
        return np.nan, np.nan, np.nan, 0, 0, 0
    return (float(np.exp(delta.mean())), float(np.exp(np.median(delta))), float((delta > 0).mean()),
            int((delta > 0).sum()), int((delta < 0).sum()), int((delta == 0).sum()))


def boot_gmr_win(xb, yb):
    """Per-rep GMR(w), win(w) and #valid concepts from (B, n) per-rep metrics; a concept undefined (NaN, <= 0) in a rep
    is skipped in that rep."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ok = np.isfinite(xb) & np.isfinite(yb) & (xb > 0) & (yb > 0)
        dl = np.where(ok, np.log(np.where(ok, xb, 1.0) / np.where(ok, yb, 1.0)), 0.0)
    cnt = ok.sum(1)
    safe = np.maximum(cnt, 1)
    return (np.where(cnt > 0, np.exp(dl.sum(1) / safe), np.nan),
            np.where(cnt > 0, (dl > 0).sum(1) / safe, np.nan), cnt)


def interval(v, q=(2.5, 97.5)):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))] if len(v) else None


def sd(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if len(v) > 1 else None


def concept_boot_ci(delta, n_boot, rng, max_cells=20_000_000):
    """95% percentile CI of the GMR when the concepts (their log ratios) are resampled with replacement."""
    n = len(delta)
    if n < 2 or n_boot < 1:
        return None
    g = np.empty(n_boot)
    step = max(1, max_cells // n)
    for s in range(0, n_boot, step):
        e = min(n_boot, s + step)
        g[s:e] = delta[rng.integers(0, n, size=(e - s, n))].mean(1)
    return interval(np.exp(g))


def tests_over_concepts(delta):
    """Two-sided Wilcoxon signed-rank p over the log ratios and sign-test p (ties dropped)."""
    from scipy import stats
    wil = sign = None
    if len(delta) and np.any(delta != 0):
        wil = float(stats.wilcoxon(delta).pvalue)
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    if wins + losses:
        sign = float(stats.binomtest(wins, wins + losses, 0.5).pvalue)
    return wil, sign


def compare(x, y, xb, yb, rng, n_concept_boot, min_test=10):
    """Every statistic of one comparison X vs Y on one metric over a concept subset. x, y: (n,) point estimates;
    xb, yb: (B, n) per-rep (document-bootstrap) metrics of the same concepts."""
    delta, ok = log_ratios(x, y)
    gmr, med, win, wins, losses, ties = gmr_win(delta)
    gb, wb, cnt = boot_gmr_win(xb, yb)
    fin = gb[np.isfinite(gb)]
    out = dict(n_concepts=int(len(x)), n_valid=int(ok.sum()), n_dropped=int(len(x) - ok.sum()), gmr=gmr,
               median_ratio=med, win=win, n_win=wins, n_loss=losses, n_tie=ties,
               gmr_ci_doc=interval(gb), gmr_sd_doc=sd(gb), win_ci_doc=interval(wb), win_sd_doc=sd(wb),
               gmr_boot_mean_doc=float(fin.mean()) if len(fin) else None,
               gmr_ci_doc_centered=[gmr + float(np.percentile(fin - fin.mean(), q)) for q in (2.5, 97.5)] if len(fin) else None,
               n_valid_doc_boot_min=int(cnt.min()) if len(cnt) else None,
               n_valid_doc_boot_mean=float(cnt.mean()) if len(cnt) else None,
               gmr_ci_concept=None, wilcoxon_p=None, sign_p=None)
    if len(delta) >= min_test:
        out["gmr_ci_concept"] = concept_boot_ci(delta, n_concept_boot, rng)
        out["wilcoxon_p"], out["sign_p"] = tests_over_concepts(delta)
    return out


def arm_summary(x, xb):
    """Median (finite values) and geometric mean (finite, > 0) of one metric across concepts, + doc-bootstrap CI."""
    fin = x[np.isfinite(x)]
    pos = fin[fin > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.isfinite(xb) & (xb > 0)
        gm = np.exp(np.where(ok, np.log(np.where(ok, xb, 1.0)), 0.0).sum(1) / np.maximum(ok.sum(1), 1))
    gm = np.where(ok.sum(1) > 0, gm, np.nan)
    return dict(n_concepts=int(len(x)), n_finite=int(len(fin)), n_positive=int(len(pos)),
                median=float(np.median(fin)) if len(fin) else None,
                gmean=float(np.exp(np.log(pos).mean())) if len(pos) else None,
                gmean_ci_doc=interval(gm), gmean_sd_doc=sd(gm))


def make_strata(types, n_chunks):
    """name -> positions within the set: concept type and size terciles of n_chunks (value thresholds)."""
    types, n_chunks = np.asarray(types), np.asarray(n_chunks)
    out = {f"type:{t}": np.flatnonzero(types == t) for t in TYPES if (types == t).any()}
    lo, hi = np.quantile(n_chunks, [1 / 3, 2 / 3])
    out["size:T1_small"] = np.flatnonzero(n_chunks <= lo)
    out["size:T2_mid"] = np.flatnonzero((n_chunks > lo) & (n_chunks <= hi))
    out["size:T3_large"] = np.flatnonzero(n_chunks > hi)
    return {k: v for k, v in out.items() if len(v)}, [float(lo), float(hi)]


def analyse_set(set_i, idx, types, n_chunks, pt, boot, n_concept_boot, seed, compare_specs=None):
    """Everything for one concept set. idx: the set's concepts (local indices into the union arrays); pt[m][arm] (n_u,),
    boot[m][arm] (B, n_u); compare_specs[m] = {name: (X, Y)} (default: the plan's three comparisons). Returns the JSON-able
    dict."""
    compare_specs = DEFAULT_COMPARE_SPECS if compare_specs is None else compare_specs
    strata, terciles = make_strata(types[idx], n_chunks[idx])
    res = dict(n_concepts=int(len(idx)), by_type={t: int((types[idx] == t).sum()) for t in TYPES},
               size_terciles_n_chunks_upper=terciles, strata_sizes={k: int(len(v)) for k, v in strata.items()})
    for mi, m in enumerate(METRICS):
        name = METRICS[m][0]
        if m not in pt:
            res[name] = dict(skipped="no neighbour graph" if m == "m2" else "not run")
            continue
        r = dict(arms={a: arm_summary(pt[m][a][idx], boot[m][a][:, idx]) for a in pt[m]}, comparisons={}, skipped={})
        for ci, (cname, (X, Y)) in enumerate(compare_specs.get(m, {}).items()):
            if X not in pt[m] or Y not in pt[m]:
                r["skipped"][cname] = f"arm(s) {[a for a in (X, Y) if a not in pt[m]]} not available for {name}"
                continue
            cx, cy = pt[m][X][idx], pt[m][Y][idx]
            bx, by = boot[m][X][:, idx], boot[m][Y][:, idx]
            c = dict(x=X, y=Y, strata={})
            c["overall"] = compare(cx, cy, bx, by, np.random.default_rng([seed, set_i, mi, ci, 0]), n_concept_boot)
            for si, (sname, pos) in enumerate(strata.items(), 1):
                c["strata"][sname] = compare(cx[pos], cy[pos], bx[:, pos], by[:, pos],
                                             np.random.default_rng([seed, set_i, mi, ci, si]), n_concept_boot)
            gb, wb, _ = boot_gmr_win(bx, by)
            c["overall"]["boot_gmr"], c["overall"]["boot_win"] = gb.tolist(), wb.tolist()
            r["comparisons"][cname] = c
        res[name] = r
    # the plan's decision rule: bypass vs base, both metrics' GMR CI above 1 (tied to the comparison NAMED bypass_vs_base)
    lows, absent = {}, []
    for m in METRICS:
        if DECISION_COMPARISON not in compare_specs.get(m, {}):
            absent.append(METRICS[m][0])
        c = res[METRICS[m][0]].get("comparisons", {}).get(DECISION_COMPARISON) if m in pt else None
        ci = c["overall"]["gmr_ci_doc"] if c else None
        lows[m] = None if ci is None else ci[0]
    res["decision_rule"] = dict(
        rule="bypass groups concepts better than base iff the 95% doc-bootstrap CI of the GMR is above 1 for both metrics",
        ci_lower={METRICS[m][0]: v for m, v in lows.items()},
        passed=None if any(v is None for v in lows.values()) else bool(all(v > 1 for v in lows.values())))
    if absent:
        res["decision_rule"]["skipped"] = f"no {DECISION_COMPARISON} comparison configured for {', '.join(absent)}"
    return res


# ------------------------------------------------------------------------------------------------ input and output
ASSIGN_SPINE = ("row_chunk", "row_pos", "chunk_doc", "chunk_ntok")


def n_differ(a, b):
    """Number of positions where two arrays of one shape differ; None when the shapes differ."""
    return None if a.shape != b.shape else int((a != b).sum())


def load_assign(paths, arms):
    """The cluster labels of `arms` from one or more assign npz files (eval_out/atlas_token_detail.py), merged by key name.
    row_chunk / row_pos / chunk_doc / chunk_ntok must be identical in every file; an arm present in several files must have
    identical labels (only the requested arms are compared). -> (dict with the four spine arrays and arm -> labels,
    {arm: [files holding it]})."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [str(p) for p in paths]
    if not paths:
        raise SystemExit("--assign: no file given")
    spine, spine_path, arrays, holders, avail = None, None, {}, {}, set()
    for p in paths:
        with np.load(p) as F:
            miss = [k for k in ASSIGN_SPINE if k not in F.files]
            if miss:
                raise SystemExit(f"{p}: missing key(s) {miss}")
            sp = {k: F[k] for k in ASSIGN_SPINE}
            if spine is None:
                spine, spine_path = sp, p
            else:
                for k in ASSIGN_SPINE:
                    if not np.array_equal(sp[k], spine[k]):
                        raise SystemExit(f"{p}: {k} differs from {spine_path}: the assign files must describe the same "
                                         f"cache rows")
            avail.update(set(F.files) - set(ASSIGN_SPINE))
            for a in arms:
                if a not in F.files:
                    continue
                L = F[a]
                if a in arrays and not np.array_equal(arrays[a], L):
                    nd = n_differ(arrays[a], L)
                    what = "different shapes" if nd is None else f"{nd:,} of {L.size:,} rows differ"
                    raise SystemExit(f"cluster arm {a!r} is in both {holders[a][0]} and {p} with different labels ({what})")
                arrays.setdefault(a, L)
                holders.setdefault(a, []).append(p)
    absent = [a for a in arms if a not in arrays]
    if absent:
        raise SystemExit(f"{', '.join(paths)} {'has' if len(paths) == 1 else 'have'} no cluster arm(s) {absent}; "
                         f"arms available: {sorted(avail)}")
    return dict(spine, **arrays), holders


def n_neighbour_diff(a, b):
    """(entries, rows) that differ between two (R, k) neighbour arrays: positions holding a different neighbour, and rows
    whose SET of neighbours differs (a different order alone does not change the graph); (None, None) when the shapes differ
    or are not (R, k)."""
    if a.shape != b.shape or a.ndim != 2:
        return None, None
    return int((a != b).sum()), int((np.sort(a, axis=1) != np.sort(b, axis=1)).any(1).sum())


def load_graph(paths, knn_arms, n_rows, row_chunk, chunk_doc, sampled, doc_ids, prefer=None):
    """The kNN graph(s) (eval_out/atlas_knn_graph.py), validated against the cache. paths: one npz or a list of npz files
    over the same rows (rows / chunk / doc identical in all), whose nbr_<arm> arrays are merged by arm. An arm in several
    files must have identical neighbours unless prefer ('first' | 'last') names the file whose copy is used (the number of
    differing entries and rows is logged). -> dict(rows, chunk, doc, nbrs, missing, files, arm_files, duplicates)."""
    if prefer not in (None, "first", "last"):
        raise ValueError(f"prefer must be None, 'first' or 'last', not {prefer!r}")
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [str(p) for p in paths]
    if not paths:
        raise SystemExit("--graph: no file given")
    spine, spine_path, cand = None, None, {}
    for j, path in enumerate(paths):
        with np.load(path) as G:
            for key in ("rows", "chunk", "doc"):
                if key not in G.files:
                    raise SystemExit(f"{path}: missing key {key!r}")
            rows = G["rows"].astype(np.int64)
            if len(rows) == 0 or (np.diff(rows) <= 0).any() or rows.min() < 0 or rows.max() >= n_rows:
                raise SystemExit(f"{path}: `rows` must be strictly increasing cache row indices")
            if not sampled[rows].all():
                raise SystemExit(f"{path}: contains final-token rows")
            chunk, doc = G["chunk"].astype(np.int64), G["doc"].astype(np.int64)
            if (chunk != row_chunk[rows]).any() or (doc != chunk_doc[chunk]).any():
                raise SystemExit(f"{path}: chunk/doc columns disagree with the cache")
            g_doc = np.searchsorted(doc_ids, doc)
            if (doc_ids[np.minimum(g_doc, len(doc_ids) - 1)] != doc).any():
                raise SystemExit(f"{path}: a graph row's document has no sampled rows")
            if spine is None:
                spine, spine_path = (rows, chunk, doc, g_doc), path
            elif not all(np.array_equal(a, b) for a, b in zip((rows, chunk, doc), spine[:3])):
                raise SystemExit(f"{path}: rows/chunk/doc differ from {spine_path}: the graph files must be over the "
                                 f"same rows")
            for arm in knn_arms:
                if f"nbr_{arm}" in G.files:
                    cand.setdefault(arm, []).append((j, path, G[f"nbr_{arm}"]))
    rows, chunk, _, g_doc = spine
    R = len(rows)
    nbrs, missing, arm_files, dups = {}, [], {}, {}
    for arm in knn_arms:
        if arm in nbrs:
            continue
        cs = cand.get(arm, [])
        if not cs:
            missing.append(arm)
            continue
        pj, ppath, pnb = cs[-1] if prefer == "last" else cs[0]
        if len(cs) > 1:
            diffs = {p: n_neighbour_diff(pnb, nb) for j, p, nb in cs if j != pj}
            differ = {p: d for p, d in diffs.items() if d != (0, 0)}
            if differ and prefer is None:
                p, (ne, nr) = next(iter(differ.items()))
                what = "different shapes" if ne is None else f"{ne:,} of {pnb.size:,} neighbour entries differ"
                raise SystemExit(f"nbr_{arm} is in both {ppath} and {p} with different neighbours ({what}); "
                                 f"pass --graph_prefer first|last to choose one file's copy")
            for p, (ne, nr) in differ.items():
                log(f"graph arm {arm!r}: " + ("shapes differ" if ne is None else
                    f"{ne:,} of {pnb.size:,} neighbour entries ({nr:,} of {R:,} rows with a different neighbour set) differ")
                    + f" between {ppath} (used, --graph_prefer {prefer}) and {p}")
            if not differ:
                log(f"graph arm {arm!r}: identical in {len(cs)} files")
            dups[arm] = dict(used=ppath, prefer=prefer, also_in=[p for j, p, _ in cs if j != pj],
                             n_entries_differ={p: d[0] for p, d in diffs.items()},
                             n_rows_differ={p: d[1] for p, d in diffs.items()})
        nb = pnb.astype(np.int64)
        if nb.ndim != 2 or len(nb) != R or nb.min() < 0 or nb.max() >= R:
            raise SystemExit(f"{ppath}: nbr_{arm} must be (R, k) positions in `rows`")
        if (g_doc[nb] == g_doc[:, None]).any():
            raise SystemExit(f"{ppath}: nbr_{arm} has a neighbour in the query's own document")
        nbrs[arm] = nb
        arm_files[arm] = ppath
    return dict(rows=rows, chunk=chunk, doc=g_doc, nbrs=nbrs, missing=missing, files=paths, arm_files=arm_files,
                duplicates=dups)


def fnum(v, d=3):
    return "  n/a " if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def fp(p):
    return "  n/a  " if p is None else (f"{p:.1e}" if p > 0 else "<1e-300")


def fci(ci, d=3):
    return "[ n/a ]" if ci is None else f"[{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def fshare(share):
    """A share of rows as a percentage that is not rounded down to 0 while some rows are in it: '0%', '12.3%', '0.0034%'."""
    if not share:
        return "0%"
    p = 100 * share
    return f"{p:.1f}%" if p >= 0.1 else f"{p:.2g}%"


def print_set(name, spec, res, unassigned=None):
    """Plain-text summary of one set. unassigned: {cluster arm: share of rows labelled -1} -> an extra column in the
    metric-1 arm table (only when --unassigned_as_cluster is on)."""
    mc, md, mp = spec
    print(f"\n=== set {name}: >= {mc} chunks, >= {md} documents, prevalence <= {mp}: {res['n_concepts']} concepts "
          f"({', '.join(f'{k} {v}' for k, v in res['by_type'].items())}) ===")
    for m, (key, short, title) in METRICS.items():
        r = res[key]
        print(f"\n-- {key}: {title}")
        if "skipped" in r and "comparisons" not in r:
            print(f"   skipped: {r['skipped']}")
            continue
        w = max([14] + [len(a) + 1 for a in r["arms"]])
        un = unassigned if m == "m1" and unassigned else None
        print(f"   {'arm':<{w}}" + (f"{'unassigned':>11}" if un else "")
              + f"{'median':>9}{'geo-mean':>10}  doc-boot 95% CI of the geo-mean")
        for a, s in r["arms"].items():
            ci = s["gmean_ci_doc"]
            flag = "!" if ci is not None and s["gmean"] is not None and not ci[0] <= s["gmean"] <= ci[1] else ""
            print(f"   {a:<{w}}" + (f"{fshare(un.get(a, 0.0)):>11}" if un else "")
                  + f"{fnum(s['median']):>9}{fnum(s['gmean']):>10}  {fci(ci)}{flag}")
        for cname, why in r["skipped"].items():
            print(f"   {cname}: skipped, {why}")
        for cname, c in r["comparisons"].items():
            print(f"   {cname} ({c['x']} vs {c['y']}), per-concept log ratio")
            print(f"      {'':<16}{'n':>6}{'drop':>6}{'GMR':>8}  {'doc-boot 95% CI':<18}{'median':>8}{'win':>7}"
                  f"{'Wilcoxon p':>12}{'sign p':>10}  concept-boot CI")
            rows = [("ALL", c["overall"])] + list(c["strata"].items())
            for sname, s in rows:
                ci = s["gmr_ci_doc"]
                flag = "!" if ci is not None and np.isfinite(s["gmr"]) and not ci[0] <= s["gmr"] <= ci[1] else " "
                print(f"      {sname:<16}{s['n_valid']:>6}{s['n_dropped']:>6}{fnum(s['gmr']):>8}  {fci(ci) + flag:<19}"
                      f"{fnum(s['median_ratio']):>8}{fnum(s['win'], 2):>7}{fp(s['wilcoxon_p']):>12}{fp(s['sign_p']):>10}"
                      f"  {fci(s['gmr_ci_concept']) if sname == 'ALL' else ''}")
    print("\n   ! = the percentile CI does not contain the point estimate: the document bootstrap is biased "
          "(see --copy_pairs; gmr_ci_doc_centered in the JSON is bias-shifted)")
    d = res["decision_rule"]
    if "skipped" in d:
        print(f"\n   decision rule ({DECISION_COMPARISON}, doc-bootstrap GMR CI above 1 for both metrics): "
              f"skipped, {d['skipped']}")
        return
    low = ", ".join(f"{k} lower bound {fnum(v)}" for k, v in d["ci_lower"].items())
    verdict = {True: "PASS", False: "FAIL", None: "undetermined (a metric was not run)"}[d["passed"]]
    print(f"\n   decision rule ({DECISION_COMPARISON}, doc-bootstrap GMR CI above 1 for both metrics): {low} -> {verdict}")


def clean(o):
    """JSON-able copy: numpy scalars/arrays to python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assign", default=DEFAULT_ASSIGN,
                    help="assign.npz, or a comma-separated list of files merged by key name (same rows; an arm in two "
                         "files must have identical labels)")
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--concepts", default=DEFAULT_CONCEPTS)
    ap.add_argument("--graph", default=DEFAULT_GRAPH,
                    help="knn_graph.npz, or a comma-separated list of graph files over the same rows (nbr_<arm> merged); "
                         "metric 2 is skipped when none exists")
    ap.add_argument("--graph_prefer", nargs="?", const="first", default=None, choices=("first", "last"),
                    help="an arm in several --graph files must have identical neighbours (otherwise an error); with this "
                         "option the copy of the first / last file is used instead (bare flag = first) and the number of "
                         "differing neighbour entries is logged")
    ap.add_argument("--cluster_arms", default="base,base_tokmean,bypass,mat10k_km")
    ap.add_argument("--knn_arms", default="base,base_tokmean,bypass,mat10k")
    ap.add_argument("--m1_compare", default=DEFAULT_M1_COMPARE,
                    help="metric-1 comparisons name=x:y,...: per concept the lift of cluster arm x / that of arm y; arms "
                         "named here are added to --cluster_arms. The decision rule uses the one named bypass_vs_base")
    ap.add_argument("--m2_compare", default=DEFAULT_M2_COMPARE,
                    help="metric-2 comparisons name=x:y,... over the graph arms (added to --knn_arms when missing)")
    ap.add_argument("--unassigned_as_cluster", action="store_true",
                    help="cluster labels of -1 (rows where a top-K readout has no active selected feature) become ONE extra "
                         "cluster per arm (max label + 1); the share of such sampled rows is recorded per arm. Without the "
                         "flag a -1 label is an error")
    ap.add_argument("--sets", default="primary:100:30:0.2,robust:50:20:0.2", help="name:min_chunks:min_docs:max_prevalence,...")
    ap.add_argument("--n_boot", type=int, default=300, help="document-bootstrap reps")
    ap.add_argument("--n_concept_boot", type=int, default=10000)
    ap.add_argument("--copy_pairs", default="across", choices=COPY_PAIRS,
                    help="bootstrap: pairs across the copies of one resampled document count as cross-document (across, the "
                         "plan) or are excluded like any within-document pair (exclude)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", default="sampled", choices=("sampled", "final", "all"),
                    help="rows scored: sampled = every row except each chunk's final token (the plan); final = ONLY the "
                         "final-token row of each chunk (the row the sequence probes read; metric 2 is then skipped); "
                         "all = every row (a --last_word cache, one row per chunk; graph from atlas_knn_graph --all_rows)")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--out_prefix", default="",
                    help="prefix of the output file names: PREFIXconcept_stats.json, PREFIXconcept_table.csv (so that a new "
                         "run does not overwrite another)")
    args = ap.parse_args(argv)
    if args.n_boot < 2 or args.n_concept_boot < 1:
        ap.error("--n_boot must be >= 2 and --n_concept_boot >= 1")
    if "/" in args.out_prefix:
        ap.error("--out_prefix is a file-name prefix and must not contain '/' (use --out_dir for the directory)")
    sets = parse_sets(args.sets)
    try:
        cmp_specs = {"m1": parse_compare(args.m1_compare, "--m1_compare"),
                     "m2": parse_compare(args.m2_compare, "--m2_compare")}
    except ValueError as e:
        ap.error(str(e))
    cl_arms = list(dict.fromkeys(split_list(args.cluster_arms)))
    knn_arms = list(dict.fromkeys(split_list(args.knn_arms)))
    added = {}                      # arms the comparisons need that --cluster_arms / --knn_arms did not list
    for m, arms, flag in (("m1", cl_arms, "--cluster_arms"), ("m2", knn_arms, "--knn_arms")):
        added[flag] = [a for a in dict.fromkeys(a for xy in cmp_specs[m].values() for a in xy) if a not in arms]
        if added[flag]:
            log(f"{METRICS[m][0]}: comparison arm(s) {added[flag]} added to {flag}")
            arms.extend(added[flag])
    timings, t_start = {}, time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table_path, stats_path = out / f"{args.out_prefix}concept_table.csv", out / f"{args.out_prefix}concept_stats.json"

    # ---- inputs
    with stage(timings, "load"):
        assign_files = split_list(args.assign)
        A, assign_arm_files = load_assign(assign_files, cl_arms)
        z = np.load(args.chunks)
        meta = json.load(open(args.concepts))
        row_chunk = A["row_chunk"].astype(np.int64)
        row_pos = A["row_pos"].astype(np.int64)
        chunk_doc = A["chunk_doc"].astype(np.int64)
        n_rows, n_chunks = len(row_chunk), len(chunk_doc)
        if (chunk_doc != z["doc_int_id"]).any() or (A["chunk_ntok"] != z["n_tok"]).any():
            raise SystemExit("assign.npz and chunks.npz describe different chunks")
        if (np.diff(z["row_start"]) != np.bincount(row_chunk, minlength=n_chunks)).any():
            raise SystemExit("assign.npz row_chunk disagrees with chunks.npz row_start")
        n_all = len(meta["concept_id"])
        if (np.asarray(meta["concept_id"]) != np.arange(n_all)).any() or z["label_ids"].max() >= n_all:
            raise SystemExit("concept metadata must list concept ids 0..n-1 covering every label id")
        types, names = np.asarray(meta["concept_type"]), meta["name"]
        smask = sampled_mask(row_chunk, row_pos, A["chunk_ntok"])
        if args.rows == "final":
            smask = ~smask
            args.graph = ""                                                 # the graph holds sampled rows only
        elif args.rows == "all":
            smask = np.ones_like(smask)
        s_idx = np.flatnonzero(smask)
        s_chunk = row_chunk[s_idx]
        doc_ids, s_doc = np.unique(chunk_doc[s_chunk], return_inverse=True)
        s_doc = s_doc.reshape(-1).astype(np.intp)
        D, N = len(doc_ids), len(s_idx)
        labels = {a: A[a].astype(np.int64)[s_idx] for a in cl_arms}
        graph = None
        graph_note = None
        graph_files = split_list(args.graph)
        present = [Path(g).exists() for g in graph_files]
        if graph_files and all(present):
            graph = load_graph(graph_files, knn_arms, n_rows, row_chunk, chunk_doc, smask, doc_ids, args.graph_prefer)
            if graph["missing"]:
                log(f"graph {args.graph} has no nbr_ for {graph['missing']}: those arms are skipped in metric 2")
            if not graph["nbrs"]:
                graph_note = f"{args.graph} holds none of the knn arms {knn_arms}"
                graph = None
        elif any(present):
            raise SystemExit(f"--graph: {[g for g, ok in zip(graph_files, present) if not ok]} do not exist (but "
                             f"{[g for g, ok in zip(graph_files, present) if ok]} do): give every file or none")
        elif not graph_files:
            graph_note = "no --graph file given"
        else:
            graph_note = f"{args.graph} does not exist" if len(graph_files) == 1 else f"none of {graph_files} exists"
        if graph is None:
            log(f"metric 2 SKIPPED: {graph_note}")
    log(f"{n_rows:,} cache rows, {N:,} scored rows (--rows {args.rows}), {n_chunks:,} chunks, {D:,} documents with sampled rows; "
        f"{n_all:,} concepts")

    # ---- concepts, eligibility, positive pairs (computed once for the union of the sets)
    with stage(timings, "concepts"):
        n_chunks_c, n_docs_c, pair_chunk, pair_concept = concept_counts(z["label_ptr"], z["label_ids"], chunk_doc, n_all)
        elig = {n: eligible_mask(n_chunks_c, n_docs_c, n_chunks, *p) for n, p in sets.items()}
        union_ids = np.flatnonzero(np.any(list(elig.values()), axis=0))
        if not len(union_ids):
            raise SystemExit("no concept is eligible in any set")
        n_u = len(union_ids)
        loc = np.full(n_all, -1, dtype=np.int64)
        loc[union_ids] = np.arange(n_u)
        set_idx = {n: loc[np.flatnonzero(m)] for n, m in elig.items()}
        keep = loc[pair_concept] >= 0
        ptr, ids = chunk_csr(pair_chunk[keep], loc[pair_concept[keep]], n_chunks)       # eligible concepts per chunk
        del pair_chunk, pair_concept, keep
        pair_row, pair_c = expand_rows(s_chunk, ptr, ids)                               # positive (sampled row, concept)
        n_pos_rows = np.bincount(pair_c, minlength=n_u)
        for n, i in set_idx.items():
            by = {t: int((types[union_ids[i]] == t).sum()) for t in TYPES}
            log(f"set {n}: {len(i)} concepts {by}")
        log(f"union {n_u} concepts, {len(pair_row):,} positive (row, concept) pairs")

    W = doc_weights(D, args.n_boot, args.seed)
    pt, boot = {}, {}

    # ---- metric 1
    with stage(timings, "metric1_prep"):
        m1 = Metric1(pair_c, pair_row, s_doc, labels, n_u, D, args.copy_pairs, args.unassigned_as_cluster)
        del pair_row, pair_c
        unassigned = m1.unassigned
        log("triples per arm: " + ", ".join(f"{a} {g.T:,}" for a, g in m1.arms.items()))
        if args.unassigned_as_cluster:
            log("unassigned (-1) sampled rows kept as one extra cluster: "
                + ", ".join(f"{a} {fshare(u['share'])}" for a, u in unassigned.items()))
    with stage(timings, "metric1_point"):
        pt["m1"] = m1.lift(np.ones(D))
    with stage(timings, "metric1_boot"):
        boot["m1"] = {a: np.full((args.n_boot, n_u), np.nan) for a in cl_arms}
        t0 = time.time()
        for b in range(args.n_boot):
            for a, x in m1.lift(W[b].astype(np.float64)).items():
                boot["m1"][a][b] = x
            if (b + 1) % 50 == 0:
                log(f"  metric 1 rep {b + 1}/{args.n_boot}  ({(time.time() - t0) / (b + 1):.2f}s/rep)")
    del m1

    # ---- metric 2
    if graph is not None:
        with stage(timings, "metric2_prep"):
            m2 = Metric2(graph["chunk"], graph["doc"], graph["nbrs"], ptr, ids, union_ids, n_all, D, args.copy_pairs)
            log(f"graph {len(graph['rows']):,} rows, k={m2.k}, arms {list(m2.H)}")
        with stage(timings, "metric2_point"):
            pt["m2"] = m2.enrich(np.ones(D))
        with stage(timings, "metric2_boot"):
            boot["m2"] = {a: np.full((args.n_boot, n_u), np.nan) for a in m2.H}
            for b in range(args.n_boot):
                for a, x in m2.enrich(W[b].astype(np.float64)).items():
                    boot["m2"][a][b] = x

    # ---- statistics
    result = dict(sets={})
    with stage(timings, "stats"):
        nc_u = n_chunks_c[union_ids]
        for si, (n, spec) in enumerate(sets.items()):
            result["sets"][n] = analyse_set(si, set_idx[n], types[union_ids], nc_u, pt, boot,
                                            args.n_concept_boot, args.seed, cmp_specs)
            result["sets"][n]["params"] = dict(min_chunks=spec[0], min_docs=spec[1], max_prevalence=spec[2])
    result["meta"] = dict(
        args=vars(args), n_cache_rows=n_rows, n_sampled_rows=N, n_final_rows_excluded=n_rows - N, n_chunks=n_chunks,
        n_documents=D, n_concepts_total=n_all, n_union_concepts=n_u, cluster_arms=cl_arms,
        metric2=dict(ran=graph is not None, reason=graph_note, knn_arms=list(pt.get("m2", {})),
                     arms_missing_from_graph=(graph or {}).get("missing", []),
                     graph_rows=None if graph is None else int(len(graph["rows"])), k=None if graph is None else m2.k),
        comparisons={m: {n: list(xy) for n, xy in c.items()} for m, c in cmp_specs.items()},
        arms_added_for_comparisons=added,
        unassigned_as_cluster=dict(enabled=bool(args.unassigned_as_cluster),
                                   share_of_sampled_rows={a: u["share"] for a, u in unassigned.items()},
                                   n_rows={a: u["n"] for a, u in unassigned.items()},
                                   cluster_id={a: u["cluster_id"] for a, u in unassigned.items()}),
        inputs=dict(assign_files=assign_files, assign_arm_files=assign_arm_files, graph_files=graph_files,
                    graph_arm_files=(graph or {}).get("arm_files", {}), graph_prefer=args.graph_prefer,
                    graph_duplicate_arms=(graph or {}).get("duplicates", {})),
        timings_s=timings)

    # ---- outputs
    first = next(iter(sets))
    fi = set_idx[first]
    cols = ["concept_id", "concept_type", "name", "n_chunks", "n_docs", "n_pos_rows"]
    if graph is not None:
        cols.append("n_graph_pos")
    cols += [f"lift_{a}" for a in pt["m1"]] + [f"enrich_{a}" for a in pt.get("m2", {})]
    with open(table_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for u in fi:
            g = union_ids[u]
            row = [int(g), types[g], names[g], int(n_chunks_c[g]), int(n_docs_c[g]), int(n_pos_rows[u])]
            if graph is not None:
                row.append(int(m2.n_pos[u]))
            wr.writerow(row + [repr(float(pt[m][a][u])) for m in ("m1", "m2") if m in pt for a in pt[m]])
    result["meta"]["total_s"] = round(time.time() - t_start, 1)
    stats_path.write_text(json.dumps(clean(result), indent=1))

    for n, spec in sets.items():
        print_set(n, spec, result["sets"][n],
                  {a: u["share"] for a, u in unassigned.items()} if args.unassigned_as_cluster else None)
    print(f"\ntimings (s): {timings}\ntotal {time.time() - t_start:.0f}s\nwrote {table_path} "
          f"({len(set_idx[first])} concepts of set {first}) and {stats_path}")


if __name__ == "__main__":
    main()
