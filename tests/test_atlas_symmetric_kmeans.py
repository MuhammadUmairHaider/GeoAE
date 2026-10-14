"""eval_out/atlas_symmetric_kmeans.py on tiny synthetic data (CPU): the row sample against eval_out/sae_controls_fit.py's,
the K-scaled schedule, codebook save / resume / validation, nearest-centroid and unit-readout labels against brute force,
the assign.npz format against eval_out/atlas_token_detail.py, and main() end to end on a synthetic dump + Atlas cache
with stub arms."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_p = _ROOT / "eval_out" / "atlas_symmetric_kmeans.py"
sys.path.insert(0, str(_p.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))             # test_sae_space's SAE builders
_spec = importlib.util.spec_from_file_location("atlas_symmetric_kmeans", _p)
SK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SK)

META_KEYS = ["row_chunk", "row_pos", "row_tok", "chunk_doc", "chunk_ntok"]


# ---------------------------------------------------------------- row sample: the call of eval_out/sae_controls_fit.py
def test_sae_controls_fit_still_makes_the_call_we_mirror():
    src = (_ROOT / "eval_out" / "sae_controls_fit.py").read_text()
    for line in ("n_train = int(len(mm) * args.train_frac)",
                 "idx = np.sort(np.random.RandomState(args.seed).choice(n_train, min(args.n_sample, n_train), replace=False))",
                 "C = balanced_fit(X, args.n_clusters, seed=args.seed, epochs=args.epochs, density_pool=args.density_pool,",
                 'ap.add_argument("--n_sample", type=int, default=200_000)',
                 'ap.add_argument("--seed", type=int, default=42)',
                 "H[s:s + 50_000] = mm[idx[s:s + 50_000]]"):
        assert line in src


def test_k2000_rows_equal_the_sae_controls_rows_at_production_size():
    n_train = int(10_000_000 * 0.95)                                    # the 4B dump: 10M rows, 95% train
    assert n_train == 9_500_000
    got = SK.sample_rows(n_train, 200_000, 42)
    ref = np.sort(np.random.RandomState(42).choice(n_train, min(200_000, n_train), replace=False))   # sae_controls_fit.py
    assert np.array_equal(got, ref) and np.array_equal(got, SK.sae_controls_rows(n_train, 42))
    assert got.dtype == np.int64 and (np.diff(got) > 0).all() and got[-1] < n_train


@pytest.mark.parametrize("n_total,frac,n_rows", [(3000, 0.95, 160), (3000, 0.95, 2850), (7001, 0.9, 5)])
def test_sample_rows_is_the_call_and_stays_in_the_train_split(n_total, frac, n_rows):
    n_train = int(n_total * frac)
    got = SK.sample_rows(n_train, n_rows, 42)
    assert np.array_equal(got, np.sort(np.random.RandomState(42).choice(n_train, n_rows, replace=False)))
    assert len(got) == n_rows and (np.diff(got) > 0).all() and got[-1] < n_train      # sorted, distinct, never validation


def test_row_sets_are_nested_across_k():
    sets = [set(SK.sample_rows(50_000, n, 42).tolist()) for n in (200, 400, 800)]    # same seed, growing n
    assert sets[0] < sets[1] < sets[2]


def test_too_small_a_train_split_is_an_error_not_a_silent_cap():
    with pytest.raises(SystemExit):
        SK.sample_rows(100, 101, 0)


def test_fingerprint_pins_the_rows():
    a = SK.sample_rows(5000, 100, 0)
    assert SK.fingerprint(a) == SK.fingerprint(a.copy()) and len(SK.fingerprint(a)) == 40
    assert SK.fingerprint(a) != SK.fingerprint(SK.sample_rows(5000, 100, 1))
    assert SK.fingerprint(a) != SK.fingerprint(a[:-1])


class FakeDump:
    def __init__(self, tmp_path, n=500, d=4):
        rng = np.random.default_rng(0)
        self.H = rng.normal(size=(n, d)).astype(np.float32)
        self.tok = rng.integers(0, 50, n).astype(np.int32)
        np.save(tmp_path / "layer_33.npy", self.H)
        np.save(tmp_path / "rows_tok.npy", self.tok)


def test_dump_read_matches_direct_indexing_in_slices(tmp_path, monkeypatch):
    f = FakeDump(tmp_path)
    monkeypatch.setattr(SK, "DUMP_READ_CHUNK", 13)                     # several slices
    d = SK.Dump(tmp_path, 33, 0.95)
    assert d.n_train == 475 and d.d == 4
    idx = d.index(100, 42)
    H, T = d.read(idx)
    assert H.dtype == np.float32 and T.dtype == np.int64
    assert np.array_equal(H, f.H[idx]) and np.array_equal(T, f.tok[idx])


# ---------------------------------------------------------------- the K-scaled schedule
def test_k_scaling_of_rows_and_batch_keeps_the_schedule_fixed():
    s = {K: SK.k_schedule(K, 100, 4096) for K in (2000, 4000, 8000)}
    assert [(v.n_rows, v.batch_size) for v in s.values()] == [(200_000, 4096), (400_000, 8192), (800_000, 16384)]
    assert {v.steps_per_epoch for v in s.values()} == {48}                # 200000 // 4096 == 400000 // 8192 == ...
    assert {v.steps_per_epoch * 40 for v in s.values()} == {1920}           # total steps at 40 epochs: identical at every K
    assert s[2000] == SK.Schedule(2000, 200_000, 4096, 48)                  # K=2000 is the sae_controls_fit.py fit


def test_k_schedule_rejects_unusable_batches():
    with pytest.raises(ValueError, match="batch size 2000 < K=4000"):
        SK.k_schedule(4000, 100, 1000)
    with pytest.raises(ValueError, match="not a multiple"):
        SK.k_schedule(7, 100, 4096)
    with pytest.raises(ValueError, match="rows"):
        SK.k_schedule(2000, 1, 4096)                                       # 2000 rows < a 4096 batch


def test_parse_ints():
    assert SK.parse_ints("500,1000,0,500", "--unit_ns", 0) == [500, 1000, 0]
    assert SK.parse_ints("2000", "--ks", 2) == [2000]
    for bad, lo in (("", 2), ("a,b", 2), ("1", 2), ("-1", 0)):
        with pytest.raises(SystemExit):
            SK.parse_ints(bad, "x", lo)


def test_cli_defaults_follow_the_spec():
    ns = SK.build_parser().parse_args(["--cache", "c", "--out_dir", "o"])
    assert (ns.layer, ns.train_frac, ns.seed) == (33, 0.95, 42)
    assert ns.spaces == "base,base_tokmean,bypass,mat10k,gs16k" and ns.ks == "2000,4000,8000"
    assert (ns.rows_per_centroid, ns.batch_per_k2000, ns.epochs, ns.density_pool) == (100, 4096, 40, 32768)
    assert ns.unit_ns == "500,1000,1500,2000,3000,0" and ns.ref_k == 2000 and ns.refit is False


def test_memory_estimate_is_at_least_the_row_matrix():
    gs = SK.fit_memory_gb(800_000, 16384, 8000, 16384, 32768)
    assert 52.4 + 1.5 < gs < 80                                            # 800k x 16384 fp32 = 52.4 GB + working set
    assert SK.fit_memory_gb(200_000, 2560, 2000, 4096, 32768) < 12


def test_too_big_fits_names_the_row_matrices_a_gpu_cannot_hold():
    sched = {K: SK.k_schedule(K, 100, 4096) for K in (2000, 4000, 8000)}
    dims = {"base": 2560, "bypass": 5120, "mat10k": 10240, "gs16k": 16384}
    ks, spaces = [2000, 4000, 8000], list(dims)
    assert SK.too_big_fits(94.0, sched, dims, ks, spaces) == []             # H100 NVL: gs16k at K=8000 holds 52.4 GB
    assert SK.too_big_fits(48.0, sched, dims, ks, spaces) == [("km8000_gs16k", 52.4)]          # an L40 cannot
    assert [k for k, _ in SK.too_big_fits(24.0, sched, dims, ks, spaces)] == [
        "km4000_gs16k", "km8000_mat10k", "km8000_gs16k"]


def test_cuda_helpers_call_the_torch_cuda_api(monkeypatch):
    dev, calls = types.SimpleNamespace(type="cuda"), []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: calls.append(("sync", d)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: 52_430_000_000)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty",)))
    assert SK.gpu_peak_gb(dev) == 52.43
    SK.clock(dev)
    SK.free_gpu(dev)
    assert ("sync", dev) in calls and ("empty",) in calls
    calls.clear()
    cpu = torch.device("cpu")
    assert SK.gpu_peak_gb(cpu) == -1.0                                      # CPU runs record -1
    SK.clock(cpu)
    SK.free_gpu(cpu)
    assert calls == []


def test_space_specs():
    assert SK.space_specs("bypass=a.pt", "base=b.npz,base_tokmean=c.npz", "mat10k=matched:m.pt,gs16k=scope:16k") == {
        "bypass": "a.pt", "base": "b.npz", "base_tokmean": "c.npz", "mat10k": "matched:m.pt", "gs16k": "scope:16k"}


# ---------------------------------------------------------------- codebook files: atomic save, validation, resume
EXPECT = dict(space="base", K=8, n_rows=160, seed=42, batch_size=80, epochs=3, density_pool=128, spec="s.npz",
              n_train=2850, rows_sha1="ab12")
STATS = dict(live=8, top10_share=0.5, eff_k=7.0, seconds=1.0, embed_seconds=0.1, fit_seconds=0.9, gpu_peak_gb=-1.0)


def test_codebook_roundtrip_is_atomic_and_validates(tmp_path):
    C = np.random.default_rng(0).normal(size=(8, 16)).astype(np.float32)
    path = SK.codebook_path(tmp_path, 8, "base")
    assert path == tmp_path / "codebooks" / "km8_base.npz"
    SK.save_codebook(path, C.astype(np.float64), EXPECT, STATS)             # float64 input is stored as float32
    assert path.exists() and not list(path.parent.glob("*.tmp*"))
    z = np.load(path, allow_pickle=False)                                  # no pickles: loadable by any downstream reader
    assert z["centroids"].dtype == np.float32 and np.array_equal(z["centroids"], C)
    assert {"centroids", "init", "reinit", *EXPECT, *STATS} <= set(z.files)
    assert SK.check_codebook(path, EXPECT, 16) is None
    assert SK.stored_stats(path) == STATS


@pytest.mark.parametrize("field,value", [("K", 9), ("epochs", 4), ("seed", 1), ("batch_size", 79), ("n_rows", 161),
                                         ("spec", "t.npz"), ("space", "bypass"), ("n_train", 1), ("rows_sha1", "cd34"),
                                         ("density_pool", 1)])
def test_codebook_validation_names_the_mismatching_field(tmp_path, field, value):
    path = SK.codebook_path(tmp_path, 8, "base")
    SK.save_codebook(path, np.zeros((8, 16), np.float32), EXPECT, STATS)
    why = SK.check_codebook(path, {**EXPECT, field: value}, 16)
    assert why and field in why


def test_codebook_validation_catches_shape_corruption_and_nan(tmp_path):
    path = SK.codebook_path(tmp_path, 8, "base")
    SK.save_codebook(path, np.zeros((8, 16), np.float32), EXPECT, STATS)
    assert "centroids" in SK.check_codebook(path, EXPECT, 17)               # wrong coordinate dimension
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"this is not an npz")
    assert "unreadable" in SK.check_codebook(bad, EXPECT, 16)
    nan = tmp_path / "nan.npz"
    SK.save_codebook(nan, np.full((8, 16), np.nan, np.float32), EXPECT, STATS)
    assert "finite" in SK.check_codebook(nan, EXPECT, 16)
    old_style = tmp_path / "old.npz"                                        # a sae_controls_fit.py file has no fit fields
    np.savez(old_style, centroids=np.zeros((8, 16), np.float32), spec="s.npz")
    assert "not stored" in SK.check_codebook(old_style, EXPECT, 16)


# ---------------------------------------------------------------- nearest centroid vs brute force
def test_nearest_centroid_matches_brute_force_for_any_chunk():
    rng = np.random.default_rng(0)
    V, C = rng.normal(size=(100, 6)).astype(np.float32), rng.normal(size=(9, 6)).astype(np.float32)
    d = ((V[:, None, :].astype(np.float64) - C[None].astype(np.float64)) ** 2).sum(-1)
    s = np.sort(d, axis=1)
    assert (s[:, 1] - s[:, 0]).min() > 1e-3                                # no near-ties: the argmin is unambiguous
    for chunk in (1, 7, 100, 1000):
        lab = SK.nearest_centroid(torch.from_numpy(V), torch.from_numpy(C), chunk=chunk)
        assert lab.dtype == torch.int64 and (lab.numpy() == d.argmin(1)).all()


# ---------------------------------------------------------------- unit readout vs brute force
def test_usage_order_is_descending_with_ties_to_the_lower_unit():
    assert SK.usage_order(np.array([5, 9, 9, 0, 7, 5])).tolist() == [1, 2, 4, 0, 5, 3]


def test_unit_sets_top_n_ascending_and_every_unit_for_zero_or_large_n():
    order = SK.usage_order(np.array([5, 9, 9, 0, 7, 5]))
    sets = SK.unit_sets(order, [0, 2, 4, 6, 100], 6)
    assert sets[0] is None and sets[6] is None and sets[100] is None       # N = 0, N >= dim: every unit
    assert sets[2].tolist() == [1, 2] and sets[4].tolist() == [0, 1, 2, 4]  # the N most used, in ascending unit id


def brute_labels(V, usage, N, nonneg):
    """numpy reference: argmax over the N most-used units (N=0: all), ties -> lower unit; -1 if nothing is > 0 (SAE)."""
    order = np.argsort(-usage, kind="stable")
    units = np.arange(V.shape[1]) if N == 0 or N >= V.shape[1] else np.sort(order[:N])
    sub = V[:, units]
    lab = units[sub.argmax(1)]
    return np.where(sub.max(1) > 0, lab, -1) if nonneg else lab


@pytest.mark.parametrize("nonneg", [False, True])
@pytest.mark.parametrize("N", [0, 1, 3, 7, 12, 50])
def test_coord_argmax_matches_brute_force(N, nonneg):
    rng = np.random.default_rng(1)
    V = rng.normal(size=(300, 12)).astype(np.float32)
    if nonneg:
        V = np.maximum(V - 1.0, 0)                                          # sparse non-negative codes; many all-zero rows
        V[::17] = 0
    full = brute_labels(V, np.zeros(12), 0, nonneg)
    usage = np.bincount(full[full >= 0], minlength=12)
    units = SK.unit_sets(SK.usage_order(usage), [N], 12)[N]
    got = SK.coord_argmax(torch.from_numpy(V), None if units is None else torch.from_numpy(units), nonneg).numpy()
    assert np.array_equal(got, brute_labels(V, usage, N, nonneg))
    assert (got == -1).any() == nonneg                                     # -1 only ever appears for SAE codes
    if not nonneg:
        assert (got >= 0).all()
    if N != 0 and N < 12:
        assert set(np.unique(got[got >= 0]).tolist()) <= set(np.sort(np.argsort(-usage, kind="stable")[:N]).tolist())


def test_coord_argmax_ties_go_to_the_lower_unit():
    V = torch.tensor([[1.0, 1.0, 0.0], [0.0, 2.0, 2.0], [3.0, 0.0, 3.0]])
    assert SK.coord_argmax(V, None, False).tolist() == [0, 1, 0]
    assert SK.coord_argmax(V, torch.tensor([1, 2]), False).tolist() == [1, 1, 2]


def test_unit_readout_with_every_unit_is_the_native_sae_top1():
    """The -1 convention: with N = 0 the readout of SAE codes is exactly SAESpace.top1_labels."""
    from geoae.interp.sae_space import SAESpace

    codes = torch.relu(torch.randn(200, 20, generator=torch.Generator().manual_seed(2)) - 1.2)
    codes[::9] = 0

    class Fixed(SAESpace):
        n_features = 20
        def encode_raw(self, h_raw, tok=None):
            return codes[h_raw[:, 0].long()]

    h = torch.arange(200).float()[:, None]
    native = Fixed().top1_labels(h)
    assert (native == -1).any()
    assert np.array_equal(SK.coord_argmax(codes, None, True).numpy(), native.numpy())


def test_usage_report():
    usage = np.array([5, 9, 9, 0, 7, 5])
    order = SK.usage_order(usage)
    r = SK.usage_report(usage, order, [0, 2, 4], n_rows=35, n_inactive=2)
    assert r["dim"] == 6 and r["n_rows"] == 35 and r["n_inactive"] == 2 and r["live_units"] == 5
    assert r["coverage_of_top_n"] == {"2": 18 / 35, "4": 30 / 35}          # N = 0 has no coverage entry
    p = usage[usage > 0] / 35
    assert np.isclose(r["eff_units"], np.exp(-(p * np.log(p)).sum())) and np.isclose(r["top10_share"], 1.0)


# ---------------------------------------------------------------- statistics
def test_count_and_label_stats():
    r = SK.count_stats([5, 3, 2, 0, 0])
    p = np.array([0.5, 0.3, 0.2])
    assert r["live"] == 3 and np.isclose(r["top10_share"], 1.0) and np.isclose(r["eff_k"], np.exp(-(p * np.log(p)).sum()))
    assert np.isclose(SK.count_stats([1, 1, 1, 1])["eff_k"], 4.0)          # uniform usage: effective K = K
    assert np.isclose(SK.count_stats(np.r_[np.ones(12), np.zeros(3)])["top10_share"], 10 / 12)
    assert SK.count_stats([0, 0]) == dict(live=0, top10_share=0.0, eff_k=0.0)
    lab = np.array([0, 0, 1, -1, 2, 2, -1, 0])
    s = SK.label_stats(lab)
    assert s["n_rows"] == 8 and s["live"] == 3 and np.isclose(s["unassigned"], 0.25)
    assert np.isclose(s["eff_k"], np.exp(-sum(q * np.log(q) for q in (3 / 6, 1 / 6, 2 / 6))))   # -1 is not a cluster
    m = SK.label_stats(lab, np.array([1, 1, 1, 1, 0, 0, 0, 0], bool))      # rows 0-3 only: 0, 0, 1, -1
    assert m["n_rows"] == 4 and m["live"] == 2 and np.isclose(m["unassigned"], 0.25)
    assert SK.label_stats(np.array([-1, -1]))["live"] == 0


def test_partition_nmi_against_sklearn_and_permutation_blindness():
    from sklearn.metrics import normalized_mutual_info_score as nmi

    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 6, 400), rng.integers(0, 9, 400)
    assert np.isclose(SK.partition_nmi(a, b), nmi(a, b, average_method="arithmetic"))
    perm = np.roll(np.arange(6), 1)
    assert np.isclose(SK.partition_nmi(a, perm[a]), 1.0)                   # relabelled clusters: same partition
    assert np.isclose(SK.partition_nmi(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])), 0.0)
    assert SK.partition_nmi(np.zeros(5, int), np.zeros(5, int)) == 1.0


# ---------------------------------------------------------------- synthetic dump + Atlas cache + stub arms
D, N_VOCAB, N_DUMP = 16, 40, 3000
SPACES = ("base", "base_tokmean", "bypass", "mat10k", "gs16k")
DIMS = {"base": D, "base_tokmean": D, "bypass": 12, "mat10k": 24, "gs16k": 40}
NONNEG = {"mat10k", "gs16k"}


def make_rows(rng, n):
    centers = np.random.default_rng(99).normal(size=(6, D)) * 3
    return (centers[rng.integers(0, 6, n)] + rng.normal(size=(n, D))).astype(np.float32)


def make_dump(path, seed=0, n=N_DUMP, vocab=N_VOCAB):
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    H, tok = make_rows(rng, n), rng.integers(0, vocab, n).astype(np.int32)
    np.save(path / "layer_33.npy", H)
    np.save(path / "rows_tok.npy", tok)
    return H, tok


def make_cache(path, n_chunks=12, seed=1, vocab=N_VOCAB):
    """AtlasCache layout: 12 chunks (3 per document), each with 4 sampled rows and the final-token row."""
    rng = np.random.default_rng(seed)
    n_tok = rng.integers(12, 30, size=n_chunks)
    pos, chunk = [], []
    for c in range(n_chunks):
        pos += [*np.sort(rng.choice(n_tok[c] - 1, size=4, replace=False)), n_tok[c] - 1]
        chunk += [c] * 5
    n = len(pos)
    path.mkdir(parents=True)
    np.save(path / "rows.npy", make_rows(rng, n))
    np.save(path / "rows_tok.npy", rng.integers(0, vocab, size=n).astype(np.int32))
    np.save(path / "rows_nxt.npy", rng.integers(0, vocab, size=n).astype(np.int32))
    np.save(path / "rows_chunk.npy", np.asarray(chunk, dtype=np.int32))
    np.save(path / "rows_pos.npy", np.asarray(pos, dtype=np.int16))
    ptr = np.zeros(n_chunks + 1, dtype=np.int64)
    np.savez(path / "chunks.npz", doc_int_id=(np.arange(n_chunks) // 3).astype(np.int32),
             chunk_id=np.arange(n_chunks, dtype=np.int32), n_tok=n_tok.astype(np.int32),
             row_start=np.arange(0, 5 * n_chunks + 1, 5, dtype=np.int64), label_ptr=ptr,
             label_ids=np.zeros(0, np.int32), tok_ptr=ptr, tok_flat=np.zeros(0, np.int32), fingerprint=np.asarray("test"))


class Stub:
    """The five spaces in the formats atlas_token_detail.embed reads: two k-means baselines (one token-mean-subtracted),
    a token-bypass AE, a token-bypass SAE (needs ids) and a Gemma-Scope-like SAE (raw residual, no ids)."""

    def __init__(self, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.mean, self.std = torch.randn(D, generator=g), torch.rand(D, generator=g) + 0.5
        self.table = torch.randn(N_VOCAB, D, generator=g) * 0.5
        self.W, self.bias = torch.randn(D, 12, generator=g), torch.randn(N_VOCAB, 12, generator=g) * 0.5
        self.Wm, self.Wg = torch.randn(D, 24, generator=g), torch.randn(D, 40, generator=g)
        cal = torch.from_numpy(make_rows(np.random.default_rng(5), 2000))      # sparse codes: ~10% of the pre-activations fire
        cal_tok = torch.from_numpy(np.random.default_rng(6).integers(0, N_VOCAB, 2000))
        self.thr_m = float((((cal - self.mean) / self.std - self.table[cal_tok]) @ self.Wm).quantile(0.9))
        self.thr_g = float((cal @ self.Wg).quantile(0.9))
        stub = self

        class AE:
            has_token_bias = True

            def encode(self, x, t):
                return x @ stub.W + stub.bias[t]

        class MatSAE:
            needs_tokens = True

            def encode_raw(self, h, tok=None):
                if tok is None:
                    raise ValueError("token-bypass SAE needs ids")
                return torch.relu(((h - stub.mean) / stub.std - stub.table[tok]) @ stub.Wm - stub.thr_m)

        class ScopeSAE:
            needs_tokens = False

            def encode_raw(self, h, tok=None):
                return torch.relu(h @ stub.Wg - stub.thr_g)

        self.arms = {"base": ("km", (None, self.mean, self.std, None, None)),
                     "base_tokmean": ("km", (None, self.mean, self.std, None, lambda t: self.table[t])),
                     "bypass": ("ae", (AE(), self.mean, self.std))}
        self.saes = {"mat10k": MatSAE(), "gs16k": ScopeSAE()}

    def coords(self, H, tok):
        """{space: float32 numpy (n, dim)} of rows H with token ids tok: the independent reference embedding."""
        h, t = torch.from_numpy(np.asarray(H, np.float32)), torch.from_numpy(np.asarray(tok, np.int64))
        x = (h - self.mean) / self.std
        out = {"base": x, "base_tokmean": x - self.table[t], "bypass": x @ self.W + self.bias[t],
               "mat10k": self.saes["mat10k"].encode_raw(h, t), "gs16k": self.saes["gs16k"].encode_raw(h, None)}
        return {k: v.numpy() for k, v in out.items()}


class Env:
    def __init__(self, tmp_path, stub):
        self.tmp, self.stub = tmp_path, stub
        self.H, self.tok = make_dump(tmp_path / "dump")
        make_cache(tmp_path / "cache")
        self.cache, self.dump = tmp_path / "cache", tmp_path / "dump"
        self.out = tmp_path / "out"

    def __call__(self, *extra, out="out"):
        """main() on the synthetic data: K in {8, 16}, 20 rows per centroid, batch 80 / 160, 3 epochs."""
        SK.main(["--activations_dir", str(self.dump), "--cache", str(self.cache), "--out_dir", str(self.tmp / out),
                 "--models", "bypass=/m/ae.pt", "--baselines", "base=/m/b.npz,base_tokmean=/m/bt.npz",
                 "--saes", "mat10k=matched:/m/mat.pt,gs16k=scope:16k", "--ks", "8,16", "--rows_per_centroid", "20",
                 "--batch_per_k2000", "20000", "--epochs", "3", "--density_pool", "128", "--unit_ns", "3,6,0",
                 "--ref_k", "8", "--controls_dir", "", *extra])
        return self.tmp / out

    def rows(self, K):
        """Independent re-derivation of one fit's training rows (the sae_controls_fit.py call)."""
        n_train = int(N_DUMP * 0.95)
        idx = np.sort(np.random.RandomState(42).choice(n_train, 20 * K, replace=False))
        return self.H[idx], self.tok[idx].astype(np.int64)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)         # stub tensors live on the CPU
    stub = Stub()
    fake = types.ModuleType("probe_chance_corrected")
    fake.load_arms = lambda models, baselines, saes, dev, sae_kmeans="": (dict(stub.arms), dict(stub.saes))
    monkeypatch.setitem(sys.modules, "probe_chance_corrected", fake)
    return Env(tmp_path, stub)


REAL_FIT = SK.balanced_fit


def load_all(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def count_fits(monkeypatch, fail_on=None):
    """Wrap the real balanced_fit: returns the list of (K, n_rows) of every call; the call number `fail_on` raises."""
    calls, real = [], REAL_FIT

    def counting(X, K, **kw):
        calls.append((K, len(X)))
        if fail_on is not None and len(calls) == fail_on:
            raise RuntimeError("killed")
        return real(X, K, **kw)
    monkeypatch.setattr(SK, "balanced_fit", counting)
    return calls


# ---------------------------------------------------------------- main() end to end
def test_main_end_to_end_matches_brute_force(env, monkeypatch):
    seen, real = [], REAL_FIT

    def spy(X, K, **kw):
        seen.append((K, X.clone(), kw))
        return real(X, K, **kw)
    monkeypatch.setattr(SK, "balanced_fit", spy)
    out = env()

    # --- the fits: K-scaled rows / batch, the same rows and embedding as an independent derivation
    assert [(K, len(X)) for K, X, _ in seen] == [(8, 160)] * 5 + [(16, 320)] * 5
    for (K, X, kw), space in zip(seen, SPACES * 2):
        assert (kw["seed"], kw["epochs"], kw["density_pool"]) == (42, 3, 128)
        assert kw["batch_size"] == {8: 80, 16: 160}[K]                      # batch = 20000 * K / 2000
        H, tok = env.rows(K)
        np.testing.assert_allclose(X.numpy(), env.stub.coords(H, tok)[space], rtol=1e-5, atol=1e-5)
        z = np.load(out / "codebooks" / f"km{K}_{space}.npz")
        assert z["centroids"].shape == (K, DIMS[space]) and z["centroids"].dtype == np.float32
        np.testing.assert_allclose(z["centroids"], real(X, K, **kw).numpy(), rtol=1e-4, atol=1e-4)   # deterministic fit
        assert (int(z["K"]), int(z["n_rows"]), int(z["n_train"]), str(z["space"])) == (K, 20 * K, 2850, space)
        d = torch.cdist(X, torch.from_numpy(z["centroids"])).argmin(1)
        st = SK.count_stats(np.bincount(d.numpy(), minlength=K))               # stats on the training rows
        for k in ("live", "top10_share", "eff_k"):
            assert np.isclose(float(z[k]), st[k])
    cb = sorted(p.name for p in (out / "codebooks").glob("*.npz"))
    assert cb == sorted([f"km{K}_{s}.npz" for K in (8, 16) for s in SPACES] + [f"unit_usage_{s}.npz" for s in SPACES])

    # --- the unit usage on the K=8 rows (160 rows), vs an independent bincount of the argmax
    H8, tok8 = env.rows(8)
    c8 = env.stub.coords(H8, tok8)
    usage = {}
    for s in SPACES:
        z = np.load(out / "codebooks" / f"unit_usage_{s}.npz")
        lab = brute_labels(c8[s], np.zeros(DIMS[s]), 0, s in NONNEG)
        usage[s] = np.bincount(lab[lab >= 0], minlength=DIMS[s])
        assert np.array_equal(z["usage"], usage[s]) and z["usage"].dtype == np.int64
        assert np.array_equal(z["order"], np.argsort(-usage[s], kind="stable"))
        assert int(z["n_rows"]) == 160 and int(z["n_inactive"]) == int((lab < 0).sum()) and bool(z["nonneg"]) == (s in NONNEG)
    assert any(int(np.load(out / "codebooks" / f"unit_usage_{s}.npz")["n_inactive"]) > 0 for s in NONNEG)   # -1 path is hit

    # --- assign.npz: keys, dtypes, and every label array against brute force on all cache rows
    z = np.load(out / "assign.npz", allow_pickle=False)
    want = [f"km{K}_{s}" for s in SPACES for K in (8, 16)] + [f"unit{n}_{s}" for s in SPACES for n in ("3", "6", "all")]
    assert sorted(z.files[:-5]) == sorted(want) and z.files[-5:] == META_KEYS
    rows, rtok = np.load(env.cache / "rows.npy"), np.load(env.cache / "rows_tok.npy").astype(np.int64)
    coords = env.stub.coords(rows, rtok)
    for s in SPACES:
        V = coords[s].astype(np.float64)
        for K in (8, 16):
            C = np.load(out / "codebooks" / f"km{K}_{s}.npz")["centroids"].astype(np.float64)
            d = ((V[:, None, :] - C[None]) ** 2).sum(-1)
            assert (np.sort(d, 1)[:, 1] - np.sort(d, 1)[:, 0]).min() > 1e-6        # no near-ties
            got = z[f"km{K}_{s}"]
            assert got.dtype == np.int32 and got.shape == (60,) and np.array_equal(got, d.argmin(1))
        for N, name in ((3, "3"), (6, "6"), (0, "all")):
            got = z[f"unit{name}_{s}"]
            assert got.dtype == np.int32
            assert np.array_equal(got, brute_labels(coords[s], usage[s], N, s in NONNEG)), (s, N)
            if s not in NONNEG:
                assert (got >= 0).all()                                    # signed spaces have no 'inactive' rows
    assert (z["unit3_mat10k"] == -1).any() and (z["unitall_mat10k"] == -1).sum() <= (z["unit3_mat10k"] == -1).sum()

    # --- meta.json
    meta = json.loads((out / "meta.json").read_text())
    assert meta["specs"] == {"base": "/m/b.npz", "base_tokmean": "/m/bt.npz", "bypass": "/m/ae.pt",
                             "mat10k": "matched:/m/mat.pt", "gs16k": "scope:16k"}
    assert meta["dims"] == DIMS and meta["device"] == "cpu" and meta["nonneg_spaces"] == ["mat10k", "gs16k"]
    idx8 = np.sort(np.random.RandomState(42).choice(2850, 160, replace=False))
    assert meta["schedule"]["8"] == dict(K=8, n_rows=160, batch_size=80, steps_per_epoch=2, total_steps=6,
                                         rows_sha1=SK.fingerprint(idx8), first_row=int(idx8[0]), last_row=int(idx8[-1]))
    assert meta["schedule"]["16"]["batch_size"] == 160 and meta["schedule"]["16"]["n_rows"] == 320
    assert set(meta["fits"]) == {f"km{K}_{s}" for K in (8, 16) for s in SPACES}
    assert all(f["resumed"] is False and f["seconds"] >= f["fit_seconds"] >= 0 and f["gpu_peak_gb"] == -1.0
               for f in meta["fits"].values())
    n_tok, row_chunk, row_pos = np.load(env.cache / "chunks.npz")["n_tok"], np.load(env.cache / "rows_chunk.npy"), np.load(env.cache / "rows_pos.npy")
    sampled = row_pos != n_tok[row_chunk] - 1
    assert meta["cache"] == dict(n_rows=60, n_chunks=12, n_sampled_rows=48) and sampled.sum() == 48
    assert set(meta["labels"]) == set(want)
    for key, rec in meta["labels"].items():                                # live / effective K / unassigned on SAMPLED rows
        lab = z[key][sampled]
        ok = lab >= 0
        cnt = np.bincount(lab[ok])
        p = cnt[cnt > 0] / ok.sum()
        assert rec["n_rows"] == 48 and rec["live"] == int((cnt > 0).sum())
        assert np.isclose(rec["eff_k"], np.exp(-(p * np.log(p)).sum())) and np.isclose(rec["unassigned"], 1 - ok.mean())
    assert meta["total_seconds"] > 0 and set(meta["unit_usage"]) == set(SPACES)
    assert meta["plan_gpu_gb"]["km16_gs16k"] == round(SK.fit_memory_gb(320, 40, 16, 160, 128), 1)   # the logged plan
    assert not list(out.rglob("*.tmp*"))


def test_assign_npz_row_metadata_is_identical_to_atlas_token_detail(env, monkeypatch, tmp_path):
    import atlas_token_detail as ATD

    probe = sys.modules["probe_chance_corrected"]
    real_load = probe.load_arms
    C0 = torch.randn(5, D, generator=torch.Generator().manual_seed(3))
    probe.load_arms = lambda models, baselines, saes, dev, sae_kmeans="": (
        {"base": ("km", (C0, env.stub.mean, env.stub.std, None, None))}, {})
    monkeypatch.setattr(sys, "argv", ["atlas_token_detail.py", "--cache", str(env.cache), "--out_dir", str(tmp_path / "td"),
                                      "--draws", str(tmp_path / "no_such_draws.json")])
    with pytest.raises(FileNotFoundError):                                 # stops right after it saved assign.npz
        ATD.main()
    probe.load_arms = real_load
    theirs = np.load(tmp_path / "td" / "assign.npz", allow_pickle=False)
    mine = np.load(env() / "assign.npz", allow_pickle=False)
    assert theirs.files == ["base", *META_KEYS] and mine.files[-5:] == META_KEYS
    for k in META_KEYS:
        assert mine[k].dtype == theirs[k].dtype and mine[k].shape == theirs[k].shape and np.array_equal(mine[k], theirs[k]), k
    assert theirs["base"].dtype == np.int32
    assert all(mine[k].dtype == np.int32 and mine[k].shape == (60,) for k in mine.files[:-5])
    assert [mine[k].dtype for k in META_KEYS] == [np.int32, np.int32, np.int32, np.int64, np.int32]


def test_resume_keeps_finished_codebooks_after_a_kill_and_skips_them(env, monkeypatch):
    calls = count_fits(monkeypatch, fail_on=3)
    with pytest.raises(RuntimeError, match="killed"):
        env()
    cb = env.out / "codebooks"
    assert sorted(p.name for p in cb.glob("km*.npz")) == ["km8_base.npz", "km8_base_tokmean.npz"]   # two finished fits kept
    assert not list(env.out.rglob("*.tmp*")) and not (env.out / "assign.npz").exists()
    meta = json.loads((env.out / "meta.json").read_text())                 # written incrementally
    assert set(meta["fits"]) == {"km8_base", "km8_base_tokmean"} and set(meta["unit_usage"]) == set(SPACES)
    kept = {p.name: p.read_bytes() for p in cb.glob("km*.npz")}

    calls = count_fits(monkeypatch)
    env()
    assert len(calls) == 8                                                # 10 fits - the 2 that were already on disk
    assert {p.name: p.read_bytes() for p in cb.glob("km8_base*.npz")} == kept    # not refitted, not rewritten
    meta = json.loads((env.out / "meta.json").read_text())
    assert meta["fits"]["km8_base"]["resumed"] is True and meta["fits"]["km16_gs16k"]["resumed"] is False
    assert (env.out / "assign.npz").exists()
    first = load_all(env.out / "assign.npz")

    monkeypatch.setattr(SK, "balanced_fit", lambda *a, **k: pytest.fail("a valid codebook was refitted"))
    env()                                                                  # everything on disk: nothing to fit
    again = load_all(env.out / "assign.npz")
    assert list(again) == list(first) and all(np.array_equal(again[k], first[k]) for k in first)
    assert json.loads((env.out / "meta.json").read_text())["fits"]["km16_gs16k"]["resumed"] is True


def test_resume_refuses_a_codebook_of_another_fit_unless_refit(env, monkeypatch):
    env()
    cb = env.out / "codebooks" / "km8_base.npz"
    old = cb.read_bytes()
    with pytest.raises(SystemExit) as e:
        env("--epochs", "4")
    assert "epochs" in str(e.value) and "--refit" in str(e.value) and cb.read_bytes() == old
    with pytest.raises(SystemExit) as e:
        env("--seed", "7")
    assert "seed" in str(e.value) or "rows_sha1" in str(e.value)
    calls = count_fits(monkeypatch)
    env("--epochs", "4", "--refit")
    assert len(calls) == 10 and int(np.load(cb)["epochs"]) == 4 and cb.read_bytes() != old


def test_a_corrupt_codebook_is_an_error_not_a_silent_refit(env):
    env()
    cb = env.out / "codebooks" / "km16_bypass.npz"
    cb.write_bytes(cb.read_bytes()[:100])                                   # a truncated file
    with pytest.raises(SystemExit, match="unreadable"):
        env()


def test_same_arguments_give_identical_codebooks_and_labels(env):
    a, b = env(out="a"), env(out="b")
    za, zb = np.load(a / "assign.npz"), np.load(b / "assign.npz")
    assert za.files == zb.files and all(np.array_equal(za[k], zb[k]) for k in za.files)
    for p in sorted((a / "codebooks").glob("km*.npz")):
        assert np.array_equal(np.load(p)["centroids"], np.load(b / "codebooks" / p.name)["centroids"])


def test_chunking_does_not_change_the_result(env, monkeypatch):
    a = env(out="a")
    monkeypatch.setattr(SK, "chunk_rows", lambda n, cap=4096: 7)
    monkeypatch.setattr(SK, "LABEL_CHUNK", 5)
    monkeypatch.setattr(SK, "DUMP_READ_CHUNK", 11)
    b = env(out="b")
    za, zb = np.load(a / "assign.npz"), np.load(b / "assign.npz")
    assert za.files == zb.files and all(np.array_equal(za[k], zb[k]) for k in za.files)
    for p in sorted((a / "codebooks").glob("*.npz")):
        for k in ("centroids", "usage"):
            if k in np.load(p).files:
                np.testing.assert_allclose(np.load(p)[k], np.load(b / "codebooks" / p.name)[k], rtol=1e-4, atol=1e-4)


def test_agreement_with_the_existing_sae_code_kmeans(env, tmp_path):
    env()
    assert json.loads((env.out / "meta.json").read_text())["agreement"] == {}     # no controls dir: skipped
    ctl = tmp_path / "controls"
    ctl.mkdir()
    cb = lambda s: np.load(env.out / "codebooks" / f"km8_{s}.npz")["centroids"]
    np.savez(ctl / "kmeans_mat10k.npz", centroids=cb("mat10k"), spec="matched:/m/mat.pt", n_rows=160, seed=42,
             n_clusters=8)                                                 # the very same fit
    np.savez(ctl / "kmeans_gs16k.npz", centroids=cb("gs16k")[np.roll(np.arange(8), 1)], spec="scope:16k")   # same partition, ids shifted
    np.savez(ctl / "kmeans_bypass.npz", centroids=np.zeros((8, 3), np.float32))           # not this space's dimension
    env("--controls_dir", str(ctl))
    agr = json.loads((env.out / "meta.json").read_text())["agreement"]
    assert set(agr) == {"mat10k", "gs16k", "bypass"}
    m = agr["mat10k"]
    assert m["label_agreement"] == 1.0 and np.isclose(m["nmi"], 1.0) and m["identical_centroids"] is True
    assert (m["n_rows"], m["K_new"], m["K_old"], m["old_n_rows"], m["old_seed"]) == (160, 8, 8, 160, 42)
    g = agr["gs16k"]
    assert g["label_agreement"] == 0.0 and np.isclose(g["nmi"], 1.0) and g["identical_centroids"] is False
    assert "dim" in agr["bypass"]["error"] and "label_agreement" not in agr["bypass"]


def test_unit_readout_of_n_at_least_the_dimension_is_the_every_unit_readout(env):
    z = load_all(env("--ks", "8", "--unit_ns", "100,5,0") / "assign.npz")
    for s in SPACES:
        assert np.array_equal(z[f"unit100_{s}"], z[f"unitall_{s}"])           # N >= dim (here also for gs16k: 40 < 100)
    assert not np.array_equal(z["unit5_mat10k"], z["unitall_mat10k"])          # a real restriction does change labels
    assert set(z) == {f"km8_{s}" for s in SPACES} | {f"unit{n}_{s}" for s in SPACES for n in ("100", "5", "all")} | set(META_KEYS)


def test_main_rejects_bad_arguments_before_fitting_anything(env):
    for extra in (("--spaces", "base,not_a_space"), ("--spaces", ""), ("--ks", "8,four"),
                  ("--ks", "4000", "--batch_per_k2000", "1000"), ("--rows_per_centroid", "1000"),
                  ("--unit_ns", "-3"), ("--epochs", "0")):
        with pytest.raises(SystemExit):
            env(*extra)
        assert not (env.out / "assign.npz").exists() and not list(env.out.glob("codebooks/km*"))


def test_dump_and_cache_dimension_mismatch_is_an_error(env, tmp_path):
    np.save(env.dump / "layer_33.npy", np.zeros((N_DUMP, D + 1), np.float32))
    with pytest.raises(SystemExit, match="dims"):
        env()


def test_sbatch_follows_the_spec_and_only_uses_flags_the_script_knows():
    import re
    import subprocess

    path = _ROOT / "scripts" / "argo" / "atlas_symmetric_kmeans.sbatch"
    txt, knn = path.read_text(), (_ROOT / "scripts" / "argo" / "atlas_knn_graph.sbatch").read_text()
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    for line in ("#SBATCH --job-name=atlas-symkm", "#SBATCH --gres=gpu:h100:1", "#SBATCH --mem=240G",
                 "#SBATCH --time=0-08:00:00"):
        assert line in txt.splitlines()
    for key in ("--account", "--partition"):                               # the same as the sibling Argo job
        assert re.search(rf"#SBATCH {key}=\S+", txt).group(0) == re.search(rf"#SBATCH {key}=\S+", knn).group(0)
    for line in ("ARM=g4b_l33_d5120; LAYER=33; K=2000", "OUT=results/delta/$ARM/evals/sae_suite/atlas_symmetric",
                 "TB=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt",
                 "KM=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5.npz",
                 "KMT=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz",
                 "MAT=$TRAIN/matryoshka_10240/last.pt", "ATLAS_CACHE=/scratch/dogar/geoae/atlas_g4b_l33",
                 '--models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT"',
                 '--saes "mat10k=matched:$MAT,gs16k=scope:16k"', "logs/atlas_symmetric_kmeans.log"):
        assert line in txt, line
    call = txt[txt.index('"$PY" -u eval_out/atlas_symmetric_kmeans.py'):]
    assert '"$@"' in call and 'exit "${PIPESTATUS[0]}"' in call              # extra arguments pass through
    known = {o for a in SK.build_parser()._actions for o in a.option_strings}
    flags = set(re.findall(r"(--[a-z_0-9]+)", call[:call.index('"$@"')]))
    assert {"--cache", "--rows_dir", "--activations_dir", "--models", "--baselines", "--saes", "--out_dir"} <= flags <= known
    for guarded in ("$TB", "$KM", "$KMT", "$MAT", "$CTL/kmeans_mat10k.npz", "$CTL/kmeans_gs16k.npz",
                    "$ACT_SRC/layer_$LAYER.npy", "$ACT_SRC/rows_tok.npy", "$ATLAS_CACHE/rows.npy", "$ATLAS_CACHE/chunks.npz"):
        assert f'"{guarded}"' in txt[:txt.index("# ---- stage")], guarded       # every input file is guarded before staging


def test_the_k2000_style_fit_reproduces_sae_controls_fit_exactly(tmp_path, monkeypatch):
    """Same procedure as eval_out/sae_controls_fit.py: run that script and this one with matching settings (one batch of 4096
    rows, K=8, 3 epochs) on real SAE adapters (Gemma-Scope-format JumpReLU + a token-bypass matched SAE, CPU): identical
    centroids, and the agreement diagnostic reads 1.0. The rows (first 95% of the dump), the embedding chunks and the
    balanced_fit arguments are what makes this hold."""
    import sae_controls_fit as SCF
    from safetensors.torch import save_file
    from test_sae_space import V as SV, matched_parts, scope_cfg, scope_params
    from geoae.interp.sae_space import load_sae_space

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    scope_dir = tmp_path / "scope"
    scope_dir.mkdir()
    save_file(scope_params(), str(scope_dir / "params.safetensors"))
    (scope_dir / "config.json").write_text(json.dumps(scope_cfg()))
    core, tf = matched_parts()
    torch.save({"recipe": {"method": "batch_topk", "activation_dim": D, "width": 40, "k": 3},
                "model_state": core.state_dict(),
                "transform_state": {"norm_mean": tf.norm_mean, "norm_std": tf.norm_std, "tb_table": tf.tb_table,
                                    "tb_index": tf.tb_index}, "provenance": {}}, tmp_path / "last.pt")
    saes_arg = f"gs=scope:{scope_dir},mat=matched:{tmp_path / 'last.pt'}"
    make_dump(tmp_path / "dump", n=5000, vocab=SV)                          # train split 4750 >= 4096 rows
    make_cache(tmp_path / "cache", vocab=SV)

    ctl = tmp_path / "controls"
    monkeypatch.setattr(sys, "argv", ["sae_controls_fit.py", "--saes", saes_arg, "--activations_dir", str(tmp_path / "dump"),
                                      "--layer", "33", "--n_sample", "4096", "--n_clusters", "8", "--epochs", "3",
                                      "--density_pool", "128", "--out_dir", str(ctl)])
    SCF.main()

    fake = types.ModuleType("probe_chance_corrected")
    fake.load_arms = lambda models, baselines, saes, dev, sae_kmeans="": (
        {}, {n: load_sae_space(sp, dev, n) for n, sp in SK.parse_saes(saes).items()})
    monkeypatch.setitem(sys.modules, "probe_chance_corrected", fake)
    out = tmp_path / "out"
    SK.main(["--activations_dir", str(tmp_path / "dump"), "--cache", str(tmp_path / "cache"), "--out_dir", str(out),
             "--saes", saes_arg, "--spaces", "gs,mat", "--ks", "8", "--rows_per_centroid", "512",
             "--batch_per_k2000", "1024000", "--epochs", "3", "--density_pool", "128", "--unit_ns", "0", "--ref_k", "8",
             "--controls_dir", str(ctl)])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["schedule"]["8"]["n_rows"] == 4096 and meta["schedule"]["8"]["batch_size"] == 4096
    for name in ("gs", "mat"):
        old = np.load(ctl / f"kmeans_{name}.npz")
        new = np.load(out / "codebooks" / f"km8_{name}.npz")
        assert int(old["n_rows"]) == int(new["n_rows"]) == 4096 and int(old["seed"]) == int(new["seed"]) == 42
        np.testing.assert_allclose(new["centroids"], old["centroids"], rtol=1e-6, atol=1e-6)
        agr = meta["agreement"][name]
        assert agr["label_agreement"] == 1.0 and np.isclose(agr["nmi"], 1.0) and agr["identical_centroids"] is True
        assert agr["old_n_rows"] == 4096 and agr["old_seed"] == 42
