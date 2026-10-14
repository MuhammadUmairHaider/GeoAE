"""eval_out/atlas_knn_graph.py on tiny synthetic data (CPU): the document-excluded kNN against a brute-force numpy
implementation, the per-chunk row sampler, and main() end to end on a synthetic cache with stub arms."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_p = Path(__file__).resolve().parents[1] / "eval_out" / "atlas_knn_graph.py"
sys.path.insert(0, str(_p.parent))
_spec = importlib.util.spec_from_file_location("atlas_knn_graph", _p)
AK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AK)


# ---------------------------------------------------------------- kNN with a group excluded
def toy(n=200, dim=8, n_groups=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32), rng.integers(0, n_groups, size=n)


def brute_force(V, group, k):
    """float64 reference straight from the coordinates: squared distances, same-group rows masked, k smallest.
    Returns (idx, dist of those, full masked matrix)."""
    V = np.asarray(V, dtype=np.float64)
    d = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    d[group[:, None] == group[None, :]] = np.inf
    idx = np.argsort(d, axis=1)[:, :k]
    return idx, np.take_along_axis(d, idx, 1), d


def test_knn_matches_brute_force():
    V, group = toy()
    k = 5
    ref_idx, ref_dist, d = brute_force(V, group, k)
    # no near-tie at the k-th boundary (far above the fp32 rounding of the expansion): the neighbour SET is unambiguous
    assert (np.sort(d, axis=1)[:, k] - ref_dist[:, -1]).min() > 1e-4
    idx, dist = AK.knn_exclude_group(torch.from_numpy(V), torch.from_numpy(group), k=k, block=37)   # 200 = 5 blocks + 15
    assert idx.dtype == torch.int64 and idx.shape == (200, k) and dist.shape == (200, k)
    idx, dist = idx.numpy(), dist.numpy()
    assert (np.sort(idx, axis=1) == np.sort(ref_idx, axis=1)).all()
    np.testing.assert_allclose(np.sort(dist, axis=1), np.sort(ref_dist, axis=1), rtol=1e-4, atol=1e-4)
    assert (np.diff(dist, axis=1) >= 0).all()                         # nearest first


@pytest.mark.parametrize("block", [1, 37, 64, 199, 200, 1000])
def test_knn_float64_exact_for_any_block(block):
    V, group = toy(seed=1)
    ref_idx, ref_dist, _ = brute_force(V, group, 5)
    idx, dist = AK.knn_exclude_group(torch.from_numpy(V.astype(np.float64)), torch.from_numpy(group), k=5, block=block)
    assert (idx.numpy() == ref_idx).all()                              # same neighbours in the same order
    np.testing.assert_allclose(dist.numpy(), ref_dist, rtol=1e-9, atol=1e-9)


def test_knn_never_returns_own_group_or_self():
    rng = np.random.default_rng(3)
    V = torch.from_numpy(rng.normal(size=(90, 4)).astype(np.float32))
    group = torch.from_numpy(np.repeat(np.arange(3), 30))             # three large groups: 60 candidates outside each
    idx, dist = AK.knn_exclude_group(V, group, k=7, block=16)
    assert (group[idx] != group[:, None]).all()
    assert (idx != torch.arange(90)[:, None]).all()
    assert idx.min() >= 0 and idx.max() < 90 and torch.isfinite(dist).all() and (dist >= 0).all()


def test_knn_duplicate_rows_have_zero_distance_not_negative():
    rng = np.random.default_rng(4)
    v = rng.normal(size=(1, 6)) * 1000                                # |v|^2 ~ 6e6: fp32 cancellation of ~1 in sq + sq - 2ab
    V = torch.from_numpy(np.concatenate([v, v, rng.normal(size=(10, 6))]).astype(np.float32))   # rows 0 and 1 coincide
    group = torch.arange(12)                                           # every row its own group: only self excluded
    idx, dist = AK.knn_exclude_group(V, group, k=3, block=5)
    assert idx[0, 0] == 1 and idx[1, 0] == 0
    assert (dist >= 0).all()                                           # clamped: rounding never reports a negative square
    assert (dist[:2, 0] < 1e-3 * dist[:2, 1]).all()                    # ~0 next to the other rows (~1e6)


def test_knn_asserts_when_a_row_lacks_k_candidates():
    V = torch.from_numpy(np.random.default_rng(5).normal(size=(12, 3)).astype(np.float32))
    group = torch.tensor([0] * 10 + [1, 2])                            # rows of the big group see only 2 candidates
    with pytest.raises(AssertionError):
        AK.knn_exclude_group(V, group, k=5, block=4)
    idx, dist = AK.knn_exclude_group(V, group, k=2, block=4)           # k = 2 is satisfiable for everyone
    assert torch.isfinite(dist).all() and (group[idx] != group[:, None]).all()


# ---------------------------------------------------------------- the row sampler
def toy_rows():
    """6 chunks of 10 tokens; stored positions per chunk (the final token is position 9)."""
    pos = [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8, 9], [4, 9], [9], [2, 3, 9], [5, 6, 7, 8]]
    row_chunk = np.concatenate([np.full(len(p), c) for c, p in enumerate(pos)]).astype(np.int32)
    row_pos = np.concatenate(pos).astype(np.int16)
    n_tok = np.full(len(pos), 10, dtype=np.int32)
    nonfinal = np.array([sum(q != 9 for q in p) for p in pos])        # [4, 5, 1, 0, 2, 4]
    return row_chunk, row_pos, n_tok, nonfinal


@pytest.mark.parametrize("per_chunk", [1, 2, 3, 100])
def test_sample_rows_counts_and_never_final(per_chunk):
    row_chunk, row_pos, n_tok, nonfinal = toy_rows()
    rows = AK.sample_rows(row_chunk, row_pos, n_tok, per_chunk, seed=0)
    assert rows.dtype == np.int64 and (np.diff(rows) > 0).all()        # sorted, no duplicates
    assert (row_pos[rows] != n_tok[row_chunk[rows]] - 1).all()         # never a final-token row
    got = np.bincount(row_chunk[rows], minlength=len(n_tok))
    assert got.tolist() == np.minimum(per_chunk, nonfinal).tolist()    # min(per_chunk, available) per chunk, 0 if none


def test_sample_rows_deterministic_per_seed():
    row_chunk, row_pos, n_tok, _ = toy_rows()
    a = AK.sample_rows(row_chunk, row_pos, n_tok, 3, seed=7)
    assert (a == AK.sample_rows(row_chunk, row_pos, n_tok, 3, seed=7)).all()
    assert not np.array_equal(a, AK.sample_rows(row_chunk, row_pos, n_tok, 3, seed=8))


def test_sample_rows_is_uniform_within_a_chunk():
    # one chunk, 6 non-final rows + the final one: every non-final row is kept with probability 3/6
    row_chunk = np.zeros(7, dtype=np.int32)
    row_pos = np.array([0, 1, 2, 3, 4, 5, 9], dtype=np.int16)
    n_tok = np.array([10], dtype=np.int32)
    hits = np.zeros(7)
    for seed in range(400):
        hits[AK.sample_rows(row_chunk, row_pos, n_tok, 3, seed=seed)] += 1
    assert hits[6] == 0
    assert (hits[:6] > 160).all() and (hits[:6] < 240).all()           # 200 +- 4 sigma, and the seeds are fixed


def test_sample_rows_no_candidates():
    rows = AK.sample_rows(np.zeros(1, dtype=np.int32), np.array([9], dtype=np.int16), np.array([10]), 3, seed=0)
    assert rows.dtype == np.int64 and len(rows) == 0


# ---------------------------------------------------------------- main() end to end, stub arms, synthetic cache
D, N_VOCAB = 6, 40


def make_cache(path, n_chunks=12, seed=0):
    """AtlasCache layout: 12 chunks (3 per document), each with 4 non-final sampled rows and the final-token row."""
    rng = np.random.default_rng(seed)
    n_tok = rng.integers(12, 30, size=n_chunks)
    pos, chunk = [], []
    for c in range(n_chunks):
        pos += [*np.sort(rng.choice(n_tok[c] - 1, size=4, replace=False)), n_tok[c] - 1]
        chunk += [c] * 5
    n = len(pos)
    path.mkdir(parents=True)
    np.save(path / "rows.npy", rng.normal(size=(n, D)).astype(np.float32))
    np.save(path / "rows_tok.npy", rng.integers(0, N_VOCAB, size=n).astype(np.int32))
    np.save(path / "rows_nxt.npy", rng.integers(0, N_VOCAB, size=n).astype(np.int32))
    np.save(path / "rows_chunk.npy", np.asarray(chunk, dtype=np.int32))
    np.save(path / "rows_pos.npy", np.asarray(pos, dtype=np.int16))
    ptr = np.zeros(n_chunks + 1, dtype=np.int64)
    np.savez(path / "chunks.npz", doc_int_id=(np.arange(n_chunks) // 3).astype(np.int32),
             chunk_id=np.arange(n_chunks, dtype=np.int32), n_tok=n_tok.astype(np.int32),
             row_start=np.arange(0, 5 * n_chunks + 1, 5, dtype=np.int64), label_ptr=ptr,
             label_ids=np.zeros(0, np.int32), tok_ptr=ptr, tok_flat=np.zeros(0, np.int32), fingerprint=np.asarray("test"))


class Stub:
    """Arms in the formats of atlas_token_detail.embed: two k-means baselines (one token-mean-subtracted) and an AE."""
    def __init__(self, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.mean, self.std = torch.randn(D, generator=g), torch.rand(D, generator=g) + 0.5
        self.table = torch.randn(N_VOCAB, D, generator=g)                 # b[tok] of the token-mean baseline
        self.W, self.bias = torch.randn(D, 9, generator=g), torch.randn(N_VOCAB, 9, generator=g)   # the AE and its bypass
        stub = self

        class AE:
            has_token_bias = True

            def encode(self, x, t):
                return x @ stub.W + stub.bias[t]

        self.arms = {"base": ("km", (None, self.mean, self.std, None, None)),
                     "base_tokmean": ("km", (None, self.mean, self.std, None, lambda t: self.table[t])),
                     "bypass": ("ae", (AE(), self.mean, self.std))}

    def coords(self, H, tok):
        x = (torch.from_numpy(H) - self.mean) / self.std
        t = torch.from_numpy(tok)
        return {"base": x, "base_tokmean": x - self.table[t], "bypass": x @ self.W + self.bias[t]}


@pytest.fixture
def run(tmp_path, monkeypatch):
    """run(space, *extra_cli) -> Stub: main() on the synthetic cache; run.cache / run.out are the directories."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)    # stub tensors live on the CPU
    stub = Stub()
    fake = types.ModuleType("probe_chance_corrected")
    fake.load_arms = lambda models, baselines, saes, dev, sae_kmeans="": (stub.arms, {})
    monkeypatch.setitem(sys.modules, "probe_chance_corrected", fake)
    make_cache(tmp_path / "cache")

    def go(space, *extra):
        AK.main(["--cache", str(tmp_path / "cache"), "--space_arms", space, "--per_chunk", "2", "--k", "3",
                 "--block", "7", "--seed", "1", "--out_dir", str(tmp_path / "out"), *extra])
        return stub
    go.cache, go.out = tmp_path / "cache", tmp_path / "out"
    return go


def test_main_graph_matches_brute_force(run):
    stub = run("base,base_tokmean,bypass")
    cache, out = run.cache, run.out
    z = np.load(out / "knn_graph.npz")
    assert set(z.files) == {"rows", "chunk", "doc", "tok", "nbr_base", "nbr_base_tokmean", "nbr_bypass"}
    row_chunk, row_pos = np.load(cache / "rows_chunk.npy"), np.load(cache / "rows_pos.npy")
    n_tok = np.load(cache / "chunks.npz")["n_tok"]
    rows = z["rows"]
    assert rows.dtype == np.int64 and len(rows) == 24                  # 12 chunks x 2 of their 4 non-final rows
    assert (rows == AK.sample_rows(row_chunk, row_pos, n_tok, 2, seed=1)).all()
    assert (row_pos[rows] != n_tok[row_chunk[rows]] - 1).all()
    assert (z["chunk"] == row_chunk[rows]).all() and (z["doc"] == row_chunk[rows] // 3).all()
    assert (z["tok"] == np.load(cache / "rows_tok.npy")[rows]).all()
    assert (z["chunk"].dtype, z["doc"].dtype, z["tok"].dtype) == (np.int32, np.int64, np.int32)
    H, tok = np.load(cache / "rows.npy")[rows], np.load(cache / "rows_tok.npy")[rows].astype(np.int64)
    for arm, V in stub.coords(H, tok).items():
        nbr = z[f"nbr_{arm}"]
        assert nbr.dtype == np.int32 and nbr.shape == (24, 3)
        assert (z["doc"][nbr] != z["doc"][:, None]).all()              # no neighbour from the query's own document
        ref_idx, _, _ = brute_force(V.numpy(), z["doc"], 3)
        assert (np.sort(nbr, axis=1) == np.sort(ref_idx, axis=1)).all(), arm
    meta = json.loads((out / "knn_graph_meta.json").read_text())
    assert meta["n_rows"] == 24 and meta["args"]["k"] == 3 and meta["args"]["per_chunk"] == 2
    assert {"torch", "device"} <= set(meta)
    assert list(meta["arms"]) == ["base", "base_tokmean", "bypass"]
    assert [meta["arms"][a]["dim"] for a in meta["arms"]] == [D, D, 9]
    assert all(meta["arms"][a]["seconds"] >= 0 for a in meta["arms"])


def test_main_keeps_finished_arms_when_killed(run, monkeypatch):
    real, calls = AK.embed, []

    def dies_on_second_arm(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("killed")
        return real(*a, **kw)
    monkeypatch.setattr(AK, "embed", dies_on_second_arm)
    with pytest.raises(RuntimeError):
        run("base,base_tokmean,bypass")
    z = np.load(run.out / "knn_graph.npz")
    assert set(z.files) == {"rows", "chunk", "doc", "tok", "nbr_base"}  # the arm finished before the kill survives
    assert list(json.loads((run.out / "knn_graph_meta.json").read_text())["arms"]) == ["base"]
    assert not (run.out / "knn_graph.tmp.npz").exists()


def test_main_rejects_unknown_space_arm_before_writing(run):
    with pytest.raises(SystemExit):
        run("base,not_an_arm")
    assert not (run.out / "knn_graph.npz").exists()


def test_main_rejects_empty_space_arms(run):
    with pytest.raises(SystemExit):
        run("")
    assert not (run.out / "knn_graph.npz").exists()
