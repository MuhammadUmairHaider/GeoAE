r"""k-nearest-neighbour graph over a fixed sample of Atlas cache rows, one graph per representation space, with every
row's neighbours taken from OTHER DOCUMENTS only.

Why. kNN label agreement with the own chunk excluded (eval_out/atlas_token_detail.py) still lets a row's neighbours be
sentences of the same document; excluding the document leaves cross-document neighbourhoods only. This script builds
that graph once per space and saves the neighbour indices; per-concept statistics (which concepts' rows are each
other's cross-document neighbours, in which space) are computed CPU-side from the saved indices, so no label is read
here.

Sample: for every chunk, `--per_chunk` of its NON-FINAL rows (the final-token row is punctuation) drawn uniformly
without replacement with a fixed seed: 3 x 100k chunks = 300,000 rows for the Gemma 3 4B layer-33 cache. Every space
embeds the same rows (atlas_token_detail.embed: base (h - mean)/std, base_tokmean minus b[tok], bypass z, SAE codes), so
the graphs are comparable arm to arm. Neighbours: the k nearest rows by squared Euclidean distance, blocked on the GPU
in true fp32 (TF32 off), rows of the query's own document (hence the row itself) excluded.

    python -u eval_out/atlas_knn_graph.py --cache C --rows_dir SHM --models bypass=<ckpt> \
        --baselines base=<npz>,base_tokmean=<npz> --saes "mat10k=matched:<last.pt>,bt10k=matched:<last.pt>" \
        --sae_kmeans "mat10k_km=matched:<last.pt>@<kmeans_mat10k.npz>" --out_dir OUT
Writes OUT/knn_graph.npz and OUT/knn_graph_meta.json, both rewritten after every arm (a killed job keeps the arms it
finished):
  rows (R,) int64   cache row index of each sample row, ascending      chunk (R,) int32   chunk of each row
  doc (R,) int64    document of each row                                tok (R,) int32     current token id
  nbr_<arm> (R, k) int32   POSITIONS IN THE SAMPLE (not cache rows) of the k nearest rows from other documents,
                           nearest first; cache row of neighbour j of sample row i is rows[nbr_<arm>[i, j]]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geoae.interp import atlas_siblings as AS  # noqa: E402
from atlas_token_detail import embed  # noqa: E402


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sample_rows(row_chunk, row_pos, n_tok, per_chunk=3, seed=0):
    """Sorted int64 cache-row indices: for every chunk, min(per_chunk, #non-final rows) of its non-final rows, drawn
    uniformly without replacement. A row is final when its token position is the chunk's last (n_tok - 1). Vectorised:
    one iid uniform key per non-final row, rows ordered by (chunk, key), the first `per_chunk` of each chunk kept."""
    row_chunk = np.asarray(row_chunk, dtype=np.int64)
    last_pos = np.asarray(n_tok, dtype=np.int64) - 1
    cand = np.flatnonzero(np.asarray(row_pos) != last_pos[row_chunk])           # non-final rows, ascending
    ch = row_chunk[cand]
    key = np.random.default_rng(seed).random(len(cand))
    order = np.lexsort((key, ch))                                               # chunk first, then the random key
    ch_s = ch[order]
    start = np.r_[0, np.flatnonzero(ch_s[1:] != ch_s[:-1]) + 1]                 # first slot of each chunk's run
    rank = np.arange(len(ch_s)) - np.repeat(start, np.diff(np.r_[start, len(ch_s)]))   # 0 = smallest key in its chunk
    return np.sort(cand[order[rank < per_chunk]]).astype(np.int64)


def knn_exclude_group(V, group, k=10, block=1024):
    """The k nearest rows of every row of V (squared Euclidean), never a row with the same `group` value (which also
    excludes the row itself). V (R, d) float tensor, group (R,) integer tensor on the same device. Returns
    (idx int64 (R, k) nearest first, dist (R, k) squared distances, clamped at 0 against fp32 cancellation). Blocked
    over query rows. A row with fewer than k candidates outside its group would get an infinite distance: asserted."""
    import torch
    n = len(V)
    assert len(group) == n, "one group value per row"
    with torch.no_grad():
        sq = (V * V).sum(1)
        idx = torch.empty(n, k, dtype=torch.long, device=V.device)
        dist = torch.empty(n, k, dtype=sq.dtype, device=V.device)
        for i in range(0, n, block):
            j = slice(i, min(i + block, n))
            d = sq[j, None] + sq[None, :] - 2 * (V[j] @ V.T)
            d.masked_fill_(group[j, None] == group[None, :], float("inf"))
            val, ix = d.topk(k, largest=False)
            idx[j], dist[j] = ix, val
    assert torch.isfinite(dist).all(), "a row has fewer than k candidates outside its group (or a non-finite coordinate)"
    return idx, dist.clamp_min(0)


def save_graph(out, graph, meta):
    """Rewrite knn_graph.npz (via a temp file, so a kill mid-write keeps the previous arms) and knn_graph_meta.json."""
    tmp = out / "knn_graph.tmp.npz"
    np.savez(tmp, **graph)
    os.replace(tmp, out / "knn_graph.npz")
    (out / "knn_graph_meta.json").write_text(json.dumps(meta, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rows_dir", default=None)
    ap.add_argument("--models", default="")
    ap.add_argument("--baselines", default="")
    ap.add_argument("--saes", default="")
    ap.add_argument("--sae_kmeans", default="")
    ap.add_argument("--space_arms", default="base,base_tokmean,bypass,mat10k",
                    help="arms a graph is built in (continuous coordinates)")
    ap.add_argument("--per_chunk", type=int, default=3, help="non-final rows sampled per chunk")
    ap.add_argument("--all_rows", action="store_true",
                    help="every cache row, final-token rows included (a --last_word cache: one row per chunk)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=1024, help="query rows per distance block")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalise every row's coordinates before the kNN (cosine neighbours); zero rows stay zero")
    ap.add_argument("--arm_suffix", default="", help="appended to every saved key: nbr_<arm><suffix>")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args(argv)
    if args.per_chunk < 1 or args.k < 1 or args.block < 1:
        ap.error("--per_chunk, --k and --block must be >= 1")

    import torch

    import probe_chance_corrected as PCC

    torch.backends.cuda.matmul.allow_tf32 = False       # neighbours in true fp32
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = AS.AtlasCache(args.cache, args.rows_dir)
    row_pos = np.load(Path(args.rows_dir or args.cache) / "rows_pos.npy", mmap_mode="r")
    log(f"cache {cache.n_rows:,} rows, {cache.n_chunks:,} chunks, d={cache.rows.shape[1]}")

    arms, saes = PCC.load_arms(args.models, args.baselines, args.saes, dev, sae_kmeans=args.sae_kmeans)
    log(f"arms {list(arms)}  saes {list(saes)}")
    space = [a for a in args.space_arms.split(",") if a]
    missing = [a for a in space if a not in arms and a not in saes]
    if not space:
        raise SystemExit("--space_arms is empty")
    if missing:
        raise SystemExit(f"--space_arms {missing} are neither loaded arms {list(arms)} nor SAEs {list(saes)}")

    # ---- the fixed sample: per_chunk non-final rows per chunk
    chunk_all = np.asarray(cache.row_chunk, dtype=np.int64)
    if args.all_rows:
        rows = np.arange(cache.n_rows, dtype=np.int64)
    else:
        rows = sample_rows(chunk_all, row_pos, cache.n_tok, args.per_chunk, args.seed)
    R = len(rows)
    if R <= args.k:
        raise SystemExit(f"only {R} sampled rows for k={args.k}")
    chunk = chunk_all[rows]
    doc = cache.doc[chunk]
    ids = np.asarray(cache.tok[rows], dtype=np.int64)
    H = np.asarray(cache.rows[rows])
    log(f"sample: {R:,} {'(all)' if args.all_rows else 'non-final'} rows from {len(np.unique(chunk)):,} chunks / {len(np.unique(doc)):,} documents "
        f"(per_chunk {args.per_chunk}, seed {args.seed}); H {H.shape} {H.dtype}")
    doc_t = torch.from_numpy(doc).to(dev)

    graph = dict(rows=rows, chunk=chunk.astype(np.int32), doc=doc, tok=ids.astype(np.int32))
    meta = dict(args=vars(args), n_rows=R, n_chunks=int(len(np.unique(chunk))), n_docs=int(len(np.unique(doc))),
                n_cache_rows=int(cache.n_rows), d_in=int(H.shape[1]), torch=torch.__version__,
                device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", arms={})
    for a in space:
        t0 = time.time()
        V = embed(H, arms[a] if a in arms else ("sae", saes[a]), dev, ids)
        if args.normalize:
            V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
        t1 = time.time()
        log(f"  {a:<13} embedded: dim {V.shape[1]}  ({t1 - t0:.0f}s)")
        nbr, dist = knn_exclude_group(V, doc_t, args.k, args.block)
        assert 0 <= int(nbr.min()) and int(nbr.max()) < R, "neighbour index outside the sample"
        assert bool((doc_t[nbr] != doc_t[:, None]).all()), "a neighbour shares its query's document"
        graph[f"nbr_{a}{args.arm_suffix}"] = nbr.cpu().numpy().astype(np.int32)
        meta["arms"][a] = dict(dim=int(V.shape[1]), seconds=round(time.time() - t0, 1), embed_seconds=round(t1 - t0, 1),
                               knn_seconds=round(time.time() - t1, 1), nn1_dist_median=float(dist[:, 0].median()),
                               nnk_dist_median=float(dist[:, -1].median()))
        del V, nbr, dist
        torch.cuda.empty_cache()
        save_graph(out, graph, meta)
        m = meta["arms"][a]
        log(f"  {a:<13} knn done in {m['knn_seconds']:.0f}s  (squared distance, median nearest {m['nn1_dist_median']:.4g}, "
            f"median k-th {m['nnk_dist_median']:.4g})  -> {out / 'knn_graph.npz'}")
    log(f"wrote {out / 'knn_graph.npz'} and knn_graph_meta.json: {len(meta['arms'])} arms, {R:,} rows, k={args.k}")


if __name__ == "__main__":
    main()
