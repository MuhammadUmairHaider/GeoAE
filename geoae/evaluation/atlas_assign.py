"""Cluster labels of every Atlas cache row for extra codebooks, made exactly as geoae/analysis/atlas_token_detail.py makes them
(probe_chance_corrected.load_arms + concept_probe.assign, 65,536-row blocks) and saved with the same spine arrays, so
geoae/analysis/atlas_concept_stats.py can merge the file with the original assign.npz (--assign a.npz,b.npz). An arm present in
both files must have identical labels, so passing an original arm again (e.g. base) checks the reproduction.

    python -u geoae/evaluation/atlas_assign.py --cache /scratch/dogar/geoae/atlas_g4b_l33 \
        --baselines "base=<balanced.npz>,plain=<plain.npz>,plain_tokmean=<plain_tokmean.npz>" --out <dir>/assign_extra.npz
SAE arms (--saes / --sae_kmeans, specs as geoae/baselines/sae_space.py) are labelled as atlas_token_detail.py labels them
(probe_chance_corrected.sae_assign): top-1 active feature, or nearest code-space centroid; -1 = no active feature.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geoae.analysis.atlas_siblings as AS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rows_dir", default=None)
    ap.add_argument("--models", default="")
    ap.add_argument("--baselines", default="")
    ap.add_argument("--saes", default="")
    ap.add_argument("--sae_kmeans", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import geoae.evaluation.chance_corrected_probe as PCC
    import geoae.analysis.concept_probe as CP
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = AS.AtlasCache(args.cache, args.rows_dir)
    row_pos = np.load(Path(args.rows_dir or args.cache) / "rows_pos.npy", mmap_mode="r")
    tok_all = np.asarray(cache.tok, dtype=np.int64)
    chunk_all = np.asarray(cache.row_chunk, dtype=np.int64)
    arms, saes = PCC.load_arms(args.models, args.baselines, args.saes, dev, sae_kmeans=args.sae_kmeans)
    lab = {}
    B = 65536
    for name in arms:
        t0 = time.time()
        parts = [CP.assign(np.asarray(cache.rows[i:i + B]), arms[name], dev, ids=tok_all[i:i + B])
                 for i in range(0, cache.n_rows, B)]
        lab[name] = np.concatenate(parts).astype(np.int32)
        print(f"labels {name}: {len(np.unique(lab[name])):,} distinct ({time.time() - t0:.0f}s)", flush=True)
    for name in saes:
        t0 = time.time()
        parts = [PCC.sae_assign(np.asarray(cache.rows[i:i + B]), saes[name], dev, tok_all[i:i + B])
                 for i in range(0, cache.n_rows, B)]
        lab[name] = np.concatenate(parts).astype(np.int32)
        print(f"labels {name}: {len(np.unique(lab[name])):,} distinct, {(lab[name] < 0).mean():.2%} rows with no active "
              f"feature (-1) ({time.time() - t0:.0f}s)", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **lab, row_chunk=chunk_all.astype(np.int32), row_pos=np.asarray(row_pos, dtype=np.int32),
                        row_tok=tok_all.astype(np.int32), chunk_doc=cache.doc.astype(np.int64),
                        chunk_ntok=np.asarray(cache.n_tok, dtype=np.int32))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
