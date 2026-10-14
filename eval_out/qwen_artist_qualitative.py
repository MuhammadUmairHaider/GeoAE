"""Recover the exact DBpedia probe rows and compare native cluster memberships.

CPU-only exploratory audit. No LM inference, training, or changes to prior results.
Run with OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 scripts/delta/py -u
    eval_out/qwen_artist_qualitative.py
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import homogeneity_completeness_v_measure
from transformers import AutoTokenizer

from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.token_bias import TokenBiasLookup

OUT = Path("results/qwen_artist_qualitative_20261009")
CKPT = Path("checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt")
BASE = OUT / "source/plain_tokmean.npz"
PRIOR = Path("results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table/probe_dense_subsets.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entropy(counts):
    p = np.asarray(list(counts), dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def fixed_point(assignments, initial, minimum=10):
    keep = initial.copy()
    while True:
        new = keep.copy()
        for c in assignments.values():
            new &= np.bincount(c[keep], minlength=int(c.max()) + 1)[c] >= minimum
        if np.array_equal(new, keep):
            return keep
        keep = new


def main():
    torch.set_num_threads(2)
    cache = np.load(OUT / "source/topic14.npz", allow_pickle=True)
    H = cache["H_last"].astype(np.float32)
    y = cache["label"].astype(int)
    ids = cache["last_token_id"].astype(np.int64)
    raw = next(Path("cache/huggingface/datasets/fancyzhx___dbpedia_14").rglob("dbpedia_14-train.arrow"))
    ds = Dataset.from_file(str(raw))
    names = ds.features["label"].names
    order = np.random.default_rng(0).permutation(len(ds))
    source_rows = []
    for idx in order:
        ex = ds[int(idx)]
        if ex.get("content"):
            source_rows.append((int(idx), ex))
        if len(source_rows) == len(y):
            break
    np.testing.assert_array_equal([ex["label"] for _, ex in source_rows], y)
    print("Matched all source labels", len(y), flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", local_files_only=True)
    rows = []
    for i, (idx, ex) in enumerate(source_rows):
        tokens = tok.encode(ex["content"], add_special_tokens=False)
        used = tokens[:192]
        assert used[-1] == ids[i], (i, used[-1], ids[i])
        rows.append(dict(row=i, source_index=idx, title=ex["title"], label=int(y[i]),
                         class_name=names[y[i]], text=ex["content"],
                         model_text=tok.decode(used), last_token_id=int(ids[i]),
                         last_token=tok.decode([int(ids[i])]), token_count=len(tokens),
                         truncated=len(tokens) > 192))
    print("Matched all final token IDs", flush=True)
    ae, mean, std, _ = load_ae_checkpoint(CKPT, "cpu", allow_token_bias=True)
    C, mu, sd, *_ = load_baseline_kmeans(BASE, torch.device("cpu"), allow_token_bias=True)
    bm = np.load(BASE, allow_pickle=True)
    tb = TokenBiasLookup(str(bm["token_bias"]), "cpu")
    assignments = {"tok_plain": [], "geoae": []}
    distances = {"tok_plain": [], "geoae": []}
    margins = {"tok_plain": [], "geoae": []}
    with torch.inference_mode():
        for s in range(0, len(H), 256):
            h, ti = torch.from_numpy(H[s:s+256]), torch.from_numpy(ids[s:s+256])
            spaces = {"tok_plain": ((h - mu) / sd - tb(ti), C),
                      "geoae": (ae.encode((h - mean) / std, ti), ae.centroids)}
            for arm, (v, centers) in spaces.items():
                dist = torch.cdist(v, centers)
                best, ix = dist.topk(2, largest=False)
                assignments[arm].extend(ix[:, 0].tolist())
                distances[arm].extend(best[:, 0].tolist())
                margins[arm].extend((best[:, 1] - best[:, 0]).tolist())
    assignments = {a: np.array(v) for a, v in assignments.items()}
    np.savez(OUT / "assignments.npz", **assignments, label=y, last_token_id=ids)
    exclusion = json.loads((OUT / "source/anchor_exclusion.json").read_text())
    ex = np.array(exclusion["topic14_anchor_indices"], dtype=int)
    initial = np.ones(len(y), dtype=bool)
    initial[ex] = False
    keep = fixed_point(assignments, initial)
    prior = json.loads(PRIOR.read_text())["rungs"]["topic14"]
    subset = "pair_fp10_tok_plain+geoae"
    assert keep.sum() == prior["common_rows"][subset], (keep.sum(), prior["common_rows"][subset])
    metrics = {}
    for arm, c in assignments.items():
        scores = homogeneity_completeness_v_measure(y[keep], c[keep])
        metrics[arm] = dict(zip(["homogeneity", "completeness", "nmi"], map(float, scores)))
        for k, val in metrics[arm].items():
            assert abs(val - prior["arms"][arm][subset][k]) < 1e-9, (arm, k, val)
    print("Exactly reproduced saved table", metrics, flush=True)
    for i, r in enumerate(rows):
        r.update(anchor_excluded=not bool(initial[i]), paired_retained=bool(keep[i]))
        for arm in assignments:
            r[arm] = int(assignments[arm][i])
            r[arm + "_distance"] = distances[arm][i]
            r[arm + "_margin"] = margins[arm][i]
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    cluster_profiles = {}
    for arm, c in assignments.items():
        cluster_profiles[arm] = {}
        for cl in sorted(set(c[keep])):
            members = np.flatnonzero(keep & (c == cl))
            counts = Counter(names[y[i]] for i in members)
            cluster_profiles[arm][int(cl)] = dict(n=len(members), classes=dict(counts.most_common()),
                row_ids=members.tolist(), last_tokens=dict(Counter(rows[i]["last_token"] for i in members).most_common()),
                truncated=sum(rows[i]["truncated"] for i in members))
    class_stats = {}
    for label, name in enumerate(names):
        target = keep & (y == label)
        arms = {}
        for arm, c in assignments.items():
            counts = Counter(c[target].tolist())
            arms[arm] = dict(clusters=len(counts), effective_clusters=math.exp(entropy(counts.values())),
                top_cluster_coverage=max(counts.values()) / target.sum(),
                weighted_class_purity=sum(n * cluster_profiles[arm][cl]["classes"].get(name, 0)
                    / cluster_profiles[arm][cl]["n"] for cl, n in counts.items()) / target.sum(),
                memberships={str(cl): n for cl, n in counts.most_common()})
        class_stats[name] = dict(n_full=int((y == label).sum()), n_after_exclusion=int((initial & (y == label)).sum()),
                                n_paired=int(target.sum()), arms=arms)
    summary = dict(model="Qwen/Qwen2.5-3B", layer=35, n_source=len(rows), n_after_exclusion=int(initial.sum()),
        n_paired=int(keep.sum()), global_metrics=metrics, classes=class_stats,
        source_mapping="DBpedia train Arrow, default_rng(0) permutation, first 3000 nonempty content rows; all labels and last-token IDs matched",
        table_reproduction="All three metrics in both arms match prior outputs within 1e-9",
        hashes={str(p): sha256(p) for p in [CKPT, BASE, OUT / "source/topic14.npz", raw, PRIOR,
                                           OUT / "source/anchor_exclusion.json", Path(__file__)]})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "cluster_profiles.json").write_text(json.dumps(cluster_profiles, indent=2, ensure_ascii=False))
    artist = [r for r in rows if r["class_name"].lower() == "artist"]
    (OUT / "artist_blind.json").write_text(json.dumps([{k: r[k] for k in ["row", "title", "model_text"]}
        for r in artist], indent=2, ensure_ascii=False))
    for name in names:
        print(name, class_stats[name], flush=True)


if __name__ == "__main__":
    main()
