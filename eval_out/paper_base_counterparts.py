"""Locate base counterparts of the GeoAE pairs in the saved removal experiment.

The discovery comparison keeps GeoAE membership FIXED across all spaces, uses
the same held-out 500k rows/global centering as paper_ranges_along_directions,
and matches memberships to native base k-means by precision/recall/F1.
The independent validation uses the removal cache: even documents fit each
space's OWN shared direction, odd documents evaluate its range separation.
No LM inference or new semantic labels are required.

    GEOAE_REMOTE=1 .venv/bin/python eval_out/paper_base_counterparts.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.closest_tokens import load_baseline_kmeans
from geoae.losses import pairwise_sq_dist
from geoae.token_bias import TokenBiasLookup

# BC_CKPT / BC_KM / BC_KMT / BC_REMOVAL / BC_RANGES override the K=2000 d6144 defaults (e.g. for the K=4000 arm).
CKPT = os.environ.get("BC_CKPT", "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt")
KM = os.environ.get("BC_KM", "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled.npz")
KMT = os.environ.get("BC_KMT", "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled_tokmean.npz")
REMOVAL = os.environ.get("BC_REMOVAL", "results/delta/d6144/evals/same_direction_removal.json")
RANGES = os.environ.get("BC_RANGES", "results/delta/d6144/evals/ranges_along_directions.json")
SPACE_NAMES = ("raw_residual", "residual", "token_centred", "geoae")


def empirical_auc(a, b):
    """Orientation-free empirical AUROC, with tied projections ranked equally."""
    ranks = rankdata(np.r_[a, b])
    auc = (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return float(max(auc, 1 - auc))


def projection_stats(a, b):
    va, vb = float(np.var(a)), float(np.var(b))
    return {"dprime": float(abs(np.mean(a) - np.mean(b)) / np.sqrt((va + vb) / 2 + 1e-12)),
            "empirical_auroc": empirical_auc(a, b), "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)), "sd_a": float(np.sqrt(va)), "sd_b": float(np.sqrt(vb))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="activations_sampled_10M")
    ap.add_argument("--cache", default="results/delta/d6144/evals/same_direction_rows.npz")
    ap.add_argument("--out", default="results/delta/d6144/evals/base_counterparts.json")
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()
    start = time.time()
    dev = torch.device("cuda")
    removal = json.loads(Path(REMOVAL).read_text())
    native = json.loads(Path(RANGES).read_text())
    tested = {frozenset((c["T"], c["S"])) for c in removal["cases"]}
    pairs = [p for p in removal["meta"]["pairs"] if frozenset((p["a"], p["b"])) in tested]
    selected = sorted({int(p[k]) for p in pairs for k in ("a", "b")})
    assert len(pairs) == len(tested)
    ae, ma, sa, _ = load_ae_checkpoint(CKPT, dev, allow_token_bias=True)
    ae.eval()
    cb, mb, sb, kb, _, _ = load_baseline_kmeans(Path(KM), dev)
    ct, mt, st, kt, _, _ = load_baseline_kmeans(Path(KMT), dev, allow_token_bias=True)
    with np.load(KMT, allow_pickle=True) as kmfile:
        tb = TokenBiasLookup(str(kmfile["token_bias"]), dev)
    ka = ae.centroids.shape[0]
    lookup = torch.full((ka,), -1, dtype=torch.long, device=dev)
    lookup[torch.tensor(selected, device=dev)] = torch.arange(len(selected), device=dev)
    pa = torch.tensor([selected.index(p["a"]) for p in pairs], device=dev)
    pb = torch.tensor([selected.index(p["b"]) for p in pairs], device=dev)

    def spaces(h, tok):
        return {"raw_residual": h, "residual": (h - mb) / sb,
                "token_centred": (h - mt) / st - tb(tok), "geoae": ae.encode((h - ma) / sa, tok)}

    dump = Path(args.dump)
    act = np.load(dump / "layer_27.npy", mmap_mode="r")
    tokens = np.load(dump / "rows_tok.npy", mmap_mode="r")
    v0 = int(len(act) * .95)
    n = len(act) - v0
    labels = np.empty(n, dtype=np.int64)
    contingency = {"residual": np.zeros((ka, kb), dtype=np.int64),
                   "token_centred": np.zeros((ka, kt), dtype=np.int64)}
    total, sums = {}, {}
    cnt = torch.zeros(len(selected), dtype=torch.float64, device=dev)

    def batches():
        for i in range(0, n, args.batch):
            h = torch.from_numpy(np.array(act[v0+i:v0+i+args.batch], copy=True)).to(dev).float()
            tok = torch.from_numpy(np.array(tokens[v0+i:v0+i+args.batch], dtype=np.int64)).to(dev)
            yield i, spaces(h, tok)

    for i, sp in batches():
        lab = pairwise_sq_dist(sp["geoae"], ae.centroids, metric=ae.metric).argmin(1)
        a = lab.cpu().numpy()
        labels[i:i+len(a)] = a
        local = lookup[lab]
        use = local >= 0
        cnt.index_add_(0, local[use], torch.ones(int(use.sum()), dtype=torch.float64, device=dev))
        for name, v in sp.items():
            total[name] = total.get(name, 0) + v.double().sum(0)
            sums.setdefault(name, torch.zeros(len(selected), v.shape[1], dtype=torch.float64, device=dev))
            sums[name].index_add_(0, local[use], v[use].double())
        for name, centroids, k in (("residual", cb, kb), ("token_centred", ct, kt)):
            b = pairwise_sq_dist(sp[name], centroids, metric="euclidean").argmin(1).cpu().numpy()
            np.add.at(contingency[name], (a, b), 1)
    means = {name: t / n for name, t in total.items()}
    mu = {name: s / cnt[:, None] - means[name] for name, s in sums.items()}
    unit = {name: torch.nn.functional.normalize(m[pa] / m[pa].norm(dim=1, keepdim=True) +
                                                m[pb] / m[pb].norm(dim=1, keepdim=True), dim=1)
            for name, m in mu.items()}
    mom = {name: torch.zeros(len(pairs), 6, dtype=torch.float64, device=dev) for name in SPACE_NAMES}
    for i, sp in batches():
        lab = torch.from_numpy(labels[i:i+len(sp["geoae"])]).to(dev)
        local = lookup[lab]
        ina, inb = (local[:, None] == pa).double(), (local[:, None] == pb).double()
        for name, v in sp.items():
            proj = (v.double() - means[name]) @ unit[name].T
            mom[name] += torch.stack((ina.sum(0), (ina*proj).sum(0), (ina*proj**2).sum(0),
                                     inb.sum(0), (inb*proj).sum(0), (inb*proj**2).sum(0)), 1)
    discovery = []
    for j, p in enumerate(pairs):
        row = {"a": p["a"], "b": p["b"], "showcase": p["showcase"], "spaces": {}}
        for name in SPACE_NAMES:
            m = mom[name][j]
            av, bv = m[1]/m[0], m[4]/m[3]
            va, vb = (m[2]/m[0] - av**2).clamp_min(0), (m[5]/m[3] - bv**2).clamp_min(0)
            aa, bb = mu[name][pa[j]], mu[name][pb[j]]
            cos = float(torch.nn.functional.cosine_similarity(aa, bb, dim=0))
            dp = float(abs(av-bv) / ((va+vb)/2 + 1e-12).sqrt())
            gap = aa-bb
            row["spaces"][name] = {"cosine": cos, "dprime": dp, "n_a": int(m[0]), "n_b": int(m[3]),
                                   "radial_share": float((gap @ unit[name][j])**2 / (gap @ gap)),
                                   "passes_geometry": cos >= .9 and dp >= 3}
        # This must reproduce the original discovery measurements up to JSON rounding.
        assert abs(row["spaces"]["geoae"]["cosine"] - p["cos"]) < .0015
        assert abs(row["spaces"]["geoae"]["dprime"] - p["dprime"]) < .015
        discovery.append(row)
    print(f"[counterparts] discovery on {n:,} rows complete ({time.time()-start:.1f}s)", flush=True)

    matches = {}
    for name, table in contingency.items():
        ac, bc = table.sum(1), table.sum(0)
        matches[name] = {}
        for a in selected:
            recall = table[a] / max(ac[a], 1)
            precision = table[a] / np.maximum(bc, 1)
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-30)
            order = np.argsort(-f1)[:5]
            matches[name][str(a)] = {"n_geoae": int(ac[a]), "top_by_f1": [
                {"base_cluster": int(b), "intersection": int(table[a,b]), "n_base": int(bc[b]),
                 "recall": float(recall[b]), "precision": float(precision[b]), "f1": float(f1[b])}
                for b in order], "top5_by_overlap_recall": float(np.sort(table[a])[-5:].sum() / max(ac[a], 1))}

    with np.load(args.cache, allow_pickle=False) as cache:
        h = torch.from_numpy(cache["h"]).to(dev).float()
        tok = torch.from_numpy(cache["tok"]).to(dev)
        cl, docs = cache["cluster"], cache["doc"]
        is_ctrl = cache["is_ctrl"].astype(bool)
    fit = docs % 2 == 0
    all_sp = {name: [] for name in SPACE_NAMES}
    for i in range(0, len(h), args.batch):
        for name, v in spaces(h[i:i+args.batch], tok[i:i+args.batch]).items():
            all_sp[name].append(v)
    all_sp = {name: torch.cat(parts) for name, parts in all_sp.items()}
    validation = []
    def idx(mask):
        return torch.from_numpy(np.flatnonzero(mask)).to(dev)
    for p in pairs:
        a, b = p["a"], p["b"]
        ia, ib = idx(fit & (cl == a)), idx(fit & (cl == b))
        ea, eb = idx(~fit & (cl == a)), idx(~fit & (cl == b))
        ctrl = idx(fit & is_ctrl & (cl != a) & (cl != b))
        row = {"a": a, "b": b, "n_fit": [len(ia), len(ib)], "n_eval": [len(ea), len(eb)], "spaces": {}}
        for name, v in all_sp.items():
            c = v[ctrl].double().mean(0)
            ua, ub = v[ia].double().mean(0)-c, v[ib].double().mean(0)-c
            u = torch.nn.functional.normalize(ua/ua.norm() + ub/ub.norm(), dim=0)
            fitcos = float(torch.nn.functional.cosine_similarity(ua, ub, dim=0))
            eua, eub = v[ea].double().mean(0)-c, v[eb].double().mean(0)-c
            evalcos = float(torch.nn.functional.cosine_similarity(eua, eub, dim=0))
            ta, tb_ = ((v[ia].double()-c) @ u).cpu().numpy(), ((v[ib].double()-c) @ u).cpu().numpy()
            xa, xb = ((v[ea].double()-c) @ u).cpu().numpy(), ((v[eb].double()-c) @ u).cpu().numpy()
            d = projection_stats(xa, xb)
            la, ha = ta.mean()-2*ta.std(), ta.mean()+2*ta.std()
            lb, hb = tb_.mean()-2*tb_.std(), tb_.mean()+2*tb_.std()
            d.update({"fit_cosine": fitcos, "eval_cosine": evalcos,
                      "gate_a_target_coverage": float(np.mean((xa>=la)&(xa<=ha))),
                      "gate_a_sibling_coverage": float(np.mean((xb>=la)&(xb<=ha))),
                      "gate_b_target_coverage": float(np.mean((xb>=lb)&(xb<=hb))),
                      "gate_b_sibling_coverage": float(np.mean((xa>=lb)&(xa<=hb))),
                      "passes_geometry": evalcos >= .9 and d["dprime"] >= 3})
            row["spaces"][name] = d
        validation.append(row)
    summaries = {}
    for name in SPACE_NAMES:
        summaries[name] = {
            "discovery_pass": sum(r["spaces"][name]["passes_geometry"] for r in discovery),
            "validation_pass": sum(r["spaces"][name]["passes_geometry"] for r in validation),
            "validation_median_cosine": float(np.median([r["spaces"][name]["eval_cosine"] for r in validation])),
            "validation_median_dprime": float(np.median([r["spaces"][name]["dprime"] for r in validation]))}
    out = {"meta": {"args": vars(args), "checkpoint": CKPT, "base_kmeans": KM, "token_kmeans": KMT,
                    "discovery_rows": n, "fresh_cache_rows": len(h), "tested_pairs": len(pairs),
                    "criteria": {"cosine": .9, "dprime": 3},
                    "centering": "discovery: global held-out mean; validation: pair-excluded fit control mean",
                    "validation_split": "even documents fit direction/gates; odd documents evaluate",
                    "native_pair_metrics_are_rounded": True, "seconds": time.time()-start},
           "summary": summaries, "discovery_fixed_labels": discovery, "validation_fixed_labels": validation,
           "membership_matches": matches,
           "native_range_pairs": {k: native["range_pairs"][k] for k in ("residual", "token_centred")}}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps(summaries, indent=2), flush=True)
    for row in validation:
        if (row["a"], row["b"]) in ((227,1662),(1453,1975),(1315,1842),(1081,1477)):
            print("EXAMPLE", row["a"], row["b"], json.dumps(row["spaces"]), flush=True)
    print(f"[counterparts] wrote {path} ({time.time()-start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
