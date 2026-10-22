"""Do a space's codebook clusters cut through its own neighbourhoods, and how many clusters does a topic spread over?
Last-word Atlas rows (one per chunk), a kNN graph of geoae/analysis/atlas_knn_graph.py and the codebook labels of
geoae/evaluation/atlas_assign.py. Per (space, codebook):
  same_cluster_all   mean over rows of the share of its k neighbours (other documents) that sit in the row's own cluster
  same_cluster_hit   the same over neighbour pairs whose chunks share a 14-topic class (single-label chunks of the three
                     draws of draws.json): of the same-topic neighbours a space finds, how many its codebook keeps together
  topic_spread_50    per 14-topic class, the number of clusters holding half of the class's chunks (mean over classes)
  topic_eff_clusters per class, exp(entropy) of the class's distribution over clusters (mean over classes)

    scripts/py geoae/analysis/atlas_cluster_knn_agreement.py --graph G --assign A --chunks C --out O
"""
from __future__ import annotations

import argparse
import json

import numpy as np

DEFAULT_DRAWS = "results/delta/d6144/interventions/atlas_classes/draws.json"
PAIRS = (("base", "base"), ("base", "plain"), ("base_tokmean", "base_tokmean"), ("base_tokmean", "plain_tokmean"),
         ("bypass", "bypass"))                                             # (kNN space, codebook living in that space)


def exclusive(label_ptr, label_ids, units):
    lut = np.full(int(label_ids.max()) + 1, -1, dtype=np.int64)
    for j, u in enumerate(units):
        lut[np.asarray(u, dtype=np.int64)] = j
    n = len(label_ptr) - 1
    chunk_of = np.repeat(np.arange(n), np.diff(label_ptr))
    j = lut[label_ids]
    pairs = np.unique(np.stack([chunk_of[j >= 0], j[j >= 0]], 1), axis=0)
    one = np.bincount(pairs[:, 0], minlength=n)[pairs[:, 0]] == 1
    return pairs[one, 0], pairs[one, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--draws", default=DEFAULT_DRAWS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    G, A, z = np.load(args.graph), np.load(args.assign), np.load(args.chunks)
    g_chunk = G["chunk"].astype(np.int64)
    assert (A["row_chunk"] == np.arange(len(A["row_chunk"]))).all() and (g_chunk == np.arange(len(g_chunk))).all()
    draws = json.load(open(args.draws))
    res = {"meta": vars(args), "pairs": {}}
    for space, cb in PAIRS:
        nbr = G[f"nbr_{space}"].astype(np.int64)
        lab = A[cb].astype(np.int64)
        same = lab[nbr] == lab[:, None]
        out = dict(same_cluster_all=float(same.mean()), draws={})
        for d in draws["draws"]:
            ch, y = exclusive(z["label_ptr"], z["label_ids"], [c["ids"] for c in d["classes"]])
            cls = np.full(len(lab), -1, dtype=np.int64)
            cls[ch] = y
            q = cls >= 0
            hit = (cls[nbr] == cls[:, None]) & q[:, None]
            spread, eff = [], []
            for c in range(len(d["classes"])):
                cnt = np.sort(np.bincount(lab[ch[y == c]]))[::-1]
                cnt = cnt[cnt > 0]
                if cnt.sum() == 0:
                    continue
                spread.append(int(np.searchsorted(np.cumsum(cnt), 0.5 * cnt.sum())) + 1)
                p = cnt / cnt.sum()
                eff.append(float(np.exp(-(p * np.log(p)).sum())))
            out["draws"][d["seed"]] = dict(same_topic_neighbours_per_query=float(hit[q].sum(1).mean()),
                                           same_cluster_hit=float(same[hit].mean()),
                                           topic_spread_50=float(np.mean(spread)), topic_eff_clusters=float(np.mean(eff)))
        dd = out["draws"].values()
        out["mean"] = {k: float(np.mean([v[k] for v in dd])) for k in next(iter(dd))}
        res["pairs"][f"{space}|{cb}"] = out
        m = out["mean"]
        print(f"space {space:13s} codebook {cb:13s}  neighbours in own cluster {out['same_cluster_all']:.3f}  | 14-topic: "
              f"same-topic neighbours/query {m['same_topic_neighbours_per_query']:.2f}, of them in own cluster "
              f"{m['same_cluster_hit']:.3f}; clusters for half a topic {m['topic_spread_50']:.1f}, "
              f"effective clusters per topic {m['topic_eff_clusters']:.0f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
