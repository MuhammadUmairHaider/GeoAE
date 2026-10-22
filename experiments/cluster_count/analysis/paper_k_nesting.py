"""Does K = 4000 refine K = 2000? Cross-K nesting of clusters and range pairs (Llama-3.2-3B L27, d6144).
On the held-out 5% of the dump (same rows as paper_ranges_along_directions), assign every row in four
partitions: base balanced k-means at K2000 and K4000 (separate fits), and the bypass AE at K2000 and
K4000 (separately trained encoders). Then, per space:
  1. agreement of the two K: NMI, and refinement purity = row share of each K4000 cluster that lies in
     its majority K2000 parent (1 = perfectly nested), plus how many K4000 children cover 80% of a K2000
     cluster's rows;
  2. AE only, origin of each K4000 range pair (a, b) via the majority K2000 parents P(a), P(b):
     "split"     P(a) == P(b): one K2000 concept divided into two ranges of one direction,
     "inherited" {P(a), P(b)} is a K2000 range pair,
     "same_ray"  P(a), P(b) lie on one K2000 ray (connected component of the range-pair graph), not a pair,
     "new"       otherwise;
  3. AE only, persistence of each K2000 range pair (A, B): are A's and B's majority children a K4000
     range pair, or on the same K4000 ray (connected component of the K4000 range-pair graph)?
     Reference: the same rate for random K2000 cluster pairs with >= 50 held-out rows each.
    GEOAE_REMOTE=1 .venv/bin/python experiments/cluster_count/analysis/paper_k_nesting.py
"""

from geoae.paths import resolve_input_path
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from geoae.checkpoint import load_ae_checkpoint  # noqa: E402
from geoae.analysis.closest_tokens import load_baseline_kmeans  # noqa: E402
from geoae.losses import pairwise_sq_dist  # noqa: E402

KMD = "e2e/checkpoints/general/llama3.2-3B/layer27/delta"
CKD = "checkpoints/llama3.2-3B/layer27"
WIDTH = os.environ.get("NEST_WIDTH", "6144")
RUN = {"6144": ("d6144", "d6144k4000"), "12288": ("d12288", "d12288k4000")}[WIDTH]
OUT = Path(os.environ.get("NEST_OUT", f"results/delta/{RUN[1]}/evals/k_nesting.json"))
MIN_N = 50
dev = torch.device("cuda")
torch.set_grad_enabled(False)

D = Path("activations_sampled_10M")
A = np.load(resolve_input_path(D / "layer_27.npy"), mmap_mode="r")
v0 = int(A.shape[0] * 0.95)
H = np.asarray(A[v0:], dtype=np.float16)
tok = np.asarray(np.load(resolve_input_path(D / "rows_tok.npy"), mmap_mode="r")[v0:], dtype=np.int64)


def assign_km(path):
    C, m, s, _, _, _ = load_baseline_kmeans(Path(path), dev)
    out = np.empty(len(H), dtype=np.int32)
    for i in range(0, len(H), 16384):
        x = (torch.from_numpy(H[i:i + 16384].copy()).to(dev).float() - m) / s
        out[i:i + len(x)] = pairwise_sq_dist(x, C, metric="euclidean").argmin(1).cpu().numpy()
    return out


def assign_ae(path):
    ae, m, s, _ = load_ae_checkpoint(path, dev, allow_token_bias=True)
    ae.eval()
    out = np.empty(len(H), dtype=np.int32)
    for i in range(0, len(H), 16384):
        h = torch.from_numpy(H[i:i + 16384].copy()).to(dev).float()
        t = torch.from_numpy(tok[i:i + 16384]).to(dev)
        z = ae.encode((h - m) / s, t)
        out[i:i + len(z)] = pairwise_sq_dist(z, ae.centroids, metric=ae.metric).argmin(1).cpu().numpy()
    del ae
    return out


lab = {
    ("base", 2000): assign_km(f"{KMD}/balanced_kmeans_k2000_dpc_sampled.npz"),
    ("base", 4000): assign_km(f"{KMD}/balanced_kmeans_k4000_dpc_sampled.npz"),
    ("ae", 2000): assign_ae(f"{CKD}/k2000_bnh_b32k_lam1_d{WIDTH}_dpc_sampled_tokbias/step_0014450.pt"),
    ("ae", 4000): assign_ae(f"{CKD}/k4000_bnh_b32k_lam1_d{WIDTH}_dpc_sampled_tokbias/step_0014450.pt"),
}


def contingency(a, b, ka, kb):
    M = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(M, (a, b), 1)
    return M


res = {"meta": {"rows": len(H), "width": WIDTH, "runs": RUN, "min_n": MIN_N}}
parents, children = {}, {}
for sp in ("base", "ae"):
    a2, a4 = lab[(sp, 2000)], lab[(sp, 4000)]
    M = contingency(a2, a4, 2000, 4000)                          # K2000 parent x K4000 child
    n4, n2 = M.sum(0), M.sum(1)
    live4, live2 = n4 > 0, n2 > 0
    purity_rows = M.max(0)[live4].sum() / n4[live4].sum()
    big4 = n4 >= MIN_N
    purity_cluster = np.median(M.max(0)[big4] / n4[big4])
    need = []
    for p in np.nonzero(n2 >= MIN_N)[0]:
        c = np.sort(M[p])[::-1].cumsum() / n2[p]
        need.append(int(np.searchsorted(c, 0.8) + 1))
    res[sp] = {"nmi": float(normalized_mutual_info_score(a2, a4)),
               "refinement_purity_rows": float(purity_rows),
               "refinement_purity_median_cluster": float(purity_cluster),
               "children_for_80pct_median": float(np.median(need)),
               "children_for_80pct_share_le2": float(np.mean(np.asarray(need) <= 2)),
               "parents_live": int(live2.sum()), "children_live": int(live4.sum())}
    parents[sp] = np.where(live4, M.argmax(0), -1)               # majority K2000 parent of each K4000 cluster
    children[sp] = np.where(live2, M.argmax(1), -1)              # majority K4000 child of each K2000 cluster
    print(sp, json.dumps(res[sp]))


def range_pairs(run):
    d = json.loads(resolve_input_path(Path(f"results/delta/{run}/evals/ranges_along_directions.json")).read_text())
    return {frozenset((p["a"], p["b"])) for p in d["range_pairs"]["geoae"]}


def rays(pairs):
    """Connected components of the range-pair graph: cluster -> ray id."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in pairs:
        a, b = tuple(p)
        parent[find(a)] = find(b)
    return {x: find(x) for x in parent}


rp2, rp4 = range_pairs(RUN[0]), range_pairs(RUN[1])
ray2, ray4 = rays(rp2), rays(rp4)
P, Ch = parents["ae"], children["ae"]

origin = {"split": 0, "inherited": 0, "same_ray": 0, "new": 0}
for p in rp4:
    a, b = tuple(p)
    pa, pb = int(P[a]), int(P[b])
    if pa == pb:
        origin["split"] += 1
    elif frozenset((pa, pb)) in rp2:
        origin["inherited"] += 1
    elif pa in ray2 and pb in ray2 and ray2[pa] == ray2[pb]:
        origin["same_ray"] += 1
    else:
        origin["new"] += 1


def persistence(pairs):
    hit_pair = hit_ray = n = 0
    for p in pairs:
        A_, B_ = tuple(p)
        ca, cb = int(Ch[A_]), int(Ch[B_])
        if ca < 0 or cb < 0:
            continue
        n += 1
        hit_pair += frozenset((ca, cb)) in rp4 or ca == cb
        hit_ray += ca == cb or (ca in ray4 and cb in ray4 and ray4[ca] == ray4[cb])
    return {"n": n, "child_pair_is_range_pair_or_same": hit_pair / max(n, 1), "children_on_same_ray": hit_ray / max(n, 1)}


rng = np.random.default_rng(0)
n2 = np.bincount(lab[("ae", 2000)], minlength=2000)
ok = np.nonzero(n2 >= MIN_N)[0]
rand = set()
while len(rand) < 5000:
    a, b = rng.choice(ok, 2, replace=False)
    rand.add(frozenset((int(a), int(b))))
res["ae_range_pairs"] = {"k2000": len(rp2), "k4000": len(rp4),
                         "k4000_origin": origin,
                         "k4000_origin_share": {k: v / len(rp4) for k, v in origin.items()},
                         "k2000_persistence": persistence(rp2),
                         "random_pair_persistence": persistence(rand)}
print(json.dumps(res["ae_range_pairs"], indent=1))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print("wrote", OUT)
