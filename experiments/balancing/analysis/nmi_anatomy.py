"""Decompose the DBpedia-14 probe score: entropies, MI, shuffled-label MI, cluster sizes (same rows as the probe)."""

from geoae.paths import resolve_input_path
import os
from pathlib import Path
ROOT = Path(os.environ.get("GEOAE_ROOT", Path(__file__).resolve().parents[3]))
os.chdir(ROOT)
import sys, json
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "eval_out")
from geoae.analysis import concept_probe as CP
from geoae.training.initialization import anchor_row_indices

dev = torch.device("cuda")
cache = "cache_ids"
excl = anchor_row_indices(cache, None, 25, 5, 42, "", 25)
H, y = CP.load_rung(cache, "topic14.npz", "label", 0)
ids = CP.load_rung_ids(cache, "topic14.npz")
keep = np.ones(len(y), bool); keep[excl["topic14"][excl["topic14"] < len(y)]] = False
H, y, ids = H[keep], y[keep], ids[keep]
enc = lambda v: np.unique(v, return_inverse=True)[1]
yi = enc(np.asarray(y).astype(str))
D = "e2e/checkpoints/general/llama3.2-3B/layer27/delta/"
from pathlib import Path
from geoae.checkpoint import load_ae_checkpoint
from geoae.analysis.closest_tokens import load_baseline_kmeans
from geoae.token_bias import TokenBiasLookup
arms = {}
ae, mu, sd, _ = load_ae_checkpoint("checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt", dev, allow_token_bias=True)
ae.eval(); arms["geoae"] = ("ae", (ae, mu, sd))
for n, f in [("res_balanced", "balanced_kmeans_k2000_dpc_sampled.npz"), ("res_plain", "plain_kmeans_k2000_dpc_sampled_seed42.npz"),
             ("tok_plain", "plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz")]:
    C, m, s_, *_ = load_baseline_kmeans(Path(D + f), dev, allow_erasure=True)
    meta = np.load(resolve_input_path(D + f), allow_pickle=True)
    tbl = TokenBiasLookup(str(meta["token_bias"]), dev) if "token_bias" in meta.files else None
    arms[n] = ("km", (C, m, s_, None, tbl))
ent = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
def parts(y, c):
    ny, nc = y.max() + 1, c.max() + 1
    J = np.bincount(y * nc + c, minlength=ny * nc).reshape(ny, nc).astype(float); N = J.sum()
    py, pc, P = J.sum(1) / N, J.sum(0) / N, J / N
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / (py[:, None] * pc[None, :])[nz])).sum())
    return ent(py), ent(pc), mi
rng = np.random.default_rng(123)
print(f"rows {len(yi)}, classes {yi.max()+1}, class sizes {np.bincount(yi).min()}-{np.bincount(yi).max()}")
for name, arm in arms.items():
    c = enc(CP.assign(H, arm, dev, ids=ids))
    hy, hc, mi = parts(yi, c)
    nmi = mi / ((hy + hc) / 2)
    sh = [parts(rng.permutation(yi), c) for _ in range(20)]
    mi0 = np.array([s[2] for s in sh]); nmi0 = mi0 / ((hy + hc) / 2)
    cnt = np.bincount(c); cnt = cnt[cnt > 0]
    # purity: share of rows whose cluster's majority topic equals their topic
    J = np.zeros((c.max() + 1, yi.max() + 1)); np.add.at(J, (c, yi), 1)
    pur = J.max(1).sum() / len(yi)
    pur0 = np.mean([ (lambda Jp: Jp.max(1).sum() / len(yi))((lambda p: (np.add.at(Jq := np.zeros_like(J), (c, p), 1), Jq)[1])(rng.permutation(yi))) for _ in range(5)])
    print(json.dumps(dict(arm=name, live=int(len(cnt)), median_size=float(np.median(cnt)), singletons=int((cnt == 1).sum()),
          rows_in_clusters_le3=round(float(cnt[cnt <= 3].sum() / len(yi)), 3), largest=int(cnt.max()),
          H_topic=round(hy, 3), H_cluster=round(hc, 3), MI=round(mi, 3), NMI=round(nmi, 4),
          MI_shuffled=round(float(mi0.mean()), 3), NMI_shuffled=round(float(nmi0.mean()), 4), NMI_shuffled_sd=round(float(nmi0.std()), 4),
          cNMI=round(float((nmi - nmi0.mean()) / (1 - nmi0.mean())), 4), purity=round(float(pur), 3), purity_shuffled=round(float(pur0), 3))))
