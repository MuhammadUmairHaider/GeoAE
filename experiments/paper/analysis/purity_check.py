"""Is GeoAE's higher topic purity an artifact of tiny clusters? Same rows and assignment as probe_chance_corrected."""

from geoae.paths import resolve_input_path
import os
from pathlib import Path
ROOT = Path(os.environ.get("GEOAE_ROOT", Path(__file__).resolve().parents[3]))
os.chdir(ROOT)
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from geoae.analysis import concept_probe as CP
from geoae.training.initialization import anchor_row_indices
from geoae.checkpoint import load_ae_checkpoint
from geoae.analysis.closest_tokens import load_baseline_kmeans
from geoae.token_bias import TokenBiasLookup

dev = torch.device("cuda")
cache = "cache_ids"
D = "e2e/checkpoints/general/llama3.2-3B/layer27/delta/"
arms = {}
ae, mu, sd, _ = load_ae_checkpoint("checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt",
                                   dev, allow_token_bias=True)
ae.eval(); arms["geoae"] = ("ae", (ae, mu, sd))
for n, f in [("res_balanced", "balanced_kmeans_k2000_dpc_sampled.npz"), ("res_plain", "plain_kmeans_k2000_dpc_sampled_seed42.npz"),
             ("tok_plain", "plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz")]:
    C, m, s_, *_ = load_baseline_kmeans(Path(D + f), dev, allow_erasure=True)
    meta = np.load(resolve_input_path(D + f), allow_pickle=True)
    tbl = TokenBiasLookup(str(meta["token_bias"]), dev) if "token_bias" in meta.files else None
    arms[n] = ("km", (C, m, s_, None, tbl))
excl = anchor_row_indices(cache, None, 25, 5, 42, "", 25)
enc = lambda v: np.unique(v, return_inverse=True)[1]


def table(c, y):
    J = np.zeros((c.max() + 1, y.max() + 1)); np.add.at(J, (c, y), 1)
    return J[J.sum(1) > 0]


def purity(J, min_size=1):
    n = J.sum(1); k = n >= min_size
    return float(J[k].max(1).sum() / max(n[k].sum(), 1)), float(n[k].sum() / n.sum())


def loo_purity(J):
    """Each row is predicted by the majority topic of the OTHER rows in its cluster (singletons: no prediction = wrong;
    ties split the credit). Free of the 'a singleton is always pure' bias."""
    tot = 0.0
    for row in J:
        for yy in np.nonzero(row)[0]:
            o = row.copy(); o[yy] -= 1
            m = o.max()
            if m == 0:
                continue
            ties = o == m
            tot += row[yy] * ties[yy] / ties.sum()
    return float(tot / J.sum())


def split_half(c, y, rng, reps=100):
    """Majority topic per cluster from a random half predicts the other half (clusters unseen in the first half = wrong)."""
    out = []
    for _ in range(reps):
        a = rng.random(len(y)) < 0.5
        Ja = np.zeros((c.max() + 1, y.max() + 1)); np.add.at(Ja, (c[a], y[a]), 1)
        seen = Ja.sum(1) > 0
        pred = Ja.argmax(1)
        b = ~a
        out.append(float((seen[c[b]] & (pred[c[b]] == y[b])).mean()))
    return float(np.mean(out))


def pairs(J):
    """Pair precision P(same topic | same cluster), pair recall P(same cluster | same topic), ARI. Singletons form no pairs."""
    c2 = lambda x: x * (x - 1) / 2
    s_ij = c2(J).sum(); s_a = c2(J.sum(1)).sum(); s_b = c2(J.sum(0)).sum(); n2 = c2(J.sum())
    exp = s_a * s_b / n2
    return float(s_ij / s_a), float(s_ij / s_b), float((s_ij - exp) / (0.5 * (s_a + s_b) - exp))


def bins(J):
    n = J.sum(1); out = {}
    for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10**9)]:
        k = (n >= lo) & (n <= hi)
        out[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = dict(clusters=int(k.sum()), rows=round(float(n[k].sum() / n.sum()), 3),
                                                          purity=round(float(J[k].max(1).sum() / max(n[k].sum(), 1)), 3))
    return out


rng = np.random.default_rng(0)
res = {}
SEQ = ["topic14", "topic4", "topic20", "language", "domain", "subjectivity", "formality", "sentiment", "sentiment_long"]
for rung, fname, key, grain, _d in CP.LADDER:
    if rung not in SEQ:
        continue
    H, y = CP.load_rung(cache, fname, key, 0)
    ids = CP.load_rung_ids(cache, fname)
    if rung in excl:
        keep = np.ones(len(y), bool); keep[excl[rung][excl[rung] < len(y)]] = False
        H, y, ids = H[keep], y[keep], ids[keep]
    yi = enc(np.asarray(y).astype(str))
    res[rung] = {}
    for name, arm in arms.items():
        c = enc(CP.assign(H, arm, dev, ids=ids))
        J = table(c, yi)
        sh = [table(c, rng.permutation(yi)) for _ in range(10)]
        r = dict(live=int(len(J)), purity=purity(J)[0], purity_shuffled=float(np.mean([purity(S)[0] for S in sh])),
                 loo_purity=loo_purity(J), loo_purity_shuffled=float(np.mean([loo_purity(S) for S in sh[:5]])),
                 split_half=split_half(c, yi, rng))
        for ms in (2, 5, 10, 20):
            p, cov = purity(J, ms); r[f"purity_min{ms}"] = p; r[f"rows_min{ms}"] = cov
        r["pair_precision"], r["pair_recall"], r["ari"] = pairs(J)
        if rung == "topic14":
            r["bins"] = bins(J)
        res[rung][name] = r
    print(rung, json.dumps({a: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "bins"}
                            for a, r in res[rung].items()}), flush=True)
print("BINS", json.dumps({a: r["bins"] for a, r in res["topic14"].items()}))
Path("results/delta/d6144/evals").mkdir(parents=True, exist_ok=True)
Path("results/delta/d6144/evals/purity_check.json").write_text(json.dumps(res, indent=1))
print("wrote results/delta/d6144/evals/purity_check.json")
