"""Gemma 3 4B L33 ablation table: K series and component ablations of the token-bypass AE (runs of 11-12 Oct 2026).

One row per AE; missing files print as a dash. Run from the repo root:
    python3 experiments/regularization/analysis/g4b_ablation_table.py

Columns and their files (all under results/delta/g4b_l33_d5120/):
  CH            Calinski-Harabasz on 1M rows, in brackets relative to k-means at the same K
                (K = 2,000 rows: balanced k-means on the residual stream, 125.8; K rows: plain k-means at that K).
                evals/cq.json, evals/kcost/cq_k<K>.json, evals/ablations/<arm>/cq.json
  Single-token  share of clusters whose members are >= 80% one current token (trivial_cur_80 of token_structure*.json)
  MMLU, ppl     with the reconstruction spliced in (mmlu*.json meta.recon_acc; splice_ppl*.json spliced.ppl)
  Atlas         last-word FineWeb-Atlas homogeneity on the rows kept for ALL K = 2,000 codebooks at once
                (evals/sae_suite/atlas_lastword/ablations/nmi_filtered_joint.json): mean of the three 14-topic draws,
                and the 31 document types. K rows have their own row sets and are printed separately.
  Steering      plain steering, alpha 0.5, three-way true positives: mean over concepts of the selectivity difference
                AE - Base in accuracy points, bootstrap over concepts, (better/worse);
                interventions/sae/range_intervention_<ds>_dprime_hzs<tag>_tp3.json. Each row has its own documents.
"""

from geoae.paths import resolve_input_path
import json
import os

import numpy as np

R = "results/delta/g4b_l33_d5120/"
E = R + "evals/"
S = R + "interventions/sae/"
A = E + "sae_suite/atlas_lastword/ablations/"
ARMS = (("nosinkhorn", "no balancing (Sinkhorn)"), ("nosinkhorn_lloyd_noreinit", "no balancing, k-means init, no re-seeding"), ("nopull", "no pull term"), ("nosep", "no separation hinge"), ("novicreg", "no variance / covariance terms"),
        ("kmppinit", "k-means++ initialization"), ("nostage", "no staged schedule"))
DRAWS = ("atlas14_s0", "atlas14_s1", "atlas14_s2")


def load(p):
    return json.load(open(resolve_input_path(p))) if resolve_input_path(os.path).exists(p) else None


def g(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def steer(tag, ds):
    d = load(f"{S}range_intervention_{ds}_dprime_hzs{tag}_tp3.json")
    if not d or "summary" not in d:
        return "—"
    x = np.array([c["z_st_global_a0.5"]["selectivity"] - c["h_st_global_a0.5"]["selectivity"] for c in d["concepts"].values()])
    rng = np.random.default_rng(0)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{100 * x.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] ({(x > 0).sum()}/{(x < 0).sum()}, n={len(x)})"


joint = g(load(A + "nmi_filtered_joint.json"), "sets")


def atlas(arm):
    if not joint or arm not in joint["doctype"]["point"]:
        return None, None
    return float(np.mean([joint[d]["point"][arm]["homogeneity"] for d in DRAWS])), joint["doctype"]["point"][arm]["homogeneity"]


rows = []
cq = load(E + "cq.json"); ts = load(E + "token_structure.json"); mm = load(E + "mmlu.json"); pp = load(E + "splice_cost/splice_ppl_ae.json")
chb = g(cq, "base", "calinski_harabasz")
rows.append(("k-means on the residual stream", chb, chb, g(ts, "base", "trivial_cur_80"), g(mm, "meta", "base_acc"), g(pp, "clean", "ppl"), *atlas("plain"), "", ""))
rows.append(("k-means, token removed", None, None, g(ts, "base_tokmean", "trivial_cur_80"), None, None, *atlas("plain_tokmean"), "", ""))
rows.append(("GeoAE, full recipe (K=2,000)", g(cq, "bypass", "calinski_harabasz"), chb, g(ts, "bypass", "trivial_cur_80"), g(mm, "meta", "recon_acc"),
             g(pp, "spliced", "ppl"), *atlas("bypass"), steer("", "ag_news"), steer("", "db14")))
N = "results/delta/g4b_l33_nobypass/evals/"
cqn = load(N + "cq.json"); tsn = load(N + "token_structure.json"); mmn = load(N + "mmlu.json")
kn = [k for k in (cqn or {}) if k != "base"]; kt = [k for k in (tsn or {}) if not k.startswith(("base", "plain"))]
rows.append(("no token bypass", g(cqn, kn[0], "calinski_harabasz") if kn else None, g(cqn, "base", "calinski_harabasz") or chb,
             g(tsn, kt[0], "trivial_cur_80") if kt else None, g(mmn, "meta", "recon_acc"), None, None, None, steer("_nobypass", "ag_news"), steer("_nobypass", "db14")))
tk = load(E + "kcost/token_structure.json") or {}
for K, sfx, tag in ((256, "256", "_k256"), (4000, "4k", None), (8000, "8k", "_k8000")):
    c = load(E + f"kcost/cq_k{K}.json"); t = load(E + "kcost/token_structure_k256.json") if K == 256 else tk
    m = load(E + f"kcost/mmlu_k{K}.json"); p = load(E + f"kcost/splice_ppl_k{K}.json")
    rows.append((f"K = {K:,}", g(c, f"bypass{sfx}", "calinski_harabasz"), g(c, f"plain{sfx}", "calinski_harabasz"), g(t, f"bypass{sfx}", "trivial_cur_80"),
                 g(m, "meta", "recon_acc"), g(p, "spliced", "ppl"), None, None, steer(tag, "ag_news") if tag else "—", steer(tag, "db14") if tag else "—"))
for arm, name in ARMS:
    D = E + f"ablations/{arm}/"
    c = load(D + "cq.json"); t = load(D + "token_structure.json"); m = load(D + "mmlu.json"); p = load(D + "splice_ppl.json")
    rows.append((name, g(c, arm, "calinski_harabasz"), g(c, "base", "calinski_harabasz") or chb, g(t, arm, "trivial_cur_80"),
                 g(m, "meta", "recon_acc"), g(p, "spliced", "ppl"), *atlas(arm), steer("_" + arm, "ag_news"), steer("_" + arm, "db14")))

f = lambda v, fmt, mul=1.0: "—" if v is None else format(mul * v, fmt)
print("| Variant | CH ↑ (× k-means) | Single-token (%) ↓ | MMLU (%) ↑ | Perplexity ↓ | Atlas 14 topics ↑ | Atlas doc type ↑ | Steering AG News: AE − Base | Steering DBpedia-14: AE − Base |")
print("|---|---|---|---|---|---|---|---|---|")
for name, ch, chb_, st, m_, p_, a14, adt, sa, sd in rows:
    chs = "—" if ch is None else f"{ch:.1f}" + (f" ({ch / chb_:.1f}×)" if chb_ else "")
    print(f"| {name} | {chs} | {f(st, '.1f', 100)} | {f(m_, '.1f', 100)} | {f(p_, '.2f')} | {f(a14, '.3f')} | {f(adt, '.3f')} | {sa or '—'} | {sd or '—'} |")

if joint:
    print(f"\nAtlas rows kept for every K = 2,000 codebook: 14 topics {[joint[d]['n_kept'] for d in DRAWS]}, document type {joint['doctype']['n_kept']:,}.")
    print("Full recipe − arm, homogeneity, per draw and document type [95% CI, document bootstrap]:")
    for arm in joint["doctype"]["point"]:
        if arm == "bypass":
            continue
        ds = [joint[d]["diff_ref_minus_arm"][arm]["homogeneity"] for d in DRAWS]
        dt = joint["doctype"]["diff_ref_minus_arm"][arm]["homogeneity"]
        print(f"  {arm:14s} " + " / ".join(f"{x['point']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]" for x in ds)
              + f" | doc type {dt['point']:+.3f} [{dt['ci'][0]:+.3f}, {dt['ci'][1]:+.3f}]")
k = g(load(A + "nmi_filtered_k256.json"), "sets")
if k:
    h = lambda st, arm: k[st]["point"][arm]["homogeneity"]
    print("\nK = 256 on its own rows, homogeneity (residual k-means / token-removed k-means / GeoAE, all K = 256): "
          f"14 topics {np.mean([h(d, 'plain256') for d in DRAWS]):.3f} / {np.mean([h(d, 'plain_tokmean256') for d in DRAWS]):.3f} / {np.mean([h(d, 'bypass256') for d in DRAWS]):.3f}; "
          f"document type {h('doctype', 'plain256'):.3f} / {h('doctype', 'plain_tokmean256'):.3f} / {h('doctype', 'bypass256'):.3f}.")
