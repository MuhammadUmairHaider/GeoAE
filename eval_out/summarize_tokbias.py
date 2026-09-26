"""Token-bypass AE (design B) vs its parent d6144_new — the decision tables.

    .venv/bin/python eval_out/summarize_tokbias.py

Reads (whatever exists, so it can run mid-sweep):
  eval_out/token_structure_tokbias.json   token-detector share, cNMI with cur/next token, recon
  eval_out/probe_tokbias.json             chance-corrected probe on cache_ids/, paired bootstrap
  eval_out/mmlu_d6144_sampled_preds.json  parent MMLU splice, per-question
  eval_out/mmlu_d6144_sampled_tokbias.json
  eval_out/cq_tokbias.json                clustering quality on the new dump

The bypass arm passes if the sequence rungs rise over d6144_new (CI excludes 0),
token-trivial clusters fall, and reconstruction / MMLU do not get worse.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

A, B = "d6144_new", "tokbias"


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def header(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def token_structure():
    d = load("eval_out/token_structure_tokbias.json")
    header("TOKEN STRUCTURE — held-out dump rows (trivial = one token >= 80% / 50% of a cluster)")
    if not d:
        print("  (missing)"); return
    keys = ["fvu", "cnmi_cur", "cnmi_next", "trivial_cur_80", "trivial_cur_50",
            "trivial_next_80", "trivial_next_50", "live", "top10", "usage_H"]
    print(f"  {'':16s}" + "".join(f"{k:>11s}" for k in d))
    for k in keys:
        print(f"  {k:16s}" + "".join(
            f"{d[a][k]:>11.4f}" if isinstance(d[a].get(k), float) else f"{str(d[a].get(k, '—')):>11s}" for a in d))


def probe():
    d = load("eval_out/probe_tokbias.json")
    header(f"CONCEPT PROBE — chance-corrected NMI on cache_ids/; Δ = {B} − {A} [95% CI]; leak = cNMI(label; token id)")
    if not d:
        print("  (missing)"); return
    arms = list(next(iter(d.values()))["arms"])
    print(f"  {'rung':16s} {'leak':>6s}" + "".join(f"{a[:10]:>11s}" for a in arms) + f"   {'Δ tokbias − d6144_new':>26s}")
    grp = {"token": [], "sequence": []}
    for rung, v in d.items():
        ar = v["arms"]
        leak = "     —" if v.get("leak") is None else f"{v['leak']:6.3f}"
        cells = "".join(f"{ar[a]['cnmi']:11.3f}" if a in ar else f"{'—':>11s}" for a in arms)
        delta = ""
        if A in ar and B in ar:
            bt = np.array(ar[B]["boots"]) - np.array(ar[A]["boots"])
            m = ar[B]["cnmi"] - ar[A]["cnmi"]
            lo, hi = m + np.quantile(bt - bt.mean(), [.025, .975])   # centred: resampled NMI is biased up
            delta = f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}"
            grp[v["grain"]].append(m)
        print(f"  {rung:16s} {leak}{cells}   {delta:>26s}")
    for g, v in grp.items():
        if v:
            print(f"  MEAN Δ {g:9s} {np.mean(v):+.3f} over {len(v)} rungs ({sum(x > 0 for x in v)} up)")


def mmlu():
    a, b = load("eval_out/mmlu_d6144_sampled_preds.json"), load("eval_out/mmlu_d6144_sampled_tokbias.json")
    header("MMLU UNDER THE SPLICE (n=2000) — paired over questions")
    if not (a and b):
        print("  (missing)"); return
    for n, x in ((A, a), (B, b)):
        m = x["meta"]
        print(f"  {n:12s} base {m['base_acc']:.4f}  recon {m['recon_acc']:.4f}  delta {m['delta']:+.4f}")
    g = np.array(a["preds"]["gold"])
    ra, rb = np.array(a["preds"]["recon"]) == g, np.array(b["preds"]["recon"]) == g
    only_a, only_b = int((ra & ~rb).sum()), int((rb & ~ra).sum())
    p = stats.binomtest(only_b, only_a + only_b).pvalue if only_a + only_b else 1.0
    print(f"  recon correct only under {A}: {only_a}, only under {B}: {only_b}  (McNemar exact p={p:.3f})")


def cq():
    d = load("eval_out/cq_tokbias.json")
    header("CLUSTERING QUALITY — new dump, 1M rows (silhouette is the stable row; ignore dunn)")
    if not d:
        print("  (missing)"); return
    print(json.dumps(d, indent=1)[:3000])


if __name__ == "__main__":
    token_structure()
    probe()
    mmlu()
    cq()
