"""Bypass side of the probe table, old recipe vs the 12 Oct 50-epoch no-balancing no-staging models: homogeneity of the
bypass AE and of plain token-removed k-means (the paper's pair filter for sequence / entity rungs, min10 for token rungs).

    python3 experiments/balancing/analysis/ns50_probe_compare.py
Old files: the four-space probe_dense_subsets.json of each model; new: <arm>_ns50/evals/probe_dense_subsets_bypasspair.json
(experiments/balancing/argo/ns50_probe_bypass_pair.sbatch) or, once the no-bypass models exist, the four-space file.
"""

from geoae.paths import resolve_input_path
import json
import os

import numpy as np

GROUPS = [("Token-level", ["surface", "pos_coarse", "pos_fine", "ner_coarse", "ner_fine"]),
          ("Sequence-level", ["sentiment", "sentiment_long", "subjectivity", "formality", "language", "domain", "topic4", "topic14", "topic20", "toxicity", "toxicity_pattern"]),
          ("Entity and relation", ["ravel_country", "ravel_continent", "ravel_language", "ioi_role", "ioi_name"])]
Q = "results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table/"
MODELS = [("Llama", "results/delta/d6144_nobypass/evals/probe_dense_subsets.json", "results/delta/d6144_ns50/evals/", "results/delta/d6144_nobypass_ns50/evals/"),
          ("Gemma", "results/delta/g4b_l33_nobypass/evals/probe_dense_subsets.json", "results/delta/g4b_l33_d5120_ns50/evals/", "results/delta/g4b_l33_nobypass_ns50/evals/"),
          ("Qwen", Q + "probe_dense_subsets_fourway.json", "results/delta/qwen25_l35_d4096_ns50/evals/", "results/delta/qwen25_l35_nobypass_ns50/evals/")]


def load(p):
    return json.load(open(resolve_input_path(p)))["rungs"] if resolve_input_path(os.path).exists(p) else None


def cell(R, rung, grp, arm):
    if R is None or rung not in R or arm not in R[rung]["arms"]:
        return None
    a = R[rung]["arms"][arm]
    key = "min10" if grp == "Token-level" else "pair_fp10_tok_plain+geoae"
    return a[key]["homogeneity"] if key in a else None


f = lambda v: "—" if v is None else f"{v:.3f}"
for name, old, newdir, nbdir in MODELS:
    O = load(old)
    N = load(nbdir + "probe_dense_subsets.json") or load(newdir + "probe_dense_subsets_bypasspair.json")
    print(f"\n### {name}: token-removed k-means vs bypass AE, homogeneity (old recipe | new recipe)")
    print("| Task | k-means | AE old | AE new | new − old |")
    print("|---|---|---|---|---|")
    for grp, rungs in GROUPS:
        acc = {"k": [], "o": [], "n": []}
        for r in rungs:
            k, o, n = cell(O, r, grp, "tok_plain"), cell(O, r, grp, "geoae"), cell(N, r, grp, "geoae")
            if o is None and n is None:
                continue
            print(f"| {r} | {f(k)} | {f(o)} | {f(n)} | {'—' if None in (o, n) else format(n - o, '+.3f')} |")
            if None not in (k, o, n):
                acc["k"].append(k); acc["o"].append(o); acc["n"].append(n)
        if acc["o"]:
            print(f"| **{grp} mean** | **{np.mean(acc['k']):.3f}** | **{np.mean(acc['o']):.3f}** | **{np.mean(acc['n']):.3f}** | **{np.mean(acc['n']) - np.mean(acc['o']):+.3f}** |")
