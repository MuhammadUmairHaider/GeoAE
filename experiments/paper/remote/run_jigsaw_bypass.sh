#!/bin/bash
# Jigsaw on the last layers, the two token-removed ("bypass") spaces only: plain token-removed k-means vs the bypass AE.
# toxicity.npz is re-cached WITH BOS (as the training dumps are); every row kept (no anchor exclusion); fixed-point
# >5-row filter over just these two spaces; labels = any flag vs none, and the exact set of the six flags.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=.venv/bin/python
LCK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
GCK=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
L=e2e/checkpoints/general/llama3.2-3B/layer27/delta
G=e2e/checkpoints/general/gemma3-4B/layer33/delta

for spec in "llama:$LCK:cache_ids:results/delta/d6144/evals:km:$L/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz" \
            "gemma:$GCK:cache/gemma3-4b/l33:results/delta/g4b_l33_d5120/evals:km:$G/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz"; do
    IFS=: read -r name ck cache out kind km <<< "$spec"
    echo "=== $name: toxicity rung (with BOS) -> $cache"
    $PY -u -m geoae.extraction.concept_suite --checkpoint $ck --out $cache --only toxicity || echo "FAILED $name cache"
    $PY - $cache/toxicity.npz <<'EOF'
import sys, collections, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
print("  rows", len(d["label"]), collections.Counter(d["label"].tolist()))
c = collections.Counter(d["label_pattern"].tolist())
print("  flag patterns", len(c), "| largest:", c.most_common(6))
EOF
    echo "=== $name: probes (token-removed k-means vs bypass AE)"
    $PY -u geoae/evaluation/alignment.py --cache $cache --fp_min 6 --rungs toxicity,toxicity_pattern --no_anchor_exclusion \
        --arms "tok_plain=km:$km,geoae=ae:$ck" --out $out/probe_jigsaw_bypass.json || echo "FAILED $name probes"
    $PY - $out/probe_jigsaw_bypass.json <<'EOF'
import json, sys
R = json.load(open(sys.argv[1]))["rungs"]
for r, x in R.items():
    for sub in ("all", "common_fp6"):
        cells = "  ".join(f"{a}: h {s[sub]['homogeneity']:.3f} c {s[sub]['completeness']:.3f} NMI {s[sub]['nmi']:.3f} ({s[sub]['clusters']} cl)"
                          for a, s in x["arms"].items())
        n = next(iter(x["arms"].values()))[sub]["n"]
        print(f"  RESULT {r:<17} {sub:<11} rows {n:>5} of {x['n']}  {cells}")
EOF
done
echo JIGSAW_BYPASS_DONE
