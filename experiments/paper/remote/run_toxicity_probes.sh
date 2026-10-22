#!/bin/bash
# Jigsaw toxicity rung (balanced, any flag vs clean) added to the last-layer four-way probe tables, Llama L27 and
# Gemma 3 4B L33. 1) extraction check: re-encode SST-2 into a scratch dir and compare with the stored rung (same
# encoder path as the existing caches); 2) cache toxicity.npz into both caches; 3) rerun the four-way probes with
# EXACTLY run_fourway_fp6.sh's arms (old files kept as *.pre_toxicity.json) and check the old rungs reproduce.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=.venv/bin/python
CHK=$HOME/tox_check; rm -rf $CHK; mkdir -p $CHK
LCK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
GCK=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt

for spec in "llama:$LCK:cache_ids" "gemma:$GCK:cache/gemma3-4b/l33"; do
    IFS=: read -r name ck cache <<< "$spec"
    echo "=== $name: extraction check (sentiment)"
    $PY -u -m geoae.extraction.concept_suite --checkpoint $ck --out $CHK/$name --only sentiment || echo "FAILED $name check"
    $PY - $CHK/$name/sentiment.npz $cache/sentiment.npz <<'EOF'
import sys, numpy as np
a, b = (np.load(p, allow_pickle=True) for p in sys.argv[1:3])
for k in ("H_last", "H_mean"):
    x, y = a[k].astype(np.float64), b[k].astype(np.float64)
    cos = (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
    rel = np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
    print(f"  {k}: shape {x.shape} vs {y.shape}  min cos {cos.min():.6f}  max rel diff {rel.max():.2e}")
print("  labels equal:", (a["label"] == b["label"]).all(), " last ids equal:", (a["last_token_id"] == b["last_token_id"]).all())
EOF
    echo "=== $name: toxicity rung -> $cache"
    $PY -u -m geoae.extraction.concept_suite --checkpoint $ck --out $cache --only toxicity || echo "FAILED $name toxicity"
done

D=e2e/checkpoints/general/gemma3-4B/layer33/delta
O=results/delta/g4b_l33_nobypass/evals
L=e2e/checkpoints/general/llama3.2-3B/layer27/delta
NL=results/delta/d6144_nobypass/evals
[ -f $O/probe_dense_subsets.pre_toxicity.json ] || cp -p $O/probe_dense_subsets.json $O/probe_dense_subsets.pre_toxicity.json
[ -f $NL/probe_dense_subsets.pre_toxicity.json ] || cp -p $NL/probe_dense_subsets.json $NL/probe_dense_subsets.pre_toxicity.json
echo "=== gemma probes"
$PY -u geoae/evaluation/alignment.py --cache cache/gemma3-4b/l33 --fp_min 6 \
    --arms "res_plain=km:$D/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz,tok_plain=km:$D/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz,geoae=ae:checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt,nobypass=ae:checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt" \
    --out $O/probe_dense_subsets.json || echo "FAILED gemma probes"
$PY experiments/paper/analysis/summarize_probe_raw_hc.py $O/probe_dense_subsets.json common_fp6 all > $O/probe_raw_hc.md
echo "=== llama probes"
$PY -u geoae/evaluation/alignment.py --cache cache_ids --fp_min 6 \
    --arms "res_plain=km:$L/plain_kmeans_k2000_dpc_sampled_seed42.npz,tok_plain=km:$L/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz,geoae=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt,nobypass=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt" \
    --out $NL/probe_dense_subsets.json || echo "FAILED llama probes"
$PY experiments/paper/analysis/summarize_probe_raw_hc.py $NL/probe_dense_subsets.json common_fp6 all > $NL/probe_raw_hc.md

echo "=== old rungs reproduce? (max |new - old| over every arm, subset and score)"
for f in $O $NL; do
    $PY - $f/probe_dense_subsets.pre_toxicity.json $f/probe_dense_subsets.json <<'EOF'
import json, sys
old, new = (json.load(open(p))["rungs"] for p in sys.argv[1:3])
worst = 0.0
for r, ro in old.items():
    for a, subs in ro["arms"].items():
        for s, x in subs.items():
            y = new[r]["arms"][a][s]
            if x is None or y is None:
                assert x is None and y is None, (r, a, s)
                continue
            for k in ("n", "clusters", "homogeneity", "completeness", "nmi", "cnmi"):
                worst = max(worst, abs(x[k] - y[k]))
print(f"  {sys.argv[2]}: {len(old)} old rungs, max diff {worst:.2e}; new rungs: {sorted(set(new) - set(old))}")
EOF
done
echo TOXICITY_PROBES_DONE
