#!/bin/bash
# What do the K = 4,000 / 8,000 Gemma 4B AEs pay for their sharper neighbourhoods? Same tools as the K = 2,000 evals,
# against plain residual k-means at matched K (density peaks seed 0, Lloyd). Outputs results/delta/g4b_l33_d5120/evals/kcost/
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
source configs/sites/local.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
PY=.venv/bin/python
E=e2e/checkpoints/general/gemma3-4B/layer33/delta
ACT=activations_gemma3_4b_sampled_struct5_10M
KM=$E/balanced_kmeans_k2000_dpc_sampled_struct5.npz; KMT=$E/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz
B=checkpoints/gemma3-4B/layer33
T2=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
T4=$B/k4000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
T8=$B/k8000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
P2=$E/plain_kmeans_k2000_dpc_sampled_struct5.npz; P4=$E/plain_kmeans_k4000_dpc_sampled_struct5.npz; P8=$E/plain_kmeans_k8000_dpc_sampled_struct5.npz
O=results/delta/g4b_l33_d5120/evals/kcost; mkdir -p $O
step() { local name=$1; shift; echo "[kcost $(date +%H:%M:%S)] start $name"; "$@" > $O/$name.log 2>&1; echo "[kcost $(date +%H:%M:%S)] done $name rc=$?"; }
for k in 4000 8000; do
  [ -f $E/plain_kmeans_k${k}_dpc_sampled_struct5.npz ] || step plain_k$k $PY -u experiments/paper/analysis/paper_plain_kmeans_grid.py --activations_dir $ACT --layer 33 --km $KM --kmt $KMT --k $k --spaces residual --out $O/plain_kmeans_k$k.json
done
step token_structure $PY -u geoae/evaluation/token_structure.py --arms "plain=$P2,bypass=$T2,plain4k=$P4,bypass4k=$T4,plain8k=$P8,bypass8k=$T8" --activations_dir $ACT --layer 33 --out $O/token_structure.json
step cq_k4000 $PY -u -m geoae.evaluation.geometry --baseline $P4 --checkpoints $T4 --names plain4k bypass4k --activations_dir $ACT --layer 33 --n_sample 1000000 --seed 0 --out $O/cq_k4000.json
step cq_k8000 $PY -u -m geoae.evaluation.geometry --baseline $P8 --checkpoints $T8 --names plain8k bypass8k --activations_dir $ACT --layer 33 --n_sample 1000000 --seed 0 --out $O/cq_k8000.json
step probe $PY -u geoae/evaluation/chance_corrected_probe.py --cache cache/gemma3-4b/l33 --models "bypass=$T2,bypass4k=$T4,bypass8k=$T8" --baselines "plain=$P2,plain4k=$P4,plain8k=$P8" --atlas_last "" --out $O/probe.json
step mmlu_k4000 $PY -u -m geoae.evaluation.mmlu --checkpoint $T4 --out $O/mmlu_k4000.json
step mmlu_k8000 $PY -u -m geoae.evaluation.mmlu --checkpoint $T8 --out $O/mmlu_k8000.json
step splice_ppl $PY -u -m geoae.evaluation.perplexity --ae $O/splice_ppl_k2000.json=$T2 --ae $O/splice_ppl_k4000.json=$T4 --ae $O/splice_ppl_k8000.json=$T8
echo "[kcost $(date +%H:%M:%S)] KCOST_DONE"
