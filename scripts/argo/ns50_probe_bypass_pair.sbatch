#!/bin/bash
# Bypass side of the probe table for the 12 Oct 50-epoch no-staging models, without waiting for the no-bypass models:
# token-removed plain k-means vs the bypass AE, same cache, filters and rungs as the four-space run of
# scripts/argo/ns50_evals.sbatch <model> pair (the pair filter only involves these two spaces). CPU only.
#   sbatch scripts/argo/ns50_probe_bypass_pair.sbatch            -> results/delta/<arm>_ns50/evals/probe_dense_subsets_bypasspair.json
#SBATCH --job-name=ns50-probe-byp
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
run() {   # arm cache tokmean-kmeans checkpoint
  local O=results/delta/$1/evals; mkdir -p "$O"
  echo "[probe-byp $(date +%H:%M:%S)] $1"
  "$PY" -u eval_out/probe_dense_subsets.py --cache "$2" --fp_min 6 --pairs tok_plain:geoae --pair_min 10 \
      --arms "tok_plain=km:$3,geoae=ae:$4" --out "$O/probe_dense_subsets_bypasspair.json" > "$O/probe_dense_subsets_bypasspair.log" 2>&1 \
      || { echo "FAILED $1"; tail -3 "$O/probe_dense_subsets_bypasspair.log"; }
}
E=e2e/checkpoints/general; C=checkpoints
run g4b_l33_d5120_ns50 cache/gemma3-4b/l33 $E/gemma3-4B/layer33/delta/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz \
    $C/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias_nobal_nostage_e50/step_0014450.pt
run d6144_ns50 /scratch/dogar/geoae/cache_ids_llama_l27_box $E/llama3.2-3B/layer27/delta/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz \
    $C/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nobal_nostage_e50/step_0014450.pt
run qwen25_l35_d4096_ns50 cache/qwen2.5-3B/layer35 $E/qwen2.5-3B/layer35/delta/plain_kmeans_k2000_dpc_struct5_train10M_tokmean_seed42.npz \
    $C/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias_nobal_nostage_e50/step_0015250.pt
echo "[probe-byp $(date +%H:%M:%S)] PROBE_BYP_DONE"
