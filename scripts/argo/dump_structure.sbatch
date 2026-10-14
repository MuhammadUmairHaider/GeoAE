#!/bin/bash
# What clusters encode on held-out dump rows (current token / position bucket / source domain / document; raw h / c / NMI,
# >5-row fixed-point filter): mid layers (Llama L14, Gemma 4B L17; six spaces incl. latent plain re-fits) vs the
# paper's last-layer bypass AEs (Llama L27 on activations_sampled_10M, Gemma 4B L33 on activations_gemma3_4b_sampled_struct5_10M).
#   sbatch --gres=gpu:l40:1 scripts/argo/dump_structure.sbatch
#SBATCH --job-name=dump-structure
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=.
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
A=/scratch/dogar/geoae/activations
OUT=results/midlast_ablation_20261006/dump_structure
mkdir -p "$OUT"
for M in llama gemma; do
  case $M in
    llama) L=14; MM=llama3.2-3B; DUMP=$A/activations_llama3_3b_l14_sampled_struct5_train10M; D=6144 ;;
    gemma) L=17; MM=gemma3-4B;   DUMP=$A/activations_gemma3_4b_l17_sampled_struct5_train10M; D=5120 ;;
  esac
  E=e2e/checkpoints/general/$MM/layer$L/delta; CK=checkpoints/$MM/layer$L
  BYP=$CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M_tokbias/step_0015250.pt
  NOB=$CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M/step_0015250.pt
  ARMS="res_plain=km:$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_seed42.npz,tok_plain=km:$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_tokmean_seed42.npz"
  ARMS="$ARMS,geoae=ae:$BYP,geoae_lplain=aekm:$BYP+$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_latent_bypass_seed42.npz"
  ARMS="$ARMS,nobypass=ae:$NOB,nobypass_lplain=aekm:$NOB+$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_latent_nobypass_seed42.npz"
  "$PY" -u eval_out/dump_structure.py --activations_dir "$DUMP" --layer $L --arms "$ARMS" --out "$OUT/${MM}_layer$L.json" \
      || echo "[warn] $MM L$L failed"
done
E=e2e/checkpoints/general/llama3.2-3B/layer27/delta
"$PY" -u eval_out/dump_structure.py --activations_dir $A/activations_sampled_10M --layer 27 \
    --arms "res_plain=km:$E/plain_kmeans_k2000_density_peaks_sampled.npz,tok_plain=km:$E/plain_kmeans_k2000_density_peaks_sampled_tokmean.npz,geoae=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt" \
    --out "$OUT/llama3.2-3B_layer27.json" || echo "[warn] L27 failed"
E=e2e/checkpoints/general/gemma3-4B/layer33/delta; B33=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
"$PY" -u eval_out/dump_structure.py --activations_dir $A/activations_gemma3_4b_sampled_struct5_10M --layer 33 \
    --arms "res_plain=km:$E/plain_kmeans_k2000_dpc_sampled_struct5.npz,tok_plain=km:$E/plain_kmeans_k2000_dpc_sampled_struct5_tokmean.npz,geoae=ae:$B33,geoae_lplain=aekm:$B33+$E/plain_kmeans_k2000_dpc_sampled_struct5_latent_bypass.npz" \
    --out "$OUT/gemma3-4B_layer33.json" || echo "[warn] G33 failed"
echo "[dump-structure] done -> $OUT"
