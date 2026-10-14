#!/bin/bash
# Evaluation sheet of the 12 Oct NO-STAGING models (no balancing, 100 epochs; configs/nobal_nostage_e50/, checkpoints ..._nobal_nostage_e50), with
# the commands that produced the paper's files for the old models; only the checkpoints and the output folders differ.
#   sbatch [--mem=260G] scripts/argo/ns50_evals.sbatch <llama|gemma|qwen> <bypass|nobypass|pair> [steps]
#     bypass    cq token mmlu ppl jigsaw atlas     -> results/delta/<arm>_ns50/evals/        (needs the bypass checkpoint)
#     nobypass  cq token mmlu ppl atlas            -> results/delta/<nobypass arm>_ns50/evals/
#     pair      probes                             -> results/delta/<nobypass arm>_ns50/evals/probe_dense_subsets.json
#                                                     (four spaces: needs BOTH checkpoints)
#   steps (optional, comma list) restricts the sheet, e.g. `... gemma bypass mmlu,ppl`. Finished outputs are skipped.
# Files and their paper counterparts:
#   cq_std.json            clustering_quality, 1M rows, seed 0, baseline = balanced k-means      (Table 1; paper_geometry_table.py)
#   token_structure.json   single-token clusters etc. against balanced + plain k-means            (ablation table)
#   mmlu.json, splice_ppl.json                                                                    (Table 1)
#   probe_jigsaw_bypass.json   toxicity rungs, token-removed plain k-means vs bypass AE, no anchor exclusion
#   atlas_lastword/assign_<which>.npz + nmi_pair10_<which>.json   last-word Atlas, pair filter (clusters >= 10 rows),
#        bypass vs plain token-removed k-means / no-bypass vs plain residual k-means; gemma bypass also vs Gemma Scope
#   probe_dense_subsets.json   res_plain:nobypass, tok_plain:geoae, pair_min 10                   (latex_pair_tables.py)
# GPU: any (H100 or L40); the dump is staged into /dev/shm for cq and token (GPFS has no page cache).
#SBATCH --job-name=ns50-evals
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=8
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
MODEL="${1:?model}"; WHICH="${2:?bypass|nobypass|pair}"; STEPS="${3:-all}"
META=/scratch/dogar/geoae/atlas_concept_meta.json
case "$MODEL" in
  llama) LAYER=27; ACT=activations_sampled_10M; E=e2e/checkpoints/general/llama3.2-3B/layer27/delta
         KM=$E/balanced_kmeans_k2000_dpc_sampled.npz; KMT=$E/balanced_kmeans_k2000_dpc_sampled_tokmean.npz
         PKM=$E/plain_kmeans_k2000_dpc_sampled_seed42.npz; PKMT=$E/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz
         C=checkpoints/llama3.2-3B/layer27; AE=$C/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nobal_nostage_e50/step_0014450.pt
         NB=$C/k2000_bnh_b32k_lam1_d6144_dpc_sampled_nobal_nostage_e50/step_0014450.pt
         CACHE=/scratch/dogar/geoae/cache_ids_llama_l27_box; ATL=/scratch/dogar/geoae/atlas_l27_lastword   # the box cache the paper tables used
         ABASE=results/delta/d6144/evals/atlas_lastword/assign_fourway.npz; SAEA=""
         RB=results/delta/d6144_ns50; RN=results/delta/d6144_nobypass_ns50 ;;
  gemma) LAYER=33; ACT=activations_gemma3_4b_sampled_struct5_10M; E=e2e/checkpoints/general/gemma3-4B/layer33/delta
         KM=$E/balanced_kmeans_k2000_dpc_sampled_struct5.npz; KMT=$E/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz
         PKM=$E/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz; PKMT=$E/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz
         C=checkpoints/gemma3-4B/layer33; AE=$C/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias_nobal_nostage_e50/step_0014450.pt
         NB=$C/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_nobypass_nobal_nostage_e50/step_0014450.pt
         CACHE=cache/gemma3-4b/l33; ATL=/scratch/dogar/geoae/atlas_g4b_l33_lastword
         ABASE=results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/assign_fourway.npz
         SAEA=results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/assign_gs16k_plain.npz
         RB=results/delta/g4b_l33_d5120_ns50; RN=results/delta/g4b_l33_nobypass_ns50 ;;
  qwen)  LAYER=35; ACT=activations_qwen2_5_3b_l35_struct5_train10M; E=e2e/checkpoints/general/qwen2.5-3B/layer35/delta
         KM=$E/balanced_kmeans_k2000_dpc_struct5_train10M.npz; KMT=$E/balanced_kmeans_k2000_dpc_struct5_train10M_tokmean.npz
         PKM=$E/plain_kmeans_k2000_dpc_struct5_train10M_seed42.npz; PKMT=$E/plain_kmeans_k2000_dpc_struct5_train10M_tokmean_seed42.npz
         C=checkpoints/qwen2.5-3B/layer35; AE=$C/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias_nobal_nostage_e50/step_0015250.pt
         NB=$C/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_nobal_nostage_e50/step_0015250.pt
         CACHE=cache/qwen2.5-3B/layer35; ATL=/scratch/dogar/geoae/atlas_qwen25_l35_lastword
         ABASE=results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table/atlas_lastword/assign.npz; SAEA=""
         RB=results/delta/qwen25_l35_d4096_ns50; RN=results/delta/qwen25_l35_nobypass_ns50 ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
case "$WHICH" in
  bypass)   CK=$AE; R=$RB; NAME=bypass;   ARMN=bypass_ns50;   PLAIN=plain42_tokmean; ALL="cq token mmlu ppl jigsaw atlas" ;;
  nobypass) CK=$NB; R=$RN; NAME=nobypass; ARMN=nobypass_ns50; PLAIN=plain42;         ALL="cq token mmlu ppl atlas" ;;
  pair)     CK=$AE; R=$RN; NAME=pair; ALL="probes"; [ -f "$NB" ] || { echo "[abort] missing $NB"; exit 1; } ;;
  *) echo "unknown sheet $WHICH"; exit 2 ;;
esac
[ "$STEPS" = all ] && STEPS="$ALL" || STEPS="${STEPS//,/ }"
[ -f "$CK" ] || { echo "[abort] missing $CK"; exit 1; }
O=$R/evals; mkdir -p "$O/atlas_lastword" "$R/logs" logs
echo "[ns50 $MODEL $WHICH] steps: $STEPS | $CK | host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
want() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
rc=0
step() {      # name outfile command...   (skipped when the output exists)
  local name=$1 out=$2; shift 2
  if [ -s "$out" ]; then echo "[ns50 $(date +%H:%M:%S)] skip $name (have $out)"; return 0; fi
  echo "[ns50 $(date +%H:%M:%S)] start $name"
  "$@" > "$R/logs/$name.log" 2>&1; local r=$?
  echo "[ns50 $(date +%H:%M:%S)] done $name rc=$r"; [ $r -ne 0 ] && { rc=1; tail -4 "$R/logs/$name.log"; }
  return 0
}
if want cq || want token; then
  SHM="/dev/shm/geoae_${SLURM_JOB_ID:-$$}"; trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
  A="$SHM/$ACT"; mkdir -p "$A"; echo "[stage] $ACT -> $A"; time cp -rL "$ACT/." "$A/"
fi
want cq    && step cq_std "$O/cq_std.json" "$PY" -u -m geoae.interp.clustering_quality --baseline "$KM" --checkpoints "$CK" \
                 --names base "$NAME" --activations_dir "$A" --layer $LAYER --n_sample 1000000 --seed 0 --out "$O/cq_std.json"
want token && step token_structure "$O/token_structure.json" "$PY" -u eval_out/token_structure.py \
                 --arms "base=$KM,base_tokmean=$KMT,plain=$PKM,plain_tokmean=$PKMT,$NAME=$CK" --activations_dir "$A" --layer $LAYER --out "$O/token_structure.json"
[ -n "${SHM:-}" ] && rm -rf "$SHM"
want mmlu  && step mmlu "$O/mmlu.json" "$PY" -u -m geoae.interp.mmlu_splice --checkpoint "$CK" --out "$O/mmlu.json"
want ppl   && step splice_ppl "$O/splice_ppl.json" "$PY" -u -m geoae.interp.splice_ppl --ae "$O/splice_ppl.json=$CK"
if want jigsaw; then
  if [ -f "$CACHE/toxicity.npz" ]; then
    step probe_jigsaw_bypass "$O/probe_jigsaw_bypass.json" "$PY" -u eval_out/probe_dense_subsets.py --cache "$CACHE" --fp_min 6 \
        --rungs toxicity,toxicity_pattern --no_anchor_exclusion --arms "tok_plain=km:$PKMT,geoae=ae:$CK" --out "$O/probe_jigsaw_bypass.json"
  else echo "[ne100] no $CACHE/toxicity.npz: jigsaw skipped"; fi
fi
if want atlas; then
  step atlas_assign "$O/atlas_lastword/assign_$NAME.npz" "$PY" -u eval_out/atlas_assign_extra.py --cache "$ATL" --models "$ARMN=$CK" \
      --out "$O/atlas_lastword/assign_$NAME.npz"
  step atlas_pair10 "$O/atlas_lastword/nmi_pair10_$NAME.json" "$PY" -u eval_out/atlas_nmi_filtered.py \
      --assign "$ABASE,$O/atlas_lastword/assign_$NAME.npz" --chunks "$ATL/chunks.npz" --concepts "$META" \
      --arms "$PLAIN,$ARMN" --ref "$ARMN" --min_rows 10 --n_boot 300 --seed 0 --out "$O/atlas_lastword/nmi_pair10_$NAME.json"
  if [ "$WHICH" = bypass ] && [ -n "$SAEA" ]; then
    step atlas_pair10_sae "$O/atlas_lastword/nmi_pair10_gs16k_plain.json" "$PY" -u eval_out/atlas_nmi_filtered.py \
        --assign "$SAEA,$O/atlas_lastword/assign_$NAME.npz" --chunks "$ATL/chunks.npz" --concepts "$META" \
        --arms "$ARMN,gs16k_plain" --ref "$ARMN" --min_rows 10 --n_boot 300 --seed 0 --out "$O/atlas_lastword/nmi_pair10_gs16k_plain.json"
  fi
fi
want probes && step probe_dense_subsets "$O/probe_dense_subsets.json" "$PY" -u eval_out/probe_dense_subsets.py --cache "$CACHE" --fp_min 6 \
                 --pairs res_plain:nobypass,tok_plain:geoae --pair_min 10 \
                 --arms "res_plain=km:$PKM,tok_plain=km:$PKMT,geoae=ae:$AE,nobypass=ae:$NB" --out "$O/probe_dense_subsets.json"
echo "[ns50 $MODEL $WHICH] finished rc=$rc $(date)"
exit "$rc"
