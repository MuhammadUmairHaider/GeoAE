#!/bin/bash
# New DOCUMENT draws of a control dataset with fixed classes, end to end. Run on the Argo login node, in the repo (the
# FineWeb-Atlas counterpart, which draws new classes, is scripts/argo/atlas_new_draws.sh):
#   bash scripts/argo/control_new_draws.sh news20            the next three unused seeds (first call: 1 2 3, then 4 5 6, ...)
#   bash scripts/argo/control_new_draws.sh news20 1 2 3      these seeds (run again after an interruption: finished parts are skipped)
#   N=5 bash scripts/argo/control_new_draws.sh news20        the next five unused seeds
#   bash scripts/argo/control_new_draws.sh db14              Gemma 3 4B, base / Gemma Scope SAE / SAE top-1 / GeoAE
# Two kinds of dataset:
#   news20, emotion6          frozen local corpora, run on Llama (MODELS default d6144; "d6144 qwen25_l35_d4096" adds Qwen).
#                             One CPU job writes the redrawn corpora (scripts/argo/control_redraw_prep.sbatch), then one
#                             single-H100 job per (model, seed): scripts/argo/control_tp2_plain.sbatch <arm> <name>_s<seed>.
#   db14, biasbios, ag_news   Hub datasets, run on Gemma (three-way true positives incl. the SAE). No corpus job: the run
#                             shuffles the split with the seed (the paper's run is seed 42). One single-H100 job per seed:
#                             scripts/argo/control_redraw_g4b.sbatch <name> <seed> (about an hour).
# The other combinations (Hub datasets on Llama / Qwen, local corpora on Gemma) are not wired.
# PREP_ONLY=1 submits the CPU job only (if any) and prints the GPU lines.
# The original run is draw 0. Report every draw that was run: the table averages each class over its draws
# (bash scripts/delta/atlas_draws_table.sh on Delta).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
NAME="${1:?usage: bash scripts/argo/control_new_draws.sh <news20|emotion6|db14|biasbios|ag_news> [seed ...]}"; shift
case "$NAME" in news20|emotion6) HUB=0 ;; db14|biasbios|ag_news) HUB=1 ;; *) echo "dataset '$NAME': news20, emotion6, db14, biasbios or ag_news"; exit 2 ;; esac
CD=results/delta/control_datasets
G4=results/delta/g4b_l33_d5120/interventions/sae
MODELS="${MODELS:-d6144}"
mkdir -p logs
if [ $# -gt 0 ]; then SEEDS=("$@")
else
  last=0
  if [ "$HUB" = 1 ]; then for f in "$G4/range_intervention_${NAME}"_s[0-9]*_dprime_hzs_tp3.json; do [ -f "$f" ] || continue
      n=${f##*/range_intervention_${NAME}_s}; n=${n%%_*}; [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt "$last" ] && last=$n; done
  else for d in "$CD/${NAME}"_s[0-9]*; do [ -d "$d" ] || continue; n=${d##*_s}; [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt "$last" ] && last=$n; done; fi
  SEEDS=(); for ((i = 1; i <= ${N:-3}; i++)); do SEEDS+=($((last + i))); done
fi
for S in "${SEEDS[@]}"; do [[ "$S" =~ ^[0-9]+$ ]] && [ "$S" -gt 0 ] && [ "$S" != 42 ] || { echo "seed '$S': need an integer above 0 and not 42 (the original draw)"; exit 2; }; done
go() { if [ "${PREP_ONLY:-0}" = 1 ]; then echo "  sbatch $*"; else echo "[new-draws] $* -> job $(sbatch --parsable "${DEP[@]}" "$@")"; fi; }
if [ "$HUB" = 1 ]; then
  DEP=(); echo "[new-draws] $NAME seeds: ${SEEDS[*]}   model: Gemma 3 4B (base, SAE, SAE top-1, GeoAE)"
  for S in "${SEEDS[@]}"; do go --job-name="g4b-${NAME}_s$S" scripts/argo/control_redraw_g4b.sbatch "$NAME" "$S"; done
  exit 0
fi
echo "[new-draws] $NAME seeds: ${SEEDS[*]}   models: $MODELS"
DEP=()
need=0; for S in "${SEEDS[@]}"; do [ -f "$CD/${NAME}_s$S/cfg.json" ] || need=1; done
if [ "$need" = 1 ]; then
  J=$(sbatch --parsable scripts/argo/control_redraw_prep.sbatch "$NAME" "${SEEDS[@]}")
  echo "[new-draws] corpus job $J (logs/slurm-control-redraw-prep-$J.out)"
  DEP=(--dependency="afterok:$J")
else echo "[new-draws] corpora exist"; fi
[ "${PREP_ONLY:-0}" = 1 ] && echo "[new-draws] GPU lines (submit after the corpus job has finished):"
for S in "${SEEDS[@]}"; do
  for M in $MODELS; do
    [ "$M" = g4b ] && { echo "[new-draws] Gemma is not wired for document redraws, skipped"; continue; }
    go --gres=gpu:h100:1 --job-name="tp2-$M-${NAME}_s$S" scripts/argo/control_tp2_plain.sbatch "$M" "${NAME}_s$S"
  done
done
