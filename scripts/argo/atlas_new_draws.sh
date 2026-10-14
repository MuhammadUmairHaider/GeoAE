#!/bin/bash
# New FineWeb-Atlas class draws for the control table, end to end. Run on the Argo login node, in the repo:
#   bash scripts/argo/atlas_new_draws.sh              the next three unused seeds (first call: 3 4 5, then 6 7 8, ...)
#   bash scripts/argo/atlas_new_draws.sh 3 4 5        these seeds (run again after an interruption: finished parts are skipped)
#   N=5 bash scripts/argo/atlas_new_draws.sh          the next five unused seeds
# Per call: one CPU job freezes the draws and writes their corpora (scripts/argo/atlas_draw_prep.sbatch), then one GPU
# job per (model, seed) starts when it has finished:
#   Llama-3.2-3B, Qwen2.5-3B   scripts/argo/control_tp2_plain.sbatch <arm> atlas_s<seed>   (alpha 0.5, as the paper's Atlas rows)
#   Gemma 3 4B                 scripts/argo/atlas_draw_g4b.sbatch <seed>                   (three-way true positives + SAE top-1)
# MODELS picks the models: default d6144 (Llama only; user, 12 Oct: the other models' Atlas rows stay on draws 0-2);
#   MODELS="d6144 qwen25_l35_d4096 g4b" bash scripts/argo/atlas_new_draws.sh 3 4 5      all three
# PREP_ONLY=1 submits the CPU job only and prints the GPU lines.
# Each draw is 14 content concepts under the rules of the first three (geoae/interp/atlas_classes.RULES); a draw is
# frozen in s<seed>/draw.json before any model sees it and is never redrawn. Report every draw that was run.
# Table over all draws, back on Delta:  bash scripts/delta/atlas_draws_table.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
AD=results/delta/d6144/interventions/atlas_classes
MODELS="${MODELS:-d6144}"
mkdir -p logs
if [ $# -gt 0 ]; then SEEDS=("$@")
else
  last=2; for d in "$AD"/s[0-9]*; do [ -d "$d" ] || continue; n=${d##*/s}; [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt "$last" ] && last=$n; done
  SEEDS=(); for ((i = 1; i <= ${N:-3}; i++)); do SEEDS+=($((last + i))); done
fi
for S in "${SEEDS[@]}"; do [[ "$S" =~ ^[0-9]+$ ]] && [ "$S" -gt 2 ] || { echo "seed '$S': need an integer above 2 (0,1,2 are the original draws)"; exit 2; }; done
echo "[new-draws] seeds: ${SEEDS[*]}   models: $MODELS"
DEP=()
need=0; for S in "${SEEDS[@]}"; do [ -f "$AD/s$S/cfg.json" ] && [ -f "$AD/s$S/data.parquet" ] && [ -f "$AD/s$S/relatedness.json" ] || need=1; done
if [ "$need" = 1 ]; then
  J=$(sbatch --parsable scripts/argo/atlas_draw_prep.sbatch "${SEEDS[@]}")
  echo "[new-draws] draw + corpus job $J (logs/slurm-atlas-draw-prep-$J.out)"
  DEP=(--dependency="afterok:$J")
else echo "[new-draws] draws and corpora exist"; fi
go() { if [ "${PREP_ONLY:-0}" = 1 ]; then echo "  sbatch $*"; else echo "[new-draws] $* -> job $(sbatch --parsable "${DEP[@]}" "$@")"; fi; }
[ "${PREP_ONLY:-0}" = 1 ] && echo "[new-draws] GPU lines (submit after the draw job has finished):"
for S in "${SEEDS[@]}"; do
  for M in $MODELS; do
    case "$M" in
      g4b) go scripts/argo/atlas_draw_g4b.sbatch "$S" ;;
      *)   go --export=ALL,ALPHAS=0.5 --gres=gpu:h100:1 --job-name="tp2-$M-s$S" scripts/argo/control_tp2_plain.sbatch "$M" "atlas_s$S" ;;
    esac
  done
done
