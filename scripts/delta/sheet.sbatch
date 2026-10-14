#!/bin/bash
# Run one eval_out command sheet on one GPU:
#
#   sbatch scripts/delta/sheet.sbatch eval_out/run_tokbias_d12288.sh evals
#   sbatch scripts/delta/sheet.sbatch eval_out/run_tokbias_interventions.sh range_db14
#   sbatch --mem=128g --time=8:00:00 scripts/delta/sheet.sbatch eval_out/run_tokbias.sh evals
#
# The sheets are long serial pipelines with their own skip/FORCE logic, so one
# sheet invocation maps onto one job: a job that hits its walltime is resubmitted
# with the same line and picks up at the first unfinished step.
#
# MEMORY. The default 96g suits d6144. `clustering_quality` holds
# n_sample x latent_dim float32 latents in RAM — 1M x 6144 = 25 GB, but
# 1M x 12288 = 49 GB — so for any d12288 sheet that runs the `cq` step, ask for
# --mem=128g on the sbatch line (the d12288 sheet's own cq step warns under
# 60 GB free and suggests dropping --n_sample to 500000).
#
# The judge steps (cjudge) call an external API and need OPENROUTER_API_KEY in
# .env at the repo root; they are cheap but are NOT GPU work, so prefer running
# those on a CPU account.
#SBATCH --job-name=geoae-sheet
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
SHEET="${1:?usage: sbatch sheet.sbatch <eval_out/run_*.sh> [steps...]}"; shift
source scripts/delta/env.sh
[ -f "$SHEET" ] || { echo "no such sheet: $SHEET"; exit 1; }
mkdir -p logs
echo "[sheet] $SHEET $*"
bash "$SHEET" "$@"
