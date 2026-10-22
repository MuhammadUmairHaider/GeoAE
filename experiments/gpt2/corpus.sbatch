#!/bin/bash
# CPU-only frozen corpus; submitted before the GPU extraction job.
#SBATCH --job-name=gpt2-corpus
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/gpt2/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source experiments/gpt2/env.sh
"$GEOAE_PYTHON" -u -m geoae.extraction.corpus \
    --config experiments/gpt2/configs/extract_all.yaml --seed 42 --workers 3 \
    2>&1 | tee -a logs/gpt2/corpus.log
