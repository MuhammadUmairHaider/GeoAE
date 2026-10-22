#!/usr/bin/env bash
# sbatch experiments/depth/run.sbatch llama [run.py options...]
# sbatch experiments/depth/run.sbatch gemma4b [run.py options...]
#SBATCH --job-name=geoae-midlayer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source scripts/env.sh
midlayer_arm="${1:?choose llama or gemma4b}"
shift
exec scripts/py experiments/depth/run.py --arm "$midlayer_arm" "$@"
