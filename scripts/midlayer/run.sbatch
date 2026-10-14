#!/usr/bin/env bash
# sbatch scripts/midlayer/run.sbatch llama [run.py options...]
# sbatch scripts/midlayer/run.sbatch gemma4b [run.py options...]
#SBATCH --job-name=geoae-midlayer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=96g
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
source scripts/delta/env.sh
midlayer_arm="${1:?choose llama or gemma4b}"
shift
exec scripts/delta/py scripts/midlayer/run.py --arm "$midlayer_arm" "$@"
