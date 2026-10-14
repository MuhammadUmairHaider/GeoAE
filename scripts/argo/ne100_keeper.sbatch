#!/bin/bash
# Keeps one checkpoint every 10 epochs of the 12 Oct runs (user: "keep checkpoint at every 10 epochs as well. we might
# have over fitting"). The runs rotate their step files (keep_checkpoints: 4), so this loop hard-links every step file
# that falls on a multiple of 10 epochs to keep_epochNNN.pt in the same folder (same filesystem: no extra space until
# the original is rotated away; rotation and --resume only look at step_*.pt). CPU only.
#   sbatch scripts/argo/ne100_keeper.sbatch            (on the box: nohup bash scripts/argo/ne100_keeper.sbatch <ckpt root> &)
#SBATCH --job-name=ne100-keeper
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
ROOT="${1:-/scratch/dogar/geoae/checkpoints}"
cd "$ROOT" || exit 1
echo "[keeper] watching $ROOT/*/layer*/*_nobal*_e[0-9]* $(date)"
while true; do
  for d in */layer*/*_nobal*_e[0-9]*; do
    [ -d "$d" ] || continue
    case "$d" in qwen*) spe=305 ;; *) spe=289 ;; esac          # steps per epoch (10M-row dumps, batch 32,768)
    for f in "$d"/step_*.pt; do
      [ -e "$f" ] || continue
      s=$((10#$(basename "$f" .pt | cut -d_ -f2))); e=$((s / spe))
      if [ $((s % spe)) -eq 0 ] && [ "$e" -gt 0 ] && [ $((e % 10)) -eq 0 ]; then
        k="$d/keep_epoch$(printf %03d "$e").pt"
        [ -e "$k" ] || { ln "$f" "$k" && echo "[keeper $(date +%H:%M:%S)] $k"; }
      fi
    done
  done
  sleep 60
done
