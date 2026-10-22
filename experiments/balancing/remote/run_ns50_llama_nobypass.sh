#!/bin/bash
# 12 Oct: Llama L27 no-bypass AE, no balancing, no staging, 50 epochs (configs/nobal_nostage_e50/llama_l27_nobypass.yaml).
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"; source configs/sites/local.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
C=configs/nobal_nostage_e50/llama_l27_nobypass.yaml
D=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_nobal_nostage_e50
R=(); compgen -G "$D/step_*.pt" >/dev/null && R=(--resume latest)
scripts/py -u -m geoae.training.mse --config $C --no_wandb "${R[@]}"
echo "NS50_EXIT $? $(date)"
