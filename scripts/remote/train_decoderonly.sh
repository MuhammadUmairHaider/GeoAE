#!/usr/bin/env bash
# Dedicated box experiment: full-input encoder, fixed token bias at decoder only.
# Launch once in tmux; stdout is redirected by the launch command documented in HANDOFF_REMOTE.md.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
CONFIG=configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly.yaml
CKDIR=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly
PROVENANCE=runs/decoderonly_d6144
mkdir -p "$PROVENANCE" logs
trap 'rc=$?; printf "DECODERONLY EXIT %s\n" "$rc"; exit "$rc"' EXIT
scripts/delta/py - <<'PY'
import torch
from geoae.config import Config
cfg = Config.from_yaml('configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly.yaml')
assert cfg.model.token_bias_mode == 'decoder_only'
assert torch.cuda.is_available(), 'The dedicated box experiment requires a GPU'
print('[experiment] decoder_only: z=encoder(x), reconstruction=decoder(z)+b[token]', flush=True)
print('[experiment] GPU:', torch.cuda.get_device_name(0), flush=True)
PY
if [ ! -f "$PROVENANCE/source.tar.gz" ]; then
    tar --exclude=__pycache__ -czf "$PROVENANCE/source.tar.gz" geoae tests/test_decoder_only_bias.py "$CONFIG" "$0"
    sha256sum "$PROVENANCE/source.tar.gz" > "$PROVENANCE/source.sha256"
    printf '%s\n' "${GEOAE_SOURCE_COMMIT:-unspecified}" > "$PROVENANCE/git_commit.txt"
fi
printf '[experiment] source commit: %s\n' "${GEOAE_SOURCE_COMMIT:-unspecified}"
RESUME=()
if compgen -G "$CKDIR/step_*.pt" > /dev/null; then RESUME=(--resume latest); fi
scripts/delta/py -u -m geoae.train --config "$CONFIG" --no_wandb "${RESUME[@]}"
