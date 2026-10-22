#!/usr/bin/env bash
# Dedicated box experiment: full-input encoder, fixed token bias at decoder only.
# Launch once in tmux; stdout is redirected by the launch command documented in HANDOFF_REMOTE.md.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source "$GEOAE_ROOT/configs/sites/local.sh"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
CONFIG=experiments/token_bypass/configs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly.yaml
CKDIR=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly
PROVENANCE=runs/decoderonly_d6144
mkdir -p "$PROVENANCE" logs
complete=0
trap 'rc=$?; if [ "$complete" -eq 1 ] && [ "$rc" -eq 0 ]; then printf "DECODERONLY COMPLETE\n"; else printf "DECODERONLY INCOMPLETE\n"; fi; printf "DECODERONLY EXIT %s\n" "$rc"; exit "$rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
scripts/py - <<'PY'
import torch
from geoae.config import Config
cfg = Config.from_yaml('experiments/token_bypass/configs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly.yaml')
assert cfg.model.token_bias_mode == 'decoder_only'
assert torch.cuda.is_available(), 'The dedicated box experiment requires a GPU'
print('[experiment] decoder_only: z=encoder(x), reconstruction=decoder(z)+b[token]', flush=True)
print('[experiment] GPU:', torch.cuda.get_device_name(0), flush=True)
PY
if [ ! -f "$PROVENANCE/source.tar.gz" ]; then
    tar --exclude=__pycache__ -czf "$PROVENANCE/source.tar.gz" geoae tests/unit/test_decoder_only_bias.py "$CONFIG" "$0"
    sha256sum "$PROVENANCE/source.tar.gz" > "$PROVENANCE/source.sha256"
    printf '%s\n' "${GEOAE_SOURCE_COMMIT:-unspecified}" > "$PROVENANCE/git_commit.txt"
fi
printf '[experiment] source commit: %s\n' "${GEOAE_SOURCE_COMMIT:-unspecified}"
RESUME=()
if compgen -G "$CKDIR/step_*.pt" > /dev/null; then RESUME=(--resume latest); fi
scripts/py -u -m geoae.training.mse --config "$CONFIG" --no_wandb "${RESUME[@]}"
scripts/py - "$CKDIR" "$CONFIG" <<'PY'
import sys
import torch
from pathlib import Path
from geoae.config import Config
from geoae.training.common import resolve_resume
cfg = Config.from_yaml(sys.argv[2])
path = resolve_resume('latest', Path(sys.argv[1]))
ckpt = torch.load(path, map_location='cpu', weights_only=False)
assert ckpt['epoch'] == cfg.train.n_epochs, 'The final training epoch was not saved'
assert ckpt['config']['model']['token_bias_mode'] == 'decoder_only'
print('[experiment] verified final checkpoint:', path.name, 'epoch', ckpt['epoch'], flush=True)
PY
complete=1
