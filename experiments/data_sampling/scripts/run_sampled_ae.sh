#!/usr/bin/env bash
# Overnight chain: freeze corpus -> extract sampled-position L27 dump -> train an AE.
#
#   ./experiments/data_sampling/scripts/run_sampled_ae.sh                # WIDTH=6144 (default), ~9 h end to end
#   WIDTH=12288 ./experiments/data_sampling/scripts/run_sampled_ae.sh    # ~20 h; reuses corpus + dump if present
#
#   step     what                                             time        disk
#   corpus   geoae.extraction.corpus  -> corpus/sampled_v1/          ~40 min     ~1 GB
#   extract  geoae.extraction.prefix       -> activations_sampled_10M/    ~1.5 h      61 GB
#   train    geoae.training.mse d6144   -> checkpoints/.../_sampled/   ~6-7 h      ~20 GB
#            geoae.training.mse d12288                                 ~17 h       ~40 GB
#
# RE-RUNNABLE. Each step is skipped when its output is already complete, so after a
# crash just start the script again:
#   corpus  : build_corpus is incremental (reuses finished source files)
#   extract : skipped if activations_sampled_10M/meta.json says mode=sampled, 10M rows
#   train   : resumes from the newest readable step_*.pt (--resume latest)
# A failing step stops the chain (set -e); its log is under logs/.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$GEOAE_ROOT/scripts/env.sh"

PY=scripts/py
EXTRACT_CFG=experiments/paper/configs/llama3.2-3b_extract_sampled.yaml
WIDTH=${WIDTH:-6144}
case "$WIDTH" in 6144|12288) ;; *) echo "WIDTH must be 6144 or 12288, got $WIDTH"; exit 1;; esac
if [ "$WIDTH" = 6144 ]; then
  TRAIN_CFG=experiments/paper/configs/llama_no_bypass.yaml
else
  TRAIN_CFG=experiments/width/configs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d12288_dpc_sampled.yaml
fi
ACTS=activations_sampled_10M
CKPT=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d${WIDTH}_dpc_sampled
mkdir -p logs

ts() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
say() { echo "[$(ts)] $*"; }

# ---- preflight ------------------------------------------------------------
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
ckpt_gb=$([ "$WIDTH" = 12288 ] && echo 45 || echo 25)
need_gb=$((62 + ckpt_gb))
[ -f "$ACTS/meta.json" ] && need_gb=$ckpt_gb
if [ "$free_gb" -lt "$need_gb" ]; then
  say "ABORT: ${free_gb} GB free, need ~${need_gb} GB"; exit 1
fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -gt 0 ]; then
  say "WARNING: $busy process(es) already on the GPU:"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
  say "continuing in 30 s (Ctrl-C to abort)"; sleep 30
fi
say "start: WIDTH=$WIDTH, ${free_gb} GB free, config $TRAIN_CFG"

# ---- 1. corpus ------------------------------------------------------------
say "step 1/3: build corpus"
$PY -u -m geoae.extraction.corpus --config "$EXTRACT_CFG" --workers 3 2>&1 | tee -a logs/corpus_sampled_v1.log
say "corpus done"

# ---- 2. extract -------------------------------------------------------------
extract_done() {
  $PY - "$ACTS" <<'EOF'
import json, sys
from pathlib import Path
m = Path(sys.argv[1]) / "meta.json"
ok = m.exists() and (lambda d: d.get("mode") == "sampled" and d.get("n_tokens") == 10_000_000)(json.load(open(m)))
sys.exit(0 if ok else 1)
EOF
}
if extract_done; then
  say "step 2/3: extract — SKIP ($ACTS/meta.json complete)"
else
  say "step 2/3: extract -> $ACTS"
  $PY -u -m geoae.extraction.prefix --config "$EXTRACT_CFG" 2>&1 | tee logs/extract_sampled_10M.log
  extract_done || { say "ABORT: extraction finished but meta.json is not complete"; exit 1; }
  say "extract done"
fi

# ---- 3. train ---------------------------------------------------------------
resume=()
if ls "$CKPT"/step_*.pt >/dev/null 2>&1; then
  resume=(--resume latest)
  say "step 3/3: train — RESUMING from newest checkpoint in $CKPT"
else
  say "step 3/3: train d${WIDTH} on $ACTS"
fi
$PY -u -m geoae.training.mse --config "$TRAIN_CFG" "${resume[@]}" 2>&1 | tee -a logs/llama_l27_d${WIDTH}_dpc_sampled.log
say "ALL DONE. Evaluate $CKPT/step_0014200.pt (epoch 50)."
