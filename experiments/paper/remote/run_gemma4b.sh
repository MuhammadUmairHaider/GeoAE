#!/usr/bin/env bash
# Gemma 3 4B arm g4b_l33_d5120 on the box (A100-40GB, one GPU, local disk): extract -> token bias -> train (lane A),
# balanced-k-means + probe caches concurrently with training (lane B), then the eval groups.
# Every step is skip-if-done, so relaunching the same command resumes after a crash/reboot:
#   tmux new -d -s g4b 'cd "$GEOAE_ROOT" && bash experiments/paper/remote/run_gemma4b.sh 2>&1 | tee -a logs/run_gemma4b.log'
# Last lines of the log are "G4B COMPLETE" or "G4B INCOMPLETE" (+ the failed steps) and "G4B EXIT <rc>".
#
#   lane A (serial): extract (experiments/paper/configs/gemma3-4b_extract_sampled_struct5.yaml, float32, layer 33, 10M rows, 102 GB)
#                    -> token-bias table (+ norm_params_layer33.npz) -> train d5120 (50 epochs, --no_wandb, resumes)
#   lane B (starts when the table and norm params exist; shares the GPU with training): g4b_km, then g4b_cache.
#                    Each step waits until nvidia-smi shows >= LANE_B_FREE_GB (default 20) GB free so training is never OOMed.
#   evals (after BOTH lanes): run_delta_evals.sh g4b_l33_d5120 evals interventions steering concept
#                    (NOT geometry/probes, so g4b_probe_acts is not needed). Serial by default; PAR=1 runs `interventions`
#                    concurrently with `evals steering concept` (only when >= EVAL_PAR_FREE_GB free). See report for the reasoning.
# Env knobs: LANE_B_FREE_GB=20  EVAL_PAR_FREE_GB=24  PAR=0   FORCE=1 (passed to eval sheets)
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
source "$GEOAE_ROOT/configs/sites/local.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
PY=scripts/py
CFG_X=experiments/paper/configs/gemma3-4b_extract_sampled_struct5.yaml
CFG_T=experiments/paper/configs/gemma.yaml
ACT=activations_gemma3_4b_sampled_struct5_10M
ARM=g4b_l33_d5120
E=e2e/checkpoints/general/gemma3-4B/layer33
TABLE=$E/token_bias_sampled_struct5.npz
CKDIR=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias
FINAL=$CKDIR/step_0014450.pt
PROV=runs/gemma4b_l33_d5120
LANE_B_FREE_GB="${LANE_B_FREE_GB:-20}"; EVAL_PAR_FREE_GB="${EVAL_PAR_FREE_GB:-24}"; PAR="${PAR:-0}"
mkdir -p "$PROV" logs "$PROV/status"
rm -f "$PROV"/status/*.fail
complete=0
trap 'rc=$?; fails=$(ls "$PROV"/status/*.fail 2>/dev/null | xargs -r -n1 basename | tr "\n" " ");
      if [ "$complete" -eq 1 ] && [ "$rc" -eq 0 ] && [ -z "$fails" ]; then echo "G4B COMPLETE"; else echo "G4B INCOMPLETE ${fails:+(failed: $fails)}"; fi
      echo "G4B EXIT $rc"; exit "$rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log() { echo "[g4b $(date +%m-%d\ %H:%M:%S)] $*"; }
fail() { touch "$PROV/status/$1.fail"; log "FAILED: $1"; }
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
wait_gpu() {   # wait_gpu <GB>: block until the GPU has that much free memory (poll 60 s)
    local need="$1"
    while [ "$(free_gb)" -lt "$need" ]; do log "waiting for >= ${need} GB free GPU memory (free: $(free_gb) GB)"; sleep 60; done
}
nvidia-smi -L || { log "no GPU"; exit 1; }
$PY - <<'PYEOF' || exit 1
import torch
from geoae.config import Config
for p in ("experiments/paper/configs/gemma3-4b_extract_sampled_struct5.yaml",
          "experiments/paper/configs/gemma.yaml"):
    Config.from_yaml(p)
assert torch.cuda.is_available()
print("[g4b] configs load; GPU:", torch.cuda.get_device_name(0), flush=True)
PYEOF

# provenance (once)
if [ ! -f "$PROV/source.tar.gz" ]; then
    tar --exclude=__pycache__ -czf "$PROV/source.tar.gz" geoae experiments/paper/scripts/run_delta_evals.sh experiments/paper/scripts/run_delta_prep.sh \
        "$CFG_X" "$CFG_T" "$0"
    sha256sum "$PROV/source.tar.gz" > "$PROV/source.sha256"
    printf '%s\n' "${GEOAE_SOURCE_COMMIT:-unspecified}" > "$PROV/git_commit.txt"
fi

# ---------------------------------------------------------------- lane A -------------------------------------------
extract_done() {
    $PY - "$ACT" <<'PYEOF'
import json, sys
from pathlib import Path
m = Path(sys.argv[1]) / "meta.json"
ok = m.exists() and (lambda d: d.get("mode") == "sampled" and d.get("n_tokens") == 10_000_000
                     and sorted(d.get("layers", [])) == [33])(json.load(open(m)))
sys.exit(0 if ok else 1)
PYEOF
}
lane_a() {
    # 1. corpus: the 4B corpus is a copy of the 1B one; build_corpus reuses every finished source (streams nothing).
    #    (online: build_corpus only needs the tokenizer, cached; but keep offline so nothing can start a download)
    HF_HUB_OFFLINE=1 $PY -u -m geoae.extraction.corpus --config "$CFG_X" --workers 3 2>&1 | tee -a logs/corpus_gemma3-4b_extract_sampled_struct5.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { fail corpus; return 1; }
    # 2. extraction
    if extract_done; then log "[skip] $ACT/meta.json complete"; else
        log "extraction start"
        HF_HUB_OFFLINE=1 $PY -u -m geoae.extraction.prefix --config "$CFG_X" 2>&1 | tee logs/extract_gemma3-4b_extract_sampled_struct5.log
        extract_done || { fail extract; return 1; }
    fi
    # 3. token-bias table (+ norm_params_layer33.npz, written next to the dump). Local disk: no staging.
    if [ -e "$TABLE" ] && [ -e "$ACT/norm_params_layer33.npz" ]; then log "[skip] token-bias table exists"; else
        mkdir -p "$E"
        HF_HUB_OFFLINE=1 $PY -u -m geoae.token_bias --activations_dir "$ACT" --layer 33 --out "$TABLE" 2>&1 \
            | tee logs/token_bias_${ACT}_l33.log
        [ -e "$TABLE" ] && [ -e "$ACT/norm_params_layer33.npz" ] || { fail token_bias; return 1; }
    fi
    touch "$PROV/status/table.ready"
    # 4. train (resumes from the latest per-epoch checkpoint)
    #    up to 3 attempts: a crash (e.g. lane B squeezing GPU memory) resumes from the last per-epoch checkpoint
    if [ -f "$FINAL" ]; then log "[skip] $FINAL exists"; else
        for attempt in 1 2 3; do
            RESUME=(); compgen -G "$CKDIR/step_*.pt" > /dev/null && RESUME=(--resume latest)
            log "training start (attempt $attempt) ${RESUME[*]:-}"
            HF_HUB_OFFLINE=1 $PY -u -m geoae.training.mse --config "$CFG_T" --no_wandb "${RESUME[@]}" 2>&1 \
                | tee -a logs/gemma3-4b_l33_d5120_train.log
            [ -f "$FINAL" ] && break
            sleep 60
        done
        [ -f "$FINAL" ] || { fail train; return 1; }
    fi
    $PY - "$FINAL" "$CFG_T" <<'PYEOF' || { fail train_verify; return 1; }
import sys, torch
from geoae.config import Config
cfg = Config.from_yaml(sys.argv[2]); c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert c["epoch"] == cfg.train.n_epochs, c["epoch"]
assert c["model_state"].get("centroids_initialized", False) and "tb_table" in c["model_state"]
print("[g4b] verified final checkpoint epoch", c["epoch"], "val_mse", c.get("val_mse"), flush=True)
PYEOF
    touch "$PROV/status/train.done"
}

# ---------------------------------------------------------------- lane B -------------------------------------------
lane_b() {
    until [ -f "$PROV/status/table.ready" ] || [ -f "$PROV/status/corpus.fail" ] || [ -f "$PROV/status/extract.fail" ] \
          || [ -f "$PROV/status/token_bias.fail" ]; do sleep 60; done
    [ -f "$PROV/status/table.ready" ] || { log "lane B: lane A failed before the table; not starting"; return 1; }
    # start only once training holds its GPU memory (first per-epoch checkpoint): the km fit puts its 1.5M x 2560
    # fp32 sample (~15 GB) on the GPU, and grabbing that before training allocates could OOM training instead.
    until compgen -G "$CKDIR/step_*.pt" > /dev/null || [ -f "$PROV/status/train.done" ] \
          || [ -f "$PROV/status/train.fail" ]; do sleep 60; done
    # prep runs online like the 1B box runs (datasets for the concept caches); the model loads from the HF cache.
    wait_gpu "$LANE_B_FREE_GB"
    log "lane B: g4b_km"
    env -u HF_HUB_OFFLINE bash experiments/paper/scripts/run_delta_prep.sh g4b_km 2>&1 | tee -a logs/g4b_prep_km.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || fail g4b_km
    wait_gpu "$LANE_B_FREE_GB"
    log "lane B: g4b_cache"
    env -u HF_HUB_OFFLINE bash experiments/paper/scripts/run_delta_prep.sh g4b_cache 2>&1 | tee -a logs/g4b_prep_cache.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || fail g4b_cache
    touch "$PROV/status/laneB.done"
}

lane_a & PID_A=$!
lane_b & PID_B=$!
wait "$PID_A"; RA=$?
wait "$PID_B"; RB=$?
log "lanes finished: A rc=$RA, B rc=$RB"
[ "$RA" -eq 0 ] || exit 1
[ -f "$FINAL" ] || exit 1

# ---------------------------------------------------------------- evals --------------------------------------------
ev() {   # ev <group...>: one run_delta_evals invocation; online (wikipedia streaming in steer/range, as in gemma_fix2.sh)
    local tag="${*// /_}"
    wait_gpu 12
    log "evals: $*"
    env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE \
        bash experiments/paper/scripts/run_delta_evals.sh "$ARM" "$@" > "logs/evals_${ARM}_${tag}.log" 2>&1 \
        || fail "evals_$tag"
    tail -3 "logs/evals_${ARM}_${tag}.log"
}
if [ "$PAR" = 1 ]; then
    ( ev interventions ) & P1=$!
    sleep 120   # let the first job's model load before checking free memory for the second chain
    ( while [ "$(free_gb)" -lt "$EVAL_PAR_FREE_GB" ]; do sleep 60; done; ev evals; ev steering; ev concept ) & P2=$!
    wait "$P1"; wait "$P2"
else
    ev evals; ev steering; ev concept; ev interventions
fi
complete=1
