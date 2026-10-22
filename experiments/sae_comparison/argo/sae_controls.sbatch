#!/bin/bash
# Controls for the SAE suite (C1 code-space balanced k-means, C2 random-SAE null, C3 top-2000-feature top-1,
# C4 non-sibling null for sibling coding). One job: fit -> probe -> token_structure -> coding null.
#   sbatch --gres=gpu:l40:1 --nodelist=argo76 --mem=200G experiments/sae_comparison/argo/sae_controls.sbatch
# Outputs: results/delta/g4b_l33_d5120/evals/sae_suite/controls/{fit_summary.json,counts_*.npz,kmeans_*.npz,
#          probe.json,token_structure.json,coding_null.json}.  Set SKIP_FIT=1 to reuse an existing fit.
#SBATCH --job-name=sae-controls
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33; K=2000
E=e2e/checkpoints/general/gemma3-4B/layer$LAYER
B=checkpoints/gemma3-4B/layer$LAYER
TB=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
KM=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
KMT=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
CACHE_IDS=cache/gemma3-4b/l$LAYER
ACT="${ACT_DIR:-activations_gemma3_4b_sampled_struct5_10M}"
SC="$DELTA_DATA/cache/gemma_scope_compare"
OUT=results/delta/$ARM/evals/sae_suite/controls
TRAIN=results/delta/$ARM/evals/gemma_scope_compare/matched_topk/training
mkdir -p "$OUT"

declare -A SPEC=(
  [bt10k]="matched:$TRAIN/batch_topk_10240/last.pt"   [bt20k]="matched:$TRAIN/batch_topk_20480/last.pt"
  [mat10k]="matched:$TRAIN/matryoshka_10240/last.pt"  [mat20k]="matched:$TRAIN/matryoshka_20480/last.pt"
  [gs16k]="scope:16k"                                  [gs262k]="scope:262k" )
NAMES="bt10k bt20k mat10k mat20k gs16k gs262k"
join() { local o=""; for x in "$@"; do o="${o:+$o,}$x"; done; echo "$o"; }

for f in "$TB" "$KM" "$KMT" "$ACT/layer_$LAYER.npy" "$CACHE_IDS" "$SC/range" "$SC/atlas" "$TRAIN"/{batch_topk,matryoshka}_{10240,20480}/last.pt; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
RC=0
run() { local log="$1"; shift; echo "[sae-controls $(date +%H:%M:%S)] $*"; "$@" 2>&1 | tee "logs/$log.log"; local r=${PIPESTATUS[0]}
        [ "$r" -eq 0 ] || { echo "[sae-controls] FAILED rc=$r: $log"; RC=1; }; }

ALL=(); for n in $NAMES; do ALL+=("$n=${SPEC[$n]}"); done
# --- fit: counts for all six (C3), code-space balanced k-means where the dense codes fit (C1) ---
[ -n "${SKIP_FIT:-}" ] || run sae_controls_fit "$PY" -u geoae/baselines/sae_controls_fit.py --saes "$(join "${ALL[@]}")" \
    --activations_dir "$ACT" --layer "$LAYER" --epochs 40 --out_dir "$OUT"   # 40 ep x ~49 steps: reinit_every=1000 fires

# --- control arms ---
CTL=("rand10k=random:10240:0:${SPEC[bt10k]#matched:}" "rand20k=random:20480:0:${SPEC[bt20k]#matched:}")
KMA=()
for n in $NAMES; do
    [ -f "$OUT/counts_$n.npz" ] && CTL+=("${n}_top2000=${SPEC[$n]}#top2000=$OUT/counts_$n.npz")
    [ -f "$OUT/kmeans_$n.npz" ] && KMA+=("${n}_km=${SPEC[$n]}@$OUT/kmeans_$n.npz")
done
echo "[sae-controls] top-1 control arms: ${CTL[*]}"; echo "[sae-controls] code-kmeans arms: ${KMA[*]:-none}"

run sae_controls_probe "$PY" -u geoae/evaluation/chance_corrected_probe.py --cache "$CACHE_IDS" \
    --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" \
    --saes "$(join "${CTL[@]}")" --sae_kmeans "$(join "${KMA[@]:-}")" \
    --atlas_last "" --out "$OUT/probe.json"
run sae_controls_tokstruct "$PY" -u geoae/evaluation/token_structure.py \
    --arms "base=$KM,base_tokmean=$KMT,bypass=$TB" \
    --saes "$(join "${CTL[@]}")" --sae_kmeans "$(join "${KMA[@]:-}")" \
    --activations_dir "$ACT" --layer "$LAYER" --out "$OUT/token_structure.json"

# --- C4: non-sibling null for sibling coding (same cached rows as the head-to-head) ---
run sae_controls_coding_null "$PY" -u experiments/paper/analysis/sae_coding_null.py --cache_dir "$SC" \
    --saes "$(join "${ALL[@]}")" --n_null 200 --out "$OUT/coding_null.json"
[ "$RC" -eq 0 ] && echo "SAE CONTROLS COMPLETE" || { echo "SAE CONTROLS INCOMPLETE"; exit 1; }
