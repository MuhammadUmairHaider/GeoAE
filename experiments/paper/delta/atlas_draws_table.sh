#!/bin/bash
# Control table over ALL draws that have finished: FineWeb-Atlas class draws (the original three +
# experiments/paper/argo/atlas_new_draws.sh) and document redraws of 20NG / Emotion (experiments/paper/argo/control_new_draws.sh; each class is
# averaged over its draws and counts once) and of DBpedia-14 / Bias in Bios / AG News on Gemma (same script).
# Run on the Delta login node that holds the Argo connection, in the repo:
#   bash experiments/paper/delta/atlas_draws_table.sh            copy the Atlas draw results from Argo, then print the table
#   SYNC=0 bash experiments/paper/delta/atlas_draws_table.sh     table from the files already here
# A draw enters when Llama-3.2-3B has finished it; Gemma and Qwen are scored on the draws they have (0-2 unless
# experiments/paper/argo/atlas_new_draws.sh was run with MODELS=... for them), so their rows repeat the paper's numbers.
# Writes results/delta/control_plain_tp_a05_alldraws.{tex,md,json}; the paper's files (control_plain_tp_a05.*, draws 0-2)
# are not touched.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "$GEOAE_ROOT"
REMOTE="${REMOTE:-argo:/home/dogar/projects/def-hsajjad/dogar/GeoAE/repo}"
AD=results/delta/d6144/interventions/atlas_classes
T2=interventions/tp2; T3=results/delta/g4b_l33_d5120/interventions/sae
if [ "${SYNC:-1}" = 1 ]; then
  rsync -a --include='s[0-9]*/' --include='s[0-9]*/**' --exclude='*' "$REMOTE/$AD/" "$AD/"
  for A in d6144 qwen25_l35_d4096; do mkdir -p "results/delta/$A/$T2"
    rsync -a --include='*atlas_s*' --include='*news20_s*' --include='*emotion6_s*' --exclude='*' "$REMOTE/results/delta/$A/$T2/" "results/delta/$A/$T2/"; done
  rsync -a --include='*_s[0-9]*/' --include='*_s[0-9]*/**' --exclude='*' "$REMOTE/results/delta/control_datasets/" results/delta/control_datasets/
  rsync -a --include='*atlas_s*_tp3.*' --include='*_s[0-9]*_dprime_hzs_tp3.*' --include='*_s[0-9]*_dprime_s1_tp3.*' --exclude='*' "$REMOTE/$T3/" "$T3/"
fi
ok() { [ -f "$1" ] && grep -q '"summary"' "$1"; }
DRAWS=()
for d in $(ls -d "$AD"/s[0-9]* | sort -V); do S=${d##*/s}
  if ok "results/delta/d6144/$T2/range_intervention_atlas_s${S}_dprime_hz_tp2.json"; then DRAWS+=("atlas_s$S")
  else echo "[draws-table] draw $S not finished on Llama, left out"; fi
done
# document redraws: a draw of <dataset> enters when Llama or Gemma (three-way and top-1 runs) has finished it
LIST=""
for B in db14 biasbios ag_news emotion6 news20; do LIST+="${LIST:+,}$B"
  for S in $(ls "results/delta/d6144/$T2" "$T3" 2>/dev/null | sed -n "s/^range_intervention_${B}_s\([0-9]*\)_dprime_.*json$/\1/p" | sort -un); do
    if ok "results/delta/d6144/$T2/range_intervention_${B}_s${S}_dprime_hz_tp2.json" \
       || { ok "$T3/range_intervention_${B}_s${S}_dprime_hzs_tp3.json" && ok "$T3/range_intervention_${B}_s${S}_dprime_s1_tp3.json"; }
    then LIST+=",${B}_s$S"; else echo "[draws-table] $B draw $S not finished, left out"; fi
  done
done
LIST+=",$(IFS=,; echo "${DRAWS[*]}")"
echo "[draws-table] datasets: $LIST"
O=results/delta/control_plain_tp_a05_alldraws
python3 experiments/paper/analysis/paper_control_plain_table.py --datasets "$LIST" \
    --stats "$O.md" --json "$O.json" > "$O.tex" 2> /dev/null
grep "GeoAE − Base" "$O.md"
grep -E "^- .*(atlas_s|_s[0-9])" "$O.md" | cut -d'|' -f1-4
echo "[draws-table] wrote $O.{tex,md,json}"
