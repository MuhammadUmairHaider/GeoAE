#!/usr/bin/env bash
# Retire the PRE-2026-09-26 data distribution from the data root, so the round-2
# campaign starts from the sampled dump only.
#
#   bash scripts/delta/clean_slate.sh            # DRY RUN: list what would go
#   bash scripts/delta/clean_slate.sh --delete   # actually delete
#
# WHAT GOES, and why each one is safe to lose:
#
#   activations/activations_diverse_10M          58 GB  the legacy L27 dump: every
#       position 4..255 of ~44k docs from an equal 5-way mix that was 43% code and
#       math. Round 2 trains on activations_sampled_10M instead (full docs to 2048
#       tokens, 64 random positions each, pretraining-like mix). Regenerable in
#       ~30 min of GPU from configs/base/llama3.2-3b_extract.yaml.
#   activations/activations_diverse_10M.raced-*  6.1 GB  a sparse leftover from two
#       extract jobs racing on 2026-09-01. Junk, no meta.json, never usable.
#   activations/activations_diverse_l14_10M      58 GB  the layer-14 twin. The L14
#       track is dormant; regenerable the same way.
#   checkpoints/.../k2000_bnh_b32k_lam1_d6144    24 GB  the k-means++ parent trained
#       ON THE LEGACY DUMP. Not a round-2 arm, and not comparable to round-2
#       checkpoints anyway (different training corpus).
#
# WHAT STAYS, and why it is NOT "old data distribution":
#
#   cache/huggingface             13 GB  the gated meta-llama/Llama-3.2-3B weights
#       plus every eval dataset. Nothing to do with the training dump, and there is
#       no HF token configured here, so a fresh gated download would 401.
#   cache/concept_suite_llama_l27 4.7 GB  the concept caches. They store BASE
#       RESIDUALS for the probe datasets (SST-2, IMDB, POS, NER, ...), never
#       latents, and concept_cache.py reads a checkpoint only for model_name and
#       target_layer — so they are independent of both the dump and the AE. Costs
#       GPU hours to rebuild for no gain.
#       (cache_ids/ DOES have to be rebuilt: the sequence rungs need last_token_id,
#       which the 2026-09-01 caches do not carry. eval_out/run_tokbias.sh's `cache`
#       step does that and symlinks the token rungs from here.)
#   results/, logs/               70 MB   old evidence; the repo's tracked copies
#       are the authoritative ones, but these are too small to be worth touching.
#
# Frees ~146 GB, taking this user's share of the allocation from ~162 GB to ~18 GB.
# The allocation is SHARED: eight users sit on /work/hdd/bbyl, ~654 GB of it not
# ours, against a 1 TB soft quota — which is why round 2 (~257 GB of new dumps and
# checkpoints) needs this space back before it starts.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/delta/env.sh

DELETE=0
[ "${1:-}" = "--delete" ] && DELETE=1

TARGETS=(
  "activations/activations_diverse_10M"
  "activations/activations_diverse_l14_10M"
  "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144"
)
# The raced leftovers carry a job-id suffix, so they are matched by glob.
while IFS= read -r d; do
  [ -n "$d" ] && TARGETS+=("${d#$DELTA_DATA/}")
done < <(find "$DELTA_DATA/activations" -maxdepth 1 -type d -name '*.raced-*' 2>/dev/null)

echo "Data root: $DELTA_DATA"
echo
total_found=0
for rel in "${TARGETS[@]}"; do
  path="$DELTA_DATA/$rel"
  if [ ! -e "$path" ]; then
    printf '  %-9s %s\n' "absent" "$rel"
    continue
  fi
  size=$(du -sh "$path" 2>/dev/null | cut -f1)
  total_found=$((total_found + 1))
  if [ "$DELETE" -eq 1 ]; then
    printf '  %-9s %-8s %s\n' "deleting" "$size" "$rel"
    rm -rf "$path"
  else
    printf '  %-9s %-8s %s\n' "would go" "$size" "$rel"
  fi
done

# Stale lock files from the old extract jobs; harmless but confusing to find later.
if [ "$DELETE" -eq 1 ]; then
  rm -f "$DELTA_DATA"/activations/.activations_diverse_*.lock
fi

echo
if [ "$DELETE" -eq 1 ]; then
  echo "Deleted $total_found item(s)."
  echo "Now: bash scripts/delta/link_data.sh   # drops the links that no longer resolve"
else
  echo "$total_found item(s) matched. Nothing was changed."
  echo "Re-run with --delete to remove them."
fi
echo
du -sh "$DELTA_DATA" 2>/dev/null || true
