#!/usr/bin/env bash
# Pack the artifacts the interpretation work needs into ONE portable kit (run on Delta, login node is fine:
# copy + sha256 only). Everything else (code, results, tests) travels by git.
#
#   bash experiments/token_bypass/portable/pack_kit.sh                 # -> $DELTA_DATA/portable/geoae_kit_d6144{/,.tar}
#
# Kit = the d6144 Llama-3.2-3B L27 token-bypass AE (epoch 50), its base (balanced k-means, K=2000, same
# dump) and token-mean control codebooks, the per-token mean table, the dump's meta.json + norm params, and
# the frozen corpus (held-out docs for closest_tokens --sampled_from). ~1.4 GB. Paths inside the kit are the
# repo-relative paths the sheets expect; experiments/token_bypass/portable/fetch_kit.sh puts them back and checks sha256.
# meta.json is copied with corpus_dir rewritten to the RELATIVE "corpus/sampled_v1" (the original is an
# absolute Delta path).
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source "$GEOAE_ROOT/scripts/env.sh"
NAME=geoae_kit_d6144
OUT="${1:-$DELTA_DATA/portable/$NAME}"
E=e2e/checkpoints/general/llama3.2-3B/layer27
FILES=(
  checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
  $E/delta/balanced_kmeans_k2000_dpc_sampled.npz
  $E/delta/balanced_kmeans_k2000_dpc_sampled_tokmean.npz
  $E/token_bias_sampled.npz
  activations_sampled_10M/norm_params_layer27.npz
)
mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  mkdir -p "$OUT/$(dirname "$f")"
  [ -f "$OUT/$f" ] && cmp -s "$f" "$OUT/$f" || cp -L "$f" "$OUT/$f"
  echo "[kit] $f"
done
mkdir -p "$OUT/corpus/sampled_v1" "$OUT/activations_sampled_10M"
cp -L corpus/sampled_v1/* "$OUT/corpus/sampled_v1/"
echo "[kit] corpus/sampled_v1/ ($(ls corpus/sampled_v1 | wc -l) files)"
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
m = json.load(open("activations_sampled_10M/meta.json"))
m["corpus_dir_original"] = m["corpus_dir"]
m["corpus_dir"] = "corpus/sampled_v1"          # relative: resolved from the repo root (env.sh cds there)
json.dump(m, open(f"{out}/activations_sampled_10M/meta.json", "w"), indent=1)
print("[kit] activations_sampled_10M/meta.json (corpus_dir -> corpus/sampled_v1)")
PY
cat > "$OUT/KIT.md" <<KIT
# $NAME
Packed $(date +%F) on $(hostname -s) from git $(git rev-parse --short HEAD) (branch $(git rev-parse --abbrev-ref HEAD)).
Unpack into a GeoAE checkout with: bash experiments/token_bypass/portable/fetch_kit.sh <this dir | $NAME.tar | hf:<repo>>
Contents: d6144 token-bypass AE (Llama-3.2-3B L27, K=2000, epoch 50), base + token-mean k-means (K=2000),
token-mean table, dump meta + norm params, frozen corpus. The dump rows themselves (61 GB) are NOT included.
KIT
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 ! -name KIT.md -printf '%P\n' | sort | xargs sha256sum > MANIFEST.sha256)
echo "[kit] MANIFEST.sha256: $(wc -l < "$OUT/MANIFEST.sha256") files"
tar -cf "$OUT.tar" -C "$(dirname "$OUT")" "$NAME"
echo "[kit] $OUT.tar ($(du -h "$OUT.tar" | cut -f1))"
cat <<NEXT
Next — make it reachable from anywhere (pick one):
  Hugging Face (private):  scripts/py -c "from huggingface_hub import HfApi as A; a=A(); a.create_repo('<user>/$NAME', private=True, exist_ok=True); a.upload_folder(folder_path='$OUT', repo_id='<user>/$NAME')"
                           then on the new machine: bash experiments/token_bypass/portable/setup_legacy.sh hf:<user>/$NAME
  copy the tar:            scp <delta>:$OUT.tar .   then: bash experiments/token_bypass/portable/setup_legacy.sh $NAME.tar
NEXT
