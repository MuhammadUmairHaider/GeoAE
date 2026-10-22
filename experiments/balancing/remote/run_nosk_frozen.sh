#!/bin/bash
# Run C (no Sinkhorn, Lloyd-initialised codebook frozen from epoch 11; stopped at epoch 48 = step_0013872):
# re-fit plain Lloyd k-means on its final latent (experiments/balancing/analysis/paper_lloyd_ep10_codebook.py: density peaks seed 42 + <= 100
# Lloyd iterations on every 6th training row), then the A/B evals for both codebooks (nosk_frozen = the frozen centres
# as trained, nosk_frozen_refit = the re-fit). MMLU once: the splice reconstructs through the AE and never uses the codebook.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1
C=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nosinkhorn_lloyd_frozen
O=results/delta/nosinkhorn
mkdir -p $O
.venv/bin/python -u experiments/balancing/analysis/paper_lloyd_ep10_codebook.py --ckpt $C/step_0013872.pt --out_ckpt $C/lloyd_refit_codebook.pt \
    --out_json $O/lloyd_refit_frozen.json || { echo NOSK_FROZEN_FAILED refit; exit 1; }
A="nosk_frozen=$C/step_0013872.pt,nosk_frozen_refit=$C/lloyd_refit_codebook.pt"
.venv/bin/python -u geoae/evaluation/token_structure.py --arms "$A" --activations_dir activations_sampled_10M --layer 27 \
    --out $O/token_structure_nosk_frozen.json || echo "FAILED token_structure"
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache_ids --models "$A" --atlas_last "" \
    --out $O/probe_nosk_frozen.json || echo "FAILED probe"
for a in nosk_frozen_refit=$C/lloyd_refit_codebook.pt nosk_frozen=$C/step_0013872.pt; do n=${a%%=*}; c=${a#*=}
  .venv/bin/python -u -m geoae.evaluation.geometry --checkpoints "$c" --names "$n" --activations_dir activations_sampled_10M \
    --layer 27 --n_sample 1000000 --seed 0 --out $O/cq_$n.json || echo "FAILED cq $n"
done
.venv/bin/python -u -m geoae.evaluation.mmlu --checkpoint $C/step_0013872.pt --out $O/mmlu_nosk_frozen.json || echo "FAILED mmlu"
echo NOSK_FROZEN_DONE
