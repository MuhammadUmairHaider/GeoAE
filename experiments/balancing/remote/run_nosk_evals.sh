#!/bin/bash
# Evaluate the no-Sinkhorn ablations with the same code and settings as the final model (d6144 token-bypass).
#   bash experiments/balancing/remote/run_nosk_evals.sh nosk_reinit=<ckpt> nosk_lloyd=<ckpt> ...
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1
O=results/delta/nosinkhorn
mkdir -p $O
ARMS=$(IFS=,; echo "$*")
TAG=$(for a in "$@"; do echo -n "${a%%=*}+"; done | sed 's/+$//')
.venv/bin/python -u geoae/evaluation/token_structure.py --arms "$ARMS" --activations_dir activations_sampled_10M --layer 27 --out $O/token_structure_$TAG.json
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache_ids --models "$ARMS" --atlas_last "" --out $O/probe_$TAG.json
for a in "$@"; do n=${a%%=*}; c=${a#*=}
  .venv/bin/python -u -m geoae.evaluation.geometry --checkpoints "$c" --names "$n" --activations_dir activations_sampled_10M \
    --layer 27 --n_sample 1000000 --seed 0 --out $O/cq_$n.json
  .venv/bin/python -u -m geoae.evaluation.mmlu --checkpoint "$c" --out $O/mmlu_$n.json
done
echo NOSK_EVALS_DONE "$TAG"
