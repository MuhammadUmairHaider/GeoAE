#!/bin/bash
# Run B's epoch-10 latent + re-fitted plain k-means codebook, scored with the paper's evaluators.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=.
D=checkpoints/llama3.2-3B/layer27/nosk_lloyd_ep10
E=results/delta/d6144/evals
CK=$D/lloyd_ep10_codebook.pt
.venv/bin/python -u experiments/balancing/analysis/paper_lloyd_ep10_codebook.py --ckpt $D/step_0002890.pt --out_ckpt $CK --out_json $E/lloyd_ep10_codebook.json || { echo FAILED; exit 1; }
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache_ids --models "lloyd_ep10=$CK" --atlas_last "" --out $E/probe_lloyd_ep10.json
.venv/bin/python -u geoae/evaluation/token_structure.py --arms "lloyd_ep10=$CK" --activations_dir activations_sampled_10M --layer 27 --out $E/token_structure_lloyd_ep10.json
.venv/bin/python -u -m geoae.evaluation.geometry --checkpoints $CK --names lloyd_ep10 --activations_dir activations_sampled_10M --layer 27 --n_sample 1000000 --seed 0 --out $E/cq_lloyd_ep10.json
echo LLOYD_EP10_DONE
