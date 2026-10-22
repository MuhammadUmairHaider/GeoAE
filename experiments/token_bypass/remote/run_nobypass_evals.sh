#!/bin/bash
# Llama L27 NO-bypass AE (the token-level partner of the token-bypass GeoAE: same dump, recipe and seed, token_bias "")
# through the final model's evaluations, plus the dense-subset / homogeneity probes next to the k-means fits and GeoAE.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1
C=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt
G=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
D=e2e/checkpoints/general/llama3.2-3B/layer27/delta
O=results/delta/d6144_nobypass/evals
mkdir -p $O
.venv/bin/python -u geoae/evaluation/token_structure.py --arms "nobypass=$C" --activations_dir activations_sampled_10M --layer 27 \
    --out $O/token_structure.json || echo "FAILED token_structure"
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache_ids --models "nobypass=$C" --atlas_last "" \
    --out $O/probe.json || echo "FAILED probe"
.venv/bin/python -u -m geoae.evaluation.geometry --checkpoints "$C" --names nobypass --activations_dir activations_sampled_10M \
    --layer 27 --n_sample 1000000 --seed 0 --out $O/cq.json || echo "FAILED cq"
.venv/bin/python -u -m geoae.evaluation.mmlu --checkpoint "$C" --out $O/mmlu.json || echo "FAILED mmlu"
.venv/bin/python -u geoae/evaluation/alignment.py \
    --arms "res_balanced=km:$D/balanced_kmeans_k2000_dpc_sampled.npz,res_plain=km:$D/plain_kmeans_k2000_dpc_sampled_seed42.npz,tok_plain=km:$D/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz,geoae=ae:$G,nobypass=ae:$C" \
    --out $O/probe_dense_subsets.json || echo "FAILED dense"
.venv/bin/python experiments/paper/analysis/summarize_probe_hc.py $O/probe_dense_subsets.json min10 common_min10 all > $O/probe_homogeneity_min10.md
echo NOBYPASS_EVALS_DONE
