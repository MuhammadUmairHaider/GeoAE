#!/bin/bash
# Gemma 3 4B L33 NO-bypass AE (Codex's gemma4b_last_plain, epoch 50, trained on activations_gemma3_4b_sampled_struct5_train10M)
# through the paper Gemma model's evaluations, on the paper Gemma dump / probe cache, plus the four-way probe table with the
# reporting rule (fixed-point subset: every kept row's cluster holds > 5 kept rows in every space; raw h / c / NMI).
# Also refits the Gemma plain k-means at seed 42 (Llama L27 and the mid layers use seed 42; the old Gemma fits were seed 0)
# and reruns the Llama L27 four-way probes with the same subset.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1
ACT=activations_gemma3_4b_sampled_struct5_10M
D=e2e/checkpoints/general/gemma3-4B/layer33/delta
C=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt
G=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
E=results/delta/g4b_l33_d5120/evals
O=results/delta/g4b_l33_nobypass/evals
mkdir -p $O
.venv/bin/python -u experiments/paper/analysis/paper_plain_kmeans_grid.py --activations_dir $ACT --layer 33 \
    --km $D/balanced_kmeans_k2000_dpc_sampled_struct5.npz --kmt $D/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz \
    --seed 42 --tag _seed42 --out $E/plain_kmeans_grid_seed42.json || echo "FAILED grid"
PL=$D/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz
PLT=$D/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz
.venv/bin/python -u geoae/evaluation/alignment.py --cache cache/gemma3-4b/l33 --fp_min 6 \
    --arms "res_balanced=km:$D/balanced_kmeans_k2000_dpc_sampled_struct5.npz,res_plain=km:$PL,tok_plain=km:$PLT,geoae=ae:$G,nobypass=ae:$C" \
    --out $O/probe_dense_subsets.json || echo "FAILED probes"
.venv/bin/python experiments/paper/analysis/summarize_probe_raw_hc.py $O/probe_dense_subsets.json common_fp6 all > $O/probe_raw_hc.md
.venv/bin/python -u geoae/evaluation/token_structure.py --arms "nobypass=$C,plain_seed42=$PL,plain_tokmean_seed42=$PLT" \
    --activations_dir $ACT --layer 33 --out $O/token_structure.json || echo "FAILED token_structure"
.venv/bin/python -u -m geoae.evaluation.geometry --checkpoints "$C" --names nobypass --activations_dir $ACT --layer 33 \
    --n_sample 1000000 --seed 0 --out $O/cq.json || echo "FAILED cq"
.venv/bin/python -u -m geoae.evaluation.mmlu --checkpoint "$C" --out $O/mmlu.json || echo "FAILED mmlu"
# Llama L27 four-way probes with the same fixed-point subset
L=e2e/checkpoints/general/llama3.2-3B/layer27/delta
NL=results/delta/d6144_nobypass/evals
.venv/bin/python -u geoae/evaluation/alignment.py --cache cache_ids --fp_min 6 \
    --arms "res_balanced=km:$L/balanced_kmeans_k2000_dpc_sampled.npz,res_plain=km:$L/plain_kmeans_k2000_dpc_sampled_seed42.npz,tok_plain=km:$L/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz,geoae=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt,nobypass=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt" \
    --out $NL/probe_dense_subsets.json || echo "FAILED llama probes"
.venv/bin/python experiments/paper/analysis/summarize_probe_raw_hc.py $NL/probe_dense_subsets.json common_fp6 all > $NL/probe_raw_hc.md
echo G4B_NOBYPASS_EVALS_DONE
