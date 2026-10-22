#!/bin/bash
# K = 256 row of the Gemma 3 4B L33 K series (11 Oct): the steps of experiments/cluster_count/remote/run_kcost_g4b.sh, run_kcost_tokmean.sh and
# run_kcost_hc.sh for one K. Plain residual and plain token-removed k-means at K = 256 (density peaks seed 0, Lloyd) are
# fitted first. Outputs results/delta/g4b_l33_d5120/evals/kcost/*_k256.* ; nothing of the K = 2,000 / 4,000 / 8,000 files is touched.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
source configs/sites/local.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
PY=.venv/bin/python
E=e2e/checkpoints/general/gemma3-4B/layer33/delta
ACT=activations_gemma3_4b_sampled_struct5_10M
KM=$E/balanced_kmeans_k2000_dpc_sampled_struct5.npz; KMT=$E/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz
T=checkpoints/gemma3-4B/layer33/k256_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
P=$E/plain_kmeans_k256_dpc_sampled_struct5.npz; PT=$E/plain_kmeans_k256_dpc_sampled_struct5_tokmean.npz
O=results/delta/g4b_l33_d5120/evals/kcost; mkdir -p $O
[ -f $T ] || { echo "[k256] missing $T"; exit 1; }
step() { local name=$1; shift; echo "[k256 $(date +%H:%M:%S)] start $name"; "$@" > $O/$name.log 2>&1; echo "[k256 $(date +%H:%M:%S)] done $name rc=$?"; }
[ -f $P ]  || step plain_k256 $PY -u experiments/paper/analysis/paper_plain_kmeans_grid.py --activations_dir $ACT --layer 33 --km $KM --kmt $KMT --k 256 --spaces residual --out $O/plain_kmeans_k256.json
[ -f $PT ] || step plain_tokmean_k256 $PY -u experiments/paper/analysis/paper_plain_kmeans_grid.py --activations_dir $ACT --layer 33 --km $KM --kmt $KMT --k 256 --spaces tokmean --out $O/plain_kmeans_tokmean_k256.json
step token_structure_k256 $PY -u geoae/evaluation/token_structure.py --arms "plain256=$P,plain_tokmean256=$PT,bypass256=$T" --activations_dir $ACT --layer 33 --out $O/token_structure_k256.json --save_labels $O/token_structure_labels_k256.npz
step cq_k256 $PY -u -m geoae.evaluation.geometry --baseline $P --checkpoints $T --names plain256 bypass256 --activations_dir $ACT --layer 33 --n_sample 1000000 --seed 0 --out $O/cq_k256.json
step mmlu_k256 $PY -u -m geoae.evaluation.mmlu --checkpoint $T --out $O/mmlu_k256.json
step splice_ppl_k256 $PY -u -m geoae.evaluation.perplexity --ae $O/splice_ppl_k256.json=$T
step probe_k256 $PY -u geoae/evaluation/chance_corrected_probe.py --cache cache/gemma3-4b/l33 --models "bypass256=$T" --baselines "plain256=$P,plain_tokmean256=$PT" --atlas_last "" --out $O/probe_k256.json --save_labels $O/probe_labels_k256.npz
step hc_tokstruct_k256 $PY -u geoae/evaluation/filtered_probe.py --labels $O/token_structure_labels_k256.npz --mode tokstruct --groups "K256=plain256,plain_tokmean256,bypass256" --n_boot 100 --out $O/hc_tokstruct_k256.json
step hc_probe_k256 $PY -u geoae/evaluation/filtered_probe.py --labels $O/probe_labels_k256.npz --mode probe --groups "K256=plain256,plain_tokmean256,bypass256" --n_boot 200 --out $O/hc_probe_k256.json
echo "[k256 $(date +%H:%M:%S)] KCOST256_DONE"
