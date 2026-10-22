#!/bin/bash
# Component ablations of the Gemma 3 4B L33 K=2,000 token-bypass AE (11 Oct): the four measurements of the paper's
# ablation table for one arm, with the commands of the main arm's sheet (experiments/paper/scripts/run_delta_evals.sh: cq, tokstruct,
# mmlu) and of experiments/paper/argo/splice_cost_g4b.sbatch (WikiText-2 perplexity), plus the probe sheet of experiments/cluster_count/remote/run_kcost_hc.sh.
#   bash experiments/regularization/remote/run_abl_g4b.sh <nopull|nosep|novicreg|kmppinit|nostage>
# Output: results/delta/g4b_l33_d5120/evals/ablations/<arm>/ ; ends with ABL_DONE <arm>.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
ARM="${1:?arm}"
cd "$GEOAE_ROOT"
source configs/sites/local.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
PY=.venv/bin/python
E=e2e/checkpoints/general/gemma3-4B/layer33/delta
ACT=activations_gemma3_4b_sampled_struct5_10M
KM=$E/balanced_kmeans_k2000_dpc_sampled_struct5.npz; KMT=$E/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz
P=$E/plain_kmeans_k2000_dpc_sampled_struct5.npz; PT=$E/plain_kmeans_k2000_dpc_sampled_struct5_tokmean.npz
T=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias_$ARM/step_0014450.pt
O=results/delta/g4b_l33_d5120/evals/ablations/$ARM; mkdir -p $O
[ -f $T ] || { echo "[abl $ARM] missing $T"; exit 1; }
while pgrep -f run_kcost_g4b_k256.sh >/dev/null; do sleep 30; done        # one evaluation sheet on the GPU at a time
step() { local name=$1; shift; echo "[abl $ARM $(date +%H:%M:%S)] start $name"; "$@" > $O/$name.log 2>&1; echo "[abl $ARM $(date +%H:%M:%S)] done $name rc=$?"; }
step cq $PY -u -m geoae.evaluation.geometry --baseline $KM --checkpoints $T --names base $ARM --activations_dir $ACT --layer 33 --n_sample 1000000 --seed 0 --out $O/cq.json
step token_structure $PY -u geoae/evaluation/token_structure.py --arms "base=$KM,base_tokmean=$KMT,plain=$P,plain_tokmean=$PT,$ARM=$T" --activations_dir $ACT --layer 33 --out $O/token_structure.json --save_labels $O/token_structure_labels.npz
step mmlu $PY -u -m geoae.evaluation.mmlu --checkpoint $T --out $O/mmlu.json
step splice_ppl $PY -u -m geoae.evaluation.perplexity --ae $O/splice_ppl.json=$T
step probe $PY -u geoae/evaluation/chance_corrected_probe.py --cache cache/gemma3-4b/l33 --models "$ARM=$T" --baselines "plain=$P,plain_tokmean=$PT" --atlas_last "" --out $O/probe.json --save_labels $O/probe_labels.npz
step hc_probe $PY -u geoae/evaluation/filtered_probe.py --labels $O/probe_labels.npz --mode probe --groups "K2000=plain,plain_tokmean,$ARM" --n_boot 200 --out $O/hc_probe.json
echo "[abl $ARM $(date +%H:%M:%S)] ABL_DONE $ARM"
