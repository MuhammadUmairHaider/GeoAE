export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"; source configs/sites/local.sh >/dev/null 2>&1; export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
until grep -q CSTEER_DONE logs/csteer_k.log; do sleep 30; done
E=e2e/checkpoints/general/llama3.2-3B/layer27/delta; O=results/delta/d6144/evals/kcost; mkdir -p $O
A="plain=$E/plain_kmeans_k2000_density_peaks_sampled.npz,plain_tokmean=$E/plain_kmeans_k2000_density_peaks_sampled_tokmean.npz,bypass=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt"
echo "[$(date +%H:%M:%S)] start llama"; .venv/bin/python -u -m geoae.evaluation.cluster_steering --arms "$A" --activations_dir activations_sampled_10M --layer 27 --out $O/cluster_steering_plain_k2000.json > $O/cluster_steering_plain_k2000.log 2>&1; echo "[$(date +%H:%M:%S)] done llama rc=$?"; echo LLAMA_DONE
