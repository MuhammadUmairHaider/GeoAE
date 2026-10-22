export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"; source configs/sites/local.sh >/dev/null 2>&1; export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
E=e2e/checkpoints/general/gemma3-4B/layer33/delta; B=checkpoints/gemma3-4B/layer33; O=results/delta/g4b_l33_d5120/evals/kcost
for k in 2000 4000 8000; do
  s=$([ $k = 2000 ] && echo "" || echo "$((k/1000))k")
  A="plain$s=$E/plain_kmeans_k${k}_dpc_sampled_struct5.npz,plain_tokmean$s=$E/plain_kmeans_k${k}_dpc_sampled_struct5_tokmean.npz,bypass$s=$B/k${k}_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt"
  echo "[$(date +%H:%M:%S)] start K=$k"
  .venv/bin/python -u -m geoae.evaluation.cluster_steering --arms "$A" --activations_dir activations_gemma3_4b_sampled_struct5_10M --layer 33 --out $O/cluster_steering_k$k.json > $O/cluster_steering_k$k.log 2>&1
  echo "[$(date +%H:%M:%S)] done K=$k rc=$?"
done; echo CSTEER_DONE
