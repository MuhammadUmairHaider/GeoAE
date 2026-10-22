# gpt_run

The campaign trains one independent token-bypass GeoAE at every GPT-2 small
transformer block (0–11), then runs the GPU evaluation suite on Argo H100.
External LLM judges are excluded, as requested.

Each AE maps 768 residual dimensions to 1,536 latent dimensions, with 2,000
clusters, batch size 32,768, 50 epochs and seed 42. Losses, initialization and
the phase schedule match the existing sampled struct5 recipe. Extraction uses
10 million shared token rows, the same ten source weights, and 1,024-token
contexts. All block outputs are captured in one frozen-model forward pass.

The isolated Argo checkout is
`/home/dogar/projects/def-hsajjad/dogar/GeoAE/gpt_run`; bulk data lives under
`/scratch/dogar/geoae`. The job requests one H100, 24 CPUs and 450 GiB RAM.
Gemma training job 284382 and evaluation job 284383 completed successfully
before this campaign was submitted. The initial pipeline job was **284386**,
named `gpt_run`, on `argo73`.

Training and evaluation each request up to 12 simultaneous layer workers.
The job profiles the actual recipe on its allocated GPU and checks remaining
allocated host RAM before choosing the worker counts. Those measurements,
memory margins and selected counts are saved in `runs/gpt_run/gpu_profile.json`.
DPC initialization shares a lock to
bound temporary memory. A CUDA OOM retries affected layers at lower
concurrency; training resumes from saved checkpoints. Batch size and losses
remain the same.

A separate synthetic CUDA smoke test passed on the allocated NVIDIA H100 NVL:
the GPU-only admission check selected 10 training workers and 9 evaluation
workers. These are preliminary limits; the campaign also checks RAM after
staging and saves its final counts with the actual-data profile. The smoke
receipt is `runs/gpt_run/cuda_smoke_profile.json`, explicitly marked as synthetic
and excluded from scientific results.

The job copies the roughly 344 GiB activation dump into shared tmpfs once.
All workers read that copy because Argo GPFS random reads otherwise dominate
training. Staging, normalization, token tables and final checkpoints are
verified before their dependent phases start. Evaluation requires the
epoch-50 checkpoint for each layer.

The evaluation suite covers balanced baselines, concept and benchmark caches,
causal reconstruction and cluster ablation, token structure, probes, MMLU,
clustering quality, numerical and semantic interventions, geometry, fixed
probe perturbations, closest tokens and word sense. Metrics that directly
apply the final head to a residual require block 11; those restrictions are
recorded in each layer's evaluation manifest. Causal splice metrics run at
every layer through the remaining GPT-2 blocks.

On Argo, inspect the submitted job and persistent progress with:

```bash
cd /home/dogar/projects/def-hsajjad/dogar/GeoAE/gpt_run
export GEOAE_ROOT="$PWD"
source scripts/gpt2/argo_env.sh
"$GEOAE_PYTHON" scripts/gpt2/status.py
cat runs/gpt_run/pipeline_state.json
tail -f logs/gpt_run/slurm-gpt_run-284386.out
```

Artifacts are organized under `runs/gpt_run`, `logs/gpt_run`,
`results/gpt_run/layer<L>` and `checkpoints/gpt2/layer<L>`.

To resume an interrupted campaign once its previous job has left the queue:

```bash
"$GEOAE_PYTHON" scripts/gpt2/submit_argo.py \
  --after-gemma 284383 --resubmit --judges skip
```

The submitter verifies the Gemma job's accounting record. Successful completed
dependencies are recorded as satisfied; active dependencies use `afterok`.

On 2026-10-04, all twelve layers were verified complete at epoch 50. The first
six layers completed their core evaluations, MMLU, probes, clustering quality
and numerical controls. Steering then encountered a GPT-2 position overflow:
the 1,024-token prompt left no room for a six-token continuation. The shared
capture and prediction helpers now use a 1,018-token prompt budget for GPT-2,
and joint-correct caches record that budget. Other model families retain their
existing prompt limit.

The original job was stopped after evaluation workers became stuck in the
NVIDIA UVM driver and the H100 node entered drain. Saved checkpoints and
completed evaluations remain available. `scripts/gpt2/recover_evals.py` resumes
only GPU evaluations, verifies the final checkpoints, and reuses validated
completed outputs. Range results need their final `summary` to count as
complete. Recovery runs at most two layers at once, with temporary staging
only for unfinished steps that read activation arrays.
