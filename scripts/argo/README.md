# GeoAE on ACENET Argo (1x H100, group partition c-hsajjad, account def-hsajjad)

1. Repo: `/home/dogar/projects/def-hsajjad/dogar/GeoAE/repo` (code only). Data root `/scratch/dogar/geoae` (empty until you copy dumps),
   linked into the repo by `bash scripts/argo/link_data.sh`. Venv `/scratch/dogar/venvs/geoae` (py3.14, torch 2.13+cu130; rebuild: `sbatch scripts/argo/setup.sbatch`).
2. Every shell/job: `source scripts/argo/env.sh` (sets UV_*, HF_HOME=/scratch/dogar/hf_cache, DELTA_DATA, GEOAE_REMOTE=1; the eval sheets and `scripts/delta/py` then run unchanged).
3. Models are in the HF cache (Llama-3.2-3B, gemma-3-1b-pt, gemma-3-4b-pt). Copy a dump from Delta over the ControlMaster, e.g.
   `rsync -a -e 'ssh -o BatchMode=yes' <delta>:/work/hdd/bimc/mhaider/GeoAE/activations/activations_sampled_struct5_10M/ argo:/scratch/dogar/geoae/activations/activations_sampled_struct5_10M/` (~40 MB/s, one stream).
4. Train (resumes automatically; ONE GPU only, the node is shared): `sbatch scripts/argo/train.sbatch configs/base/<run>.yaml [train args]`; Gemma-4B dump (102 GB) needs `--mem=240G`.
5. Check the setup any time with `sbatch scripts/argo/smoke.sbatch` (driver, bf16 matmul, Llama forward, pytest; baseline 375 tests, 3 fail on a missing topic14.npz).
   Do not run `uv sync` bare: it deletes `openai`/`umap-learn`; use `uv run --no-sync`, `scripts/delta/py`, or re-run setup.sbatch.
