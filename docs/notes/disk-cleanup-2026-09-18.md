---
name: disk-cleanup-2026-09-18
description: 2026-09-18 disk-full cleanup (282 GB) and 2026-09-26 cleanup (127 GB) — which dumps, models and intermediate checkpoints are gone and what that rules out
metadata:
  type: project
---
The 484 GB disk filled (d3072_dpc crashed saving its epoch-31 checkpoint). Other projects
were only ~10 GB; GeoAE itself was 382 GB. User chose to delete:

- **Other-track data (gone):** `activations_gemma3_4b/layer_33.npy` (80 GB),
  `activations_diverse_10M_l14/layer_14.npy` (59 GB), `cache_l14/` (L14 concept caches, 5 GB).
  Their `meta*.json` and `norm_params_*.npz` were KEPT beside the deleted dumps.
- **HF models (gone, re-downloadable):** gemma-3-4b-pt, gemma-3-12b-pt, Qwen3.5-9B.
  Llama-3.2-3B is still cached.
- **Intermediate checkpoints (gone):** 210 files, 105 GB. These runs now hold ONLY
  `best_val.pt` + the final step: L27 d6144 dpc / dpc_tanh / seeded_atlas (step_0014200),
  L14 d6144 seeded (step_0013950), and the 6 sweeps/cov_l27 runs (step_0000450).
  -> Per-epoch trajectory analyses are no longer possible for these runs, and
  step_0007952 (epoch 28, used for the tanh28-vs-gelu28 evals) no longer exists.

Still on disk and needed: `activations_diverse_10M/layer_27.npy` (the L27 training dump),
`cache/` (L27 concept caches), all other checkpoints incl. the in-progress d3072_dpc run.
Any Gemma, Qwen or L14 work now needs the dump re-extracted first.
Related: [[gemma3-12b-setup]], [[tanh-encoder-arm]].

**2026-09-26 cleanup (127 GB, disk 87% -> 62%):** every run under `checkpoints/` now holds only
`best_val.pt` + its final step (230 files removed) — incl. d12288_dpc, d6144_dpc_sampled,
d6144_dpc_sampled_tokbias (bypass AE), d3072_dpc, lam2_d3072_pd, ft_sup_enc/full (kept step_0015620).
-> No per-epoch trajectory analysis is possible for ANY run anymore.
User declined deleting: Gemma-1B L25 dump (`runs/gemma3_1b_l25_d2304_dpc_seed42/activations`, 45 GB),
old `e2e/` step files (21 GB), both L27 training dumps. `uv cache prune` freed only 0.4 GB
(the 18 GB archive is live; only `uv cache clean` would free it).
