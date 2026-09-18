---
name: dpc-density-peaks-init
description: Density-peaks centroid init (centroid_init dpc / reinit_mode peaks) — L27 results, the encoder-free dpc control, and eval gotchas
metadata:
  type: project
---
Density-peaks init (Rodriguez & Laio) added 2026-09-14: `centroid_init: dpc`, `fill_mode: peaks` (inside seeded), `reinit_mode: peaks`. Selector is `geoae.seeded_init.density_peaks_select`. Suppression radius uses the median kNN radius; the global pairwise scale swallowed whole modes.

L27 K=2000 d6144 dpc arm, finished 2026-09-15, vs k-means++ parent:
- Init density at the 69th pct (the coverage fill was at the 21st). Reinits 260 vs 2394 (9x fewer), but init and reinit were changed together.
- concept_probe NMI: token mean 0.2507 (kmeanspp 0.2179, balanced k-means 0.2523); sequence mean 0.2419 (worse than kmeanspp 0.2531). Wins pos/surface/ner, loses ravel/topic/atlas. The split follows codebook coverage: rungs with ~1100 live clusters won, rungs with ~43 lost.
- kNN-10 agreement agrees with NMI (pos_coarse 0.871 vs 0.801 balanced). UMAP shows no global change, so the gain is local only. topic14 UMAP: both AE arms merge Company/WrittenWork/Film/Artist, which balanced k-means keeps apart. That is the encoder, not the init.
- DB14 K=14 Hungarian: dpc 0.4946, below kmeanspp 0.4997, seeded_atlas 0.5575, raw zscore 0.6823.

**Why:** balanced k-means on raw activations uses k-means++ init plus farthest reinit, so it cannot tell whether dpc's gains come from the init or the encoder. The user asked for a dpc-init encoder-free baseline (the request was lost once).
**How to apply:** use `fit_balanced_kmeans --init dpc --reinit peaks` (added 2026-09-17). Target npz: `e2e/checkpoints/general/llama3.2-3B/layer27/balanced_kmeans_k2000_dpc.npz`. Compare as a 2x2: {k-means++, dpc} x {encoder, none}. Raw residual space has weaker density contrast (52nd pct of picks vs 69th in the AE latent). The `_dpc_tanh` arm has the same density settings and only swaps GELU for tanh. clustering_quality `--names` is aligned with (baseline, *checkpoints). For the probe use `--anchor_seed 42` to match `train.seed`. See [[fineweb-atlas-concept-alignment]].
