---
name: d3072-width-arm
description: d3072 vs d6144 dpc (L27 K=2000, both ep50, 2026-09-19) — halving latent width amplifies the range-intervention effect in both directions and steepens the base-quality moderation (rho −0.77 vs −0.29), at 2.5x the MMLU cost
metadata:
  type: project
---
Arm: `k2000_bnh_b32k_lam1_d3072_dpc` (latent 3072, gelu, dpc init), evaluated at
step_0014200 (ep50) vs d6144 dpc at step_0014200. **best_val.pt is epoch 10 (pre-clustering,
centroids never initialised) — never evaluate it.** Crashed once at ep31 on a full disk;
resumed cleanly. Eval sheet: eval_out/run_d3072_evals.sh.

- **Costs, all worse:** val_mse 0.095 vs 0.035; MMLU under recon −0.066 vs −0.027;
  wiki ppl under recon +2.20 vs +0.30.
- **Clustering, d6144 better on every scale-free metric:** silhouette 0.026 vs 0.036,
  Dunn 0.134 vs 0.151, CH 699 vs 856, effective rank 180 vs 217.
- **Probes/kNN: tie.** Probe NMI token +0.0004 (p=.96), sequence −0.001 (p=.77);
  kNN-10 d3072 >= d6144 on 5/5 rungs by ~0.01.
- **Range interventions: width is a dial on the effect, both directions.** Matched-erasure
  collateral margin (neg = AE better), transport a1:
  DB14 d3072 −0.111 (p=.003, 7/7) vs d6144 −0.069; bias_in_bios d3072 +0.107 (p<.0001,
  AE better 2/17) vs d6144 +0.030. rm_range_comp: DB14 −0.114 vs −0.060; bios +0.123 vs +0.044.
- **User's hypothesis gets much stronger:** pooled 41 concepts, transport a1 z−h by tercile of
  independent base quality: d3072 +0.225 / −0.077 / −0.125 (spearman −0.77, p<.0001) vs
  d6144 +0.070 / +0.028 / −0.032 (−0.29, p=.067). Narrower = more help where base is weak,
  more harm where base is strong. Consistent with compression acting as a regulariser on edits.

**Width series tooling (2026-09-20).** `eval_out/run_d12288_evals.sh` runs the whole suite
for any arm (step-selectable: `./eval_out/run_d12288_evals.sh cq range_db14`, `FORCE=1` to redo),
and `eval_out/summarize_width_series.py` prints d3072/d6144/d12288 side by side from whatever
JSONs exist. Two traps it encodes: the d6144 steer/range JSONs were run on **best_val (ep47)**,
not the epoch-matched step_0014200 (the summary flags this per file); and the summary recomputes
matched-erasure collateral with its own explicit definition (interpolate comp_drop to
E = min over spaces of max tgt_drop across the alpha grid), so its DB14 transport number for
d3072 is −0.028, NOT the −0.111 quoted above from the earlier ad-hoc analysis — directions agree,
magnitudes do not. Compare arms only within one of the two definitions.
d12288 (ep50, val_mse 0.0193): its best_val IS valid (ep49, centroids initialised) — the ep10
best_val trap was d3072-specific — but use step_0014200 to stay epoch-matched.

Related: [[biasbios-range-reversal]], [[range-interventions-h-vs-z]], [[disk-cleanup-2026-09-18]].
