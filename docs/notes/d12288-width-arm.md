---
name: d12288-width-arm
description: d12288 (4x) vs d6144 vs d3072 at L27 K=2000, all ep50 (2026-09-20) — width is nearly free and wins on cost/geometry/topic-probes/DB14 interventions, but the bias_in_bios deficit is width-invariant and token-level probes regress
metadata:
  type: project
---
Arm `k2000_bnh_b32k_lam1_d12288_dpc`, evaluated at step_0014200 (ep50). Its best_val is ep49
and valid (the ep10 pre-clustering trap was d3072-only), but use step_0014200 to stay
epoch-matched. Eval sheet `eval_out/run_d12288_evals.sh`, tables from
`eval_out/summarize_width_series.py`. Full suite ran 18:00-20:50, no failures.

- **Cost halves per doubling, so width is nearly free:** val_mse 0.0949/0.0349/0.0193,
  MMLU delta -0.066/-0.0265/-0.0130, wiki ppl +2.20/+0.29/+0.09 (d3072/d6144/d12288).
- **Geometry: d12288 best on every stable metric.** silhouette 0.025/0.036/0.050,
  CH 699/856/1072, DB 3.18/2.97/2.70, separability 109/176/294, intra-var 0.418/0.275/0.186,
  effective rank 180/217/290. But relative rank FALLS 5.9% -> 3.5% -> 2.4% of dims.
- **Probes trade token structure for document structure** (paired over rungs, vs d6144):
  token/surface -0.032 (p=.040; pos_fine .465->.407), doc/topic +0.029 (p=.0017; language
  .670->.740, topic14 .443->.472, ravel_continent .092->.136), affect/other flat. Mean NMI
  0.2503 vs 0.2445. This is the clearest "wider = more semantic, less lexical" evidence yet.
- **DB14 interventions: best arm so far, and the mechanism is clean.** Whole-concept steering
  at a=1: z-h -0.0723 (p=.0001) — beats the tanh arm's -0.075 (p=.003) on significance —
  with IDENTICAL target erasure (-0.963 both) and collateral 0.327 (z) vs 0.399 (h), i.e. 18%
  less collateral for the same erasure. Matched-erasure collateral now significant on three
  operators: global -0.039*, range -0.059*, transport -0.045*, vs only st_range at d6144.
- **bias_in_bios does NOT improve — the deficit is structural, not capacity.** Every mode still
  favours base. Only the catastrophic removals soften (rm_full_comp -0.378/-0.174/-0.116,
  rm_range_zero -0.152/-0.102/-0.003 = now a tie). Matched-erasure collateral is flat-to-WORSE
  than d6144: global +0.025/+0.025, transport +0.047/+0.047, salient +0.045->+0.059,
  range +0.057->+0.073. **Scaling width does not fix the generalisation failure in
  [[biasbios-range-reversal]].**
- **The narrow-is-a-regulariser effect is a d3072 phenomenon.** Base-quality moderation
  (x = h rm_range_comp selectivity from other arms' runs, leave-one-out; y = z-h at transport
  a1): rho -0.69 / -0.20 / -0.32, tercile spread collapses 0.27 -> 0.08 -> 0.07. WITHIN dataset
  it is only real at d3072 (db14 -0.79, bios -0.77, both p<.01); at d6144 and d12288 the
  within-dataset rho is null (p>=.49) and the pooled rho is mostly the DB14-vs-bios split.
- **Effects shrink but noise shrinks faster.** DB14 range, mean|z-h| over 16 modes
  0.098/0.081/0.074 with per-concept SD 0.301/0.165/0.115 -> effect/noise 0.32/0.49/0.64.
  d12288 is not "more like the base and therefore duller"; it is more reliable. Not trivial
  identity convergence either: it still cuts collateral 18% at matched erasure.
- **Found in the detailed report pass (2026-09-20):** (1) DB14 range: AE significantly better on
  9/16 operators at d12288 vs 5/16 at d6144; bios: base ahead on 12/12 at 4x, 8 significant.
  (2) topic14 probe NMI rises .443->.472 but kNN-10 FALLS .940->.930 — the topic gain is in the
  cluster partition, not local neighbourhoods; POS kNN .873->.815 / .849->.806. All-rung probe
  mean p=.40 — a reallocation, not a lift. (3) Per-concept z-h does NOT replicate across arms
  (DB14 rho 0.00, bios 0.21 between 2x and 4x; binomial SE ~0.095 at n_eval 50 / n_comp 80);
  base-side h measures do (rho .64-.95). Never tell per-concept stories from one run.
  (4) z range gate fires ~0.03 more than h on OTHER concepts at 4x (gap <=0.01 at 1x/2x), so the
  DB14 win isn't cleaner ranges. (5) z-steer ppl at a=1: 6.57 vs h 6.52 (d3072 was 8.45).
  Report: https://claude.ai/artifact/HowUJ4XUF2PTzHwufqRCVu (builder was in session scratchpad).
- **Base / encoder-free controls added 2026-09-21 (probe + kNN run in ONE invocation with the
  three arms, so exact):** balanced k-means (Sinkhorn, NO encoder) MATCHES OR BEATS every AE arm
  on both probe grain means — token 0.252 vs d12288 0.246 (p=.58), sequence 0.259 vs 0.254 (p=.75)
  — and on kNN topic14 (.967 vs .930), language (.994 vs .988), ravel_country (.922 vs .908).
  Plain unbalanced k-means is far behind (token 0.161, sequence 0.107). The AE's real wins over
  balanced: surface/POS/NER group 0.265 vs 0.254 (p=.006), language NMI .740 vs .623, topic4.
  It LOSES topic14 (.472 vs .569) and atlas_content (.296 vs .327). Same conclusion as
  [[fineweb-atlas-concept-alignment]] and [[llm-judge-autointerp]]: the balancing carries most
  of the gain; the encoder's value shows up in interventions, not in topics.
- **CORRECTION to the "token structure is lost" claim:** with the probe tool's OWN token/sequence
  split (LADDER grain), token rungs are FLAT (d6144 0.249 -> d12288 0.246, p=.82) and sequence
  rungs gain (0.241 -> 0.254, p=.061). The loss is specific to surface/POS/NER; the semantic
  token rungs (RAVEL country/continent/language, IOI) gain. Don't say "token-level structure".
- **Raw k-means (base residual, K=2000, same 1M sample) geometry:** silhouette -0.094 (negative),
  CH 133 vs 1072, DB 2.06 (flattered), separability 121, balance H 0.53, 377 EMPTY clusters,
  effective K 270, effective rank 568 (HIGHER than any AE: raw spans more directions). Every arm
  beats it on silhouette/CH/usage. Column is borrowed from cq_init_arms (15 Sep, same seed/sample,
  bridge arm 3 epochs older) -> ±0.003; one ~1h re-run with --baseline refreshes it AND replaces
  the stale Dunn row, since the user fixed dunn/silhouette seeding on 21 Sep.
- **Projections (2026-09-21):** `figures/d12288_50_base/{pca,tsne,umap,centroids,variance}/<rung>.png`,
  five spaces per row (plain km, balanced km, d3072, d6144, d12288), 6k points, 8 classes, from
  `concept_geometry --projections tsne,umap,pca,centroids,variance`. UMAP and t-SNE agree with the
  kNN numbers: on topic14 the encoder-free controls (0.967) show tighter islands than d12288 (0.930).
  Fixed a layout bug in `scatter_row` while doing it — the note at y=0.985 was drawn through the
  two-line panel titles and the legend was anchored below the canvas; now `subplots_adjust(top=0.79,
  bottom=0.20)` with the legend at y=0.005. Affects every concept_geometry figure.
- **Encoder-free controls (2026-09-21, 1M tokens):** geometry is the AE's clearest win — balanced
  k-means on the raw residual gets silhouette −0.027 (dpc-init −0.034, raw k-means −0.094) vs
  d12288 +0.050, CH 225 vs 1072; balancing alone does not buy separation. Probes are the opposite:
  balanced k-means (no encoder) 0.327 on doc/topic vs d12288 0.310 (p=.34, ns) and 0.2557 vs 0.2503
  over all 22 rungs (p=.56) — the AE is significantly ahead only on token rungs (+0.011, p=.006).
  kNN-10 on the raw residual beats every arm on topic14 (.967 vs .930) and country. Controls:
  `balanced_kmeans_k2000{,_dpc}.npz`, `baseline_kmeans_k2000_refit.npz`; re-run via
  `--baselines "name=path,..."` (concept_probe/-geometry) and `--baseline <npz>` (clustering_quality).
- Caveat carried by every steer/range comparison: the d6144 reference JSONs were run on
  best_val (ep47), not step_0014200 (ep50). Only d3072 vs d12288 is strictly epoch-matched.

**Follow-ups done 2026-09-21:** `eval_out/run_d6144_ep50_evals.sh` re-runs the d6144 steer +
DB14/bios ranges at step_0014200 so the intervention series becomes epoch-matched (outputs get a
`_d6144_50` suffix; `summarize_width_series.py` now takes candidate-path lists and picks them up
automatically, leaving the ep47 files untouched). The reliability / effect-vs-spread / moderation
analyses quoted above are now sections of `summarize_width_series.py`, not ad-hoc code. Metric fix
in [[cq-dunn-and-steer-sign-traps]] means the cq numbers above predate it — re-run to refresh.

Related: [[d3072-width-arm]], [[biasbios-range-reversal]], [[range-interventions-h-vs-z]],
[[tanh-encoder-arm]], [[cq-dunn-and-steer-sign-traps]].
