---
name: supervised-finetune-ladder
description: in-batch supervision built 2026-09-21 (geoae/supervised.py + sup_* config knobs + fit_balanced_kmeans --space latent) and the 3-tier recovery ladder on the d12288 arm
metadata:
  type: project
---
**RESULTS (2026-09-21, d12288 parent).** The recovery is in the PARTITION, not the
representation. Tier 0 (supervised codebook on the frozen latent): probe NMI 0.2503 ->
0.2721 over 22 rungs (+0.0218, p=.009, 15/22), semantic rungs +0.0256 (p=.026). The
unsupervised control in the same space (latent_dpc) moved +0.0005 (p=.92) — so the gain
is the LABELS, not the refit; run that control or the result is uninterpretable. But the
gain concentrates where anchors are dense: atlas rungs +0.095 (3/3, p=.022) vs +0.008 on
the other 12 semantic rungs (p=.096). Read it as "25-shot prototypes make a better
codebook for the classes they cover", not as generalisation. Raw balanced k-means still
wins topic14 (0.569 vs 0.497). Tier 0 costs nothing — same encoder, so MMLU/ppl/geometry
are unchanged by construction. Seeded fit: 785 class means (448 of them atlas) + 1,215
peaks fill, 78th density pct, 2000/2000 live.
**Tier 1 (encoder fine-tune, decoder frozen) was NET NEGATIVE and confounded by a config
error of mine:** all rungs -0.0269, transfer -0.0184 (p=.019, 4/15). It did fit what it
was shown (5 non-conflicting supervised rungs +0.0305, p=.023, 5/5; sup loss 6.26 -> 4.6)
and cost nothing (val_mse 0.0193 -> 0.0175, MMLU -0.0130 -> -0.0100, silhouette 0.0470 ->
0.0493, eff rank 290 -> 299) — the damage is specific to concept structure.
**CAUSE: conflicting multi-labels.** ioi.npz's 24,000 rows carry BOTH ioi_role (3 classes)
and ioi_name (44), NMI(role,name)=0.0006 — fully orthogonal labelings of the SAME rows, so
each row entered the pool twice with contradictory positives/negatives. Both IOI rungs
collapsed (0.274 -> 0.039) and entity-flavoured transfer rungs (ravel_* -0.06..-0.08)
went with them. Nested pairs were fine (pos_coarse/fine NMI 0.87, ner 0.82, both up);
surface vs pos_* is 0.35, the remaining mild case. ioi_role is now removed from both
ft configs. **Before any rung goes in sup_rungs, check pairwise NMI of rungs sharing a
cache file.** A clean tier 1 rerun with the fixed list is the honest test of "does
supervision help the encoder"; tier 2 as originally configured would have repeated the bug.

- **Centroids are an EMA buffer, not a gradient parameter** (`register_buffer` in model.py,
  and total_loss gets `centroids.detach()`). So a "centroids-only gradient fine-tune" does
  not exist: the codebook-only question is answered OFFLINE by re-fitting on the frozen
  latent (`fit_balanced_kmeans --space latent`), which is tier 0.
- **Tier ladder:** t0 = refit codebook on frozen latent, supervised (`--init seeded`) AND
  unsupervised (`--init dpc`) — without the dpc control a t0 win can't be attributed to
  labels rather than to refitting at all. t1 = 5-epoch fine-tune, decoder frozen
  (`freeze: decoder`). t2 = full. Sheet: `eval_out/run_supervised_tiers.sh`.
- **New code:** `geoae/supervised.py` (LabelledPool with class/row holdouts, SupCon loss,
  bn_eval guard); `sup_frac/sup_rungs/sup_per_class/sup_m_per_class/sup_holdout_*/freeze`
  in TrainConfig, `lambda_sup` in LossConfig; `fit_balanced_kmeans --space latent --init seeded`;
  concept_probe accepts latent-space codebooks (npz carries `space` + `ae_checkpoint`, kind "aekm").
- **Design choices that matter:** labelled rows are ADDED to the step, not swapped into the
  batch, so Sinkhorn marginals never see the off-distribution labelled corpus; the labelled
  forward runs with BatchNorm in eval mode so it cannot move the running stats.
- **Calibration measured in the smoke:** λ_sup 0.02 × sup 6.26 = 0.125 = 18% of total loss
  (parent total 0.58) — in the intended 10-20% band, no tuning needed. Pool from the 7
  lexical rungs = 132,287 rows / 150 classes / 1.5 GB, ~7 replays per row per epoch at
  sup_frac 0.10. Resume lands at epoch 51 with tau pinned 0.100 (configs set tau_start =
  tau_end and zipf_alpha_start = end, else a resume REWINDS both and re-anneals).
- **Label supply is the real constraint:** doc/topic rungs hold only ~57k rows total (the
  atlas ones all from 7,742 FineWeb chunks) vs ~1.17M token-level rows. That is why the
  supervised rungs are lexical only and the semantic rungs are left untouched as the
  transfer measure. Scaling doc labels = extend the atlas extraction or use FineWeb
  metadata (URL domain, language id).
- Fine-tune checkpoints land at `step_0015620.pt` (14200 + 5x284). Read supervised rungs as
  "did it fit", semantic rungs as the result; `sup_manifest.json` in checkpoints_dir lists
  the 38 whole classes held out.

Related: [[d12288-width-arm]], [[init-arms-dpc-results]], [[db14-kmeanspp-diagnosis]],
[[fineweb-atlas-concept-alignment]].
