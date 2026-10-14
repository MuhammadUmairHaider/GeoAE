# Targeted Probe Perturbation (k50) audit

The implementation is `geoae/interp/probe_perturbation.py`. It ablates **input
representation coordinates**, not the probe's output units. A multiclass probe
still has all its class outputs after editing. It remains fixed throughout the
sweep; no language-model forward pass or probe retraining occurs.

## What the requested percentage means

For target class c, let A_c(0) be the initial target-class accuracy (recall), and
A_c(k) its accuracy after replacing the first k ranked coordinates with their
training means. The metric is:

```
k50 = first sampled k for which A_c(k) <= 0.5 * A_c(0)
percentage removed = 100 * k50 / representation_width
```

Thus 80% starting accuracy gives a 40% threshold. This is the literal 50% relative
drop requested; unlike the BiasBios gender operating point it does **not** subtract
a chance baseline. The original arithmetic was correct for this definition.
With 200 steps, k50 is measured on a roughly 0.5%-spaced coordinate grid. It is
not an exact minimum over every k or over all possible coordinate subsets.
The full curves and the preceding sampled k are saved, including nonmonotonicity.

## Problems found and fixed

1. **Unreached thresholds became 100% removal.** Both targeted and random sweeps
   substituted the full width when no crossing existed. New results use null
   plus an explicit status. Zero initial target accuracy or absent target rows
   make the threshold undefined rather than trivially achieved at k=0.
2. **Selectivity was measured after removing all coordinates.** At that endpoint
   the output is determined by probe biases and the ranking no longer matters.
   Selectivity now means target drop minus complement drop at the sampled k50.
3. **The evaluation cohorts differed by representation.** Each probe retained only
   its own correct predictions. The corrected evaluation uses all common test
   rows and records initial accuracy. There is no per-probe correctness filter.
4. **Fit/validation splits differed between spaces.** A single stratified split
   is now generated before fitting either representation, with source row IDs saved.
5. **Normalization differed and used validation rows.** Both raw and AE spaces
   now use coordinate statistics from the shared probe-fitting rows only.
   Ablation to zero therefore consistently replaces features with their train mean.
6. **Absolute probe weights were not softmax-invariant.** Adding the same weight
   vector to every class leaves predictions unchanged but changed old rankings.
   New rankings use target-versus-mean-class weight contrasts. A train-only
   d-prime selector is also available with `--ranking dprime`.
7. **Random aggregation hid important distinctions.** The crossing of an averaged
   random curve is not the mean or median of individual crossings. Both are now
   labeled separately; individual trials and censoring counts are saved. The
   localization ratio uses median trial k50 only when every trial reaches it.
8. **Comparison summaries could use different class sets.** Results report reach
   counts and a comparison mean over the same classes reaching k50 in every space.
   Reached-only means remain explicitly conditional, not unconditional averages.
9. **Probe fitting could be dominated by the original summed L1 penalty.** The
   audited trainer uses validation cross-entropy, convergence histories, zero L1
   by default, and light AdamW decay. Optional L1 is a mean penalty. These are
   documented defaults, not a claim that hyperparameters are optimally tuned.
10. Non-finite caches now fail instead of being silently repaired using test-set
    statistics. Class vocabulary size no longer depends on the number of labels
    appearing in a small subsample. Outputs refuse overwrites and save progressively.

The test-set k50 remains a descriptive response-curve metric. When using a budget
to claim held-out collateral preservation, a separate `validation_selected`
record chooses k50 on validation and evaluates that unchanged k on test data.
Its test target accuracy need not cross exactly 50%.

## Existing results checked

The four legacy `results/tpp*.json` DBpedia files were checked directly. Every
saved importance-ranked target curve reached its threshold, so that part of
their recorded k50 arithmetic was not invalid. Some random mean curves did not:
for example, `MeanOfTransportation` in the raw phased/semisupervised runs, and
`Building` in the semisupervised AE run. Those localization ratios used a false
full-width fallback.

The phased run compared 9,212 base-correct test rows with 8,659 AE-correct rows.
Consequently, its reported 336.3 versus 92.7 mean coordinates is not a controlled
same-cohort comparison. Recomputing selectivity from saved curves cannot repair
the training, scaling, ranking, or cohort differences; a fresh run is required.

`eval_out/tpp_audit/legacy_result_audit.json` preserves the per-class findings.
The original implementation was copied to `eval_out/tpp_audit/probe_before_audit.py`.

## Data and validation

The DBpedia cached labels match the source Arrow dataset after its documented
seed-42 extraction shuffle. Both activation arrays are finite. The 50,000-row
training cache has three duplicate texts; the 10,000-row test cache has no text
duplicates or train overlap. Audited text hashes are available at
`eval_out/tpp_audit/dbpedia_source_hashes.npz`, and the launcher uses them to remove
the three training duplicates before splitting.

Regression tests cover literal threshold arithmetic, censoring, zero/missing
baselines, nonmonotonicity, softmax invariance, direct-versus-incremental ablation,
validation-only budget selection, random trial aggregation, shared-cohort CLI
execution, missing classes, and non-finite caches. BiasBios regression tests are
also run because it imports the same probe class.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest \
  tests/test_probe_perturbation.py tests/test_bias_probe.py

# Full evaluation; choose fresh outputs for every configuration.
bash eval_out/run_tpp_audited.sh

# Discriminative selector comparison.
TPP_OUTPUT=results/tpp_dbpedia_dpc_audited_dprime.json \
TPP_PLOTS=figures/tpp_dbpedia_dpc_audited_dprime \
  bash eval_out/run_tpp_audited.sh --ranking dprime
```

Smaller k50 indicates greater vulnerability of this frozen readout to this
coordinate-ranking/ablation procedure. It does not, by itself, establish cleaner
semantic disentanglement. Report initial accuracy, collateral, censoring, count
and fraction: the DPC AE has 6,144 coordinates while the base has 3,072.
