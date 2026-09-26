Implementation plan: fine-tune GeoAE on enriched token activations

Status: proposed implementation, 2026-09-19. This document specifies the work;
the fine-tuning code and experiments have not been run.

The hypothesis is that a short adaptation stage on a more informative activation
distribution can preserve distinctions that the current AE loses. Start with
the trained Llama-3.2-3B layer-27, d6144, K=2000 DPC AE. Train the AE while keeping
the language model frozen. The first experiment changes the activation mixture
and keeps the reconstruction and geometry objective fixed between arms.

The comparison is:

| Arm | Starting point | Adaptation data | Purpose |
|---|---|---|---|
| Parent | Explicit epoch-50 checkpoint | None | Measure the starting behavior |
| Replay | Same parent | 100% original training activations | Control for additional optimization |
| Enriched | Same parent | 80% original, 20% enriched activations | Test the enrichment intervention |

Enriched minus Replay is the main experimental contrast. Enriched minus Parent
alone cannot separate enrichment from the effect of additional training. The
80/20 mix and the numerical settings below are starting defaults, not tuned
optima. This experiment tests an enrichment bundle; separating new subject matter
from new prompt formats would require a subsequent ablation.

**1. Pin the parent, data, and evaluation populations.**

Use `checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/step_0014200.pt`
when available. Verify its saved epoch, architecture, layer, and initialized
centroids, and record a content fingerprint. On a fresh Delta checkout, regenerate
the parent using the existing extraction/training recipe and record the new
fingerprint. Both adaptation arms must use that exact regenerated parent. An
arbitrary `best_val.pt` is not a substitute for a completed geometry checkpoint.

Pin source dataset revisions and model/tokenizer revisions when building the
experiment manifests. Keep all paths relative to the repo or configurable data
root. Persist source IDs and normalized-text hashes rather than relying on a
dataset's current iteration order. Reserve evaluation documents before preparing
the enriched corpus.

Build common fit/evaluation manifests for DBpedia-14, bias_in_bios, and AG News.
Reserve AG News entirely from enriched training as the first transfer check.
Use separate original documents for fitting directions/ranges and evaluating
edits. Freeze complement document lists, prompt templates, generation settings,
PPL text IDs, and concept lists. Keep the existing teacher-class exclusion in
bias_in_bios fixed for this experiment and report its reason.

**2. Prepare a small, reproducible enrichment corpus.**

The first corpus uses official training splits of DBpedia-14 and bias_in_bios,
which the repo already supports. Target 5,000 source documents from each dataset
for the pilot, sampling across their labels with explicit quotas. Labels guide
sampling and diagnostics; the initial AE objective does not predict them. Record
any quota shortfalls rather than silently substituting another source.

Assign source documents to adaptation train/validation, 95/5, before making
variants. All variants of a source document stay in the same split. Group known
entity duplicates and near-duplicate texts where detectable, exclude normalized
text duplicates of reserved evaluation documents, and write an overlap report.
This does not establish that the frozen LLM or the historical AE pretraining dump
never encountered related material; document that limit explicitly.

Produce three deterministic views of each source document:

- Original text.
- A classification instruction followed by the text and an unanswered response
  cue, such as `Category:` or `Profession:`.
- A differently worded question about the same source text followed by `Answer:`.

Use versioned template files with several variants. Reserve a template family
for validation/evaluation. Do not put the gold answer before an activation
designated as a pre-answer state. Learned paraphrase generation, synthetic factual
descriptions, and external generation APIs can be added later; deterministic
formatting is sufficient for the first experiment and regenerates from git.

Add `geoae/prepare_finetune_data.py` to produce JSONL records containing source
dataset/revision/split/ID, original-text hash, group ID, label, assigned split,
template version, and rendered text. Store corpus sizes, quotas, exclusions, and
the manifest fingerprint alongside the records.

**3. Extract both contextual and endpoint activations.**

Add `geoae/extract_finetune.py`, reusing the existing frozen-LM loading and
layer-hook utilities. Process the full rendered context, then select rows to
store. Use a configurable 1024-token limit; shorten the document body to preserve
the instruction and final response cue. Log truncation and retained positions.

For each view retain its final non-padding state and up to 15 reproducibly sampled
body/prompt states. Keep endpoint and contextual position metadata. Sample the
enriched portion of training batches half from endpoints and half from contextual
states, with dataset/template quotas, to avoid burying the decision states among
ordinary tokens. Original-data replay still supplies broad token coverage.

This addresses an observed mismatch without changing evaluation semantics:
directions/ranges are fitted on final prompt states, while current intervention
hooks edit all positions, including generated tokens. Endpoint-only adaptation
would therefore omit part of the distribution affected by edits.

Write separate train/validation memmaps and row metadata. Ten thousand documents
times three views times at most 16 states gives at most 480,000 retained rows,
about 2.95 GB for one 3072-wide fp16 layer, before metadata. Count processed LM
tokens separately from retained activation rows. Preserve the existing overflow
and finiteness checks. The pilot only extracts this additional corpus.

**4. Add a dedicated fine-tuning path.**

Add `geoae/finetune.py`, `geoae/finetune_config.py`, and
`geoae/finetune_data.py`. Keep ordinary pretraining resume behavior intact.
Provide distinct initialization and resume modes:

- `--init_checkpoint` loads the parent weights, all model buffers, centroids,
  normalization, and effective assignment settings, then creates a fresh optimizer
  at the fine-tuning learning rate and starts an adaptation step counter.
- `--resume` restores an interrupted adaptation, including optimizer, completed
  step, sampler, diagnostics history, and RNG state. It must not reset the run.

Preserve BatchNorm running statistics/counters, centroid EMA counts, and
`sinkhorn_g` at initialization. Use the checkpoint's exact mean/std for both data
sources and validation; do not recompute normalization on the enriched data or
overwrite normalization caches belonging to the original dump.

Derive the effective reconstruction, clustering, separation, and VICReg weights,
temperature, and Zipf exponent from the parent's saved epoch and configuration.
Hold them fixed during the short adaptation. Reuse the same `total_loss` call,
including `sep_mode`, `sep_margin`, and `var_gamma`. Extending the original
`n_epochs` would change its schedules and is not the desired implementation.

Keep decoder renormalization, BatchNorm updates, centroid EMA, and the parent's
peak-based dead-centroid reseeding rule identical between arms. Preserve reseeding
cadence using parent step plus adaptation step. Initialize the usage-history
window identically in both arms and checkpoint it on adaptation saves. Retaining
the starting centroids does not mean freezing their later EMA updates.

Mix original and enriched rows within each batch and shuffle the combined batch
before one forward pass. Alternating source-specific batches changes BatchNorm
and Sinkhorn behavior and would add another experimental difference. Use a
deterministic quota accumulator for fractional row counts and independently seeded
source permutations. Select original rows only from the historical training split.

Checkpoint sampler progress at the last completed optimizer step, not at a
prefetched DataLoader cursor. Save Python, NumPy, CPU/CUDA RNG state and restore
it after constructing the model/loaders. Guarantee reproducible sample order;
numerical reproducibility is conditional on hardware and deterministic kernels.

The checkpoint format retains `model_state`, `norm_mean`, `norm_std`, and the
effective configuration expected by existing evaluators. Add parent fingerprint,
adaptation step, explicit effective runtime settings, data fingerprints, and
exposure counters. Update `geoae/checkpoint.py` to prefer explicit runtime settings
and retain its legacy fallback. In particular, `zipf_alpha` is not a model buffer:
reconstructing it from adaptation epoch 1 would silently load the wrong assignment
rule. Test all evaluation checkpoint-loading paths against this case.

**5. Run a bounded pilot with fixed validation.**

| Setting | Initial value |
|---|---|
| Model | Parent d6144 architecture, all AE parameters trainable |
| Global batch | 32,768 rows, matching the parent |
| Optimizer | Fresh optimizer of the parent's type/settings, lower LR |
| Learning rate | 2.8e-5, one tenth of the current parent's configured LR |
| Adaptation budget | 300 updates per arm; 9,830,400 row exposures |
| Replay fraction | 1.0 for Replay; 0.8 for Enriched |
| Pilot seed | 42 for both arms |
| Saved checkpoints | Steps 0, 100, 300; final step is the primary comparison |
| Confirmation seeds | 43 and 44, with the same parent and fixed data manifests |

The enriched arm sees approximately 1.97 million enriched-row exposures during
this pilot; report unique rows/documents and repetition counts separately. These
seeds measure adaptation variability conditional on one parent, not pretraining
variability. A hardware-driven batch reduction must apply to both arms and be
recorded as a new experiment setting; gradient accumulation does not automatically
preserve BatchNorm or Sinkhorn semantics.

At step zero, require reconstruction and assignment outputs to match the parent
before adaptation. Validate on fixed original and enriched holdout rows, reporting
each source, template, and position stratum separately. Aggregate by row count,
not an unweighted average of potentially unequal batches. Record reconstruction
MSE/FVE, loss components, centroid usage/churn, latent scale, and actual exposures.
Use final fixed-step checkpoints for the main experiment; validation can diagnose
problems without silently selecting different training durations for each arm.

**6. Make the intervention comparison fair across arms.**

Add `geoae/interp/eval_manifest.py` and a `--eval_manifest` option to the range
and dense-steering harnesses. In this mode bypass checkpoint-specific
joint-correct set rebuilding and per-checkpoint refiltering. Existing behavior
remains available for reproducing historical results.

Include versioned prompt/template definitions in the manifest and pass them
explicitly through `_shared.py` prompt construction. The current hard-coded
per-dataset prompt is the legacy default. Evaluate the same source documents under
the existing template and a reserved template family, with separate results per
template. Group template variants of one document when estimating uncertainty.

Persist per-document baseline and intervention outcomes with IDs. First report
reconstruction accuracy/damage on the entire fixed population. Then use one
common mask of documents correctly classified by the base LM and every compared
AE for conditional intervention comparisons. Apply that same mask to every arm
and report coverage by concept; losing baseline correctness must remain visible.
Fit directions/ranges on the fixed fit documents, without arm-specific filtering.

Add `geoae/interp/compare_finetune.py` to check manifest compatibility and summarize:

- Range and transport steering with d-prime saliency, `tao=2`, and 30% salient
  dimensions; dense steering as a secondary comparison.
- A prespecified alpha grid, initially `[0, 0.25, 0.5, 1, 2, 4]`, and collateral
  at 50% and 80% target erasure, always relative to each substrate's own baseline.
- Piecewise interpolation at those prespecified erasure targets. Use the first
  crossing in increasing-alpha order, flag non-monotone curves, and never
  extrapolate. Report paired eligible concept counts and actual reach. Any later
  tuning of an operating alpha uses a separate calibration split.
- Mean-replacement range removal as a fixed-operation secondary result; it is
  not automatically a matched-erasure result.
- Per-concept and per-domain Enriched-minus-Replay differences, paired confidence
  intervals, and each adaptation seed. Keep uncertainty across documents,
  concepts, and seeds distinguishable; do not treat all tokens as independent.

Compute the base-h reference once or require identical h results as a wiring
check. Standardize the new report's sign: collateral(z) minus collateral(h),
negative is better; preserve historical metric fields with their documented
signs. Keep checkpoint width, intervention positions, and dimension-selection
settings constant between adaptation arms.

Evaluate DBpedia and bios as adaptation domains and AG News plus held-out prompt
templates as transfer checks. AG News has only four concepts, so it alone cannot
establish broad transfer. Run reconstruction PPL and MMLU on the final confirmation
checkpoints to quantify behavioral cost. Reuse the existing concept-probe suite
as a secondary geometry check.

The intended result is reduced bios collateral compared with Replay, retained
DBpedia selectivity, and preserved reconstruction/general behavior. Proposed
regression tolerances to freeze before the pilot: original-holdout MSE and
reconstruction PPL no more than 5% above either Parent or Replay; DBpedia collateral
no more than 0.01 above either reference at each reachable target; full-cohort
reconstruction accuracy on each of DBpedia, bios, and AG News, and MMLU accuracy,
no more than one percentage point below either reference. Also report retention
of the parent/base-correct documents by arm and domain, targeting at least 95%,
along with final common-mask coverage. A conditional bios win accompanied by
excessive reconstruction failures is not a useful improvement. These are
experimental budgets, not known optimal thresholds. Report uncertainty and
inconclusive results. Improvement restricted to enriched domains is evidence of
adaptation, not broad generalization.

**7. Package, verify, and run on Delta.**

Commit recipe YAMLs under `configs/finetune/`, template/source definitions, the
preparation/extraction/training/report entry points, and documentation. Generated
text corpora, activations, and checkpoints remain regenerable artifacts. Record
manifest hashes and small summaries in results. No new credentials or generation
service are required beyond the existing model/dataset access.

Add Delta wrappers for enrichment preparation/extraction and adaptation. Stage
both activation pools on local storage when feasible, retain manifests with the
run, and resume only that arm's adaptation checkpoint. Use one GPU per independent
arm/seed, scheduled concurrently. This pilot does not depend on adding DDP; shared
Sinkhorn/centroid synchronization would be a separate implementation project.
The workflow from a fresh git clone regenerates the parent if it is absent.

Implement and test in this order:

| Work unit | Deliverable and verification |
|---|---|
| A: contracts and manifests | Source/template/config schemas; document-group split and overlap tests; fixed eval manifest |
| B: enriched extraction | Correct layer/positions, preserved response cue, padding exclusion, finite rows, deterministic metadata; tiny frozen-LM smoke |
| C: checkpoint and trainer | Step-zero fidelity; correct lower LR/fresh optimizer; unchanged normalization/runtime state; mixed batch quotas |
| D: interruption handling | Continuous versus interrupted tiny CPU run matches consumed rows, optimizer, BN/centroid/dual/history states, and outputs within tolerance |
| E: evaluation/report | Identical populations/complements across arms; visible baseline failures; interpolation/coverage tests; legacy checkpoint compatibility |
| F: Delta integration | Shell syntax and command smoke checks; 10-update GPU adaptation save/reload; one-concept range smoke; then 300-update pilot |

Run the affected existing checkpoint, resume, data, model, loss, and intervention
tests with the new meaningful tests. The first full experiment begins only after
the technical smoke checks pass. Inspect the pilot once, then repeat both arms
with the confirmation seeds using the frozen recipe; do not silently retune on
final evaluation results.

After this experiment, consider two separate extensions: a small concept
supervision term and a short KL+MSE adaptation stage. The latter needs objective
parity fixes first: the current E2E loss call does not propagate all separation
and variance settings. Neither extension is part of the initial enrichment test.

Relevant existing implementation: `geoae/train.py`, `geoae/train_common.py`,
`geoae/data.py`, `geoae/checkpoint.py`, `geoae/extract.py`,
`geoae/interp/_shared.py`, `geoae/interp/range_intervention_compare.py`,
`geoae/interp/steering_concept_compare.py`, and `HANDOFF_DELTA.md`.
