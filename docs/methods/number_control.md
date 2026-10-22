# Number agreement control

This constructed task tests whether range edits change next-token grammatical
number while preserving unrelated predictions. It uses the existing DPC L27
checkpoint without training another autoencoder.

## Why this task

The source is `results/closest_tokens_dpc.json`, rather than a judge's description.
Cluster 352 has 95 ` is` tokens among 103 assignments. Cluster 1603 has 85 ` are`
tokens among 181 assignments, alongside ` is`, ` be`, and other auxiliaries.
Cluster 1482 has 134 ` have` and 52 ` has` among 201 assignments. These examples
motivate testing an auxiliary-number distinction; they do not establish that the
clusters already disentangle number. The evaluation reads a state BEFORE the
auxiliary, whereas closest-token examples describe states AFTER their marked
token. Success therefore requires a predictive feature rather than token copying.

## Protocol

- Fit: 16 subject nouns, three templates, both subject numbers and both distractor
  numbers: 192 prompts. Example: `The gardener near the windows`.
- Test: 12 different subject nouns and three different templates: 144 prompts.
  Example: `The singers beside the gate` (actual noun allocation follows seed 42).
- Readout: next-token logits for ` is` versus ` are`. Both continuations must be
  distinct single tokens for the checkpoint's tokenizer. This is forced-choice
  grammatical accuracy; it is not unrestricted generation accuracy. Full-vocabulary
  counterpart top-1 and KL are also recorded.
- The distractor's number is balanced independently of the subject. Matching and
  opposite-number distractors are reported separately.
- Fit ranges and apply interventions at the last prompt token only.
- Run singular suppression and plural suppression separately. Each direction is
  applied identically to all evaluation prompts, including the complement and
  neutral prompts; test labels do not choose which rows receive the edit.
- Use global, salient, range and transport edits; tao 2, selected fraction 0.3,
  and alpha 0.5, 1, 2. Use `--saliency dprime` for an additional saliency control.

## Controls and measurements

1. Original residual, unedited model, and unedited GeoAE reconstruction.
2. GeoAE in its learned coordinates.
3. Three seeded orthogonal coordinate transforms. Refit ranges after rotation and
   invert before decoding. These transforms preserve reconstruction and Euclidean
   geometry but change coordinate-wise gates. They are structured orthogonal
   transforms, not Haar-distributed rotations. Global mean-difference edits must
   agree after inverse rotation, up to numerical precision.
4. Shuffled fit labels with the original class counts; true test labels are never
   shuffled. This is a negative control, not a guaranteed zero-effect arm.
5. Sixteen neutral completion prompts, scored for top-1 changes and full-vocabulary
   KL against each space's own baseline. Reconstruction damage relative to the
   original LM is saved separately. This small neutral set is not a perplexity
   benchmark or a comprehensive general-capability evaluation.

Report all test examples and a common subset correct under both the original and
reconstructed models. Selectivity is target accuracy drop minus complement drop.
Also save the full-vocabulary counterpart hit rate, pair probability, logit-margin
change, KL, gate firing, decoded edit norms and individual-example measurements.
Equal alpha or edited coordinate fractions need not imply equal perturbation size;
use the saved decoded norms to inspect that confound. Variation across prompts is
not independent of subject noun: any bootstrap should group by subject lemma.

## Completed d12288 replication

The epoch-50 `d12288` DPC checkpoint was evaluated with d-prime saliency on
2026-09-22 using the same 30% coordinate fraction, tao 2, alpha sweep, three
rotations and shuffled-label control. The full result is
`results/range_number_d12288_50_dprime.json`; the audited summary is
`eval_out/number_d12288_50_review.json` and the figure prefix is
`figures/number_control/d12288_50_review`.

At alpha 1, strict target flips were 80.6% base versus 95.1% AE for the range
edit (paired difference +14.6 percentage points, noun-bootstrap 95% interval
[+10.4, +18.8]) and 81.9% versus 95.8% for transport (+13.9 points,
[+10.4, +17.4]). All 144 test prompts were correct under both unedited
baselines, every raw-base arm reproduced the earlier run exactly, and the
shuffled-label arm produced no strict flips. Thirty percent selects 922 raw
coordinates but 3,686 d12288 latent coordinates, so this is a matched-fraction
rather than matched-count or matched-norm comparison.

## Commands

The launcher refuses to overwrite existing result files. The current GPU training
job is left running; execute these commands when evaluation should use the GPU.

```bash
# Inspect/save the full dataset and run configuration, without loading a model.
NUMBER_OUTPUT=eval_out/number_control_manifest.json \
  bash eval_out/run_number_control.sh --dry_run

# Small end-to-end run: 24 fit prompts, 24 test prompts, one rotation, alpha 1.
NUMBER_OUTPUT=results/range_number_dpc_smoke.json \
  bash eval_out/run_number_control.sh --smoke

# Full control experiment: all three rotations and the shuffled-label arm.
bash eval_out/run_number_control.sh
```

The completed result above applies only to its stated checkpoint and protocol.
For future variants, setup checks validate the experimental invariants and task
construction but do not substitute for running the LM evaluation.
