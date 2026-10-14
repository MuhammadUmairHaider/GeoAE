# Gemma 3 1B: extraction → DPC AE → evaluation

Status: **extraction and 50-epoch AE training complete; evaluations in progress.**
See the [results report](gemma3_1b_dpc_results.md) or its
[HTML version](gemma3_1b_dpc_results.html) for the latest dated snapshot.
The report is refreshed on request with
`.venv/bin/python eval_out/update_gemma_report.py`.

## Recipe

| Setting | Value |
|---|---|
| Frozen language model | `google/gemma-3-1b-pt` (pretrained, not instruction-tuned) |
| Residual location | After decoder layer **25**, the last of 26 layers |
| Residual / latent width | **1,152 → 2,304** (2× overcomplete) |
| AE | GELU, BatchNorm, K=2,000, density-peaks initialization and peak reseeding |
| Extraction | 10M token budget; web/wiki/code/math/pile; 256 tokens/document; skip first 4 |
| Compute / storage dtype | Frozen LM bfloat16; activation `.npy` float32; no clipping |
| Training | 50 epochs, batch 32,768, LR 2.8e−4, seed 42, 5% validation tail |
| Schedule | Reconstruction warmup; VICReg from epoch 6; clustering from epoch 11 |
| Objective | Same MSE-based DPC objective as the main Llama arm; **not** KL end-to-end training |
| Evaluation checkpoint | Final epoch, verified initialized centroids and optimizer LR; never an unverified `best_val.pt` |
| Artifacts | `runs/gemma3_1b_l25_d2304_dpc_seed42/` |

Configuration: [gemma3-1b_l25_k2000_d2304_dpc.yaml](../configs/base/gemma3-1b_l25_k2000_d2304_dpc.yaml).

## Space and activation-range check

Machine availability and activation ranges were measured on 23 September 2026
using the actual frozen model and the same layer, document length and
leading-token skip as extraction. That probe used the earlier 5M recipe; the
storage estimate below reflects the current 10M budget:

- **225 GiB disk free after caching the model**; approximately 109 GiB host RAM available at preflight.
- 10M × 1,152 × 4-byte activations = **46.08 GB / 42.92 GiB**. The launcher additionally requires 20 GiB of free headroom for model/cache/checkpoint/temp artifacts, for **62.92 GiB** total required free space.
- Available GPU: NVIDIA A100, 40 GB. The full recipe requires at least 24 GiB GPU memory and 20 GiB available host RAM; it runs stages serially.
- Probe: **22,852 retained tokens**, 100 documents, 20 documents per domain across all five source domains.
- Largest channel: **1038**, median absolute activation **27,392**, maximum **55,552**. The next-largest channel's maximum was 5,952.
- **Zero nonfinite values and zero float16-range overflows in this sample.** This is not a guarantee for the full corpus.

This model does have a pronounced large-activation channel, but the sampled values
do not approach bfloat16's maximum (**3.3895×10³⁸**). Float16's maximum is only
**65,504**. Clipping at the bfloat16 maximum would not change these values, and
would not solve a future float16-storage overflow. Clipping at the float16 maximum
would alter the residual stream and would need its own reconstruction/behavior
control. There is adequate space, so this recipe preserves values in float32.
Per-channel normalization is fitted by the AE trainer on its training split.

Probe artifact: [gemma3_1b_l25_activation_probe.json](../eval_out/gemma3_1b_l25_activation_probe.json).
To repeat without overwriting that result:

```bash
.venv/bin/python -u -m geoae.gemma_activation_probe \
  --out eval_out/gemma3_1b_l25_activation_probe_repeat.json
```

The probe loads the model but does not write the full activation dump or train the AE.

## Commands

From the repository root:

```bash
# Preview commands only: no files, downloads or GPU jobs.
.venv/bin/python -m geoae.gemma_dpc plan

# Check free space, GPU/RAM, model config/access and number-task tokenization.
# Downloads metadata/tokenizer if needed, but does not load model weights.
.venv/bin/python -m geoae.gemma_dpc check

# Explicitly launch the full extraction -> training -> eval sequence.
.venv/bin/python -u -m geoae.gemma_dpc run

# Read verified completion markers.
.venv/bin/python -m geoae.gemma_dpc status
```

Hugging Face model access must already be authorized. Existing HF login/environment
credentials are used; the launcher does not accept a model license on your behalf.
W&B is disabled for this recipe, and no paid judge/API evaluation is launched.

To run a subset, prerequisites are added automatically:

```bash
# Extraction + validation only.
.venv/bin/python -u -m geoae.gemma_dpc run --stages validate_activations

# Through training and number-control evaluation.
.venv/bin/python -u -m geoae.gemma_dpc run --stages number

# Finish the remaining pipeline after an interrupted training run.
.venv/bin/python -u -m geoae.gemma_dpc run --resume-training
```

Running `run` with no stage selection includes every configured stage. Selecting
`summary` also includes every prerequisite. To change the recipe, copy/edit the
YAML and use a **new** artifact directory:

```bash
.venv/bin/python -m geoae.gemma_dpc plan \
  --config configs/base/gemma3-1b_l25_k2000_d2304_dpc.yaml \
  --run-dir runs/gemma3_1b_l25_dpc_second_run
```

## Evaluation coverage

| Stage family | AE measurement | Explicit base/control |
|---|---|---|
| MMLU, 2,000 examples | AE reconstruction splice | Original frozen Gemma |
| Cluster geometry, 1M sampled states | Learned AE codebook | Balanced raw k-means, balanced raw DPC, plain raw k-means |
| Concept probes, 22 tasks | Cluster-label NMI and best-cluster F1 | All three raw-space codebooks |
| Local geometry | PCA/t-SNE and logged neighborhood diagnostics on five tasks | Raw h / balanced-base codebook |
| DB14 dense steering | AE edits at α=.5,1,2,4,8 | Paired raw-h edits |
| DB14 / AG News / BiasBios range suppression | d′ top-30%, τ=2; range/full removal and global/salient/range/transport steering | Same operators on raw h, paired joint-correct samples |
| Number agreement | 144 arms; α=.5,1,2; d′ top-30%, last-position edits | Raw h, three structured rotations, shuffled-label control |

Each benchmark cache is built anew with Gemma 1B, including POS/NER, sequence
tasks, RAVEL, IOI and Atlas; no Llama or older Gemma activation cache is reused.
Atlas uses last-token states from the BF16 model, stored as float32; document labels
must be single-label, tone uses the rarest label and content the most common label,
with support ≥25 and deterministic label-ID tie-breaking. These are descriptive
cluster probes, not new held-out supervised classifier scores. DPC uses no label
anchors, so `--exclude_anchors` is deliberately absent.

The baseline codebooks use the AE's normalization and the same Gemma corpus.
The balanced baselines use uniform Sinkhorn assignments, while the AE uses
Zipf balancing, so these comparisons do not isolate the encoder alone. Baseline
reseeding is checked every 125 steps, matching the configured AE interval; the
fitter's default of 1,000 steps would never fire during this 240-step baseline run.
Their fit/geometry samples can overlap: geometry is a descriptive diagnostic,
not a held-out generalization estimate. The 5% AE validation split is a row tail,
not a document-level disjointness guarantee.

All 28 BiasBios classes are requested; Llama's teacher exclusion is **not** copied
to Gemma. Low joint-correct coverage can make some classes unmeasurable; inspect
the logs and evaluated concept counts. The 1B model may have lower baseline
accuracy than Llama. Number-control all-example and joint-correct results are kept
separate; the old reviewer that assumed 100% baseline accuracy is not reused.

This first launch recipe does **not** include the separately audited DBpedia TPP,
BiasBios gender-removal probe, linear/MLP utility suites, morphology, or paid LLM
judges. Those need additional Gemma-specific labeled caches and/or protocol work;
the launcher does not silently reuse their Llama artifacts or claim they were run.

## Artifact layout and failure handling

```text
runs/gemma3_1b_l25_d2304_dpc_seed42/
  config.yaml                 # resolved config actually passed to the trainer
  manifest.json               # commands, config and source hashes
  preflight.json
  activations/                # layer_25.npy, meta.json, train normalization
  activation_validation.json # all-row finite scan, dimensions, source coverage
  checkpoints/               # rotating step files, best_val, sealed eval_final.pt
    selected.json             # final checkpoint epoch, SHA256, actual optimizer LR
  baselines/                  # 3 matched-K raw-space codebooks
  cache/                      # model-specific 22-task and joint-correct caches
  evals/                      # raw JSON evaluations, including both base and AE
  figures/                    # PCA/t-SNE
  logs/                       # one append-only log per stage
  state/                      # completion records with output size/mtime
  summary.json                # consolidated machine-readable results
  summary.md                  # base-versus-AE tables and source links
```

The launcher holds an exclusive per-run lock, stops on the first failing stage,
and marks completion only after successful process exit and output checks. A
changed config/source manifest or modified completed artifact is rejected. It
does not overwrite previous results or treat partially written JSON as success.
For an interrupted non-training stage, preserve/move aside its partial outputs
before rerunning; `--resume-training` applies specifically to training checkpoints.
The extractor itself is not resumable. The final checkpoint is hard-linked, so
sealing it does not double its disk use and does not select a warmup best-val model.

### Recover the initial single-class DBpedia cache

The initial run stopped at `validate_cache`: DBpedia's training split is ordered
in 40,000-row class blocks, so the old 20,000-row streaming shuffle buffer sampled
only class 0. Sequence-task sampling now shuffles the full dataset before taking
examples and checks label diversity before encoding.

For that existing run, use the repair command before resuming:

```bash
.venv/bin/python -u scripts/repair_gemma_concept_cache.py
.venv/bin/python -u -m geoae.gemma_dpc run
```

The repair rebuilds the seven sequence-task caches in a staging directory,
validates the complete concept suite, and preserves the old files and provenance
before installing the replacement caches. It accepts only this specific sampling
source change. Extraction, AE training, baseline fits, and the other caches are
reused. The repair itself does not launch evaluations; the second command does.

Some existing eval CLIs use fixed output filenames or permissive defaults. The
wrapper isolates MMLU's fixed output in this run's `evals/`, passes explicit cache,
baseline and checkpoint arguments, validates all 22 concept rungs, and verifies
the number run's `complete` status and 144 arms. Legacy MMLU scoring is generated
one-token A/B/C/D rather than constrained-choice likelihood; interpret small-model
accuracy accordingly. Clipping is neither enabled nor needed for safe storage.
