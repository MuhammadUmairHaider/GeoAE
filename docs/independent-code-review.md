# Independent source review — 2026-09-19

Reviewed source at commit `9844f3ac4b1833b251ebf7dc47f25615a6da9c38`.

This assessment comes from the executable code, configuration, tests, and fresh synthetic checks. It does not use previous agents' conclusions, saved experiment scores, handoff documents, or the earlier fine-tuning plan as evidence. Source comments and docstrings were excluded from the main code reads. This is a review of what the implementation supports, not a ranking of trained checkpoints.

Scope: the maintained `geoae/` package (76 Python files, 20,094 lines), the test directory (19 Python files including its empty initializer), 89 YAML configurations under `configs/` and `dbpedia/configs/`, package/dependency configuration, and training/evaluation launchers. The directory tree was inventoried. Generated activation dumps, checkpoints, plots, logs, historical audit snapshots, agent notes, credentials, and migration history were not treated as research evidence.

**Conclusion:** improving the distribution of contexts is a sensible next training experiment. However, the implementation has a more fundamental ambiguity: it optimizes latent geometry, while useful control depends on decoded changes and downstream behavior. Compression and better initialization are hypotheses worth testing separately. Neither follows as a necessary explanation from the code.

## 1. What is actually being learned

The language model already supplies pretrained representations. The AE learns a map of its frozen residual activations:

```text
text → frozen LM → residual h → standardize → encoder → dense z → linear decoder → reconstructed h
                                                        ↓
                                                 distances to centroids
                                                        ↓
                                                 assignments and geometry losses
```

The encoder is a linear map, optionally followed by BatchNorm and an activation. The decoder is linear, has no bias, and normally has unit-norm columns. Reconstruction uses `decoder(z)` directly; it does not reconstruct from a cluster identifier or a weighted combination of centroids. There is no sparse activation objective in the current AE. See [model.py](../geoae/model.py), especially construction at lines 127–146 and forward at 281–330, and [losses.py](../geoae/losses.py).

Consequences:

- Increasing the number of clusters changes the partition and regularization, not the capacity of a quantized reconstruction bottleneck.
- A wider dense latent is not itself information compression. Narrower latents restrict the linear reconstruction subspace; whether this removes nuisance variation or useful information needs behavioral measurement.
- Cluster membership is a partition of contextual states. Overlapping concepts and selective coordinate edits are additional properties to establish, not properties directly guaranteed by that partition.

The existing E2E path uses teacher/student KL with a frozen LM. That is already a distillation-style training route; reinforcement learning is not needed to use it. KL preservation alone still does not train a desired counterfactual edit.

## 2. Geometry has two separate gaps to causal control

Let `W` be the decoder matrix and `S = diag(norm_std)`. For a latent change `delta_z`, the corresponding raw residual change is `S W delta_z`. Its squared size is:

```text
delta_z.T @ W.T @ S.T @ S @ W @ delta_z
```

The geometry losses primarily use distances in `z`, rather than this decoder-induced metric. With a 6,144-dimensional latent and 3,072-dimensional residual, `W` necessarily has a nullspace of dimension at least 3,072. Some latent changes can therefore have no decoded effect. Unit decoder-column norms do not remove that nullspace. This is an architectural possibility, not evidence that the trained model actually places its concepts there.

Separately, Euclidean cluster geometry survives an orthogonal rotation of latent coordinates. Coordinate selection and interval gating generally do not. Thus good separation is insufficient to establish useful coordinate alignment, even with a full-rank decoder. Variance/covariance penalties can favor particular statistical structure, but do not explicitly align coordinates with the concepts being edited.

Recommended diagnostics before changing the loss:

- Compare latent and decoded distances between relevant centroids and concept means.
- Measure decoder-visible versus nullspace components of actual edit directions.
- Retain orthogonal-rotation controls for range edits.
- Compare interventions at matched decoded edit magnitude or matched collateral damage, using validation to select operating points.

The number-control harness already has rotation controls and decoded edit norms: [range_number_control.py](../geoae/interp/range_number_control.py), lines 151–170 and 350–394. Extend this implementation rather than creating another unrelated harness.

## 3. Dense latent steering is a fixed decoded direction plus reconstruction

The current dense edit in [neuronlens.py](../geoae/interp/neuronlens.py), line 160, satisfies exactly:

```text
h_edited = h_reconstructed - alpha * S W r_z
```

When `r_z` is a difference of latent class means, `S W r_z` is the difference of reconstructed residual class means. This follows from decoder linearity. A successful dense latent edit can be useful, but does not by itself establish a special nonlinear intervention mechanism. Range-gated edits are different because their selected coordinates depend on the current state.

Add these explicit controls:

| Arm | Purpose |
| --- | --- |
| Original residual, no edit | Measure baseline behavior on all examples. |
| AE reconstruction, no edit | Measure reconstruction damage. |
| Original residual plus decoded latent edit | Isolate the edit from reconstruction. |
| Reconstruction plus the same decoded edit | Verify equivalence to the existing latent implementation. |
| Raw mean-difference edit | Compare direction quality. |
| Range edits in learned and rotated coordinates | Test whether coordinate alignment matters. |

For general latent edits, the residual-preserving splice is `h + S W (z_edited - z)`. Keep the existing full-reconstruction arm too: these answer different questions. In particular, preserving the original residual may allow a compact control representation without requiring that representation to reconstruct every detail of the LM state.

## 4. Better context sampling has a concrete basis in the extractor

[extract.py](../geoae/extract.py) iterates a hardcoded source list, takes document prefixes, and stores consecutive token states after skipping leading positions. Default document length is 256 tokens. The extraction loop does not shuffle each source. Round-robin document selection is also not an exact token-balanced mixture when document lengths differ.

The main activation dump does not retain a row-level mapping to document identity, source, token identity, and position. [data.py](../geoae/data.py), line 51, then splits the dump by a contiguous row boundary. Different rows from the same document can straddle that boundary; unseen documents, domains, and templates are not explicitly defined.

This supports a specific data hypothesis: **more distinct contexts, positions, and task formats per training budget may matter more than more adjacent token rows.** It does not establish how large that gain will be.

Build extraction around document manifests and selectable activation positions. Save dataset revision, document/content hash, source, window offset, token position, tokenizer/model revision, layer, and computation/storage dtype. Split and deduplicate documents before extraction. Sample windows beyond document starts, preserve sufficient preceding context, cap rows per document, and explicitly mix ordinary token states with meaningful final-context/readout states. Adding labeled benchmark-domain data should be described as targeted adaptation; generalization requires separately held-out domains or constructions.

There is already partial infrastructure worth reusing: `manifold_map.py` records document/token ownership, `prepare_atlas_probe.py` handles grouped data, and the audited probe paths save source-row splits and perform deduplication.

## 5. Current evaluation answers several different questions

The main classification steering/range harnesses construct examples that both the base LM and the particular AE reconstruction classify correctly. The cache is keyed by AE fingerprint, so each checkpoint can have a different population: [_shared.py](../geoae/interp/_shared.py), lines 195–335. A within-run `z - h` comparison is paired, but differences between such comparisons still involve changing populations. Report a fixed complete test population first, then any explicitly defined common-correct subset as a secondary analysis.

Classification uses greedy generation of at most six tokens and exact string matching. Confidence uses only the first token of the class name. These measure formatting and completion as well as semantic classification. Use full-label likelihoods or a constrained readout alongside generation, and report invalid-format outputs separately. Ensure truncation preserves the response suffix.

Cluster assignments are inconsistent across tools: DBpedia evaluation uses `Q.argmax` in batches of 512, whereas concept probes use nearest-centroid distances. Sinkhorn assignments depend on batch composition; nearest-centroid assignment does not. Several direct `torch.cdist` paths also ignore the configured cosine metric. Establish one explicit pointwise evaluation assignment rule and report balanced training assignment separately.

The number-agreement harness is a stronger starting point: held-out nouns/templates, matched fit/edit positions, all-example results, rotations, label shuffles, full-vocabulary KL, and recorded examples. The audited BiasBios/probe paths also have useful safeguards: shared splits, train-only standardization, validation-selected budgets, and explicit limitations on interpreting fixed-probe ablation as information erasure.

Morphology evaluation holds words out of the direction fit, but its corpus is not proven held out of AE training. Its standard error treats positions as independent despite shared words/documents, and its sparse, signed target-position edits differ from the global one-direction edits used for its perplexity measurement. Use grouping for uncertainty estimates and align the intervention being measured.

Additional interpretation limits: token-string entropy is not semantic monosemanticity; best cluster/class matching on the same sample is descriptive rather than held-out prediction; low-dimensional projections are visualization rather than evidence of causal selectivity. The direct-logit tool's last-layer `exact` label also overstates its calculation: multiplying by RMSNorm gains omits the state-dependent normalization factor and its change under an intervention.

## 6. Concrete implementation issues to fix before the next comparison

| Priority | Finding and source | Practical consequence |
| --- | --- | --- |
| High | Streaming resets the accumulation counter and EMA lists at every epoch, without flushing or clearing pending gradients: [train_stream.py](../geoae/e2e/train_stream.py), lines 320–371. | A partial group from one epoch contributes to the next full group with the wrong weight; its EMA data are discarded. Final pending gradients are never applied or checkpointed. Reproduced with the real loop and a toy model. |
| High | Cached training computes geometry on shuffled token batches; streaming computes it on one document at a time before gradient accumulation. | BatchNorm, covariance, separation and Sinkhorn are different objectives in the two paths. Accumulation does not produce equivalent geometric batches. Token counts alone cannot make these comparisons fair. |
| High | `--resume` restores optimizer parameter groups after constructing the optimizer with the new configuration: [train_common.py](../geoae/train_common.py), line 393. | A requested fine-tuning learning rate can silently revert to the saved rate. Add a distinct model-only warm start; keep strict continuation separate. |
| High | `best_val.pt` minimizes MSE across all phases: [train.py](../geoae/train.py), lines 438–443. | It can select a reconstruction-warmup checkpoint before centroid initialization. Record selection phase and retain separate reconstruction and geometry-qualified choices; do not assume a filename selects a trained clustering model. |
| Medium | `use_sinkhorn` is a plain attribute, omitted by checkpoint save/load: [model.py](../geoae/model.py), line 88; [checkpoint.py](../geoae/checkpoint.py), line 59. | Loading a no-Sinkhorn model reenables Sinkhorn. This changes assignments, though it does not directly change encoder/decoder reconstruction. Reproduced. |
| Medium | E2E callers omit configured `sep_mode`/`sep_margin`; the E2E loss does not accept `var_gamma`. Streaming also hardcodes k-means++ initialization and uses the default reseeding mode. | A shared YAML does not mean shared effective objectives or initialization. Apply each option or reject it explicitly. |
| Medium | Base validation uses a shuffled loader with `drop_last=True`; `run_validation` divides by `max(n,1)`: [train.py](../geoae/train.py), lines 71–98 and 172–173. | Small validation sets produce zero batches and report MSE/FVE `(0, 0)`; larger sets can omit a varying tail. Use all validation rows and fail on empty evaluation. |
| Medium | Extraction ignores `extraction.data_sources`; CLI `--max_doc_tokens` and `--skip_leading` are shadowed by existing dataclass fields: [extract.py](../geoae/extract.py), lines 231 and 380–381. | Intended data-distribution changes may not occur. CLI override failure reproduced. |
| Medium | Sliding-window scoring starts at `min(stride, T-1)` after the first window: [inference_benchmark.py](../geoae/interp/inference_benchmark.py), line 165. | It skips target tokens at the default half-window stride; other strides can duplicate many targets. Track the last scored absolute token. Reproduced. |
| Medium | Several DBpedia Qwen configurations omit `extraction.model_name`. | They resolve to Llama metadata despite Qwen-sized activations. Downstream model loading or metadata validation can be wrong. Check resolved configuration against extraction metadata. |
| Conditional | Head-only KL uses `lm_head(final_norm(h))`: [e2e/logits.py](../geoae/e2e/logits.py), line 72. Installed Gemma text-LM implementations can additionally soft-cap logits. | The shortcut is wrong when that option is enabled. Reproduced with a tiny local Gemma model configured with soft-capping; no claim is made that a particular saved run enabled it. Validate shortcuts against full forward and fail on mismatch. |

Continuation has additional state gaps: RNG/sampler position and anchor-pool usage are not saved; streaming fast-forward estimates consumed tokens from epoch budgets rather than storing actual source positions; mid-epoch best checkpoints resume at the next epoch. Normalization caches are keyed by filename without a data/split fingerprint. Fix these before calling resumed runs equivalent to uninterrupted runs.

There are further localized reporting issues worth tidying after the main fixes: some PCA comparisons use the baseline's normalization for the AE; LLM-judge spectrum examples can overlap detection positives; intrusion scoring divides by requested trials even after an early break; several scripts decide completion from file existence alone. These are not reasons to discard every evaluation, but they need explicit protocol coverage.

## 7. What I would implement next

**First, establish one reliable experiment contract.** Put resolved configuration, data/split hashes, selected checkpoint phase, assignment rule, edit position, reconstruction policy, effective learning rate, and runtime versions in a manifest. Fix the confirmed training/evaluation issues above. Add residual-preserving edits and decoded-direction equivalence controls to the existing number-control harness. Use a fixed test set and validation-only selection of edit strength. Measure target success, complement damage, neutral-task KL and decoded edit size together.

**Then run a data-only fine-tuning comparison.** Start from the same encoder/decoder checkpoint and fixed normalization, with fresh optimizer state and an explicit short schedule. Specify whether BatchNorm statistics adapt. Include the untouched parent, continuation on the original sampler, and continuation on the broader context sampler. Match sampled activation rows and optimizer updates across the two trained arms. Hold architecture, geometry weights and centroid/reseeding policy fixed. This distinguishes more optimization from better data coverage.

Use broad natural text and varied task formats without placing the final evaluation examples in training. If behavior preservation is needed, the existing KL path supplies a supervised/distillation-style constraint after its identity checks are repaired. Do not add a new counterfactual objective in the first data comparison; that would change the question being tested.

**Test compression and initialization separately after that.** Compare a bottleneck, square latent and wider latent under the same data protocol, with a simple linear/PCA baseline where applicable. Judge useful control at comparable collateral cost, not latent separation alone. For initialization, distinguish encoder/decoder warm starts from centroid initialization: the present k-means++, density-peak and labeled-anchor options initialize centroids after an AE warmup. Some configurations also change repeated reseeding, so they are not pure initialization ablations. Reuse a common warmup checkpoint and hold subsequent reseeding fixed to isolate centroid choice. Labeled anchor reuse is supervision and needs its own grouped holdout.

If broader sampling improves reconstruction or probes but does not improve controlled edits, that is evidence to investigate decoder alignment and the edit objective next. If it improves held-out controlled edits at fixed collateral cost, the distribution hypothesis has direct support. Neither outcome requires an RL stage.

## Verification performed during this review

The existing CPU suite passed **184 tests** using `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync pytest -q`. Additional checks used synthetic tensors, temporary checkpoints, patched toy training components, and a randomly initialized tiny local transformer; they did not load experimental checkpoints or call external model APIs.

| Fresh check | Observation |
| --- | --- |
| Linear-decoder nullspace | A latent displacement of norm 10 decoded to approximately `8.0e-7` in a random 4-by-8 decoder. |
| Dense latent steering identity | Maximum error against reconstruction minus decoded direction was approximately `2.4e-7`. |
| Small validation set | Three rows with batch size eight yielded zero batches and returned `(0.0, 0.0)`. |
| Sinkhorn batch dependence | An unchanged point's winning assignment changed when only its batch companion changed. |
| No-Sinkhorn checkpoint round trip | Saved `False`, loaded `True`. |
| Fine-tuning through resume | Requested LR `0.001`, effective restored LR `0.01`. |
| Streaming partial accumulation | Three documents per epoch, accumulation two: optimizer gradients were `[1.0, 1.5]` although each complete group should contribute `1.0`; final pending gradient was `0.5`. |
| Extraction overrides | Requested document length 777 and leading skip 9; actual extraction call received 256 and 4. |
| Sliding-window targets | For 24 tokens, window eight, stride four: scored 19 of 23 targets. Stride two scored 51 entries with 28 repeats. |
| Conditional Gemma head shortcut | With soft-capping explicitly enabled at 0.2, full versus shortcut logits differed by about `0.049`. |

No training implementation was changed and no research training/evaluation job was launched. This review document is the only repository file added by this independent review; pre-existing edits and externally appearing files were left alone.
