# GeoAE codebase audit and cleanup proposal

GeoAE should have one maintained source tree, a short paper reproduction path, and documented access to its additional research tools. The current workspace mixes these responsibilities, and it also lacks implementations that its own evaluation code imports. Reconcile the source copies before removing files.

This is a proposal based on the local workspace audit of October 21–22, 2026. No source, configurations, results, frozen snapshots, or remote files were deleted or rewritten. The only repository addition from this audit is this document.

## Decisions

1. Build the maintained checkout from the preserved Git history plus the newer research changes. Keep both original copies until that reconciliation is verified.
2. Organize research by experiment family under `experiments/`, with each family's configs, execution plan, analysis, and README together. Use the final paper to define the main documented workflow. Keep nonlinear decoders, KL training, matched SAEs, generation, judging, and other existing capabilities accessible as extensions.
3. Keep the package at `geoae/` initially. A move to `src/` would change path resolution, snapshot imports, and source hashes.
4. Move executable code out of `eval_out/` and `logs/`. Give algorithms, recipes, results, and historical evidence separate homes.
5. Extract demonstrated duplicate implementations. Preserve scientifically different algorithms and evaluation protocols.
6. Remove regenerable caches from the working tree. Move historical artifacts out of the default development view after verifying their archive and consumers.

There is no evidence supporting deletion of an entire current source feature family as dead code. Many tools are invoked through `python -m`, shell scripts, or frozen plans rather than imported by another Python module.

## Inventory

The inventory excludes `.venv/`, bytecode, interpreter caches, Ruff caches, pytest caches, and editable-install metadata. Counts are before adding this proposal. Byte totals are logical file sizes, rather than filesystem allocation.

| Current area | Files | Bytes | Meaning |
| --- | ---: | ---: | --- |
| `geoae/` | 102 | 1,476,040 | 101 Python modules and one README; 31,214 Python lines |
| `scripts/` | 68 | 582,355 | 30 Python files, 21 shell files, 15 Slurm files, one executable wrapper, one HTML template |
| `configs/` | 119 | 717,992 | 118 YAML recipes and one README |
| `tests/` | 53 | 361,069 | 52 test modules plus package initialization; 8,038 lines |
| `eval_out/` | 44 | 495,169 | Primarily source: 35 Python files and five shell launchers |
| `logs/` | 145 | 27,066,100 | 105 logs, 34 shell launchers, five Python files, one YAML |
| `results/` | 1,355 | 1,471,175,091 | Evaluation evidence, closest-token exports, figures, logs, and cached control corpora |
| `runs/` | 294 | 99,801,144 | Run records, results, frozen recipes, source archives, orchestration snapshots |
| `reports/` | 115 | 60,342,932 | Final reports, presentation, workbook, exports, builders, previews, ZIPs |
| `docs/` | 52 | 7,110,739 | Paper, guides, historical reports, notes, migration records |
| `backup_20261012/` | 5 | 102,916 | Historical source files; all differ from their current counterparts |

Across the workspace there are 2,374 files and 1,670,184,851 bytes under this exclusion policy. The environment occupies roughly another 1.1 GiB. Removing the environment would not simplify the code; keep it local and ignored.

The source is more scattered than the package count suggests: `eval_out/` contributes another 8,935 lines of Python and shell, while executable files under `logs/` contribute 930 lines. A directory named like output currently holds part of the research implementation.

## Problems to resolve before deleting source

### The two source copies contain different functionality

The research root has no `.git/`. The preserved checkout at `../code/GeoAE` is clean at commit `1121151`, the initial clean import from Delta. The current setup status explicitly requires reconciling them: [setup status](LAPTOP_SETUP_STATUS.md), Remaining work and boundaries.

| Area | Current files | Preserved checkout files | Identical at same path | Changed at same path | Current only | Preserved only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `geoae/` | 102 | 133 | 78 | 24 | 0 | 31 |
| `scripts/` | 68 | 154 | 32 | 6 | 30 | 116 |
| `configs/` | 119 | 164 | 103 | 6 | 10 | 55 |
| `tests/` | 53 | 85 | 30 | 7 | 16 | 48 |
| `docs/` | 52 | 20 | 17 | 0 | 35 | 3 |

The 31 files present only under the preserved package include 29 Python files, a pinned-source receipt, and a requirements file. They contain the Gemma Scope loader, additional SAE comparisons and controls, matched BatchTopK and Matryoshka training/evaluation, and extended control analyses. The 24 changed current files contain newer decoder, checkpoint, intervention, loader, and data changes.

For the maintained checkout, reconcile each changed file and retain the missing capabilities. Neither copying the older tree over this one nor discarding the older tree preserves the full system.

The current ignore policy requires an explicit provenance policy during Git recovery. `.gitignore` ignores all of `runs/` and large checkpoint/array formats, while intentionally retaining small result JSONs. An initial `git add` would omit run manifests and frozen source evidence. Ignored or untracked status is never a deletion criterion; do not use a blanket `git clean` to implement this proposal.

Confirmed missing dependencies in the current root:

| Consumer | Dependency absent here | Consequence |
| --- | --- | --- |
| `geoae/interp/sae_space.py:215–232` | `geoae.matched_topk.common`, `geoae.interp.gemma_scope` | The documented matched, Scope, and random SAE loading branches cannot complete |
| `scripts/deltaai/nonlinear_paper_evals.py:320–325` | `geoae.interp.gemma_scope._validate_config` | Gemma SAE preflight depends on the other source tree |
| `scripts/deltaai/qwen_paper_plan.py:134` | `eval_out/atlas_assign_extra.py` | Generated Atlas assignment command names a missing script |
| `scripts/deltaai/nonlinear_paper_plan.py:150` | The same Atlas script | The generic paper plan has the same dependency |

These implementations exist in the preserved checkout. Recovering them is source reconciliation, not a new experiment.

### The paper evaluation implementation is assembled from two trees

`qwen_paper_evals.prepare` copies `geoae/`, `eval_out/`, and `docs/paper/` from an external `--paper-repo`, overlays selected current training files, and patches evaluator source text. See [Qwen preparation](../scripts/deltaai/qwen_paper_evals.py), lines 150–177, and [generic preparation](../scripts/deltaai/nonlinear_paper_evals.py), lines 114–145.

Both preparation paths then require `docs/paper/main.tex`, while the active local file is `docs/paper/paper.tex`. This is why a locally importable model package does not establish that the full paper suite is self-contained.

For new runs, the reconciled checkout should already contain the intended evaluator behavior. Snapshot that single tree once. Retire source-text patching from the maintained execution path after equivalent behavior is verified. Preserve existing patched snapshots and their receipts exactly.

### The paper authoring directory is incomplete

`docs/paper/` contains only `paper.tex`. It loads the ACL style at line 5 and requests `custom.bib` at line 985. Neither file is in that directory. The sole local `custom.bib` is in `runs/nonlinear_decoder_validation/deltaai/review/paper_source/`; all 79 active citation keys occur there. The report's `raw_source_artifacts.zip` includes the paper but omits the bibliography and ACL style.

The authoring directory needs the bibliography, the required ACL template resources or a documented external template dependency, and a build command with its output directory. A final paper PDF can remain a deliverable; LaTeX auxiliary files should stay ignored.

Paper comments identify generators missing from this root:

- `eval_out/paper_geometry_table.py` at paper line 420.
- `eval_out/paper_control_plain_table.py` at line 728.
- `eval_out/ksweep_probe_pair10.py` at line 1106.

Those generators exist in the preserved checkout. Some cluster-count appendix result paths, including Gemma K=4000/K=8000 results, are also absent locally. Map them to their authoritative cluster artifacts before claiming that all paper tables can be regenerated.

### Several advertised options do not have consistent trainer behavior

These are static findings, separate from any claim about historical numerical results:

| Issue | Evidence | Required treatment |
| --- | --- | --- |
| KL trainers do not pass configured separation mode or margin | `e2e/train.py:327–337`, `e2e/train_stream.py:343–351`; defaults in `e2e/losses.py:59–60` | Either support the options explicitly or reject them for that trainer |
| KL variance loss ignores configured `var_gamma` | `e2e/losses.py:102–103` calls `variance_loss(z)` without gamma | Make effective behavior explicit and test it |
| Last-token KL training, cached or on-the-fly, crashes when reinitialization is disabled with zero | `config.py:124` documents zero as never; `e2e/train.py:388` performs modulo by zero | Guard the interval; the streaming trainer already does |
| Pipeline config only controls extraction | `pipeline.py:41–62` chooses a different training config by layer | Pass one resolved recipe through all phases |
| Dataset selection changes module globals | `interp/_shared.py:71–82` | Pass an explicit dataset specification to prompts, prediction, and cohort construction |

Fix these in separate, reviewable behavior changes. A structural cleanup should not silently change an old recipe's scientific meaning or rewrite a frozen source snapshot.

### Local test support is narrower than the README suggests

The supported Mac smoke command passed all 45 tests:

```bash
bash scripts/remote/setup_laptop.sh --smoke-only
```

An offline full-suite run produced **565 passed, 6 failed, 7 skipped**. Five failures reached hardcoded `/dev/shm` temporary directories in Qwen bundle/evaluation and generic evaluation code. macOS has no such directory. Staging policy needs to be passed into the executor, with a portable temporary directory for CPU tests and an explicit memory-aware policy for cluster runs.

The remaining failure was `test_refine_places_centroid_off_the_data_points`. Direct subtraction confirmed that the unrefined selections (`refine_k=0`) exactly equal existing rows. Default float32 `torch.cdist` reported a small nonzero distance; its non-matmul mode returned zero. Test exact row membership directly rather than weakening the assertion arbitrarily. This failure does not establish a defect in the refinement algorithm.

All seven skipped source-compatibility tests passed when pointed at the preserved local checkout:

```bash
GEOAE_PAPER_REPO=/Users/muhammadumairhaider/Work/code/GeoAE \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -m pytest -p no:cacheprovider tests/test_qwen_paper_compat.py
```

The default fixture still assumes `/home/exouser/code/GeoAE` (`tests/test_qwen_paper_compat.py:27`). The reconciled checkout needs an explicit local reference or small preserved fixtures while the patcher remains.

No real language-model weights, research datasets, cluster jobs, or paid judge APIs were used for these checks. Existing tests did exercise small synthetic CPU training fixtures.

## Proposed maintained tree

Keep `geoae/` at its present level during the first cleanup. The tree below specifies responsibility boundaries. Specialist tools remain separate files where combining them would produce a large module; the ellipses represent retained capabilities, not deletion.

```text
GeoAE/
├── README.md
├── AGENTS.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── geoae/
│   ├── __init__.py              # Small public model/config API
│   ├── cli.py                   # Thin routing to supported commands
│   ├── config.py
│   ├── model.py
│   ├── losses.py
│   ├── checkpoint.py
│   ├── lm_arch.py
│   ├── hooks.py
│   ├── token_bias.py
│   ├── training/
│   │   ├── mse.py
│   │   ├── kl_cached.py
│   │   ├── kl_stream.py
│   │   ├── data.py
│   │   ├── schedules.py
│   │   ├── initialization.py
│   │   ├── supervision.py
│   │   └── checkpoints.py
│   ├── extraction/
│   │   ├── sources.py
│   │   ├── corpus.py
│   │   ├── structured.py
│   │   ├── prefix.py
│   │   ├── sampled.py
│   │   └── benchmarks.py
│   ├── baselines/
│   │   ├── kmeans.py            # Plain fitter; balanced fitter remains named
│   │   ├── balanced_kmeans.py
│   │   ├── token_erasure.py
│   │   ├── sae_space.py
│   │   ├── gemma_scope.py
│   │   └── matched_topk/        # Includes pinned implementation provenance
│   ├── evaluation/
│   │   ├── geometry.py
│   │   ├── alignment.py
│   │   ├── faithfulness.py
│   │   ├── cohorts.py
│   │   ├── control.py
│   │   ├── ranges.py
│   │   └── ...                 # Separate scorers/adapters where needed
│   ├── analysis/
│   │   ├── closest_tokens.py
│   │   ├── logit_attribution.py
│   │   ├── concept_probe.py
│   │   ├── concept_steer_generate.py
│   │   ├── concept_remove_generate.py
│   │   ├── llm_judge.py
│   │   └── ...                 # Remaining probes, judges, plots, diagnostics
│   └── workflow/
│       ├── assets.py
│       ├── snapshot.py
│       ├── plan.py
│       ├── execute.py
│       ├── slurm.py
│       └── summary.py
├── configs/
│   ├── sites/                  # Delta, DeltaAI, Argo runtime/resources
│   └── examples/               # MSE, KL, extraction, advanced features
├── experiments/
│   ├── README.md               # Index of experiment questions and status
│   ├── paper/
│   │   ├── README.md           # Table → command → inputs → result mapping
│   │   ├── configs/            # Explicit reviewed paper recipes
│   │   ├── plan.py             # Exact protocol; calls shared package code
│   │   ├── tables.py
│   │   └── artifacts.json      # References to frozen runs and evidence
│   ├── nonlinear_decoders/
│   │   ├── README.md           # Question, comparisons, findings, commands
│   │   ├── configs/            # Model × linear/MLP/residual-MLP arms
│   │   ├── plan.py
│   │   ├── analyze.py          # Experiment-specific comparisons and plots
│   │   ├── results/            # Selected small summary tables
│   │   └── artifacts.json      # Full results, checkpoints, source receipts
│   ├── token_bypass/
│   ├── width/
│   ├── cluster_count/
│   ├── balancing/
│   ├── initialization/
│   └── sae_comparison/
├── scripts/
│   ├── setup_laptop.sh
│   ├── setup_cluster.sh
│   └── slurm/
│       ├── delta.sbatch
│       ├── deltaai.sbatch
│       └── argo.sbatch
├── tests/
│   ├── unit/
│   ├── workflows/
│   └── integration/
└── docs/
    ├── architecture.md
    ├── usage.md
    ├── research.md
    ├── artifacts.json          # Shared artifacts and experiment manifest index
    ├── paper/
    │   ├── paper.tex
    │   ├── custom.bib
    │   ├── ...                 # Required template assets, with provenance
    │   └── README.md           # Build and table-generation instructions
    └── history/
        ├── CHAT_HISTORY.md
        ├── notes/
        ├── audits/
        └── migration/
```

Historical experiments that retain reusable operations need an indexed replay recipe or supported command. A dated launcher can be retired after that operation is available elsewhere. The final active tree should not carry a large unindexed `legacy/` package.

Keep the existing seven installed commands during migration. They can delegate to the new modules. Update direct `python -m` commands, import paths, tests, documentation, and snapshot logic as one feature family moves. Remove compatibility shims only after checking their actual consumers; the one-line `GeoSepAE` alias is a very low priority.

Keep historical artifacts in a separate verified archive, for example a sibling `GeoAE-artifacts/` and the established Delta archive. This is a proposed local organization, not a rename of any existing cluster path. During transition, retain existing local `runs/`, `results/`, and frozen paths until their consumers have an explicit mapping.

The intended reading order is README → model and loss → one trainer → the paper recipe → the relevant evaluator. Optional analyses should be discoverable from one command index.

### Experiment directories

Each experiment family has one directory that answers its research question: what changes, what stays fixed, how to run the comparison, and what the results support. For example, `experiments/nonlinear_decoders/` owns the linear-versus-MLP-versus-residual-MLP comparison across model families. Model and decoder arms are configs within that experiment, rather than separate copies of training code.

Keep experiment-specific configs beside their plan and analysis. The root `configs/` holds site settings and general usage examples. A recipe has one authoritative editable location; other experiments reference a shared baseline or its frozen run rather than keeping independently editable copies of it.

An experiment folder can contain:

- `README.md`: question, hypothesis, changed and fixed settings, status, commands, and interpretation.
- `configs/`: explicit arm definitions and scientific settings.
- `plan.py` or a declarative plan: calls the shared workflow and specifies the exact protocol. Add Python only when custom logic is needed.
- `analyze.py` or `tables.py`: comparisons, figures, and tables specific to that question.
- `results/`: selected small summaries that make the experiment understandable locally.
- `artifacts.json`: references to complete outputs, frozen run manifests, checkpoints, source identities, and authoritative storage locations.

Only add files the experiment needs; this is not a mandatory scaffold. The additional family names in the tree describe existing lines of research and are not instructions to create empty directories.

Reusable models, losses, fitters, metrics, cohort selection, and execution safeguards stay in `geoae/`. An experiment may own a new method while it is specific to that question; move it into the shared package when another workflow needs it. This keeps experimental reasoning together without duplicating the implementation.

Full activation dumps, weights, paid judge caches, and frozen run directories retain their established storage and identities. The experiment folder provides a readable route to them. Dated reruns belong to run records under the experiment family; create a separate experiment directory when the research question changes.

## Capability preservation map

The final paper defines the primary path: Llama, Gemma, and Qwen; bypass and no-bypass models; linear decoder; geometric training; plain and token-removed k-means; Gemma Scope comparisons; geometry and faithfulness; token/sequence alignment; concept control; appendix ablations. The paper does not describe the later nonlinear decoder experiments.

| Capability to retain | Existing implementation families | Destination |
| --- | --- | --- |
| Linear, MLP, residual MLP; encoder variants; exact nonlinear decoded edits | `model.py`, checkpoint/config, intervention consumers | Core model and documented decoder options |
| Fixed token bypass, decoder-only bias, unknown tokens | `token_bias.py`, model, activation loaders | Core transform and explicit model options |
| Uniform/Zipf/generalized Sinkhorn, no-Sinkhorn ablation, hard/soft EMA, Euclidean/cosine | Model, losses, training common helpers | Named model/training settings |
| MSE reconstruction training | `train.py`, `data.py`, `train_common.py` | `training/mse.py` |
| Cached and on-the-fly last-token KL; streamed all-position KL | `e2e/` | Separate KL trainers sharing small helpers |
| Schedules, initialization/reseeding, SupCon fine-tuning, resume and retention | `train_common.py`, `seeded_init.py`, `supervised.py` | Training utilities with explicit trainer support |
| Prefix/sampled extraction, structured/frozen corpora, dataset-specific extraction | Extract/build modules, DBpedia/bias/GoEmotions adapters | Extraction package and dataset adapters |
| Plain and balanced k-means; token/PCA erasure | Fitters, baseline adapters, token erasure | Baselines package; preserve distinct algorithms |
| Pretrained Scope and matched BatchTopK/Matryoshka | Current `sae_space.py` plus missing older source | Restored baselines and SAE evaluation |
| Geometry, token/sequence alignment, Atlas bootstrap protocols | `interp/` and `eval_out/` | Reusable evaluators plus exact paper recipes |
| MMLU, WikiText perplexity, splice loss, quantization, ablation, speed | Current evaluation/interp modules | Separate named faithfulness scorers |
| Joint-correct cohorts, label control, ranges, sibling removal, number/morphology controls | `_shared.py`, intervention tools, paper ranges | Cohort/data layer and explicit intervention methods |
| Probes, erasure studies, token interpretation, DLA, plots, trajectories | Remaining `interp/`, bias/DBpedia analysis | Documented optional analyses |
| Generation, removal, judging, judge agreement and caches | Generate/judge families | Optional analysis commands and declared dependencies |
| Asset receipts, source snapshots, command plans, staged data, locking, resume, Slurm | Argo/DeltaAI launch families | Workflow package with thin site wrappers |
| Historical widths, no-bypass, cluster counts, decoder-only, repair and smoke campaigns | Launchers/configs under scripts/logs/runs | Indexed recipes or historical replay tools |

Unavailable dependencies should produce a clear capability error. Declare a small number of useful extras, such as judge providers and optional visualization/SAE dependencies. Current `llm_judge.py` dynamically uses OpenAI and Anthropic SDKs that are absent from the project dependencies.

## Consolidation with evidence

| Current implementations | Shared implementation to extract | Behavior to preserve |
| --- | --- | --- |
| Older `interp/sae_controls.py:25–70`; current `fit_balanced_kmeans.py:335–379` | Tensor-level balanced fitter | Seeding, visit mask, reseeding, update order; older file explicitly documents copying the loop |
| `bias/extract.py:57–78`; `dbpedia/extract.py:78–99` | Capture hook and pooling primitives | Dataset labels, prompts, hashes, schema remain adapter-specific |
| `extract.py:60–101`; `closest_tokens.py:84–133` | Source streaming and tokenization | Priming, exhaustion, truncation, sampling and reporting |
| `cluster_steer_generate.py:216–224`; `concept_steer_generate.py:681–689` | Generation with hook lifecycle | Cleanup and continuation slicing |
| `concept_steer_generate.py:691–696`; `concept_remove_generate.py:245–250` | Continuation perplexity helper | Prompt/continuation boundaries |
| `train.py:41–68`; `e2e/train.py:50–66` | Checkpoint serialization | Metric names, normalization, provenance and optimizer state |
| `train_common.py:108–145`; `checkpoint.py:86–110` | Architecture/config mapping | Fresh initialization remains distinct from restoration |
| `e2e/train.py:190–207`; `train_common.py:513–536` | Initialization pool encoding | Batch types, row budgets and minimum rows |
| Qwen-specific and generic DeltaAI plan/eval/summary families | One executor and model-specific plans | Qwen/Llama/Gemma protocol differences and exact outputs |
| Argo and DeltaAI prepare/run/submit families | Shared snapshot/submission functions | Site modules, resource limits, account names and staging differences |
| Remote control/confidence wrappers | Recipe-driven evaluator invocation | Dataset/cohort rules, saliency, alpha and confidence/accuracy distinctions |

Do not merge plain k-means with balanced k-means, cached KL with streaming KL, training assignment diagnostics with hard-label geometry metrics, or fixed linear directions with state-dependent nonlinear decoder changes. Share numerical primitives only where their inputs and meaning are identical.

Preserve the workflow safeguards: hashes, output validation, checkpoint identity, partial-submit receipts, no duplicate submission, locks, completed-step reuse, and resume integrity. They solve recorded failure cases.

Model compatibility also includes parameter names, persistent buffers, and constructor RNG consumption. `model.py:190–239` deliberately preserves legacy decoder names, matched-arm initialization, and centroid/EMA/Sinkhorn state. Compare state dictionaries, assignments, decoded edits, and initialization alongside forward outputs. Preserve normalization, temperature, and balancing state when restoring checkpoints. Reconstruction-best, final, and resumable checkpoints have different roles; the guards documented in `tests/test_train_resume.py` must survive consolidation.

## Configuration cleanup

All 118 current YAML files load through `Config.from_yaml`. No complete parsed YAML files are duplicates when comments are removed and every path field is retained.

There are exactly two duplicate pairs after applying all dataclass defaults:

- The Gemma final-bypass recipe under `configs/base/` and `configs/nonlinear_decoder/gemma3-4b_l33/reference_bypass.yaml`.
- The Llama final-bypass recipe under `configs/base/` and `configs/nonlinear_decoder/llama3.2-3b_l27/reference_bypass.yaml`.

These are canonicalization candidates. Preserve the frozen copies used by historical runs.

The 14,104 YAML lines consist of 7,672 full-comment lines, 615 blanks, and 5,817 content lines. Keep recipe-specific explanations and the change from its parent; put general configuration documentation in the guide. Existing loading already applies overrides to defaults, so concise explicit YAML does not require another configuration framework.

Critical exception: the no-Sinkhorn recipes encode their differentiating behavior in comments instructing callers to pass `--no_sinkhorn`. See the Llama `...tokbias_nosinkhorn.yaml:3–5` and the layer-16 KL `...vicreg_nosink.yaml:1`. The maintained recipe must encode this setting as machine-readable input, and the resolved runtime configuration must record it.

Keep readable general examples under `configs/examples/`, and explicit reviewed scientific recipes under `experiments/<family>/configs/`. Generate seed replicas through the existing utility or CLI overrides, then save the fully resolved config and command into each run. Keep historical seed recipes with their provenance.

The unused `ExtractionConfig.pack_size`, `skip_first_n`, and `data_sources` can leave the documented schema after legacy YAML/checkpoint loading handles them. Deleting these dataclass fields immediately would make previously valid configs fail because unknown keys are rejected.

Keep filesystem locations and site resources separate from scientific choices. Resolve environment/site settings once and record the effective values. Do not rewrite the absolute paths inside existing frozen manifests.

## Disposition of the current roots

| Path | Disposition | Condition |
| --- | --- | --- |
| `geoae/` | Keep and reorganize | Reconcile missing source and preserve current changes first |
| `tests/` | Keep and reorganize | Separate unit, workflow, and explicit integration tests |
| `configs/` | Experiment recipes move beside their experiment; retain general examples and site settings here | Preserve effective settings, runtime flags and frozen copies |
| `scripts/` | Thin setup/site wrappers | Move reusable Python into the package; historical recipes into an index |
| `eval_out/` | Extract maintained source, then retire output-like source directory | Map all 35 Python/five shell files, including old-only dependencies |
| `logs/` | Extract launcher/config source; archive historical logs | Do not delete the 40 non-log files with the directory |
| `results/` | External artifacts plus selected small local evidence | Verify hashes/locations and update consumers before removing local copies |
| `runs/` | Preserve frozen run records | Do not mutate source archives, configs, orchestration, hashes or state evidence |
| `reports/` | Keep a delivery entry point; archive complete dated bundle | Reusable extraction moves to package; report-specific builders stay with report |
| `backup_20261012/` | Historical source snapshot | All five files differ; preserve before leaving active tree |
| `activations_*/` | Metadata belongs in artifact catalog | These local directories contain small metadata, not the large activation arrays |
| `corpus/` | Preserve manifests/catalog locations | Required to identify frozen extraction input |
| `dbpedia/`, `biasbios/` | Preserve cache/activation metadata | Dataset code already lives in the package; map data paths separately |
| `RUNBOOK_DELTA_ROUND2.md` | Historical campaign runbook | Extract current setup instructions, retain original as history |
| `AGENTS.md` | Keep concise continuation instructions | Update links only after corresponding moves |
| `pyproject.toml`, `uv.lock`, `LICENSE` | Keep | Preserve locked environment and verify packaging deliberately |
| `README.md` | Rewrite around current entry points | Current text describes absent predecessor trees and outdated research defaults |
| `.venv/` | Local ignored environment | Rebuild only when setup changes require it |
| `geoae.egg-info/` | Local generated install metadata | Ignore; it may be regenerated on installation |
| `.ruff_cache/`, bytecode | Delete from working copy when cleaning | Regenerable and unrelated to scientific evidence |

Specific source hidden under `logs/` includes pair-probe runs, no-bypass evaluations, token-removed baselines, cluster-count sweeps, and `g4b_final_config_from_ckpt.yaml`. Convert the useful launch commands to explicit recipes. Preserve `patch_keep10_20261012.py` and `patch_nobal_20261012.py` as historical source mutations rather than normal execution commands.

`runs/gpt2_2x/` is a post-hoc GeoAE compatibility smoke record with source backups. Do not move it into GeoPretrain solely because its name contains GPT-2.

Historical maintenance scripts such as `scripts/delta/clean_slate.sh`, `organize_delta_results.sh`, and the pinned Gemma cache repair belong with their historical receipts. They should leave the normal research command list.

`scripts/remote/import_project_chats.py` is provenance tooling. Moving it is optional and requires updating AGENTS/history links; it is unrelated to model execution.

## Deletion and deduplication candidates

### Regenerable working files

The currently observed `.ruff_cache/` and `scripts/remote/__pycache__/import_project_chats.cpython-312.pyc` can be removed from the working copy. Apply the same rule to other current bytecode and pytest caches if present. Exclude frozen source archives from cache-cleanup globs.

These changes save little space; their purpose is a clean workspace.

### Generated report previews and duplicate delivery packaging

After retaining the verified final deliverables and QA receipts, local copies of these can leave the active tree:

```text
reports/nonlinear_decoder_20261021/report_preview/
reports/nonlinear_decoder_20261021/slides_preview/
reports/nonlinear_decoder_20261021/nonlinear_decoder_presentation_bundle.zip
```

The preview directories occupy approximately 4.7 MiB together. Keep one verified complete delivery bundle in the external artifact archive. Different formats such as editable slides, PDF, SVG, and PNG are distinct deliverables; a similar filename is not evidence that one is dispensable.

### Exact content duplicates that need path aliases

Hashing the artifact/document/log trees found 50 nonempty duplicate groups representing 51,734,711 redundant bytes, excluding overlap inside ZIPs. A further zero-byte group includes `laneB.done` and `table.ready` status markers; preserve their roles rather than deleting them as empty files.

The largest example is:

```text
results/delta/base_balanced_kmeans_k2000_dpc_sampled/interp/closest_tokens.json
results/delta/base_balanced_kmeans_k2000_dpc_sampled/interp/closest_tokens_rejudge.json
```

These files are byte-identical at 38,709,816 bytes each. Another example is `results/delta/d6144/interp/atlas_bypass.json`, duplicated under `rejudge_seed1/` and `vs_tokmean/`.

Archive deduplication can retain one content object plus aliases for original paths and roles. Local removal should follow consumer mapping. A repeated filename may be used by a frozen result extractor or cache loader.

The active paper is byte-identical to four frozen paper copies under `runs/nonlinear_decoder_validation/deltaai/review/`. Its SHA-256 is:

```text
cf2aefe5b9e1882ac56397f547c85242ea65b113036565aa60cd25ccb32e0203
```

Keep those frozen copies. The report extractor and builder record their path and hash (`reports/nonlinear_decoder_20261021/scripts/extract_metrics.py:224–232`, `build_report.py:507–519`).

The authoring copy has 262 full-comment lines in 1,344 lines, including superseded drafts and results paragraphs. Remove superseded prose only from the maintained authoring copy after preserving its original hash; retain meaningful method/generator references and update those paths.

## Documentation and packaging

Maintain three short guides: architecture, usage, and current research state. Link method-specific audit notes from the relevant evaluation command. Keep historical notes and chats available through an index.

The README should explain the actual model, the paper and nonlinear distinction, supported setup, one complete recipe, output locations, and the command index. Its claims about `geosep/`, a publishing-ready successor, and a placeholder citation URL need replacing with the present project state.

The package currently derives its root from `Path(__file__).parent.parent` (`paths.py:7`) and resolves configs/data relative to it. Setuptools discovers Python packages, but the project does not explicitly package sibling YAML config resources. Editable installation succeeds without establishing that a wheel installed elsewhere can find those defaults.

Keep the existing layout first. Then verify a built wheel in an isolated environment, with explicit resource handling or caller-supplied paths. Do not call the code distributable based only on imports from this checkout.

Avoid a general plugin system, a new config inheritance engine, dozens of tiny dependency extras, or a universal trainer superclass. Straightforward functions, a small step schema, and explicit recipes are sufficient.

## Implementation order and completion checks

### Phase 1 Reconciled baseline

Create an isolated worktree or branch from the preserved Git history. Record relative paths and hashes for both original copies; import the newer changes explicitly and retain the missing source capabilities. Keep the current environment and original archives intact.

Completion: every changed/current-only/preserved-only source file has a disposition; the model and existing CLIs import; the missing SAE and Atlas dependencies are present; test results are recorded.

### Phase 2 Portable execution and honest configuration

Address the five temporary-directory failures and the numeric assertion. Make the paper checkout reference explicit. Add trainer capability validation and resolve the ignored-option/zero-reinitialization issues as separate behavior changes.

Completion: the complete local CPU test suite passes without accidental skips caused by an obsolete path. Unsupported options fail before training. Existing frozen recipes and remote execution snapshots remain unchanged.

### Phase 3 One paper path

Complete the authoring bundle, recover the missing generators, and build a table/protocol map. Reconcile the canonical evaluator implementations with the current nonlinear edits. Use the reconciled source directly for new snapshots.

Completion: every paper table/appendix result has a command or preserved generator, exact input/artifact references, checkpoint selection, evaluation cohort/filter rules, and a known output schema. The paper builds when its documented LaTeX dependencies are available. Missing local large artifacts have explicit authoritative locations.

### Phase 4 Shared algorithms and workflow

Extract the proven duplicates in small commits. Parameterize site resources, staging, and model profiles. Make the executor consume an explicit plan and own source snapshots, locks, resume checks, and output validation.

Completion: small synthetic fixtures preserve old outputs or documented tolerances, decoded edits remain exact, checkpoint roundtrips remain compatible, and failure/resume/submission tests still pass. Qwen protocol parity precedes retiring the Qwen-only runner.

### Phase 5 Tree and documentation

Move one capability family at a time. Update installed commands, `python -m` paths, package resources, test imports, defaults, documentation, and new-run snapshot rules together. Keep temporary compatibility aliases where they have consumers.

Completion: no maintained executable source remains under output/log directories; the primary workflow has one documented route; all additional capabilities are indexed; a wheel/resource check passes if packaging is claimed.

### Phase 6 Archive and deletion

Build the artifact catalog and verify the external copy. Move dated reports, results, repair scripts, snapshots, and historical recipes out of default navigation. Remove regenerable clutter and deduplicated local copies only after path consumers have been handled.

Completion: every removed local artifact has either a verified archived equivalent or an explicit regeneration reason; all frozen run hashes remain valid; current status/comparison links still open; no remote job source, cluster data path, or paid judge cache has changed.

The success criterion is one readable implementation of each supported operation, one explicit recipe for each reported protocol, and a verifiable path to its evidence. A smaller file count alone is not sufficient.

## Source reconciliation register

These source-area files exist in `../code/GeoAE/geoae/` and are absent from the current `geoae/`. This is a retention inventory; it does not imply they should be copied without checking compatibility.

```text
campaign_config.py
interp/_context_logits.py
interp/atlas_geometry.py
interp/chain_control.py
interp/concept_steer_extra.py
interp/concept_steer_supervised_subset.py
interp/gemma_scope.py
interp/representation_dispersion.py
interp/sae_coding.py
interp/sae_compare.py
interp/sae_controls.py
interp/sae_data.py
interp/sae_matched.py
interp/sae_removal.py
interp/sae_report.py
interp/ssl_concept_controls.py
matched_topk/__init__.py
matched_topk/_reference/SOURCE.json
matched_topk/_reference/__init__.py
matched_topk/_reference/dictionary.py
matched_topk/_reference/trainers/__init__.py
matched_topk/_reference/trainers/batch_top_k.py
matched_topk/_reference/trainers/matryoshka_batch_top_k.py
matched_topk/_reference/trainers/trainer.py
matched_topk/benchmark.py
matched_topk/common.py
matched_topk/evaluate.py
matched_topk/report.py
matched_topk/requirements.txt
matched_topk/stage.py
matched_topk/train.py
```

These overlapping files differ between the current and preserved package. Preserve and review the current changes explicitly:

```text
bias/extract.py
bias/probe.py
checkpoint.py
config.py
data.py
dbpedia/extract.py
interp/_shared.py
interp/cluster_steer_generate.py
interp/clustering_quality.py
interp/concept_remove_generate.py
interp/concept_steer_generate.py
interp/concept_steer_judge.py
interp/concept_suite.py
interp/faithfulness_ablate.py
interp/logit_attribution.py
interp/mmlu_splice.py
interp/neuronlens.py
interp/range_intervention_compare.py
interp/range_number_control.py
interp/splice_ppl.py
interp/steering_morph_compare.py
model.py
train.py
train_common.py
```
