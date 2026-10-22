# Matched Llama and Gemma AEs at middle and final layers

The pipeline supports plain and token-bypass AEs at Llama layers 14/27 and
Gemma 4B layers 17/33. Every training uses the same required ten-source `struct5`
recipe. Each model/layer shares one activation dump and evaluation controls;
plain and bypass checkpoints and results remain separate.

| Arm | Frozen model | Block | Residual → latent | Extraction config |
|---|---|---:|---:|---|
| `llama` | `meta-llama/Llama-3.2-3B` | 14 of 28 | 3072 → 6144 | [Llama extraction](../configs/llama3.2-3b_l14_extract_sampled_struct5.yaml) |
| `gemma4b` | `google/gemma-3-4b-pt` | 17 of 34 | 2560 → 5120 | [Gemma extraction](../configs/gemma3-4b_l17_extract_sampled_struct5.yaml) |

Every run uses exactly 10M training rows and 526,316 separate validation rows
(10,526,316 extracted rows with a 95/5 split),
K=2000, seed 42, batch 32768, a BatchNorm/GeLU encoder, DPC initialization,
hinge separation and the parent's 50-epoch schedule. The model encodes
`x - b[current_token]` and reconstructs `decoder(z) + b[current_token]`.
Each layer gets its own train-only token table and normalization statistics.
Gemma activations retain float32 storage, matching the existing Gemma parent.
The frozen text corpora and source quotas are reused, including the 5% structured
source and its benchmark-overlap exclusions.

## Run

Use the existing project environment (`scripts/delta/py` selects the configured
venv on Delta, Argo, or a local machine). The two entry points run corpus setup,
extraction, table construction, training, detailed evaluations and reporting:

```bash
bash scripts/midlayer/train_llama.sh --dry-run
bash scripts/midlayer/train_gemma4b.sh --dry-run

# Run one arm in a tmux session with a GPU:
bash scripts/midlayer/train_llama.sh | tee logs/midlayer_llama.log
bash scripts/midlayer/train_gemma4b.sh | tee logs/midlayer_gemma4b.log
```

To run the six new variants serially on one GPU:

```bash
scripts/delta/py scripts/midlayer/run.py --arm new
```

The trainers resume the numerical latest checkpoint and verify its full recipe
before resuming. They verify epoch 50, model, layer, width, K, initialized
centroids and bypass mode before marking training complete. Existing extraction
and token-table artifacts are reused after model/layer/source/shape/statistics
checks. A wrong artifact raises an error. Checkpoint-directory locks prevent two
trainers from writing the same run concurrently.

Use `--steps training` for corpus/extraction/table/training only, or
`--steps train` when the dump and table already exist. `--steps evaluate,report`
runs the detailed suite on a trained arm. `--eval-steps gpu` runs the suite
without API judges; `--judges required` treats unavailable API credentials as a
failure. `--judges auto` records missing API judgments as pending and leaves
the evaluation status partial. `--force-evals` reruns evaluation stages.

For Slurm, source the site's existing environment, create `logs/`, and choose the
site's GPU partition/account as usual:

```bash
source scripts/delta/env.sh   # or scripts/argo/env.sh
mkdir -p logs
sbatch --partition="$MID_LAYER_GPU_PARTITION" scripts/midlayer/run.sbatch llama
sbatch --partition="$MID_LAYER_GPU_PARTITION" scripts/midlayer/run.sbatch gemma4b
```

The supplied job template requests one GPU, 16 CPUs and 96 GB RAM. On Delta,
`DELTA_DATA` puts new dumps under the existing bulk-data root instead of the code
filesystem. Existing checkpoint/corpus/cache links remain managed by the site's
`link_data.sh`. A custom `--data-root` changes the location of the **new dump**;
set up the site's other data links before running. Raw dumps require about
61 GB for Llama and 102 GB for Gemma, plus sidecars, checkpoints and evaluation
caches. Extraction restarts from the beginning after an interrupted dump;
training resumes at the latest saved epoch.

## Detailed evaluation coverage

The middle-layer runner reuses the existing metric implementations:

```bash
scripts/delta/py scripts/midlayer/evaluate.py llama --steps whole --dry-run
scripts/delta/py scripts/midlayer/evaluate.py gemma4b --steps whole --dry-run
scripts/delta/py scripts/midlayer/evaluate.py all --steps gpu
scripts/delta/py scripts/midlayer/evaluate.py all --steps api --judges required
scripts/delta/py scripts/midlayer/report.py --arm all
```

| Group | Measurements/artifacts |
|---|---|
| `prep` | Matched raw and token-mean balanced k-means; POS/NER and nine sequence-task caches with token IDs; RAVEL/IOI caches; DBpedia/BiasBios probe activations and source hashes; word-sense data |
| `evals` | Live reconstruction loss recovered and per-cluster ablation; current-token/document cluster structure; chance-corrected concept NMI with bootstrap comparisons; MMLU per-subject accuracy and paired flips; clustering quality/rank/usage; document-disjoint dispersion and Vendi with native and fixed base groupings |
| `interventions` | Number agreement with rotated and shuffled controls; supervised DBpedia steering; DBpedia and BiasBios range removal/transport using d-prime saliency; number control in the token-mean residual |
| `steering` | Cluster-directed generation with fluency and perplexity measurements; final-block-only dump-row transfer is recorded as unavailable |
| `concept` | Matched DBpedia/BiasBios generation using label directions in h/z, raw/bypass cluster transport and raw/bypass cluster directions |
| `geometry` | PCA/t-SNE/UMAP/variance panels on concept caches |
| `probes` | Gender/profession localization and DBpedia/BiasBios fixed-probe coordinate removal, with mean-fill ablation and raw/token-mean/bypass comparisons |
| `interp` | Held-out document-context examples for raw/token-mean/bypass clusters; CoarseWSD20 and WiC word-sense evaluation |
| `api` | Cluster/concept generation judgments, cluster interpretation, multi-judge agreement, atlas summaries and blinded pairwise judgments |

`whole` includes all groups above. `gpu` includes all except `api`; it also
includes the CPU/data preparation stages. Individual stage names are accepted.
UMAP uses the existing optional `umap-learn` dependency; unavailable requested
UMAP figures leave geometry partial and are listed in the manifest.
External judgments use the existing `OPENROUTER_API_KEY` setup. Human/agent
answer export/import is available separately as `pair_export`/`pair_import`.

Middle-layer behavior is measured by running the remaining transformer blocks.
Concept steering and generation splice the selected block in a complete model
forward. For cluster generation, middle-layer target signatures replay real
document contexts through the full text decoder. Applying the final head
directly to a middle-layer vector would produce a logit-lens proxy instead of
the model's actual prediction.

Two existing final-block metrics have no valid dump-row-only middle-layer
equivalent: predicted-next-token cluster purity and cluster transfer (`csteer`).
The former is explicitly omitted with `--skip_next`; the latter is recorded as
skipped with its reason. Live reconstruction, MMLU, range edits and generation
remain included. Thus unavailable metrics are visible in the manifest rather
than replaced with misleading numbers.

## Artifacts and review

Results are under:

```text
results/midlayer_run/llama3.2-3B/layer14/
results/midlayer_run/gemma3-4B/layer17/
```

Each tree holds `evals/`, `interventions/`, `steering/`, `concept/`, `probes/`,
`figures/`, `interp/`, `baseline/`, `token_mean/`, `joint_correct/` and stage logs.
`pipeline_state.json` records commands and completion/skips/failures;
`trainer_state.json` records training-stage progress. After evaluation,
`report.md` links detailed artifacts and measured headline results, while
`report.json` indexes scalar summaries. Per-concept, per-subject, paired
comparisons, strengths and examples remain in the original metric JSON files.
Missing judgments and failed or malformed artifacts are reported explicitly.

The configs and runners prepare these experiments; GPU training and measured
results require running the commands above.

## Six-run campaign on two Argo H100s

The requested six new trainings run in two independent queues. Every training is
followed by its GPU evaluations before that queue advances to the next arm:

| H100 queue | First | Second | Third |
|---|---|---|---|
| Llama | L14 bypass | L14 plain | L27 plain |
| Gemma 4B | L17 bypass | L17 plain | L33 plain |

The final-layer bypass variants are also supported. A reference is reused only
when its checkpoint and completed ten-source dump pass validation. Missing or
incompatible references remain pending; the six-run campaign never substitutes
an older dataset or silently adds another training.

After both GPU queues finish, a CPU-only job runs API judgments serially and
writes reports. Each GPU job requests one H100, 12 CPUs, 240 GiB RAM and a 14-day
ceiling. Only one model runs at a time on each GPU. The 240 GiB includes RAM
staging of the current layer; both variants reuse that copy before it is removed
for the next layer. Checkpoints retain canonical dataset paths and the verified
source fingerprint across staging and resume.

```bash
# On Argo after deployment and credential setup:
source scripts/argo/env.sh
scripts/delta/py scripts/midlayer/campaign.py --dry-run
scripts/delta/py scripts/midlayer/campaign.py --submit
```

The campaign records each submitted job immediately in
`runs/midlast_ablation_20261006/campaign.json`, prevents duplicate submissions,
and supports `--submit --resume-submission` if submission stopped between jobs.
Progress is in `lane_llama.json`, `lane_gemma4b.json` and `judges_state.json` in
that directory; Slurm logs are under `logs/midlast_ablation_20261006/`.

All eight variants can be addressed by `llama_mid_bypass`, `llama_mid_plain`,
`llama_last_bypass`, `llama_last_plain` and the four equivalent `gemma4b_*`
names. `--arm new` selects the six new trainings; `--arm all` selects all eight.
Evaluation uses positional arm names:

```bash
scripts/delta/py scripts/midlayer/run.py --arm llama_last_plain --dry-run
scripts/delta/py scripts/midlayer/evaluate.py gemma4b_last_plain --steps whole --dry-run
scripts/delta/py scripts/midlayer/report.py --arm all --run-name midlast_ablation_20261006
```

Explicit variant results are under
`results/<campaign>/<family>/layer<L>/{plain,bypass}/`. Reports identify the AE
mode. Some underlying tools retain their historical `bypass` argument/result
key when evaluating the plain AE; the variant directory and pipeline metadata
identify the actual architecture.

## Required ten-source dataset

| Domain | Share |
|---|---:|
| edu_web | 35% |
| web | 15% |
| wiki | 15% |
| pile | 10% |
| code | 8% |
| math | 5% |
| structured | 5% |
| wiki_de | 3% |
| wiki_fr | 2% |
| wiki_es | 2% |

`dataset.py` enforces the exact HF identities/configurations and field names,
Pile exclusions, structured eight-item packing, RAVEL/IOI guard statistics,
all ten frozen corpus files, and actual activation row shares. A nine-source
manifest fails validation. Compatible individual corpus files can be reused,
but the builder must add the missing structured source and complete the
manifest before extraction or training. `corpus_step.zip` is a source-code
bundle containing this builder and recipe, rather than frozen text records.
