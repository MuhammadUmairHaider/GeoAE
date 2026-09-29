# GeoAE round 2 on Delta — command sheet

For the checkout at **`/projects/bbyl/mhaider/GeoAE`**, branch `clean`. Written
2026-09-26. Supersedes `HANDOFF_DELTA.md` §0 and §3 for bring-up and for the runs
below; the findings, traps (§6) and run inventory (§4) there still hold.

**Plan: one arm per question, single seed.** Replicates are deferred (§7 has them
ready if the margins turn out to be too tight to read). That means the existing
`eval_out/run_tokbias*.sh` sheets apply directly — they are wired to exactly these
checkpoints, so nothing has to be driven by hand.

| | question | arm | GPU-h |
|---|---|---|---:|
| A | — | the new sampled dump | 2 |
| B | base vs bypass, on the new data | `d6144_dpc_sampled_tokbias` | 13 |
| C | — | its full eval suite | 8 |
| D | is the help/hurt split a data artefact? | `..._struct5_tokbias` | 20 |
| E | does bypass survive 4x width? | `d12288_dpc_sampled_tokbias` | 27 |

~70 GPU-h total, comfortably inside `bbyl-delta-gpu`'s 201 h. No `sbatch` line needs
`-A`: `env.sh` exports `SBATCH_ACCOUNT`. CPU-only steps go to `$DELTA_CPU_ACCOUNT`
(bbyl has no CPU allocation).

---

## 0. Start here

```bash
cd /projects/bbyl/mhaider/GeoAE
source scripts/delta/env.sh
```

Bring-up is **done**: venv on `/work/nvme/bbyl`, data links resolve, DBpedia
activations copied, `clean` current with origin at `7ddce85`. Optional:
`uv run --no-sync wandb login` (or add `--no_wandb` to each train line), and
`sbatch scripts/delta/verify.sbatch` to prove the GPU path (324 tests + a real Llama
forward, ~0.1 h).

### 0a. Storage — nothing needs deleting

The bulk data root is **`/work/hdd/bimc/mhaider/GeoAE`** (`$DELTA_DATA`), set in
`scripts/delta/site.sh`. GPU hours are still charged to `bbyl-delta-gpu`: **compute
account and data location are independent** — group membership grants access, not
whoever pays for the node.

Why not `/work/hdd/bbyl`: every allocation has three storage tiers, so four
allocations give twelve directories and ~5 TB free between them. `/work/hdd/bbyl`
happens to be the crowded one — 822 GB of its 1 TB with **eight users** on it, ~178 GB
free against a ~250 GB campaign. `/work/hdd/bimc` is empty with the full 1 TB.

```
                   USED    QUOTA     FREE
/projects/bbyl     130G     500G     370G   <- code; the repo is here
/work/nvme/bbyl      6G     500G     494G   <- the venv is here (fast metadata)
/work/hdd/bbyl     822G    1000G     178G   <- crowded; holds the 2026-09 caches
/work/hdd/bimc       0G    1000G    1000G   <- $DELTA_DATA: dumps + checkpoints
/work/hdd/bgrb     404G    1000G     596G   <- fallback, still holds the old dumps
TOTAL             3549G    8500G    4951G
```

The 1 TB figures are **project quotas, not disks**: the underlying Lustre filesystem
is petabytes, and `df` inside one of these trees reports the quota as the filesystem
size, which makes it look far smaller than it is. To move the data, change the one
`DELTA_DATA` line in `site.sh` and re-run `scripts/delta/link_data.sh`; nothing else
in the repo names a data path.

Already copied into the new root and verified through the links: the gated Llama
weights and eval datasets (`cache/huggingface`, 13 GB — loads offline, no token), the
L27 concept caches (4.7 GB) and the DBpedia-14 activations (673 MB, 14 classes) — all
independent of the training dump, so they carry over.

The concept caches were **moved up** from `cache/concept_suite_llama_l27/` to
`cache/` while doing this. Every tool defaults to `--cache cache` and expects
`cache/<rung>.npz`, and `eval_out/run_tokbias.sh`'s `cache` step symlinks the token
rungs as `ln -s ../cache/pos.npz`; the Sept-1 campaign had written them one level
down, so left alone they would have been silently invisible and every probe eval
would have rebuilt them from scratch. Verified: `cache/pos.npz` and `cache/ner.npz`
carry `token_id`, which is what makes them reusable for `cache_ids/`.

**`clean_slate.sh` is now OPTIONAL.** It retires the pre-2026-09-26 data distribution
still sitting on `/work/hdd/bbyl` (legacy L27 dump 58 G, L14 twin 58 G, a raced
leftover 6.1 G, the k-means++ checkpoint trained on the legacy dump 24 G = ~146 GB).
Nothing in this campaign reads any of it, and nothing here needs the space, so run it
when you want that allocation's quota back rather than as a prerequisite:

```bash
bash scripts/delta/clean_slate.sh            # dry run, read the list
DELTA_DATA=/work/hdd/bbyl/$USER/GeoAE bash scripts/delta/clean_slate.sh --delete
```

---

## 1. Stage A — extract the new dump (~2 GPU-h + ~40 CPU-min)

This is the "new dataset setting": full documents up to 2048 tokens with 64 random
positions each (~157k docs vs the legacy dump's ~43k), in a pretraining-like mix
(fineweb-edu 40 / fineweb 15 / en-wiki 15 / Pile tail 10 / code 8 / finemath 5 /
de-fr-es wiki 7) instead of the legacy equal 5-way mix that was 43% code and math.
10M rows x 3072 x fp16 = **61 GB**.

```bash
# A1. corpus -> corpus/sampled_v1   (~40 min, 0.3 GB, CPU account, no GPU hours)
CJOB=$(sbatch --parsable -A "$DELTA_CPU_ACCOUNT" scripts/delta/corpus.sbatch)
echo "corpus job $CJOB"

# A2. dump + token-bias table       (~1.75 h, 61 GB + 0.3 GB, one A100)
DJOB=$(SKIP_CORPUS=1 sbatch --parsable --dependency=afterok:$CJOB \
  scripts/delta/extract_sampled.sbatch)
echo "extract job $DJOB"

squeue -u $USER
```

`build_corpus` only streams and tokenises — it never touches a GPU — so it is split
onto a CPU allocation rather than spending ~40 min of an A100's hours on network I/O.
`SKIP_CORPUS=1` tells the GPU job the corpus is already built. Both steps are
resumable: resubmitting the same line skips whatever is already complete.

**The corpus is not the old box's.** `corpus/sampled_v1` is not on Delta and cannot be
copied from it, so this re-streams from the hub and gets a different document sample.
Every Delta arm shares this one corpus and so is mutually comparable; against old-box
numbers it is replicate noise on two levels. That was already the rule
(`HANDOFF_DELTA.md` §0, "Comparability").

Checks when A2 finishes:

```bash
scripts/delta/py - <<'PY'
import json; m = json.load(open("activations_sampled_10M/meta.json"))
print(m["n_tokens"], m["n_docs"], m["source_share"])
print("median norm", m["outliers"]["median_norm"], m["outliers"]["frac_over"])
PY
ls -la e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz
tail -3 logs/token_bias_activations_sampled_10M.log
quota | grep work/hdd/bimc
```

Expect 10,000,000 rows from ~157k docs, `source_share` equal to the config weights
(40/15/15/10/8/5/3/2/2 %), `>10x` outliers ~1e-7, and a token-bias table of ~305 MB
covering ~98% of rows with ~0.17 variance removed (the old box's numbers). The sampled
extractor writes **no `validation.json`** — that file belonged to the legacy dump; the
median L27 norm lives in `meta.json` under `outliers`. `scripts/delta/verify.sbatch`
prints the median straight from the model (54.9 on 2026-09-19), so the two are
cross-checkable if a dump looks wrong; the sampled dump reaches positions up to 2047,
so a small offset from the prefix-only figure is expected, a large one is not.

Checked 2026-09-26 on dt-login03, before the first Delta extraction: all ten sources
(nine HF streams + the struct5 structured source) open and yield text under
datasets 5.0.0 / transformers 5.14.1; the Pile `meta.pile_set_name` exclude filter
fires (~half the raw stream dropped, as intended — a missing key would have silently
let Pile-CC/GitHub/Wikipedia through); `tests/test_extract_sampled.py` and
`tests/test_token_bias.py` pass (27).

---

## 2. Stage B — the bypass AE on the new dump (~13 GPU-h)

```bash
# B1. the main arm. ~9 h (10.8 min/epoch x 50).
sbatch --dependency=afterok:$DJOB scripts/delta/train.sbatch \
  configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias.yaml
```

> **Confirm steps/epoch from the first epoch in the log**, then stop worrying about
> it. Everything downstream calls epoch 50 `step_0014450.pt`, i.e. **289 steps/epoch**;
> the legacy dump gave 284. That filename is hardcoded inside the eval sheets, so if
> this dump reports a different number, fix it there once, before the evals.
>
> ```bash
> tail -f logs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias.log
> ```

```bash
# B2. THE BASE ARM — balanced k-means on the plain residual, no encoder: same K, dpc
#     init, Sinkhorn balancing. This is the "base" in every base-vs-bypass number.
#     It only borrows a checkpoint's normalisation, so any saved epoch will do; pick
#     it by glob so this does not depend on steps/epoch. ~1 h.
E=e2e/checkpoints/general/llama3.2-3B/layer27
B=checkpoints/llama3.2-3B/layer27
until ls $B/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_*.pt >/dev/null 2>&1; do sleep 60; done
CK=$(ls -1 $B/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_*.pt | head -1)
echo "borrowing normalisation from $CK"

sbatch --mem=96g scripts/delta/eval.sbatch geoae.interp.fit_balanced_kmeans \
  --checkpoint $CK --activations activations_sampled_10M/layer_27.npy \
  --init dpc --reinit peaks --n_clusters 2000 --seed 42 \
  --out $E/balanced_kmeans_k2000_dpc_sampled.npz

# B3. The encoder-free CONTROL: the same balanced k-means on x - b[token]. Keep this
#     one — on the old box it TIED the bypass AE on clustering, which is why the
#     clustering win is credited to the per-token subtraction rather than to the
#     encoder (docs/notes/token-bypass-design-b.md). It is the arm that says whether
#     the encoder is doing anything at all. ~1 h.
sbatch --mem=96g scripts/delta/eval.sbatch geoae.interp.fit_balanced_kmeans \
  --checkpoint $CK --activations activations_sampled_10M/layer_27.npy \
  --init dpc --reinit peaks --n_clusters 2000 --seed 42 \
  --token_bias $E/token_bias_sampled.npz \
  --out $E/balanced_kmeans_k2000_dpc_sampled_tokmean.npz

# B4. Probe caches. cache_ids/ MUST be rebuilt: the sequence rungs need
#     last_token_id, which the 2026-09-01 caches do not carry. The token rungs are
#     symlinked from cache/, which clean_slate.sh kept. ~1 h.
sbatch scripts/delta/sheet.sbatch eval_out/run_tokbias.sh cache
#     Absent from cache/ and needed by some rungs:
sbatch scripts/delta/eval.sbatch geoae.interp.benchmark_cache --bench ravel --checkpoint $CK --out cache/ravel.npz
sbatch scripts/delta/eval.sbatch geoae.interp.benchmark_cache --bench ioi   --checkpoint $CK --out cache/ioi.npz
```

---

## 3. Stage C — the full eval suite (~8 GPU-h)

The sheets are wired to exactly this checkpoint, so they run as-is.

```bash
# geometry, chance-corrected probes, MMLU splice, clustering quality  (~1.5 h)
sbatch --time=8:00:00 scripts/delta/sheet.sbatch eval_out/run_tokbias.sh evals

# number control, dense steering, DB14 + bias_in_bios range interventions  (~5 h)
sbatch --time=10:00:00 scripts/delta/sheet.sbatch eval_out/run_tokbias_interventions.sh

# generation-level cluster steering  (~20 min GPU) then the LLM judge (API, not GPU)
sbatch scripts/delta/sheet.sbatch eval_out/run_cluster_steer_generate.sh gen
sbatch -A "$DELTA_CPU_ACCOUNT" -p cpu scripts/delta/sheet.sbatch \
  eval_out/run_cluster_steer_generate.sh judge      # needs OPENROUTER_API_KEY in .env

# decision tables
sbatch -A "$DELTA_CPU_ACCOUNT" -p cpu scripts/delta/sheet.sbatch eval_out/run_tokbias.sh summary
```

Each sheet skips finished steps and reports failures at the end; `FORCE=1` redoes one.

> Traps these sheets already handle, listed so a surprising number is recognisable:
> `--saliency dprime` not mean-|a| (mean-|a| picks the shifted-offset dims of a GELU
> latent, and two earlier conclusions were artefacts of it); comparison at **matched
> erasure**, not fixed α; bias_in_bios `teacher` excluded because its few-shot accuracy
> is 0%. **Ignore `clustering_quality`'s `dunn`** — unseeded subsample, swings ~50%
> between runs. `steer` selectivity's sign is the negation of `range`'s.
>
> Each AE rebuilds its own joint-correct document set (keyed on the weight
> fingerprint), so each intervention tool spends a while predicting first. The
> committed `dbpedia/joint_correct_*.json` belong to the old-box checkpoints and will
> not be reused. Expected, not a failure.

---

## 4. Stage D — struct5, the data-artefact test (~20 GPU-h)

Optional and independent of E. Asks whether the help/hurt split is a training-sample
artefact: the structured source is the kind of text the intervention evals are built
from, and the pretraining-like mix has almost none of it.

```bash
# D1. corpus: SHARES corpus/sampled_v1, so this only adds the structured source
#     (CounterFact true facts + NQ-open Q&A, 8 items per doc, RAVEL-entity and
#     IOI-template overlaps DROPPED so the probe benchmarks stay clean).
#     Must run AFTER A1 ($CJOB): both jobs rewrite corpus/sampled_v1/manifest.json from
#     the copy they read at start, so running them concurrently loses one's entries.
SCJOB=$(sbatch --parsable -A "$DELTA_CPU_ACCOUNT" --dependency=afterok:$CJOB \
  scripts/delta/corpus.sbatch configs/base/llama3.2-3b_extract_sampled_struct5.yaml)

# D2. dump + ITS OWN token-bias table. TABLE= is MANDATORY: b is the per-token mean
#     over THIS dump's train split, so the no-struct table belongs to the no-struct
#     dump. extract_sampled.sbatch ABORTS on a non-default config with TABLE unset,
#     precisely so this cannot go wrong silently.
SDJOB=$(SKIP_CORPUS=1 TABLE=$E/token_bias_sampled_struct5.npz \
  sbatch --parsable --dependency=afterok:$SCJOB \
  scripts/delta/extract_sampled.sbatch configs/base/llama3.2-3b_extract_sampled_struct5.yaml)

# D3. the arm, plus its own base arm on its own dump.
sbatch --dependency=afterok:$SDJOB scripts/delta/train.sbatch \
  configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_struct5_tokbias.yaml

CK5=$(ls -1 $B/k2000_bnh_b32k_lam1_d6144_dpc_sampled_struct5_tokbias/step_*.pt | head -1)
sbatch --mem=96g scripts/delta/eval.sbatch geoae.interp.fit_balanced_kmeans \
  --checkpoint $CK5 --activations activations_sampled_struct5_10M/layer_27.npy \
  --init dpc --reinit peaks --n_clusters 2000 --seed 42 \
  --out $E/balanced_kmeans_k2000_dpc_sampled_struct5.npz
```

Then Stage C's tools against the struct5 checkpoint and its base fit.

> **Compare Δ(z−h) between dumps, never z between dumps.** The two dumps give
> different base residual populations, so raw struct5-vs-no-struct numbers are not a
> comparison. The question is only whether the *margin* moves.

---

## 5. Stage E — 4x width (~27 GPU-h)

Optional and independent of D.

```bash
sbatch --time=36:00:00 scripts/delta/train.sbatch \
  configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias.yaml
# ~18-20 h, ~22 GB peak GPU, 41 ckpts x ~1.31 GB = 54 GB. Resubmit the same line if
# it hits the walltime — it resumes from the newest checkpoint.

sbatch --mem=128g --time=8:00:00 scripts/delta/sheet.sbatch eval_out/run_tokbias_d12288.sh evals
sbatch --time=12:00:00 scripts/delta/sheet.sbatch eval_out/run_tokbias_d12288.sh interventions
sbatch scripts/delta/sheet.sbatch eval_out/run_tokbias_d12288.sh steering
sbatch scripts/delta/sheet.sbatch eval_out/run_tokbias_d12288.sh summary
```

`--mem=128g` on the `evals` line is required, not cautious: `clustering_quality`
holds `n_sample × latent_dim` float32 in RAM, and 1M × 12288 = **49 GB**. That sheet
also has a checkpoint guard that refuses to score an AE whose centroids were never
initialised, whose width is wrong, or that is missing its token-bias table — let it
run, it catches the failure that wastes an eval night.

---

## 6. Disk and budget

| stage | GPU-h | disk |
|---|---:|---:|
| A. sampled dump + table | 2 | 61 GB |
| B. d6144 bypass + 2 k-means + caches | 13 | 38 GB |
| C. full evals | 8 | — |
| D. struct5 dump + arm + evals | 20 | 95 GB |
| E. d12288 arm + evals | 27 | 54 GB |
| **total** | **~70** | **~248 GB** |

`$DELTA_DATA` is `/work/hdd/bimc`, which starts empty with a 1 TB quota, so ~248 GB
of campaign lands in ~1000 GB of headroom with nobody else on it. That is the reason
the data root is not `/work/hdd/bbyl` (~178 GB free, eight users). Still worth a
`quota | grep work/hdd/bimc` before D or E, but it is no longer close.

---

## 7. If the margins turn out too tight to read — seeds

Deferred, not discarded. The three headline claims are all single-seed, and several
numbers in `docs/notes/` rest on margins of 0.03-0.05, which a single run cannot
separate from noise. Replicate configs for seeds 43-46 are already committed
(`..._tokbias_s43.yaml` and so on, `keep_checkpoints: 3`, ~2.4 GB each), and more come
from:

```bash
scripts/delta/py scripts/make_seed_config.py \
  --base configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias.yaml \
  --seeds 47 48
```

Two rules if you do run them:

- **Refit the base arm at the matched seed** (`fit_balanced_kmeans --seed N`). One base
  fit compared against several AE seeds folds the base's seed variance into the AE's.
- **Average the within-run margin z − h**, not raw z. Evaluation populations are
  rebuilt per AE, so raw cross-arm gaps under ~0.03 are noise
  (`docs/notes/init-arms-dpc-results.md`).

The sheets are wired to the seed-42 checkpoints, so replicates have to be driven tool
by tool — `git log` for this file's earlier revision has that loop written out.

---

## 8. Traps

- **`best_val.pt` can be a pre-clustering epoch.** Val MSE rises once the cluster loss
  starts at epoch 11; on the tanh arm `best_val.pt` is epoch 5 with all-zero
  centroids. Always evaluate the final `step_*.pt`.
- **Delta checkpoints are new replicates.** Compare Delta arms with each other; the
  committed `results/*.json` belong to the old box.
- **b32k runs collapse transiently and self-heal by epoch 40.** A variance-hinge spike
  and a jump in dying clusters around epochs 12-20 is not a reason to kill a run
  (`docs/notes/b32k-transient-collapse.md`).
- **The W&B cluster/silhouette/entropy panels do not measure geometry.** They measure
  the Sinkhorn output on one batch, and the correctly-computed silhouette flips the
  sign (`docs/notes/train-diag-metrics-misleading.md`). Judge on val MSE and the
  offline evals.
- **Training stages the 61 GB dump to node-local `/tmp` per job.** Correct whether or
  not Delta gives each job a private `/tmp`; do not "optimise" it into a node-shared
  path without testing that, or an epilog could pull the dump out from under a
  neighbouring job mid-run.

---

## 9. What to report

Base vs bypass AE only, in plain units
(`docs/notes/report-base-vs-bypass-only.md`):

| claim | metric | old box |
|---|---|---|
| topic clustering | chance-corrected NMI, topic14 | 0.60 bypass vs 0.45 base |
| intervention parity | bias_in_bios collateral at matched erasure | first AE at parity with base |
| steering | usable on-target continuations per 100 | ~15 bypass vs ~6.5 base |

Keep `km_tokmean` in the tables you look at even though it is not in the report: if
the encoder-free control ties the bypass AE again, the clustering win belongs to the
per-token subtraction, and that is the finding rather than a footnote.
