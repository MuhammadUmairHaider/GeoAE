# GeoAE — scaling to Gemma / Qwen on another server

Companion to `HANDOFF_HPC.md` (which holds the science state, the eval pipeline
and the general traps). THIS file is only about what changes when you move off
Llama-3.2-3B. Read both.

---

## 0. What you are testing, in one line

Whether the encoder beats an **encoder-free control** (`fit_balanced_kmeans` =
identity encoder + the same Sinkhorn balancing). On Llama it mostly does not.
Every result below must be reported against that control or it means nothing.

---

## 1. THE TRAP THAT WILL COST YOU A WEEK: fp16 overflow

`extraction.dtype` defaults to **float16**. Gemma scales embeddings by sqrt(d),
so its residual stream is enormous:

| model | observed max\|x\| | fp16 ceiling | verdict |
|---|---|---|---|
| llama-3.2-3B L27 | 162 | 65,504 | fp16 fine |
| **gemma-3-4b-pt L33** | **158,720** | 65,504 | **2.4x OVER — fp16 destroys it** |

A previous fp16 gemma dump had **70% of tokens non-finite**. That makes `cdist`
return `inf`, `argmin` return 0, and produces a *plausible-looking* "77/2000
cluster collapse" that wasted real time. It does not crash.

**Every non-Llama extract config MUST set `extraction.dtype: float32`.**

Current state in this repo:

```
gemma3-4b_extract.yaml       dtype: float32   OK
gemma3-4b_l22_extract.yaml   dtype: float32   OK
gemma3-12b_extract.yaml      dtype: float32   OK
qwen_last_extract.yaml       UNSET -> float16 *** FIX BEFORE RUNNING ***
```

`extract.py` already counts non-finite elements and prints
`Fix: set extraction.dtype: float32` — do not ignore that line. After any
extract, verify independently:

```python
a = np.load(f"{dir}/layer_{L}.npy", mmap_mode="r")
assert a.dtype == np.float32
assert np.isfinite(a[:200_000]).all()
print(a.dtype, a.shape, float(np.abs(a[:200_000]).max()))
```

## 2. Disk — float32 roughly doubles everything

Per 10M tokens, ONE layer:

| model | hidden | fp16 | **float32 (what you must use)** |
|---|---|---|---|
| llama-3.2-3B | 3072 | 57 GB | 114 GB |
| gemma-3-4b-pt | 2560 | 48 GB | **95 GB** |
| gemma-3-12b-pt | 3840 | 72 GB | **143 GB** |
| Qwen3.5-9B | 4096 | 76 GB | **153 GB** |

Plus per training run: `keep_checkpoints x checkpoint_size`, where a checkpoint
is roughly `3 x (2*hidden*latent + K*latent) x 4` bytes (model + Adam state).
At latent=2*hidden, K=2000 that is ~0.5 GB for llama-scale, ~0.9 GB for Qwen.
40 kept checkpoints = 20-36 GB per run.

**Budget ~150 GB per model-layer plus ~30 GB per training arm.** Extraction
aborts up front if the target directory cannot hold the dump — believe it.

Consider fewer tokens rather than fp16: 5M tokens at float32 beats 10M at
float16, because the latter is silently corrupt.

## 3. Model geometry already established here

| model | hidden | layer used | notes |
|---|---|---|---|
| llama-3.2-3B | 3072 | 27 (final) , 14 (mid) | the reference track |
| gemma-3-4b-pt | 2560 | 33 (final of 34), 22 | Gemma Scope 2 covers L22 |
| gemma-3-12b-pt | 3840 | 47 (final) | |
| Qwen3.5-9B | 4096 | 31 (final) | |

**Gemma loading trap:** Gemma 3 loads as `Gemma3ForConditionalGeneration`; the
decoder is nested under `.language_model`, not at the top level. `hooks.py`
handles this via `locate_lm_parts`, but if you add a new family, verify the hook
attaches to the *decoder* layers and not the vision tower.

**Gemma Scope 2** (`google/gemma-scope-2-*`) provides public SAE baselines for
the gemma family. NOTHING in this repo has ever loaded an SAE — no loader, no
`sae_lens`, no weights. If you want the SAE comparison, that is new work; take
top-1 active feature per token as the "cluster" so the existing
`closest_tokens`/`llm_judge` pipeline applies unchanged.

## 4. Pipeline order for a new model

```bash
# 1. EXTRACT — check dtype: float32 in the config FIRST
python -m geoae.extract --config configs/base/<model>_extract.yaml \
       --n_tokens 10000000 --out_dir activations_<model>
#    then verify finiteness (snippet in section 1)

# 2. TRAIN — copy a llama config, change model_name/hidden_size/target_layer/
#    activations_dir/checkpoints_dir. Keep everything else identical so the
#    comparison to the llama track is one-variable.
python -m geoae.train --config configs/base/<model>_<layer>.yaml 2>&1 | tee logs/<name>.log

# 3. ENCODER-FREE CONTROL — ~2 min, needed by every comparison.
#    --activations takes the .npy FILE not the directory; --out is REQUIRED.
python -m geoae.interp.fit_balanced_kmeans --checkpoint <ckpt.pt> \
  --activations activations_<model>/layer_<L>.npy --n_clusters 2000 \
  --out checkpoints/<model>/layer<L>/balanced_kmeans_k2000.npz

# 4. CONCEPT CACHE — PER MODEL AND PER LAYER. Never reuse across either.
python -m geoae.interp.concept_suite    --checkpoint <ckpt.pt> --out cache_<model>_l<L>
python -m geoae.interp.benchmark_cache --bench ravel --checkpoint <ckpt.pt> --out cache_<model>_l<L>/ravel.npz
python -m geoae.interp.benchmark_cache --bench ioi   --checkpoint <ckpt.pt> --out cache_<model>_l<L>/ioi.npz

# 5. EVALS  (see HANDOFF_HPC.md section 6 for the full set)
python -m geoae.interp.concept_probe --cache cache_<model>_l<L> \
  --models "ae=<ckpt.pt>" --baselines "bal=<balanced.npz>" --out results/<name>.json
python -m geoae.interp.closest_tokens --checkpoint <ckpt.pt> --out results/ct_<name>.json
python -m geoae.interp.rank_clusters results/ct_<name>.json
python -m geoae.interp.llm_judge results/ct_<name>.json results/ct_<name>_bal.json \
  --provider openrouter --model google/gemini-2.5-flash-lite \
  --n_clusters 2000 --min_assigned 0 --concurrency 16 --out results/judge_<name>.json
```

`concept_suite` does NOT build ravel/ioi — that is `benchmark_cache`, and its
`--out` is a FILE path (passing a directory writes `<dir>.npz`, and passing
`x.npz` to a dir arg produced `x.npz.npz` here).

## 5. Semi-supervised seeded init — the mechanism in full

Implemented in `geoae/seeded_init.py`, switched on with
`centroid_init: "seeded"` and `reinit_mode: "anchor"`. Everything below is
config-driven; no code change is needed to port it to a new model.

### 5.1 Why bother

k-means++ chooses each next centroid with probability proportional to D^2, the
squared distance to the nearest existing centroid. That deliberately seeks the
points FURTHEST from what it already has, which minimises worst-case k-means
cost and is the wrong objective for landing a centroid on a semantic mode.
`reinit_dead_clusters` had the same bias in sharper form: it reseeded a dead
centroid from the HIGHEST-RECONSTRUCTION-LOSS sample in the batch — close to a
definition of an outlier.

Precedent: on DBpedia-14 at K=14, seeding each centroid from 5-10 labelled
examples per class reached ~93% purity where k-means++ reproduced ~55%.

### 5.2 Stage 1 — anchors (runs once, at `clustering_start_epoch`)

```
for each rung in the concept cache (19 of them):
    for each class with >= anchor_min_examples (5) examples:
        take up to anchor_per_class (25) examples, chosen with RandomState(seed)
-> pool of individual labelled examples          e.g. 11,053 rows (L27), 10,095 (L14)
-> encode all of them with the CURRENT encoder
-> per-class mean latent                          511 classes (L27), 470 (L14)
-> DEDUP: drop any class mean within
   min_sep_frac (0.25) x the median inter-anchor distance of one already kept
-> anchors                                        427 (L27, trained encoder)
```

Dedup matters because the rungs overlap: `ravel_country` and `ravel_language`
label the SAME tokens with correlated labels (a token tagged `Afghanistan` is
also tagged `Pashto,Dari`), so their class means nearly coincide. ~84 of 511
classes are absorbed this way.

Anchor count depends on how trained the encoder is — an untrained encoder gives
499 because fewer means are close enough to merge. Expect ~420 at epoch 11.

### 5.3 Stage 2 — fill the remaining K - n_anchor centroids

~1,580 of K=2000 are still unassigned. These come from a greedy
**density x coverage** rule over a batch of encoded latents:

```
score(x) = density(x)^density_power  *  d2(x, nearest already-selected)
pick argmax, repeat
```

* `density(x)` = 1 / (mean distance to the k=32 nearest of 8,192 reference
  points sampled from the batch). Higher = denser = less outlier-y.
* the `d2` factor is what stops every centroid collapsing into the single
  densest mode — it preserves k-means++'s coverage property.
* `density_power: 0` reduces this to greedy k-means++ exactly. `1.0` is the
  default.

**Calibration warning.** At `density_power: 1.0` the coverage term still
dominates: measured on a trained llama L27 checkpoint, selected points sat at
density 0.0105 against a pool median of 0.0173 — still below-median density,
just less extreme than pure coverage (0.0089). If clusters still look
outlier-seeded, raise `density_power` to 2.0-3.0 before concluding the idea
fails.

### 5.4 Stage 3 — reinit (`reinit_mode: "anchor"`, every `reinit_every` steps)

The individual labelled examples from 5.2 are retained as an `AnchorPool`.
When clusters die, they are reseeded from it **before** any unsupervised
fallback:

```
priority   per CLASS:  (distance from class mean to nearest LIVE centroid)
                       / (that class's own radius)
           -> a COVERAGE question: "does this concept have a home, relative to
              how tight it is?"  NOT a distance question.
placement  the MEAN of the anchor_mean_k (5) members closest to the class mean,
           i.e. its most typical members.
consume    the whole class is marked used once seeded, so the next reinit moves
           on to the next unserved concept instead of re-seeding the same one.
fallback   once the pool is exhausted -> D^2 sampling over the batch
           (k-means++'s rule, drawn probabilistically rather than greedily).
```

Two implementation details that matter:

* **Anchors are re-encoded at every reinit, never cached.** The encoder keeps
  training after epoch 11, so latents captured at init go stale within an epoch.
  The pool is ~10k rows, so a forward pass is free next to a reinit cycle.
* Training logs `(N from labelled anchors, M from D^2; X anchors left)` each
  cycle. **If that prints `0 from labelled anchors` while the pool is non-empty,
  the anchor path is not firing** and you are silently getting k-means++
  behaviour. That is the line to check at epoch 11.

### 5.5 v1 vs v2 — the rule that was wrong, and the evidence

The FIRST implementation picked the individual unused example *farthest* from
any live centroid. That is farthest-first selection — the same outlier-seeking
rule as k-means++, merely restricted to a labelled subset — and within a class
the farthest point is that class's most ATYPICAL member. Both runs are on disk;
the difference is stark:

| clusters reinitialised | ep11-15 | 16-20 | 21-30 | 31-40 | 41-50 | total |
|---|---|---|---|---|---|---|
| kmeans++ parent | 235 | 41 | 34 | 5 | 0 | 315 |
| v1 farthest-first | 1169 | 210 | 59 | 31 | 1 | 1470 |
| **v2 class-centre** | 786 | **18** | **15** | **3** | 1 | **823** |

v2 cut settled-phase churn (post-epoch-15) 8x versus v1 and 2x versus the
k-means++ parent. It did NOT fix the epoch-11 shock (786 vs the parent's 235) —
seeding many centroids into dense regions makes them compete immediately, and
the losers die at once. **That initial burst is the remaining open problem.**

### 5.6 Results on llama L14 (v2, anchors held out of the eval)

| | v2 seeded | kmeans++ | balanced (no encoder) |
|---|---|---|---|
| reinit churn, settled | **37** | 80 | — |
| MEAN TOKEN NMI | **0.2841** | 0.2665 | 0.2617 |
| MEAN TOKEN F1 | 0.2226 | 0.2190 | **0.2442** |
| MEAN SEQ NMI | 0.1543 | 0.1518 | **0.1609** |
| word intrusion (token-wtd) | 0.349 | 0.338 | **0.422** |
| usage perplexity | 1243 | **1492** | **1585** |
| val_mse | 0.02878 | 0.02990 | — |

Gains concentrate on **many-class entity rungs** — `ioi_name` 0.5965 vs 0.2859
for the control, all three RAVEL rungs — and it LOSES on few-class rungs
(`ioi_role` 0.3144 vs 0.4764, worst of all arms). That is the expected shape:
those rungs contribute the most anchors (ravel_country alone is 130 classes /
2,520 pool rows), so few-shot seeding hands the partition categories it would
never find unsupervised, while a 3-class rung contributes 75 rows and just loses
the capacity.

It costs usage balance (perplexity 1243 vs 1492) and raises surface coherence
(2.09 vs 1.94 — more single-token clusters). **It does not beat the
encoder-free control on anything except token NMI.**

### 5.7 Porting it to gemma / qwen

1. Build that model+layer's concept cache first (section 4 step 4) and point
   `anchor_cache` at it. **It must match `target_layer`** — a mismatched cache
   silently seeds from a different layer's geometry and every downstream number
   is quietly wrong.
2. Check the anchor count printed at `clustering_start_epoch` is in the low
   hundreds. Far fewer means the cache is thin; far more means dedup is not
   merging (raise `min_sep_frac`).
3. **Evaluate with `concept_probe --exclude_anchors`.** Anchors are drawn from
   the same caches the probe scores on — without the holdout the seeded rungs
   report an inflated result. On the first seeded run 37% of `ravel_country`
   and 27% of `ravel_language` eval rows were also anchors. Holding them out
   shrank the gains but did not erase them (ravel_country 0.222 -> 0.203, still
   +30% over k-means++), so this is legitimate class-level few-shot transfer —
   but it is SUPERVISED, and a seeded arm is not comparable to an unsupervised
   one without saying so.

### 5.8 Config block to copy

```yaml
train:
  centroid_init: "seeded"
  anchor_cache: "cache_<model>_l<L>"   # MUST match target_layer
  anchor_per_class: 25                 # labelled examples per anchor (5-100)
  anchor_min_examples: 5               # skip classes with fewer than this
  anchor_mean_k: 5                     # reinit seed = mean of N most typical
  anchor_rungs: ""                     # empty = all rungs in the cache
  density_power: 1.0                   # 0 = pure coverage; raise to 2-3 if needed
  reinit_mode: "anchor"                # unserved labelled class first, then D^2
```

## 6. Which numbers to trust (short version)

1. **word intrusion, token-weighted** — behavioural, chance 1/6, real headroom.
2. **NMI / F1, ALWAYS split token vs sequence** — `concept_probe` prints
   `MEAN TOKEN` / `MEAN SEQUENCE` automatically. The two move in opposite
   directions; a grand mean hides the only interesting structure.
3. **usage**: `rank_clusters` Zipf alpha + perplexity, or `ema_cluster_size`
   from a checkpoint.
4. NOT semantic coherence 1-5 — 96% of clusters score 4-5 in every space.
5. NOT `cluster/dying`, `eff_K`, `cluster/silhouette` from the training log —
   they measure the Sinkhorn `Q`, not geometry.

**No metric family agrees with another.** Report several; a conclusion resting
on one number has repeatedly failed here.

## 7. Traps specific to running many arms

- **Sequential, not concurrent.** Two runs reading different 56 GB dumps
  thrash the page cache: epochs went 10 min -> 84 min. Warm one file first with
  `dd if=<layer>.npy of=/dev/null bs=8M` (~60s) and run one at a time.
- **`2>&1 | tee`, not `| tee`.** Two runs died at the seeded init and looked
  like clean shutdowns because the traceback went to stderr and was never logged.
- **Untracked files get lost.** A config created after a commit vanished (git
  clean). `git add` new configs immediately.
- **`meta.checkpoint` in a `closest_tokens` JSON is a bare path.** Renaming a
  checkpoint directory silently repoints it, and `llm_judge` then draws hard
  negatives from the WRONG model's centroids — it produced 0.635 instead of
  0.453 with no error. Do not rename checkpoint dirs that results point at.
- **`best_val.pt` is usually the wrong checkpoint for phased runs** (val_mse
  peaks before clustering starts). Use the last `step_*.pt`; check `epoch` and
  `ema_cluster_size` if in doubt.
- **Do not judge cluster health before ~epoch 40.** One run sat at 83% of tokens
  in a single cluster through epochs 15-30 and self-healed by 31.
- **Do not report anything below ~50 units** without a sensitivity check at two
  thresholds. Five results reversed under proper power in one session.
