# GeoAE — handoff for NCSA Delta (2026-09-18, updated 2026-09-19)

Moving off the Jetstream box (`circuits`, 484 GB disk, full) to NCSA Delta.
**Nothing large is transferred.** Code, configs, small results and notes are in
git; activation dumps and checkpoints are regenerated on Delta from code.

Supersedes `HANDOFF_HPC.md` (2026-09-03), which is still accurate for the older
findings it lists.

**Read with this:** `docs/notes/MEMORY.md` indexes 16 notes, one finding each.
These were Claude's machine-local memory at
`~/.claude/projects/-home-exouser-RepresentationAE-GeoAE/memory/`, copied into the
repo because that path doesn't exist on Delta. To give a Claude session on Delta
the same context, copy them into its memory directory (the path is derived from
the working directory, so it will differ).

---

## 1. State of play

The project goal is a latent space more geometrically separated and more
semantically organised than Llama-3.2-3B's layer-27 residual stream. After the
2026-09-03 handoff, the work split into three threads.

**Centroid initialisation.** Density-peaks init (`centroid_init: dpc`) replaced
k-means++ in the main arm. It cuts dead-cluster churn ~9x (260 vs 2,394 reinits)
and helps token-level concepts (POS, NER), but loses on topic/sequence rungs. Arm
choice makes no measurable difference to steering once the evaluation-set
confound is removed (see `docs/notes/init-arms-dpc-results.md`).

**tanh encoder** (`..._dpc_tanh`). The best dense-steering arm so far (latent vs
base margin −0.075, p=.003, 13/14 concepts), but effective rank halves and
reconstruction is worse (val MSE 0.049 vs 0.035). Its `best_val.pt` is epoch 5 —
see traps. (`docs/notes/tanh-encoder-arm.md`)

**NeuronLens range interventions**, base residual (h) vs AE latent (z), via the
new `geoae.interp.range_intervention_compare`:
- Rank dims by d′ saliency (`--saliency dprime`), not mean |a|. Two earlier
  conclusions were artefacts of mean-|a|: the "zero-replacement destroys the
  latent" collapse and a null at matched erasure.
- Range gating beats full-neuron ablation in both spaces, on every dataset. This
  is the paper's claim and it replicates robustly.
- AE vs base is **dataset-dependent**. DBpedia-14: the AE breaks ~25% fewer
  other-class docs at matched erasure (p≈.002–.015). bias_in_bios, 27
  professions: the AE is marginally worse (+0.03, p≤.015).
- Per concept, the AE margin shrinks as the base gets better: +0.09 where the
  base is weakest, −0.03 where it is strongest. After controlling for dataset,
  base quality isn't significant, so there is a residual dataset effect.
  (`docs/notes/range-interventions-h-vs-z.md`, `docs/notes/biasbios-range-reversal.md`)

**Latent width** (d3072 vs d6144, both at epoch 50). Going from 2x to 1x hidden
size cost a lot of reconstruction (val MSE 0.095 vs 0.035; MMLU under
reconstruction −0.066 vs −0.027) and tied on probes and kNN. On range
interventions it amplified the effect in both directions: a bigger AE advantage
on DBpedia, a bigger disadvantage on bias_in_bios, and a much steeper base-quality
moderation (rho −0.77 vs −0.29). d6144 is still the better model.
(`docs/notes/d3072-width-arm.md`)

**Standing caveat** from earlier work: an encoder-free control (identity encoder
+ the same Sinkhorn balancing, `geoae.interp.fit_balanced_kmeans`) matches or
beats the AE on most concept metrics. Balancing carries most of the gain
(`docs/notes/fineweb-atlas-concept-alignment.md`).

---

## 2. Delta essentials

Checked against docs.ncsa.illinois.edu/systems/delta on 2026-09-18.

| | |
|---|---|
| Login | `ssh <user>@login.delta.ncsa.illinois.edu` — password + Duo; SSH keys are disabled. Use tmux on a fixed node (`dt-login01..04`). |
| `$HOME` = `/u/$USER` | **100 GB, 750k files.** Code only. `env.sh` moves the HF, uv, wandb and triton caches off it. |
| `/work/hdd/<project>` | 1 TB default per project, not purged, **no backups**. The repo, dumps and checkpoints live here. |
| `/tmp` on GPU nodes | ~1.5 TB local NVMe, wiped after each job. `train.sbatch` stages the dump here. |
| GPU partitions | `gpuA100x4` (default), `gpuA40x4`, `gpuA100x8`, `gpuH200x8`; 48 h max walltime |
| Account | `accounts` lists them; pass `-A <account>` to every `sbatch` |
| Bulk transfer | Globus collection "NCSA Delta" (not needed here — nothing is transferred) |

---

## 3. Bring-up, in order

```bash
# 0. login node, inside tmux
export DELTA_PROJECT=<your allocation dir under /work/hdd>
git clone -b clean https://github.com/MuhammadUmairHaider/GeoAE.git /work/hdd/$DELTA_PROJECT/$USER/GeoAE
bash /work/hdd/$DELTA_PROJECT/$USER/GeoAE/scripts/delta/setup.sh
source /work/hdd/$DELTA_PROJECT/$USER/GeoAE/scripts/delta/env.sh
uv run huggingface-cli login        # Llama-3.2-3B is gated
uv run wandb login                  # or add --no_wandb to training

# 1. re-extract the L27 dump (~30 min, ~60 GB)
sbatch -A <account> scripts/delta/extract.sbatch

# 2. train — stages the dump to /tmp and auto-resumes if checkpoints exist
sbatch -A <account> scripts/delta/train.sbatch configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc.yaml

# 3. concept caches for the probe evals (need any AE checkpoint for model + layer,
#    so run after the first epoch has saved). ~16 GB, cache/ is gitignored.
CK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/step_0000284.pt
sbatch -A <account> scripts/delta/eval.sbatch geoae.interp.concept_suite --checkpoint $CK --out cache
sbatch -A <account> scripts/delta/eval.sbatch geoae.interp.benchmark_cache --bench ravel --checkpoint $CK --out cache/ravel.npz
sbatch -A <account> scripts/delta/eval.sbatch geoae.interp.benchmark_cache --bench ioi   --checkpoint $CK --out cache/ioi.npz
sbatch -A <account> scripts/delta/eval.sbatch geoae.interp.atlas_cache --checkpoint $CK
#    atlas_doc/tone/content.npz and atlas8k_last.npz: see geoae/interp/prepare_atlas_probe.py
#    (not re-verified in this migration — check --help before relying on it).

# 4. evals: any module via eval.sbatch. Every result JSON records its own
#    checkpoint and settings in its `meta` block; eval_out/*.sh holds the
#    command sheets for the init-arm, tanh and range runs.
```

**Throughput reference (old box, one A100 40 GB):** 7.2 min/epoch for d3072,
10.5 min/epoch for d6144, 50 epochs per run. DB14 range eval ~45 min; bias_in_bios
~3 h including building its document set.

---

## 4. Runs to regenerate

All are layer 27 unless noted. Status is as of the old box.

| config (`configs/base/llama3.2-3b_…`) | status | why it matters |
|---|---|---|
| `l27_…_lam1_d6144_dpc` | done, ep 50 | **main arm**; every range/steering result |
| `l27_…_lam1_d6144` | done, ep 50 | k-means++ parent, init comparison |
| `l27_…_lam1_d6144_seeded_atlas` | done, ep 50 | labelled-anchor init |
| `l27_…_lam1_d6144_dpc_tanh` | done, ep 50 | tanh arm |
| `l27_…_lam1_d3072_dpc` | done, ep 50, fully evaluated | width arm at 1x hidden. `best_val.pt` is epoch 10 (pre-clustering) — evaluate `step_0014200.pt` |
| `l27_…_lam1_d12288_dpc` | never run | 4x hidden, completes the width series; ~17 h on one A100 40 GB, ~1 GB checkpoints. Eval sheet: `eval_out/run_d12288_evals.sh` |
| `l27_…_lam1_d6144_seeded_peaks` | never run | anchors + density-peaks fill |
| `l14_…_lam1_d6144*` | checkpoints exist; see `logs/` for completion | layer-14 track; needs the L14 dump |

The `lam2_*`, `d768*` and `cov*` configs are older sweeps. Their logs are in
`logs/`; check those and `docs/notes/` before re-running them.

The Gemma and Qwen tracks (`GEMMA3_4B_*_COMMANDS.md`,
`HANDOFF_SCALING_GEMMA_QWEN.md`) also regenerate from code. Gemma must extract
in float32.

---

## 5. What's in git and what isn't

| in git | not in git (regenerate) |
|---|---|
| all code, tests (180), configs, docs | activation dumps (`activations_*`) |
| `results/*.json` — the evidence for every note (tracked by default since 2026-09-19) | closest-token dumps `results/ct_*`, `results/closest_tokens_*` (~440 MB, regenerable) |
| `dbpedia/joint_correct_*.json` — pins each eval's document set (tracked by default) | checkpoints, `sweeps/`, `e2e/checkpoints/` |
| `eval_out/` result JSONs and command sheets | `cache/` concept caches, `wandb/` |
| `docs/notes/` research notes; `logs/` training + eval logs (16 MB) | HF / uv caches |

Joint-correct document sets are keyed on the AE's weight fingerprint. A
retrained checkpoint gets a new one, so its set is rebuilt automatically. The
committed sets only reproduce results for bit-identical checkpoints.

---

## 5b. Multi-GPU: not built yet

Extraction, training and evals are all single-GPU today (one GPU per sbatch job;
Delta's scheduler runs independent jobs concurrently). To be designed on Delta:
- **Extraction:** shard documents across a node's GPUs and merge into the same
  single-file dump. Keep the merged row order deterministic — the last 5% of rows
  is the validation split.
- **Independent runs and evals:** one per GPU, packed onto a node.
- **DDP for one run** would need Sinkhorn balancing, EMA centroid updates and
  dead-cluster reinit synchronised across ranks; per-rank Sinkhorn on a sub-batch
  changes the balancing, so results would stop being comparable to earlier runs.

## 6. Traps that cost real time

- **`best_val.pt` can be a pre-clustering epoch.** Val MSE rises once the cluster
  loss starts at epoch 11. On the tanh run it never recovered, so `best_val.pt`
  is epoch 5 with all-zero centroids. Evaluate the final `step_*.pt` for those
  runs. `--resume` refuses to resume from an older checkpoint than the newest in
  the same directory, because that would overwrite later epochs.
- **Evaluation populations are rebuilt per AE.** Cross-arm steering gaps under
  ~0.03 are noise. Compare within-run z − h, not raw z. (`docs/notes/init-arms-dpc-results.md`)
- **α is not comparable across latents.** The same class-gap step erases
  different amounts in different latents. Compare at matched erasure, not at
  fixed α.
- **Mean-|a| saliency picks the shifted-offset dims of GELU latents.** Use d′.
- **Few-shot accuracy gates which classes can be evaluated.** bias_in_bios
  `teacher` is 0% (always predicted "professor"). Exclude it with `--concepts`;
  the joint-correct build now only scans requested classes.
- **Disk.** Checkpoint saves are now atomic and check free space first, and
  `--resume latest` skips truncated files. Four 40-epoch d6144 runs are still
  80 GB, so watch the `/work/hdd` quota (`quota`).
- **`clustering_quality` holds `n_sample × latent_dim` float32 latents in RAM**:
  1M × 6144 = 25 GB, 1M × 12288 = 49 GB. Request `--mem` accordingly (eval.sbatch
  asks for 64g, too little for d12288 at 1M). Its sklearn Calinski-Harabasz used to
  add a full float64 copy on top, which got evals OOM-killed; that is fixed.
- **b32k runs have transient collapses** (variance-hinge spike, dying clusters
  up) that self-heal. Don't kill a run for one. (`docs/notes/b32k-transient-collapse.md`)

---

## 7. Open questions, in priority order

1. **Width series: run `d12288_dpc`** (4x hidden). If compression drives the
   help/hurt split, 4x should flatten it. Note that no run has a true bottleneck
   yet: d3072 equals the hidden size, and the only sub-hidden run (`d768`) stopped
   at epoch 2. Its "compression" comes from the cluster loss pulling latents
   toward 2,000 centroids, not from a narrower code.
1b. **Is the help/hurt split a training-sample artefact?** Candidate tests: per
   concept, does coverage in the training dump or preservation of the concept's
   class-mean direction predict the AE margin; an encoder-free PCA control; a
   data-scale dose-response. The old box trained on 9.3M unique tokens × 50 epochs;
   Delta can hold a much larger, more diverse dump.
2. **Separate base quality from dataset** in the range results. That needs a
   dataset whose own concepts span a wide range of base-intervention quality.
   bias_in_bios sits near the ceiling. A GoEmotions-28 extractor exists
   (`geoae/goemotions/extract.py`) but isn't wired into the harness.
3. **tanh with d′ saliency.** It was queued and never run.
4. **Range operators at α > 2**, to tell precision from reach where one space
   saturates first.
