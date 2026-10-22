# Laptop setup and recovered research state

Setup completed October 21, 2026 (America/New_York). This Mac workspace is the
development home base. The original instructions remain in
[LAPTOP_HANDOFF.md](LAPTOP_HANDOFF.md); machine-readable verification evidence is
in [laptop_setup_evidence.json](laptop_setup_evidence.json).

## Setup verdict

The laptop handoff checks passed. The old Jetstream VM is ready for the user to
shelve. This session did not shelve or delete it, submit jobs, restart experiments,
download research datasets, or move GPU work onto the Mac.

| Check | Result |
| --- | --- |
| Mac | Apple M5, Mac17,4, native arm64, 16 GiB RAM, macOS 27.0.1 (26A434) |
| Package manager | uv 0.13.5 installed in `~/.local/bin`; available in a new zsh login shell |
| Supported setup | `bash scripts/remote/setup_laptop.sh` succeeded |
| Local interpreter | `.venv/bin/python`, CPython 3.12.15, native arm64 |
| Dependencies | PyTorch 2.13.0; all 81 installed packages compatible |
| CPU checks | All 45 tests passed in 3.60 seconds |
| Lockfile | `uv.lock` unchanged |
| Current review | `core_comparison.csv` parses: three decoder rows, 13 columns |
| Delta SSH | Direct Mac login as `mhaider`; archive directory accessible |
| Argo SSH | Direct Mac login as `dogar`; run and evaluation directories accessible |

Use `.venv/bin/python` as the project interpreter. System `/usr/bin/python3`
remains Python 3.9.6. Recheck the installed environment with:

```bash
bash scripts/remote/setup_laptop.sh --smoke-only
```

The tests use synthetic CPU tensors and temporary fixtures, with Hugging Face
downloads disabled. The local environment has no CUDA requirement. GeoPretrain
has separate dependencies and was not installed by the GeoAE setup script.

The existing SSH config had uppercase-only aliases. Both cases now work:
`Host Delta delta` and `Host Argo argo`; both use the handoff's keepalive settings.
The original config is backed up at `~/.ssh/config.before-geoae-laptop-setup`.
The user authenticated in macOS Terminal and completed MFA. Temporary connection
masters expire after 30 minutes of inactivity; future logins can prompt again.
No proxy through Jetstream is configured for either cluster.

## Preservation checks

The Delta archive is
`/projects/bbyl/mhaider/jetstream_migration_20261021/`.

- Its `control/status.json` and `control/verification.json` report **verified**
  for **8,482 selected files**, with full content checks and whole-second metadata
  verification. Original nanosecond timestamps have a separate manifest.
- The 160-entry skipped-artifact inventory distinguishes existing Delta/Argo
  originals, duplicate weights, regenerable caches, and nonportable environments.
  This is a selective preservation set; preliminary broad-copy leftovers are
  outside that verification claim.
- The Mac tar's SHA-256 matches the local sidecar and Delta's actual tar:
  `f821ece2a7136c8166306c6092446786350bdd15aafa4bb747367f8b528a1609`.
- All 6,169 regular tar members were checked against extraction. Research files,
  results, context and Git objects match. Git status refreshed index stat caches;
  indexed content, modes, paths and flags remain identical.
- `packed/geoae-llm-judge-cache.tar.gz` passed a fresh remote `sha256sum -c`.
  Its separate source-content verification receipt is also successful. It stays
  packed on Delta; no paid API calls were made.
- `control/tmp-verification.json` verifies all 46 selected staged `/tmp` research
  artifacts under the archive's `tmp_research/` directory.
- Saved chat text and memory are included. The snapshot cutoff is
  `2026-10-22T02:34:26Z` (October 21, 10:34 p.m. EDT). Credentials and live editor
  or tool state were intentionally excluded.

## Research recovered

The main sources are the saved-chat [index](../../work_context/readable_chats/INDEX.md),
the [main project review](../../work_context/readable_chats/codex-785bfea6ecb4.md),
the [nonlinear decoder conversation](../../work_context/readable_chats/codex-4d9e40eb9e9d.md),
the [migration conversation](../../work_context/readable_chats/codex-0697c079d7b3.md),
and Claude notes under
`../work_context/claude_projects/-home-exouser-GeoPretrain/memory/`.
Those memory notes mostly stop October 17; later result artifacts supersede old
lists of unfinished jobs.

### GeoAE: nonlinear decoders on frozen LLM activations

The current [Qwen comparison](../runs/nonlinear_decoder_validation/deltaai/review/core_comparison.csv)
uses Qwen2.5-3B block 35, latent dimension 4096, K=2000, and 50 epochs. Both
nonlinear arms finished all 18 evaluations. The linear baseline was reused.
All arms retain token bypass: `z = E(h - b_t)`, `h_hat = D(z) + b_t`.

| Decoder | CH ↑ | Reconstruction FVU ↓ | Splice perplexity ↓ | Control selectivity gain, pp ↑ |
| --- | ---: | ---: | ---: | ---: |
| Linear | 664.05 | 0.029106 | 8.66617 | 6.6439 |
| MLP | 1047.87 | 0.031684 | 8.48182 | 4.0269 |
| Residual MLP | 710.70 | 0.027692 | 8.55323 | 3.9354 |

Full MLP leads on measured geometry, matched topic alignment and perplexity;
residual MLP has the lowest reconstruction error. At steering alpha=0.5,
linear has higher selectivity gain in six of seven dataset/draw rows; MLP slightly
leads on 20 Newsgroups. Control document pools differ, so these comparisons are
descriptive. One seed, differing capacity/bias/constraints, and a row-based
train/validation split limit conclusions. MLP's MMLU gain is not significant.

The checked presentation, detailed report, unpooled workbook and source CSVs
remain under `reports/nonlinear_decoder_20261021/`.

The saved other-model status had recovery jobs 3428745/3428746/3428747 pending.
A fresh read of runtime JSONs on Delta during setup showed:

- Gemma residual job **3428747** has completed GPU preflight and training and is
  running evaluations. Its runtime state supersedes the saved pending snapshot.
- Llama residual's available evaluation state still has 15/16 steps complete,
  with `control_atlas_s5` and its dependent summary missing. Its recovery
  allocation state file is absent.
- Gemma MLP's recovery allocation/evaluation state files are absent. Confirm its
  scheduler state before taking further action.

These are filesystem observations, not a fresh scheduler accounting report.
The Delta login's accounting database lists `delta` and `linux`; a DeltaAI
cluster name was not established there. No replacement jobs were submitted.
Runtime snapshots and timestamps are retained in the evidence JSON.

### GeoPretrain: GPT-2 feed-forward geometry regularization

The twelve single-layer runs, layer-5/layer-7 hinge follow-ups, and all five
matched SFT pilots completed in the preserved reports. They are single-seed
pilots. Layer 5 without separation improves language prediction but harms
agreement; layer 7 is a contextual-selectivity lead with language-loss costs.
Hinge repairs some layer-5 agreement failures but introduces other regressions.
SFT improves stopping and looping while worsening language/grammar retention;
it does not establish a useful assistant or a consistent GeoFF advantage.

Read the latest reports in `../GeoPretrain/analyses/layerwise/`,
`layerwise/hinge_followup/`, and `posttraining/` before quoting detailed numbers.
Next research decisions are repeatability/shared-state branching, independent
seeds, and separating pull/push effects while holding covariance fixed. Some
geometry/history buffers are absent from checkpoints, which limits exact resume
and branch reproducibility. A representation-use learning curve remains missing.

## Remaining work and boundaries

There are no remaining blockers to the supported GeoAE laptop setup or shelving
handoff. Development follow-ups remain:

1. Reconcile research directories with preserved Git histories before normal
   branch-based development. The research roots have no `.git`. Keep both copies:
   `../code/GeoAE` is at `1121151`; `../code/GeoPretrain` is at `0cad079` with
   preserved uncommitted changes. They differ materially from the research roots.
2. Refresh DeltaAI scheduler status and recover/collect Llama and Gemma outputs;
   inspect frozen manifests before changing anything.
3. Plan matched-document and multi-seed comparisons before replacing the linear
   baseline or interpreting geometry gains as general semantic improvement.

Keep large data and GPU work on the clusters. Argo's authoritative pretraining
runs remain `/scratch/dogar/geoae/pretrain/runs/`, with evaluations under
`/scratch/dogar/geoae/pretrain/evalresults/`. Delta pretraining assets remain
`/work/hdd/bimc/mhaider/GeoAE/pretrain/`. Current nonlinear Qwen work is under
`/projects/bbyl/mhaider/GeoAE/nonlinear_qwen_20261018_deltaai/`; recovery work is
under `nonlinear_recovery_20261021_deltaai/`. The migration `home/` is an archive,
not a new experiment working directory. Preserve existing immutable source
snapshots, hashes, manifests and absolute cluster paths.

Labels remain evaluation-only; activation-space geometry uses Euclidean
distances. Argo campaign GPU work uses H100s, leaving L40s free. Keep model/dump
computation off login nodes. Overleaf remains the paper working copy.

The final cloud action is **Actions → Shelve**, then verify **Shelved** and
**0.00 SUs/hour**. Shelving preserves an image for later unshelving. Shelved
instances cost zero SUs; stopped instances still cost 50% of their usual SUs.
See the current [Jetstream SU policy](https://docs.jetstream-cloud.org/general/access/)
and [shelving instructions](https://docs.jetstream-cloud.org/getting-started/instance-management/).
