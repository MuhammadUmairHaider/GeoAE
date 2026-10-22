# Mac home base and Delta research archive

Use the Mac for editing, CPU checks, reviewing results, and connecting directly to Delta and Argo. Keep activation extraction, model loading, and the large training/evaluation jobs on the appropriate cluster.

The local workspace is `~/work` (`$HOME/work`). After extraction, GeoAE is under `~/work/GeoAE` and the saved conversations are under `~/work/work_context`.

## Migration locations

The migration destination on Delta is:

```text
/projects/bbyl/mhaider/jetstream_migration_20261021/
  home/                                  Selected Jetstream research files
  control/selected-files.txt              Preservation file list
  control/verification.json               Verification of that selected set
  control/skipped-artifacts.jsonl         Omitted files and canonical locations/reasons
  handoff/geoae-macos-handoff.tar.gz       Lightweight Mac workspace
  packed/geoae-llm-judge-cache.tar.gz      Separately packed paid judge cache
  packed/geoae-llm-judge-cache.tar.gz.sha256
```

This is selective preservation of nonredundant Jetstream work. Existing Argo runs and datasets remain authoritative on Argo. Existing Delta artifacts are referenced at their current paths instead of copied again. Regenerable caches and proven duplicate copies are recorded in `control/skipped-artifacts.jsonl`.

The laptop bundle contains lightweight code, reports, and selected research results, including `GeoAE/`, `GeoPretrain/`, and `code/`. It excludes heavy weights and activation dumps. `GeoAE/` includes `runs/nonlinear_decoder_validation/deltaai/review/core_comparison.csv`. `code/GeoAE/` and `code/GeoPretrain/` preserve the original working checkouts and their Git histories. They can differ from the research directories; keep both copies until those differences have been reviewed.

The bundle also contains `work_context/` with saved conversations and project memory notes. These are included in the same download as the code.

The selected archive retains unique Jetstream pretraining runs, `GeoPretrain/analyses/`, and lightweight `diag/` and `perf/` evidence. The Jetstream runs include `gpt2s_base_mb32_s1337`, `gpt2s_base_mb32_s2`, and `gpt2s_geo0p1_k2000nosep_s2`; duplicate diagnostic copies of their weights need not be preserved twice. Local reports, calibration records, evaluation fixtures, and analysis outputs remain useful even when their underlying training checkpoints stay on Argo.

`home/` may also contain completed files from the preliminary broad copy. Their presence does not mean every original home file was copied or verified. `control/verification.json` covers `control/selected-files.txt`; use that scope and the skipped-artifact inventory when assessing completion.

## Connect directly from the Mac

Add these entries to `~/.ssh/config` using a text editor. Use the authentication and MFA required for each account; set an `IdentityFile` only if you already have an authorized Mac key.

```sshconfig
Host delta
    HostName login.delta.ncsa.illinois.edu
    User mhaider
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host argo
    HostName argo.ace-net.ca
    User dogar
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then test access from the Mac terminal:

```bash
ssh delta 'hostname; test -d /projects/bbyl/mhaider/jetstream_migration_20261021/home'
ssh argo 'hostname; test -d /scratch/dogar/geoae/pretrain/runs'
```

Both connections go directly from the Mac to the cluster and do not require Jetstream to remain running. Confirm each host fingerprint through the cluster's usual access instructions when first connecting.

## Download, verify, and extract

Use an empty destination so existing local files are not overwritten:

```bash
mkdir -p "$HOME/work"
cd "$HOME/work"
scp delta:/projects/bbyl/mhaider/jetstream_migration_20261021/handoff/geoae-macos-handoff.tar.gz .
ssh delta 'sha256sum /projects/bbyl/mhaider/jetstream_migration_20261021/handoff/geoae-macos-handoff.tar.gz'
shasum -a 256 geoae-macos-handoff.tar.gz
```

The two printed SHA-256 values must match. Check the archive contents and extract only after they match:

```bash
tar -tzf geoae-macos-handoff.tar.gz | head -30
tar -xzf geoae-macos-handoff.tar.gz
cd GeoAE
test -f runs/nonlinear_decoder_validation/deltaai/review/core_comparison.csv
```

Keep the downloaded tar as a local recovery copy. Reconfigure credentials on the Mac separately; Linux virtual environments and authentication files do not belong in the laptop setup.

## Saved chats and project memory

From the extracted workspace parent, open `work_context/readable_chats/INDEX.md` to browse saved Codex and Claude conversations. The index links to readable Markdown transcripts and separates the main Codex conversations from helper sessions. This migration conversation is included through the snapshot cutoff recorded in `work_context/CONTEXT_SNAPSHOT.json`.

`work_context/codex_sessions/` and `work_context/claude_projects/` retain filtered JSONL chat records with visible user/assistant text. Claude project memory notes remain under each project's `memory/` directory. Private reasoning, tool logs, account settings, and authentication files are excluded. These are portable reference files; the package does not install chat history into an application.

When continuing work on the Mac, give the assistant `GeoAE/docs/LAPTOP_HANDOFF.md`, the relevant transcript from the index, and any applicable project memory notes. Keep historical cluster paths in the transcripts as provenance; use the direct Mac-to-cluster connections above for new work.

## Optional paid judge-cache download

Existing paid LLM judge responses are preserved separately as a tar with about 111,000 cache entries, avoiding that many loose files on the Delta archive. They are not model weights and do not need to be regenerated with new paid API calls.

Download and extract this optional cache from the **workspace parent**, the directory containing `GeoAE/`:

```bash
cd "$HOME/work"
scp delta:/projects/bbyl/mhaider/jetstream_migration_20261021/packed/geoae-llm-judge-cache.tar.gz .
scp delta:/projects/bbyl/mhaider/jetstream_migration_20261021/packed/geoae-llm-judge-cache.tar.gz.sha256 .
cat geoae-llm-judge-cache.tar.gz.sha256
shasum -a 256 geoae-llm-judge-cache.tar.gz
```

Compare the two SHA-256 values. After they match:

```bash
tar -tzf geoae-llm-judge-cache.tar.gz | head -10
tar -xzf geoae-llm-judge-cache.tar.gz
cd GeoAE
```

The archive contains `GeoAE/cache/llm_judge/`, so extracting inside `GeoAE/` would create an extra nested `GeoAE/` directory. Leave this cache packed on Delta unless loose files are needed for a particular run. Cached responses can be reused with matching evaluator inputs; this step does not configure credentials or start any API work.

## Native Mac environment

The checked-in `uv.lock` contains torch 2.13 wheels for Apple Silicon with macOS 14 or newer. It contains no Intel macOS torch wheel. On Apple Silicon, run a native terminal: `uname -m` should print `arm64`. A Rosetta terminal reports `x86_64` and will be refused by the setup script.

Install `uv` if needed, for example with an existing Homebrew installation:

```bash
brew install uv
bash scripts/remote/setup_laptop.sh
```

The setup creates a local `.venv` with Python 3.12 from the existing lock and runs small CPU tests. The initial installation needs internet access for Python and packages. It does not download models, corpora, activation dumps, or benchmark datasets, and it does not contact API judges.

If an incompatible `.venv` already exists, move it aside before setup; do not copy the Jetstream `.venv`. The script leaves `uv.lock` unchanged.

Once Python and package wheels are cached, an installation without network access is available:

```bash
bash scripts/remote/setup_laptop.sh --offline
```

This fails if a required interpreter, wheel, or build dependency is missing from the local cache. To rerun only the installed environment's CPU checks, with model/dataset downloads disabled:

```bash
bash scripts/remote/setup_laptop.sh --smoke-only
```

On an Intel Mac, or an older macOS release, use the Mac for source editing and remote work first. With Python 3.10+ installed, this dependency-free syntax check is available:

```bash
bash scripts/remote/setup_laptop.sh --source-only
```

A full Intel environment would require separately chosen compatible dependencies. Do not rewrite the research lock simply to make installation pass. DeltaAI's paper training preflight requires both ARM64 and CUDA; Apple Silicon alone does not satisfy it. General GeoAE training selects CUDA or CPU and does not automatically select Apple's MPS backend.

## Everyday work

Open the extracted `GeoAE/` in the local IDE. Use its `.venv/bin/python` as the interpreter on supported Macs. Run the smoke checks after code changes, and inspect the copied comparison CSVs and reports locally.

The `code/` checkouts preserve Git history but may predate current research changes. Inspect them before deciding how to reconcile the current research files into a development branch:

```bash
git -C ../code/GeoAE status --short
git -C ../code/GeoAE log -1 --oneline
```

For new remote experiments, first upload code into a separate candidate directory. The following example copies source and tests without bulk artifacts or local environments:

```bash
ssh delta 'mkdir -p /projects/bbyl/mhaider/GeoAE_laptop_candidate'
rsync -av --exclude '__pycache__/' \
  geoae configs scripts tests pyproject.toml uv.lock README.md \
  delta:/projects/bbyl/mhaider/GeoAE_laptop_candidate/
```

Connect to Delta to inspect that candidate, configure its environment and existing data paths, and prepare a new run using the appropriate existing Delta or DeltaAI launcher. The upload does not submit a job. The standard Delta setup is documented in `RUNBOOK_DELTA_ROUND2.md`; DeltaAI launchers are under `scripts/deltaai/` and have their own runtime and allocation requirements. Use direct `ssh argo` for Argo workflows; `GeoPretrain/scripts/env_argo.sh` and the analysis submission manifests retain the cluster-specific paths.

Existing prepared experiments contain absolute paths, source hashes, and file-identity checks. Keep their source snapshots, configs, and manifests intact. Do not point those manifests at the Mac or replace a running campaign's source with a new laptop checkout. The migration `home/` directory is an archive; current remote jobs continue to use their established run directories.

Retrieve newly generated summaries from the actual remote run directory with `scp` or `rsync`, and save them beside the corresponding local results. Argo's existing pretraining runs remain under `/scratch/dogar/geoae/pretrain/runs/`; Delta's existing pretraining assets remain under `/work/hdd/bimc/mhaider/GeoAE/pretrain/`. Refer to `control/skipped-artifacts.jsonl` for omitted artifact locations. Fetch individual files when needed instead of bringing all activation dumps onto the laptop.

`GeoPretrain/` is included for continuity. Its dependencies and workflows are separate; `setup_laptop.sh` installs GeoAE only.

## Finish retiring Jetstream

Check the selected archive's status, verification report, and inventory from the Mac:

```bash
ssh delta 'cat /projects/bbyl/mhaider/jetstream_migration_20261021/control/status.json; cat /projects/bbyl/mhaider/jetstream_migration_20261021/control/verification.json'
ssh delta 'wc -l /projects/bbyl/mhaider/jetstream_migration_20261021/control/selected-files.txt /projects/bbyl/mhaider/jetstream_migration_20261021/control/skipped-artifacts.jsonl'
```

Require a successful verification report for the selected preservation set. A missing report means that verification is incomplete. Review the skipped-artifact inventory to distinguish existing cluster originals from regenerable caches; a successful selected-file check is not a claim that the entire Jetstream home was mirrored. Confirm the separate packed judge-cache checksum as well. Global account credentials, runtime editor/tool state, and downloadable global caches are outside this research handoff; saved chat text and project memory are included in `work_context/`.

After the selected Delta archive is verified, the Mac bundle's checksum matches, the comparison file opens locally, and direct Mac access to both clusters works, use **Actions → Shelve** in Jetstream. Shelved instances consume zero SUs; stopped instances still consume 50% of their normal SUs. See the [Jetstream SU policy](https://docs.jetstream-cloud.org/general/access/) and [shelving instructions](https://docs.jetstream-cloud.org/getting-started/instance-management/). Confirm that shelving completes and the UI shows a zero burn rate.

Shelving preserves the instance's disk contents for later unshelving. Keep the instance until the migration checks are complete. This handoff does not shelve or delete the VM.
