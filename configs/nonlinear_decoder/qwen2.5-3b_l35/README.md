Qwen2.5-3B final-block decoder experiments
=======================================

Two new runs reuse the completed 50-epoch linear-decoder checkpoint as their
baseline. Both train from scratch with seed 42 on the existing block-35 activation
dump. The encoder remains Linear(2048, 4096), BatchNorm, GELU, with the same token
bypass, clustering objectives, and training schedule as `reference_bypass.yaml`.

| New run | Decoder |
| --- | --- |
| `mlp` | Linear(4096, 4096), GELU, Linear(4096, 2048); both biases enabled |
| `residual_mlp` | Original bias-free Linear(4096, 2048) plus Linear(4096, 512), GELU, bias-free Linear(512, 2048) |

The residual output projection starts at zero. Its skip decoder retains the
existing column renormalization. The MLP decoder and residual nonlinear branch
are not column-renormalized. The MLP therefore differs from the baseline in
parameter count, biases, and normalization as well as nonlinearity; only these
two requested runs are prepared.

The shared dump has 10,526,316 FP16 rows: 10,000,000 train rows and 526,316
validation rows, width 2048, sampled throughout 2048-token contexts with 5%
structured prompts. These recipes preserve the reference row split. One document
straddles its boundary (16 train positions, 48 validation positions), so this is
not a strict document holdout. Each run has batch size 32,768 and 50 epochs:
305 updates per epoch, 15,250 updates total.

Argo preparation expects a reviewed source snapshot at:

```text
/project/def-hsajjad/dogar/GeoAE/nonlinear_qwen_20261018/source
```

Stage source code into that directory without credentials, caches, activation
dumps, or checkpoints. The preparation tool references existing data and the
baseline in `/project/def-hsajjad/dogar/GeoAE/repo`; it does not copy the dataset or
modify the old checkout. Run from the source snapshot:

```bash
/scratch/dogar/venvs/geoae/bin/python scripts/argo/prepare_nonlinear_qwen.py
/scratch/dogar/venvs/geoae/bin/python scripts/argo/validate_nonlinear_qwen.py --build-model
/scratch/dogar/venvs/geoae/bin/python scripts/argo/submit_nonlinear_qwen.py
```

The last command calls `sbatch --test-only` for the two arms and allocates no
resources. `--print-only` only prints the commands. A separate invocation with
`--submit` explicitly submits the two training jobs. Default resources per job:
one L40, 12 CPUs, 160 GB RAM, and three days on `c-hsajjad` under `def-hsajjad`.
`--gpu h100` selects an H100 instead; the architecture recipes remain unchanged.

Preparation writes absolute runtime configs, per-arm output directories, and an
immutable source/config/dataset manifest. The layer array has a clearly labeled
head/tail sample fingerprint; smaller sidecars, normalization, token table,
baseline checkpoint, and source files have complete SHA256 hashes. Every launch
verifies them again, checks CUDA, stages only the required activation files in
`/dev/shm`, verifies that copy, and records runtime provenance. It refuses to
overwrite an arm's checkpoint outputs and prevents concurrent jobs for that arm.
The verified Argo interpreter is fixed to `/scratch/dogar/venvs/geoae/bin/python`;
`PYTHONPATH` and an import-location check select the snapshot over its old
editable installation. No LM, Hugging Face download, `.env`, or API key is needed.

DeltaAI paper evaluations
-------------------------

The DeltaAI training jobs have a separate evaluation snapshot based on
`/projects/bbyl/mhaider/GeoAE/docs/paper/main.tex` and that checkout's paper
evaluators. The prepared suite is at
`/projects/bbyl/mhaider/GeoAE/nonlinear_qwen_20261018_deltaai/evaluation/paper_suite`.
It uses the final epoch-50 checkpoint for each arm and the existing linear
checkpoint, with one GPU allocation per arm running four successive stages.
The current queued jobs (`3410592`, `3410593`) use `bimc-dtai-gh` with a
24-hour limit, 12 CPUs, and 160G RAM each; training used `bbyl-dtai-gh`.
Their account, QoS, and time limits were updated in place on October 19, 2026.
The original 48-hour submission receipts remain as an audit record; the live
resource changes are recorded in the suite's `resource_update.json` and the
local mirrored evaluation records.

1. One-million-state geometry, reconstruction/token diagnostics, MMLU (2,000
   questions), and WikiText-2 perplexity.
2. All nine paper alignment tasks, paired cluster-support filtering, and the
   three FineWeb-Atlas draws with document-bootstrap intervals.
3. DBpedia-14, Bias in Bios, AG News, 20 Newsgroups, and the three Atlas draws:
   fresh jointly correct documents, per-concept accuracy/confidence, and
   pooled concept-bootstrap summaries at steering strengths 0.4 and 0.5.
4. A Qwen extension of the range-structure appendix: pair counts and
   target/sibling/random loss and KL under range and whole-direction edits.

The evaluation source preserves the nonlinear checkpoint loader and exact
state-conditioned decoder changes. Its isolated ARM environment, copied paper
caches, frozen task definitions, source hashes, and complete command plans are
recorded in its manifest. Training source and recipes remain in their original
snapshot. Per-step completion records validate outputs and their hashes before
skipping a completed evaluation. A failed metric does not prevent independent
later stages from being attempted. The consolidated launcher reuses one
node-local activation copy across all four stages and records the result of each
stage. The original evaluated source and command plans remain frozen.

Run the submitter with the explicit prepared directory; without `--submit` it
only checks Slurm requests. Recorded submissions are never duplicated:

```bash
python scripts/deltaai/qwen_paper_bundle.py submit --eval-dir \
  /projects/bbyl/mhaider/GeoAE/nonlinear_qwen_20261018_deltaai/evaluation/paper_suite
```

Local job records and validation evidence are under
`runs/nonlinear_decoder_validation/deltaai/evaluation/`; the parent
`status.json` records the training and evaluation job IDs together.
