# BiasBios evaluation audit

The existing dual-probe benchmark in `geoae/bias/probe.py` required corrections
before interpreting its raw-versus-AE comparison. This is a **fixed linear probe
ablation** experiment, not an intervention in the language model. Lower accuracy
of the frozen gender probe does not establish that gender information is erased;
that would require, at minimum, training a fresh probe on the modified features.

## Findings and corrections

- The raw and AE spaces previously drew different validation splits because they
  consumed the global RNG sequentially. Both now share saved source row indices.
- Raw features were standardized by default; AE features were not. Both now use
  per-coordinate mean/std fitted only on probe-training rows. Ablating to zero
  consistently means replacing a coordinate with its training mean.
- Original standardization included the held-out validation rows. It no longer does.
- The original binary chance threshold used ordinary accuracy despite label
  imbalance. The train cache has 27,104 label-0 and 22,896 label-1 biographies;
  the test cache has 10,773 and 9,227. Gender operating points now use balanced
  accuracy. A constant prediction has balanced accuracy 0.5 even on imbalanced data.
- The original half-drop threshold was baseline accuracy divided by two, often
  below chance. It now halves the excess balanced accuracy above 0.5.
- Unreached thresholds previously silently became `k = all dimensions` and the
  efficiency score assumed chance had been reached. Unreached points now remain
  null; the misleading efficiency ratio is removed.
- Previously the test curve selected k and the same test rows measured retained
  profession accuracy. Budgets are now selected on validation curves, then applied
  unchanged to the test set. The complete test curves are descriptive.
- Softmax weight magnitude depends on an arbitrary common weight row that cannot
  affect predictions. Probe rankings now use class-centered weight contrasts.
  A train-only d-prime ranking and five random rankings are also evaluated.
- The old default L1 coefficient (0.1 times a sum of all weights) could dominate
  probe fitting, especially the 28-class probe. New defaults use no L1 and light
  AdamW decay (1e-4); optional L1 uses a mean. Validation CE chooses the checkpoint,
  and convergence histories, best epochs and training/validation metrics are saved.
  These defaults are explicit choices, not a claim of optimal hyperparameter tuning.
- Profession outputs always use the 28-class label space, even when a small
  smoke sample omits a class. The previous unique-label count could underallocate
  outputs for noncontiguous labels.
- Checkpoint model/layer compatibility and finite values are validated. Outputs
  refuse overwrites and are saved after each completed representation.

## Cache audit

The cached 50,000 training and 20,000 test rows match the source dataset labels
after reconstructing extraction's seed-42 shuffle. All activations are finite.
There are 49,977 unique training texts and 19,994 unique test texts; 13 text hashes
occur in both sets. The new evaluation retains the first training occurrence,
removes training-overlapping test texts, and deduplicates test texts. This leaves
49,977 training-pool rows and 19,981 test rows before optional subsampling.
The validation split is drawn from this deduplicated training pool.

`geoae/bias/audit_cache.py` reproduces the audit against local source Arrow files.
For this workspace the source directory is:

```
/home/exouser/.cache/huggingface/datasets/LabHC___bias_in_bios/default/0.0.0/052f01de644dba841176e0449528b41f27d94a61
```

The checked report and row hashes are `eval_out/biasbios_cache_audit.json` and
`eval_out/biasbios_source_hashes.npz`. Hashes identify exact stripped texts; they
do not guarantee that different biographies describe different people, or that
different texts do not become identical after truncation/tokenization.

## Running

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest tests/test_bias_probe.py

BIOS_OUTPUT=results/biasbios_dpc_audited_smoke.json \
BIOS_PLOTS=figures/biasbios_dpc_audited_smoke \
  bash eval_out/run_biasbios_audited.sh --smoke

bash eval_out/run_biasbios_audited.sh
```

The launcher defaults to CPU to coexist with the currently active GPU jobs.
Set `BIOS_DEVICE=cuda` to use the GPU when desired. Raw features have 3,072
coordinates and the DPC AE has 6,144: equal fractions do not mean equal numbers
of coordinates or equal perturbation norms. Both count and fraction are saved.

Schema version 2 is deliberately different from the old report: each ranking
contains validation and test curves plus validation-selected operating points.
Profession accuracy, balanced accuracy, prediction changes, probe-distribution
KL, and per-profession gender recall gaps are reported. This is not a comprehensive
fairness evaluation. The original separate `linear_probe_compare` utility was
reviewed but not modified; it does not apply this new text deduplication.
