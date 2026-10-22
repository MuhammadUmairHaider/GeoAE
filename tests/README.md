# Verification

The default suite uses CPU tensors, temporary files and stubs; it does not download models or datasets, contact judges or submit jobs.

- `unit/`: model and loss contracts, algorithms, checkpoint loading, extraction and evaluation calculations.
- `workflows/`: training/resume behavior, command dispatch, campaign plans, source snapshots, staging and recovery.
- `integration/`: import side effects and compatibility against the preserved external paper checkout. The latter uses `../code/GeoAE` by default, or `GEOAE_PAPER_REPO`; it skips explicitly if that optional reference is unavailable.

From the project root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest
bash scripts/setup_laptop.sh --smoke-only
```

The [test relocation ledger](../docs/history/audits/test_paths.json) maps previous names. The pinned matched-SAE implementation is tested through maintained adapters and remains byte-identical to its recorded source.
