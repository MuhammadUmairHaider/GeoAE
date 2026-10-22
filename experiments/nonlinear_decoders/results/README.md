# Saved comparisons

[references.json](references.json) records the saved Qwen tables and their hashes. Paths are relative to the research artifact root in [the catalog](../../../docs/artifacts.json). These are recorded results, not a live scheduler query. Read [research.md](../../../docs/research.md) before interpreting them.

Rebuild tables and figures from saved JSON into a new output directory:

```bash
python -m experiments.nonlinear_decoders.analyze --output-dir /path/to/new-analysis
```

Use `--review-dir` to select another copy of the same review input layout. This reads saved results and does not start model evaluation or call a judge. The original builder remains with its frozen review inputs.
