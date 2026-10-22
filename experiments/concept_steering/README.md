# Concept steering

This family owns campaign-specific generation, qualitative review and judge recipes. Shared interventions, generation and scoring live in `geoae/evaluation` and `geoae/analysis`; the [command catalog](../../docs/usage.md) exposes them through `geoae evaluate` and `geoae analyze`.

- `analysis/`: saved qualitative reports and explicit judge item/scoring tools.
- `scripts/`: eraser/movie control campaign.
- `argo/` and `remote/`: model- and resource-specific launch recipes.

Inspect a recipe before executing it: generation loads models and judge scoring may call a paid API. Saved inputs and reports are located through [the artifact catalog](../../docs/artifacts.json); frozen output caches and remote paths retain their original meaning.
