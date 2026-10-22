# Runtime site profiles

Source `scripts/env.sh` from any working directory. It resolves the checkout,
selects Delta, Argo or local defaults, then sources the matching profile here.
Set `GEOAE_ROOT` and `GEOAE_SITE` to choose explicitly. Cluster cache, data,
allocation and interpreter defaults retain the previous storage paths.

Use `bash scripts/setup_laptop.sh` on the supported Mac. Use
`bash scripts/setup_cluster.sh --site delta` or `--site argo` on that cluster.
`scripts/py` selects the configured interpreter without changing dependencies.
