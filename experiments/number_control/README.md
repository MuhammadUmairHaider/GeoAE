# Number control

The range/number benchmark asks whether continuous latent edits shift numeric predictions as intended. Shared implementation lives in `geoae/evaluation/number_control.py`; inspect its interface with `geoae evaluate number-control --help`.

`scripts/run_number_control.sh` preserves the explicit cluster recipe. It loads model/checkpoint inputs and must run on the cluster. Saved result inputs live in the store indexed by [the artifact catalog](../../docs/artifacts.json); use a new output directory for another run. The [method note](../../docs/methods/number_control.md) records the evaluation design.
