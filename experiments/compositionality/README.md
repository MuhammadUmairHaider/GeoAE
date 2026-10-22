# Compositionality

SugarCrepe controls ask whether representation geometry and recovered model behavior distinguish compositional image-caption alternatives. Shared benchmark/evaluation primitives remain in `geoae/`; this directory owns the comparison recipes.

`analysis/` contains the saved-result comparison, geometry and rescoring programs. `delta/` contains the explicit cluster launch recipes. Read each program's arguments before execution; geometry/rescoring can require model or activation inputs on the cluster.

Saved results live in the sibling artifact store indexed by [docs/artifacts.json](../../docs/artifacts.json). Use a new output directory when rerunning analysis, and keep historical input/checkpoint paths intact.
