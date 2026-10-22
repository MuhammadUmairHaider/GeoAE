#!/bin/bash
# Rebuild plain_kmeans_grid.json from the log (the run finished but could not write: output dir missing), then probe + token structure.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=.
D=e2e/checkpoints/general/gemma3-4B/layer33/delta
ACT=activations_gemma3_4b_sampled_struct5_10M
E=results/delta/g4b_l33_d5120/evals
mkdir -p $E
.venv/bin/python - <<'PY'
import ast, json, re
log = open("logs/g4b_grid.log").read().replace("\r", "\n")
res = {"meta": {"note": "rebuilt from logs/g4b_grid.log: the run completed but its output directory was missing; fits saved as npz",
                "activations_dir": "activations_gemma3_4b_sampled_struct5_10M", "layer": 33, "K": 2000, "stride": 6, "max_iter": 100,
                "geometry_sample": "1,000,000 rows, RandomState(0) over all rows (same draw as cq.json)"},
       "fits": {}, "heldout_usage": {}, "geometry": {}}
for m in re.finditer(r"plain (residual|tokmean): (\d+) iterations, last changed ([0-9.]+) -> (\S+)", log):
    res["fits"][m.group(1)] = {"iterations": int(m.group(2)), "last_changed": float(m.group(3)), "npz": m.group(4)}
for m in re.finditer(r"held-out usage: (\{.*\})", log):
    res["heldout_usage"].update(ast.literal_eval(m.group(1)))
for m in re.finditer(r"geometry (\w+): (.*?) \(\d+s\)", log):
    res["geometry"][m.group(1)] = {k: float(v) for k, v in (kv.split(" ") for kv in m.group(2).split(", "))}
assert len(res["fits"]) == 2 and len(res["heldout_usage"]) == 4 and len(res["geometry"]) == 4, res
json.dump(res, open("results/delta/g4b_l33_d5120/evals/plain_kmeans_grid.json", "w"), indent=1)
print("rebuilt plain_kmeans_grid.json", {k: round(v["calinski_harabasz"], 1) for k, v in res["geometry"].items()})
PY
ARMS="base=$D/balanced_kmeans_k2000_dpc_sampled_struct5.npz,base_tokmean=$D/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz,plain=$D/plain_kmeans_k2000_dpc_sampled_struct5.npz,plain_tokmean=$D/plain_kmeans_k2000_dpc_sampled_struct5_tokmean.npz"
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache/gemma3-4b/l33 --baselines "$ARMS" \
  --atlas_last "" --out $E/probe_plain_kmeans_grid.json
.venv/bin/python -u geoae/evaluation/token_structure.py --arms "$ARMS" --activations_dir $ACT --layer 33 \
  --out $E/token_structure_plain_kmeans_grid.json
echo G4B_REST_DONE
