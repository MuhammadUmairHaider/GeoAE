#!/bin/bash
# Prove the environment works on a GPU NODE before committing hours to a campaign:
#
#   sbatch scripts/delta/verify.sbatch
#
# ~2 min, ~0.1 GPU-h. Checks, in order:
#   1. the GPUs Slurm actually gave us, and free space on node-local /tmp (the
#      training jobs stage a 61 GB dump there)
#   2. the toolchain versions and that CUDA is usable from the wheel
#   3. the full test suite
#   4. a real Llama-3.2-3B forward pass, loaded OFFLINE from $HF_HOME, reporting the
#      layer-27 residual scale — the number to compare against a dump's
#      validation.json when a dump looks wrong
#
# WHY A GPU NODE and not the login node: pytest is ~10x faster here (~40 s vs 6m32s
# on dt-login03, where it is thread-contended, not CPU-bound), and only here can the
# CUDA build and the gpu-bind actually be exercised. Never size anything from
# login-node timings.
#
# NOTE ON THE TEST COUNT: pyproject.toml already puts `-q` in addopts, so passing
# `-q` again makes it `-qq` and SUPPRESSES the "N passed" summary line. This runs
# without an extra -q so the count is printed.
#SBATCH --job-name=geoae-verify
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
source scripts/delta/env.sh
mkdir -p logs

echo "== node ================================================================="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "/tmp:"; df -h /tmp | tail -1
echo "data root ($DELTA_DATA):"; df -h "$DELTA_DATA" | tail -1

echo
echo "== toolchain ============================================================"
uv run --no-sync python - <<'PY'
import torch, transformers, numpy
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| transformers", transformers.__version__, "| numpy", numpy.__version__)
print("device_count", torch.cuda.device_count(),
      "| capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "no CUDA device visible"
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok, trace", float((x @ x.T).diagonal().sum()))
PY

echo
echo "== tests ================================================================"
uv run --no-sync python -m pytest tests/

echo
echo "== Llama-3.2-3B forward (offline, from \$HF_HOME) ========================"
HF_HUB_OFFLINE=1 uv run --no-sync python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from geoae.lm_arch import decoder_layers

name = "meta-llama/Llama-3.2-3B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
layers = decoder_layers(model)
print(f"decoder blocks found: {len(layers)}")

captured = {}
h = layers[27].register_forward_hook(lambda m, i, o: captured.setdefault(
    "h", (o[0] if isinstance(o, tuple) else o).detach().float()))
ids = tok("The capital of France is Paris, a city known for", return_tensors="pt").to("cuda")
with torch.no_grad():
    model(**ids)
h.remove()

a = captured["h"][0]
print(f"layer-27 residual: shape {tuple(a.shape)}  std {a.std():.3f}  "
      f"median token norm {a.norm(dim=-1).median():.2f}")
print("compare the median token norm against a dump's validation.json "
      "(the 2026-09 L27 dump reported 55.83)")
PY

echo
echo "== done ================================================================="
