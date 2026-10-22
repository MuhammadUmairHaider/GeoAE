#!/bin/bash
# GPU smoke test for the Argo setup (1 H100, <= 20 min):
#   sbatch scripts/slurm/argo_smoke.sbatch
# Checks: driver/CUDA, torch + bf16 matmul, `import geoae`, compute-node internet, cold `import torch` timing of
# the /scratch vs /home venv (if both exist), Llama-3.2-3B offline forward pass, and the pytest suite
# (counts from a junit xml, because the pytest summary line is sometimes suppressed).
#SBATCH --job-name=geoae-smoke
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
now() { date +%s.%N; }
echo "== host =="; hostname -f; date
echo "== nvidia-smi =="; nvidia-smi | head -4; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== cold import torch timing (this node has not touched these venvs before) =="
for V in /scratch/dogar/venvs/geoae /home/dogar/venvs/geoae; do
  [ -x "$V/bin/python" ] || continue
  for i in 1 2; do s=$(now); "$V/bin/python" -c "import torch"; e=$(now); echo "import torch  $V  run$i: $(echo "$e - $s" | bc) s"; done
done

echo "== torch / cuda / bf16 matmul / import geoae =="
"$PY" - <<'PYEOF'
import time, torch, transformers, numpy
print("python", __import__("sys").version.split()[0], "| torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| transformers", transformers.__version__, "| numpy", numpy.__version__)
print("cuda available:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0), "| n visible:", torch.cuda.device_count())
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
c = a @ b; torch.cuda.synchronize(); t = time.time()
for _ in range(20): c = a @ b
torch.cuda.synchronize(); dt = (time.time() - t) / 20
print(f"bf16 matmul 8192^3: ok={bool(torch.isfinite(c.float()).all())}, {2 * 8192**3 / dt / 1e12:.0f} TFLOP/s")
import geoae; print("import geoae ok:", geoae.__file__)
PYEOF
echo "rc import/matmul: $?"

echo "== compute-node internet =="
curl -sI --max-time 10 https://huggingface.co | head -1 || echo "no internet (curl rc=$?)"
echo "curl rc: ${PIPESTATUS[0]}"
env | grep -i -E '^(https?_proxy|no_proxy)=' | sed 's/=.*@/=<cred>@/'

echo "== Llama-3.2-3B offline forward pass =="
HF_HUB_OFFLINE=1 "$PY" - <<'PYEOF'
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
t = time.time()
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
m = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype=torch.bfloat16).cuda().eval()
print(f"loaded in {time.time() - t:.0f}s; params {sum(p.numel() for p in m.parameters()) / 1e9:.2f} B")
x = tok("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = m(**x, output_hidden_states=True)
nxt = tok.decode(out.logits[0, -1].argmax().item())
print("n hidden states:", len(out.hidden_states), "| layer-27 hidden:", tuple(out.hidden_states[27].shape), "| next token:", repr(nxt))
PYEOF
echo "rc llama: $?"

echo "== pytest =="
mkdir -p "$ARGO_SCRATCH/geoae/logs"
JX="$ARGO_SCRATCH/geoae/logs/pytest-smoke-${SLURM_JOB_ID:-local}.xml"
timeout 900 "$PY" -m pytest -q -p no:cacheprovider -rf --junitxml="$JX" 2>&1 | tail -40
echo "pytest exit code: ${PIPESTATUS[0]}"
"$PY" - "$JX" <<'PYEOF'
import sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot(); s = r if r.tag == "testsuite" else r.find("testsuite")
print({k: s.get(k) for k in ("tests", "failures", "errors", "skipped", "time")})
for tc in s.iter("testcase"):
    if tc.find("failure") is not None or tc.find("error") is not None:
        print("FAILED:", tc.get("classname"), tc.get("name"))
PYEOF
echo "== done =="; date
