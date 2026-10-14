"""
Honest cluster usage for the status panel: hard-assignment EMA counts from each run's
latest checkpoint (ckpt["model_state"]["ema_cluster_size"]), NOT the logged `dying`
field, which is Sinkhorn balancing telemetry (docs/notes/train-diag-metrics-misleading.md).

    scripts/delta/py scripts/delta/panel_usage.py <cache.json> <ckpt_dir> [<ckpt_dir> ...]

Results are cached by checkpoint path + mtime, so a refresh only loads new checkpoints.
"""
import json
import math
import sys
from pathlib import Path

import torch


def stats(ck: Path) -> dict:
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    c = sd.get("model_state", sd)["ema_cluster_size"].double()
    u = c / c.sum()
    K = len(u)
    nz = u[u > 0]
    perp = math.exp(-(nz * nz.log()).sum().item())
    s = u.sort(descending=True).values
    return dict(ckpt=ck.name, k=K, starved=int((u < 0.1 / K).sum()), zero=int((c == 0).sum()),
                perplexity=round(perp), perp_frac=round(perp / K, 3),
                top10=round(100 * s[:10].sum().item(), 1), top1=round(100 * s[0].item(), 2))


def main() -> None:
    cache_path = Path(sys.argv[1])
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for d in map(Path, sys.argv[2:]):
        cks = sorted(d.glob("step_*.pt"))
        if not cks:
            continue
        ck = cks[-1]
        key = f"{ck}:{ck.stat().st_mtime_ns}"
        if cache.get(str(d), {}).get("key") != key:
            cache[str(d)] = dict(key=key, **stats(ck))
    cache_path.write_text(json.dumps(cache, indent=1))


if __name__ == "__main__":
    main()
