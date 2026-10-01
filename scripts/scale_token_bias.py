"""Scale an existing fixed token table without refitting its means or normalization."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def scale_table(source: Path, out: Path, alpha: float) -> None:
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    with np.load(source, allow_pickle=True) as src:
        data = {key: src[key] for key in src.files}
    if float(data.get("bias_scale", 1.0)) != 1.0:
        raise ValueError("Use the original unscaled table as source")
    table = data["table"]
    scaled = (table.astype(np.float32) * alpha).astype(table.dtype)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled table contains nonfinite entries")
    data["table"] = scaled
    data["bias_scale"] = np.asarray(alpha)
    data["source_token_bias"] = np.asarray(str(source))
    # Variance-removed diagnostics were measured on the original table, not this one.
    if "report" in data:
        data["source_report"] = data.pop("report")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents concurrent preparers from overwriting a table.
    with out.open("xb") as f:
        np.savez(f, **data)
    print(f"[token-bias-scale] alpha={alpha:g}: {source} -> {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    scale_table(args.source, args.out, args.alpha)


if __name__ == "__main__":
    main()
