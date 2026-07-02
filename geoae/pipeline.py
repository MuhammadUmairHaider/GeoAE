"""
End-to-end GeoAE pipeline orchestrator (MSE pipeline only).

Phases:
  extract   — extract residual stream activations from a frozen LLM
  train     — train the concept-cluster autoencoder (MSE faithfulness)
  evaluate  — causal splice evaluation + per-cluster ablation

The e2e KL pipeline is driven directly via its own CLIs (`geoae-e2e-extract`,
`geoae-e2e-train`, `geoae-e2e-train-stream`) — see docs/training.md.

Usage (from the GeoAE package root)::

    python -m geoae.pipeline extract --n_tokens 100000
    python -m geoae.pipeline train --layer 27 --n_clusters 128
    python -m geoae.pipeline eval --layer 27 --n_clusters 128
    python -m geoae.pipeline all --layer 27 --no_wandb
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from geoae.paths import BASE_CONFIGS_DIR, PACKAGE_ROOT


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a command, streaming output. Exits on non-zero status."""
    print(f"\n[pipeline] $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(cwd or PACKAGE_ROOT))
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_id_of(args: argparse.Namespace) -> str:
    return f"layer{args.layer}_k{args.n_clusters}_d{args.latent_dim}"


def phase_extract(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "geoae.extract", "--config", args.config]
    if args.n_tokens:
        cmd += ["--n_tokens", str(args.n_tokens)]
    run(cmd)


def phase_train(args: argparse.Namespace) -> None:
    layer_config = BASE_CONFIGS_DIR / f"layer_{args.layer}.yaml"
    if not layer_config.exists():
        layer_config = BASE_CONFIGS_DIR / "full_run_v2.yaml"
    run_id = run_id_of(args)
    cmd = [
        sys.executable, "-m", "geoae.train",
        "--config", str(layer_config),
        "--n_clusters", str(args.n_clusters),
        "--latent_dim", str(args.latent_dim),
        "--checkpoints_dir", f"checkpoints/{run_id}",
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")
    run(cmd)


def phase_eval(args: argparse.Namespace) -> None:
    run_id = run_id_of(args)
    ckpt = PACKAGE_ROOT / "checkpoints" / run_id / "best_val.pt"
    if not ckpt.exists():
        print(f"[pipeline] ERROR: checkpoint not found at {ckpt}")
        print("           Run the train phase first.")
        sys.exit(1)
    cmd = [
        sys.executable, "-m", "geoae.evaluate",
        "--checkpoint", str(ckpt),
        "--layer", str(args.layer),
        "--experiment", "all",
        "--n_eval", str(args.n_eval),
        "--out", f"results_{run_id}.json",
    ]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoAE end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "phase", choices=["extract", "train", "eval", "all"],
        help="Pipeline phase to run",
    )
    parser.add_argument("--layer", type=int, default=27, help="Target layer")
    parser.add_argument("--n_clusters", type=int, default=128, help="Number of clusters K")
    parser.add_argument("--latent_dim", type=int, default=2048, help="Latent dimension L")
    parser.add_argument("--n_tokens", type=int, default=None, help="(extract) token budget")
    parser.add_argument("--n_eval", type=int, default=500, help="(eval) number of texts")
    parser.add_argument(
        "--config",
        default=str(BASE_CONFIGS_DIR / "default.yaml"),
        help="(extract) YAML config path",
    )
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    if args.phase in ("extract", "all"):
        print("\n" + "=" * 60)
        print("  PHASE 1: Activation Extraction")
        print("=" * 60)
        phase_extract(args)

    if args.phase in ("train", "all"):
        print("\n" + "=" * 60)
        print(f"  PHASE 2: Training  (layer={args.layer}, K={args.n_clusters})")
        print("=" * 60)
        phase_train(args)

    if args.phase in ("eval", "all"):
        print("\n" + "=" * 60)
        print(f"  PHASE 3: Causal Evaluation  (layer={args.layer})")
        print("=" * 60)
        phase_eval(args)

    print(f"\n[pipeline] Done. Run id: {run_id_of(args)}")


if __name__ == "__main__":
    main()
