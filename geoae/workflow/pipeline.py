"""
End-to-end GeoAE pipeline orchestrator (MSE pipeline only).

Phases:
  extract   — extract residual stream activations from a frozen LLM
  train     — train the concept-cluster autoencoder (MSE faithfulness)
  evaluate  — causal splice evaluation + per-cluster ablation

The e2e KL pipeline is driven directly via its own CLIs (`geoae-e2e-extract`,
`geoae-e2e-train`, `geoae-e2e-train-stream`) — see docs/training.md.

Usage (from the GeoAE package root)::

    python -m geoae.workflow.pipeline extract --n_tokens 100000
    python -m geoae.workflow.pipeline train --layer 27 --n_clusters 128
    python -m geoae.workflow.pipeline eval --layer 27 --n_clusters 128
    python -m geoae.workflow.pipeline all --layer 27 --no_wandb
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from geoae.config import Config
from geoae.paths import BASE_CONFIGS_DIR, PACKAGE_ROOT, resolve_input_path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a command, streaming output. Exits on non-zero status."""
    print(f"\n[pipeline] $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(cwd or PACKAGE_ROOT))
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_id_of(args: argparse.Namespace) -> str:
    return f"layer{args.layer}_k{args.n_clusters}_d{args.latent_dim}"


def phase_extract(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "geoae.extraction.prefix", "--config", args.config,
           "--seed", str(args.seed)]
    run(cmd)


def phase_train(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "geoae.training.mse",
        "--config", args.config,
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")
    run(cmd)


def phase_eval(args: argparse.Namespace) -> None:
    run_id = run_id_of(args)
    ckpt = resolve_input_path(args.checkpoints_dir) / "best_val.pt"
    if not ckpt.exists():
        print(f"[pipeline] ERROR: checkpoint not found at {ckpt}")
        print("           Run the train phase first.")
        sys.exit(1)
    cmd = [
        sys.executable, "-m", "geoae.evaluation.faithfulness",
        "--checkpoint", str(ckpt),
        "--layer", str(args.layer),
        "--experiment", "all",
        "--n_eval", str(args.n_eval),
        "--seed", str(args.seed),
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
    parser.add_argument("--layer", type=int, default=None, help="Override target/extracted layer")
    parser.add_argument("--n_clusters", type=int, default=None, help="Override number of clusters K")
    parser.add_argument("--latent_dim", type=int, default=None, help="Override latent dimension L")
    parser.add_argument("--n_tokens", type=int, default=None, help="(extract) token budget")
    parser.add_argument("--n_eval", type=int, default=500, help="(eval) number of texts")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML recipe shared by extraction, training, and checkpoint selection",
    )
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    supplied_config = args.config is not None
    config_path = Path(args.config).resolve() if supplied_config else BASE_CONFIGS_DIR / "default.yaml"
    cfg = Config.from_yaml(config_path)
    if args.layer is not None:
        cfg.data.target_layer = cfg.extraction.target_layer = args.layer
        cfg.extraction.layers = [args.layer]
    if args.n_clusters is not None:
        cfg.model.n_clusters = args.n_clusters
    if args.latent_dim is not None:
        cfg.model.latent_dim = args.latent_dim
    if args.n_tokens is not None:
        cfg.extraction.n_tokens = args.n_tokens
    cfg.validate()
    args.layer = cfg.data.target_layer
    args.n_clusters = cfg.model.n_clusters
    args.latent_dim = cfg.model.latent_dim
    args.seed = cfg.train.seed
    if not supplied_config:
        cfg.train.checkpoints_dir = f"checkpoints/{run_id_of(args)}"
    args.checkpoints_dir = cfg.train.checkpoints_dir

    # CLI overrides reach every subprocess through one recipe. The checkpoint
    # retains its effective config; the user's source YAML is never rewritten.
    with TemporaryDirectory(prefix="geoae-pipeline-") as temporary:
        effective = Path(temporary) / "config.yaml"
        effective.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        args.config = str(effective)
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
