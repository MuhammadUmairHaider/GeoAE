"""Entry point for the matched nonlinear-decoder workflow.

Prepare a source snapshot, prepare matched training/evaluation manifests, inspect
preflights, then run or submit explicitly. This file dispatches to the actual
implementations below; it never starts a second stage automatically.

Examples (use a cluster for model/data checks and GPU work)::

    python -m experiments.nonlinear_decoders.plan prepare source --run-dir /path/to/new-run
    python -m experiments.nonlinear_decoders.plan prepare training --help
    python -m experiments.nonlinear_decoders.plan prepare evaluation --help
    python -m experiments.nonlinear_decoders.plan preflight cpu --run-dir /path/to/run
    python -m experiments.nonlinear_decoders.plan run evaluation --help
    python -m experiments.nonlinear_decoders.plan submit campaign --campaign /path/to/campaign

Submission defaults to scheduler validation. Only an explicit --submit submits;
campaign jobs retain their held/release workflow. Existing frozen runs continue
to use the launchers and manifests preserved inside their original directories.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = Path(__file__).resolve().parent / "deltaai"
ROUTES = {
    ("prepare", "training"): ("nonlinear_paper_training", "prepare"),
    ("prepare", "evaluation"): ("nonlinear_paper_evals", "prepare"),
    ("prepare", "campaign"): ("nonlinear_paper_campaign", "prepare"),
    ("prepare", "qwen-evaluation"): ("qwen_paper_evals", "prepare"),
    ("preflight", "cpu"): ("nonlinear_paper_evals", "preflight"),
    ("preflight", "gpu"): ("nonlinear_paper_evals", "gpu-preflight"),
    ("run", "training"): ("nonlinear_paper_training", "run"),
    ("run", "evaluation"): ("nonlinear_paper_evals", "run"),
    ("run", "campaign"): ("nonlinear_paper_campaign", "run"),
    ("run", "qwen-evaluation"): ("qwen_paper_bundle", "run"),
    ("submit", "campaign"): ("nonlinear_paper_campaign", "submit"),
    ("submit", "qwen-evaluation"): ("qwen_paper_bundle", "submit"),
    ("validate", "training"): ("nonlinear_paper_training", "validate"),
    ("validate", "evaluation"): ("nonlinear_paper_evals", "validate"),
    ("validate", "campaign"): ("nonlinear_paper_campaign", "verify"),
}


def prepare_source(arguments: list[str]) -> None:
    from geoae.workflow.snapshot import atomic_json, copy_evaluation_sources

    parser = argparse.ArgumentParser(description="Freeze source and recipes only; no data, model or job loading.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(arguments)
    run = args.run_dir.resolve()
    source = run / "source"
    if source.exists():
        raise ValueError(f"Source snapshot already exists: {source}; use a new run directory")
    source.mkdir(parents=True)
    hashes = copy_evaluation_sources(ROOT, source)
    atomic_json(run / "source_manifest.json", {"source_root": str(source), "source_sha256": hashes})
    print(f"Prepared source snapshot: {source}. No jobs submitted.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "submit", "validate"))
    parser.add_argument("target", help="source, training, evaluation, campaign, qwen-evaluation, cpu, or gpu")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected implementation")
    args = parser.parse_args(argv)
    if (args.action, args.target) == ("prepare", "source"):
        prepare_source(args.arguments)
        return
    route = ROUTES.get((args.action, args.target))
    if route is None:
        parser.error(f"Unsupported action/target: {args.action} {args.target}")
    # The cluster collectors also run directly as Python 3.6 scripts, so their
    # sibling imports remain usable outside the installed Python package.
    sys.path.insert(0, str(RUNNERS))
    module = importlib.import_module(route[0])
    previous = sys.argv
    try:
        sys.argv = [str(RUNNERS / (route[0] + ".py")), route[1], *args.arguments]
        module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
