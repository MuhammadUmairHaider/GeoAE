"""Qwen adapter for the shared, model-generic paper command plan.

The historical Qwen API takes an asset mapping. All commands and validation
now come from the same implementation used for Llama and Gemma.
"""
from pathlib import Path

from nonlinear_paper_plan import (
    CONTROL_DATASETS as CONTROL_DATASETS,
    PAPER_RUNG_NAMES as PAPER_RUNG_NAMES,
    REQUIRED_ASSETS as REQUIRED_ASSETS,
    build_steps as _build_steps,
)


def build_steps(assets: dict, checkpoint: Path, output: Path, python: str) -> list[dict]:
    return _build_steps({"assets": assets, "target_layer": 35, "model_name": "Qwen/Qwen2.5-3B"},
                        checkpoint, output, python)
