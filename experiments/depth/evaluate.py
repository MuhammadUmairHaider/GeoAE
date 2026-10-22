#!/usr/bin/env python3
"""Detailed, model-isolated evaluations of Llama, Gemma and Qwen block AEs.

Reuse the metric tools and resumable research runner from experiments.gpt2.evaluate.
``--dry-run`` prints the complete recipe without loading tensors or writing files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gpt2 import evaluate as research  # noqa: E402


_MIDDLE_PROFILES = {
    "llama": {
        "family": "llama3.2-3B", "model_name": "meta-llama/Llama-3.2-3B",
        "layer": 14, "hidden_size": 3072, "latent_dim": 6144, "n_layers": 28,
        "n_clusters": 2000, "n_epochs": 50,
        "ckdir": "k2000_bnh_b32k_lam1_d6144_dpc_sampled_struct5_tokbias",
        "activations_dir": "activations_llama3_3b_l14_sampled_struct5_10M",
        "corpus_dir": "corpus/sampled_v1",
        "train_config": "experiments/depth/configs/llama3.2-3b_l14_k2000_bnh_b32k_lam1_d6144_dpc_sampled_struct5_tokbias.yaml",
        "extract_config": "experiments/depth/configs/llama3.2-3b_l14_extract_sampled_struct5.yaml",
    },
    "gemma4b": {
        "family": "gemma3-4B", "model_name": "google/gemma-3-4b-pt",
        "layer": 17, "hidden_size": 2560, "latent_dim": 5120, "n_layers": 34,
        "n_clusters": 2000, "n_epochs": 50,
        "ckdir": "k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias",
        "activations_dir": "activations_gemma3_4b_l17_sampled_struct5_10M",
        "corpus_dir": "corpus/gemma3_4b_sampled_struct5_v1",
        "train_config": "experiments/depth/configs/gemma3-4b_l17_k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias.yaml",
        "extract_config": "experiments/depth/configs/gemma3-4b_l17_extract_sampled_struct5.yaml",
    },
}


def _variant_profile(family: str, position: str, mode: str) -> dict:
    profile = deepcopy(_MIDDLE_PROFILES[family])
    if position == "last":
        old_layer = profile["layer"]
        profile["layer"] = profile["n_layers"] - 1
        profile["train_config"] = profile["train_config"].replace(
            f"_l{old_layer}_", f"_l{profile['layer']}_").replace("_struct5", "_struct5_train10M")
        profile["extract_config"] = profile["extract_config"].replace(
            f"_l{old_layer}", f"_l{profile['layer']}").replace(".yaml", "_train10M.yaml")
        profile["activations_dir"] = ("activations_sampled_struct5_10M" if family == "llama"
                                       else "activations_gemma3_4b_sampled_struct5_10M")
    if mode == "plain":
        profile["ckdir"] = profile["ckdir"].removesuffix("_tokbias")
        profile["train_config"] = profile["train_config"].replace("_tokbias.yaml", ".yaml")
    profile.update(mode=mode, variant=mode, layer_role=position,
                   arm=f"{family}_{position}_{mode}")
    return profile


CANONICAL_ARMS = tuple(f"{family}_{position}_{mode}"
                       for family in _MIDDLE_PROFILES for position in ("mid", "last")
                       for mode in ("bypass", "plain"))
# Six requested new trainings. Final-block bypass references are evaluated
# separately only after their source provenance has been verified.
NEW_ARMS = tuple(arm for arm in CANONICAL_ARMS if not arm.endswith("last_bypass"))
PROFILES = {arm: _variant_profile(*arm.split("_")) for arm in CANONICAL_ARMS}
for _alias in _MIDDLE_PROFILES:
    # Preserve the original middle-block spellings and their artifact paths.
    PROFILES[_alias] = deepcopy(PROFILES[f"{_alias}_mid_bypass"])
    PROFILES[_alias].pop("variant")

for _profile in PROFILES.values():
    _profile["n_train_rows"] = 10_000_000
    _profile["n_tokens"] = 10_526_316
    _profile["val_frac"] = 0.05
    _profile["activations_dir"] = _profile["activations_dir"].replace("_10M", "_train10M")
    _profile["ckdir"] = _profile["ckdir"].replace("_struct5", "_struct5_train10M")
    _profile["token_bias_filename"] = "token_bias_sampled_struct5_train10M.npz"
    _profile["baseline_suffix"] = "sampled_struct5_train10M"

# This final-block plain recipe is also the paper's Gemma no-bypass arm.
PROFILES["gemma4b_last_plain"]["train_config"] = "experiments/paper/configs/gemma_no_bypass.yaml"

# Qwen is an independent approved pair. Keep the existing all/new campaign
# selections stable so its pending jobs cannot enter the Llama/Gemma judges.
QWEN_ARMS = ("qwen_last_bypass", "qwen_last_plain")
for _arm in QWEN_ARMS:
    _mode = _arm.rsplit("_", 1)[1]
    PROFILES[_arm] = {
        "family": "qwen2.5-3B", "model_name": "Qwen/Qwen2.5-3B",
        "layer": 35, "hidden_size": 2048, "latent_dim": 4096, "n_layers": 36,
        "n_clusters": 2000, "n_epochs": 50,
        "ckdir": "k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M"
                 + ("_tokbias" if _mode == "bypass" else ""),
        "activations_dir": "activations_qwen2_5_3b_l35_struct5_train10M",
        "corpus_dir": "corpus/qwen2_5_3b_struct5_v1",
        "train_config": f"experiments/paper/configs/qwen{'' if _mode == 'bypass' else '_no_bypass'}.yaml",
        "extract_config": "experiments/paper/configs/qwen2.5-3b/l35_extract_struct5_train10M.yaml",
        "mode": _mode, "variant": _mode, "layer_role": "last", "arm": _arm,
        "n_train_rows": 10_000_000, "n_tokens": 10_526_316, "val_frac": 0.05,
        "token_bias_filename": "token_bias_struct5_train10M.npz",
        "baseline_suffix": "struct5_train10M",
    }


def options_for(arm: str, args) -> argparse.Namespace:
    """Return a standalone runner namespace; callers need no ML imports."""
    values = vars(args).copy()
    values["profile"] = deepcopy(PROFILES[arm])
    for name in ("activations_dir", "corpus_dir"):
        values[name] = values.get(name) or PROFILES[arm][name]
    return argparse.Namespace(**values)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", nargs="?", choices=(*PROFILES, "all", "new"), default="all",
                   help="One variant, all eight comparisons, or the six new training arms")
    p.add_argument("--steps", default="whole", help="prep, evals, interventions, steering, concept, geometry, "
                   "probes, interp, api, gpu, whole; or comma-separated individual stages")
    p.add_argument("--run-name", default="midlayer_run", help="Campaign directory under results/")
    p.add_argument("--activations-dir", default=None, help="Override one arm's sampled activation dump")
    p.add_argument("--corpus-dir", default=None, help="Override one arm's frozen sampled corpus")
    p.add_argument("--checkpoint", default=None, help="Override one arm's checkpoint; final-checkpoint guard still applies")
    p.add_argument("--dry-run", action="store_true", default=bool(os.environ.get("DRY")))
    p.add_argument("--force", action="store_true", default=bool(os.environ.get("FORCE")))
    p.add_argument("--judges", choices=("auto", "skip", "required"), default="auto")
    p.add_argument("--judge-model", default="google/gemini-2.5-flash-lite")
    p.add_argument("--agreement-judges", default="google/gemini-2.5-flash-lite,openai/gpt-4o-mini,deepseek/deepseek-chat-v3.1")
    p.add_argument("--pair-model", default=os.environ.get("PAIR_MODEL", "anthropic/claude-sonnet-5.5"))
    p.add_argument("--judge-concurrency", type=int, default=int(os.environ.get("JUDGE_CONCURRENCY", "6")))
    p.add_argument("--pair-pairs", type=int, default=int(os.environ.get("PAIR_PAIRS", "100")))
    p.add_argument("--pair-controls", type=int, default=int(os.environ.get("PAIR_CONTROLS", "25")))
    p.add_argument("--print-code-files", action="store_true", help="Print the transitive evaluation code transfer list")
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        p.error("--run-name must be one directory name")
    if args.arm in ("all", "new") and any((args.activations_dir, args.corpus_dir, args.checkpoint)):
        p.error("Path overrides require one specific arm")
    if args.judge_concurrency < 1 or args.pair_pairs < 1 or args.pair_controls < 0:
        p.error("Judge concurrency and pairs must be positive; controls must be nonnegative")
    os.chdir(ROOT)
    arms = list(CANONICAL_ARMS if args.arm == "all" else NEW_ARMS if args.arm == "new" else [args.arm])
    try:
        if args.print_code_files:
            files = set()
            for arm in arms:
                options = options_for(arm, args)
                layer = options.profile["layer"]
                plan = research.build_steps(layer, Path("<checkpoint>"), Path(options.activations_dir), options)
                files.update(research.code_files(plan, options.profile))
            print(json.dumps(sorted(files), indent=2))
            return 0
        results = [research.run_layer(PROFILES[arm]["layer"], options_for(arm, args)) for arm in arms]
        return 0 if all(results) else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
