#!/usr/bin/env python3
"""Write an artifact index and measured results for matched plain and bypass AEs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.depth.evaluate import PROFILES, CANONICAL_ARMS, NEW_ARMS, options_for, parser as evaluation_parser
from experiments.gpt2.evaluate import build_steps, profile_paths, write_state


def read_json(path: Path):
    return json.loads(path.read_text())


def scalar_metrics(value, prefix="", limit=200) -> dict:
    """Index scalar summaries; per-item lists remain in their original artifacts."""
    metrics = {}

    def visit(item, name):
        if len(metrics) >= limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if key not in {"per_text", "per_item", "per_subject", "preds", "concepts", "commands"}:
                    visit(child, f"{name}.{key}" if name else key)
        elif isinstance(item, (int, float, bool)) or item is None:
            metrics[name] = item

    visit(value, prefix)
    return metrics


def make_report(arm: str, run_name: str) -> tuple[Path, dict, str]:
    options = options_for(arm, evaluation_parser().parse_args([arm, "--run-name", run_name]))
    layer = PROFILES[arm]["layer"]
    paths = profile_paths(layer, options)
    out = paths["out"]
    manifest = out / "pipeline_state.json"
    state = read_json(manifest) if manifest.exists() else {}
    checkpoint = Path(state.get("checkpoint", "<checkpoint>"))
    plan = build_steps(layer, checkpoint, Path(options.activations_dir), options)
    report = {"arm": arm, "profile": PROFILES[arm], "status": state.get("status", "not_run"),
              "manifest": str(manifest), "checkpoint": str(checkpoint), "stages": {}, "artifacts": {}}
    mode = PROFILES[arm].get("mode", "bypass")
    label = "token-bypass AE" if mode == "bypass" else "plain AE"
    report["ae_mode"] = mode
    lines = [f"# {PROFILES[arm]['model_name']} layer {layer}: {label}", "",
             f"Evaluation status: **{report['status']}**. Checkpoint: `{checkpoint}`.", "",
             f"Base and {label} are the primary comparison. The token-mean control measures the contribution "
             "of subtracting token identity before encoding.", "",
             "| Evaluation | Status | Detailed artifacts |", "|---|---|---|"]
    selected = set(state.get("requested", []))
    for step in plan:
        if step.name in ("pair_export", "pair_import") and step.name not in selected:
            continue
        entry = state.get("steps", {}).get(step.name, {})
        status = entry.get("status", "not_run")
        links = []
        artifacts = []
        all_outputs = [*step.outputs, *getattr(step, "optional_outputs", [])]
        for path in all_outputs:
            if not path.is_file():
                continue
            artifacts.append(str(path))
            try:
                display = str(path.relative_to(out))
                href = display
            except ValueError:
                display, href = str(path), str(path.resolve())
            links.append(f"[{display}]({href})")
            if path.suffix == ".json":
                try:
                    contents = read_json(path)
                    report["artifacts"][str(path)] = {"metrics": scalar_metrics(contents),
                                                       "scalar_limit": 200, "detail_file": str(path)}
                except (OSError, ValueError) as exc:
                    report["artifacts"][str(path)] = {"error": str(exc)}
                    status = "invalid_artifact"
        if status == "complete" and any(not path.is_file() for path in step.outputs):
            status = "incomplete_artifact"
        if status in ("invalid_artifact", "incomplete_artifact"):
            report["status"] = "failed"
        report["stages"][step.name] = {"status": status, "reason": entry.get("reason"), "artifacts": artifacts,
                                     "optional_missing": entry.get("optional_missing", [])}
        reason = f" — {entry['reason']}" if entry.get("reason") else ""
        if entry.get("optional_missing"):
            reason += " — requested optional figures unavailable (see report.json)"
        lines.append(f"| {step.name} | {status}{reason} | {', '.join(links) or 'Pending'} |")
    headlines = []
    mmlu = out / "evals/mmlu.json"
    if mmlu.is_file() and report["stages"]["mmlu"]["status"] == "complete":
        try:
            meta = read_json(mmlu)["meta"]
            reconstruction_label = "bypass" if mode == "bypass" else "plain AE"
            headlines.append(f"MMLU accuracy: base {100 * meta['base_acc']:.2f}% → {reconstruction_label} reconstruction "
                             f"{100 * meta['recon_acc']:.2f}% ({100 * meta['delta']:+.2f} percentage points, "
                             f"{meta['n']} questions).")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    core = out / "evals/core_eval.json"
    if core.is_file() and report["stages"]["core_eval"]["status"] == "complete":
        try:
            recovered = read_json(core)["loss_recovered"]["mean_loss_recovered"]
            headlines.append(f"Mean language-model loss recovered by reconstruction: {100 * recovered:.2f}%.")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if headlines:
        lines.extend(["", "## Measured results", "", "\n\n".join(headlines)])
    else:
        lines.extend(["", "No measured headline results are available yet."])
    lines.extend(["", "The JSON artifacts retain full per-concept, per-subject, per-strength and paired results. "
                  "`report.json` indexes scalar summaries; lists and examples remain in the linked files.", "",
                  "The dump-row transfer metric (`csteer`) requires the final block and is recorded as unavailable "
                  "at middle layers; it is included at the final block. Generation uses the full model; middle-layer cluster signatures replay "
                  "real document contexts through the remaining blocks. Predicted-next-token cluster purity "
                  "also requires the final block, so token-structure evaluation omits it at middle layers.", ""])
    lines[2] = f"Evaluation status: **{report['status']}**. Checkpoint: `{checkpoint}`."
    return out, report, "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=(*PROFILES, "all", "new"), default="all")
    p.add_argument("--run-name", default="midlayer_run")
    args = p.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        p.error("--run-name must be one directory name")
    try:
        os.chdir(ROOT)
        for arm in CANONICAL_ARMS if args.arm == "all" else NEW_ARMS if args.arm == "new" else [args.arm]:
            out, data, markdown = make_report(arm, args.run_name)
            out.mkdir(parents=True, exist_ok=True)
            data["written_at"] = datetime.now(timezone.utc).isoformat()
            write_state(out / "report.json", data)
            temp = out / "report.md.tmp"
            temp.write_text(markdown)
            temp.replace(out / "report.md")
            print(f"[report] {out / 'report.md'} ({data['status']})")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
