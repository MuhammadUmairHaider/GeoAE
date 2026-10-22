"""Consolidate completed external-label experiments without changing cohorts."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATASETS = ["medmentions", "streusle", "py150", "measeval", "fewnerd"]


def number(value):
    return f"{value:.4f}" if value is not None else "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/delta/d6144/evals/dataset_neuronlens")
    args = ap.parse_args()
    root = Path(args.root)
    outputs = {name: json.loads((root / f"{name}.json").read_text()) for name in DATASETS}
    consolidated = {"method": "modified NeuronLens, matching common target damage across all three spaces",
                    "primary_rule": "case level = half minimum maximum target damage; first upward crossing; complete pair means",
                    "primary": {}, "fixed_0.25": {}, "qualifying_shared_direction_pairs": []}
    lines = ["# External-labelled NeuronLens results", "",
             "Same frozen Llama-3.2-3B layer-27 model and d6144 token-bypass AE for every dataset.",
             "Same labelled examples in base, token-centred base and AE; ranges and edits fitted on fit documents only.",
             "Lower sibling/random next-token ΔNLL means less collateral predictive damage at matched target ΔNLL.",
             "These are bounded descriptive tests, not full-corpus benchmarks or semantic-classifier erasure scores.", "",
             "## Primary matched-target comparison", "",
             "Each removal case is matched across all three spaces at half the smallest maximum target ΔNLL reached by the sweep.",
             "Interpolation uses the first upward crossing in actual strength order. Both orientations are averaged per complete pair.",
             "Mean target loss differs between datasets and can vary between cases within a dataset.", "",
             "| Dataset | Pairs | Mean target ΔNLL | Base sibling | AE sibling | Base random | AE random |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, output in outputs.items():
        primary = output["summary"]["all_three"]["matched"]
        fixed = output["summary"]["all_three"]["fixed"]["0.25"]
        consolidated["primary"][key] = {k: v for k, v in primary.items() if k != "records"}
        consolidated["fixed_0.25"][key] = {k: v for k, v in fixed.items() if k != "records"}
        values = primary["arms"]
        lines.append(f"| {output['meta']['name']} | {primary['n_complete_pairs']} | {number(primary['L_mean'])} | " +
                     " | ".join(number(values[arm][group]) for arm, group in
                                (("base", "sibling"), ("ae", "sibling"), ("base", "random"), ("ae", "random"))) + " |")
        for pair in output["geometry"]:
            for space, geometry in pair["spaces"].items():
                if geometry["fit_cosine"] >= 0.9 and geometry["eval"]["projected_dprime"] >= 3.0:
                    consolidated["qualifying_shared_direction_pairs"].append(
                        {"dataset": key, "pair": pair["pair"], "space": space, "geometry": geometry})
    lines += ["", "## Fixed target ΔNLL = 0.25 nats", "",
              "All selected cases reach this level in all three spaces; no pair exclusions.", "",
              "| Dataset | Base sibling | AE sibling | Base random | AE random |",
              "|---|---:|---:|---:|---:|"]
    for key, output in outputs.items():
        values = consolidated["fixed_0.25"][key]["arms"]
        assert consolidated["fixed_0.25"][key]["n_complete_cases"] == len(output["cases"])
        lines.append(f"| {output['meta']['name']} | " + " | ".join(
            number(values[a][g]) for a, g in (("base", "sibling"), ("ae", "sibling"),
                                            ("base", "random"), ("ae", "random"))) + " |")
    lines += ["", "## Token-centred control (primary matching)", "",
              "| Dataset | Target ΔNLL | Sibling ΔNLL | Random ΔNLL |", "|---|---:|---:|---:|"]
    for output in outputs.values():
        values = output["summary"]["all_three"]["matched"]["arms"]["token_centred"]
        lines.append(f"| {output['meta']['name']} | " + " | ".join(number(values[g]) for g in
                     ("target", "sibling", "random")) + " |")
    lines += ["", "## Pair-level primary results", "",
              "| Dataset | Label pair | Mean target | Base sibling | AE sibling | Base random | AE random |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for output in outputs.values():
        records = output["summary"]["all_three"]["matched"]["records"]
        for pair_id, pair in enumerate(output["meta"]["plan"]["pairs"]):
            orientations = [r for r in records if r["pair_id"] == pair_id]
            if len(orientations) != 2:
                continue
            label_names = output["meta"]["plan"].get("label_names", {})
            label_pair = " / ".join(label_names.get(label, label) for label in pair)
            values = [statistics.mean(r["level"] for r in orientations)] + [
                statistics.mean(r["values"][a][g] for r in orientations)
                for a, g in (("base", "sibling"), ("ae", "sibling"), ("base", "random"), ("ae", "random"))]
            lines.append(f"| {output['meta']['name']} | {label_pair} | " + " | ".join(map(number, values)) + " |")
    lines += ["", "## Scope and validation", "",
              "- 16 label pairs / 32 removal cases; all meet minimum labelled fit/eval support. No primary matching exclusions.",
              "- Native external semantic categories chosen before looking at representations. Py150 roles are derived syntax labels.",
              "- MedMentions and STREUSLE use official document splits; Py150 additionally separates repositories; MeasEval additionally separates source articles.",
              "- Few-NERD supplies sentence splits without source article IDs. No exact text is shared between fit and eval in any adapter.",
              "- Repeated words can occur in both splits; these are not lexical-disjoint or same-word-controlled sense tests.",
              "- At most 200 fit and 120 eval spans per label, at most 4 per document; 256-token causal context windows. Random controls are non-annotated same-corpus positions.",
              "- Completed-span anchors exclude tokenizer pieces extending beyond the annotation. Some gold roles can still require later evidence, especially MeasEval relations.",
              "- Method matches the earlier project variant: d-prime saliency, ±2σ gates, complement mean replacement; grid p=.003,.01,.03,.1,.3,.5.",
              "- AE edits add the decoded latent difference to the clean residual, preserving the no-edit baseline exactly.",
              "- Saved H is float32; row sidecars are hashed. Clean-head log-probability sanity, finite scores, matching/reachability, offsets and split overlap were checked.",
              "- Pair groups can share labels. Results are descriptive; no confidence intervals or significance claims are made.",
              f"- Shared-direction checks (fit cosine ≥0.9 and held-out projected d′ ≥3) found {len(consolidated['qualifying_shared_direction_pairs'])} qualifying label-pair/space combinations. These broad category tests do not confirm the earlier AE cluster range pairs.",
              "", "## Reproduction", "",
              "Run the adapters in geoae/extraction/datasets/med_streusle.py, dataset_code_quantity.py and dataset_fewnerd.py, retaining their provenance manifests.",
              "The per-dataset plan JSONs and hashes are embedded in each result. On the GPU box, run each dataset sequentially:", "",
              "```bash", "GEOAE_REMOTE=1 HF_HUB_OFFLINE=1 .venv/bin/python geoae/extraction/datasets/neuronlens.py \\",
              "  --name MedMentions --data data/neuronlens_datasets/medmentions.jsonl \\",
              "  --plan data/neuronlens_datasets/medmentions.plan.json \\",
              "  --out results/delta/d6144/evals/dataset_neuronlens/medmentions.json \\",
              "  --cache results/delta/d6144/evals/dataset_neuronlens/medmentions_rows.npz", "```", "",
              "Use the corresponding dataset name/path for the other four. Reuse verified activation caches for identical settings.",
              "Consolidate with `python geoae/extraction/datasets/neuronlens_report.py`.", ""]
    (root / "summary.json").write_text(json.dumps(consolidated, indent=2, allow_nan=False))
    (root / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:22]))
    print(f"Saved {root / 'summary.md'}")


if __name__ == "__main__":
    main()
