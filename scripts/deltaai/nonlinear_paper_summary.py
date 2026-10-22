"""Paired control summaries, including Gemma's jointly evaluated SAE controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qwen_paper_summary import (ALPHAS, DATASETS, GROUP_NAMES, _aggregate, _atomic_text,
                                _bootstrap_interval, _concept, _confidence, _vector)


def write_control_summary(root: Path, *, sae: bool = False, datasets=None) -> dict:
    spaces = ("h", "z", "s", "s1") if sae else ("h", "z")
    suffix = "hzss1_tp3" if sae else "hz_tp2"
    concepts, sources = [], {}
    for dataset in DATASETS if datasets is None else datasets:
        path = root / "interventions" / f"range_intervention_{dataset}_dprime_{suffix}.json"
        data = json.loads(path.read_text())
        if not data.get("summary") or not data.get("concepts"):
            raise ValueError(f"Incomplete control result: {path}")
        if sae and not data["meta"].get("tp_all"):
            raise ValueError(f"Gemma SAE controls must share three-way true positives: {path}")
        sources[dataset] = {"path": str(path), "n_concepts": len(data["concepts"]),
                            "n_eval_by_class": data["meta"].get("n_eval_by_class", {})}
        for cid, record in data["concepts"].items():
            alias = "atlas_s0" if dataset.startswith("atlas_") else dataset
            concept = _concept(alias, str(cid), record, data["meta"])
            concept["dataset"] = dataset
            nt, nc = concept["target_documents"], concept["complement_documents"]
            for space in spaces[2:]:
                reference = record["ref"][space]
                correct = _vector(reference, "doc_correct", nt + nc, f"{dataset} {cid} {space}", correctness=True)
                tc, cc = _confidence(reference, nt, nc, f"{dataset} {cid} {space}")
                concept["baseline"][space] = {
                    "target_accuracy": 100 * float(correct[:nt].mean()),
                    "complement_accuracy": 100 * float(correct[nt:].mean()),
                    "target_confidence": 100 * tc, "complement_confidence": 100 * cc}
                for alpha in ALPHAS:
                    edited = record[f"{space}_st_global_a{alpha}"]
                    ec = _vector(edited, "doc_correct", nt + nc, f"{dataset} {cid} {space}", correctness=True)
                    etc, ecc = _confidence(edited, nt, nc, f"{dataset} {cid} {space}")
                    td = 100 * float((correct[:nt] - ec[:nt]).mean())
                    cd = 100 * float((correct[nt:] - ec[nt:]).mean())
                    concept["alphas"][alpha][space] = {
                        "target_drop": td, "collateral": cd, "selectivity": td - cd,
                        "target_accuracy": 100 * float(ec[:nt].mean()),
                        "complement_accuracy": 100 * float(ec[nt:].mean()),
                        "target_confidence": 100 * etc, "complement_confidence": 100 * ecc,
                        "target_confidence_drop": 100 * (tc - etc),
                        "collateral_confidence_drop": 100 * (cc - ecc),
                        "confidence_selectivity": 100 * (tc - etc - cc + ecc)}
            concepts.append(concept)
    groups = []
    for name in list(dict.fromkeys(GROUP_NAMES.values())) + ["All concepts"]:
        subset = [c for c in concepts if name == "All concepts" or c["dataset_group"] == name]
        group = _aggregate(name, subset)
        if not subset:
            raise ValueError(f"No concepts scored for {name}")
        for space in spaces[2:]:
            group["baseline"][space] = {key: float(np.mean([c["baseline"][space][key] for c in subset]))
                                         for key in subset[0]["baseline"][space]}
            for alpha in ALPHAS:
                group["alphas"][alpha][space] = {
                    key: float(np.mean([c["alphas"][alpha][space][key] for c in subset]))
                    for key in subset[0]["alphas"][alpha][space]}
        for alpha in ALPHAS:
            comparisons = {}
            for left, right in [(space, "h") for space in spaces[1:]] + ([("z", "s"), ("z", "s1")] if sae else []):
                differences = np.asarray([c["alphas"][alpha][left]["selectivity"] -
                                          c["alphas"][alpha][right]["selectivity"] for c in subset])
                comparisons[f"{left}_minus_{right}"] = {
                    "mean": float(differences.mean()),
                    "ci95": _bootstrap_interval(differences, [c["dataset_group"] for c in subset]),
                    "better": int((differences > 1e-9).sum()), "worse": int((differences < -1e-9).sum()),
                    "ties": int((np.abs(differences) <= 1e-9).sum())}
            group["alphas"][alpha]["comparisons"] = comparisons
        groups.append(group)
    result = {"status": "complete", "meta": {
        "units": "accuracy and confidence percentages; drops and differences in percentage points",
        "spaces": list(spaces), "bootstrap": "10,000 concept pairs; pooled row stratified by dataset; seed 0",
        "true_positive_filter": "base, nonlinear AE, and SAE" if sae else "base and nonlinear AE",
        "confidence": "correct label's first token probability", "alphas": list(ALPHAS)},
        "sources": sources, "groups": groups, "per_concept": concepts}
    _atomic_text(root / "control_summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = ["# Control comparisons", "", "Drops and selectivity differences are percentage points; means weight concepts equally.", ""]
    for alpha in ALPHAS:
        rows += [f"## Alpha {alpha}", "", "| Dataset | Concepts | Space | Target drop | Collateral | Selectivity gain vs base |",
                 "|---|---:|---|---:|---:|---:|"]
        for group in groups:
            for space in spaces[1:]:
                values = group["alphas"][alpha]
                rows.append(f"| {group['dataset']} | {group['n_concepts']} | {space} | "
                            f"{values[space]['target_drop']:.2f} | {values[space]['collateral']:.2f} | "
                            f"{values['comparisons'][space + '_minus_h']['mean']:+.2f} |")
        rows.append("")
    _atomic_text(root / "control_summary.md", "\n".join(rows))
    return result
