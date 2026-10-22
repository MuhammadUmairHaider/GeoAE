"""Strict, paired paper control summaries from completed per-document outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


DATASETS = ("db14", "biasbios", "ag_news", "news20", "atlas_s0", "atlas_s1", "atlas_s2")
GROUP_NAMES = {
    "db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News",
    "news20": "20 Newsgroups", "atlas_s0": "FineWeb-Atlas",
    "atlas_s1": "FineWeb-Atlas", "atlas_s2": "FineWeb-Atlas",
}
ALPHAS = ("0.4", "0.5")
N_BOOTSTRAP = 10_000


def _vector(record: dict, key: str, size: int, context: str, *, correctness: bool = False) -> np.ndarray:
    if key not in record:
        raise ValueError(f"{context}: missing {key}")
    value = np.asarray(record[key], dtype=float)
    if value.ndim != 1 or len(value) != size or not np.isfinite(value).all():
        raise ValueError(f"{context}: {key} must contain {size} finite values")
    if correctness and not np.isin(value, [0, 1]).all():
        raise ValueError(f"{context}: {key} must contain binary correctness values")
    if not correctness and ((value < 0) | (value > 1)).any():
        raise ValueError(f"{context}: {key} must contain probabilities in [0, 1]")
    return value


def _confidence(record: dict, nt: int, nc: int, context: str) -> tuple[float, float]:
    if "doc_conf" in record:
        values = _vector(record, "doc_conf", nt + nc, context)
        return float(values[:nt].mean()), float(values[nt:].mean())
    # Earlier valid artifacts carry only these means.  Accuracy always comes
    # from per-document correctness, never from rounded aggregate drop fields.
    try:
        values = np.asarray([record["tgt_conf"], record["comp_conf"]], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: missing usable confidence records") from exc
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{context}: invalid confidence means")
    return float(values[0]), float(values[1])


def _bootstrap_interval(differences: np.ndarray, strata: list[str]) -> list[float | None]:
    """Canonical concept-paired bootstrap, stratified in the pooled paper row."""
    d = np.asarray(differences, dtype=float)
    labels = np.asarray(strata)
    if len(d) < 2:
        return [None, None]
    rng = np.random.default_rng(0)
    parts = []
    for label in np.unique(labels):
        values = d[labels == label]
        parts.append(values[rng.integers(0, len(values), (N_BOOTSTRAP, len(values)))])
    values = np.concatenate(parts, axis=1).mean(axis=1)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _concept(dataset: str, class_id: str, record: dict, meta: dict) -> dict:
    context = f"{dataset} concept {class_id}"
    try:
        target = list(record["eval_idx"]["tgt"])
        complement = list(record["eval_idx"]["comp"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{context}: missing target/complement document indices") from exc
    nt, nc = len(target), len(complement)
    if not nt or not nc or len(set(target)) != nt or len(set(complement)) != nc or set(target) & set(complement):
        raise ValueError(f"{context}: invalid target/complement document indices")
    baseline, edits = {}, {alpha: {} for alpha in ALPHAS}
    reference_arrays = {}
    for space in ("h", "z"):
        try:
            reference = record["ref"][space]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{context}: missing own unedited {space} reference") from exc
        correct = _vector(reference, "doc_correct", nt + nc, f"{context} ref {space}", correctness=True)
        tc, cc = _confidence(reference, nt, nc, f"{context} ref {space}")
        reference_arrays[space] = correct
        baseline[space] = {
            "target_accuracy": 100 * float(correct[:nt].mean()),
            "complement_accuracy": 100 * float(correct[nt:].mean()),
            "target_confidence": 100 * tc, "complement_confidence": 100 * cc,
        }
        for alpha in ALPHAS:
            key = f"{space}_st_global_a{alpha}"
            if key not in record:
                raise ValueError(f"{context}: missing {key}; control evaluation is incomplete")
            edited = record[key]
            ec = _vector(edited, "doc_correct", nt + nc, f"{context} {key}", correctness=True)
            etc, ecc = _confidence(edited, nt, nc, f"{context} {key}")
            td = 100 * float((correct[:nt] - ec[:nt]).mean())
            cd = 100 * float((correct[nt:] - ec[nt:]).mean())
            tcd, ccd = 100 * (tc - etc), 100 * (cc - ecc)
            edits[alpha][space] = {
                "target_drop": td, "collateral": cd, "selectivity": td - cd,
                "target_accuracy": 100 * float(ec[:nt].mean()),
                "complement_accuracy": 100 * float(ec[nt:].mean()),
                "target_confidence": 100 * etc, "complement_confidence": 100 * ecc,
                "target_confidence_drop": tcd, "collateral_confidence_drop": ccd,
                "confidence_selectivity": tcd - ccd,
            }
    for alpha in ALPHAS:
        edits[alpha]["delta_selectivity"] = edits[alpha]["z"]["selectivity"] - edits[alpha]["h"]["selectivity"]
        edits[alpha]["delta_confidence_selectivity"] = (
            edits[alpha]["z"]["confidence_selectivity"] - edits[alpha]["h"]["confidence_selectivity"])
    classes = meta.get("classes", [])
    name = classes[int(class_id)] if class_id.isdigit() and int(class_id) < len(classes) else class_id
    return {"dataset": dataset, "dataset_group": GROUP_NAMES[dataset], "class_id": class_id,
            "class_name": name, "target_documents": nt, "complement_documents": nc,
            "eval_idx": {"tgt": target, "comp": complement}, "baseline": baseline, "alphas": edits}


def _aggregate(name: str, concepts: list[dict]) -> dict:
    result = {"dataset": name, "n_concepts": len(concepts),
              "n_target_documents": sum(c["target_documents"] for c in concepts),
              "n_complement_appearances": sum(c["complement_documents"] for c in concepts),
              "baseline": {}, "alphas": {}}
    if not concepts:
        result["reason"] = "No classes met the evaluation protocol"
        return result
    for space in ("h", "z"):
        result["baseline"][space] = {
            key: float(np.mean([c["baseline"][space][key] for c in concepts]))
            for key in concepts[0]["baseline"][space]
        }
    for alpha in ALPHAS:
        row = {space: {key: float(np.mean([c["alphas"][alpha][space][key] for c in concepts]))
                       for key in concepts[0]["alphas"][alpha][space]}
               for space in ("h", "z")}
        differences = np.asarray([c["alphas"][alpha]["delta_selectivity"] for c in concepts])
        conf_differences = np.asarray([c["alphas"][alpha]["delta_confidence_selectivity"] for c in concepts])
        strata = [c["dataset_group"] for c in concepts]
        row.update({"delta_selectivity": float(differences.mean()),
                    "delta_selectivity_ci95": _bootstrap_interval(differences, strata),
                    "delta_confidence_selectivity": float(conf_differences.mean()),
                    "delta_confidence_selectivity_ci95": _bootstrap_interval(conf_differences, strata),
                    "better": int((differences > 1e-9).sum()), "worse": int((differences < -1e-9).sum()),
                    "ties": int((np.abs(differences) <= 1e-9).sum())})
        result["alphas"][alpha] = row
    return result


def _atomic_text(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _markdown(summary: dict) -> str:
    lines = ["# Qwen paper control evaluation", "", "Accuracy and confidence use percentages; drops and differences use percentage points.",
             "Each concept has equal weight. FineWeb-Atlas pools its three frozen topic draws. Intervals use 10,000 paired concept bootstraps (seed 0); the overall row is stratified by dataset.", ""]
    for alpha in ("0.5", "0.4"):
        lines += [f"## Plain steering, alpha = {alpha}", "",
                  "| Dataset | Concepts | Target documents | Target drop: base | Target drop: AE | Collateral: base | Collateral: AE | Delta selectivity [95% CI] | Better / worse |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in summary["groups"]:
            if not group["n_concepts"]:
                lines.append(f"| {group['dataset']} | 0 | 0 | — | — | — | — | — | — |")
                continue
            row = group["alphas"][alpha]
            lo, hi = row["delta_selectivity_ci95"]
            interval = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "[not estimable]"
            lines.append(f"| {group['dataset']} | {group['n_concepts']} | {group['n_target_documents']} | "
                         f"{row['h']['target_drop']:.2f} | {row['z']['target_drop']:.2f} | "
                         f"{row['h']['collateral']:.2f} | {row['z']['collateral']:.2f} | "
                         f"{row['delta_selectivity']:+.2f} {interval} | {row['better']} / {row['worse']} |")
        lines.append("")
    lines += ["## Starting predictions", "", "Equal-concept means on the target documents, before editing, using each space's own reference.", "",
              "| Dataset | Base accuracy | AE accuracy | Base confidence | AE confidence |", "|---|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        if group["n_concepts"]:
            b = group["baseline"]
            lines.append(f"| {group['dataset']} | {b['h']['target_accuracy']:.2f} | {b['z']['target_accuracy']:.2f} | "
                         f"{b['h']['target_confidence']:.2f} | {b['z']['target_confidence']:.2f} |")
    lines += ["", "Target document counts sum the scored classes; complement documents are reused between concept edits. Class exclusions, source hashes, confidence changes, and every per-concept result are retained in control_summary.json.", ""]
    return "\n".join(lines)


def write_control_summary(arm_output: Path) -> dict:
    """Validate all seven completed runs and atomically write both summaries."""
    root = Path(arm_output)
    concepts, sources = [], {}
    for dataset in DATASETS:
        path = root / "interventions" / f"range_intervention_{dataset}_dprime_hz_tp2.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed paper control result: {path}")
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict) or not all(key in data for key in ("meta", "concepts", "summary")):
            raise ValueError(f"{path}: control result is incomplete (meta, concepts, summary required)")
        if not isinstance(data["concepts"], dict) or not isinstance(data["summary"], dict):
            raise ValueError(f"{path}: invalid concepts/summary records")
        meta = data["meta"]
        if meta.get("dataset", dataset) != dataset:
            raise ValueError(f"{path}: metadata identifies a different dataset")
        available_alphas = {f"{float(alpha):.1f}" for alpha in meta.get("alphas", ALPHAS)}
        if not set(ALPHAS) <= available_alphas:
            raise ValueError(f"{path}: both alpha 0.4 and 0.5 are required")
        records = [_concept(dataset, str(key), value, meta) for key, value in data["concepts"].items()]
        concepts.extend(records)
        counts = {str(key): int(value) for key, value in meta.get("n_eval_by_class", {}).items()}
        scored = {record["class_id"] for record in records}
        unscored = {key: value for key, value in counts.items() if key not in scored}
        sources[dataset] = {
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "checkpoint": meta.get("checkpoint"), "eval_sha": meta.get("eval_sha"),
            "n_scored_classes": len(records), "n_metadata_classes": len(meta.get("classes", counts)),
            "n_unscored_classes": len(unscored), "unscored_class_evaluation_counts": unscored,
            "classes_below_minimum20": {key: value for key, value in unscored.items() if value < 20},
            "other_unscored_classes": {key: value for key, value in unscored.items() if value >= 20},
            "classes": meta.get("classes"), "n_fit_by_class": meta.get("n_fit_by_class"),
            "n_eval_by_class": counts, "protocol_excluded_class_ids": ["26"] if dataset == "biasbios" else [],
            "unedited_dataset_accuracy": {"h": meta.get("base_acc"), "z": meta.get("zrecon_acc")},
            "unedited_dataset_confidence": {"h": meta.get("base_conf"), "z": meta.get("zrecon_conf")},
        }
    names = list(dict.fromkeys(GROUP_NAMES.values()))
    groups = [_aggregate(name, [record for record in concepts if record["dataset_group"] == name]) for name in names]
    groups.append(_aggregate("All concepts", concepts))
    result = {"status": "complete", "meta": {
        "units": "accuracy/confidence percentages; drops/selectivity/differences in percentage points",
        "main_alpha": 0.5, "alphas": [0.4, 0.5], "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": 0,
        "bootstrap": "paired concept bootstrap; overall row stratified by dataset; Atlas draws pooled as one dataset",
        "aggregation": "equal weight per scored concept; differences use each space's own unedited reference",
        "confidence": "probability of the correct label's first generated token",
        "target_document_count": "sum of scored-class target documents; Atlas draws count independently",
    }, "sources": sources, "groups": groups, "per_concept": concepts}
    json_text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    md_text = _markdown(result)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_text(root / "control_summary.json", json_text)
    _atomic_text(root / "control_summary.md", md_text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm_output", type=Path, required=True)
    args = parser.parse_args()
    result = write_control_summary(args.arm_output)
    print(f"Wrote control summaries: {len(result['per_concept'])} scored concepts in {args.arm_output}")


if __name__ == "__main__":
    main()
