from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.nonlinear_decoders.deltaai.qwen_paper_summary import DATASETS, _bootstrap_interval, write_control_summary


def record(nt=2, delta=100):
    nc = 2
    h_target = [1] * nt if delta > 0 else [0] * nt
    z_target = [0] * nt if delta > 0 else [1] * nt
    baseline = {"doc_correct": [1] * (nt + nc), "doc_conf": [.8] * (nt + nc)}
    result = {"eval_idx": {"tgt": list(range(nt)), "comp": list(range(nt, nt + nc))},
              "ref": {"h": dict(baseline), "z": dict(baseline)}}
    for alpha in ("0.4", "0.5"):
        for space, target in (("h", h_target), ("z", z_target)):
            result[f"{space}_st_global_a{alpha}"] = {
                "doc_correct": target + [1] * nc,
                "doc_conf": [.8 if value else .2 for value in target] + [.8] * nc,
                "tgt_drop": -999, "selectivity": -999,
            }
    return result


def write_inputs(root: Path, records_by_dataset=None):
    directory = root / "interventions"
    directory.mkdir(parents=True)
    for dataset in DATASETS:
        records = (records_by_dataset or {}).get(dataset, {"0": record()})
        payload = {"meta": {"dataset": dataset, "alphas": [.4, .5], "classes": ["first", "second"],
                            "n_eval_by_class": {key: len(value["eval_idx"]["tgt"]) for key, value in records.items()}},
                   "concepts": records, "summary": {"st_global_a0.5": {"n": len(records)}}}
        path = directory / f"range_intervention_{dataset}_dprime_hz_tp2.json"
        path.write_text(json.dumps(payload))


def test_uses_each_space_own_document_reference_and_ignores_rounded_drops(tmp_path):
    item = record()
    item["ref"]["z"]["doc_correct"] = [1, 0, 1, 1]
    for alpha in ("0.4", "0.5"):
        item[f"h_st_global_a{alpha}"]["doc_correct"] = [0, 1, 1, 1]
        item[f"z_st_global_a{alpha}"]["doc_correct"] = [0, 0, 1, 1]
    write_inputs(tmp_path, {"db14": {"0": item}})
    result = write_control_summary(tmp_path)
    concept = next(c for c in result["per_concept"] if c["dataset"] == "db14")
    assert concept["baseline"]["h"]["target_accuracy"] == 100
    assert concept["baseline"]["z"]["target_accuracy"] == 50
    assert concept["alphas"]["0.5"]["h"]["target_drop"] == 50
    assert concept["alphas"]["0.5"]["z"]["target_drop"] == 50
    assert concept["alphas"]["0.5"]["delta_selectivity"] == 0
    assert json.loads((tmp_path / "control_summary.json").read_text())["status"] == "complete"
    assert "Plain steering, alpha = 0.5" in (tmp_path / "control_summary.md").read_text()


def test_equal_concept_pooling_atlas_draws_and_paired_stratified_bootstrap(tmp_path):
    write_inputs(tmp_path, {"db14": {"0": record(nt=50, delta=100), "1": record(nt=2, delta=-100)}})
    result = write_control_summary(tmp_path)
    rows = {row["dataset"]: row for row in result["groups"]}
    assert rows["DBpedia-14"]["alphas"]["0.5"]["delta_selectivity"] == 0
    assert rows["FineWeb-Atlas"]["n_concepts"] == 3
    assert rows["All concepts"]["n_concepts"] == 8
    assert rows["All concepts"]["alphas"]["0.5"]["delta_selectivity"] == 75
    # Every concept has a constant paired difference: large common variation in
    # h/z disappears before bootstrapping rather than inflating uncertainty.
    assert _bootstrap_interval(np.array([7., 7., 7.]), ["A", "A", "B"]) == [7., 7.]
    actual = rows["All concepts"]["alphas"]["0.5"]["delta_selectivity_ci95"]
    # The only varying stratum has two concepts; its four equally likely
    # resamples yield overall differences 50, 75, 75, 100.
    assert actual == [50., 100.]


def test_missing_or_partial_control_refuses_to_publish(tmp_path):
    write_inputs(tmp_path)
    path = tmp_path / "interventions/range_intervention_atlas_s2_dprime_hz_tp2.json"
    data = json.loads(path.read_text())
    del data["concepts"]["0"]["z_st_global_a0.4"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        write_control_summary(tmp_path)
    assert not (tmp_path / "control_summary.json").exists()
    assert not (tmp_path / "control_summary.md").exists()
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing completed"):
        write_control_summary(tmp_path)


def test_rejects_document_misalignment(tmp_path):
    item = record()
    item["z_st_global_a0.5"]["doc_correct"] = [1]
    write_inputs(tmp_path, {"db14": {"0": item}})
    with pytest.raises(ValueError, match="must contain 4"):
        write_control_summary(tmp_path)
