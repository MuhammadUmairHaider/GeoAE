"""Five-arm reporting and common RAW reconstruction sanity requirements."""
import json

import pytest

from geoae.matched_topk import report


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sanity(fve=.8, delta=.3, *, finite=True):
    return {"mean_l0": 21., "raw_fve": fve, "clean_nll_nats": 2.,
            "spliced_nll_nats": 2. + delta, "spliced_delta_nll_nats": delta,
            "finite_gate_passed": finite, "rows": 128,
            "sample": "same held-out random controls"}


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "plot_validation", lambda *_: None)
    for kind in ("range", "atlas"):
        base = tmp_path / "evaluation" / kind
        put_json(base / "geoae_sanity.json", sanity(.97, .08))
        put_json(base / "gemma_scope_16k_sanity.json", sanity(.63, .46))
        for index, name in enumerate(report.expected_models()):
            dest = base / name
            provenance = {"analysis_plan_sha256": "frozen-plan", "model": name}
            put_json(dest / "complete.json", {"checkpoint": f"training/{name}/last.pt",
                     "checkpoint_sha256": name, "smoke": False})
            put_json(dest / "coding.json", {"provenance": provenance, "analysis": {
                "summary": {"interpretation_counts": {"same_feature_intensity_separation": 1}},
                "pairs": [1, 2]}})
            put_json(dest / "removal.json", {"analysis": {"paired_comparisons": []}})
            matching = {"comparisons": [], "status_counts": {}, "matches": []}
            for arm in ("sae", "geoae_range"):
                matching["matches"].append({"pair_id": 0, "target": "a", "arm": arm,
                    "level": .05, "feature_budget": 1, "status": "observed_interpolated",
                    "family": "one-family", "evaluation": {
                        "sibling": {"delta_nll": float(index)}, "random": {"delta_nll": 0.}}})
            put_json(dest / "observed_sae_matching.json", {"analysis": matching})
            put_json(dest / "sanity.json", sanity(.8 + .01 * index))
            # This deliberately disagrees with evaluation RAW FVE. The report
            # must never substitute normalized bypass validation FVE for it.
            put_json(tmp_path / "training" / name / "validation.jsonl",
                     {"visits": 500_000_000, "fve": .5, "mse": .1, "mean_l0": 21})
    return tmp_path


def test_default_report_has_four_k20_and_one_k100_no_dropped_width(evaluation):
    report.run(evaluation)
    summary = json.loads((evaluation / "summary.json").read_text())
    text = (evaluation / "report.md").read_text()
    assert summary["status"] == "complete"
    assert len(summary["expected_models"]) == 5
    assert len(summary["models"]) == 10
    assert "batch_topk_20480_k100" in summary["expected_models"]
    assert all("5120" not in name for name in summary["expected_models"])
    assert "six SAEs" not in text and "All 5 matched SAEs" in text
    k_comparisons = [row for row in summary["comparisons"] if row["factor"] == "k"]
    assert len(k_comparisons) == 2
    for row in k_comparisons:
        assert (row["left"], row["right"]) == ("batch_topk_20480_k100", "batch_topk_20480")
        assert row["n_common_targets"] == 1
        assert row["difference_sibling"]["mean"] == 3.
    model = summary["models"]["range/batch_topk_20480_k100"]
    assert model["training_provenance"]["analysis_plan_sha256"] == "frozen-plan"
    assert model["evaluation_checkpoint"]["checkpoint"].endswith("last.pt")


def test_report_shows_same_raw_units_and_absolute_spliced_loss_for_all(evaluation):
    report.run(evaluation)
    text = (evaluation / "report.md").read_text()
    summary = json.loads((evaluation / "summary.json").read_text())
    # Baselines and SAEs share one table, with measured values and NLL units.
    assert "RAW residual FVE | Clean next-token NLL | Spliced next-token NLL | Spliced ΔNLL" in text
    assert "| range | geoae | 21.0000 | 0.9700 | 2.0000 | 2.0800 | 0.0800 |" in text
    assert "| range | gemma_scope_16k | 21.0000 | 0.6300 | 2.0000 | 2.4600 | 0.4600 |" in text
    assert "| range | batch_topk_10240 | 21.0000 | 0.8000 | 2.0000 | 2.3000 | 0.3000 |" in text
    assert "token mean is added back" in text
    assert "SST centers the original RAW residual per dimension" in text
    assert "bypass validation FVE is a different metric" in text
    assert summary["models"]["range/batch_topk_10240"]["sanity"]["raw_fve"] == .8
    assert "mean(raw, axis=0)" in summary["raw_fve_definition"]


def test_missing_required_reference_prevents_complete_report(evaluation):
    (evaluation / "evaluation" / "atlas" / "gemma_scope_16k_sanity.json").unlink()
    report.run(evaluation)
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert len(summary["missing"]) == 1
    assert "atlas/gemma_scope_16k_sanity.json" in summary["missing"][0]
    # The optional 262k arm is not required for the five-SAE comparison.
    assert all("262k" not in entry for entry in summary["missing"])
    assert "Missing required artifacts:" in (evaluation / "report.md").read_text()


def test_failed_or_missing_raw_loss_sanity_cannot_be_hidden(evaluation):
    base = evaluation / "evaluation" / "range"
    put_json(base / "batch_topk_20480_k100" / "sanity.json", sanity(finite=False))
    value = sanity()
    del value["clean_nll_nats"]
    put_json(base / "geoae_sanity.json", value)
    report.run(evaluation)
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert len(summary["failed_sanity"]) == 1
    assert "clean_nll_nats" in summary["missing"][0]


def test_requested_widths_and_k_are_flexible():
    names = report.expected_models([128, 256], ["matryoshka_256_k100"])
    assert names == ["batch_topk_128", "batch_topk_256", "matryoshka_128",
                     "matryoshka_256", "matryoshka_256_k100"]
    comparisons = report.requested_comparisons(names)
    assert ("matryoshka_256_k100", "matryoshka_256", "k") in comparisons
    assert ("matryoshka_128", "batch_topk_128", "trainer") in comparisons
    assert ("batch_topk_128", "batch_topk_256", "width") in comparisons
    with pytest.raises(ValueError, match="Invalid matched SAE"):
        report.expected_models(extra_models=["unknown_20480_k100"])
