"""Reports must distinguish measured results, missing artifacts, and partial API work."""
import json
from pathlib import Path

from experiments.depth import report


def output_root():
    return Path("results/midlayer_run/llama3.2-3B/layer14")


def test_report_does_not_fabricate_results_before_evaluation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out, data, text = report.make_report("llama", "midlayer_run")
    assert out == output_root()
    assert data["status"] == "not_run"
    assert "No measured headline results" in text
    assert "MMLU accuracy:" not in text
    assert data["stages"]["mmlu"]["status"] == "not_run"
    assert not list(tmp_path.iterdir())


def test_report_preserves_partial_api_and_full_measurements(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = output_root()
    (out / "evals").mkdir(parents=True)
    (out / "pipeline_state.json").write_text(json.dumps({"status": "partial", "requested": ["mmlu", "judge"],
        "steps": {"mmlu": {"status": "complete"}, "judge": {"status": "skipped", "reason": "API key missing"}}}))
    (out / "evals/mmlu.json").write_text(json.dumps({"meta": {
        "base_acc": .5, "recon_acc": .45, "delta": -.05, "n": 100}, "preds": {"base": [0] * 100}}))
    _, data, text = report.make_report("llama", "midlayer_run")
    assert data["status"] == "partial"
    assert data["stages"]["judge"]["reason"] == "API key missing"
    assert "50.00% → bypass reconstruction 45.00% (-5.00 percentage points" in text
    assert not any("preds" in key for artifact in data["artifacts"].values() for key in artifact["metrics"])


def test_report_does_not_trust_complete_marker_with_missing_or_invalid_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = output_root()
    (out / "evals").mkdir(parents=True)
    (out / "pipeline_state.json").write_text(json.dumps({"status": "complete", "requested": ["mmlu", "probe"],
        "steps": {"mmlu": {"status": "complete"}, "probe": {"status": "complete"}}}))
    (out / "evals/mmlu.json").write_text("{interrupted")
    _, data, text = report.make_report("llama", "midlayer_run")
    assert data["status"] == "failed"
    assert data["stages"]["mmlu"]["status"] == "invalid_artifact"
    assert data["stages"]["probe"]["status"] == "incomplete_artifact"
    assert "MMLU accuracy:" not in text
    assert "Evaluation status: **failed**" in text


def test_report_omits_headlines_from_failed_stage_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = output_root()
    (out / "evals").mkdir(parents=True)
    (out / "pipeline_state.json").write_text(json.dumps({"status": "failed", "requested": ["mmlu"],
        "steps": {"mmlu": {"status": "failed"}}}))
    (out / "evals/mmlu.json").write_text(json.dumps({"meta": {
        "base_acc": .5, "recon_acc": .45, "delta": -.05, "n": 100}}))
    _, data, text = report.make_report("llama", "midlayer_run")
    assert data["stages"]["mmlu"]["status"] == "failed"
    assert "MMLU accuracy:" not in text
