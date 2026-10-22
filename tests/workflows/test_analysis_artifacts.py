"""Saved analyses read the artifact store and write new checkout-local outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from geoae import paths
from experiments.paper.analysis import plot_gemma_scope_compare as scope_plot
from experiments.paper.analysis import render_evaluation_report as report_renderer


ROOT = Path(__file__).resolve().parents[2]


def test_scope_figure_reads_archived_json_and_keeps_outputs_local(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    archive = tmp_path / "archive"
    source = archive / "results/scope/atlas/width_16k/coding.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"n_features": 16384, "pairs": [
        {"interpretation": "unresolved"}], "summary": {
            "n_labels": 2, "n_heldout_supported": 0}}))
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(paths, "PACKAGE_ROOT", checkout)
    monkeypatch.setattr(paths, "ARTIFACT_ROOT", archive)

    outputs = scope_plot.render_figures("results/scope", dpi=30)

    assert outputs and all(Path(p).is_relative_to(checkout) for p in outputs)
    summary = json.loads(next(Path(p) for p in outputs if p.endswith("_source.json")).read_text())
    assert summary["result_root"] == str(archive / "results/scope")
    assert summary["input_runs"][0]["sources"]["coding"]["path"] == str(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    assert sorted(p.relative_to(archive) for p in archive.rglob("*") if p.is_file()) == [
        source.relative_to(archive)]


def test_report_renderer_rebases_citations_without_changing_preserved_markdown(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    source = checkout / "docs/history/reports/report.md"
    citation = tmp_path / "archive/results/evidence.json"
    source.parent.mkdir(parents=True)
    citation.parent.mkdir(parents=True)
    citation.write_text('{"measured": true}')
    original = f"# Saved report\n\n## Evidence\n\n[Result]({os.path.relpath(citation, source.parent)}#detail)\n"
    source.write_text(original)
    output = checkout / "results/paper/report.html"
    monkeypatch.setattr(report_renderer, "ROOT", checkout)
    monkeypatch.setattr(report_renderer, "SOURCE", source)
    monkeypatch.setattr(report_renderer, "OUTPUT", output)

    report_renderer.render()

    page = output.read_text()
    assert f'href="{os.path.relpath(citation, output.parent)}#detail"' in page
    assert f'href="{os.path.relpath(source, output.parent)}"' in page
    assert source.read_text() == original
    assert citation.read_text() == '{"measured": true}'


@pytest.mark.parametrize("missing_config", [False, True])
def test_dbpedia_launcher_uses_canonical_modules_and_validates_before_compute(
        tmp_path, missing_config):
    checkout = tmp_path / "checkout"
    config_dir = checkout / "experiments/dbpedia/configs/llama3.2-3B/layer27"
    config_dir.mkdir(parents=True)
    (config_dir / "unprompted_mean_gelu.yaml").write_text("{}\n")
    if not missing_config:
        (config_dir / "unprompted_mean_semisup.yaml").write_text("{}\n")
    interpreter = checkout / "scripts/py"
    interpreter.parent.mkdir()
    interpreter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
    interpreter.chmod(0o755)
    commands = tmp_path / "commands.txt"
    result = subprocess.run(["bash", str(ROOT / "experiments/dbpedia/scripts/run_experiment.sh")],
                            cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "GEOAE_ROOT": str(checkout),
                                 "COMMAND_LOG": str(commands), "MODEL": "llama3.2-3B",
                                 "LAYER": "27", "MODE": "unprompted", "POOLING": "mean",
                                 "VARIANTS": "gelu semisup"})
    if missing_config:
        assert result.returncode != 0
        assert "MISSING config" in result.stderr
        assert not commands.exists()
    else:
        assert result.returncode == 0, result.stderr
        calls = commands.read_text().splitlines()
        assert len(calls) == 4
        assert calls[0].startswith("-m geoae.extraction.dbpedia ")
        assert all(call.startswith("-m geoae.training.mse --config experiments/dbpedia/configs/")
                   for call in calls[1:3])
        assert calls[3].startswith("-m geoae.evaluation.dbpedia --compare ")
        assert (checkout / "dbpedia/results/llama3.2-3B/layer27").is_dir()
