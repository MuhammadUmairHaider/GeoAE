"""Recovery preserves completed artifacts and repairs interrupted GPU stages."""
from argparse import Namespace
import fcntl
import json
from pathlib import Path

import pytest

from experiments.gpt2 import evaluate, recover_evals as recovery


def options(root, **overrides):
    values = dict(layers="0", run_name="gpt_run", activations_dir=root / "activations",
                  corpus_dir="corpus/gpt2_sampled_struct5_v1", concurrency=2, dry_run=True,
                  tmpfs_dir=root / "tmpfs")
    values.update(overrides)
    return Namespace(**values)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_plan(tmp_path, monkeypatch, *, layer=0, partial=True):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    folder = Path(f"checkpoints/gpt2/layer{layer}") / evaluate.CKDIR
    folder.mkdir(parents=True)
    checkpoint = folder / "step_0014500.pt"
    checkpoint.touch()
    out = tmp_path / f"results/gpt_run/layer{layer}"
    steer = evaluate.Step("steer", [["unused"]], [out / "steer.json"])
    ranges = evaluate.Step("range_db14", [["unused"]], [out / "range.json"])
    write_json(steer.outputs[0], dict(meta={}, concepts={"0": {"result": 1}}))
    write_json(ranges.outputs[0], dict(meta={}, concepts={}) if partial else dict(meta={}, concepts={}, summary={}))
    old = dict(checkpoint=str(checkpoint), status="failed", requested=["steer", "range_db14"],
               steps={"steer": {"status": "complete"}, "range_db14": {"status": "running"}})
    write_json(out / "pipeline_state.json", old)
    monkeypatch.setattr(evaluate, "build_steps", lambda *unused: [steer, ranges])
    return checkpoint, out, steer, ranges


def test_partial_range_snapshot_requires_final_summary(tmp_path):
    path = tmp_path / "range.json"
    step = evaluate.Step("range_db14", [], [path])
    write_json(path, dict(meta={}, concepts={"0": {}}))
    assert not evaluate.complete(step)
    write_json(path, dict(meta={}, concepts={"0": {}}, summary={}))
    assert evaluate.complete(step)
    path.write_text('{"meta":')
    assert not evaluate.complete(step)


def test_killed_json_write_and_invalid_status_file_are_incomplete(tmp_path):
    path = tmp_path / "metric.json"
    path.write_text('{"partial":')
    assert not evaluate.complete(evaluate.Step("core_eval", [], [path]))
    write_json(path, [])
    assert not evaluate.complete(evaluate.Step("number_hb", [], [path], complete_status=True))


def test_cancelled_running_range_selected_completed_steer_preserved(tmp_path, monkeypatch):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch)
    args = options(tmp_path)
    plan = recovery.plans_for(args, [0])[0]
    assert plan["selected"] == ["range_db14"]
    assert not plan["stage_arrays"]
    assert plan["checkpoint"] == checkpoint
    assert "--force" not in plan["command"]
    assert plan["command"][plan["command"].index("--judges") + 1] == "skip"
    assert all("geoae.training.mse" not in word and "concurrency.py" not in word for word in plan["command"])


def test_dry_run_prints_exact_retry_and_makes_no_changes(tmp_path, monkeypatch, capsys):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(evaluate, "guard_checkpoint", lambda *unused: pytest.fail("dry run loaded checkpoint tensors"))
    monkeypatch.setattr(recovery, "run_layers", lambda *unused: pytest.fail("dry run launched processes"))
    assert recovery.recover(options(tmp_path)) == 0
    printed = capsys.readouterr().out
    assert "--steps range_db14" in printed
    assert "steer," not in printed
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert not (tmp_path / "runs").exists()


def test_recovery_refuses_active_original_pipeline(tmp_path, monkeypatch):
    fixture_plan(tmp_path, monkeypatch)
    runs = tmp_path / "runs/gpt_run"
    runs.mkdir(parents=True)
    monkeypatch.setattr(evaluate, "guard_checkpoint", lambda *unused: None)
    with (runs / ".pipeline.lock").open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Original pipeline"):
            recovery.recover(options(tmp_path, dry_run=False))
    assert not (runs / "recovery").exists()


def test_recovery_refuses_live_layer_evaluator(tmp_path, monkeypatch):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, "guard_checkpoint", lambda *unused: None)
    with (out / ".evaluation.lock").open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="evaluator is still active"):
            recovery.recover(options(tmp_path, dry_run=False))


def test_recovery_archives_evidence_and_preserves_scheduler_history(tmp_path, monkeypatch):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch)
    runs = tmp_path / "runs/gpt_run"
    original = dict(job_id="284386", status="interrupted", layers=[0], error="cancelled after driver stall",
                    stages={"train": {"status": "complete"}, "eval_gpu": {"status": "failed", "failed_layers": [0]}})
    manifest = dict(jobs={"pipeline": "284386"}, model="gpt2")
    write_json(runs / "pipeline_state.json", original)
    write_json(runs / "campaign.json", manifest)
    old_campaign_bytes = (runs / "campaign.json").read_bytes()
    old_pipeline_bytes = (runs / "pipeline_state.json").read_bytes()
    write_json(out / "logs/range_db14.log", {"old_error": "position overflow"})
    steer_bytes = steer.outputs[0].read_bytes()
    verified = []
    monkeypatch.setattr(evaluate, "guard_checkpoint", lambda path, layer: verified.append((path, layer)))
    monkeypatch.setenv("SLURM_JOB_ID", "recovery_99")
    monkeypatch.setattr(recovery, "host_memory_snapshot", lambda: {"available_bytes": 128 * recovery.GIB})
    calls = []

    def finish(stage, layers, concurrency, command_for, state, state_path, logs):
        calls.append((stage, layers, concurrency, command_for(0)))
        write_json(ranges.outputs[0], dict(meta={}, concepts={}, summary={}))
        layer_state = json.loads((out / "pipeline_state.json").read_text())
        layer_state["steps"]["range_db14"]["status"] = "complete"
        layer_state["status"] = "complete"
        write_json(out / "pipeline_state.json", layer_state)
        return True

    monkeypatch.setattr(recovery, "run_layers", finish)
    assert recovery.recover(options(tmp_path, dry_run=False)) == 0
    assert verified == [(checkpoint, 0)]
    assert calls[0][:3] == ("eval_batch0", [0], 1)
    assert calls[0][3][calls[0][3].index("--steps") + 1] == "range_db14"
    assert steer.outputs[0].read_bytes() == steer_bytes
    assert (runs / "campaign.json").read_bytes() == old_campaign_bytes
    attempt = next((runs / "recovery").iterdir())
    assert (attempt / "original_pipeline_state.json").read_bytes() == old_pipeline_bytes
    assert (attempt / "layer0/range_db14.log").is_file()
    final = json.loads((runs / "pipeline_state.json").read_text())
    assert final["job_id"] == "284386"
    assert final["recovery_job_id"] == "recovery_99"
    assert final["status"] == "complete"
    assert final["stages"]["train"] == {"status": "complete"}
    assert final["original_error"] == original["error"]


def test_only_missing_activation_reading_phase_requires_staging(tmp_path, monkeypatch):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch)
    heavy = evaluate.Step("cq", [["unused"]], [out / "cq.json"])
    monkeypatch.setattr(evaluate, "build_steps", lambda *unused: [steer, ranges, heavy])
    plan = recovery.plans_for(options(tmp_path), [0])[0]
    assert plan["stage_arrays"]
    assert plan["selected"] == ["range_db14", "cq"]


def test_valid_array_metric_with_stale_failure_record_needs_no_staging(tmp_path, monkeypatch):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch)
    cq = evaluate.Step("cq", [["unused"]], [out / "cq.json"])
    write_json(cq.outputs[0], {"metric": 1})
    monkeypatch.setattr(evaluate, "build_steps", lambda *unused: [steer, ranges, cq])
    plan = recovery.plans_for(options(tmp_path), [0])[0]
    assert plan["selected"] == ["range_db14", "cq"]
    assert not plan["stage_arrays"]


def test_checkpoint_mismatch_prevents_artifact_reuse(tmp_path, monkeypatch):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch)
    (checkpoint.parent / "step_0014501.pt").touch()
    with pytest.raises(ValueError, match="different checkpoint"):
        recovery.plans_for(options(tmp_path), [0])


def test_interruption_before_last_block_skips_recovers_complete_state(tmp_path, monkeypatch):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch, partial=False)
    skipped = [evaluate.Step(name, [["must never launch"]], [out / f"{name}.json"], last_block_only=True)
               for name in ("csteer", "cgen", "cgen_concept")]
    monkeypatch.setattr(evaluate, "build_steps", lambda *unused: [steer, ranges, *skipped])
    state_path = out / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    state["requested"] += [step.name for step in skipped]
    write_json(state_path, state)
    plan = recovery.plans_for(options(tmp_path), [0])[0]
    assert plan["selected"] == ["range_db14", "csteer", "cgen", "cgen_concept"]
    assert not plan["stage_arrays"]
    args = recovery.recipe_args(options(tmp_path))
    args.steps = ",".join(plan["selected"])
    args.activations_dir = tmp_path / "activations"
    args.dry_run, args.force, args.judges = False, False, "skip"
    monkeypatch.setattr(evaluate, "code_files", lambda unused: [])
    assert evaluate.run_layer(0, args)
    final = json.loads(state_path.read_text())
    assert final["status"] == "complete"
    assert all(final["steps"][step.name]["status"] == "skipped" for step in skipped)
    assert recovery.final_outcomes(options(tmp_path), {0: plan})["0"]["status"] == "complete"


def dump_files(root):
    source = root / "activations"
    source.mkdir()
    names = ["meta.json", "docs.npy", *[f"rows_{name}.npy" for name in ("doc", "pos", "tok", "src", "norm")]]
    names += [f"layer_{layer}.npy" for layer in (0, 1, 11)]
    names += [f"norm_params_layer{layer}.npz" for layer in (0, 1, 11)]
    for name in names:
        (source / name).write_bytes(name.encode())
    return source


def test_stage_selection_preserves_all_sidecars_and_only_batch_arrays(tmp_path):
    source = dump_files(tmp_path)
    selected = {path.name for path in recovery.staging_files(source, [0, 1])}
    assert {"meta.json", "docs.npy", "rows_doc.npy", "rows_pos.npy", "rows_tok.npy", "rows_src.npy", "rows_norm.npy"} <= selected
    assert {"layer_0.npy", "layer_1.npy", "norm_params_layer0.npz", "norm_params_layer1.npz"} <= selected
    assert "layer_11.npy" not in selected


def test_batch_reduces_to_one_when_two_workers_exceed_host_allocation(tmp_path, monkeypatch):
    dump_files(tmp_path)
    (tmp_path / "tmpfs").mkdir()
    plans = {layer: {"stage_arrays": True} for layer in (0, 1)}
    monkeypatch.setattr(recovery, "host_memory_snapshot", lambda: {"available_bytes": 32 * recovery.GIB})
    batch, files, evidence = recovery.admitted_batch(options(tmp_path), plans, [0, 1])
    assert batch == [0]
    assert evidence["staged_layers"] == [0]
    assert "layer_1.npy" not in {p.name for p in files}
    assert evidence["worker_budget_bytes"] == 16 * recovery.GIB
    monkeypatch.setattr(recovery, "host_memory_snapshot", lambda: {"available_bytes": 16 * recovery.GIB})
    with pytest.raises(RuntimeError, match="cannot admit"):
        recovery.admitted_batch(options(tmp_path), plans, [0, 1])


def test_staged_batch_passes_shared_dump_and_cleans_it_after_children(tmp_path, monkeypatch):
    checkpoint, out, steer, ranges = fixture_plan(tmp_path, monkeypatch, partial=False)
    dump_files(tmp_path)
    (tmp_path / "tmpfs").mkdir()
    cq = evaluate.Step("cq", [["unused"]], [out / "cq.json"])
    monkeypatch.setattr(evaluate, "build_steps", lambda *unused: [steer, ranges, cq])
    monkeypatch.setattr(evaluate, "guard_checkpoint", lambda *unused: None)
    monkeypatch.setattr(recovery, "host_memory_snapshot", lambda: {"available_bytes": 128 * recovery.GIB})
    seen = []

    def finish(stage, layers, concurrency, command_for, state, state_path, logs):
        command = command_for(0)
        dump = Path(command[command.index("--activations-dir") + 1])
        assert dump != tmp_path / "activations"
        assert (dump / "layer_0.npy").read_bytes() == b"layer_0.npy"
        assert not (dump / "layer_11.npy").exists()
        assert (dump / "meta.json").read_bytes() == b"meta.json"
        seen.append(dump)
        write_json(cq.outputs[0], {"metric": 1})
        current = json.loads((out / "pipeline_state.json").read_text())
        current["steps"].update(range_db14={"status": "complete"}, cq={"status": "complete"})
        current["status"] = "complete"
        write_json(out / "pipeline_state.json", current)
        return True

    monkeypatch.setattr(recovery, "run_layers", finish)
    assert recovery.recover(options(tmp_path, dry_run=False)) == 0
    assert seen and all(not path.exists() for path in seen)
    assert (tmp_path / "activations/layer_0.npy").read_bytes() == b"layer_0.npy"
