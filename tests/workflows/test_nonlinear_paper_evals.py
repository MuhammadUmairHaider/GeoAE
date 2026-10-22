"""Protect model identities and resumable, matched nonlinear paper evaluation."""
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "experiments/nonlinear_decoders/deltaai"))
    return importlib.import_module("nonlinear_paper_plan"), importlib.import_module("nonlinear_paper_evals")


def profile(plan, model="llama"):
    gemma = model == "gemma"
    return {"model_name": "google/gemma-3-4b-pt" if gemma else "meta-llama/Llama-3.2-3B",
            "target_layer": 33 if gemma else 27, "hidden_size": 2560 if gemma else 3072,
            "latent_dim": 5120 if gemma else 6144, "n_clusters": 2000,
            "epochs": 50, "final_step": 14450,
            "assets": {**{key: "/assets/" + key for key in plan.REQUIRED_ASSETS},
                       "atlas_control_classes": {"atlas_s0": [0, 2], "atlas_s1": "1,3", "atlas_s2": [4, 5]},
                       **({"sae": "scope:/assets/sae", "sae_dir": "/assets/sae", "sae_kmeans": "/assets/sae_codebook.npz"} if gemma else {})}}


@pytest.mark.parametrize("training_matches", [True, False])
@pytest.mark.parametrize("external_checkout", [True, False])
def test_default_model_preparation_uses_complete_current_source(modules, tmp_path, training_matches, external_checkout):
    plan, runner = modules
    checkout = Path(__file__).resolve().parents[2]
    asset = tmp_path / "fixture.asset"
    asset.write_text("fixture")
    p = profile(plan)
    p["assets"].update({key: str(asset) for key in plan.REQUIRED_ASSETS if key != "atlas_control_classes"})
    profile_path = tmp_path / "profile.json"
    runner.atomic_json(profile_path, p)
    training = tmp_path / "training"
    training.mkdir()
    training_source = training / "source"
    for name in runner.CORE_OVERLAYS:
        target = training_source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkout / name, target)
    if not training_matches:
        (training_source / "geoae/checkpoint.py").write_text("# Different, legitimately frozen checkpoint semantics\n")
    core_hashes = {name: runner.digest(training_source / name) for name in runner.CORE_OVERLAYS}
    runner.atomic_json(training / "training_manifest.json", {"profile": p,
        "runtime": {"python": sys.executable, "modules": []}, "dataset": {"files": {"fixture.npy": {}}},
        "source_root": str(training_source), "source_sha256": core_hashes})
    args = SimpleNamespace(run_dir=training, profile=profile_path,
                           paper_repo=checkout if external_checkout else None, python=None)
    if not training_matches:
        with pytest.raises(ValueError, match="Evaluation core differs from frozen training source: geoae/checkpoint.py"):
            runner.prepare(args)
        assert not (training / "evaluation").exists()
        return
    runner.prepare(args)
    manifest = runner.load_manifest(training / "evaluation")
    assert manifest["paper_origin"] == str(checkout / "docs/paper/paper.tex")
    assert manifest["nonlinear_overlays_sha256"] == manifest["nonlinear_evaluator_patches"] == {}
    assert manifest["training_core_sha256"] == core_hashes
    assert len(manifest["plans"]["mlp"]) == 18
    for step in manifest["plans"]["mlp"]:
        command = step["command"]
        target = (command[command.index("-m") + 1].replace(".", "/") + ".py" if "-m" in command
                  else next(argument for argument in command if argument.endswith(".py")))
        assert (Path(manifest["source_root"]) / target).is_file()


def test_historical_model_checkout_is_rejected_before_reading_inputs_or_creating_output(modules, tmp_path):
    _, runner = modules
    historical = tmp_path / "historical"
    (historical / "geoae/interp").mkdir(parents=True)
    old_source = historical / "geoae/interp/range_intervention_compare.py"
    old_source.write_text("# historical evaluator\n")
    with pytest.raises(ValueError, match="reconciled source layout.*historical"):
        runner.prepare(SimpleNamespace(run_dir=tmp_path / "training", profile=tmp_path / "profile.json",
                                       paper_repo=historical, python=None))
    assert not (tmp_path / "training").exists()
    assert old_source.read_text() == "# historical evaluator\n"


@pytest.mark.parametrize("model", ["llama", "gemma"])
def test_plan_uses_profile_identity_and_exact_final_checkpoint(modules, model):
    plan, runner = modules
    p = profile(plan, model)
    manifest = {"profile": p, "training_run_dir": "/runs/model"}
    ck = runner.checkpoint_path(manifest, "residual_mlp")
    assert str(ck) == "/runs/model/checkpoints/residual_mlp/step_0014450.pt"
    steps = plan.build_steps(p, ck, Path("/eval/results"), "/env/python")
    count = 19 if model == "gemma" else 18
    assert len(steps) == count
    assert [s["group"] for s in steps] == ["core"] * 4 + ["alignment"] * (7 if model == "gemma" else 6) + ["controls"] * 7 + ["ranges"]
    assert len({step["name"] for step in steps}) == count
    for step in steps:
        command = step["command"]
        assert not any("Qwen" in value for value in command)
        if "--layer" in command:
            assert command[command.index("--layer") + 1] == str(p["target_layer"])
        assert not any("step_0015250" in value for value in command)
    controls = [s for s in steps if s["group"] == "controls"]
    for step in controls:
        command = step["command"]
        assert command[command.index("--checkpoint") + 1] == str(ck)
        assert command[command.index("--substrates") + 1] == ("h,z,s,s1" if model == "gemma" else "h,z")
        assert ("--tp_all" in command) == (model == "gemma")
        assert ("--sae" in command) == (model == "gemma")
    if model == "gemma":
        probe = next(s for s in steps if s["name"] == "probe_alignment")["command"]
        assert "scope16:geoae" in probe[probe.index("--pairs") + 1]
        assert "scope16=saekm:scope:/assets/sae@/assets/sae_codebook.npz" in probe[probe.index("--arms") + 1]
        assert any(s["name"] == "atlas_pair10_sae" for s in steps)


def test_gemma_never_silently_omits_the_paper_sae_control(modules):
    plan, _ = modules
    p = profile(plan, "gemma")
    p["assets"].pop("sae")
    with pytest.raises(ValueError, match="SAE"):
        plan.build_steps(p, Path("ae.pt"), Path("out"), "python")


def test_llama_uses_exact_paper_control_draw_five_without_changing_alignment_draws(modules):
    plan, _ = modules
    p = profile(plan)
    p["assets"].update(atlas_control_datasets=["atlas_s5"], atlas_control_classes={"atlas_s5": [0, 1, 2]})
    steps = plan.build_steps(p, Path("/final.pt"), Path("/results"), "/python")
    controls = [step for step in steps if step["group"] == "controls"]
    assert len(steps) == 16 and len(controls) == 5
    assert controls[-1]["name"] == "control_atlas_s5"
    assert all("control_atlas_s0" != step["name"] for step in steps)
    atlas = next(step for step in steps if step["name"] == "atlas_pair10")
    assert atlas["command"][atlas["command"].index("--draws") + 1] == p["assets"]["atlas_draws"]


@pytest.mark.parametrize("model", ["llama", "gemma"])
def test_checkpoint_guard_derives_dimensions_layer_and_final_step_from_profile(modules, tmp_path, model):
    plan, runner = modules
    p = profile(plan, model)
    data = {"config": {"model": {key: p[key] for key in ("hidden_size", "latent_dim", "n_clusters")},
                       "extraction": {"model_name": p["model_name"]}, "data": {"target_layer": p["target_layer"]}},
            "epoch": 50, "step": 14450, "model_state": {"centroids_initialized": torch.tensor(True)}}
    data["config"]["model"]["decoder_type"] = "mlp"
    path = tmp_path / "checkpoint.pt"
    torch.save(data, path)
    assert runner.verify_checkpoint(path, "mlp", p)["step"] == 14450
    with pytest.raises(ValueError, match="architecture"):
        runner.verify_checkpoint(path, "residual_mlp", p)
    wrong = dict(p, final_step=15250)
    with pytest.raises(ValueError, match="Expected final"):
        runner.verify_checkpoint(path, "mlp", wrong)
    with pytest.raises(ValueError, match="dimensions"):
        runner.verify_checkpoint(path, "mlp", dict(p, hidden_size=2048))
    with pytest.raises(ValueError, match="language model"):
        runner.verify_checkpoint(path, "mlp", dict(p, model_name="other/model"))
    with pytest.raises(ValueError, match="layer"):
        runner.verify_checkpoint(path, "mlp", dict(p, target_layer=1))


def test_resume_requires_same_command_checkpoint_and_validated_output_hash(modules, tmp_path):
    _, runner = modules
    output = tmp_path / "mmlu.json"
    output.write_text(json.dumps({"meta": {}, "preds": [1]}))
    step = {"name": "mmlu", "command": ["python", "eval.py", "--checkpoint", "final.pt"],
            "outputs": [str(output)], "json_required_keys": {str(output): ["meta", "preds"]}}
    record = {"state": "complete", "checkpoint_sha256": "A", "command_sha256": runner.command_sha256(step),
              "output_sha256": {str(output): runner.digest(output)}}
    assert runner.completed(step, record, "A")
    assert not runner.completed(step, record, "B")
    assert not runner.completed(dict(step, command=["different-evaluator"]), record, "A")
    output.write_text(json.dumps({"meta": {}, "preds": [2]}))
    assert not runner.completed(step, record, "A")
    output.write_text("{partial")
    assert not runner.completed(step, record, "A")


def test_control_completion_requires_all_sae_spaces_and_both_strengths(modules, tmp_path):
    _, runner = modules
    output = tmp_path / "control.json"
    record = {"ref": {space: {} for space in ("h", "z", "s", "s1")},
              **{space + "_st_global_a" + alpha: {} for space in ("h", "z", "s", "s1") for alpha in ("0.4", "0.5")}}
    output.write_text(json.dumps({"meta": {}, "summary": {"done": True}, "concepts": {"0": record}}))
    step = {"name": "control_db14", "command": ["python", "--substrates", "h,z,s,s1"],
            "outputs": [str(output)], "json_required_keys": {str(output): ["meta", "concepts", "summary"]}}
    runner.validate_outputs(step)
    record.pop("s1_st_global_a0.5")
    output.write_text(json.dumps({"meta": {}, "summary": {}, "concepts": {"0": record}}))
    with pytest.raises(ValueError, match="space s1"):
        runner.validate_outputs(step)


def test_external_activation_sidecars_are_checked_without_recopying_big_arrays(modules, tmp_path):
    _, runner = modules
    original, staged = tmp_path / "original", tmp_path / "staged"
    original.mkdir(); staged.mkdir()
    for directory in (original, staged):
        (directory / "metadata.json").write_text('{"layer": 27}')
        (directory / "activations.npy").write_bytes(b"activation data")
    manifest = {"assets": {"activations_dir": str(original)}, "asset_identities": {
        str(original / "metadata.json"): {"sha256": runner.digest(original / "metadata.json")}}}
    runner.verify_staged_activations(manifest, staged)
    (staged / "metadata.json").write_text('{"layer": 33}')
    with pytest.raises(ValueError, match="sidecar hash"):
        runner.verify_staged_activations(manifest, staged)


def test_declared_staged_files_do_not_require_unrelated_transfer_receipts(modules, tmp_path):
    _, runner = modules
    original, staged = tmp_path / "original", tmp_path / "staged"
    original.mkdir(); staged.mkdir()
    (original / "layer_27.npy").write_bytes(b"same")
    (staged / "layer_27.npy").write_bytes(b"same")
    (original / "unrelated_audit.json").write_text("{}")
    runner.verify_staged_activations({"assets": {"activations_dir": str(original)},
                                     "activation_files": ["layer_27.npy"], "asset_identities": {}}, staged)


def test_cpu_preflight_help_ignores_missing_final_checkpoint_and_preserves_module_or_script(modules):
    _, runner = modules
    step = {"command": ["env", "CACHE=/cache", "/env/python", "-u", "-m", "geoae.evaluation.mmlu",
                         "--checkpoint", "/not-trained-yet.pt", "--n", "2000"]}
    assert runner.help_command(step, "/env/python") == ["/env/python", "-u", "-m", "geoae.evaluation.mmlu", "--help"]
    step["command"] = ["/env/python", "-u", "geoae/evaluation/alignment.py", "--out", "out.json"]
    assert runner.help_command(step, "/env/python") == ["/env/python", "-u", "geoae/evaluation/alignment.py", "--help"]


def test_gemma_summary_compares_all_spaces_on_the_same_documents(modules, tmp_path):
    plan, _ = modules
    summary = importlib.import_module("nonlinear_paper_summary")
    directory = tmp_path / "interventions"
    directory.mkdir()
    for dataset in plan.CONTROL_DATASETS:
        ref = {"doc_correct": [1, 1, 1, 1], "doc_conf": [0.8, 0.8, 0.8, 0.8]}
        record = {"eval_idx": {"tgt": [0, 1], "comp": [2, 3]}, "ref": {
            space: ref for space in ("h", "z", "s", "s1")}}
        for alpha in ("0.4", "0.5"):
            for space, correct in (("h", [0, 1, 1, 1]), ("z", [0, 0, 1, 1]),
                                   ("s", [0, 0, 0, 1]), ("s1", [1, 1, 1, 1])):
                record[f"{space}_st_global_a{alpha}"] = {"doc_correct": correct, "doc_conf": [0.2] * 4}
        data = {"meta": {"tp_all": True, "classes": ["concept"]}, "concepts": {"0": record}, "summary": {"complete": True}}
        (directory / f"range_intervention_{dataset}_dprime_hzss1_tp3.json").write_text(json.dumps(data))
    result = summary.write_control_summary(tmp_path, sae=True)
    pooled = result["groups"][-1]["alphas"]["0.5"]
    assert pooled["comparisons"]["z_minus_h"]["mean"] == 50
    assert pooled["comparisons"]["z_minus_s"]["mean"] == 50
    assert pooled["comparisons"]["z_minus_s1"]["mean"] == 100
    assert result["groups"][-1]["n_target_documents"] == 14
    path = directory / "range_intervention_db14_dprime_hzss1_tp3.json"
    data = json.loads(path.read_text()); data["meta"]["tp_all"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="three-way true positives"):
        summary.write_control_summary(tmp_path, sae=True)
