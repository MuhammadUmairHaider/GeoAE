"""Command routing forwards tool arguments without loading or running models."""
import subprocess
import sys
from pathlib import Path

import pytest

from geoae import cli


@pytest.mark.parametrize("arguments,module", [
    (["train", "--config", "a recipe.yaml", "--no_wandb"], "geoae.training.mse"),
    (["extract-sampled", "--help"], "geoae.extraction.sampled"),
    (["evaluate", "control", "--dataset", "db14", "--alphas", "-0.5,1"], "geoae.evaluation.control"),
    (["analyze", "closest-tokens", "--help"], "geoae.analysis.closest_tokens"),
    (["baseline", "matched-train", "--help"], "geoae.baselines.matched_topk.train"),
])
def test_dispatch_forwards_tool_arguments_and_restores_argv(monkeypatch, arguments, module):
    original = sys.argv
    observed = []

    def run(target, **kwargs):
        observed.append((target, kwargs, sys.argv.copy()))

    monkeypatch.setattr(cli.runpy, "run_module", run)
    cli.main(arguments)
    count = 2 if arguments[0] in cli.GROUPS else 1
    assert observed == [(module, {"run_name": "__main__", "alter_sys": True},
                         ["geoae " + " ".join(arguments[:count]), *arguments[count:]])]
    assert sys.argv is original


def test_tool_exit_is_preserved_and_argv_restored(monkeypatch):
    original = sys.argv

    def run(*args, **kwargs):
        raise SystemExit(7)

    monkeypatch.setattr(cli.runpy, "run_module", run)
    with pytest.raises(SystemExit) as error:
        cli.main(["train", "--help"])
    assert error.value.code == 7 and sys.argv is original


@pytest.mark.parametrize("arguments", [[], ["--help"], ["evaluate", "--help"], ["commands"]])
def test_inventory_and_dispatcher_help_do_not_execute_tools(monkeypatch, capsys, arguments):
    def forbidden(*args, **kwargs):
        raise AssertionError("Listing commands must not execute a tool")

    monkeypatch.setattr(cli.runpy, "run_module", forbidden)
    try:
        cli.main(arguments)
    except SystemExit as error:
        assert error.code == 0
    output = capsys.readouterr().out
    assert "geoae" in output
    if arguments == ["commands"]:
        assert "geoae evaluate alignment -> python -m geoae.evaluation.alignment" in output
        assert "geoae train-kl -> python -m geoae.training.kl_cached" in output
        assert "geoae corpus -> python -m geoae.extraction.corpus" in output


def test_invalid_tool_fails_before_execution(monkeypatch):
    monkeypatch.setattr(cli.runpy, "run_module", lambda *a, **k: pytest.fail("Unexpected execution"))
    with pytest.raises(SystemExit) as error:
        cli.main(["evaluate", "unknown-tool"])
    assert error.value.code == 2


def test_registered_routes_resolve_to_shipped_modules():
    root = Path(__file__).resolve().parents[2]
    missing = [module for module in cli.command_routes().values()
               if not (root / Path(*module.split("."))).with_suffix(".py").is_file()]
    assert not missing


def test_sampled_entry_selects_sampled_mode(monkeypatch):
    from geoae.extraction import prefix, sampled

    observed = []
    monkeypatch.setattr(prefix, "main", lambda **kwargs: observed.append(kwargs))
    sampled.main()
    assert observed == [{"mode": "sampled"}]


def test_sampled_override_reaches_extractor_even_with_prefix_recipe(tmp_path, monkeypatch):
    from geoae.extraction import prefix, sampled

    recipe = tmp_path / "prefix.yaml"
    recipe.write_text("extraction:\n  mode: prefix\n  n_tokens: 7\n  layers: [2]\n  target_layer: 2\n")
    observed = []
    monkeypatch.setattr(sampled, "extract_sampled", lambda **kwargs: observed.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["geoae extract-sampled", "--config", str(recipe)])
    prefix.main(mode="sampled")
    assert len(observed) == 1 and observed[0]["n_tokens"] == 7 and observed[0]["layers"] == [2]


def test_command_inventory_is_lightweight_in_fresh_interpreter():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from geoae.cli import main; main(['commands']); "
         "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
