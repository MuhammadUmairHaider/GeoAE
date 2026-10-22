"""Recipe aliases preserve complete configurations and snapshot independence."""
import hashlib
import json
from pathlib import Path

import pytest

from geoae.config import Config
from geoae.paths import config_path

ROOT = Path(__file__).resolve().parents[2]


def effective_hash(path):
    payload = json.dumps(Config.from_yaml(path).to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_all_recorded_configuration_aliases_resolve_to_one_existing_owner():
    ledger = json.loads((ROOT / "experiments/config_paths.json").read_text())
    for old, owner in ledger["paths"].items():
        assert config_path(old).resolve() == (ROOT / owner).resolve()
        assert config_path(old).is_file()


def test_canonicalization_retains_recorded_complete_effective_configs_and_bytes():
    receipt = json.loads((ROOT / "docs/history/audits/config_consolidation.json").read_text())
    assert sum(record["action"] == "rename" for record in receipt["records"]) == 6
    for record in receipt["records"]:
        path = ROOT / record["canonical"]
        assert not (ROOT / record["previous"]).exists()
        assert config_path(record["previous"]).resolve() == path.resolve()
        assert effective_hash(path) == record["effective_config_sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["after_sha256"]
        assert record["effective_config_equal"]
        if record["action"] == "rename":
            assert record["before_sha256"] == record["after_sha256"]


@pytest.mark.parametrize("absolute", [False, True])
def test_explicit_snapshot_base_uses_its_own_map_and_ignores_working_directory(tmp_path, monkeypatch, absolute):
    source, elsewhere = tmp_path / "snapshot", tmp_path / "elsewhere"
    owner = source / "experiments/paper/configs/llama.yaml"
    owner.parent.mkdir(parents=True)
    owner.write_text("model: {hidden_size: 16}\n")
    old = Path("configs/base/historical.yaml")
    (source / "experiments/config_paths.json").write_text(json.dumps({"paths": {str(old): str(owner.relative_to(source))}}))
    unrelated = elsewhere / old
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("unrelated working-directory file")
    monkeypatch.chdir(elsewhere)
    assert config_path(source / old if absolute else old, base=source) == owner
    assert config_path(old, base=source / "missing") == source / "missing" / old


def test_all_main_paper_names_are_available_with_the_recorded_bypass_modes():
    for name in ("llama", "gemma", "qwen"):
        bypass = Config.from_yaml(ROOT / "experiments/paper/configs" / f"{name}.yaml")
        plain = Config.from_yaml(ROOT / "experiments/paper/configs" / f"{name}_no_bypass.yaml")
        assert bypass.model.token_bias and bypass.model.token_bias_mode == "bypass"
        assert not plain.model.token_bias
        assert bypass.model.hidden_size == plain.model.hidden_size
        assert bypass.model.latent_dim == plain.model.latent_dim
