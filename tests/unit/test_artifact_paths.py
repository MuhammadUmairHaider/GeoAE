"""Artifact lookup must never redirect a working output into frozen evidence."""
from pathlib import Path

import pytest

from geoae import paths


@pytest.fixture
def stores(tmp_path, monkeypatch):
    checkout, archive = tmp_path / "checkout", tmp_path / "archive"
    checkout.mkdir()
    archive.mkdir()
    monkeypatch.setattr(paths, "PACKAGE_ROOT", checkout)
    monkeypatch.setattr(paths, "ARTIFACT_ROOT", archive)
    return checkout, archive


@pytest.mark.parametrize("destination", ["corpus/sampled_v1", "activations_sampled_10M", "runs/experiment", "checkpoints/run"])
def test_working_output_stays_local_when_archive_has_same_relative_path(stores, destination):
    checkout, archive = stores
    archived = archive / destination
    archived.mkdir(parents=True)
    sentinel = archived / "metadata.json"
    sentinel.write_bytes(b"frozen evidence\n")

    output = paths.resolve_path(destination)
    assert output == checkout / destination
    output.mkdir(parents=True)
    (output / "metadata.json").write_bytes(b"new work\n")
    assert sentinel.read_bytes() == b"frozen evidence\n"
    assert paths.resolve_input_path(destination) == output


def test_explicit_input_lookup_finds_archive_without_creating_working_files(stores):
    checkout, archive = stores
    relative = Path("results/saved.json")
    saved = archive / relative
    saved.parent.mkdir()
    saved.write_text('{"saved": true}\n')
    assert paths.resolve_input_path(relative) == saved
    assert not (checkout / relative).exists()
    assert paths.resolve_path(relative) == checkout / relative


def test_explicit_input_locations_are_not_substituted_by_an_archive(stores, tmp_path):
    checkout, archive = stores
    (archive / "shared").mkdir()
    absolute = tmp_path / "explicit" / "shared"
    assert paths.resolve_input_path(absolute) == absolute
    assert paths.resolve_path(absolute) == absolute
    base = tmp_path / "chosen_base"
    assert paths.resolve_input_path("shared", base=base) == base / "shared"
    assert paths.resolve_path("shared", base=base) == base / "shared"
    assert paths.resolve_input_path("missing") == checkout / "missing"
