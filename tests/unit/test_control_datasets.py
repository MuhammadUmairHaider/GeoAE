import json

import geoae.evaluation.cohorts as shared


def test_control_cfg_loads_from_env_dir(tmp_path, monkeypatch):
    d = tmp_path / "emotion6"
    d.mkdir()
    cfg = {"classes": ["sadness", "joy"], "dataset_name": "local:emotion6", "dataset_split": "train",
           "text_column": "text", "label_column": "label", "local_parquet": "data.parquet",
           "prompt_head": 'Pick one.\n\n{{ I am sad:sadness}}\n\n{{"{}":', "max_new": 6}
    (d / "cfg.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("GEOAE_CONTROL_DATASETS_DIR", str(tmp_path))
    shared.DATASET_CONFIGS.pop("emotion6", None)
    try:
        got = shared.set_dataset("emotion6")
        assert shared.CLASSES == ["sadness", "joy"]
        assert got["local_parquet"] == str(d / "data.parquet")
        assert shared.build_prompt("hello world").endswith('{"hello world":')
    finally:
        shared.DATASET_CONFIGS.pop("emotion6", None)


def test_db14_unchanged():
    shared.set_dataset("db14")
    assert len(shared.CLASSES) == 14
