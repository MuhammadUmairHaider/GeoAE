"""The matched AE comparisons must use the user's complete ten-source mix."""
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from geoae.config import Config
from experiments.depth import dataset, evaluate


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("arm", evaluate.CANONICAL_ARMS)
def test_every_profile_uses_the_ten_source_recipe_and_matching_training_dump(arm):
    profile = evaluate.PROFILES[arm]
    extraction = ROOT / profile["extract_config"]
    report = dataset.validate_extraction_config(extraction)
    training = Config.from_yaml(ROOT / profile["train_config"])
    assert report["source_mix"] == {"edu_web": 35, "web": 15, "wiki": 15, "pile": 10,
                                    "code": 8, "math": 5, "structured": 5,
                                    "wiki_de": 3, "wiki_fr": 2, "wiki_es": 2}
    assert report["source_fingerprint"] == dataset.SOURCE_FINGERPRINT
    assert report["activations_dir"] == training.data.activations_dir == profile["activations_dir"]
    assert report["layers"] == [training.data.target_layer] == [profile["layer"]]
    assert report["model"] == training.extraction.model_name == profile["model_name"]
    assert "struct5" in profile["extract_config"]
    assert bool(training.model.token_bias) == (profile["mode"] == "bypass")


def test_user_mix_is_immutable_and_report_mutation_cannot_change_fingerprint():
    with pytest.raises(TypeError):
        dataset.USER_MIX["edu_web"] = 40
    path = ROOT / evaluate.PROFILES["llama_mid_plain"]["extract_config"]
    original = dataset.validate_extraction_config(path)
    altered = dataset.validate_extraction_config(path)
    altered["source_mix"]["edu_web"] = 40
    altered["source_specs"][0]["weight"] = 999
    assert dataset.validate_extraction_config(path) == original


def guarded_stats():
    return {"guard": {"ravel_entities": 2291, "ioi_templates": 28},
            "counterfact": {"raw": 32, "dropped_entity": 8, "dropped_template": 0, "kept": 24},
            "nq_open": {"raw": 32, "dropped_entity": 8, "dropped_template": 0, "kept": 24},
            "docs": {"fact_lists": 3, "qa_lists": 3}}


@pytest.fixture
def artifacts(tmp_path):
    original = Config.from_yaml(ROOT / evaluate.PROFILES["llama_mid_plain"]["extract_config"])
    ex = original.extraction
    ex.activations_dir = str(tmp_path / "dump")
    ex.corpus_dir = str(tmp_path / "corpus")
    ex.n_tokens = 1000
    ex.hidden_size = 3
    cfg = {"extraction": vars(ex)}
    config = tmp_path / "extract.yaml"
    config.write_text(yaml.safe_dump(cfg))
    corpus = Path(ex.corpus_dir)
    corpus.mkdir()
    manifest = {"model_tokenizer": ex.model_name, "seed": 42, "n_tokens": ex.n_tokens,
                "context_len": ex.context_len, "positions_per_doc": ex.positions_per_doc,
                "skip_leading": ex.skip_leading, "margin": 1.3, "sources": []}
    sources = []
    row_ids = []
    for index, source in enumerate(ex.sources):
        domain = source["domain"]
        rows = source["weight"] * 10
        record = {"domain": domain, "file": f"{domain}.jsonl.gz",
                  "docs": math.ceil(rows / ex.positions_per_doc), "est_rows": rows,
                  "spec": deepcopy(source), "stats": guarded_stats() if domain == "structured" else {}}
        manifest["sources"].append(record)
        with gzip.open(corpus / record["file"], "wt") as stream:
            stream.write(json.dumps({"text": "An actual frozen document."}) + "\n")
        sources.append({**deepcopy(source), "rows": rows, "docs": record["docs"], "stats": {}})
        row_ids.extend([index] * rows)
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    dump = Path(ex.activations_dir)
    dump.mkdir()
    meta = {"model": ex.model_name, "layers": ex.layers, "mode": "sampled", "n_tokens": ex.n_tokens,
            "hidden_size": ex.hidden_size, "dtype": ex.dtype, "context_len": ex.context_len,
            "positions_per_doc": ex.positions_per_doc, "skip_leading": ex.skip_leading, "seed": 42,
            "sources": sources, "corpus_manifest": deepcopy(manifest)}
    (dump / "meta.json").write_text(json.dumps(meta))
    np.save(dump / "layer_14.npy", np.zeros((ex.n_tokens, ex.hidden_size), dtype=ex.dtype))
    for name in ("rows_doc", "rows_pos", "rows_tok", "rows_norm"):
        np.save(dump / f"{name}.npy", np.zeros(ex.n_tokens, dtype=np.int32))
    np.save(dump / "rows_src.npy", np.array(row_ids, dtype=np.uint8))
    np.save(dump / "docs.npy", np.zeros(10, dtype=[("src", np.uint8)]))
    return config, cfg, corpus, manifest, dump, meta


@pytest.mark.parametrize("mutation", ["nine_sources", "domain_type", "weight", "hf_name", "hf_config", "field",
                                      "pile_exclude", "structured_kind", "items_per_doc"])
def test_config_guard_rejects_wrong_source_recipe(artifacts, mutation):
    config, cfg, *_ = artifacts
    sources = cfg["extraction"]["sources"]
    if mutation == "nine_sources":
        sources.pop()
        sources[0]["weight"] = 40
    elif mutation == "domain_type":
        sources[0]["domain"] = ["edu_web"]
    elif mutation == "weight":
        sources[0]["weight"] = 40
    elif mutation == "hf_name":
        sources[0]["name"] = "other/fineweb"
    elif mutation == "hf_config":
        sources[0]["config"] = "default"
    elif mutation == "field":
        sources[4]["field"] = "text"
    elif mutation == "pile_exclude":
        sources[3]["exclude"]["values"].remove("Github")
    elif mutation == "structured_kind":
        sources[-1]["kind"] = "hf"
    elif mutation == "items_per_doc":
        sources[-1]["items_per_doc"] = 4
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        dataset.validate_extraction_config(config)


def test_corpus_reuses_old_source_weights_at_current_quota_without_margin(artifacts):
    config, _, corpus, manifest, *_ = artifacts
    manifest["sources"][0]["spec"]["weight"] = 40
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    result = dataset.check_corpus(config)
    assert result["dataset_provenance"]["source_mix"]["edu_web"] == 35
    assert result["sources"][0]["est_rows"] == 350  # quota, below the builder's 455-row margin
    assert len(result["dataset_provenance"]["manifest_sha256"]) == 64


@pytest.mark.parametrize("mutation", ["nine_sources", "domain_type", "missing_file", "wrong_spec", "below_quota",
                                      "no_docs", "different_seed", "different_tokenizer", "guard",
                                      "filter_counts", "packing", "empty_file"])
def test_corpus_guard_rejects_incomplete_or_wrong_provenance(artifacts, mutation):
    config, _, corpus, manifest, *_ = artifacts
    structured = manifest["sources"][-1]
    if mutation == "nine_sources":
        manifest["sources"].pop()
    elif mutation == "domain_type":
        structured["domain"] = ["structured"]
    elif mutation == "missing_file":
        (corpus / "structured.jsonl.gz").unlink()
    elif mutation == "wrong_spec":
        manifest["sources"][0]["spec"]["name"] = "other/data"
    elif mutation == "below_quota":
        manifest["sources"][0]["est_rows"] = 349
    elif mutation == "no_docs":
        structured["docs"] = 0
    elif mutation == "different_seed":
        manifest["seed"] = 43
    elif mutation == "different_tokenizer":
        manifest["model_tokenizer"] = "google/gemma-3-4b-pt"
    elif mutation == "guard":
        structured["stats"].pop("guard")
    elif mutation == "filter_counts":
        structured["stats"]["counterfact"]["kept"] = 25
    elif mutation == "packing":
        structured["stats"]["docs"]["fact_lists"] = 2
    elif mutation == "empty_file":
        (corpus / "structured.jsonl.gz").write_bytes(b"")
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        dataset.check_corpus(config)


def test_dump_checks_complete_arrays_and_records_ten_source_provenance(artifacts):
    config, *_ = artifacts
    result = dataset.check_dump(config)
    provenance = result["dataset_provenance"]
    assert provenance["source_fingerprint"] == dataset.SOURCE_FINGERPRINT
    assert provenance["training_rows"] == 950
    assert provenance["source_rows"] == {domain: weight * 10 for domain, weight in dataset.USER_MIX.items()}


def test_dump_accepts_weighted_scheduler_rounding_larger_than_one_document(artifacts):
    config, _, _, _, dump, meta = artifacts
    # The real Gemma extraction has an edu_web deficit of 73 rows. The
    # scheduler balances relative quota completion, so small sources can
    # overshoot and leave a larger absolute deficit in a heavier source.
    ids = np.load(dump / "rows_src.npy")
    ids[:37] = 7  # wiki_fr
    ids[37:73] = 8  # wiki_es
    np.save(dump / "rows_src.npy", ids)
    meta["sources"][0]["rows"] -= 73
    meta["sources"][7]["rows"] += 37
    meta["sources"][8]["rows"] += 36
    (dump / "meta.json").write_text(json.dumps(meta))
    assert dataset.check_dump(config)["dataset_provenance"]["source_rows"]["edu_web"] == 277


@pytest.mark.parametrize("mutation", ["old_nine_sources", "wrong_hf", "wrong_pile", "old_corpus", "row_drift",
                                      "metadata_count", "source_ids", "noninteger_ids", "short_array"])
def test_dump_guard_rejects_old_mix_or_actual_row_provenance_drift(artifacts, mutation):
    config, _, _, _, dump, meta = artifacts
    if mutation == "old_nine_sources":
        meta["sources"].pop()
        meta["sources"][0]["weight"] = 40
    elif mutation == "wrong_hf":
        meta["sources"][0]["name"] = "different/fineweb"
    elif mutation == "wrong_pile":
        meta["sources"][3]["exclude"]["values"].pop()
    elif mutation == "old_corpus":
        meta["corpus_manifest"]["sources"].pop()
    elif mutation == "row_drift":
        ids = np.load(dump / "rows_src.npy")
        ids[:80] = 7  # More than one 64-row group over the smallest quota.
        np.save(dump / "rows_src.npy", ids)
        meta["sources"][0]["rows"] -= 80
        meta["sources"][7]["rows"] += 80
    elif mutation == "metadata_count":
        meta["sources"][0]["rows"] += 1
    elif mutation == "source_ids":
        ids = np.load(dump / "rows_src.npy")
        ids[0] = 10
        np.save(dump / "rows_src.npy", ids)
    elif mutation == "noninteger_ids":
        np.save(dump / "rows_src.npy", np.load(dump / "rows_src.npy").astype(np.float32))
    elif mutation == "short_array":
        np.save(dump / "layer_14.npy", np.zeros((999, 3), dtype=np.float16))
    (dump / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        dataset.check_dump(config)
