"""Verify the requested ten-source corpus before either AE variant is trained.

The source recipe is fixed independently of model, layer, and token bypass.
Imports of Config and the array validator are delayed until their checks run.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
USER_MIX = MappingProxyType({
    "edu_web": 35, "web": 15, "wiki": 15, "pile": 10, "code": 8,
    "math": 5, "structured": 5, "wiki_de": 3, "wiki_fr": 2, "wiki_es": 2,
})
_HF_SPECS = {
    "edu_web": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "web": ("HuggingFaceFW/fineweb", "sample-10BT", "text"),
    "wiki": ("wikimedia/wikipedia", "20231101.en", "text"),
    "pile": ("monology/pile-uncopyrighted", None, "text"),
    "code": ("codeparrot/codeparrot-clean", None, "content"),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus", "text"),
    "wiki_de": ("wikimedia/wikipedia", "20231101.de", "text"),
    "wiki_fr": ("wikimedia/wikipedia", "20231101.fr", "text"),
    "wiki_es": ("wikimedia/wikipedia", "20231101.es", "text"),
}
_PILE_EXCLUSIONS = frozenset({"Pile-CC", "Github", "DM Mathematics", "Wikipedia (en)"})
_SOURCE_KEYS = frozenset({"domain", "weight", "kind", "name", "config", "split", "field",
                          "exclude", "shuffle_buffer", "items_per_doc"})
_DUMP_SOURCE_KEYS = frozenset({"stats", "docs", "rows", "skipped_short"})


def _canonical_source(source: dict, *, metadata: bool = False) -> dict:
    if not isinstance(source, dict):
        raise ValueError("Each source must be a mapping")
    unknown = set(source) - _SOURCE_KEYS - (_DUMP_SOURCE_KEYS if metadata else set())
    if unknown:
        raise ValueError(f"Source {source.get('domain')!r} has unexpected fields: {sorted(map(str, unknown))}")
    domain = source.get("domain")
    if not isinstance(domain, str) or domain not in USER_MIX:
        raise ValueError(f"Unexpected source domain {domain!r}; the ten-source mix is required")
    value = {"domain": domain, "weight": source.get("weight"),
             "kind": source.get("kind", "hf")}
    if domain == "structured":
        value["items_per_doc"] = source.get("items_per_doc")
        if any(key in source for key in ("name", "config", "split", "field", "exclude")):
            raise ValueError("Structured source must use the guarded structured generator")
    else:
        value.update(name=source.get("name"), config=source.get("config"),
                     split=source.get("split", "train"), field=source.get("field", "text"))
        if "items_per_doc" in source:
            raise ValueError(f"HF source {domain!r} cannot define items_per_doc")
        if source.get("exclude") is not None:
            exclude = source["exclude"]
            if not isinstance(exclude, dict) or set(exclude) != {"key", "values"}:
                raise ValueError(f"Invalid exclusion specification for {domain}")
            values = exclude["values"]
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise ValueError(f"Invalid exclusion values for {domain}")
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate exclusion values for {domain}")
            value["exclude"] = {"key": exclude["key"], "values": sorted(values)}
    if "shuffle_buffer" in source:
        _integer(source["shuffle_buffer"], f"{domain}.shuffle_buffer", positive=True)
    return value


def _expected_specs() -> list[dict]:
    result = []
    for domain, weight in USER_MIX.items():
        source = {"domain": domain, "weight": weight}
        if domain == "structured":
            source.update(kind="structured", items_per_doc=8)
        else:
            name, config, field = _HF_SPECS[domain]
            source.update(name=name, config=config, field=field)
            if domain == "pile":
                source["exclude"] = {"key": "meta.pile_set_name", "values": sorted(_PILE_EXCLUSIONS)}
        result.append(_canonical_source(source))
    return sorted(result, key=lambda source: source["domain"])


def _integer(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


_SOURCE_RECIPE_JSON = _json(_expected_specs())
SOURCE_FINGERPRINT = hashlib.sha256(_SOURCE_RECIPE_JSON.encode()).hexdigest()


def _validate_sources(sources, *, metadata: bool = False) -> list[dict]:
    if not isinstance(sources, list) or len(sources) != len(USER_MIX):
        raise ValueError("Exactly ten source specifications are required, including structured 5%")
    normalized = [_canonical_source(source, metadata=metadata) for source in sources]
    domains = [source["domain"] for source in normalized]
    if len(set(domains)) != len(domains) or set(domains) != set(USER_MIX):
        raise ValueError("The ten source domains must each occur exactly once")
    normalized.sort(key=lambda source: source["domain"])
    expected = json.loads(_SOURCE_RECIPE_JSON)
    for actual, wanted in zip(normalized, expected):
        if actual != wanted:
            raise ValueError(f"Source {actual['domain']!r} does not match the requested ten-source recipe: "
                             f"{actual!r}; expected {wanted!r}")
    return normalized


def _load_config(config_path):
    from geoae.config import Config
    path = Path(config_path)
    return Config.from_yaml(path if path.is_absolute() else ROOT / path)


def _provenance(ex, sources: list[dict]) -> dict:
    return {"source_mix": dict(USER_MIX), "source_specs": sources,
            "source_fingerprint": SOURCE_FINGERPRINT, "model": ex.model_name,
            "n_tokens": ex.n_tokens, "training_rows": int(ex.n_tokens * 0.95), "corpus_dir": ex.corpus_dir,
            "activations_dir": ex.activations_dir, "layers": list(ex.layers),
            "sampling": {"mode": ex.mode, "seed": 42, "context_len": ex.context_len,
                         "positions_per_doc": ex.positions_per_doc, "skip_leading": ex.skip_leading}}


def validate_extraction_config(config_path: str | Path) -> dict:
    """Return JSON-ready provenance after checking the immutable source recipe."""
    cfg = _load_config(config_path)
    ex = cfg.extraction
    sources = _validate_sources(ex.sources)
    if ex.mode != "sampled":
        raise ValueError("The ten-source campaign requires sampled extraction")
    _integer(ex.n_tokens, "extraction.n_tokens", positive=True)
    _integer(ex.context_len, "extraction.context_len", positive=True)
    _integer(ex.positions_per_doc, "extraction.positions_per_doc", positive=True)
    _integer(ex.skip_leading, "extraction.skip_leading")
    if ex.context_len <= ex.skip_leading or not ex.corpus_dir:
        raise ValueError("Sampled extraction requires a corpus and usable token positions")
    if cfg.train.seed != 42:
        raise ValueError("The paired dataset recipe requires seed 42")
    return _provenance(ex, sources)


def _check_structured_stats(stats, *, documents: int) -> None:
    if not isinstance(stats, dict) or not isinstance(stats.get("guard"), dict):
        raise ValueError("Structured corpus lacks benchmark leakage-guard statistics")
    for key in ("ravel_entities", "ioi_templates"):
        _integer(stats["guard"].get(key), f"structured.guard.{key}", positive=True)
    packed = stats.get("docs")
    if not isinstance(packed, dict):
        raise ValueError("Structured corpus lacks packed document statistics")
    for source, document_key in (("counterfact", "fact_lists"), ("nq_open", "qa_lists")):
        counts = stats.get(source)
        if not isinstance(counts, dict):
            raise ValueError(f"Structured corpus lacks {source} guard statistics")
        for key in ("raw", "dropped_entity", "dropped_template", "kept"):
            _integer(counts.get(key), f"structured.{source}.{key}", positive=key in ("raw", "kept"))
        if counts["raw"] != counts["kept"] + counts["dropped_entity"] + counts["dropped_template"]:
            raise ValueError(f"Inconsistent structured {source} filtering statistics")
        if _integer(packed.get(document_key), f"structured.docs.{document_key}", positive=True) != counts["kept"] // 8:
            raise ValueError(f"Inconsistent structured {source} packing statistics")
    if documents > packed["fact_lists"] + packed["qa_lists"]:
        raise ValueError("Structured corpus document count exceeds its guarded packed documents")


def _check_manifest(manifest, ex, *, directory: Path | None = None) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Corpus manifest must be a mapping")
    expected = {"model_tokenizer": ex.model_name, "seed": 42, "context_len": ex.context_len,
                "positions_per_doc": ex.positions_per_doc, "skip_leading": ex.skip_leading}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Corpus {key}={manifest.get(key)!r}, expected {value!r}")
    if _integer(manifest.get("n_tokens"), "corpus.n_tokens", positive=True) < ex.n_tokens:
        raise ValueError("Corpus was built for fewer tokens than the campaign requires")
    records = manifest.get("sources")
    if not isinstance(records, list) or len(records) != len(USER_MIX):
        raise ValueError("Corpus manifest must contain all ten source files")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Corpus source records must be mappings")
    if any(not isinstance(record.get("domain"), str) for record in records):
        raise ValueError("Corpus source domains must be strings")
    by_domain = {record.get("domain"): record for record in records}
    if len(by_domain) != len(records) or set(by_domain) != set(USER_MIX):
        raise ValueError("Corpus source domains do not match the ten-source campaign")
    for source in ex.sources:
        domain = source["domain"]
        record = by_domain[domain]
        spec = record.get("spec")
        if isinstance(spec, dict):
            _canonical_source(spec)
        if not isinstance(spec, dict) or _json({key: value for key, value in spec.items() if key != "weight"}) != _json(
                {key: value for key, value in source.items() if key != "weight"}):
            raise ValueError(f"Corpus source {domain!r} has a different source identity")
        documents = _integer(record.get("docs"), f"{domain}.docs", positive=True)
        rows = _integer(record.get("est_rows"), f"{domain}.est_rows", positive=True)
        quota = source["weight"] / sum(USER_MIX.values()) * ex.n_tokens
        if rows < quota:
            raise ValueError(f"Corpus source {domain!r} has {rows} estimated rows, below quota {quota:g}")
        if rows > documents * ex.positions_per_doc:
            raise ValueError(f"Corpus source {domain!r} row yield exceeds its document capacity")
        if record.get("file") != f"{domain}.jsonl.gz":
            raise ValueError(f"Corpus source {domain!r} does not name its required source file")
        if domain == "structured":
            _check_structured_stats(record.get("stats"), documents=documents)
        if directory is not None:
            path = directory / record["file"]
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Corpus source file is missing or empty: {path}")
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                first = json.loads(stream.readline())
            if not isinstance(first, dict) or not isinstance(first.get("text"), str) or not first["text"]:
                raise ValueError(f"Corpus source file has no usable first document: {path}")


def check_corpus(config_path: str | Path) -> dict:
    """Verify corpus reuse identity, all ten files, quotas, and structured guards.

    Manifest source weights may reflect an earlier corpus build. Corpus reuse,
    like build_corpus, matches source identity without weight and checks the
    current extraction quota rather than requiring the builder's margin.
    """
    provenance = validate_extraction_config(config_path)
    ex = _load_config(config_path).extraction
    directory = Path(ex.corpus_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    _check_manifest(manifest, ex, directory=directory)
    provenance["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return {**manifest, "dataset_provenance": provenance}


def check_dump(config_path: str | Path) -> dict:
    """Verify completed arrays and the full ten-source extraction provenance."""
    provenance = validate_extraction_config(config_path)
    from scripts.gpt2 import lifecycle
    config = Path(config_path)
    meta = lifecycle.check_extraction(config if config.is_absolute() else ROOT / config)
    ex = _load_config(config_path).extraction
    _validate_sources(meta.get("sources"), metadata=True)
    _check_manifest(meta.get("corpus_manifest"), ex)
    import numpy as np
    directory = Path(ex.activations_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    row_sources = np.load(directory / "rows_src.npy", mmap_mode="r")
    if not np.issubdtype(row_sources.dtype, np.integer):
        raise ValueError("Dump row source IDs must be integers")
    actual_counts = np.zeros(len(USER_MIX), dtype=np.int64)
    for start in range(0, len(row_sources), 1_000_000):
        chunk = np.asarray(row_sources[start:start + 1_000_000], dtype=np.int64)
        if np.any(chunk < 0) or np.any(chunk >= len(USER_MIX)):
            raise ValueError("Dump row source IDs lie outside the ten-source recipe")
        actual_counts += np.bincount(chunk, minlength=len(USER_MIX))
    total = 0
    counts = {}
    for index, source in enumerate(meta["sources"]):
        domain = source["domain"]
        rows = _integer(source.get("rows"), f"dump.{domain}.rows", positive=True)
        _integer(source.get("docs"), f"dump.{domain}.docs", positive=True)
        if rows != int(actual_counts[index]):
            raise ValueError(f"Dump source {domain!r} metadata rows disagree with rows_src.npy")
        quota = USER_MIX[domain] / sum(USER_MIX.values()) * ex.n_tokens
        # The scheduler balances the fraction of each quota still unfilled.
        # One document from the smallest source advances that fraction by
        # positions_per_doc / smallest_quota; larger sources can consequently
        # miss their own quota by more than a single document's row count.
        tolerance = ex.positions_per_doc * max(1.0, USER_MIX[domain] / min(USER_MIX.values()))
        if abs(rows - quota) > tolerance:
            raise ValueError(f"Dump source {domain!r} rows {rows} drift from requested quota {quota:g}")
        total += rows
        counts[domain] = rows
    if total != ex.n_tokens:
        raise ValueError("Dump source row counts do not sum to the completed extraction")
    provenance["source_rows"] = counts
    provenance["corpus_manifest_sha256"] = hashlib.sha256(_json(meta["corpus_manifest"]).encode()).hexdigest()
    return {**meta, "dataset_provenance": provenance}
