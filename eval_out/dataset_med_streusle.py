#!/usr/bin/env python3
"""Prepare official gold-labelled corpora for span-end NeuronLens evaluations.

No model representations or scores are used to choose labels or examples.
MedMentions uses the ST21pv PubTator corpus and official PMID train/test splits.
STREUSLE uses official train/test files, joined by review document, and scene
supersenses. Character offsets are zero-based, half-open in emitted document text.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import urllib.request
from typing import Any


SOURCES = {
    "medmentions": {
        "repo": "chanzuckerberg/MedMentions",
        "commit": "14578232bb6cdf290fea9dd7fee60a3e01ee6fd5",
        "files": [
            "st21pv/data/corpus_pubtator.txt.gz",
            "full/data/corpus_pubtator_pmids_trng.txt",
            "full/data/corpus_pubtator_pmids_test.txt",
            "full/data/corpus_pubtator_pmids_dev.txt",
            "README.md",
            "st21pv/ReadMe.md",
        ],
        "license": "CC0; see official README",
    },
    "streusle": {
        "repo": "nert-nlp/streusle",
        "commit": "8ba61fe4f216e7967500a862554a4fff79d25f5d",
        "files": [
            "train/streusle.ud_train.json",
            "test/streusle.ud_test.json",
            "README.md",
            "FORMAT.md",
            "LICENSE.txt",
        ],
        "license": "CC BY-SA 4.0; source text redistributed with permission",
    },
}

MED_NAMES = {
    "T005": "Virus", "T007": "Bacterium", "T017": "Anatomical Structure",
    "T022": "Body System", "T031": "Body Substance", "T033": "Finding",
    "T037": "Injury or Poisoning", "T038": "Biologic Function",
    "T058": "Health Care Activity", "T062": "Research Activity",
    "T074": "Medical Device", "T082": "Spatial Concept",
    "T091": "Biomedical Occupation or Discipline", "T092": "Organization",
    "T097": "Professional or Occupational Group", "T098": "Population Group",
    "T103": "Chemical", "T168": "Food", "T170": "Intellectual Product",
    "T201": "Clinical Attribute", "T204": "Eukaryote",
}

MED_PAIRS = [
    {"labels": ["T017", "T038"], "name": "anatomical structure versus biological function"},
    {"labels": ["T017", "T022"], "name": "anatomical structure versus body system"},
    {"labels": ["T007", "T204"], "name": "bacteria versus eukaryotic organisms"},
    {"labels": ["T103", "T168"], "name": "chemical versus food"},
]

STREUSLE_PAIRS = [
    {"labels": ["n.LOCATION", "n.TIME"], "name": "spatial versus temporal nouns"},
    {"labels": ["n.PERSON", "n.GROUP"], "name": "individual versus collective social entities"},
    {"labels": ["v.cognition", "v.communication"], "name": "cognitive versus communicative verbs"},
    {"labels": ["n.FOOD", "n.ARTIFACT"], "name": "food versus manufactured objects"},
]


def raw_path(root: pathlib.Path, dataset: str, source: str) -> pathlib.Path:
    return root / f"{dataset}_raw_{source.replace('/', '__')}"


def download(root: pathlib.Path, dataset: str) -> dict[str, Any]:
    src = SOURCES[dataset]
    provenance = {
        "repository": f"https://github.com/{src['repo']}",
        "commit": src["commit"], "license": src["license"], "files": [],
    }
    for file in src["files"]:
        url = f"https://raw.githubusercontent.com/{src['repo']}/{src['commit']}/{file}"
        path = raw_path(root, dataset, file)
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=90).read())
        data = path.read_bytes()
        provenance["files"].append({
            "source_path": file, "url": url, "local_path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        })
    (root / f"{dataset}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def medmentions(root: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = set(raw_path(root, "medmentions", "full/data/corpus_pubtator_pmids_trng.txt").read_text().split())
    test = set(raw_path(root, "medmentions", "full/data/corpus_pubtator_pmids_test.txt").read_text().split())
    dev = set(raw_path(root, "medmentions", "full/data/corpus_pubtator_pmids_dev.txt").read_text().split())
    assert not (train & test or train & dev or test & dev)
    source = gzip.decompress(raw_path(root, "medmentions", "st21pv/data/corpus_pubtator.txt.gz").read_bytes()).decode("utf-8")
    docs: list[dict[str, Any]] = []
    skipped = collections.Counter()
    for block in source.strip().split("\n\n"):
        lines = block.splitlines()
        pmid, kind, title = lines[0].split("|", 2)
        pmid2, kind2, abstract = lines[1].split("|", 2)
        assert pmid == pmid2 and kind == "t" and kind2 == "a"
        if pmid not in train and pmid not in test:
            assert pmid in dev
            skipped["dev_documents"] += 1
            continue
        text = title + " " + abstract
        spans = []
        labels_by_offset: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
        for line in lines[2:]:
            fields = line.split("\t")
            assert fields[0] == pmid and len(fields) == 6
            start, end = map(int, fields[1:3])
            mention, type_id, concept_id = fields[3:]
            assert text[start:end] == mention, (pmid, start, end, mention, text[start:end])
            if type_id not in MED_NAMES:
                skipped["unknown_or_multiple_type_mentions"] += 1
                continue
            labels_by_offset[start, end].add(type_id)
            spans.append({"start": start, "end": end, "label": type_id,
                          "text": mention, "concept_id": concept_id})
        clean_spans = []
        seen = set()
        for span in spans:
            offset = span["start"], span["end"]
            if len(labels_by_offset[offset]) != 1:
                skipped["ambiguous_exact_offset_mentions"] += 1
                continue
            key = *offset, span["label"]
            if key in seen:
                skipped["duplicate_mentions"] += 1
                continue
            seen.add(key)
            clean_spans.append(span)
        docs.append({"doc_id": f"medmentions:{pmid}", "split": "fit" if pmid in train else "eval",
                     "text": text, "spans": clean_spans})
    return docs, {
        "split_policy": "Official training PMIDs -> fit; official test PMIDs -> eval; dev excluded.",
        "text_policy": "Official PubTator title + one space + abstract, preserving original offsets.",
        "label_policy": "Original ST21pv semantic type ID; exact-offset multilabel ambiguities excluded.",
        "label_names": MED_NAMES, "proposed_pairs": MED_PAIRS, "skipped": dict(skipped),
    }


def streusle(root: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    docs = []
    skipped = collections.Counter()
    document_split: dict[str, str] = {}
    for official_split, split in [("train", "fit"), ("test", "eval")]:
        sentences = json.loads(raw_path(root, "streusle", f"{official_split}/streusle.ud_{official_split}.json").read_text())
        groups: dict[str, dict[str, Any]] = {}
        for sentence in sentences:
            # UD IDs are reviews-<review-id>-<sentence-id>.
            source_doc = sentence["sent_id"].rsplit("-", 1)[0]
            if source_doc in document_split:
                assert document_split[source_doc] == split
            document_split[source_doc] = split
            document = groups.setdefault(source_doc, {"doc_id": f"streusle:{source_doc}",
                                                      "split": split, "text": "", "spans": []})
            offset = len(document["text"]) + (1 if document["text"] else 0)
            document["text"] += ("\n" if document["text"] else "") + sentence["text"]
            positions = {}
            cursor = 0
            for token in sentence["toks"]:
                start = sentence["text"].find(token["word"], cursor)
                assert start >= 0, (sentence["sent_id"], token, sentence["text"], cursor)
                end = start + len(token["word"])
                positions[token["#"]] = start, end
                cursor = end
            for expression in list(sentence["swes"].values()) + list(sentence["smwes"].values()):
                label = expression.get("ss")
                if not label or not label.startswith(("n.", "v.", "p.")):
                    skipped["unlabelled_or_nonsemantic_expressions"] += 1
                    continue
                nums = expression["toknums"]
                if nums != list(range(min(nums), max(nums) + 1)):
                    skipped["noncontiguous_multiword_expressions"] += 1
                    continue
                start, end = positions[min(nums)][0], positions[max(nums)][1]
                document["spans"].append({
                    "start": offset + start, "end": offset + end, "label": label,
                    "text": sentence["text"][start:end], "lemma": expression["lexlemma"],
                    "single_word": len(nums) == 1, "sentence_id": sentence["sent_id"],
                    "coding_supersense": expression.get("ss2"),
                })
        docs.extend(groups.values())
    return docs, {
        "split_policy": "Official UD training reviews -> fit; official test reviews -> eval; dev excluded.",
        "text_policy": "Original sentence text, joined with one newline in source review order; exact token substrings aligned.",
        "label_policy": "Scene supersense ss; n.*, v.*, p.* retained. Special labels and noncontiguous MWEs excluded. Coding ss2 retained as metadata, never duplicated as a second label.",
        "proposed_pairs": STREUSLE_PAIRS, "skipped": dict(skipped),
        "preposition_note": "Unmerged preposition roles have too little official-test support for a pair with >=30 eval instances per label: p.Locus=78, p.Purpose=27, p.Time=24, p.Goal=18. Proposed pairs therefore use supported noun/verb semantic classes. This is not a same-word-controlled benchmark.",
    }


def write_dataset(root: pathlib.Path, name: str, documents: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    label_counts: dict[str, collections.Counter[str]] = {"fit": collections.Counter(), "eval": collections.Counter()}
    label_docs: dict[str, dict[str, set[str]]] = {"fit": collections.defaultdict(set), "eval": collections.defaultdict(set)}
    lexical_counts: dict[str, dict[str, collections.Counter[str]]] = {"fit": collections.defaultdict(collections.Counter), "eval": collections.defaultdict(collections.Counter)}
    document_counts = collections.Counter()
    output = root / f"{name}.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for doc in documents:
            split = doc["split"]
            document_counts[split] += 1
            for span in doc["spans"]:
                assert 0 <= span["start"] < span["end"] <= len(doc["text"])
                assert doc["text"][span["start"]:span["end"]] == span["text"]
                label_counts[split][span["label"]] += 1
                label_docs[split][span["label"]].add(doc["doc_id"])
                lexical_counts[split][span["label"]][span["text"].lower()] += 1
            stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
    assert not ({d["doc_id"] for d in documents if d["split"] == "fit"} &
                {d["doc_id"] for d in documents if d["split"] == "eval"})
    summary = {
        "dataset": name, "path": str(output),
        "schema": "JSONL documents: doc_id, split (fit/eval), text, spans[{start,end,label}]; offsets zero-based half-open, extra span fields are metadata.",
        **metadata, "documents": dict(document_counts),
        "label_support": {label: {
            "fit_spans": label_counts["fit"][label], "eval_spans": label_counts["eval"][label],
            "fit_documents": len(label_docs["fit"][label]), "eval_documents": len(label_docs["eval"][label]),
            "fit_top_forms": lexical_counts["fit"][label].most_common(8),
            "eval_top_forms": lexical_counts["eval"][label].most_common(8),
        } for label in sorted(label_counts["fit"] | label_counts["eval"])},
        "canonical_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    for pair in summary["proposed_pairs"]:
        pair["support"] = {label: summary["label_support"][label] for label in pair["labels"]}
    (root / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data/neuronlens_datasets"))
    parser.add_argument("--only", choices=["medmentions", "streusle"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ([args.only] if args.only else ["medmentions", "streusle"]):
        download(args.out_dir, name)
        documents, metadata = {"medmentions": medmentions, "streusle": streusle}[name](args.out_dir)
        summary = write_dataset(args.out_dir, name, documents, metadata)
        print(json.dumps({"dataset": name, "path": summary["path"], "documents": summary["documents"],
                          "pairs": [{"labels": p["labels"], "support": {label: {k: v for k, v in p["support"][label].items() if k.endswith("spans") or k.endswith("documents")} for label in p["labels"]}} for p in summary["proposed_pairs"]],
                          "skipped": summary["skipped"]}, indent=2))


if __name__ == "__main__":
    main()
