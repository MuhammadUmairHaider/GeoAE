"""Prepare official Few-NERD supervised spans for the NeuronLens comparison.

Run with the project's datasets-enabled Python. The original data is tokenized,
so text is reconstructed by joining tokens with single spaces; span offsets are
exact in that reconstruction. The native fine tags have no BIO boundaries:
adjacent tokens with the same non-O label are treated as one mention.

Split units are source sentences, not Wikipedia articles. Official train/test
splits and exact-text duplicate exclusion are preserved, but named entities may
recur across splits. No AE activations or clustering results enter selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


PAIRS = [
    ["other-biologything", "other-livingthing"],
    ["other-disease", "other-medical"],
    ["organization-government/governmentagency", "organization-politicalparty"],
    ["person-politician", "person-athlete"],
]
CONTROL_LABELS = ["product-food", "other-chemicalthing", "other-astronomything", "event-election"]
LABELS = {label for pair in PAIRS for label in pair} | set(CONTROL_LABELS)


def canonical(row, names, source_split):
    tokens = row["tokens"]
    labels = [names[tag] for tag in row["fine_ner_tags"]]
    if len(tokens) != len(labels):
        raise ValueError("Few-NERD token/tag length mismatch")
    starts, ends, pos = [], [], 0
    for token in tokens:
        starts.append(pos)
        pos += len(token)
        ends.append(pos)
        pos += 1
    text = " ".join(tokens)
    spans = []
    i = 0
    while i < len(tokens):
        label = labels[i]
        end = i + 1
        while end < len(tokens) and labels[end] == label:
            end += 1
        if label in LABELS:
            spans.append({"start": starts[i], "end": ends[end - 1], "label": label,
                          "token_start": i, "token_end": end})
        i = end
    source_id = str(row["id"])
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"doc_id": f"fewnerd:{source_split}:{source_id}",
            "split": "fit" if source_split == "train" else "eval", "text": text,
            "spans": spans, "source_split": source_split, "source_id": source_id,
            "text_sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/neuronlens_datasets/fewnerd.jsonl")
    parser.add_argument("--fit_per_label", type=int, default=600)
    parser.add_argument("--eval_per_label", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    from datasets import load_dataset

    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    names = dataset["train"].features["fine_ner_tags"].feature.names
    missing = LABELS - set(names)
    if missing:
        raise ValueError(f"Unknown native fine labels: {sorted(missing)}")
    selected, seen, counts, source_rows, duplicate_skips = [], set(), {}, {}, {}
    for source_split, cap in (("train", args.fit_per_label), ("test", args.eval_per_label)):
        split_count = Counter()
        indices = list(range(len(dataset[source_split])))
        random.Random(args.seed + (0 if source_split == "train" else 1)).shuffle(indices)
        source_rows[source_split] = len(indices)
        duplicate_skips[source_split] = 0
        for row_idx in indices:
            rec = canonical(dataset[source_split][row_idx], names, source_split)
            if not rec["spans"]:
                continue
            if rec["text_sha256"] in seen:
                duplicate_skips[source_split] += 1
                continue
            keep = []
            for span in rec["spans"]:
                if split_count[span["label"]] < cap:
                    keep.append(span)
                    split_count[span["label"]] += 1
            if keep:
                rec["spans"] = keep
                seen.add(rec["text_sha256"])
                selected.append(rec)
            if all(split_count[label] >= cap for label in LABELS):
                break
        counts["fit" if source_split == "train" else "eval"] = dict(sorted(split_count.items()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"dataset": "DFKI-SLT/few-nerd", "configuration": "supervised",
                "source": "https://huggingface.co/datasets/DFKI-SLT/few-nerd",
                "official_source": "https://github.com/thunlp/Few-NERD",
                "license": "CC-BY-SA-4.0", "args": vars(args), "pairs": PAIRS,
                "control_labels": CONTROL_LABELS, "selected_documents": len(selected),
                "selected_counts": counts, "source_rows": source_rows,
                "duplicate_skips": duplicate_skips,
                "fingerprints": {split: dataset[split]._fingerprint for split in ("train", "test")},
                "sampling": "Seeded random sentence order, retain spans up to cap per selected native label.",
                "split_limitation": "Official sentence splits; no article IDs supplied, entity names can overlap.",
                "text_reconstruction": "Original tokens joined with single spaces; exact offsets in reconstructed text.",
                "span_boundary_limitation": "Native fine tags have no BIO boundaries; adjacent identical tags merged.",
                "annotation_selection": "A priori native categories, no activation or AE concept selection."}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(out), "documents": len(selected), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
