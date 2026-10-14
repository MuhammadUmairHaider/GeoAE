#!/usr/bin/env python3
"""Prepare independent labelled Py150 and MeasEval span corpora.

Only standard-library dependencies are used. Source archives are downloaded to
data/neuronlens_datasets/source. No model or representation is inspected.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tokenize
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/neuronlens_datasets"
SOURCE = OUTPUT / "source"
MEASEVAL_REV = "1fa738b6bc9b72c84c88a80344ca3ab39a310a44"
MEASEVAL_URL = f"https://codeload.github.com/harperco/MeasEval/zip/{MEASEVAL_REV}"
PY150_URL = "https://files.sri.inf.ethz.ch/data/py150_files.tar.gz"


def download(url: str, path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=90) as response, path.open("wb") as out:
            import shutil
            shutil.copyfileobj(response, out)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finish(name: str, docs: list[dict], metadata: dict) -> None:
    doc_ids = [d["doc_id"] for d in docs]
    assert len(doc_ids) == len(set(doc_ids)), "duplicate document IDs"
    group_sets = {}
    counts = {}
    texts = {}
    for split in ("fit", "eval"):
        selected = [d for d in docs if d["split"] == split]
        group_sets[split] = {d["group_id"] for d in selected}
        texts[split] = {hashlib.sha256(d["text"].encode()).hexdigest() for d in selected}
        label_counts = collections.Counter(s["label"] for d in selected for s in d["spans"])
        counts[split] = {"documents": len(selected), "groups": len(group_sets[split]),
                         "spans": sum(label_counts.values()), "labels": dict(sorted(label_counts.items()))}
        for d in selected:
            for s in d["spans"]:
                assert 0 <= s["start"] < s["end"] <= len(d["text"])
                assert d["text"][s["start"]:s["end"]] == s["text"]
    assert not group_sets["fit"] & group_sets["eval"], "fit/eval group overlap"
    assert not texts["fit"] & texts["eval"], "fit/eval exact-text overlap"
    path = OUTPUT / f"{name}.jsonl"
    with path.open("w") as out:
        for d in docs:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    metadata.update({"counts": counts, "output": str(path.relative_to(ROOT)),
                     "output_sha256": sha256(path), "offsets": "Python Unicode character offsets; end exclusive",
                     "group_overlap": 0, "exact_text_overlap": 0})
    (OUTPUT / f"{name}.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(name, json.dumps(counts, sort_keys=True), flush=True)


def prepare_measeval() -> None:
    archive = SOURCE / "measeval.zip"
    download(MEASEVAL_URL, archive)
    docs = []
    bad = []
    with zipfile.ZipFile(archive) as z:
        paths = sorted(n for n in z.namelist() if "/tsv/" in n and n.endswith(".tsv"))
        eval_articles = {Path(n).stem.split("-")[0] for n in paths if "/data/eval/" in n}
        for path in paths:
            official_split = path.split("/data/")[1].split("/")[0]
            if official_split not in {"train", "eval"}:
                continue
            doc_id = Path(path).stem
            article = doc_id.split("-")[0]
            if official_split == "train" and article in eval_articles:
                continue
            # Decode bytes directly: newline normalization would invalidate offsets.
            text_path = path.replace("/tsv/", "/txt/").replace(".tsv", ".txt")
            if text_path not in z.namelist():
                text_path = path.replace("/tsv/", "/text/").replace(".tsv", ".txt")
            text = z.read(text_path).decode("utf8")
            spans = {}
            for row in csv.DictReader(io.StringIO(z.read(path).decode("utf8")), delimiter="\t"):
                start, end = int(row["startOffset"]), int(row["endOffset"])
                label = row["annotType"]
                if not (0 <= start < end <= len(text)) or text[start:end] != row["text"]:
                    bad.append({"doc_id": doc_id, "row": row})
                    continue
                key = (start, end, label)
                if key not in spans:
                    spans[key] = {"start": start, "end": end, "label": label, "text": row["text"],
                                  "annotation_sets": [], "annotation_ids": []}
                spans[key]["annotation_sets"].append(row["annotSet"])
                spans[key]["annotation_ids"].append(row["annotId"])
            docs.append({"doc_id": f"measeval:{doc_id}", "group_id": f"measeval-article:{article}",
                         "split": "fit" if official_split == "train" else "eval",
                         "source_split": official_split, "text": text,
                         "spans": sorted(spans.values(), key=lambda s: (s["start"], s["end"], s["label"]))})
    finish("measeval", docs, {
        "source": "https://github.com/harperco/MeasEval", "revision": MEASEVAL_REV,
        "archive_url": MEASEVAL_URL, "archive_sha256": sha256(archive),
        "labels": "Official gold annotType: Quantity, MeasuredEntity, MeasuredProperty, Qualifier",
        "split_policy": "Official eval paragraphs; official train paragraphs excluding every eval source article. Trial excluded.",
        "deduplication": "Repeated (start,end,label) mentions in separate annotation sets count once; metadata retained.",
        "excluded_invalid_annotations": bad,
        "suggested_pairs": [["MeasuredEntity", "MeasuredProperty"], ["Quantity", "MeasuredProperty"]],
        "caveat": "Qualifier has lower fit support. Relation gold labels may use later context; extract at span end and interpret causally.",
    })


def code_spans(text: str) -> list[dict]:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def charpos(row: int, col: int) -> int:
        return starts[row - 1] + col

    def astpos(node: ast.AST, end: bool = False) -> int:
        row = node.end_lineno if end else node.lineno
        bytecol = node.end_col_offset if end else node.col_offset
        col = len(lines[row - 1].encode("utf8")[:bytecol].decode("utf8"))
        return charpos(row, col)

    tokens = [(t.string, charpos(*t.start), charpos(*t.end), t.type)
              for t in tokenize.generate_tokens(io.StringIO(text).readline)
              if t.type not in {tokenize.ENDMARKER, tokenize.ENCODING}]
    token_starts = [t[1] for t in tokens]
    found = {}

    def add(start: int, end: int, label: str):
        if start < end:
            found[start, end, label] = {"start": start, "end": end, "label": label, "text": text[start:end]}

    def token_between(value: str, start: int, end: int):
        for i in range(bisect.bisect_left(token_starts, start), len(tokens)):
            v, a, b, _ = tokens[i]
            if a >= end:
                break
            if v == value:
                return a, b
        return None

    def identifier(node: ast.AST, label: str):
        if isinstance(node, ast.Name):
            add(astpos(node), astpos(node, True), label)
        elif isinstance(node, ast.Attribute):
            end = astpos(node, True)
            add(end - len(node.attr), end, label)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        identifier(child, "AssignmentTarget")
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                loc = token_between("=", astpos(targets[-1], True), astpos(node.value))
                if loc:
                    add(*loc, "AssignmentOperator")
        elif isinstance(node, ast.Call):
            identifier(node.func, "CallCallee")
            loc = token_between("(", astpos(node.func, True), astpos(node, True))
            if loc:
                add(*loc, "CallOpenParen")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            loc = token_between(node.name, astpos(node), starts[node.lineno])
            if loc:
                add(*loc, "FunctionDefName")
        elif isinstance(node, ast.Return):
            loc = token_between("return", astpos(node), astpos(node, True))
            if loc:
                add(*loc, "ReturnKeyword")
        elif isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for i, op in enumerate(node.ops):
                if isinstance(op, (ast.In, ast.NotIn)):
                    loc = token_between("in", astpos(operands[i], True), astpos(operands[i + 1]))
                    if loc:
                        add(*loc, "MembershipOperator")
    return sorted(found.values(), key=lambda s: (s["start"], s["end"], s["label"]))


def prepare_py150(max_fit: int, max_eval: int, max_chars: int) -> None:
    archive = SOURCE / "py150_files.tar.gz"
    source_archive = SOURCE / "py150_data.tar.gz"
    download(PY150_URL, archive)
    if not source_archive.exists():
        import shutil
        with tarfile.open(archive) as tar, tar.extractfile("data.tar.gz") as inp, source_archive.open("wb") as out:
            shutil.copyfileobj(inp, out)
    with tarfile.open(archive) as tar:
        original = {split: set(tar.extractfile(name).read().decode().splitlines())
                    for split, name in [("fit", "python100k_train.txt"), ("eval", "python50k_eval.txt")]}
    allowed, source_licenses = {}, {}
    manifest_hashes = {}
    for split, official in [("fit", "train"), ("eval", "eval")]:
        path = SOURCE / f"py150_open_{official}_manifest.json"
        download(f"https://raw.githubusercontent.com/google-research-datasets/eth_py150_open/master/{official}__manifest.json", path)
        manifest_hashes[official] = sha256(path)
        allowed[split] = set()
        for row in json.loads(path.read_text()):
            filepath = "data/" + row["filepath"]
            repo = "/".join(row["filepath"].split("/")[:2])
            # Additional repository holdout, while preserving official file splits.
            held_out = int(hashlib.sha256(repo.encode()).hexdigest()[:8], 16) % 4 == 0
            if filepath in original[split] and held_out == (split == "eval"):
                allowed[split].add(filepath)
                source_licenses[filepath] = row["license"]
    selected = {"fit": [], "eval": []}
    allowed_all = allowed["fit"] | allowed["eval"]
    with tarfile.open(source_archive) as tar:
        candidates = {m.name for m in tar.getmembers()
                      if m.isfile() and 0 < m.size <= max_chars * 4 and m.name in allowed_all}
    candidate_order = {
        split: sorted((n for n in candidates if n in allowed[split]),
                      key=lambda n: hashlib.sha256(("geoae-py150-v1:" + n).encode()).hexdigest())[:limit * 10]
        for split, limit in [("fit", max_fit), ("eval", max_eval)]}
    requested = set(candidate_order["fit"] + candidate_order["eval"])
    raw_files = {}
    # Read compressed members in one forward pass instead of repeatedly seeking.
    with tarfile.open(source_archive, "r|gz") as tar:
        for member in tar:
            if member.name in requested:
                raw_files[member.name] = tar.extractfile(member).read()
    failures = collections.Counter()
    for split, limit in [("fit", max_fit), ("eval", max_eval)]:
        per_repo = collections.Counter()
        candidates_split = candidate_order[split]
        for name in candidates_split:
            if len(selected[split]) >= limit:
                break
            repo = "/".join(name.split("/")[1:3])
            if per_repo[repo] >= 8:
                continue
            raw = raw_files[name]
            try:
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                text = raw.decode(encoding)
                if len(text) > max_chars:
                    failures["too_long"] += 1
                    continue
                spans = code_spans(text)
            except (SyntaxError, UnicodeError, tokenize.TokenError, ValueError, IndentationError):
                failures["not_python3_parseable"] += 1
                continue
            if not spans:
                failures["no_roles"] += 1
                continue
            selected[split].append({"doc_id": f"py150:{name}", "group_id": f"py150-repo:{repo}",
                                    "split": split, "source_split": "train" if split == "fit" else "eval",
                                    "source_path": name, "source_license": source_licenses[name],
                                    "text": text, "spans": spans})
            per_repo[repo] += 1
    # Remove exact duplicate eval text even though repository/file IDs are distinct.
    fit_texts = {hashlib.sha256(d["text"].encode()).hexdigest() for d in selected["fit"]}
    eval_docs = [d for d in selected["eval"] if hashlib.sha256(d["text"].encode()).hexdigest() not in fit_texts]
    finish("py150", selected["fit"] + eval_docs, {
        "source": "https://www.sri.inf.ethz.ch/py150", "archive_url": PY150_URL,
        "archive_sha256": sha256(archive),
        "redistributable_manifest_source": "https://github.com/google-research-datasets/eth_py150_open",
        "manifest_sha256": manifest_hashes,
        "labels": "Derived from Python standard-library AST and tokenizer; not native gold semantic labels.",
        "split_policy": "Official train/eval file manifests plus repository SHA256 modulo 4 holdout (eval=0; fit=1,2,3).",
        "sampling": f"SHA256 file order, max 8 files/repository, <= {max_chars} chars/file, caps {max_fit} fit and {max_eval} eval files.",
        "parser": "Python 3 standard-library ast; original Python 2-only sources are excluded without conversion.",
        "excluded_counts": dict(failures),
        "duplicate_eval_text_removed": len(selected["eval"]) - len(eval_docs),
        "suggested_pairs": [["AssignmentOperator", "MembershipOperator"], ["AssignmentTarget", "CallCallee"],
                            ["FunctionDefName", "CallCallee"]],
        "causal_caveat": "Identifier roles can require following code; operator/keyword anchors are visible by span end. Interpret this as a structural control.",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["measeval", "py150", "both"], default="both")
    parser.add_argument("--max-fit", type=int, default=400)
    parser.add_argument("--max-eval", type=int, default=240)
    parser.add_argument("--max-chars", type=int, default=16000)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.dataset in {"measeval", "both"}:
        prepare_measeval()
    if args.dataset in {"py150", "both"}:
        prepare_py150(args.max_fit, args.max_eval, args.max_chars)


if __name__ == "__main__":
    main()
