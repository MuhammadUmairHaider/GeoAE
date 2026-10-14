"""
Structured-prompt documents for the sampled extractor (`kind: structured`).

WHY. The probe ladder includes short, templated, fact-bearing prompts (RAVEL
entity attributes, IOI name roles). Web text rarely looks like that, so a small
slice of short structured text keeps that register in the training
distribution. It is gated behind a source weight and is 0 in the default mix.

WHAT. Two existing, human-curated sources, never synthetic templates of our own:
  * CounterFact (azhx/counterfact) — the TRUE fact of each record, rendered as a
    sentence: "The mother tongue of Danielle Darrieux is French."
  * NQ-open (google-research-datasets/nq_open) — "Question: ...?\nAnswer: ..."
Items are packed `items_per_doc` to a document (a fact list / a Q&A list), so a
structured doc has a normal context length instead of being 12 tokens long, most of
them near BOS.

GUARD, WHICH IS THE POINT OF THIS MODULE. The corpus must not overlap the
evaluation benchmarks it is meant to help with, or probe gains become template
memorisation:
  * RAVEL: any item that contains a RAVEL entity (all splits) as a whole-word
    phrase is DROPPED. This is deliberately aggressive — "Paris" anywhere in an
    item removes it — because RAVEL's labels are exactly country/continent/
    language facts about those entities.
  * IOI: every IOI template (all splits) is compiled to a regex with its name /
    place / object slots as wildcards; any item matching one is DROPPED.
Drop counts are returned in the stats and land in the extraction meta.json.
"""
from __future__ import annotations

import ast
import random
import re
from typing import Iterable, Iterator

_WORD = re.compile(r"[^\W_]+(?:['’.\-][^\W_]+)*", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _as_obj(v):
    """HF streaming sometimes yields nested fields as their str(repr)."""
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

class LeakageGuard:
    """Rejects text that contains a banned entity phrase or matches a banned template."""

    def __init__(self, entities: Iterable[str] = (), templates: Iterable[str] = ()):
        self.entities: set[tuple[str, ...]] = set()
        for e in entities:
            w = tuple(_words(e))
            if w:
                self.entities.add(w)
        self.max_n = max((len(w) for w in self.entities), default=0)
        self.patterns = [self._compile(t) for t in set(templates) if t]

    @staticmethod
    def _compile(template: str) -> re.Pattern:
        # "As {name_A} and {name_B} left the {place}, ..." -> slots become .+?
        parts = re.split(r"\{[^{}]*\}", template)
        body = r".+?".join(re.escape(p) for p in parts)
        return re.compile(body, re.DOTALL)

    def entity_hit(self, text: str) -> bool:
        if not self.entities:
            return False
        w = _words(text)
        for n in range(1, self.max_n + 1):
            for i in range(len(w) - n + 1):
                if tuple(w[i:i + n]) in self.entities:
                    return True
        return False

    def template_hit(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)


def load_benchmark_guard() -> tuple[LeakageGuard, dict]:
    """Guard built from every split of mib-bench/ravel and mib-bench/ioi."""
    from datasets import load_dataset
    ents, tmpls = set(), set()
    for split, ds in load_dataset("mib-bench/ravel").items():
        ents.update(e for e in ds["entity"] if e)
    for split, ds in load_dataset("mib-bench/ioi").items():
        tmpls.update(t for t in ds["template"] if t)
    return LeakageGuard(ents, tmpls), {"ravel_entities": len(ents), "ioi_templates": len(tmpls)}


# ---------------------------------------------------------------------------
# Item sources
# ---------------------------------------------------------------------------

def counterfact_fact(record: dict) -> str | None:
    rw = _as_obj(record.get("requested_rewrite"))
    if not isinstance(rw, dict):
        return None
    prompt, subj = rw.get("prompt"), rw.get("subject")
    tgt = _as_obj(rw.get("target_true"))
    tgt = tgt.get("str") if isinstance(tgt, dict) else None
    if not prompt or not subj or not tgt or "{}" not in prompt:
        return None
    return f"{prompt.format(subj)} {tgt}."


def nq_item(record: dict) -> str | None:
    q, ans = record.get("question"), _as_obj(record.get("answer"))
    if isinstance(ans, (list, tuple)):
        ans = ans[0] if ans else None
    if not q or not ans:
        return None
    q = q.strip().rstrip("?")
    return f"Question: {q[0].upper() + q[1:]}?\nAnswer: {ans}"


def filter_items(items: Iterable[str], guard: LeakageGuard, stats: dict, key: str) -> list[str]:
    kept = []
    stats.setdefault(key, {"raw": 0, "dropped_entity": 0, "dropped_template": 0, "kept": 0})
    s = stats[key]
    for it in items:
        if it is None:
            continue
        s["raw"] += 1
        if guard.entity_hit(it):
            s["dropped_entity"] += 1
            continue
        if guard.template_hit(it):
            s["dropped_template"] += 1
            continue
        kept.append(it)
    s["kept"] = len(kept)
    return kept


def pack_docs(items: list[str], per_doc: int, sep: str) -> list[str]:
    return [sep.join(items[i:i + per_doc]) for i in range(0, len(items), per_doc)
            if len(items[i:i + per_doc]) == per_doc]


def iter_structured_docs(seed: int = 42, items_per_doc: int = 8,
                         guard: LeakageGuard | None = None,
                         stats: dict | None = None) -> Iterator[str]:
    """Yields packed structured documents, alternating fact lists and Q&A lists.

    Loads both item sources in full (small: ~22k + ~88k rows), filters them
    through the guard, shuffles with `seed`, and interleaves until both run out.
    `stats` (if given) is filled in place for the extraction meta.
    """
    from datasets import load_dataset
    stats = {} if stats is None else stats
    if guard is None:
        guard, gstats = load_benchmark_guard()
        stats["guard"] = gstats

    cf = load_dataset("azhx/counterfact", split="train")
    facts = filter_items((counterfact_fact(r) for r in cf), guard, stats, "counterfact")
    nq = load_dataset("google-research-datasets/nq_open", split="train")
    qas = filter_items((nq_item(r) for r in nq), guard, stats, "nq_open")

    rng = random.Random(seed)
    rng.shuffle(facts)
    rng.shuffle(qas)
    a = pack_docs(facts, items_per_doc, "\n")
    b = pack_docs(qas, items_per_doc, "\n\n")
    stats["docs"] = {"fact_lists": len(a), "qa_lists": len(b)}
    for i in range(max(len(a), len(b))):
        if i < len(a):
            yield a[i]
        if i < len(b):
            yield b[i]
