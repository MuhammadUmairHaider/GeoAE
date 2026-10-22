"""Pure helpers for the Atlas class-intervention corpus: FineWeb-Atlas CONTENT concepts used exactly like DBpedia-14 classes.

A "draw" is K = 14 content concepts sampled uniformly at random (seeded) from the concepts that have >= MIN_CHUNKS chunks
in the 300k-chunk activation cache, subject to rules fixed BEFORE any model run (RULES below, written to draws.json):

  * no two drawn concepts share a name token sequence (case-insensitive containment, atlas_siblings.name_contains),
  * the chunk Jaccard of every pair in the cache is <= MAX_JACCARD,
  * every class keeps >= MIN_ELIGIBLE EXCLUSIVE chunks (chunks carrying the class and none of the other 13).
A violation rejects the whole draw and redraws from the same seeded stream. Everything is numpy/scipy so it unit-tests
on a laptop; the orchestration is in eval_out/atlas_classes_{draw,prep,joint_correct}.py.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

import geoae.analysis.atlas_siblings as AS

RULES = dict(
    seeds=[0, 1, 2],
    K=14,
    content_only=True,
    merge_duplicate_names=True,            # case-insensitive duplicate names are one unit (as in atlas_siblings)
    min_chunks=600,                        # chunks of the concept in the 300k cache
    min_eligible=400,                      # exclusive chunks per class in a draw
    max_jaccard=0.02,                      # pairwise chunk Jaccard in the cache
    name_containment=True,                 # token-wise, case-insensitive, either direction
    name_forbidden_chars=["{", "}", '"', "newline", "tab"],   # would corrupt the {{"text":Label}} prompt/score format -> not in the universe
    name_max_chars=40,
    n_split_buckets=100,
    fewshot_below=10,                      # doc bucket < 10        -> few-shot exemplar pool
    fit_below=50,                          # 10 <= bucket < 50      -> fit docs ; >= 50 -> test docs   (document level)
    cap_per_class_split=600,               # chunks written per class and split
    min_joint_correct_test=40,             # a class needs this many joint-correct TEST docs to be intervened on
    exemplar_tok_range=[30, 100],
    n_exemplars=4,
    max_draw_attempts=200000,
)


# --------------------------------------------------------------------------------------------- universe
def name_ok(name: str, rules=RULES) -> bool:
    s = str(name)
    if not s.strip() or len(s) > rules["name_max_chars"]:
        return False
    return not any(ch in s for ch in ('{', '}', '"', '\n', '\r', '\t'))


def build_units(names, descriptions, concept_ids, counts, is_content):
    """Content concepts -> units (case-insensitive duplicate names merged). Each unit: ids (sorted concept ids), pid
    (primary id), name, desc, count (Atlas-wide chunk count, summed over members)."""
    rows = np.flatnonzero(np.asarray(is_content))
    nm = [names[i] for i in rows]
    cnt = np.asarray(counts, dtype=np.float64)[rows]
    units = []
    for g in AS.merge_duplicate_names(nm, cnt):
        pm = AS.primary_member(g, cnt)
        pr = int(rows[pm])
        units.append(dict(ids=sorted(int(concept_ids[rows[m]]) for m in g), pid=int(concept_ids[pr]), name=str(names[pr]),
                          desc=str(descriptions[pr]), count=int(sum(cnt[m] for m in g))))
    return units


# --------------------------------------------------------------------------------------------- draw / reject
def pair_jaccard(sub) -> np.ndarray:
    """(K, K) chunk Jaccard of the columns of a chunk x K 0/1 sparse matrix (diagonal = 1)."""
    S = sub.tocsc().astype(np.int64)
    inter = np.asarray((S.T @ S).todense(), dtype=np.float64)
    c = np.diag(inter).copy()
    return inter / np.maximum(c[:, None] + c[None, :] - inter, 1.0)


def exclusive_chunks(sub, k: int) -> np.ndarray:
    """Chunk indices carrying column k and no other column of `sub`."""
    S = sub.tocsc()
    rowsum = np.asarray(S.sum(axis=1)).ravel()
    col = S.indices[S.indptr[k]:S.indptr[k + 1]]
    return col[rowsum[col] == 1]


def check_draw(sub, names, rules=RULES):
    """(ok, reasons, stats) for one candidate draw. `sub` is the chunk x K 0/1 membership of the K drawn units."""
    K = len(names)
    reasons = []
    J = pair_jaccard(sub)
    iu = np.triu_indices(K, 1)
    jmax = float(J[iu].max()) if K > 1 else 0.0
    if jmax > rules["max_jaccard"]:
        reasons.append("jaccard")
    if rules["name_containment"]:
        if any(AS.name_contains(names[i], names[j]) for i in range(K) for j in range(i + 1, K)):
            reasons.append("name_containment")
    S = sub.tocsc()
    n_excl = [int(len(exclusive_chunks(S, k))) for k in range(K)]
    if min(n_excl) < rules["min_eligible"]:
        reasons.append("min_eligible")
    return (not reasons), reasons, dict(max_jaccard=jmax, n_exclusive=n_excl)


def draw_classes(seed: int, M, names, candidates, rules=RULES):
    """Seeded draw loop. M: chunk x unit csc membership; names: unit names; candidates: unit indices that pass the
    universe filter (name_ok, >= min_chunks). Returns dict(units, attempts, rejections{reason: n}, stats)."""
    rng = np.random.default_rng(seed)
    cand = np.asarray(sorted(int(c) for c in candidates), dtype=np.int64)
    rej: dict[str, int] = {}
    for attempt in range(1, rules["max_draw_attempts"] + 1):
        pick = [int(x) for x in rng.choice(cand, size=rules["K"], replace=False)]
        ok, reasons, stats = check_draw(M[:, pick], [names[i] for i in pick], rules)
        if ok:
            return dict(units=pick, attempts=attempt, n_rejected=attempt - 1, rejections=rej, stats=stats)
        for r in reasons:
            rej[r] = rej.get(r, 0) + 1
    raise RuntimeError(f"no valid draw for seed {seed} in {rules['max_draw_attempts']} attempts")


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------------------------- splits / prompt
def doc_bucket(doc_ids, seed: int, n: int = 100) -> np.ndarray:
    """Deterministic document-level bucket in [0, n) (hash of doc id and seed): all chunks of a document share it."""
    with np.errstate(over="ignore"):
        x = np.asarray(doc_ids, dtype=np.uint64) + (np.uint64(seed) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(n)).astype(np.int64)


def split_of_bucket(b, rules=RULES) -> np.ndarray:
    """0 = few-shot pool, 1 = fit, 2 = test."""
    b = np.asarray(b)
    return np.where(b < rules["fewshot_below"], 0, np.where(b < rules["fit_below"], 1, 2))


def clean_text(t: str) -> str:
    return " ".join(str(t).split())


def _esc(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def build_prompt_head(names, exemplars) -> str:
    """The db14 template with these class names and (text, class index) exemplars. Output is a str.format template:
    exemplar braces are doubled so only the final {} slot is substituted."""
    head = ("Choose from one of these categories: " + ", ".join(_esc(n) for n in names) +
            ". Be careful distinguishing between similar categories.\n\n")
    for text, k in exemplars:
        head += "{{ " + _esc(clean_text(text)) + ":" + _esc(names[k]) + "}}\n\n"
    return head + '{{"{}":'


# --------------------------------------------------------------------------------------------- collateral matrix
def comp_by_class(labels, base_correct, correct) -> dict:
    """Per-other-class collateral on the complement sample of one (target, op, substrate) evaluation.
    labels/base_correct/correct are aligned over the COMPLEMENT docs. {str(class): {n, base_acc, acc, drop}}, drop =
    base_acc - acc (positive = the edit hurt that class)."""
    labels = np.asarray(labels)
    b = np.asarray(base_correct, dtype=float)
    a = np.asarray(correct, dtype=float)
    out = {}
    for o in sorted(set(int(x) for x in labels)):
        m = labels == o
        out[str(o)] = dict(n=int(m.sum()), base_acc=round(float(b[m].mean()), 4), acc=round(float(a[m].mean()), 4),
                           drop=round(float(b[m].mean() - a[m].mean()), 4))
    return out


def drop_matrix(concept_recs: dict, key: str, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) matrix of per-class accuracy drops, row = target class, col = other class, NaN where
    unavailable (diagonal, classes not intervened on / absent from the complement). concept_recs: results['concepts']."""
    D = np.full((n_classes, n_classes), np.nan)
    for c, rec in concept_recs.items():
        r = rec.get(key)
        if not r or "comp_by_class" not in r:
            continue
        for o, v in r["comp_by_class"].items():
            D[int(c), int(o)] = v["drop"]
    return D
