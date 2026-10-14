"""
Contrastive form steering: base residual stream (h) vs GeoAE latent (z).

Picks a morphological / orthographic axis with a clean two-token readout --
"performed" vs "perform", "events" vs "event", "Google" vs "google" -- and asks
whether the axis is steerable, comparing the model's own residual stream against
the AE latent on identical positions in the same pass.

WHY MORE THAN ONE AXIS.  The first run (past tense, llama L27) came back a dead
heat: hit rate 0.254 base vs 0.273 latent, +0.019 +/- 0.027, tie at every alpha,
with cos(r_h, decode(r_z)) = 0.979. Sifting closest_tokens then showed why the
obvious follow-up ("test a feature the AE clusters on") will not help: measured
by whether the two forms of a word land in different clusters, past tense ALREADY
splits at 71-78%, as cluster-organising as any axis except -ly. Cluster
organisation does not predict steerability.

The structural reason is that latent_dim == hidden_size with FVE ~0.998 makes the
AE close to an invertible reparameterisation, so NO linear intervention can
separate the two spaces. Hence the second mode below.

AXES (--axis)          marked form -> unmarked form        split rate (llama/gemma)
  past                 performed -> perform                71% / 78%
  plural               events    -> event                  71% / 71%
  case                 Google    -> google                 62% / 59%   (semantics held constant)
  ing                  running   -> run                    74% / 68%
  ly                   closely   -> close                  100% / 100%

MODES
  linear   (default)  r = mean(marked) - mean(unmarked); h' = h -/+ alpha*r, and
                      the same in z. alpha is a multiple of that space's own
                      mean-difference norm, so alpha=1 moves a point by exactly
                      the marked->unmarked gap and the two spaces are comparable.
  centroid (--centroid)  move z toward the nearest counterpart-enriched CLUSTER
                      centroid: z' = z + alpha*(C[k*] - z), alpha=1 landing on the
                      centroid. This is a Voronoi-defined, nonlinear edit with no
                      h equivalent -- the first intervention that the AE's
                      structure can express and the base residual stream cannot.

Protocol (shared by both modes)
  1. MINE wiki+C4+pile, derive counterpart forms by ordered orthographic rules,
     keep a pair only if both forms are real words in the same corpus and single
     leading-space tokens. RedPajama is unusable: -1T and -V2 are script-based
     loaders current `datasets` refuses, and the -Sample / SlimPajama mirrors 404.
  2. CERTIFY IN CONTEXT. Spelling cannot tell "United" from "performed" (and
     United -> Unite is a legitimate pair, so lexical filters pass it). The model
     can: keep a position only if the clean top-1 next token IS the target form
     and the counterpart is in the top-k.
  3. TWO SLOTS, BIDIRECTIONAL -- marked positions steer one way, unmarked the
     other. A degradation direction moves both the same way; a real feature moves
     them oppositely.
  4. HELD-OUT WORDS. The direction is fitted on one word set and evaluated on a
     disjoint one, separating a general feature from memorised per-word offsets.

Arms, all on identical positions: `h`, `z`, `zcent` (centroid mode only), and
`zrecon` (encode/decode, no steer) which separates reconstruction damage from
steering effect.

Usage:
    python -m geoae.interp.steering_morph_compare \
      --checkpoint e2e/.../best_val.pt --axis plural \
      --out results_morph_plural.json

    python -m geoae.interp.steering_morph_compare \
      --checkpoint e2e/.../best_val.pt --axis past --centroid \
      --out results_morph_past_centroid.json

    python -m geoae.interp.steering_morph_compare --report_from <results.json>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.evaluate import load_ae_from_checkpoint, sanity_check_splice
from geoae.hooks import SplicingHook
from geoae.interp import _shared as shared
from geoae.interp.closest_tokens import DEFAULT_SOURCES, open_sources, stream_docs
from geoae.seeding import seed_everything

WORD_RE = re.compile(r"\b([A-Za-z]+)\b")

# Mining corpus: the prose subset of closest_tokens' sources. Code and math are
# excluded -- inflected word forms live in prose.
MINE_DOMAINS = ("wiki", "web", "pile")

# "-ed" spellings that are not past tense. The contextual filter removes these on
# its own; the list is kept as an explicit control set, so every run reports how
# much the direction disturbs positions it should not touch.
CONTROL_WORDS = ["United", "red", "sacred", "wicked", "indeed", "hundred",
                 "need", "speed", "bed", "seed", "embed", "naked"]

# Irregular past tenses escape any suffix rule, which makes them the sharper test
# for the `past` axis: there is no shared spelling for a direction to latch onto.
IRREGULAR_PAIRS = [
    ("led", "lead"), ("built", "build"), ("written", "write"), ("made", "make"),
    ("found", "find"), ("held", "hold"), ("told", "tell"), ("sold", "sell"),
    ("kept", "keep"), ("left", "leave"), ("sent", "send"), ("spent", "spend"),
    ("brought", "bring"), ("bought", "buy"), ("taught", "teach"), ("caught", "catch"),
    ("became", "become"), ("began", "begin"), ("gave", "give"), ("took", "take"),
]


# ---------------------------------------------------------------------------
# Axes. Each supplies: a detector for the MARKED surface form, ordered guesses at
# the UNMARKED counterpart, and an optional extra validity test on the pair.
# ---------------------------------------------------------------------------

def _dedup(cands):
    return list(dict.fromkeys(c for c in cands if len(c) >= 2))


def _cand_past(w: str) -> list[str]:
    out = []
    if w.endswith("ied") and len(w) > 4:
        out.append(w[:-3] + "y")               # carried -> carry
    stem = w[:-2]
    if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
        out.append(stem[:-1])                  # stopped -> stop
    out += [w[:-1], stem]                      # used -> use ; performed -> perform
    return _dedup(out)


def _cand_plural(w: str) -> list[str]:
    out = []
    if w.endswith("ies") and len(w) > 4:
        out.append(w[:-3] + "y")               # countries -> country
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        out.append(w[:-2])                     # boxes -> box
    out.append(w[:-1])                         # events -> event
    return _dedup(out)


def _cand_ing(w: str) -> list[str]:
    stem = w[:-3]
    out = []
    if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
        out.append(stem[:-1])                  # running -> run
    out += [stem + "e", stem]                  # using -> use ; performing -> perform
    return _dedup(out)


def _cand_ly(w: str) -> list[str]:
    out = []
    if w.endswith("ily") and len(w) > 4:
        out.append(w[:-3] + "y")               # easily -> easy
    out += [w[:-2] + "e", w[:-2]]              # gently -> gentle ; closely -> close
    return _dedup(out)


def _cand_case(w: str) -> list[str]:
    return [w[0].lower() + w[1:]]              # Google -> google


def _s_forms(base: str) -> list[str]:
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return [base[:-1] + "ies"]
    if base.endswith(("s", "x", "z", "ch", "sh")):
        return [base + "es"]
    return [base + "s"]


def _ing_forms(base: str) -> list[str]:
    forms = [base + "ing"]
    if base.endswith("e"):
        forms.append(base[:-1] + "ing")
    if len(base) >= 3 and base[-1] not in "aeiouwxy" and base[-2] in "aeiou":
        forms.append(base + base[-1] + "ing")
    return forms


def _verbhood(base: str, counts: Counter, min_count: int) -> bool:
    """The unmarked form must inflect like a verb in this corpus."""
    return (any(counts.get(f, 0) >= min_count for f in _s_forms(base))
            and any(counts.get(f, 0) >= min_count for f in _ing_forms(base)))


def _not_plural_already(base: str, counts: Counter, min_count: int) -> bool:
    return not base.endswith("s")


AXES = {
    "past": dict(
        marked_re=re.compile(r"^[a-z]{3,}ed$"), candidates=_cand_past, extra=_verbhood,
        slots=("past", "base"), irregulars=True,
        blurb="performed -> perform"),
    "plural": dict(
        marked_re=re.compile(r"^[a-z]{4,}s$"), candidates=_cand_plural, extra=_not_plural_already,
        slots=("plural", "singular"), irregulars=False,
        blurb="events -> event"),
    "ing": dict(
        marked_re=re.compile(r"^[a-z]{5,}ing$"), candidates=_cand_ing, extra=_verbhood,
        slots=("ing", "base"), irregulars=False,
        blurb="running -> run"),
    "ly": dict(
        marked_re=re.compile(r"^[a-z]{5,}ly$"), candidates=_cand_ly, extra=None,
        slots=("adverb", "adjective"), irregulars=False,
        blurb="closely -> close"),
    "case": dict(
        marked_re=re.compile(r"^[A-Z][a-z]{2,}$"), candidates=_cand_case, extra=None,
        slots=("capitalised", "lowercase"), irregulars=False,
        blurb="Google -> google  (semantics held constant)"),
}


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

def single_token(tokenizer, word: str) -> int | None:
    """Token id if ' word' is exactly one token, else None."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def build_lexicon(axis: dict, counts: Counter, tokenizer, min_count: int,
                  min_par: int, n_words: int) -> list[dict]:
    """(marked, unmarked) pairs surviving the lexical filters, most frequent first."""
    out = []
    for marked, c_marked in counts.items():
        if not axis["marked_re"].match(marked) or c_marked < min_count:
            continue
        marked_id = single_token(tokenizer, marked)
        if marked_id is None:
            continue
        best = None
        for unmarked in axis["candidates"](marked):
            c_un = counts.get(unmarked, 0)
            if c_un < min_count:
                continue
            if axis["extra"] is not None and not axis["extra"](unmarked, counts, min_par):
                continue
            unmarked_id = single_token(tokenizer, unmarked)
            if unmarked_id is None:
                continue
            if best is None or c_un > best["unmarked_count"]:
                best = dict(marked=marked, unmarked=unmarked, marked_id=marked_id,
                            unmarked_id=unmarked_id, marked_count=c_marked,
                            unmarked_count=c_un)
        if best is not None:
            out.append(best)
    out.sort(key=lambda p: -min(p["marked_count"], p["unmarked_count"]))
    return out[:n_words]


def add_irregulars(pairs: list[dict], counts: Counter, tokenizer, min_count: int) -> list[dict]:
    have = {p["marked"] for p in pairs}
    for marked, unmarked in IRREGULAR_PAIRS:
        if marked in have or counts.get(marked, 0) < min_count or counts.get(unmarked, 0) < min_count:
            continue
        mid, uid = single_token(tokenizer, marked), single_token(tokenizer, unmarked)
        if mid is None or uid is None:
            continue
        pairs.append(dict(marked=marked, unmarked=unmarked, marked_id=mid, unmarked_id=uid,
                          marked_count=counts[marked], unmarked_count=counts[unmarked],
                          irregular=True))
    return pairs


def drop_role_collisions(pairs: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Remove pairs whose surface form plays both roles somewhere in the lexicon.

    " found" is the past of "find" AND the unmarked form of "founded"; on the
    plural axis " times" is the plural of "time" while " time" is itself the
    unmarked form of another pair. One token id cannot be both slots, so keeping
    them would silently label half their occurrences with the wrong slot.
    """
    marked_ids = {p["marked_id"] for p in pairs}
    unmarked_ids = {p["unmarked_id"] for p in pairs}
    clash = marked_ids & unmarked_ids
    if not clash:
        return pairs, []
    kept = [p for p in pairs if p["marked_id"] not in clash and p["unmarked_id"] not in clash]
    dropped = sorted({p["marked"] for p in pairs if p not in kept})
    return kept, dropped


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------

class PositionPatch:
    """
    Linear steering applied ONLY at flagged positions.

    `signs` is a (B, T) float tensor: 0 leaves a position untouched, +1/-1 sets
    the direction there. Patching one position per example keeps h and z
    symmetric -- both arms perturb the same site and leave everything else at its
    original value, so the comparison isolates the intervention.
    """

    def __init__(self, mode: str, r: torch.Tensor, step: float, ae, mean, std):
        self.mode = mode          # "h" | "z" | "zrecon"
        self.r, self.step = r, step
        self.ae, self.mean, self.std = ae, mean, std
        self.signs: torch.Tensor | None = None

    @torch.no_grad()
    def __call__(self, hs: torch.Tensor) -> torch.Tensor:
        if self.signs is None:
            return hs
        sg = self.signs.to(hs.device)
        m = sg != 0
        if not bool(m.any()):
            return hs
        h = hs.float()
        sel, sgn = h[m], sg[m].unsqueeze(1)
        r = self.r.to(sel.device)
        if self.mode == "h":
            sel = sel - self.step * sgn * r
        else:
            z = self.ae.encoder((sel - self.mean) / self.std)
            if self.mode == "z":
                z = z - self.step * sgn * r
            sel = self.ae.decoder(z) * self.std + self.mean
        out = h.clone()
        out[m] = sel
        return out.to(hs.dtype)


class CentroidPatch:
    """
    Cluster-space steering, in two flavours. Neither has an h equivalent: the
    target is defined by the AE's own Voronoi partition, not by a direction in
    activation space. That is the point -- with latent_dim == hidden_size and
    FVE ~0.998 the AE is close to an invertible reparameterisation, so every
    LINEAR edit ties with the base by construction.

    mode="toward"  z' = z + alpha * (C[k*] - z),  alpha=1 landing on the centroid.
        Measured to be degenerate for a two-token readout: the centroid is the
        cluster MEAN, so interpolating toward it discards which word this is, and
        the model can no longer emit the specific counterpart form. On plural at
        llama L27 it scored hit 0.011 against linear-z's 0.363 while pushing wiki
        ppl to 14.3. Kept because that contrast is the informative part.

    mode="delta"   z' = z + alpha * (C[k*] - C[k_source])
        The cluster-space analogue of a mean-difference direction: it applies the
        centroid-to-centroid offset while PRESERVING the token's own offset from
        its cluster, so word identity survives the edit.

    `targets` is a dense (B, T, L) tensor -- the destination centroid for
    "toward", the centroid difference for "delta" -- dense rather than packed so
    it cannot fall out of step with the boolean mask's row-major ordering.
    """

    def __init__(self, ae, mean, std, alpha: float, mode: str = "delta"):
        self.ae, self.mean, self.std, self.alpha = ae, mean, std, alpha
        self.mode = mode
        self.mask: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None

    @torch.no_grad()
    def __call__(self, hs: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return hs
        m = self.mask.to(hs.device)
        if not bool(m.any()):
            return hs
        h = hs.float()
        sel = h[m]
        tgt = self.targets.to(hs.device)[m]
        z = self.ae.encoder((sel - self.mean) / self.std)
        z = z + self.alpha * ((tgt - z) if self.mode == "toward" else tgt)
        sel = self.ae.decoder(z) * self.std + self.mean
        out = h.clone()
        out[m] = sel
        return out.to(hs.dtype)


def batch_docs(docs: list[torch.Tensor], pad_id: int, batch_size: int):
    """Right-padded batches, so absolute token positions stay valid."""
    for s in range(0, len(docs), batch_size):
        chunk = docs[s:s + batch_size]
        T = max(d.shape[0] for d in chunk)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        am = torch.zeros((len(chunk), T), dtype=torch.long)
        for i, d in enumerate(chunk):
            ids[i, :d.shape[0]] = d
            am[i, :d.shape[0]] = 1
        yield s, ids, am


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def mine_corpus(sources, n_tokens, tokenizer, max_doc_tokens, n_wiki_keep):
    """Stream prose, keeping tokenised docs, word counts, and raw wiki text."""
    docs, counts, wiki_txt = [], Counter(), []
    for ids, domain in stream_docs(sources, n_tokens, tokenizer,
                                   min_len=24, max_len=max_doc_tokens):
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        counts.update(WORD_RE.findall(text))
        docs.append(ids[0])
        if domain == "wiki" and len(wiki_txt) < n_wiki_keep:
            wiki_txt.append(text)
    return docs, counts, wiki_txt


def find_candidates(docs, lexicon, control_ids, max_ctx_per_word):
    """Token positions whose id is a target form; position 0 has no predictor."""
    by_word = defaultdict(lambda: {"marked": [], "unmarked": []})
    controls = []
    marked_of = {p["marked_id"]: p for p in lexicon}
    unmarked_of = {p["unmarked_id"]: p for p in lexicon}
    for di, ids in enumerate(docs):
        for i, tid in enumerate(ids.tolist()):
            if i == 0:
                continue
            if tid in marked_of:
                w = marked_of[tid]["marked"]
                if len(by_word[w]["marked"]) < max_ctx_per_word:
                    by_word[w]["marked"].append((di, i))
            elif tid in unmarked_of:
                w = unmarked_of[tid]["marked"]
                if len(by_word[w]["unmarked"]) < max_ctx_per_word:
                    by_word[w]["unmarked"].append((di, i))
            elif tid in control_ids and len(controls) < 400:
                controls.append((di, i, control_ids[tid], tid))
    return by_word, controls


def kl_div(logp_clean: torch.Tensor, logits_steer: torch.Tensor) -> float:
    logp_steer = F.log_softmax(logits_steer.float(), dim=-1)
    p = logp_clean.exp()
    return float((p * (logp_clean - logp_steer)).sum())


def make_global_centroid(ae, mean, std, alpha: float, C_pool: torch.Tensor,
                         C_all: torch.Tensor, mode: str = "delta"):
    """
    Cluster-space steering applied at EVERY position, for the wiki-perplexity
    probe. The per-position arm uses each token's own nearest eligible centroid,
    so the global analogue recomputes that per token; "delta" additionally needs
    each token's SOURCE centroid, hence C_all.
    """
    @torch.no_grad()
    def fn(hs: torch.Tensor) -> torch.Tensor:
        B, T, D = hs.shape
        x = (hs.reshape(B * T, D).float() - mean) / std
        z = ae.encoder(x)
        pool = C_pool.to(z.device)
        tgt = pool[torch.cdist(z, pool).argmin(1)]
        if mode == "toward":
            z = z + alpha * (tgt - z)
        else:
            src = C_all.to(z.device)[torch.cdist(z, C_all.to(z.device)).argmin(1)]
            z = z + alpha * (tgt - src)
        recon = (ae.decoder(z) * std + mean).reshape(B, T, D)
        return recon.to(hs.device).to(hs.dtype)
    return fn


# ---------------------------------------------------------------------------
# Reporting (self-contained, so `--report_from` can re-render a finished run)
# ---------------------------------------------------------------------------

def _paired_hit_diff(results: dict, alpha: float, arm_a="h", arm_b="z"):
    """
    (mean, se, n, paired?) for arm_b-minus-arm_a hit rate on held-out positions.

    The arms steer the SAME positions, so this is a paired quantity whenever the
    per-example rows are stored; the gaps are small enough that comparing two
    rate columns by eye is not decidable. Runs written before rows were stored
    fall back to the unpaired (conservative) two-proportion SE.
    """
    rows = results.get("rows")
    if rows:
        aa = {r["ex"]: r for r in rows if r["arm"] == arm_a and r["alpha"] == alpha
              and r["split"] == "heldout" and r["slot"] != "control"}
        bb = {r["ex"]: r for r in rows if r["arm"] == arm_b and r["alpha"] == alpha
              and r["split"] == "heldout" and r["slot"] != "control"}
        common = sorted(set(aa) & set(bb))
        if len(common) >= 2:
            d = np.array([bb[i]["hit"] - aa[i]["hit"] for i in common], dtype=float)
            return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d), True
    a = (results["arms"].get(f"{arm_a}_a{alpha}") or {}).get("heldout")
    b = (results["arms"].get(f"{arm_b}_a{alpha}") or {}).get("heldout")
    if not a or not b:
        return None
    p1, p2, n = a["hit_rate"], b["hit_rate"], min(a["n"], b["n"])
    se = float(np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)) if n else float("nan")
    return p2 - p1, se, n, False


def _verdict(pr):
    if pr is None:
        return "-", float("nan"), float("nan")
    dm, dse, _, paired = pr
    v = ("tie (within noise)" if abs(dm) < 2 * dse
         else ("latent better" if dm > 0 else "base better"))
    return v + ("" if paired else " *unpaired"), dm, dse


def render_report(results: dict) -> None:
    m, arms, ppl = results["meta"], results["arms"], results["ppl"]
    alphas = m["alphas"]
    axis = m.get("axis", "past")
    slots = m.get("slot_labels", ["past", "base"])
    # Runs predating the multi-axis rename used *_verbs keys.
    n_heldout = m.get("n_heldout_words", m.get("n_heldout_verbs", "?"))
    blurb = m.get("axis_blurb", AXES.get(axis, {}).get("blurb", ""))
    ho_n = (arms[f"h_a{alphas[0]}"]["heldout"] or {}).get("n", 0)

    print("\n" + "=" * 94)
    print(f"HELD-OUT WORDS — BASE RESIDUAL STREAM vs GeoAE LATENT   [axis: {axis} — {blurb}]")
    print("=" * 94)
    print(f"  base   (h) = the model's own residual stream at layer {m['layer']}, untouched by the AE")
    print("  latent (z) = encode -> steer -> decode through the AE")
    print(f"  slots      = {slots[0]} positions steered toward {slots[1]}, and the reverse")
    print(f"  {ho_n} scored positions across {n_heldout} words held OUT of the direction fit")

    print("\n--- ON TARGET (higher is better) " + "-" * 60)
    print(f"{'':>6}  {'base (h)':^16}{'latent (z)':^16}  {'latent - base':^20}")
    print(f"{'alpha':>6}  {'flip':>7}{'hit':>9}{'flip':>7}{'hit':>9}  {'hit diff':>9}{'+/- se':>8}  verdict")
    for a in alphas:
        h, z = arms[f"h_a{a}"]["heldout"], arms[f"z_a{a}"]["heldout"]
        if not h or not z:
            continue
        v, dm, dse = _verdict(_paired_hit_diff(results, a))
        print(f"{a:>6}  {h['flip_rate']:>7.3f}{h['hit_rate']:>9.3f}"
              f"{z['flip_rate']:>7.3f}{z['hit_rate']:>9.3f}  {dm:>+9.3f}{dse:>8.3f}  {v}")

    cent = sorted(a for a in results.get("centroid_alphas", []))
    if cent:
        print("\n--- CENTROID MODE: z' = z + alpha*(C[k*] - z), no h equivalent " + "-" * 30)
        print(f"  target = nearest {slots[1]}-enriched cluster centroid; alpha=1 lands on it")
        print(f"  coverage: {results.get('centroid_coverage', float('nan')):.1%} of positions had an "
              f"eligible target cluster")
        print("  toward = z + a*(C[k*] - z), lands ON the centroid and so discards word identity")
        print("  delta  = z + a*(C[k*] - C[k_src]), applies the centroid offset and KEEPS identity")
        print(f"{'':>6}  {'toward':^26}{'delta':^26}")
        print(f"{'alpha':>6}  {'hit':>7}{'top1chg':>9}{'ppl':>9}{'hit':>8}{'top1chg':>9}{'ppl':>9}")
        for a in cent:
            zt, zd = arms.get(f"zcent_a{a}"), arms.get(f"zcdelta_a{a}")
            if not zt or not zt["heldout"]:
                continue
            t, d = zt["heldout"], (zd or {}).get("heldout")
            nan = float("nan")
            print(f"{a:>6}  {t['hit_rate']:>7.3f}{t['top1_change']:>9.3f}"
                  f"{ppl.get(f'zcent_a{a}', nan):>9.2f}"
                  f"{(d['hit_rate'] if d else nan):>8.3f}{(d['top1_change'] if d else nan):>9.3f}"
                  f"{ppl.get(f'zcdelta_a{a}', nan):>9.2f}")
        best_lin = max((arms[f"z_a{a}"]["heldout"]["hit_rate"] for a in alphas
                        if arms[f"z_a{a}"]["heldout"]), default=float("nan"))
        def _best(pref):
            return max((arms[f"{pref}_a{a}"]["heldout"]["hit_rate"] for a in cent
                        if arms.get(f"{pref}_a{a}", {}).get("heldout")), default=float("nan"))
        print(f"  best hit — linear-z {best_lin:.3f} | centroid-toward {_best('zcent'):.3f} "
              f"| centroid-delta {_best('zcdelta'):.3f}")

    print("\n--- COLLATERAL DAMAGE (lower is better) " + "-" * 53)
    print(f"{'':>6}  {'wiki ppl':^18}{'control top1chg':^18}{'full-vocab KL':^18}")
    print(f"{'alpha':>6}  {'base':>8}{'latent':>10}{'base':>8}{'latent':>10}{'base':>8}{'latent':>10}")
    for a in alphas:
        h, z = arms[f"h_a{a}"], arms[f"z_a{a}"]
        hc, zc = h["control"], z["control"]
        nan = float("nan")
        print(f"{a:>6}  {ppl[f'h_a{a}']:>8.2f}{ppl[f'z_a{a}']:>10.2f}"
              f"{(hc['top1_change'] if hc else nan):>8.3f}{(zc['top1_change'] if zc else nan):>10.3f}"
              f"{h['heldout']['kl']:>8.3f}{z['heldout']['kl']:>10.3f}")

    zr = arms["zrecon_a0.0"]
    print("\n--- REFERENCE POINTS " + "-" * 72)
    print(f"  no intervention at all         : wiki ppl {ppl['base']:.3f}")
    print(f"  AE reconstruction, NO steering : wiki ppl {ppl['zrecon_a0.0']:.3f} | "
          f"control top1chg {(zr['control'] or {}).get('top1_change', float('nan')):.3f} | "
          f"residual hit {(zr['heldout'] or {}).get('hit_rate', float('nan')):.3f}")
    print("      ^ the floor the latent arm carries before any steering happens")
    print(f"  cluster move rate (linear z)   : {results['cluster_move_rate']}")
    print(f"  cos(r_base, decode(r_latent))  : {m['cos_rh_dec_rz']:+.4f}")
    print(f"  mean-diff gap |r|              : base {m['gap_h']:.2f} / latent {m['gap_z']:.2f}"
          f"   (mean norms {m['mean_norm_h']:.1f} / {m['mean_norm_z']:.1f})")

    print("\n--- HOW TO READ " + "-" * 77)
    print('  hit  = the full-vocab argmax became the counterpart form. Strict success.')
    print("  flip = the pair ordering reversed only. This can be true while the model emits NEITHER")
    print("         form, so a high flip beside a low hit means the position was destroyed, not steered.")
    print("  alpha (linear)   = multiples of each space's OWN mean-difference norm, so alpha=1 moves a")
    print("         point by exactly the marked->unmarked gap in that space. That makes the columns")
    print("         comparable across base and latent.")
    print("  alpha (centroid) = interpolation fraction toward the target centroid; 1.0 lands on it.")


# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--report_from", default=None,
                    help="Re-render the tables from a finished results JSON and exit.")
    ap.add_argument("--axis", default="past", choices=sorted(AXES),
                    help="Contrastive axis to steer. " +
                         "; ".join(f"{k}: {v['blurb']}" for k, v in AXES.items()))
    ap.add_argument("--centroid", action="store_true",
                    help="Also run the centroid-steering arm (latent-only intervention).")
    ap.add_argument("--centroid_alphas", nargs="+", type=float, default=[0.25, 0.5, 1.0],
                    help="Interpolation fraction toward the target centroid; 1.0 lands on it.")
    ap.add_argument("--centroid_purity", type=float, default=0.6,
                    help="Min counterpart-slot share for a cluster to be an eligible target.")
    ap.add_argument("--centroid_min_fit", type=int, default=2,
                    help="Min fit positions in a cluster for it to be an eligible target.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Splice layer; defaults to the checkpoint's target_layer and must match it.")
    ap.add_argument("--alphas", nargs="+", type=float, default=None,
                    help="Linear step as a multiple of the space's own mean-difference norm.")
    ap.add_argument("--n_mine_tokens", type=int, default=4_000_000)
    ap.add_argument("--max_doc_tokens", type=int, default=256)
    ap.add_argument("--min_count", type=int, default=20, help="Min corpus count for both forms")
    ap.add_argument("--min_paradigm", type=int, default=3, help="Min count for base+s / base+ing")
    ap.add_argument("--n_words", type=int, default=200)
    ap.add_argument("--max_ctx_per_word", type=int, default=20)
    ap.add_argument("--top_k", type=int, default=50, help="Counterpart must be in the clean top-k")
    ap.add_argument("--min_ctx", type=int, default=1,
                    help="Min certified positions for a word to enter the fit/held-out split")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_ppl", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.report_from:
        render_report(json.load(open(args.report_from)))
        return
    if not args.checkpoint:
        raise SystemExit("[morph] --checkpoint is required (or use --report_from)")

    explicit_alphas = args.alphas is not None
    if args.smoke:
        args.n_mine_tokens, args.n_words = 1_000_000, 40
        args.max_ctx_per_word, args.n_ppl = 6, 8
        if not explicit_alphas:
            args.alphas = [0.5, 2.0]
        args.centroid_alphas = args.centroid_alphas[:2]
    if args.alphas is None:
        args.alphas = [0.25, 0.5, 1.0, 2.0, 4.0]

    axis = AXES[args.axis]
    slot_labels = list(axis["slots"])
    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae, norm, cfg = load_ae_from_checkpoint(Path(args.checkpoint), device)
    mean_t, std_t = norm["mean"].float(), norm["std"].float()
    model_name = cfg["extraction"]["model_name"]
    ckpt_layer = cfg["data"]["target_layer"]
    if args.layer is None:
        args.layer = ckpt_layer
    elif args.layer != ckpt_layer:
        raise SystemExit(f"[morph] --layer {args.layer} != checkpoint target_layer {ckpt_layer}")
    print(f"[morph] axis={args.axis} ({axis['blurb']}) | LM={model_name} layer={args.layer} "
          f"K={ae.n_clusters} ae_sha={shared.ae_fingerprint(ae)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = load_lm(model_name, device_map="auto")
    sanity_check_splice(lm, args.layer, tokenizer, device)

    # --- 1. mine -----------------------------------------------------------
    print("[morph] Opening prose sources:")
    sources = open_sources([s for s in DEFAULT_SOURCES if s["domain"] in MINE_DOMAINS])
    docs, counts, wiki_txt = mine_corpus(sources, args.n_mine_tokens, tokenizer,
                                         args.max_doc_tokens, args.n_ppl * 2)
    print(f"[morph] {len(docs)} docs, {len(counts)} distinct words")

    lexicon = build_lexicon(axis, counts, tokenizer, args.min_count, args.min_paradigm, args.n_words)
    if axis["irregulars"]:
        lexicon = add_irregulars(lexicon, counts, tokenizer, args.min_count)
    lexicon, collided = drop_role_collisions(lexicon)
    if collided:
        print(f"[morph] dropped {len(collided)} role-colliding forms: {collided[:8]}")
    n_irr = sum(1 for p in lexicon if p.get("irregular"))
    print(f"[morph] lexicon: {len(lexicon)} pairs ({n_irr} irregular). "
          f"e.g. {[(p['marked'], p['unmarked']) for p in lexicon[:8]]}")
    if len(lexicon) < 4:
        raise SystemExit("[morph] lexicon too small — raise --n_mine_tokens or lower --min_count")

    control_ids = {}
    for w in CONTROL_WORDS:
        tid = single_token(tokenizer, w)
        if tid is not None:
            control_ids[tid] = w
    by_word, controls = find_candidates(docs, lexicon, control_ids, args.max_ctx_per_word)
    pair_of = {p["marked"]: p for p in lexicon}

    cands = []
    for word, slots in by_word.items():
        for slot in ("marked", "unmarked"):
            for di, pos in slots[slot]:
                cands.append(dict(di=di, pos=pos, word=word, slot=slot))
    for di, pos, w, tid in controls:
        cands.append(dict(di=di, pos=pos, word=w, slot="control", tid=tid))
    print(f"[morph] {len(cands)} candidate positions before the contextual filter")

    cand_by_doc = defaultdict(list)
    for c in cands:
        cand_by_doc[c["di"]].append(c)

    # --- 2. clean pass: capture h, apply the contextual filter --------------
    hook = SplicingHook(lm, args.layer)
    captured: list[torch.Tensor] = []
    hook.activate(lambda hs: (captured.append(hs.detach()) or hs))

    kept, H, drop = [], [], Counter()
    doc_order = sorted(cand_by_doc)
    sub = [docs[d] for d in doc_order]
    for s, ids, am in tqdm(list(batch_docs(sub, tokenizer.pad_token_id, args.batch_size)),
                           desc="[morph] clean pass"):
        logits = lm(input_ids=ids.to(device), attention_mask=am.to(device)).logits
        hs = captured.pop()
        for b in range(ids.shape[0]):
            for c in cand_by_doc[doc_order[s + b]]:
                pre = c["pos"] - 1
                lg = logits[b, pre].float()
                if c["slot"] == "control":
                    if int(lg.argmax()) != c["tid"]:
                        drop["control_not_top1"] += 1
                        continue
                    ld = float("nan")
                else:
                    p = pair_of[c["word"]]
                    tid = p["marked_id"] if c["slot"] == "marked" else p["unmarked_id"]
                    other = p["unmarked_id"] if c["slot"] == "marked" else p["marked_id"]
                    if int(lg.argmax()) != tid:
                        drop["not_top1"] += 1
                        continue
                    if other not in set(lg.topk(args.top_k).indices.tolist()):
                        drop["counterpart_outside_topk"] += 1
                        continue
                    ld = float(lg[p["unmarked_id"]] - lg[p["marked_id"]])
                kept.append({**c, "ld_clean": ld, "idx": len(H)})
                H.append(hs[b, pre].float().cpu().numpy())
        captured.clear()
    hook.deactivate()

    n_ctrl = sum(1 for k in kept if k["slot"] == "control")
    print(f"[morph] {len(kept) - n_ctrl} target + {n_ctrl} control positions survived; "
          f"dropped {drop['not_top1']} not-top-1, {drop['counterpart_outside_topk']} "
          f"counterpart-outside-top-{args.top_k}")
    if len(kept) - n_ctrl < 20:
        raise SystemExit("[morph] too few certified positions — raise --n_mine_tokens")

    H = np.stack(H).astype(np.float32)
    Ht = torch.from_numpy(H).to(device)
    Z = ae.encoder((Ht - mean_t) / std_t)

    # --- 3. fit / held-out word split --------------------------------------
    per_word = Counter(k["word"] for k in kept if k["slot"] != "control")
    eligible = sorted(w for w, n in per_word.items() if n >= args.min_ctx)
    rng.shuffle(eligible)
    fit_words = set(eligible[: len(eligible) // 2])
    heldout_words = set(eligible[len(eligible) // 2:])
    print(f"[morph] {len(eligible)} eligible words -> {len(fit_words)} fit / {len(heldout_words)} held-out")
    if not fit_words or not heldout_words:
        raise SystemExit("[morph] not enough eligible words — raise --n_mine_tokens")

    fit_m = [k["idx"] for k in kept if k["slot"] == "marked" and k["word"] in fit_words]
    fit_u = [k["idx"] for k in kept if k["slot"] == "unmarked" and k["word"] in fit_words]
    if len(fit_m) < 5 or len(fit_u) < 5:
        raise SystemExit(f"[morph] fit set too thin ({len(fit_m)} {slot_labels[0]} / "
                         f"{len(fit_u)} {slot_labels[1]}) — raise --n_mine_tokens")

    def _dir(space: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Unit direction plus the mean-difference norm, which sets the alpha unit."""
        r = space[fit_m].mean(0) - space[fit_u].mean(0)
        return r / r.norm(), float(r.norm())

    r_h, scale_h = _dir(Ht)
    r_z, scale_z = _dir(Z)
    norm_h, norm_z = float(Ht.norm(dim=1).mean()), float(Z.norm(dim=1).mean())
    cos_hz = float(F.cosine_similarity(r_h, ae.decoder(r_z.unsqueeze(0))[0], dim=0))
    print(f"[morph] fit positions: {len(fit_m)} {slot_labels[0]} / {len(fit_u)} {slot_labels[1]}")
    print(f"[morph] mean-diff gap |r|: h={scale_h:.2f} z={scale_z:.2f}  "
          f"(mean |h|={norm_h:.1f} |z|={norm_z:.1f})  cos(r_h, dec(r_z))={cos_hz:+.3f}")

    # --- 4. centroid targets (only when --centroid) -------------------------
    C = ae.centroids.detach()
    lab0 = torch.cdist(Z, C).argmin(1)
    cent_target = torch.zeros_like(Z)     # destination centroid   (mode "toward")
    cent_delta = torch.zeros_like(Z)      # centroid difference    (mode "delta")
    has_target = torch.zeros(len(kept), dtype=torch.bool)
    elig_toward_u = elig_toward_m = None
    if args.centroid:
        # A cluster is an eligible destination for "steer toward the unmarked
        # form" when the FIT positions landing in it are mostly unmarked. Using
        # fit-only counts keeps the held-out evaluation honest.
        cm, cu = Counter(), Counter()
        for k in kept:
            if k["word"] not in fit_words or k["slot"] == "control":
                continue
            (cm if k["slot"] == "marked" else cu)[int(lab0[k["idx"]])] += 1
        eps = 0.5
        scores, tot = {}, {}
        for kcl in set(cm) | set(cu):
            n = cm[kcl] + cu[kcl]
            tot[kcl] = n
            scores[kcl] = (cu[kcl] + eps) / (n + 2 * eps)
        elig_toward_u = sorted(k for k, s in scores.items()
                               if s >= args.centroid_purity and tot[k] >= args.centroid_min_fit)
        elig_toward_m = sorted(k for k, s in scores.items()
                               if s <= 1 - args.centroid_purity and tot[k] >= args.centroid_min_fit)
        print(f"[morph] centroid targets: {len(elig_toward_u)} clusters enriched for "
              f"{slot_labels[1]}, {len(elig_toward_m)} for {slot_labels[0]} "
              f"(purity>={args.centroid_purity}, >={args.centroid_min_fit} fit points)")
        for pool, name in ((elig_toward_u, slot_labels[1]), (elig_toward_m, slot_labels[0])):
            if not pool:
                print(f"[morph]   ! no clusters enriched for {name} — centroid arm will be empty")
        Cu = C[elig_toward_u] if elig_toward_u else None
        Cm = C[elig_toward_m] if elig_toward_m else None
        for k in kept:
            pool = Cu if k["slot"] in ("marked", "control") else Cm
            if pool is None or len(pool) == 0:
                continue
            i = int(torch.cdist(Z[k["idx"]].unsqueeze(0), pool).argmin())
            cent_target[k["idx"]] = pool[i]
            cent_delta[k["idx"]] = pool[i] - C[int(lab0[k["idx"]])]
            has_target[k["idx"]] = True
        cov = float(has_target.float().mean())
        print(f"[morph] centroid coverage: {cov:.1%} of kept positions have an eligible target")

    # --- 5. steering passes: clean once per batch, then every arm x alpha ---
    SIGN = {"marked": 1.0, "unmarked": -1.0, "control": 1.0}
    kept_by_doc = defaultdict(list)
    for k in kept:
        kept_by_doc[k["di"]].append(k)
    order2 = sorted(kept_by_doc)
    sub2 = [docs[d] for d in order2]

    arms = ([("h", a) for a in args.alphas]
            + [("z", a) for a in args.alphas]
            + ([("zcent", a) for a in args.centroid_alphas] if args.centroid else [])
            + ([("zcdelta", a) for a in args.centroid_alphas] if args.centroid else [])
            + [("zrecon", 0.0)])
    patches = {}
    for arm, alpha in arms:
        if arm == "h":
            patches[(arm, alpha)] = PositionPatch("h", r_h, alpha * scale_h, ae, mean_t, std_t)
        elif arm == "z":
            patches[(arm, alpha)] = PositionPatch("z", r_z, alpha * scale_z, ae, mean_t, std_t)
        elif arm == "zcent":
            patches[(arm, alpha)] = CentroidPatch(ae, mean_t, std_t, alpha, mode="toward")
        elif arm == "zcdelta":
            patches[(arm, alpha)] = CentroidPatch(ae, mean_t, std_t, alpha, mode="delta")
        else:
            patches[(arm, alpha)] = PositionPatch("zrecon", r_z, 0.0, ae, mean_t, std_t)

    def split_of(word):
        return "fit" if word in fit_words else ("heldout" if word in heldout_words else "other")

    L = int(C.shape[1])
    rows = []
    for s, ids, am in tqdm(list(batch_docs(sub2, tokenizer.pad_token_id, args.batch_size)),
                           desc="[morph] steering"):
        ids_d, am_d = ids.to(device), am.to(device)
        signs = torch.zeros(ids.shape, dtype=torch.float32)
        cmask = torch.zeros(ids.shape, dtype=torch.bool)
        ctgt = torch.zeros(ids.shape + (L,), dtype=torch.float32) if args.centroid else None
        cdel = torch.zeros(ids.shape + (L,), dtype=torch.float32) if args.centroid else None
        meta = []
        for b in range(ids.shape[0]):
            for k in kept_by_doc[order2[s + b]]:
                signs[b, k["pos"] - 1] = SIGN[k["slot"]]
                if args.centroid and bool(has_target[k["idx"]]):
                    cmask[b, k["pos"] - 1] = True
                    ctgt[b, k["pos"] - 1] = cent_target[k["idx"]].cpu()
                    cdel[b, k["pos"] - 1] = cent_delta[k["idx"]].cpu()
                meta.append((b, k))
        if not meta:
            continue
        bs = torch.tensor([b for b, _ in meta], device=device)
        pres = torch.tensor([k["pos"] - 1 for _, k in meta], device=device)

        # Gather only the rows we score: full (B, T, V) logits in float would be
        # gigabytes at a 262k-vocab model.
        out = lm(input_ids=ids_d, attention_mask=am_d).logits
        lgc = out[bs, pres].float()
        del out
        logp_clean = F.log_softmax(lgc, dim=-1)

        for (arm, alpha), patch in patches.items():
            if isinstance(patch, CentroidPatch):
                if not bool(cmask.any()):
                    continue
                patch.mask = cmask
                patch.targets = ctgt if patch.mode == "toward" else cdel
            else:
                patch.signs = signs
            hook.activate(patch)
            out = lm(input_ids=ids_d, attention_mask=am_d).logits
            lgs = out[bs, pres].float()
            del out
            hook.deactivate()
            for j, (_, k) in enumerate(meta):
                if arm in ("zcent", "zcdelta") and not bool(has_target[k["idx"]]):
                    continue          # no eligible destination; not a no-op result
                delta = flip = hit = None
                if k["slot"] != "control":
                    p = pair_of[k["word"]]
                    ld_s = float(lgs[j, p["unmarked_id"]] - lgs[j, p["marked_id"]])
                    sgn = SIGN[k["slot"]]
                    delta = sgn * (ld_s - k["ld_clean"])
                    flip = float(sgn * k["ld_clean"] < 0 and sgn * ld_s > 0)
                    counterpart = p["unmarked_id"] if k["slot"] == "marked" else p["marked_id"]
                    hit = float(int(lgs[j].argmax()) == counterpart)
                rows.append(dict(
                    arm=arm, alpha=alpha, ex=k["idx"], word=k["word"], slot=k["slot"],
                    split=split_of(k["word"]), delta=delta, flip=flip, hit=hit,
                    top1_change=float(int(lgs[j].argmax()) != int(lgc[j].argmax())),
                    kl=kl_div(logp_clean[j], lgs[j]),
                ))

    # --- 6. cluster movement (linear z arm, no LM needed) -------------------
    sgn_vec = torch.tensor([SIGN[k["slot"]] for k in kept], device=device,
                           dtype=torch.float32).unsqueeze(1)
    cluster_move = {}
    for alpha in args.alphas:
        lab1 = torch.cdist(Z - alpha * scale_z * sgn_vec * r_z, C).argmin(1)
        cluster_move[f"a{alpha}"] = round(float((lab0 != lab1).float().mean()), 4)

    # --- 7. global damage: the step applied at EVERY position ---------------
    from geoae.interp import neuronlens as nl
    ppl_txt = wiki_txt[: args.n_ppl]
    ppl = {"base": round(shared.perplexity(lm, tokenizer, ppl_txt, device), 3)}
    for arm, alpha in arms:
        if arm == "h":
            fn = nl.make_h_steer((alpha * scale_h * r_h).cpu().numpy(), 1.0, device)
        elif arm == "z":
            fn = nl.make_z_steer((alpha * scale_z * r_z).cpu().numpy(), 1.0, ae,
                                 mean_t, std_t, device)
        elif arm in ("zcent", "zcdelta"):
            if not elig_toward_u:
                continue
            fn = make_global_centroid(ae, mean_t, std_t, alpha, C[elig_toward_u], C,
                                      mode="toward" if arm == "zcent" else "delta")
        else:
            fn = nl.make_z_gate(np.full(L, np.inf), np.full(L, -np.inf), np.zeros(L),
                                ae, mean_t, std_t, device, gate=False)
        hook.activate(fn)
        ppl[f"{arm}_a{alpha}"] = round(shared.perplexity(lm, tokenizer, ppl_txt, device), 3)
        hook.deactivate()

    # --- 8. aggregate ------------------------------------------------------
    def agg(arm, alpha, split=None, slot=None, control=False):
        sel = [r for r in rows if r["arm"] == arm and r["alpha"] == alpha
               and (r["slot"] == "control") == control
               and (split is None or r["split"] == split)
               and (slot is None or r["slot"] == slot)]
        if not sel:
            return None
        out = {"n": len(sel),
               "top1_change": round(float(np.mean([r["top1_change"] for r in sel])), 4),
               "kl": round(float(np.mean([r["kl"] for r in sel])), 4)}
        if not control:
            out["delta_logit_diff"] = round(float(np.mean([r["delta"] for r in sel])), 4)
            out["flip_rate"] = round(float(np.mean([r["flip"] for r in sel])), 4)
            out["hit_rate"] = round(float(np.mean([r["hit"] for r in sel])), 4)
        return out

    results = {
        "meta": {
            "axis": args.axis, "axis_blurb": axis["blurb"], "slot_labels": slot_labels,
            "centroid_mode": bool(args.centroid),
            "checkpoint": args.checkpoint, "model_name": model_name, "layer": args.layer,
            "ae_sha": shared.ae_fingerprint(ae), "n_clusters": int(ae.n_clusters),
            "alphas": args.alphas, "seed": args.seed,
            "n_docs": len(docs), "n_lexicon": len(lexicon), "n_irregular": n_irr,
            "n_role_collisions_dropped": len(collided),
            "filter_drops": dict(drop),
            "n_kept_target": len(kept) - n_ctrl, "n_kept_control": n_ctrl,
            "n_fit_words": len(fit_words), "n_heldout_words": len(heldout_words),
            "alpha_unit": "mean-difference norm |r| in each space",
            "gap_h": round(scale_h, 3), "gap_z": round(scale_z, 3),
            "mean_norm_h": round(norm_h, 3), "mean_norm_z": round(norm_z, 3),
            "cos_rh_dec_rz": round(cos_hz, 4),
            "n_fit_marked": len(fit_m), "n_fit_unmarked": len(fit_u),
            "top_k": args.top_k, "min_count": args.min_count,
            "sources": [s["domain"] for s in sources],
        },
        "lexicon": [{k: v for k, v in p.items() if not k.endswith("_id")} for p in lexicon],
        "ppl": ppl,
        "cluster_move_rate": cluster_move,
        "centroid_alphas": args.centroid_alphas if args.centroid else [],
        "centroid_coverage": float(has_target.float().mean()) if args.centroid else None,
        "arms": {},
    }
    if args.centroid:
        results["meta"]["centroid_target_clusters"] = {
            slot_labels[1]: len(elig_toward_u or []), slot_labels[0]: len(elig_toward_m or [])}
    for arm, alpha in arms:
        results["arms"][f"{arm}_a{alpha}"] = {
            "heldout": agg(arm, alpha, split="heldout"),
            "fit": agg(arm, alpha, split="fit"),
            "heldout_marked_slot": agg(arm, alpha, split="heldout", slot="marked"),
            "heldout_unmarked_slot": agg(arm, alpha, split="heldout", slot="unmarked"),
            "control": agg(arm, alpha, control=True),
        }

    out_path = Path(args.out) if args.out else Path(
        f"results_morph_{args.axis}_{Path(args.checkpoint).parent.name}.json")
    results["rows"] = rows
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    render_report(results)
    print(f"\n[morph] Saved -> {out_path}")


if __name__ == "__main__":
    main()
