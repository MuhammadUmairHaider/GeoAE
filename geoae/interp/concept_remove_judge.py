"""
LLM judge for concept_remove_generate.py: does an edit that's meant to REMOVE class c from the
model's behaviour actually stop the continuation reading as class c (on TARGET prompts), while
leaving COMPLEMENT prompts (other classes) reading as their own class?

Every continuation -- unsteered, the z-substrate's own recon baseline (--z_mode splice only),
and every (substrate, op, alpha) edit -- is judged by the SAME forced choice: the prompt's
TRUE class plus 3 distractors from the dataset's class list, chosen deterministically per
(dataset, true_class, prompt index) via concept_steer_judge.pick_candidates (reused UNCHANGED:
its crc32 key is exactly (dataset, concept, prompt-index), and here "concept" is simply the
PROMPT's own true class and "prompt index" its fixed position in that concept-run's prompt
list -- both are properties of the prompt alone, so the same call gives the identical
candidate set for every substrate/op/alpha judging that same prompt).

Headline metrics (all "clean", i.e. read against FLUENT continuations, per 100, net of the
same measure on the UNSTEERED continuation -- see the scoring functions below):
  clean_removal   target prompts that stopped reading as c, beyond the unsteered rate
  collateral      complement prompts that broke (non-fluent OR off their own class), beyond
                  the unsteered rate -- LOWER is better
  clean_selectivity = clean_removal - collateral
plus the raw (non-fluency-gated) tgt_drop/comp_drop/selectivity, for reference.

    set -a; source .env; set +a
    python -u -m geoae.interp.concept_remove_judge --gen eval_out/concept_remove_generate.json \
        --provider openrouter --model google/gemini-2.5-flash-lite --out eval_out/concept_remove_judge.json
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Reused, UNMODIFIED, from concept_steer_judge.py / llm_judge.py (per the no-modify constraint
# on those files -- these are their own module-level, importable names).
from geoae.interp.concept_steer_judge import boot_mean_ci, paired_diff, pick_candidates
from geoae.interp.llm_judge import LLM

LETTERS = "ABCD"
FLUENT_THRESHOLD = 0.6
OP_ALPHA_ORDER_HINT = ("rm_range_comp",)   # printed first; st_range_a*/st_transport_a* follow, by alpha


def build_prompt(prompt: str, cont: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(options))
    return (
        "A language model was given a prompt and, from that point, wrote a continuation.\n\n"
        f"Prompt: {prompt!r}\nContinuation: {cont!r}\n\n"
        f"Which topic is the CONTINUATION about? If none clearly, pick the closest.\n\n{opts}\n\n"
        "Answer with a single letter (A, B, C or D) and nothing else."
    )


def sort_keys(keys) -> list[str]:
    """rm_range_comp first, then st_range_a*/st_transport_a* ordered by op then alpha."""
    def keyfn(k):
        if k in OP_ALPHA_ORDER_HINT:
            return (0, k, 0.0)
        op, _, a = k.rpartition("_a")
        try:
            return (1, op, float(a))
        except ValueError:
            return (2, k, 0.0)
    return sorted(keys, key=keyfn)


# ---------------------------------------------------------------------------
# Pure, unit-testable scoring (numpy in, dict/float out)
# ---------------------------------------------------------------------------

def drop_and_selectivity(hit_tgt_uns, hit_tgt_edit, hit_comp_uns, hit_comp_edit) -> dict:
    """Raw (non-fluency-gated) tgt_drop/comp_drop/selectivity: kept_* = share judged as the
    prompt's own true class; drop = unsteered kept - edited kept (positive = removed more)."""
    kt_u, kt_e = float(np.mean(hit_tgt_uns)), float(np.mean(hit_tgt_edit))
    kc_u, kc_e = float(np.mean(hit_comp_uns)), float(np.mean(hit_comp_edit))
    tgt_drop, comp_drop = kt_u - kt_e, kc_u - kc_e
    return {"kept_target_unsteered": kt_u, "kept_target_edited": kt_e, "tgt_drop": tgt_drop,
            "kept_comp_unsteered": kc_u, "kept_comp_edited": kc_e, "comp_drop": comp_drop,
            "selectivity": tgt_drop - comp_drop}


def clean_removal_score(hit_tgt_edit, d2_tgt_edit, hit_tgt_uns, d2_tgt_uns,
                         threshold: float = FLUENT_THRESHOLD) -> float:
    """Share of TARGET prompts with a FLUENT continuation NOT judged as c (edited), minus the
    same share on the unsteered continuations. Higher = more genuine removal, beyond whatever
    the model already didn't say unprompted."""
    hit_tgt_edit, hit_tgt_uns = np.asarray(hit_tgt_edit, bool), np.asarray(hit_tgt_uns, bool)
    fluent_e = np.asarray(d2_tgt_edit, float) >= threshold
    fluent_u = np.asarray(d2_tgt_uns, float) >= threshold
    removed_e = float(np.mean(fluent_e & ~hit_tgt_edit))
    removed_u = float(np.mean(fluent_u & ~hit_tgt_uns))
    return removed_e - removed_u


def collateral_score(hit_comp_edit, d2_comp_edit, hit_comp_uns, d2_comp_uns,
                      threshold: float = FLUENT_THRESHOLD) -> float:
    """Share of COMPLEMENT prompts that BROKE (non-fluent OR no longer judged as their own
    class) under the edit, minus the same share on the unsteered continuations. LOWER = less
    collateral damage."""
    hit_comp_edit, hit_comp_uns = np.asarray(hit_comp_edit, bool), np.asarray(hit_comp_uns, bool)
    fluent_e = np.asarray(d2_comp_edit, float) >= threshold
    fluent_u = np.asarray(d2_comp_uns, float) >= threshold
    bad_e = float(np.mean(~fluent_e | ~hit_comp_edit))
    bad_u = float(np.mean(~fluent_u | ~hit_comp_uns))
    return bad_e - bad_u


def matched_at_removal_level(removal_by_cell, collateral_by_cell, target_removal: float) -> float:
    """Interpolate collateral at a target clean_removal level across one substrate's (op,
    alpha) cells pooled over concepts. The cells mix operators, so their order says nothing
    about strength: sort them by removal and interpolate (no running-max, which would carry one
    operator's removal onto another's cells). NaN outside the observed removal range."""
    rem = np.asarray(removal_by_cell, dtype=float)
    order = np.argsort(rem, kind="stable")
    rem_o = rem[order]
    col_o = np.asarray(collateral_by_cell, dtype=float)[order]
    if target_removal < rem_o.min() or target_removal > rem_o.max():
        return float("nan")
    return float(np.interp(target_removal, rem_o, col_o))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="output of concept_remove_generate.py")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--cache_dir", default="cache/llm_judge")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    G = json.load(open(args.gen))
    dataset = G["meta"]["datasets"]
    z_mode = G["meta"]["z_mode"]
    all_classes = list(G["meta"]["class_descriptions"].keys())
    class_desc = G["meta"]["class_descriptions"]
    concepts = G["concepts"]
    llm = LLM(args.provider, args.model, Path(args.cache_dir), dry_run=args.dry_run)

    jobs = []   # (cid, src, i, cand, order, prompt_text); src: "unsteered" | "z_recon_baseline" | (sub, key)
    for cid, rec in concepts.items():
        prompts, true_names = rec["prompts"]["text"], rec["prompts"]["true_class_name"]
        n = len(prompts)
        option_cache: dict[int, tuple] = {}

        def opts_for(i):
            if i not in option_cache:
                # SAME crc32 pattern/function as concept_steer_generate's steering test, just
                # keyed by (dataset, this prompt's true class, its fixed position) -- both are
                # properties of the prompt alone, so identical for every substrate/op/alpha.
                option_cache[i] = pick_candidates(dataset, true_names[i], i, all_classes)
            return option_cache[i]

        sources = {"unsteered": rec["unsteered"]["text"]}
        if "z_recon_baseline" in rec:
            sources["z_recon_baseline"] = rec["z_recon_baseline"]["text"]
        for sub in ("h", "z"):
            for key, cell in rec.get("edits", {}).get(sub, {}).items():
                sources[(sub, key)] = cell["text"]
        for src, texts in sources.items():
            for i in range(n):
                cand, order = opts_for(i)
                shown = [class_desc[cand[k]] for k in order]
                jobs.append((cid, src, i, cand, order, build_prompt(prompts[i], texts[i], shown)))
    print(f"[judge] {len(jobs):,} judgements across {len(concepts)} concepts")

    # DEDUPE identical prompt strings before calling the LLM -- e.g. the unsteered text is
    # judged once per concept but may be byte-identical to a z_recon_baseline in delta mode's
    # absence, and options are shared across every source for a given prompt index (copied
    # inline pattern from concept_steer_judge.py's main(), which isn't a standalone function).
    unique_prompts = list(dict.fromkeys(j[5] for j in jobs))
    prompt_idx = {t: i for i, t in enumerate(unique_prompts)}
    print(f"[judge] {len(unique_prompts):,} unique prompts ({args.model}); cached ones are free")

    def ask(text):
        return "A" if args.dry_run else llm.ask(text, max_tokens=5)

    with ThreadPoolExecutor(args.workers) as ex:
        unique_answers = list(ex.map(ask, unique_prompts))
    matches = [re.search(r"[ABCD]", (ans or "").upper()) for ans in unique_answers]
    bad = sum(1 for m in matches if not m)
    print(f"[judge] unparseable answers: {bad} of {len(unique_prompts)} unique (counted as misses)")

    hits: dict = {}   # cid -> src -> {i: hit}
    for (cid, src, i, cand, order, text) in jobs:
        m = matches[prompt_idx[text]]
        hit = float(bool(m) and order[LETTERS.index(m.group(0))] == 0)
        hits.setdefault(cid, {}).setdefault(src, {})[i] = hit

    def hit_arr(cid, src, n):
        d = hits.get(cid, {}).get(src, {})
        return np.array([d.get(i, np.nan) for i in range(n)])

    # ---- per-concept, per-(substrate,op/alpha) scoring ----------------------------------
    all_keys = sorted({key for rec in concepts.values() for sub in ("h", "z")
                       for key in rec.get("edits", {}).get(sub, {})})
    all_keys = sort_keys(all_keys)
    per_cell: dict = {"h": {k: [] for k in all_keys}, "z": {k: [] for k in all_keys}}   # -> list of per-concept dicts
    concept_order: dict = {"h": {k: [] for k in all_keys}, "z": {k: [] for k in all_keys}}

    for cid, rec in concepts.items():
        n = len(rec["prompts"]["text"])
        n_tgt = rec["prompts"]["n_target"]
        tgt, comp = slice(0, n_tgt), slice(n_tgt, n)
        d2_uns = np.asarray(rec["unsteered"]["distinct2"])
        hit_uns = hit_arr(cid, "unsteered", n)
        for sub in ("h", "z"):
            base_src = "z_recon_baseline" if (sub == "z" and z_mode == "splice" and "z_recon_baseline" in rec) \
                else "unsteered"
            hit_base = hit_uns if base_src == "unsteered" else hit_arr(cid, "z_recon_baseline", n)
            d2_base = d2_uns if base_src == "unsteered" else np.asarray(rec["z_recon_baseline"]["distinct2"])
            for key, cell in rec.get("edits", {}).get(sub, {}).items():
                hit_e = hit_arr(cid, (sub, key), n)
                d2_e = np.asarray(cell["distinct2"])
                sc = drop_and_selectivity(hit_base[tgt], hit_e[tgt], hit_base[comp], hit_e[comp])
                sc["clean_removal"] = clean_removal_score(hit_e[tgt], d2_e[tgt], hit_base[tgt], d2_base[tgt])
                sc["collateral"] = collateral_score(hit_e[comp], d2_e[comp], hit_base[comp], d2_base[comp])
                sc["clean_selectivity"] = sc["clean_removal"] - sc["collateral"]
                sc["fluent_target"] = float(np.mean(d2_e[tgt] >= FLUENT_THRESHOLD))
                sc["fluent_comp"] = float(np.mean(d2_e[comp] >= FLUENT_THRESHOLD))
                sc["ppl_ratio"] = float(np.exp(np.mean(cell["log_ppl_ratio"])))
                per_cell[sub][key].append(sc)
                concept_order[sub][key].append(cid)

    # ---- pooled table (mean [95% CI over concepts]) --------------------------------------
    FIELDS = ("kept_target_unsteered", "kept_target_edited", "kept_comp_unsteered", "kept_comp_edited",
              "tgt_drop", "comp_drop", "clean_removal", "collateral", "clean_selectivity",
              "fluent_target", "fluent_comp", "ppl_ratio")
    pooled = {"h": {}, "z": {}}
    print("\n[judge] pooled over concepts (mean [95% CI]); h = base residual, z = bypass AE latent")
    for key in all_keys:
        print(f"  {key}:")
        for sub in ("h", "z"):
            rows = per_cell[sub][key]
            if not rows:
                continue
            pooled[sub][key] = {}
            summary_bits = []
            for f in FIELDS:
                vals = np.array([r[f] for r in rows])
                m, lo, hi = boot_mean_ci(vals, rng, args.n_boot)
                pooled[sub][key][f] = {"mean": m, "ci": [lo, hi]}
                summary_bits.append(f"{f} {100*m:+.1f}" if f != "ppl_ratio" else f"ppl x{m:.2f}")
            print(f"    {sub}: " + "  ".join(summary_bits) + f"   (n={len(rows)})")

    # ---- paired per-concept z - h difference (clean_selectivity & selectivity) ----------
    print("\n[judge] paired per-concept z - h difference (centred bootstrap CI, Wilcoxon p)")
    paired = {}
    for key in all_keys:
        h_ids, z_ids = concept_order["h"][key], concept_order["z"][key]
        common = [cid for cid in h_ids if cid in z_ids]
        if not common:
            continue
        h_by_cid = dict(zip(h_ids, per_cell["h"][key]))
        z_by_cid = dict(zip(z_ids, per_cell["z"][key]))
        paired[key] = {}
        for metric in ("clean_selectivity", "selectivity"):
            h_vals = np.array([h_by_cid[cid][metric] for cid in common])
            z_vals = np.array([z_by_cid[cid][metric] for cid in common])
            diff, mask = paired_diff(z_vals, h_vals)
            n = len(diff)
            m, lo, hi = boot_mean_ci(diff, rng, args.n_boot)
            eps = 1e-9
            wins, ties, loses = int((diff > eps).sum()), int((np.abs(diff) <= eps).sum()), int((diff < -eps).sum())
            p_val = float("nan")
            if n >= 1 and np.any(np.abs(diff) > eps):
                from scipy.stats import wilcoxon
                try:
                    p_val = float(wilcoxon(diff).pvalue)
                except ValueError:
                    p_val = float("nan")
            paired[key][metric] = {"mean_diff": m, "ci": [lo, hi], "wilcoxon_p": p_val,
                                   "n": n, "z_wins": wins, "ties": ties, "z_loses": loses}
        cs = paired[key]["clean_selectivity"]
        print(f"  {key:<20s} clean_sel z-h {100*cs['mean_diff']:+.1f} [{100*cs['ci'][0]:+.0f},{100*cs['ci'][1]:+.0f}]  "
              f"p={cs['wilcoxon_p']:.3f}  z wins/ties/loses {cs['z_wins']}/{cs['ties']}/{cs['z_loses']}  (n={cs['n']})")

    # ---- collateral at MATCHED clean_removal (20/40/60 per 100), per substrate ------------
    print("\n[judge] collateral at MATCHED clean_removal (pooled across ops/alphas)")
    matched = {}
    for sub in ("h", "z"):
        keys_here = [k for k in all_keys if k in pooled[sub]]
        if not keys_here:
            continue
        removal = [pooled[sub][k]["clean_removal"]["mean"] for k in keys_here]
        collat = [pooled[sub][k]["collateral"]["mean"] for k in keys_here]
        matched[sub] = {}
        bits = []
        for target in (0.2, 0.4, 0.6):
            v = matched_at_removal_level(removal, collat, target)
            matched[sub][str(target)] = v
            bits.append(f"{int(100*target)}%: {100*v:.1f}" if v == v else f"{int(100*target)}%: (out of range)")
        print(f"  {sub}: " + "  ".join(bits))

    res = {"meta": {**vars(args), "gen": args.gen, "dataset": dataset, "z_mode": z_mode,
                    "n_jobs": len(jobs), "n_unique_prompts": len(unique_prompts)},
          "pooled": pooled, "paired": paired, "matched_at_removal": matched,
          "per_concept": {sub: {key: {concepts[cid]["class_name"]: row for cid, row in
                                      zip(concept_order[sub][key], per_cell[sub][key])}
                                for key in all_keys} for sub in ("h", "z")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n[judge] wrote {args.out}  (calls {llm.n_calls}, cached {llm.n_cached})")


if __name__ == "__main__":
    main()
