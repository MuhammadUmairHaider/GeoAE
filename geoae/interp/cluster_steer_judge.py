"""
LLM judge for cluster_steer_generate.py: did the steered continuation take on the TARGET
cluster's content? Forced choice among the target and the same 3 distractor clusters the
automatic score used (chance .25), shown as short profiles: the words the cluster raises
next and the words its members sit on. The judge never sees alpha or the codebook name.

alpha 0 judges the UNSTEERED continuation against each target's candidate set — the chance
anchor. Codebooks are compared per alpha and at matched fluency cost (continuation
perplexity under the unedited model, from the generation file), CIs over targets.

Uses geoae.interp.llm_judge.LLM (disk-cached by prompt, so re-runs are free):
    set -a; source .env; set +a
    python -u -m geoae.interp.cluster_steer_judge --gen eval_out/cluster_steer_generate.json \
        --provider openrouter --model google/gemini-2.5-flash-lite --out eval_out/cluster_steer_judge.json
"""
from __future__ import annotations

import argparse
import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from geoae.interp.llm_judge import LLM

LETTERS = "ABCD"


def profile_text(p: dict) -> str:
    nxt = ", ".join(repr(w.strip()) for w in p["raises_next"][:15] if w.strip())
    on = ", ".join(repr(w.strip()) for w in p["members_on"][:8] if w.strip())
    return f"tends to write next: {nxt}; its examples sit on words like: {on}"


def build_prompt(prompt: str, cont: str, profiles: list[dict]) -> str:
    opts = "\n".join(f"{LETTERS[k]}) {profile_text(p)}" for k, p in enumerate(profiles))
    return (
        "A language model's internal states were grouped into clusters. Each cluster below is "
        "described by the words the model tends to write next when in that cluster, and the words "
        "its examples sit on.\n\n"
        f"{opts}\n\n"
        "The model was then given a prompt and wrote a continuation. Which cluster's words and "
        "subject matter does the CONTINUATION lean toward most? If none clearly, pick the closest.\n\n"
        f"Prompt: {prompt!r}\nContinuation: {cont!r}\n\n"
        "Answer with a single letter (A, B, C or D) and nothing else."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="output of cluster_steer_generate.py")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--cache_dir", default="cache/llm_judge")
    ap.add_argument("--max_alpha", type=float, default=0.75, help="judge alphas up to this (beyond, text degenerates)")
    ap.add_argument("--n_prompts", type=int, default=20, help="judge the first N prompts per target")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="build prompts, call nothing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    G = json.load(open(args.gen))
    prompts = G["meta"]["prompts"][: args.n_prompts]
    unsteered = G["generations"]["unsteered"]["text"]
    llm = LLM(args.provider, args.model, Path(args.cache_dir), dry_run=args.dry_run)

    jobs = []   # (arm, alpha, target_idx, prompt_idx, order, prompt_text)
    for arm, rec in G["arms"].items():
        profs = [rec["profiles"][str(t)] for t in rec["targets"]]
        dis = rec["distractors"]
        for a in sorted(rec["alphas"], key=float):
            if float(a) > args.max_alpha:
                continue
            texts = rec["alphas"][a]["text"]
            for i in range(len(profs)):
                for p in range(len(prompts)):
                    cont = unsteered[p] if float(a) == 0.0 else texts[i][p]
                    cand = [i] + list(dis[i][p])
                    # zlib.crc32, NOT hash(): Python's str hash is salted per process, so hash()
                    # reshuffled the options on every run — prompts changed, the cache missed,
                    # and judgements were not reproducible.
                    order = list(np.random.default_rng(zlib.crc32(f"{arm}|{i}|{p}".encode())).permutation(4))
                    shown = [profs[cand[k]] for k in order]
                    jobs.append((arm, a, i, p, order, build_prompt(prompts[p], cont, shown)))
    print(f"[judge] {len(jobs):,} judgements ({args.model}); cached ones are free")

    def ask(job):
        if args.dry_run:
            return "A"
        return llm.ask(job[5], max_tokens=5)

    with ThreadPoolExecutor(args.workers) as ex:
        answers = list(ex.map(ask, jobs))

    res = {"meta": {**vars(args), "gen": args.gen}, "arms": {}}
    bad = 0
    for (arm, a, i, p, order, _), ans in zip(jobs, answers):
        m = re.search(r"[ABCD]", (ans or "").upper())
        if not m:
            bad += 1
        hit = float(bool(m) and order[LETTERS.index(m.group(0))] == 0)   # position of the true target
        r = res["arms"].setdefault(arm, {}).setdefault(a, {})
        r.setdefault("hit", {}).setdefault(i, {})[p] = hit
    print(f"[judge] unparseable answers: {bad} (counted as misses)")

    def boot_rows(M):
        idx = rng.integers(0, len(M), size=(args.n_boot, len(M)))
        b = M.mean(1)[idx].mean(1)
        return float(M.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    summary = {}
    print("\n[judge] LLM forced choice, target vs 3 other clusters (chance .25), mean [95% CI over targets]")
    for arm, by_a in res["arms"].items():
        summary[arm] = {}
        for a, r in sorted(by_a.items(), key=lambda kv: float(kv[0])):
            M = np.array([[r["hit"][i][p] for p in sorted(r["hit"][i])] for i in sorted(r["hit"])])
            gl = np.asarray(G["arms"][arm]["alphas"][a]["log_ppl_ratio"])[:, : M.shape[1]]
            summary[arm][a] = {"hit": boot_rows(M), "ppl_x": float(np.exp(gl.mean()))}
            h = summary[arm][a]["hit"]
            print(f"  {arm:6s} a{a:<5} {h[0]:.3f} [{h[1]:.3f},{h[2]:.3f}]   ppl x{summary[arm][a]['ppl_x']:.2f}")
            r["hit"] = M.tolist()
    res["summary"] = summary

    print("\n[judge] at MATCHED fluency cost (continuation perplexity x): LLM-judged hit rate [95% CI]")
    matched = {}
    for lev in (1.1, 1.25, 1.5, 2.0):
        row = {}
        for arm, by_a in res["arms"].items():
            al = sorted(by_a, key=float)
            Ms = [np.asarray(by_a[a]["hit"]) for a in al]
            Ls = [np.asarray(G["arms"][arm]["alphas"][a]["log_ppl_ratio"])[:, : Ms[0].shape[1]] for a in al]

            def at(rows=None):
                pick = (lambda v: v) if rows is None else (lambda v: v[rows])
                xs = np.maximum.accumulate(np.array([pick(Lm).mean() for Lm in Ls]))
                ys = np.array([pick(M).mean() for M in Ms])
                return float(np.interp(np.log(lev), xs, ys, left=np.nan, right=np.nan))
            est = at()
            nt = Ms[0].shape[0]
            bs = [at(rng.integers(0, nt, nt)) for _ in range(500)]
            row[arm] = (est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))
        matched[str(lev)] = row
        print(f"  ppl x{lev:<4}  " + "   ".join(
            f"{k} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" if v[0] == v[0] else f"{k} (out of range)"
            for k, v in row.items()))
    res["matched_fluency"] = matched

    # Perplexity FALLS for repetition loops ('the the the' is easy to predict), so the share of
    # degenerate continuations (distinct-bigram share < 0.6) is the second, fairer fluency axis.
    print("\n[judge] at MATCHED share of degenerate continuations: LLM-judged hit rate [95% CI]")
    matched_d = {}
    for lev in (0.1, 0.2, 0.3, 0.5):
        row = {}
        for arm, by_a in res["arms"].items():
            al = sorted(by_a, key=float)
            Ms = [np.asarray(by_a[a]["hit"]) for a in al]
            Ds = [(np.asarray(G["arms"][arm]["alphas"][a]["distinct2"])[:, : Ms[0].shape[1]] < 0.6).astype(float)
                  for a in al]

            def at(rows=None):
                pick = (lambda v: v) if rows is None else (lambda v: v[rows])
                xs = np.maximum.accumulate(np.array([pick(Dm).mean() for Dm in Ds]))
                ys = np.array([pick(M).mean() for M in Ms])
                return float(np.interp(lev, xs, ys, left=np.nan, right=np.nan))
            est = at()
            nt = Ms[0].shape[0]
            bs = [at(rng.integers(0, nt, nt)) for _ in range(500)]
            row[arm] = (est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))
        matched_d[str(lev)] = row
        print(f"  degenerate {int(100 * lev):>2d}%  " + "   ".join(
            f"{k} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" if v[0] == v[0] else f"{k} (out of range)"
            for k, v in row.items()))
    res["matched_degenerate"] = matched_d

    # The headline scoring after a blind human read of the text: the raw hit rate mixes ~25%
    # chance, per-cell judge priors (some profiles win on unsteered text) and loops of the
    # target's own word, which the judge counts as hits. So score only FLUENT continuations
    # (distinct-bigram share >= 0.6), net of the SAME cell's unsteered judgement; and report
    # usable = fluent share x net hit (on-target continuations per 100 attempts, above baseline).
    print("\n[judge] FLUENT continuations only, net of the same cell's unsteered hit (per 100) [95% CI over targets]")
    print(f"  {'alpha':>6}  " + "   ".join(f"{arm}: fluent share / net hit / usable" for arm in res["arms"]))
    net = {}
    alphas = sorted({a for by_a in res["arms"].values() for a in by_a if float(a) > 0}, key=float)
    for a in alphas:
        cells = []
        for arm, by_a in res["arms"].items():
            if a not in by_a or "0.0" not in by_a:
                cells.append(f"{arm}: —"); continue
            h, h0 = np.asarray(by_a[a]["hit"]), np.asarray(by_a["0.0"]["hit"])
            ok = np.asarray(G["arms"][arm]["alphas"][a]["distinct2"])[:, : h.shape[1]] >= 0.6
            d = np.where(ok, h - h0, np.nan)
            with np.errstate(all="ignore"):
                per_t = np.nanmean(d, 1)
            per_t = per_t[~np.isnan(per_t)]
            bs = [np.mean(per_t[rng.integers(0, len(per_t), len(per_t))]) for _ in range(1000)] if len(per_t) else [np.nan]
            m = float(np.nanmean(d)) if ok.any() else float("nan")
            net.setdefault(a, {})[arm] = {"fluent_share": float(ok.mean()), "net_hit": m,
                                          "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                          "usable": float(ok.mean() * m) if m == m else float("nan")}
            v = net[a][arm]
            cells.append(f"{arm}: {100 * v['fluent_share']:3.0f}% / {100 * m:+5.1f} [{100 * v['ci'][0]:+.0f},"
                         f"{100 * v['ci'][1]:+.0f}] / {100 * v['usable']:4.1f}")
        print(f"  {a:>6}  " + "   ".join(cells))
    res["net_fluent"] = net
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"[judge] wrote {args.out}  (calls {llm.n_calls}, cached {llm.n_cached})")


if __name__ == "__main__":
    main()
