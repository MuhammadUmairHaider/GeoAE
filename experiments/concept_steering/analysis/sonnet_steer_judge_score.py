"""Score a blind agent's judgements of a stratified sample of concept-steering continuations (long generation).

Sample (built 10 Oct from concept_steer_generate_sae_db14.json): per concept 2 of the 20 neutral prompts (seeded), every
handle at alphas 0.3 / 0.5 / 0.75 / 1.0 plus the unsteered continuation; identical (prompt, continuation, options) strings
are judged once. The agent answered, per item, the two questions of concept_steer_judge.py: which of four topic
descriptions the continuation leans toward (target + 3 other classes, chance 0.25) and whether it is fluent.
Scores are concept_steer_judge.net_fluent_scores': per concept and (handle, alpha)
  fluent share = share of the concept's prompts judged fluent
  net on-topic = mean over FLUENT prompts of (hit - hit of the unsteered continuation of the same prompt)
  usable       = fluent share x net on-topic                       (0 when nothing is fluent)
Per handle the alpha with the highest mean usable is its best alpha; handles are compared at their best alphas, paired over
concepts, with a 95% bootstrap CI over concepts (10k, seed 0).
Also: agreement of the agent with the Gemini 2.5 Flash-Lite answers in cache/llm_judge on every item that judge has seen.

    python experiments/concept_steering/analysis/sonnet_steer_judge_score.py --key <key.json> --answers <dir with answers_*.csv>
"""

from geoae.paths import resolve_input_path
import argparse
import csv
import glob
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--key", required=True)
ap.add_argument("--answers", required=True)
ap.add_argument("--gen", default="", help="generation source for metadata (defaults to the key's gen)")
ap.add_argument("--items", default="", help="blind items for cache agreement (defaults to <key directory>/items)")
ap.add_argument("--cache_dir", default="cache/llm_judge")
ap.add_argument("--judge_model", default="google/gemini-2.5-flash-lite")
ap.add_argument("--no_agreement", action="store_true", help="skip the comparison with cached Gemini answers (slow on Lustre)")
ap.add_argument("--out", default="")
args = ap.parse_args()
NAMES = {"label": "Supervised", "label_z": "Supervised AE latent direction",
         "base_dir": "Residual k-means clusters", "bypass_dir": "GeoAE clusters",
         "nobypass_dir": "No-bypass AE clusters", "sae_km_dir": "k-means on SAE codes", "sae_f1_dir": "SAE top-1 feature",
         "sae_dir": "SAE top-5 features"}
K = json.load(open(resolve_input_path(args.key)))
expected_items = {k["item"] for k in K["key"]}
if not expected_items or len(expected_items) != K["n_items"]:
    ap.error("key n_items must equal the number of distinct mapped item ids")
if any(type(i) is not int or i < 0 for i in expected_items):
    ap.error("key item ids must be nonnegative integers")
concepts = sorted({k["concept"] for k in K["key"]})
ci = {c: i for i, c in enumerate(concepts)}
P = K["P"]
if type(P) is not int or P < 1:
    ap.error("key P must be a positive integer")
slot = {c: sorted({k["p"] for k in K["key"] if k["concept"] == c}) for c in concepts}
if any(len(ps) != P for ps in slot.values()):
    ap.error("every concept must have exactly P prompt indices in the key")
mapped_cells = set()
for k in K["key"]:
    cell = (k["concept"], k["handle"], k["alpha"], k["p"])
    if cell in mapped_cells:
        ap.error(f"duplicate logical judgment in key: {cell}")
    mapped_cells.add(cell)
    if k["correct"] not in ("A", "B", "C", "D"):
        ap.error(f"invalid correct topic letter in key for item {k['item']}")
required_cells = {(c, h, a, p) for c in concepts for p in slot[c]
                  for h, a in [("unsteered", "0.0")]
                  + [(h, a) for h in K["handles"] for a in K["alphas"]]}
if mapped_cells != required_cells:
    ap.error("key must contain every concept/prompt for unsteered text and each requested handle/alpha")
ans = {}
answer_files = sorted(resolve_input_path(Path(args.answers)).glob("answers_*.csv"))
if not answer_files:
    ap.error(f"no answers_*.csv files found in {args.answers}")
for f in answer_files:
    with resolve_input_path(f).open(newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if len(row) != 3:
                ap.error(f"{f}:{line}: expected exactly id,LETTER,YES/NO")
            ident, topic, fluent = (x.strip() for x in row)
            topic, fluent = topic.upper(), fluent.upper()
            if not ident.isdigit() or topic not in ("A", "B", "C", "D") or fluent not in ("YES", "NO"):
                ap.error(f"{f}:{line}: invalid id, topic letter, or fluency answer")
            item_id = int(ident)
            if item_id not in expected_items:
                ap.error(f"{f}:{line}: unknown item id {item_id}")
            if item_id in ans:
                ap.error(f"{f}:{line}: duplicate answer for item id {item_id}")
            ans[item_id] = (topic, fluent == "YES")
missing = sorted(expected_items - set(ans))
if missing:
    ap.error(f"missing answers for {len(missing)} items: {missing[:20]}; refusing to score incomplete judgments")
dataset = K.get("dataset", "unknown dataset")
dataset_label = "DBpedia-14" if dataset == "db14" else f"Atlas ({dataset})" if dataset.startswith("atlas") else dataset
items_dir = args.items or str(Path(args.key).parent / "items")
print(f"# Blind agent judge, {dataset_label} long-generation steering sample\n\nitems judged {len(ans)} of {K['n_items']}")
hit = {}
flu = {}
for k in K["key"]:
    key = (k["handle"], k["alpha"])
    hit.setdefault(key, np.full((len(concepts), P), np.nan))
    flu.setdefault(key, np.zeros((len(concepts), P), bool))
    j = slot[k["concept"]].index(k["p"])
    hit[key][ci[k["concept"]], j] = float(ans[k["item"]][0] == k["correct"])
    flu[key][ci[k["concept"]], j] = ans[k["item"]][1]
hit0 = hit[("unsteered", "0.0")]
print(f"unsteered continuations: on target {np.nanmean(hit0):.0%} (chance 25%), fluent {flu[('unsteered', '0.0')].mean():.0%}\n")


def scores(h, a):
    f = flu[(h, a)]
    d = np.where(f, hit[(h, a)] - hit0, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        net = np.nanmean(d, axis=1)
    fs = f.mean(1)
    return fs, net, np.where(np.isnan(net), 0.0, fs * net), np.nanmean(hit[(h, a)], axis=1)


def boot(v):
    v = np.asarray(v, float)
    b = v[np.random.default_rng(0).integers(0, len(v), (10000, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def paired_stats(v):
    v = np.asarray(v, float)
    m, lo, hi = boot(v)
    return {"diff": m, "ci": [lo, hi], "better": int((v > 0).sum()),
            "worse": int((v < 0).sum()), "tied": int((v == 0).sum()), "n_concepts": len(v)}


print("## Usable on-target generations per 100 (fluent share x net on-topic), by steering strength\n")
print("| Method | " + " | ".join(f"α {a}" for a in K["alphas"]) + " | Best α | Usable at best α [95% CI] | Fluent | On target (raw) |")
print("|---|" + "---|" * (len(K["alphas"]) + 4))
best, res = {}, {
    "meta": {"dataset": dataset, "concepts": concepts, "n_concepts": len(concepts),
             "P": P, "prompt_indices": slot, "alphas": K["alphas"], "handles": K["handles"],
             "n_items": K["n_items"], "n_answers": len(ans), "n_judgments": len(K["key"]),
             "key": args.key, "answers": args.answers, "answer_files": [str(f) for f in answer_files],
             "gen": args.gen or K.get("gen"), "items": items_dir,
             "agreement_skipped": args.no_agreement,
             "agreement_model": None if args.no_agreement else args.judge_model,
             "bootstrap": {"unit": "concept", "n_resamples": 10000, "seed": 0,
                           "interval": "percentile", "confidence": 0.95}},
    "unsteered": {"on_target": hit0.mean(1).tolist(), "on_target_mean": float(hit0.mean()),
                  "fluent": flu[("unsteered", "0.0")].mean(1).tolist(),
                  "fluent_mean": float(flu[("unsteered", "0.0")].mean()),
                  "both": (hit0 * flu[("unsteered", "0.0")]).mean(1).tolist(),
                  "both_mean": float((hit0 * flu[("unsteered", "0.0")]).mean())},
}
for h in K["handles"]:
    per = {a: scores(h, a) for a in K["alphas"]}
    best[h] = max(K["alphas"], key=lambda a: per[a][2].mean())
    fs, net, us, raw = per[best[h]]
    m, lo, hi = boot(us)
    res[h] = {"best_alpha": best[h], "usable": us.tolist(), "usable_mean": m, "ci": [lo, hi],
              "per_alpha": {a: {"usable": float(per[a][2].mean()), "fluent": float(per[a][0].mean()),
                                "on_target_raw": float(np.nanmean(per[a][3]))} for a in K["alphas"]}}
    print(f"| {NAMES.get(h, h)} | " + " | ".join(f"{100 * per[a][2].mean():.1f}" for a in K["alphas"])
          + f" | {best[h]} | {100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] | {fs.mean():.0%} | {np.nanmean(raw):.0%} |")
print("\n## Paired differences at each method's best α (usable per 100, first − second), concepts better / worse\n")
print("| Pair | Difference [95% CI] | Better / worse |\n|---|---|---|")
pairs = [("bypass_dir", "sae_dir"), ("bypass_dir", "sae_f1_dir"), ("bypass_dir", "sae_km_dir"), ("bypass_dir", "nobypass_dir"),
         ("bypass_dir", "base_dir"), ("bypass_dir", "label"), ("sae_dir", "label"), ("sae_dir", "base_dir")]
res["pairs"] = {}
for a, b in pairs:
    if a in res and b in res:
        d = np.asarray(res[a]["usable"]) - np.asarray(res[b]["usable"])
        m, lo, hi = boot(d)
        res["pairs"][f"{a}-{b}"] = paired_stats(d)
        print(f"| {NAMES[a]} − {NAMES[b]} | {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {int((d > 0).sum())} / {int((d < 0).sum())} |")

# ---- agreement with the Gemini judge on items it has already answered (cache lookups only, no API call)
if not args.no_agreement:
    from geoae.analysis.concept_steer_judge import build_prompt  # noqa: E402
    from geoae.analysis.llm_judge import fluency_prompt
    from geoae.analysis.llm_judge import parse_fluency  # noqa: E402


def cached(prompt):
    h = hashlib.sha256(f"openrouter\0{args.judge_model}\0{prompt}".encode()).hexdigest()[:32]
    p = Path(args.cache_dir) / f"{h}.json"
    try:
        return json.loads(resolve_input_path(p).read_text())["response"] if resolve_input_path(p).exists() else None
    except (ValueError, KeyError):
        return None


items = {}
for f in ([] if args.no_agreement else sorted(resolve_input_path(glob).glob(str(Path(items_dir) / "items_*.jsonl")))):
    for line in open(resolve_input_path(f)):
        it = json.loads(line)
        items[it["id"]] = it
t_same = t_n = f_same = f_n = 0
for i, it in items.items():
    if i not in ans:
        continue
    shown = [it["options"][x] for x in "ABCD"]
    g = cached(build_prompt(it["prompt"], it["continuation"], shown))
    if g is not None and re.search(r"[ABCD]", g.upper()):
        t_n += 1
        t_same += re.search(r"[ABCD]", g.upper()).group(0) == ans[i][0]
    gf = cached(fluency_prompt(it["prompt"], it["continuation"]))
    if gf is not None and parse_fluency(gf) is not None:
        f_n += 1
        f_same += parse_fluency(gf) == ans[i][1]
res["agreement_with_gemini"] = {"topic": [int(t_same), t_n], "fluent": [int(f_same), f_n]}
if not args.no_agreement:
    print(f"\n## Agreement with the Gemini judge on items it had already scored\n\ntopic choice: {t_same}/{t_n} = "
          f"{t_same / max(t_n, 1):.0%};  fluency: {f_same}/{f_n} = {f_same / max(f_n, 1):.0%}")
# ---- on target as the main score, fluency not required (user, 10 Oct): net on-target over ALL prompts, fluency reported beside it
print("\n## On target, fluency not required: on-target rate above the unsteered text (points), all prompts\n")
print("| Method | " + " | ".join(f"α {a}: on target / net / fluent" for a in K["alphas"]) + " | Best α | Net on target at best α [95% CI] |")
print("|---|" + "---|" * (len(K["alphas"]) + 2))
res["target"] = {}
for h in K["handles"]:
    per = {a: (np.nanmean(hit[(h, a)], axis=1), np.nanmean(hit[(h, a)] - hit0, axis=1), flu[(h, a)].mean(1)) for a in K["alphas"]}
    ba = max(K["alphas"], key=lambda a: per[a][1].mean())
    m, lo, hi = boot(per[ba][1])
    res["target"][h] = {"best_alpha": ba, "net": per[ba][1].tolist(), "net_mean": m, "ci": [lo, hi],
                        "per_alpha": {a: {"on_target": float(per[a][0].mean()), "net": float(per[a][1].mean()),
                                          "fluent": float(per[a][2].mean())} for a in K["alphas"]}}
    print(f"| {NAMES.get(h, h)} | " + " | ".join(f"{per[a][0].mean():.0%} / {100 * per[a][1].mean():+.1f} / {per[a][2].mean():.0%}"
                                                 for a in K["alphas"])
          + f" | {ba} | {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
print("\n| Pair (net on target at best α) | Difference [95% CI] | Better / worse |\n|---|---|---|")
res["target_pairs"] = {}
for a, b in pairs:
    if a in res["target"] and b in res["target"]:
        d = np.asarray(res["target"][a]["net"]) - np.asarray(res["target"][b]["net"])
        m, lo, hi = boot(d)
        res["target_pairs"][f"{a}-{b}"] = {**paired_stats(d),
                                         "alpha_first": res["target"][a]["best_alpha"],
                                         "alpha_second": res["target"][b]["best_alpha"]}
        print(f"| {NAMES[a]} − {NAMES[b]} | {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {int((d > 0).sum())} / {int((d < 0).sum())} |")
# ---- on target AND fluent (plain share of continuations judged both), per strength; GeoAE vs every other handle, paired over concepts
print("\n## On target and fluent: share of continuations judged both\n")
print("| Method | " + " | ".join(f"α {a}: on target / fluent / both" for a in K["alphas"]) + " |")
print("|---|" + "---|" * len(K["alphas"]))
res["both"] = {}
for h in K["handles"]:
    res["both"][h] = {a: {"on_target": np.nanmean(hit[(h, a)], axis=1).tolist(), "fluent": flu[(h, a)].mean(1).tolist(),
                          "both": (np.nan_to_num(hit[(h, a)]) * flu[(h, a)]).mean(1).tolist()} for a in K["alphas"]}
    print(f"| {NAMES.get(h, h)} | " + " | ".join(
        f"{np.mean(res['both'][h][a]['on_target']):.0%} / {np.mean(res['both'][h][a]['fluent']):.0%} / {np.mean(res['both'][h][a]['both']):.0%}"
        for a in K["alphas"]) + " |")
REF = "bypass_dir"
res["per_alpha_pairs"] = {}
for a in K["alphas"]:
    print(f"\n| {NAMES[REF]} minus (α {a}) | On target [95% CI] | better / worse / tied | Fluent [95% CI] | On target and fluent [95% CI] | better / worse / tied |")
    print("|---|---|---|---|---|---|")
    res["per_alpha_pairs"][a] = {}
    for h in K["handles"] if REF in K["handles"] else []:
        if h == REF:
            continue
        cells = []
        stats = {}
        for kk in ("on_target", "fluent", "both"):
            d = np.asarray(res["both"][REF][a][kk]) - np.asarray(res["both"][h][a][kk])
            m, lo, hi = boot(d)
            stats[kk] = paired_stats(d)
            cells.append((f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]", f"{int((d > 0).sum())} / {int((d < 0).sum())} / {int((d == 0).sum())}"))
        print(f"| {NAMES.get(h, h)} | {cells[0][0]} | {cells[0][1]} | {cells[1][0]} | {cells[2][0]} | {cells[2][1]} |")
        res["per_alpha_pairs"][a][f"{REF}-{h}"] = stats
if args.out:
    Path(args.out).write_text(json.dumps(res, indent=1))
