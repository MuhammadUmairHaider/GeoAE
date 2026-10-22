"""Build blind item files for agent judges from a concept-steering generation file (concept_steer_generate.py /
concept_steer_extra.py), for experiments/concept_steering/analysis/sonnet_steer_judge_score.py.

Per concept and prompt: the unsteered continuation and every handle's continuation at the chosen strengths. Each item
shows only the prompt, the continuation and four topic descriptions (the target plus three other classes of the dataset,
chosen and ordered by concept_steer_judge.pick_candidates, so they are identical for every handle and strength).
Identical (prompt, continuation, options) strings become ONE item; items are shuffled, so neither the method nor the
strength can be read off the position. The key (item -> concept, handle, strength, prompt, correct letter) is written to a
separate path: keep it out of the directory the judges are pointed at.

    python experiments/concept_steering/analysis/sonnet_steer_judge_items.py --gen <gen.json> --dataset atlas_s0 --alphas 0.5 \
        --out_dir <dir>/items --key <private dir>/key.json
"""

from geoae.paths import resolve_input_path
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from geoae.analysis.concept_steer_judge import LETTERS
from geoae.analysis.concept_steer_judge import pick_candidates  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--alphas", default="0.5", help="comma list of strengths to judge")
ap.add_argument("--prompts_per_concept", type=int, default=0, help="0 = all prompts; else a seeded draw per concept")
ap.add_argument("--chunk", type=int, default=225, help="max items per file")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--key", required=True)
args = ap.parse_args()

G = json.load(open(resolve_input_path(args.gen)))
ds = args.dataset
prompts, desc = G["meta"]["prompts"], G["meta"]["concept_descriptions"][ds]
st, un = G["generations"]["steered"][ds], G["generations"]["unsteered"]["text"]
alphas = [a for a in args.alphas.split(",") if a]
items, key, index = [], [], {}


def add(concept, h, a, p, text):
    cand, order = pick_candidates(ds, concept, p, list(desc.keys()))
    shown = [desc[cand[k]] for k in order]
    k = (prompts[p], text, tuple(shown))
    if k not in index:
        index[k] = len(items)
        items.append({"prompt": prompts[p], "continuation": text, "options": dict(zip("ABCD", shown))})
    key.append({"item": index[k], "concept": concept, "handle": h, "alpha": a, "p": p, "correct": LETTERS[list(order).index(0)]})


handles = [h for h in G["meta"]["handles"] if any(h in st[c] for c in st)]
for concept in sorted(st):
    ps = range(len(prompts))
    if args.prompts_per_concept:
        ps = sorted(map(int, np.random.default_rng(zlib.crc32(f"sonnet|{concept}".encode())).choice(
            len(prompts), args.prompts_per_concept, replace=False)))
    for p in ps:
        add(concept, "unsteered", "0.0", p, un[p])
        for h in handles:
            if h in st[concept]:
                for a in alphas:
                    add(concept, h, a, p, st[concept][h][a]["text"][p])
perm = np.random.default_rng(args.seed).permutation(len(items))
inv = {int(o): i for i, o in enumerate(perm)}
items = [items[int(o)] for o in perm]
for i, it in enumerate(items):
    it["id"] = i
for k in key:
    k["item"] = inv[k["item"]]
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
n_files = -(-len(items) // args.chunk)
size = -(-len(items) // n_files)
for c in range(n_files):
    part = items[c * size:(c + 1) * size]
    with open(out / f"items_{c + 1}.jsonl", "w") as f:
        for it in part:
            f.write(json.dumps({"id": it["id"], "prompt": it["prompt"], "continuation": it["continuation"],
                                "options": it["options"]}, ensure_ascii=False) + "\n")
    print(f"items_{c + 1}.jsonl: {len(part)} items (ids {part[0]['id']}..{part[-1]['id']})")
Path(args.key).parent.mkdir(parents=True, exist_ok=True)
json.dump({"dataset": ds, "gen": args.gen, "alphas": alphas, "P": args.prompts_per_concept or len(prompts), "handles": handles, "key": key,
           "n_items": len(items), "concepts": sorted(st)}, open(args.key, "w"))
missing = {h: [c for c in sorted(st) if h not in st[c]] for h in handles}
print(f"concepts {len(st)}, handles {handles}, strengths {alphas}; judgements {len(key)}, unique items {len(items)}, files {n_files}")
print("handles missing for some concepts:", {h: v for h, v in missing.items() if v} or "none")
