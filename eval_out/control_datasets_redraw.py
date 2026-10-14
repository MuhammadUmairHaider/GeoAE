"""Document redraws of a frozen control corpus (news20, emotion6): the same classes, prompt and few-shot exemplars, a new
random sample of documents. For seed N: apply the filter of eval_out/control_datasets_prep.py with the frozen character
cap, drop the exemplar texts, shuffle with seed N, keep up to 600 documents per class, shuffle again. The original corpus
is the seed-0 draw and is not touched.

Writes results/delta/control_datasets/<name>_s<N>/{data.parquet,cfg.json}; a seed whose files exist is skipped, so the
same line can be run again. geoae.interp._shared finds <name>_s<N> on disk as a dataset name.

    python eval_out/control_datasets_redraw.py news20 --seeds 1,2,3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import control_datasets_prep as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=["news20", "emotion6"])
    ap.add_argument("--seeds", required=True, help="comma list of seeds above 0 (0 is the original corpus)")
    args = ap.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer
    seeds = [int(x) for x in args.seeds.split(",")]
    assert all(s > 0 for s in seeds), "seed 0 is the original corpus"
    todo = [s for s in seeds if not (P.OUT / f"{args.name}_s{s}" / "cfg.json").exists()]
    for s in sorted(set(seeds) - set(todo)):
        print(f"[{args.name}] seed {s}: {P.OUT / f'{args.name}_s{s}'} exists, kept")
    if not todo:
        return
    cfg0 = json.load(open(P.OUT / args.name / "cfg.json"))
    old = pd.read_parquet(P.OUT / args.name / "data.parquet")
    raw, classes, source, _ = P.LOADERS[args.name]()
    assert classes == cfg0["classes"], "class list differs from the frozen corpus"
    d = raw.copy()
    d["text"] = d["text"].astype(str).str.strip()
    d = d[(d.text.str.len() >= P.MIN_LEN) & (d.text.str.len() <= cfg0["char_cap"])]
    d = d[~d.text.isin([e["text"] for e in cfg0["exemplars"]])]          # exemplars never appear in the data
    assert set(old.text) <= set(d.text), "the frozen corpus is not a subset of the filtered source: the source changed"
    toks = {}
    for k, v in P.TOKENIZERS.items():
        try:
            toks[k] = AutoTokenizer.from_pretrained(v, local_files_only=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{args.name}] tokenizer {v} not available here, prompt length not checked for it ({type(e).__name__})")
    assert toks, "no tokenizer available for the prompt-length check"
    print(f"[{args.name}] eligible documents per class: {[int((d.label == i).sum()) for i in range(len(classes))]}")
    for seed in todo:
        x = d.sample(frac=1.0, random_state=seed).groupby("label", group_keys=False).head(P.PER_CLASS)
        x = x.sample(frac=1.0, random_state=seed).reset_index(drop=True)[["text", "label"]]
        mp = {k: max(len(i) for i in t([cfg0["prompt_head"].format(v) for v in x.text])["input_ids"]) for k, t in toks.items()}
        assert max(mp.values()) <= 1000, f"seed {seed}: prompt of {mp} tokens would be truncated"
        shared = [int(x.text[x.label == i].isin(set(old.text[old.label == i])).sum()) for i in range(len(classes))]
        out = P.OUT / f"{args.name}_s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        x.to_parquet(out / "data.parquet", index=False)
        cfg = dict(cfg0, dataset_name=f"local:{args.name}_s{seed}", seed=seed, redraw_of=args.name,
                   filter=cfg0["filter"].replace("(seed 0)", f"(seed {seed})"), max_prompt_tokens=mp,
                   n_per_class={classes[i]: int((x.label == i).sum()) for i in range(len(classes))},
                   n_shared_with_original={classes[i]: shared[i] for i in range(len(classes))})
        json.dump(cfg, open(out / "cfg.json", "w"), indent=1)
        print(f"[{args.name}] seed {seed}: {len(x)} rows -> {out}; {sum(shared)} ({sum(shared) / len(x):.0%}) also in the "
              f"original corpus; max prompt tokens {mp}")


if __name__ == "__main__":
    main()
