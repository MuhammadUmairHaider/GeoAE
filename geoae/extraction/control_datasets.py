"""Four frozen control classification corpora for the range-intervention harness (emotion6, news20, trec6, jigsaw2).

Writes results/delta/control_datasets/<name>/{data.parquet,cfg.json}. Deterministic (seed 0). Run on the box:
    .venv/bin/python geoae/extraction/control_datasets.py [name ...]
Rules: strip text, drop empty, keep 20 <= len <= char cap (1000, tightened if a full prompt exceeds 1000 tokens under
either tokenizer), seeded shuffle, few-shot exemplars picked FIRST and excluded from data.parquet, cap 600 docs/class.
Exemplars are single-line, <= 200 chars, no braces. cfg.json mirrors the atlas cfg format (no split_column).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SEED, PER_CLASS, MIN_LEN = 0, 600, 20
import os
OUT = Path(os.environ.get("GEOAE_CONTROL_OUT") or Path(__file__).resolve().parents[2] / "results/delta/control_datasets")
TOKENIZERS = {"llama": "meta-llama/Llama-3.2-3B", "gemma": "google/gemma-3-4b-pt"}

NEWS_MAP = {"alt.atheism": "atheism", "comp.graphics": "graphics", "comp.os.ms-windows.misc": "windows",
            "comp.sys.ibm.pc.hardware": "pc hardware", "comp.sys.mac.hardware": "mac hardware",
            "comp.windows.x": "x window system", "misc.forsale": "for sale", "rec.autos": "cars",
            "rec.motorcycles": "motorcycles", "rec.sport.baseball": "baseball", "rec.sport.hockey": "hockey",
            "sci.crypt": "cryptography", "sci.electronics": "electronics", "sci.med": "medicine",
            "sci.space": "space", "soc.religion.christian": "christianity", "talk.politics.guns": "guns",
            "talk.politics.mideast": "middle east", "talk.politics.misc": "politics",
            "talk.religion.misc": "religion"}


def load_emotion6():
    from datasets import load_dataset
    d = load_dataset("dair-ai/emotion", split="train").to_pandas()
    classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    return d[["text", "label"]].copy(), classes, "dair-ai/emotion train", {}


def load_news20():
    from datasets import load_dataset
    parts = [load_dataset("SetFit/20_newsgroups", split=s).to_pandas() for s in ("train", "test")]
    d = pd.concat(parts, ignore_index=True)
    m = d.groupby("label")["label_text"].agg(lambda s: sorted(set(s)))
    assert all(len(v) == 1 for v in m), "label<->label_text not one-to-one"
    assert sorted(m.index) == list(range(20))
    classes = [NEWS_MAP[m[i][0]] for i in range(20)]
    assert len(set(classes)) == 20
    n0 = len(d)
    d = d.drop_duplicates(subset="text")
    return d[["text", "label"]].copy(), classes, "SetFit/20_newsgroups train+test (exact-duplicate texts removed)", \
        {"label_text_by_id": {int(i): m[i][0] for i in m.index}, "rows_before_dedupe": n0}


def load_trec6():
    from datasets import load_dataset
    parts = [load_dataset("CogComp/trec", revision="refs/convert/parquet", split=s).to_pandas()
             for s in ("train", "test")]
    d = pd.concat(parts, ignore_index=True).rename(columns={"coarse_label": "label"})
    classes = ["abbreviation", "thing", "explanation", "person", "location", "number"]   # v2: "entity" -> "thing", "description" -> "explanation"
    return d[["text", "label"]].copy(), classes, "CogComp/trec (refs/convert/parquet) train+test, coarse_label", {}


def load_jigsaw2():
    from huggingface_hub import hf_hub_download
    csv = hf_hub_download("thesofakillers/jigsaw-toxic-comment-classification-challenge", "train.csv",
                          repo_type="dataset")
    d = pd.read_csv(csv)
    flags = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    d["label"] = (d[flags].sum(axis=1) > 0).astype(int)
    d["only_toxic"] = (d["toxic"] == 1) & (d[flags[1:]].sum(axis=1) == 0)
    d = d.rename(columns={"comment_text": "text"})
    return d[["text", "label", "only_toxic"]].copy(), ["clean", "toxic"], \
        "thesofakillers/jigsaw-toxic-comment-classification-challenge train.csv; toxic = any of 6 flags", {}


LOADERS = {"emotion6": load_emotion6, "news20": load_news20, "trec6": load_trec6, "jigsaw2": load_jigsaw2}
INSTR = {
    "emotion6": "Choose the emotion expressed in this text from: sadness, joy, love, anger, fear, surprise.",
    "news20": "Choose the newsgroup topic of this post from: {cl}. Be careful distinguishing between similar topics.",
    "trec6": "Choose the type of answer this question asks for from: abbreviation, thing, explanation, person, "
             "location, number.",
    "jigsaw2": "Is this comment clean or toxic? Choose from: clean, toxic.",
}
# news20 exemplars: single-line posts are mostly off-topic one-liners, so require an on-topic keyword (exemplar choice only)
NEWS_KW = {"hockey": r"hockey|nhl|stanley cup|goalie|playoff", "space": r"nasa|orbit|space|shuttle|satellite|launch",
           "for sale": r"for sale|asking|obo|\$\d", "middle east": r"israel|arab|palestin|turk|armenia|lebanon"}
# prompt_version 2: hand-written, keyword-clear exemplars (the first probe showed the data-drawn exemplars left
# several classes at ~0 accuracy). (text, label id) in prompt order; not drawn from the data, so nothing to exclude.
EXEMPLARS_V2 = {
    "emotion6": [
        ("i feel so sad and depressed since my dog died", 0), ("i feel so angry and furious at him for lying to me", 3),
        ("i feel so happy and joyful this morning", 1), ("i feel so scared and terrified of being alone in the dark", 4),
        ("i feel so loving and affectionate towards my wife", 2),
        ("i feel surprised and amazed that they threw me a party", 5),
        ("i feel delighted and cheerful about the good news", 1),
        ("i feel shocked and astonished by the unexpected news", 5),
        ("i feel miserable and heartbroken today", 0), ("i feel tender and devoted to my children", 2),
        ("i feel enraged and hostile about the unfair rules", 3),
        ("i feel afraid and frightened that something bad will happen", 4)],
    "trec6": [
        ("What does NASA stand for ?", 0), ("What kind of animal is a Dalmatian ?", 1),
        ("How do airplanes stay in the air ?", 2), ("Who wrote the novel Moby Dick ?", 3),
        ("Where is the Eiffel Tower located ?", 4), ("How many legs does a spider have ?", 5),
        ("What spice is the most expensive in the world ?", 1), ("What is the abbreviation for Doctor of Philosophy ?", 0),
        ("Why is the sky blue ?", 2), ("What country borders Spain to the west ?", 4),
        ("What year did World War II end ?", 5), ("Which actor played Batman in the 1989 film ?", 3)],
}
QUOTED = {"trec6"}
NEWS_EX = ["hockey", "space", "for sale", "middle east"]


EX_MIN = {"emotion6": 60, "news20": 50, "trec6": 30, "jigsaw2": 30}   # too-short snippets carry no label signal


def ok_ex(t: str, lo: int = 30) -> bool:
    return "\n" not in t and "\r" not in t and lo <= len(t) <= 200 and "{" not in t and "}" not in t


def build(name, raw, classes, cap):
    d = raw.copy()
    d["text"] = d["text"].astype(str).str.strip()
    d = d[(d.text.str.len() >= MIN_LEN) & (d.text.str.len() <= cap)]
    d = d.sample(frac=1.0, random_state=SEED).reset_index(drop=True)   # seeded shuffle
    ex_idx = []
    if name in EXEMPLARS_V2:
        ex_list = [dict(text=t, label=classes[c], label_id=c) for t, c in EXEMPLARS_V2[name]]
        d = d[~d.text.isin([e["text"] for e in ex_list])]   # exemplars never appear in the data
        d = d.groupby("label", group_keys=False).head(PER_CLASS).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        head = INSTR[name].format(cl=", ".join(classes)) + "\n\n"
        for e in ex_list:
            if name in QUOTED:   # exemplar shape matches the final '{"{}":' (fixes a stray ` "label` generation on trec6)
                head += '{{"' + e["text"] + '":' + e["label"] + "}}\n\n"
            else:
                head += "{{ " + e["text"] + ":" + e["label"] + "}}\n\n"
        return d[["text", "label"]], ex_list, head + '{{"{}":'
    if name == "news20":
        want = [classes.index(c) for c in NEWS_EX]
    else:
        want = list(range(len(classes)))
    for c in want:
        sub = d[(d.label == c) & d.text.map(lambda t: ok_ex(t, EX_MIN[name]))]
        if name == "news20":
            sub = sub[sub.text.str.contains(NEWS_KW[classes[c]], case=False, regex=True)]
        if name == "jigsaw2" and c == 1:
            sub = sub[sub.only_toxic & (sub.text.str.len() <= 120)]   # mildest: toxic flag only, short
        if name == "jigsaw2" and c == 0:
            sub = sub[sub.text.str.len() <= 120]
        assert len(sub), f"{name}: no exemplar for class {c}"
        ex_idx.append(sub.index[0])
    ex = d.loc[ex_idx]
    d = d.drop(index=ex_idx)
    d = d.groupby("label", group_keys=False).head(PER_CLASS)
    d = d.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    ex_list = [dict(text=r.text, label=classes[int(r.label)], label_id=int(r.label)) for r in ex.itertuples()]
    head = INSTR[name].format(cl=", ".join(classes)) + "\n\n"
    for e in ex_list:
        head += "{{ " + e["text"] + ":" + e["label"] + "}}\n\n"
    head += '{{"{}":'
    return d[["text", "label"]], ex_list, head


def main():
    from geoae.checkpoint import load_tokenizer
    toks = {k: load_tokenizer(v, local_files_only=True) for k, v in TOKENIZERS.items()}
    names = sys.argv[1:] or list(LOADERS)
    for name in names:
        raw, classes, source, extra = LOADERS[name]()
        for cap in (1000, 800, 600, 500, 400, 300):
            d, ex_list, head = build(name, raw, classes, cap)
            prompts = [head.format(t) for t in d.text]
            mp = {}
            for k, t in toks.items():
                ids = t(prompts)["input_ids"]
                mp[k] = max(len(x) for x in ids)
            if max(mp.values()) <= 1000:
                break
            print(f"[{name}] cap {cap} chars -> max prompt {mp}; tightening")
        assert max(mp.values()) <= 1000, mp
        # prompt must also parse with .format
        longest = max(max(len(t(c, add_special_tokens=False)["input_ids"]) for t in toks.values()) for c in classes)
        max_new = max(6, 2 + longest)
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        d.to_parquet(out / "data.parquet", index=False)
        cfg = dict(classes=classes, dataset_name=f"local:{name}", dataset_split="train", text_column="text",
                   label_column="label", local_parquet="data.parquet", prompt_head=head, max_new=max_new,
                   seed=SEED, exemplars=ex_list, source=source,
                   filter=f"strip; {MIN_LEN} <= len(text) <= {cap} chars; seeded shuffle (seed 0); exemplars removed; "
                          f"cap {PER_CLASS}/class",
                   prompt_version=2 if name in EXEMPLARS_V2 else 1, char_cap=cap, n_per_class={classes[i]: int((d.label == i).sum()) for i in range(len(classes))},
                   max_prompt_tokens=mp, **extra)
        json.dump(cfg, open(out / "cfg.json", "w"), indent=1)
        print(f"\n== {name}: {len(d)} rows, char_cap {cap}, max_new {max_new}, max_prompt_tokens {mp}")
        for i, c in enumerate(classes):
            print(f"  {i:>2} {c:<18}{(d.label == i).sum():>6}")
        for e in ex_list:
            print(f"  EX [{e['label']}] {e['text']!r}")


if __name__ == "__main__":
    main()
