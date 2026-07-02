"""
Shared few-shot prompted-classification machinery for the concept-compare
tools (causal_concept_compare, steering_concept_compare): dataset prompt
configs, tokenisation, post-layer activation capture, greedy-generation
classification, and perplexity.

Call `set_dataset(name)` once in main() — the helpers read the module-level
CLASSES / PROMPT_HEAD that it installs.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from geoae.interp import neuronlens as nl

CLASSES: list[str] = []
PROMPT_HEAD: str = ""


def set_dataset(name: str) -> dict:
    """Install CLASSES / PROMPT_HEAD for `name` and return its config dict."""
    global CLASSES, PROMPT_HEAD
    cfg = DATASET_CONFIGS[name]
    CLASSES = cfg["classes"]
    PROMPT_HEAD = cfg["prompt_head"]
    return cfg


DATASET_CONFIGS = {
    "db14": {
        "classes": ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
                    "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
                    "Plant", "Album", "Film", "WrittenWork"],
        "dataset_name": "fancyzhx/dbpedia_14",
        "dataset_split": "test",
        "text_column": "content",
        "prompt_head": (
            "Choose from one of these categories: Company, EducationalInstitution, "
            "Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, "
            "Village, Animal, Plant, Album, Film, WrittenWork. Be careful distinguishing "
            "between similar categories.\n\n"
            "{{ Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972.:Company}}\n\n"
            "{{ Dubai Gem Private School (DGPS) is a British school located in the Oud Metha area of Dubai United Arab Emirates. Dubai Gem Nursery is located in Jumeirah. Together the institutions enroll almost 1500 students aged 3 to 18.:EducationalInstitution}}\n\n"
            "{{ Martin Marty McKinnon (born 5 July 1975 in Adelaide) is a former Australian rules footballer who played with Adelaide Geelong and the Brisbane Lions in the Australian Football League (AFL).McKinnon was recruited by Adelaide in the 1992 AFL Draft with their first ever national draft pick. He was the youngest player on Adelaide's list at the time and played for Central District in the SANFL when not appearing with Adelaide.:Athlete}}\n\n"
            "{{ The Wedell-Williams XP-34 was a fighter aircraft design submitted to the United States Army Air Corps (USAAC) before World War II by Marguerite Clark Williams widow of millionaire Harry P. Williams former owner and co-founder of the Wedell-Williams Air Service Corporation.:MeanOfTransportation}}\n\n"
            '{{"{}":'
        )
    },
    "emotions": {
        "classes": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "dataset_name": "dair-ai/emotion",
        "dataset_split": "test",
        "text_column": "text",
        "prompt_head": (
            'Choose from one of these: anger, fear, joy, love, sadness, surprise\n'
            '     {{"I can\'t believe how wonderful this day has been!":joy}}\n'
            '     {{"Missing you more with each passing day":sadness}}\n'
            '     {{"How dare they treat me like this!":anger}}\n'
            '     {{"I\'m getting butterflies just thinking about tomorrow":fear}}\n'
            '     {{"You mean everything to me":love}}\n'
            '     {{"I didn\'t expect this to happen at all":surprise}}\n'
            '     {{"{}":'
        )
    },
    "ag_news": {
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "dataset_name": "fancyzhx/ag_news",
        "dataset_split": "test",
        "text_column": "text",
        "prompt_head": (
            'Choose from one of these categories: World, Sports, Business, Sci/Tech. Be careful distinguishing between similar topics.\n\n'
            '            {{ China unveils new space station module set for launch next year:World}}\n\n'
            '            {{ Lionel Messi nets two goals as PSG defeat Marseille 3-1:Sports}}\n\n'
            '            {{ Tesla shares jump 8% after delivery numbers beat expectations:Business}}\n\n'
            '            {{ Researchers develop a new algorithm to speed up quantum error correction:Sci/Tech}}\n\n'
            '            {{"{}":'
        )
    }
}


def build_prompt(text: str) -> str:
    return PROMPT_HEAD.format(text)


def _tok(tokenizer, texts, device):
    return tokenizer([build_prompt(t) for t in texts], padding=True, truncation=True,
                     max_length=1024, return_tensors="pt").to(device)


@torch.no_grad()
def capture_h(lm, tokenizer, texts, layer, device, batch_size=16):
    """Post-layer (pre-norm) residual at orig_lens-1 — same point as reference mask_layer."""
    H = []
    for s in tqdm(range(0, len(texts), batch_size), desc="capture", leave=False):
        enc = _tok(tokenizer, texts[s:s + batch_size], device)
        H.append(nl.capture_post_layer(lm, enc, layer))
    return np.concatenate(H, axis=0)


@torch.no_grad()
def predict(lm, tokenizer, texts, labels, device, batch_size=16, max_new=6):
    """Few-shot classification via greedy generation. Returns correct/gold_conf/pred."""
    preds, gold_conf, correct = [], [], []
    for s in tqdm(range(0, len(texts), batch_size), desc="predict", leave=False):
        bt, bl = texts[s:s+batch_size], labels[s:s+batch_size]
        enc = _tok(tokenizer, bt, device)
        gen = lm.generate(**enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
                          output_scores=True, return_dict_in_generate=True,
                          pad_token_id=tokenizer.pad_token_id)
        new = gen.sequences[:, enc["input_ids"].shape[1]:]
        conf0 = gen.scores[0].softmax(-1).float().cpu()        # first new-token dist
        for i in range(len(bt)):
            gtxt = tokenizer.decode(new[i], skip_special_tokens=True)
            pred = gtxt.split("}")[0].strip('"').strip()
            gold = CLASSES[int(bl[i])]
            preds.append(pred); correct.append(pred == gold)
            gtok = tokenizer.encode(gold, add_special_tokens=False)[0]
            gold_conf.append(float(conf0[i, gtok]))
    return {"pred": preds, "gold_conf": gold_conf, "correct": correct}


@torch.no_grad()
def perplexity(lm, tokenizer, texts, device):
    tot_loss, tot_tok = 0.0, 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(device)
        loss = lm(**enc, labels=enc["input_ids"]).loss
        n = enc["input_ids"].shape[1]
        tot_loss += loss.item() * n; tot_tok += n
    return float(np.exp(tot_loss / max(tot_tok, 1)))
