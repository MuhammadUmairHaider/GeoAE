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
    },
    "biasbios": {
        "classes": [
            "accountant", "architect", "attorney", "chiropractor", "comedian",
            "composer", "dentist", "dietitian", "dj", "filmmaker",
            "interior_designer", "journalist", "model", "nurse", "painter",
            "paralegal", "pastor", "personal_trainer", "photographer", "physician",
            "poet", "professor", "psychologist", "rapper", "software_engineer",
            "surgeon", "teacher", "yoga_teacher",
        ],
        "dataset_name": "LabHC/bias_in_bios",
        "dataset_split": "test",
        "text_column": "hard_text",
        "label_column": "profession",   # this dump has no "label" column
        "prompt_head": (
            "Choose the profession of the person described in this biography from: "
            "accountant, architect, attorney, chiropractor, comedian, composer, "
            "dentist, dietitian, dj, filmmaker, interior_designer, journalist, "
            "model, nurse, painter, paralegal, pastor, personal_trainer, "
            "photographer, physician, poet, professor, psychologist, rapper, "
            "software_engineer, surgeon, teacher, yoga_teacher.\n\n"
            '{{ He is also the project lead of and major contributor to the open '
            'source assembler/simulator "EASy68K.":professor}}\n\n'
            '{{ She is able to assess, diagnose and treat minor illness conditions:nurse}}\n\n'
            '{{ Born in Long Beach, CA he began his musical studies at an early age:composer}}\n\n'
            '{{"{}":'
        ),
    },
}


# Same prompt, classes and text column as `emotions`, but drawn from the TRAIN
# split. The test split holds only 66 `surprise` documents — below the default
# need = n_fit + n_eval = 130, and the joint-correct filter cuts that further —
# so a full-size run is impossible on test. Train has 572. The AE is trained on
# general web text (C4/wiki/code/math/pile) and never on this dataset, so
# evaluating against its train split leaks nothing.
DATASET_CONFIGS["emotions_train"] = {
    **DATASET_CONFIGS["emotions"],
    "dataset_split": "train",
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


# ---------------------------------------------------------------------------
# Joint-correct document set (shared by steering / causal / probe tools)
# ---------------------------------------------------------------------------

def ae_fingerprint(ae) -> str:
    """Short content hash of an AE's weights.

    The joint-correct set is filtered by THIS AE's reconstruction, so it is
    checkpoint-specific: a later epoch of the same run, at the same path, is a
    different filter. Hashing the weights (rather than the path, which gets
    overwritten in place by best_val.pt) makes that detectable.
    """
    import hashlib
    h = hashlib.sha1()
    for k, v in sorted(ae.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def build_joint_correct_set(
    ds,
    dataset_cfg,
    tokenizer,
    lm,
    device,
    hook,
    z_recon,
    corr_path,
    need: int,
    dataset_name: str,
    model_name: str | None = None,
    ae_sha: str | None = None,
    batch_size: int = 16,
    tag: str = "jc",
    classes: list[int] | None = None,
):
    """
    Documents that BOTH the base LM and the AE-recon splice classify correctly.

    This is the evaluation population for the steering / causal-concept
    comparisons, so it is specific to (prompt, dataset, MODEL, AE checkpoint) —
    not just the prompt. The on-disk cache therefore records `model_name`, and a
    cache written by a different model (or by a version that predates the field,
    which reads back as None) is rebuilt rather than trusted. Reusing another
    model's set silently swaps the evaluation population and skips this
    checkpoint's reconstruction filter entirely.

    `hook` must be an activated-on-demand SplicingHook and `z_recon` the
    replacement fn; both are toggled around the recon pass.

    `batch_size` drives peak memory, which is set by the MLP intermediate, not
    the vocab head: bs x seq x intermediate x 2 B. On Gemma 3 12B
    (intermediate=15360, prompts truncated to 1024) batch 64 needs ~830 MB per
    MLP tensor with several live at once, which OOMs a 40 GB card already
    holding 24.4 GB of weights. 16 is the same default predict/capture_h use.
    """
    import hashlib
    import json
    from collections import Counter
    from pathlib import Path

    corr_path = Path(corr_path)
    prompt_sha = hashlib.sha1(PROMPT_HEAD.encode("utf-8")).hexdigest()[:12]
    n_classes = len(CLASSES)
    correct = None

    if corr_path.exists():
        cached = json.load(open(corr_path))
        cached_model = cached.get("model_name") if isinstance(cached, dict) else None
        cached_ae = cached.get("ae_sha") if isinstance(cached, dict) else None
        cached_ds = cached.get("dataset") if isinstance(cached, dict) else None
        # `dataset` must match too: variants that share a prompt_head (emotions vs
        # emotions_train) produce the SAME prompt_sha, so without this check a set
        # built on one split would be silently reused for the other.
        if (isinstance(cached, dict) and cached.get("prompt_sha") == prompt_sha
                and cached_model is not None and cached_model == model_name
                and cached_ae is not None and cached_ae == ae_sha
                and cached_ds == dataset_name
                and cached.get("docs")):
            correct = cached["docs"]
            cnt = Counter(d["label"] for d in correct)
            # A cache may cover only a subset of classes (see `classes` below), so
            # judge coverage against what was built, and say which are missing.
            built = cached.get("classes_built") or list(range(n_classes))
            min_c = min((cnt.get(c, 0) for c in built), default=0)
            absent = [CLASSES[c] for c in range(n_classes) if cnt.get(c, 0) == 0]
            if absent:
                print(f"[{tag}] cache covers {len(built)}/{n_classes} classes; "
                      f"not in it: {', '.join(absent)}")
            print(f"[{tag}] Loaded joint-correct set: {len(correct)} docs "
                  f"(min/class={min_c}, prompt_sha={prompt_sha}, model={cached_model}). "
                  f"To force a rebuild, delete {corr_path}")
            if min_c < need:
                print(f"[{tag}] WARNING: cache min/class={min_c} < n_fit+n_eval={need} "
                      f"(looks like a --smoke cache); fit/eval slices will auto-shrink "
                      f"and be noisy. Delete {corr_path} to rebuild at full size.")
        else:
            if not isinstance(cached, dict):
                why = "legacy format (no prompt hash)"
            elif cached.get("prompt_sha") != prompt_sha:
                why = "built under a DIFFERENT prompt"
            elif cached_model != model_name:
                why = f"built on a DIFFERENT model ({cached_model!r} != {model_name!r})"
            elif cached_ae != ae_sha:
                why = f"built on a DIFFERENT AE checkpoint ({cached_ae!r} != {ae_sha!r})"
            elif cached_ds != dataset_name:
                why = f"built on a DIFFERENT dataset ({cached_ds!r} != {dataset_name!r})"
            else:
                why = "empty docs list"
            print(f"[{tag}] Cache {corr_path} is stale ({why}); rebuilding (sha={prompt_sha}).")

    if correct is not None:
        return correct

    print(f"[{tag}] Building joint base+recon correct set …")
    label_col = dataset_cfg.get("label_column", "label")
    # Only build the classes that will actually be evaluated. On a 28-way task one
    # hopeless class (bias_in_bios "teacher": 0% few-shot, 4051 docs) otherwise costs
    # ~8000 predictions to return nothing.
    build = list(range(n_classes)) if classes is None else sorted(set(classes))
    by_class = {c: [] for c in range(n_classes)}
    for ex in ds:
        by_class[int(ex[label_col])].append(ex[dataset_cfg["text_column"]].strip())

    def _both_correct(texts, c):
        bs = len(texts)
        r_base = predict(lm, tokenizer, texts, [c] * bs, device=device, batch_size=bs)
        hook.activate(z_recon)
        r_recon = predict(lm, tokenizer, texts, [c] * bs, device=device, batch_size=bs)
        hook.deactivate()
        return [t for i, t in enumerate(texts)
                if r_base["correct"][i] and r_recon["correct"][i]]

    correct = []
    for c in build:
        print(f"[{tag}] Finding joint-correct predictions for class {c} ({CLASSES[c]}) ...")
        class_correct, batch_texts = [], []
        for text in by_class[c]:
            batch_texts.append(text)
            if len(batch_texts) == batch_size:
                class_correct += [{"text": t, "label": c} for t in _both_correct(batch_texts, c)]
                batch_texts = []
                if len(class_correct) >= need:
                    break
        if len(class_correct) < need and batch_texts:
            class_correct += [{"text": t, "label": c} for t in _both_correct(batch_texts, c)]
        class_correct = class_correct[:need]
        print(f"  Found {len(class_correct)} joint-correct docs for class {c}")
        correct += class_correct

    corr_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"prompt_sha": prompt_sha, "dataset": dataset_name,
               "model_name": model_name, "ae_sha": ae_sha,
               "classes": CLASSES, "classes_built": build, "docs": correct},
              open(corr_path, "w"))
    cnt = Counter(d["label"] for d in correct)
    print(f"[{tag}] joint-correct: {len(correct)} docs "
          f"(min/class={min(cnt.get(c, 0) for c in range(n_classes))}, "
          f"prompt_sha={prompt_sha}, model={model_name}, ae={ae_sha}) -> {corr_path}")
    return correct
