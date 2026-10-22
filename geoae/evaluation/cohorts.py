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

import geoae.evaluation.interventions as nl

CLASSES: list[str] = []
PROMPT_HEAD: str = ""
MAX_NEW: int = 6          # generated tokens per prediction; a config may raise it ("max_new") for long class names

# FineWeb-Atlas class corpora (atlas_s<N>): 14 random Atlas content concepts used like db14 classes.
# Their classes, prompt and local parquet are frozen by eval_out/atlas_classes_{draw,prep}.py into
# <ATLAS_CLASSES_DIR>/s<N>/cfg.json and loaded lazily by set_dataset (nothing to import for db14/biasbios).
# atlas_s0 / atlas_s1 / atlas_s2 are the three original draws; every later draw that has been prepared
# (experiments/paper/argo/atlas_new_draws.sh) is found on disk.
def atlas_classes_dir():
    import os
    from pathlib import Path
    d = os.environ.get("GEOAE_ATLAS_CLASSES_DIR", "results/delta/d6144/interventions/atlas_classes")
    p = Path(d)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def _atlas_class_datasets():
    seeds = {0, 1, 2}
    for f in atlas_classes_dir().glob("s*/cfg.json"):
        if f.parent.name[1:].isdigit():
            seeds.add(int(f.parent.name[1:]))
    return tuple(f"atlas_s{s}" for s in sorted(seeds))


ATLAS_CLASS_DATASETS = _atlas_class_datasets()


def _load_atlas_cfg(name: str) -> dict:
    import json
    d = atlas_classes_dir() / f"s{name[len('atlas_s'):]}"
    f = d / "cfg.json"
    if not f.exists():
        raise FileNotFoundError(f"{name}: {f} missing; run geoae/extraction/atlas_classes_prep.py first (stage prep)")
    cfg = json.load(open(f))
    cfg["local_parquet"] = str(d / cfg["local_parquet"])
    return cfg


# Frozen control corpora (emotion6 / news20 / trec6 / jigsaw2), written by geoae/extraction/control_datasets.py to
# <CONTROL_DATASETS_DIR>/<name>/{cfg.json,data.parquet} and loaded lazily by set_dataset like the atlas draws.
# <name>_s<N> is a document redraw of <name> (same classes and prompt, geoae/extraction/control_redraw.py), found on disk.
def control_datasets_dir():
    import os
    from pathlib import Path
    d = os.environ.get("GEOAE_CONTROL_DATASETS_DIR", "results/delta/control_datasets")
    p = Path(d)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def _control_datasets():
    import re
    base = ("emotion6", "news20", "trec6", "jigsaw2")
    more = sorted(f.parent.name for f in control_datasets_dir().glob("*_s*/cfg.json")
                  if re.fullmatch("(" + "|".join(base) + r")_s\d+", f.parent.name))
    return base + tuple(more)


CONTROL_DATASETS = _control_datasets()


def _load_control_cfg(name: str) -> dict:
    import json
    d = control_datasets_dir() / name
    f = d / "cfg.json"
    if not f.exists():
        raise FileNotFoundError(f"{name}: {f} missing; run geoae/extraction/control_datasets.py first")
    cfg = json.load(open(f))
    cfg["local_parquet"] = str(d / cfg["local_parquet"])
    return cfg


def set_dataset(name: str) -> dict:
    """Install CLASSES / PROMPT_HEAD (and MAX_NEW) for `name` and return its config dict."""
    global CLASSES, PROMPT_HEAD, MAX_NEW
    if name not in DATASET_CONFIGS and name in ATLAS_CLASS_DATASETS:
        DATASET_CONFIGS[name] = _load_atlas_cfg(name)
    if name not in DATASET_CONFIGS and name in CONTROL_DATASETS:
        DATASET_CONFIGS[name] = _load_control_cfg(name)
    cfg = DATASET_CONFIGS[name]
    CLASSES = cfg["classes"]
    PROMPT_HEAD = cfg["prompt_head"]
    MAX_NEW = int(cfg.get("max_new", 6))
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


def _prompt_max_length(lm, max_new=None):
    """Reserve GPT-2's finite absolute-position window for the continuation.

    Generation prepares its own position_ids, so the forward padding hook
    cannot prevent a 1024-token prompt from exhausting GPT-2's position table
    on subsequent decoding steps. Capture the same shortened prompt used for
    prediction, keeping the fitted activation and intervention aligned.
    Other model families retain the existing 1024-token prompt limit.
    """
    cfg = getattr(lm, "config", None)
    if getattr(cfg, "model_type", None) != "gpt2":
        return 1024
    context = int(cfg.max_position_embeddings)
    reserve = MAX_NEW if max_new is None else int(max_new)
    if not 0 < reserve < context:
        raise ValueError(
            f"max_new_tokens must be between 1 and {context - 1} for GPT-2's "
            f"{context}-position context window; got {reserve}."
        )
    return min(1024, context - reserve)


def _tok(tokenizer, texts, device, lm=None, max_new=None):
    return tokenizer([build_prompt(t) for t in texts], padding=True, truncation=True,
                     max_length=_prompt_max_length(lm, max_new), return_tensors="pt").to(device)


@torch.no_grad()
def capture_h(lm, tokenizer, texts, layer, device, batch_size=16, return_ids=False, max_new=None):
    """Post-layer (pre-norm) residual at orig_lens-1 — same point as reference mask_layer.
    With return_ids, also the token id at that position: (H, ids)."""
    H, I = [], []
    for s in tqdm(range(0, len(texts), batch_size), desc="capture", leave=False):
        enc = _tok(tokenizer, texts[s:s + batch_size], device, lm=lm, max_new=max_new)
        if return_ids:
            h, ids = nl.capture_post_layer(lm, enc, layer, return_ids=True)
            H.append(h); I.append(ids)
        else:
            H.append(nl.capture_post_layer(lm, enc, layer))
    H = np.concatenate(H, axis=0)
    return (H, np.concatenate(I)) if return_ids else H


@torch.no_grad()
def predict(lm, tokenizer, texts, labels, device, batch_size=16, max_new=None):
    """Few-shot classification via greedy generation. Returns correct/gold_conf/pred."""
    max_new = MAX_NEW if max_new is None else max_new
    preds, gold_conf, correct = [], [], []
    for s in tqdm(range(0, len(texts), batch_size), desc="predict", leave=False):
        bt, bl = texts[s:s+batch_size], labels[s:s+batch_size]
        enc = _tok(tokenizer, bt, device, lm=lm, max_new=max_new)
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
    # Mode affects reconstruction even for identical tensors. Preserve existing
    # plain/bypass cache keys, but distinguish the decoder-only experiment.
    if getattr(ae, "token_bias_mode", "bypass") == "decoder_only":
        h.update(b"token_bias_mode=decoder_only")
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
    need_by_split: dict | None = None,
    max_try: int | None = None,
    extra_recons: dict | None = None,
    extra_key: str | None = None,
):
    """
    Documents that BOTH the base LM and the AE-recon splice classify correctly.

    `extra_recons` ({name: replacement fn}, e.g. an SAE-reconstruction splice) adds further splices a document must also
    survive; `extra_key` describes them (and any extra data source) and is stored in / checked against the cache, so a
    pool built without them is never reused for a run that asks for them. Both default to None = the original behaviour.

    This is the evaluation population for the steering / causal-concept
    comparisons, so it is specific to (prompt, dataset, MODEL, AE checkpoint) —
    not just the prompt. The on-disk cache therefore records `model_name`, and a
    cache written by a different model (or by a version that predates the field,
    which reads back as None) is rebuilt rather than trusted. Reusing another
    model's set silently swaps the evaluation population and skips this
    checkpoint's reconstruction filter entirely.

    `need_by_split` ({"fit": n, "test": m}; used with a dataset that has a `split_column`, e.g. the Atlas class corpora)
    builds the pool per document-level split instead: up to n joint-correct fit docs then m joint-correct test docs per
    class, each doc tagged {"split": ...}, trying at most `max_try` docs per (class, split). Per-(class, split) counts of
    docs tried / base-correct / recon-correct / both are stored under "stats" in the cache file (additive).

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
    prompt_max_length = _prompt_max_length(lm)
    limited_context = prompt_max_length < 1024
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
                and cached.get("extra") == extra_key
                and (not limited_context or
                     (cached.get("prompt_max_length") == prompt_max_length
                      and cached.get("max_new_tokens") == MAX_NEW))
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
            elif cached.get("extra") != extra_key:
                why = f"built with DIFFERENT extra filters ({cached.get('extra')!r} != {extra_key!r})"
            elif limited_context and (cached.get("prompt_max_length") != prompt_max_length
                                      or cached.get("max_new_tokens") != MAX_NEW):
                why = "built under a DIFFERENT generation context budget"
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

    stats = {}
    tried, xstat = {}, {}

    def _both_correct(texts, c, st=None):
        bs = len(texts)
        r_base = predict(lm, tokenizer, texts, [c] * bs, device=device, batch_size=bs)
        hook.activate(z_recon)
        r_recon = predict(lm, tokenizer, texts, [c] * bs, device=device, batch_size=bs)
        hook.deactivate()
        # extra splices are tried only on the documents that survived so far (a document failing the base model or
        # an earlier splice cannot enter the pool, so the later forward passes would be wasted on it)
        alive = [i for i in range(bs) if r_base["correct"][i] and r_recon["correct"][i]]
        for name, fn in (extra_recons or {}).items():
            st_x = xstat.setdefault(name, {}).setdefault(c, [0, 0])          # [entered, survived]
            st_x[0] += len(alive)
            if not alive:
                continue
            hook.activate(fn)
            r_x = predict(lm, tokenizer, [texts[i] for i in alive], [c] * len(alive), device=device, batch_size=len(alive))
            hook.deactivate()
            alive = [i for i, ok in zip(alive, r_x["correct"]) if ok]
            st_x[1] += len(alive)
        tried[c] = tried.get(c, 0) + bs
        if st is not None:
            st["n_tried"] += bs
            st["base_correct"] += int(sum(r_base["correct"]))
            st["recon_correct"] += int(sum(r_recon["correct"]))
            st["both_correct"] += int(sum(b and r for b, r in zip(r_base["correct"], r_recon["correct"])))
        return [texts[i] for i in alive]

    correct = []
    if need_by_split:
        split_col = dataset_cfg["split_column"]
        by_cs = {c: {"fit": [], "test": []} for c in range(n_classes)}
        for ex in ds:
            by_cs[int(ex[label_col])][ex[split_col]].append(ex[dataset_cfg["text_column"]].strip())
        for c in build:
            for sp in ("fit", "test"):
                st = stats[f"{c}/{sp}"] = {"n_tried": 0, "base_correct": 0, "recon_correct": 0, "both_correct": 0}
                print(f"[{tag}] Finding joint-correct {sp} docs for class {c} ({CLASSES[c]}) ...")
                got, texts_ = [], by_cs[c][sp][:max_try] if max_try else by_cs[c][sp]
                for s0 in range(0, len(texts_), batch_size):
                    got += [{"text": t, "label": c, "split": sp}
                            for t in _both_correct(texts_[s0:s0 + batch_size], c, st)]
                    if len(got) >= need_by_split[sp]:
                        break
                got = got[:need_by_split[sp]]
                print(f"  Found {len(got)} joint-correct {sp} docs for class {c} "
                      f"(tried {st['n_tried']}: base {st['base_correct']}, recon {st['recon_correct']})")
                correct += got
    else:
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
               "prompt_max_length": prompt_max_length, "max_new_tokens": MAX_NEW,
               "classes": CLASSES, "classes_built": build, "docs": correct,
               **({"extra": extra_key, "tried_by_class": {str(c): n for c, n in tried.items()},
                   "extra_stats": {n: {str(c): {"entered": v[0], "survived": v[1]} for c, v in d.items()}
                                   for n, d in xstat.items()}} if extra_key is not None else {}),
               **({"stats": stats, "need_by_split": need_by_split, "max_try": max_try} if need_by_split else {})},
              open(corr_path, "w"))
    cnt = Counter(d["label"] for d in correct)
    print(f"[{tag}] joint-correct: {len(correct)} docs "
          f"(min/class={min(cnt.get(c, 0) for c in range(n_classes))}, "
          f"prompt_sha={prompt_sha}, model={model_name}, ae={ae_sha}) -> {corr_path}")
    return correct
