"""
Cache activations + ground truth for a LADDER of concept types, so we can ask
WHICH KINDS of structure a GeoAE latent acquires and when.

Each rung is a genuinely different claim about what a cluster could mean. They
are cached separately because this project keeps finding that a result at one
rung predicts nothing about another: the morphology axes all tie while genre
alignment differs; token-presence and token-fraction readouts differ 15x on the
same clusters; balanced k-means with no encoder matches the AE on concepts while
plain k-means loses badly.

  RUNG              SOURCE                              GRAIN        CLASSES
  surface           derived from the token string       token        ~7
  pos_coarse        batterydata/pos_tagging             token        12
  pos_fine          same, Penn tags                     token        44   <- NN vs NNS
  ner_coarse        DFKI-SLT/few-nerd                   token        8
  ner_fine          same                                token        66   <- art-film vs
                                                                             building-airport
  sentiment         stanfordnlp/sst2                    sentence     2
  sentiment_long    stanfordnlp/imdb                    document     2
  subjectivity      SetFit/subj                         sentence     2
  formality         pavlick-formality-scores            sentence     3 (bucketed)
  language          papluca/language-identification     sentence     20
  domain            code / math / prose                 document     3
  topic4            fancyzhx/ag_news                    document     4
  topic14           fancyzhx/dbpedia_14                 document     14
  topic20           SetFit/20_newsgroups                document     20
  (atlas: document 31, tone 587, content 12786, entity 3386 — cached separately)

Token-level rungs store EVERY token's activation with its label. Sequence-level
rungs store both the LAST-token and the MEAN-POOLED activation per document,
since a token-level AE has no sequence representation of its own and the two
pooling choices are different questions.

Activations are float32 and asserted finite: gemma-scale layers overflow fp16's
65504 ceiling and a cache silently went 70% inf earlier in this project.

Usage:
    python -m geoae.interp.concept_suite --checkpoint <any ckpt> --out cache/
    python -m geoae.interp.concept_suite --checkpoint <ckpt> --only topic14,ner
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.hooks import SplicingHook

POS_COARSE = {
    "NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN",
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB",
    "VBZ": "VERB", "MD": "VERB",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON", "EX": "PRON",
    "DT": "DET", "PDT": "DET", "WDT": "DET",
    "IN": "ADP", "TO": "PART", "RP": "PART", "POS": "PART",
    "CC": "CONJ", "CD": "NUM", "UH": "INTJ", "FW": "X",
}

# name -> (hf_name, config, split, text_field, label_field, n)
SEQ_SOURCES = {
    "sentiment":      ("stanfordnlp/sst2", None, "train", "sentence", "label", 3000),
    "sentiment_long": ("stanfordnlp/imdb", None, "train", "text", "label", 2000),
    "subjectivity":   ("SetFit/subj", None, "train", "text", "label", 3000),
    "language":       ("papluca/language-identification", None, "train", "text", "labels", 3000),
    "topic4":         ("fancyzhx/ag_news", None, "train", "text", "label", 3000),
    "topic14":        ("fancyzhx/dbpedia_14", None, "train", "content", "label", 3000),
    "topic20":        ("SetFit/20_newsgroups", None, "train", "text", "label", 3000),
}


def surface_class(s: str) -> str:
    if s.strip() == "":
        return "whitespace"
    t = s.strip()
    if re.fullmatch(r"[^\w\s]+", t):
        return "punct"
    if re.fullmatch(r"[0-9][0-9.,:/]*", t):
        return "numeric"
    if not s[:1].isspace() and s[:1] != " " and t.isalpha():
        return "subword"
    if t[:1].isupper():
        return "capitalised"
    if t.isalpha():
        return "lowercase"
    return "mixed"


class Encoder:
    """Runs the frozen LM once and hands back layer-L residuals."""

    def __init__(self, model_name, layer, dev):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.lm = load_lm(model_name, device_map="auto")
        self.hook = SplicingHook(self.lm, layer)
        self.cap: list[torch.Tensor] = []
        self.hook.activate(lambda hs: (self.cap.append(hs.detach()) or hs))
        self.dev = dev

    @torch.no_grad()
    def run(self, id_seqs):
        T = max(len(x) for x in id_seqs)
        inp = torch.full((len(id_seqs), T), self.tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(id_seqs), T), dtype=torch.long)
        for i, x in enumerate(id_seqs):
            inp[i, : len(x)] = torch.tensor(x)
            am[i, : len(x)] = 1
        self.lm(input_ids=inp.to(self.dev), attention_mask=am.to(self.dev), use_cache=False)
        hs = self.cap.pop()
        self.cap.clear()
        return hs


def write(out: Path, name: str, **arrays):
    out.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        if isinstance(v, np.ndarray) and v.dtype == np.float32 and not np.isfinite(v).all():
            raise SystemExit(f"[suite] {name}/{k}: non-finite — refusing to write a poisoned cache")
    np.savez(out / f"{name}.npz", **arrays)
    n = next(iter(arrays.values())).shape[0]
    mb = sum(v.nbytes for v in arrays.values()) / 1e6
    print(f"[suite] {name:<16} {n:>7,} rows  {mb:>7.0f} MB")


def cache_token_source(enc, out, name, sents, tags_of, bs=32, max_words=64):
    """Every token gets its own activation and its word's gold label."""
    H, LAB, TID, FIRST = [], [], [], []
    for s in tqdm(range(0, len(sents), bs), desc=f"[suite] {name}", leave=False):
        seqs, metas = [], []
        for r in sents[s : s + bs]:
            words, tags = tags_of(r)
            words, tags = words[:max_words], tags[:max_words]
            ids, meta = [], []
            for wi, (w, t) in enumerate(zip(words, tags)):
                sub = enc.tok.encode((" " if wi else "") + w, add_special_tokens=False)
                if not sub:
                    continue
                ids += sub
                meta += [(t, j == 0) for j in range(len(sub))]
            if ids:
                seqs.append(ids)
                metas.append(meta)
        if not seqs:
            continue
        hs = enc.run(seqs)
        for i, meta in enumerate(metas):
            n = len(meta)
            H.append(hs[i, :n].float().cpu().numpy())
            LAB += [m[0] for m in meta]
            FIRST += [m[1] for m in meta]
            TID += seqs[i][:n]
    H = np.concatenate(H).astype(np.float32)
    toks = np.array([enc.tok.decode([t]) for t in TID], dtype=object)
    write(out, name, H=H, label=np.array(LAB, dtype=object),
          token_id=np.array(TID), word_first=np.array(FIRST, dtype=bool),
          surface=np.array([surface_class(t) for t in toks], dtype=object))


def cache_seq_source(enc, out, name, texts, labels, bs=16, max_len=192):
    """Last-token AND mean-pooled activation per document."""
    LAST, MEAN = [], []
    for s in tqdm(range(0, len(texts), bs), desc=f"[suite] {name}", leave=False):
        chunk = [enc.tok.encode(t, add_special_tokens=False)[:max_len] or [enc.tok.eos_token_id]
                 for t in texts[s : s + bs]]
        hs = enc.run(chunk)
        for i, ids in enumerate(chunk):
            n = len(ids)
            LAST.append(hs[i, n - 1].float().cpu().numpy())
            MEAN.append(hs[i, :n].float().mean(0).cpu().numpy())
    write(out, name, H_last=np.stack(LAST).astype(np.float32),
          H_mean=np.stack(MEAN).astype(np.float32),
          label=np.array(labels, dtype=object))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="any AE ckpt — supplies model_name + layer")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--only", default=None, help="comma list of source names")
    ap.add_argument("--n_token_sents", type=int, default=6000)
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else None
    out = Path(args.out)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, ck = load_ae_checkpoint(args.checkpoint, "cpu")
    model_name = ck["config"]["extraction"]["model_name"]
    layer = ck["config"]["data"]["target_layer"]
    print(f"[suite] {model_name} layer {layer} -> {out}/")
    enc = Encoder(model_name, layer, dev)

    def todo(n):
        return (want is None or n in want) and not (out / f"{n}.npz").exists()

    if todo("pos"):
        ds = load_dataset("batterydata/pos_tagging", split="train", streaming=True)
        sents = [r for r in itertools.islice(iter(ds), args.n_token_sents) if len(r["words"]) >= 3]
        cache_token_source(enc, out, "pos", sents, lambda r: (r["words"], r["labels"]))

    if todo("ner"):
        ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split="train", streaming=True)
        feat = ds.features["fine_ner_tags"].feature.names
        coarse = ds.features["ner_tags"].feature.names
        sents = [r for r in itertools.islice(iter(ds), args.n_token_sents) if len(r["tokens"]) >= 3]
        cache_token_source(enc, out, "ner", sents,
                           lambda r: (r["tokens"], [feat[i] for i in r["fine_ner_tags"]]))
        # coarse labels alongside, same row order
        lab = []
        for r in sents:
            words = r["tokens"][:64]
            for wi, w in enumerate(words):
                sub = enc.tok.encode((" " if wi else "") + w, add_special_tokens=False)
                lab += [coarse[r["ner_tags"][wi]]] * len(sub)
        np.save(out / "ner_coarse.npy", np.array(lab, dtype=object))

    for name, (hf, cfg, split, tf, lf, n) in SEQ_SOURCES.items():
        if not todo(name):
            continue
        # MUST shuffle: several of these (dbpedia_14, imdb, ag_news) ship sorted
        # by label, so taking the first N off a raw stream yields ONE class.
        ds = load_dataset(hf, cfg, split=split, streaming=True).shuffle(seed=0, buffer_size=20000)
        rows = [r for r in itertools.islice(iter(ds), n * 2) if r.get(tf)][:n]
        cache_seq_source(enc, out, name, [r[tf] for r in rows], [r[lf] for r in rows])

    if todo("formality"):
        ds = load_dataset("osyvokon/pavlick-formality-scores", split="train",
                          streaming=True).shuffle(seed=0, buffer_size=20000)
        rows = [r for r in itertools.islice(iter(ds), 6000) if r.get("sentence")]
        sc = np.array([r["avg_score"] for r in rows])
        lo, hi = np.quantile(sc, [0.33, 0.67])
        lab = ["informal" if s < lo else "formal" if s > hi else "neutral" for s in sc]
        cache_seq_source(enc, out, "formality", [r["sentence"] for r in rows], lab)

    if todo("domain"):
        texts, labs = [], []
        for hf, cfg, field, tag, n in [
            ("google/code_x_glue_ct_code_to_text", "python", "code", "code", 1200),
            ("math-ai/StackMathQA", None, "Q", "math", 1200),
            ("wikimedia/wikipedia", "20231101.en", "text", "prose", 1200)]:
            ds = load_dataset(hf, cfg, split="train", streaming=True)
            for r in itertools.islice(iter(ds), n):
                v = r.get(field)
                if v:
                    texts.append(v[:1200]); labs.append(tag)
        cache_seq_source(enc, out, "domain", texts, labs)

    print("[suite] done")


if __name__ == "__main__":
    main()
