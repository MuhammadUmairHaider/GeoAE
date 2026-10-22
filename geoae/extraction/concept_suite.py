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
  toxicity          Jigsaw toxic comments (Kaggle 2018) comment      2 (balanced: any flag vs clean; WITH BOS)
  toxicity_pattern  same rows, exact set of the 6 flags  comment      ~30
  biasbios          LabHC/bias_in_bios test split       biography    28 professions x 150 (WITH BOS); gender alongside
  (atlas: document 31, tone 587, content 12786, entity 3386 — cached separately)

Token-level rungs store EVERY token's activation with its label. Sequence-level
rungs store both the LAST-token and the MEAN-POOLED activation per document,
since a token-level AE has no sequence representation of its own and the two
pooling choices are different questions.

Activations are float32 and asserted finite: gemma-scale layers overflow fp16's
65504 ceiling and a cache silently went 70% inf earlier in this project.

Usage:
    python -m geoae.extraction.concept_suite --checkpoint <any ckpt> --out cache/
    python -m geoae.extraction.concept_suite --checkpoint <ckpt> --only topic14,ner
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from geoae.checkpoint import load_tokenizer

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
        self.tok = load_tokenizer(model_name)
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


def cache_seq_source(enc, out, name, texts, labels, bs=16, max_len=192, bos=False, **extra):
    """Last-token AND mean-pooled activation per document, plus the last token's id
    (a token-bypass AE needs it to encode H_last).

    bos=True prepends the tokenizer's BOS, as the training dumps do (extract_sampled tokenises with the
    default add_special_tokens=True); the mean then skips the BOS position. The older rungs were cached
    without BOS. A tokenizer without a BOS token (Qwen2.5) adds none, exactly as in its training dump.
    `extra` arrays are stored alongside, one entry per text."""
    LAST, MEAN, LAST_ID = [], [], []
    has_bos = bos and enc.tok.bos_token_id is not None
    for s in tqdm(range(0, len(texts), bs), desc=f"[suite] {name}", leave=False):
        chunk = [enc.tok.encode(t, add_special_tokens=bos)[:max_len] or [enc.tok.eos_token_id]
                 for t in texts[s : s + bs]]
        if has_bos:
            assert all(ids[0] == enc.tok.bos_token_id for ids in chunk), "tokenizer did not prepend BOS"
        hs = enc.run(chunk)
        for i, ids in enumerate(chunk):
            n = len(ids)
            LAST.append(hs[i, n - 1].float().cpu().numpy())
            LAST_ID.append(ids[-1])
            lo = 1 if has_bos and n > 1 else 0
            MEAN.append(hs[i, lo:n].float().mean(0).cpu().numpy())
    write(out, name, H_last=np.stack(LAST).astype(np.float32),
          H_mean=np.stack(MEAN).astype(np.float32),
          last_token_id=np.array(LAST_ID, dtype=np.int64),
          label=np.array(labels, dtype=object), **extra)


def sample_sequence_rows(hf, config, split, text_field, label_field, n, seed=0):
    """Sample across the full split, including datasets sorted by class.

    A streaming shuffle only mixes a bounded prefix. DBpedia has 40,000-row
    class blocks, so a 20,000-row buffer still yields a single-class cache.
    Map-style shuffling permutes indices across the complete dataset instead.
    """
    ds = load_dataset(hf, config, split=split, streaming=False).shuffle(seed=seed)
    rows = list(itertools.islice((r for r in ds if r.get(text_field)), n))
    if len(rows) != n:
        raise ValueError(f"{hf}: expected {n} nonempty texts, got {len(rows)}")
    if len({r[label_field] for r in rows}) < 2:
        raise ValueError(f"{hf}: sampled fewer than two classes; refusing to encode")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="any AE ckpt — supplies model_name + layer "
                    "(optional when --model_name and --layer are both given)")
    ap.add_argument("--model_name", default=None, help="LM name; with --layer, replaces --checkpoint")
    ap.add_argument("--layer", type=int, default=None, help="decoder layer; with --model_name, replaces --checkpoint")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--only", default=None, help="comma list of source names")
    ap.add_argument("--n_token_sents", type=int, default=6000)
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else None
    out = Path(args.out)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Only model_name + layer are read; the AE itself is discarded, so a token-bypass
    # checkpoint is fine here (its bias is never used).
    if args.checkpoint:
        if args.model_name is not None or args.layer is not None:
            raise SystemExit("[suite] --checkpoint is mutually exclusive with --model_name/--layer")
        _, _, _, ck = load_ae_checkpoint(args.checkpoint, "cpu", allow_token_bias=True)
        model_name = ck["config"]["extraction"]["model_name"]
        layer = ck["config"]["data"]["target_layer"]
    else:
        if args.model_name is None or args.layer is None:
            raise SystemExit("[suite] give --checkpoint, or both --model_name and --layer")
        model_name, layer = args.model_name, args.layer
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
        rows = sample_sequence_rows(hf, cfg, split, tf, lf, n)
        cache_seq_source(enc, out, name, [r[tf] for r in rows], [r[lf] for r in rows])

    if todo("formality"):
        ds = load_dataset("osyvokon/pavlick-formality-scores", split="train",
                          streaming=True).shuffle(seed=0, buffer_size=20000)
        rows = [r for r in itertools.islice(iter(ds), 6000) if r.get("sentence")]
        sc = np.array([r["avg_score"] for r in rows])
        lo, hi = np.quantile(sc, [0.33, 0.67])
        lab = ["informal" if s < lo else "formal" if s > hi else "neutral" for s in sc]
        cache_seq_source(enc, out, "formality", [r["sentence"] for r in rows], lab)

    if todo("toxicity"):
        # Jigsaw 2018 train split, six binary flags. Only ~10% of comments carry a flag, so draw 1,500 flagged and
        # 1,500 unflagged comments from one seeded shuffle of the whole split. Stored labels: `label` = any flag vs
        # none, `label_pattern` = the exact set of flags (lossless single label), `flags` = the six columns.
        # Encoded WITH BOS, as the training dumps are. The CSV is read through the hub file cache because a
        # data_files load of a hub repo cannot be resolved offline.
        csv = hf_hub_download("thesofakillers/jigsaw-toxic-comment-classification-challenge", "train.csv",
                              repo_type="dataset")
        ds = load_dataset("csv", data_files=csv, split="train").shuffle(seed=0)
        flags = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        picked = {"toxic": [], "clean": []}
        for r in ds:
            if not (r["comment_text"] or "").strip():
                continue
            lab = "toxic" if any(int(r[f]) for f in flags) else "clean"
            if len(picked[lab]) < 1500:
                picked[lab].append((r["comment_text"], [int(r[f]) for f in flags]))
            if all(len(v) == 1500 for v in picked.values()):
                break
        rows = picked["toxic"] + picked["clean"]
        F = np.array([f for _, f in rows], dtype=np.int8)
        pattern = ["+".join(n for n, v in zip(flags, f) if v) or "clean" for f in F]
        cache_seq_source(enc, out, "toxicity", [t for t, _ in rows], ["toxic"] * 1500 + ["clean"] * 1500,
                         bos=True, flags=F, flag_names=np.array(flags, dtype=object),
                         label_pattern=np.array(pattern, dtype=object))

    if todo("biasbios"):
        # Bias in Bios (De-Arteaga et al. 2019) test split, hard_text (the sentence naming the profession removed).
        # 150 biographies per profession, up to 75 per gender within each profession and the other gender filling the
        # rest (only rapper, dj and a few male-rare professions fall short), so the profession label is balanced and
        # gender is close to independent of profession. Labels: `label` = profession, `label_gender` = gender;
        # `source_index` = row in the seeded shuffle of the split. Encoded WITH BOS, as the training dumps are.
        from geoae.extraction.biasbios import GENDERS
        from geoae.extraction.biasbios import PROFESSIONS
        ds = load_dataset("LabHC/bias_in_bios", split="test").shuffle(seed=0)
        prof, gen = np.asarray(ds["profession"]), np.asarray(ds["gender"])
        pick = []
        for p in range(len(PROFESSIONS)):
            idx = [np.flatnonzero((prof == p) & (gen == g)) for g in (0, 1)]
            take = [min(75, len(i)) for i in idx]
            for g in (0, 1):
                if take[g] < 75:
                    take[1 - g] = min(150 - take[g], len(idx[1 - g]))
            pick += list(idx[0][:take[0]]) + list(idx[1][:take[1]])
        pick = np.asarray(pick, dtype=np.int64)
        cache_seq_source(enc, out, "biasbios", ds.select(pick.tolist())["hard_text"],
                         [PROFESSIONS[i] for i in prof[pick]], bos=True,
                         label_gender=np.array([GENDERS[i] for i in gen[pick]], dtype=object), source_index=pick)

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
