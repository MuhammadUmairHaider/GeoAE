"""
Build the token-level concept cache used by training_dynamics.

The sequence-level caches already exist (FineWeb-Atlas chunks; DBpedia last-token
activations under dbpedia/activations/). What was missing is TOKEN-level ground
truth, so the concept ladder had a hole at its concrete end: we could ask whether
a cluster predicts a document's genre but not whether it predicts a token's part
of speech.

This pass runs the LM over a POS-tagged corpus (batterydata/pos_tagging, Penn
Treebank tags) and stores, per LM token:

  activation      layer-L residual, float32 (NOT fp16 — gemma-scale layers
                  overflow fp16's 65504 ceiling and 70% of a cached tensor
                  silently became inf last time; the writer asserts finiteness)
  pos             gold Penn tag of the word this token belongs to
  token_id        so surface classes and monosemanticity are derivable
  word_first      whether this token starts a word (subword continuations carry
                  their word's tag but are marked, since a tag on "##ing" is a
                  different claim than a tag on a whole word)

Alignment: each whitespace word is tokenised separately with a leading space, so
LM tokens map to words by construction rather than by offset arithmetic.

Usage:
    python -m geoae.interp.concept_cache --checkpoint <any ckpt for model+layer> \
      --n_sentences 4000 --out cache/pos_l27
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.hooks import SplicingHook

# Penn tags collapsed to coarse classes. Both are kept: the fine tag asks whether
# a cluster separates VBD from VBN (a hard, detailed distinction), the coarse one
# whether it separates verbs from nouns at all.
COARSE = {
    "NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN",
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON",
    "DT": "DET", "PDT": "DET", "WDT": "DET",
    "IN": "ADP", "TO": "PART", "RP": "PART", "POS": "PART",
    "CC": "CONJ", "CD": "NUM", "MD": "VERB", "UH": "INTJ", "EX": "PRON", "FW": "X",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="any AE ckpt — supplies model_name + layer")
    ap.add_argument("--n_sentences", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_words", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, ck = load_ae_checkpoint(args.checkpoint, "cpu")
    model_name = ck["config"]["extraction"]["model_name"]
    layer = ck["config"]["data"]["target_layer"]
    print(f"[cache] {model_name} layer {layer}")

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    lm = load_lm(model_name, device_map="auto")
    hook = SplicingHook(lm, layer)
    cap: list[torch.Tensor] = []
    hook.activate(lambda hs: (cap.append(hs.detach()) or hs))

    ds = load_dataset("batterydata/pos_tagging", split="train", streaming=True)
    sents = [r for r in itertools.islice(iter(ds), args.n_sentences) if 3 <= len(r["words"])]
    print(f"[cache] {len(sents)} sentences")

    H, POS, TID, FIRST, WORD = [], [], [], [], []
    with torch.no_grad():
        for s in tqdm(range(0, len(sents), args.batch_size), desc="[cache] encoding"):
            batch = sents[s:s + args.batch_size]
            seqs, metas = [], []
            for r in batch:
                words, tags = r["words"][: args.max_words], r["labels"][: args.max_words]
                ids, per_tok = [], []
                for wi, (w, t) in enumerate(zip(words, tags)):
                    sub = tok.encode((" " if wi else "") + w, add_special_tokens=False)
                    if not sub:
                        continue
                    ids += sub
                    per_tok += [(t, j == 0, w) for j in range(len(sub))]
                if ids:
                    seqs.append(ids); metas.append(per_tok)
            if not seqs:
                continue
            T = max(len(x) for x in seqs)
            inp = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(seqs), T), dtype=torch.long)
            for i, x in enumerate(seqs):
                inp[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
            lm(input_ids=inp.to(dev), attention_mask=am.to(dev), use_cache=False)
            hs = cap.pop(); cap.clear()
            for i, meta in enumerate(metas):
                n = len(meta)
                H.append(hs[i, :n].float().cpu().numpy())
                POS += [m[0] for m in meta]
                FIRST += [m[1] for m in meta]
                WORD += [m[2] for m in meta]
                TID += seqs[i][:n]
    hook.deactivate()

    H = np.concatenate(H).astype(np.float32)
    if not np.isfinite(H).all():
        raise SystemExit("[cache] non-finite activations — refusing to write a poisoned cache")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out) + ".npz", H=H, token_id=np.array(TID, dtype=np.int64),
             word_first=np.array(FIRST, dtype=bool))
    np.save(str(out) + "_pos.npy", np.array(POS))
    np.save(str(out) + "_coarse.npy", np.array([COARSE.get(p, "OTHER") for p in POS]))
    np.save(str(out) + "_word.npy", np.array(WORD, dtype=object), allow_pickle=True)
    print(f"[cache] {H.shape[0]:,} tokens, {H.nbytes/1e9:.2f} GB, "
          f"{len(set(POS))} fine tags -> {out}.npz")


if __name__ == "__main__":
    main()
