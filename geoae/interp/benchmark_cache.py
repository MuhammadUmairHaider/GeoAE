"""
Token-level concept caches from SAE-interpretability benchmarks (MIB).

The existing token rungs (POS, NER) test whether a cluster tracks a token's
grammatical or entity category. Both are strongly predictable from the token
STRING, so a clustering can score well on them by being lexical. These two
benchmarks are chosen because they cannot be won that way.

RAVEL  (mib-bench/ravel, also used by SAEBench)
    One entity token carries THREE independent labels — Country, Continent,
    Language. The same token "Biu" is simultaneously Nigeria / Africa / English.
    So this measures DISENTANGLEMENT: a cluster that is pure on Country while
    mixing Language has isolated one attribute from the others, which is exactly
    the property a good latent should have and which purity on a single label
    set cannot detect.

IOI  (mib-bench/ioi)
    The label is the token's ROLE — first subject mention, indirect object, or
    second subject mention — and the same NAME appears in every role across the
    dataset. Token identity therefore carries zero information about the label,
    so any purity above chance is positional/syntactic structure rather than
    lexical memorisation. This is the control the POS/NER rungs lack.

    Roles follow the IOI template "As {A} and {B} left the {place}, {C} gave ...":
    S1 = first mention of the subject, IO = the indirect object (the answer),
    S2 = the repeated subject. IO is what the model must predict.

NOT INCLUDED: pyvene/axbench-concept500 is sequence-level — its `input`/`output`
pair carries one concept label for a whole generation — so it cannot produce
token-level purity. Use it for a large-vocabulary sequence rung if wanted, but
it does not answer the token-level question.

Activations are float32 and finiteness is asserted before writing: gemma-scale
layers overflow fp16's 65504 ceiling, and 70% of a cached tensor silently became
inf the last time that was skipped.

Usage:
    python -m geoae.interp.benchmark_cache --bench ravel \
        --checkpoint <any ckpt for model+layer> --n_rows 6000 --out cache/ravel
    python -m geoae.interp.benchmark_cache --bench ioi \
        --checkpoint <any ckpt for model+layer> --n_rows 6000 --out cache/ioi

Then add to concept_probe.LADDER (already done by --register):
    ("ravel_country",   "ravel.npz", "Country",   "token", "RAVEL entity attribute")
    ("ravel_continent", "ravel.npz", "Continent", "token", "RAVEL entity attribute")
    ("ravel_language",  "ravel.npz", "Language",  "token", "RAVEL entity attribute")
    ("ioi_role",        "ioi.npz",   "role",      "token", "IOI S1/IO/S2 position")
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.hooks import SplicingHook

RAVEL_ATTRS = ["Country", "Continent", "Language"]


def last_span(offsets, lo, hi):
    """Index of the LAST token whose char span overlaps [lo, hi)."""
    hit = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo and b > a]
    return hit[-1] if hit else None


def find_occurrences(text, needle):
    """Char spans of every occurrence of `needle` in `text`."""
    out, i = [], text.find(needle)
    while i != -1:
        out.append((i, i + len(needle)))
        i = text.find(needle, i + 1)
    return out


def build_ravel(n_rows):
    """-> list of (prompt, char_span_of_query_entity, {label_name: value}, entity)"""
    ds = load_dataset("mib-bench/ravel", split="train", streaming=True)
    items = []
    for r in itertools.islice(iter(ds), n_rows * 3):
        ent, prompt = r.get("entity"), r.get("prompt")
        if not ent or not prompt:
            continue
        labs = {a: r.get(a) for a in RAVEL_ATTRS}
        if any(v in (None, "") for v in labs.values()):
            continue
        occ = find_occurrences(prompt, ent)
        if not occ:
            continue
        items.append((prompt, occ[-1], labs, ent))     # LAST mention = the query
        if len(items) >= n_rows:
            break
    return items


def build_ioi(n_rows):
    """-> list of (prompt, char_span, {'role': S1|IO|S2, 'name': str}, name)"""
    ds = load_dataset("mib-bench/ioi", split="train", streaming=True)
    items = []
    for r in itertools.islice(iter(ds), n_rows * 3):
        prompt = r.get("prompt")
        md = r.get("metadata")
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except json.JSONDecodeError:
                continue
        if not prompt or not isinstance(md, dict):
            continue
        subj, io = md.get("subject"), md.get("indirect_object")
        if not subj or not io or subj == io:
            continue
        s_occ = find_occurrences(prompt, subj)
        i_occ = find_occurrences(prompt, io)
        if len(s_occ) < 2 or len(i_occ) < 1:
            continue                                    # need S1 and S2 present
        for span, role, nm in ((s_occ[0], "S1", subj),
                               (i_occ[0], "IO", io),
                               (s_occ[1], "S2", subj)):
            items.append((prompt, span, {"role": role, "name": nm}, nm))
        if len({id(x) for x in items}) >= n_rows * 3 or len(items) >= n_rows * 3:
            break
    return items[: n_rows * 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, choices=["ravel", "ioi"])
    ap.add_argument("--checkpoint", required=True, help="any AE ckpt — supplies model_name + layer")
    ap.add_argument("--n_rows", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, ck = load_ae_checkpoint(args.checkpoint, "cpu")
    model_name = ck["config"]["extraction"]["model_name"]
    layer = ck["config"]["data"]["target_layer"]
    print(f"[bench] {args.bench}: {model_name} layer {layer}")

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    items = build_ravel(args.n_rows) if args.bench == "ravel" else build_ioi(args.n_rows)
    print(f"[bench] {len(items)} labelled token positions")
    if not items:
        raise SystemExit("[bench] nothing extracted — schema may have changed upstream")

    lm = load_lm(model_name, device_map="auto")
    hook = SplicingHook(lm, layer)
    cap: list[torch.Tensor] = []
    hook.activate(lambda hs: (cap.append(hs.detach()) or hs))

    H, LAB, TID, SURF = [], [], [], []
    with torch.no_grad():
        for s in tqdm(range(0, len(items), args.batch_size), desc="[bench] encoding"):
            batch = items[s:s + args.batch_size]
            enc = tok([b[0] for b in batch], return_offsets_mapping=True,
                      return_tensors="pt", padding=True, truncation=True, max_length=512)
            offs = enc.pop("offset_mapping")
            lm(input_ids=enc["input_ids"].to(dev),
               attention_mask=enc["attention_mask"].to(dev), use_cache=False)
            hs = cap.pop(); cap.clear()
            for i, (_p, (lo, hi), labs, surf) in enumerate(batch):
                n = int(enc["attention_mask"][i].sum())
                t = last_span(offs[i][:n].tolist(), lo, hi)
                if t is None:
                    continue
                H.append(hs[i, t].float().cpu().numpy())
                LAB.append(labs)
                TID.append(int(enc["input_ids"][i, t]))
                SURF.append(surf)
    hook.deactivate()

    H = np.stack(H).astype(np.float32)
    if not np.isfinite(H).all():
        raise SystemExit("[bench] non-finite activations — refusing to write a poisoned cache")
    keys = RAVEL_ATTRS if args.bench == "ravel" else ["role", "name"]
    cols = {k: np.array([d[k] for d in LAB], dtype=object) for k in keys}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out) + ".npz", H=H, token_id=np.array(TID, dtype=np.int64),
             surface=np.array(SURF, dtype=object), **cols)
    for k in keys:
        v = cols[k]
        print(f"[bench]   {k}: {len(set(v.tolist()))} classes, "
              f"most common {sorted(set(v.tolist()))[:4]}")
    print(f"[bench] {H.shape[0]:,} tokens x {H.shape[1]} -> {out}.npz "
          f"({H.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
