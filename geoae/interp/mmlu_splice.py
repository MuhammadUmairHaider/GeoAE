"""
MMLU under the AE reconstruction splice: base LM vs LM with layer L replaced by
decode(encode(h)) at every position.

The same protocol as `causal_concept_compare --mmlu` (cais/mmlu "all" test,
shuffled with --seed, first --n questions, "Question/choices/Answer:" prompt,
left padding, greedy single-token answer, batch 16) and the same output schema,
so results sit next to eval_out/mmlu_*.json. Unlike that tool it supports
token-bypass AEs: the splice gets each position's current token id from a
TokenIdTap on the input embedding. For a plain AE it reproduces the existing
numbers (check: d6144_new recon_acc 0.5390 at n=2000, seed 42).

    .venv/bin/python -u -m geoae.interp.mmlu_splice --checkpoint <ckpt> --out eval_out/mmlu_<tag>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.hooks import SplicingHook, TokenIdTap
from geoae.interp import neuronlens as nl
from geoae.seeding import seed_everything


def mmlu_prompt(ex) -> str:
    p = f"Question: {ex['question']}\n"
    for i, ch in enumerate(ex["choices"]):
        p += f"{chr(65 + i)}. {ch}\n"
    return p + "Answer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, mean, std, ck = load_ae_checkpoint(args.checkpoint, dev, allow_token_bias=True)
    ae.eval()
    model_name, layer = ck["config"]["extraction"]["model_name"], ck["config"]["data"]["target_layer"]
    print(f"[mmlu] {model_name} layer {layer}  token bypass: {ae.has_token_bias}")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    lm = load_lm(model_name, device_map="auto")
    hook, tap = SplicingHook(lm, layer), TokenIdTap(lm)
    recon = nl.make_z_gate(np.zeros(1), np.zeros(1), np.zeros(1), ae, mean, std, dev,
                           gate=False, tap=tap)

    mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=args.seed)
    mm = mm.select(range(min(args.n, len(mm))))
    prompts = [mmlu_prompt(ex) for ex in mm]
    golds = [int(ex["answer"]) for ex in mm]
    subjects = [ex["subject"] for ex in mm]

    @torch.no_grad()
    def run():
        preds = []
        for s in tqdm(range(0, len(prompts), args.bs), desc="mmlu", leave=False):
            enc = tok(prompts[s:s + args.bs], padding=True, truncation=True, max_length=1024,
                      return_tensors="pt").to(dev)
            gen = lm.generate(**enc, max_new_tokens=1, do_sample=False, num_beams=1,
                              pad_token_id=tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                t = tok.decode(row, skip_special_tokens=True).strip().upper()
                preds.append(ord(t[0]) - 65 if t and t[0] in "ABCD" else -1)
        return preds

    base_p = run()
    hook.activate(recon)
    recon_p = run()
    hook.deactivate()
    tap.remove()

    ba = float(np.mean([p == g for p, g in zip(base_p, golds)]))
    ra = float(np.mean([p == g for p, g in zip(recon_p, golds)]))
    subj = {}
    for sub, b, r, g in zip(subjects, base_p, recon_p, golds):
        d = subj.setdefault(sub, {"n": 0, "b": 0, "r": 0})
        d["n"] += 1; d["b"] += (b == g); d["r"] += (r == g)
    flips = {"base_only": int(sum(b == g and r != g for b, r, g in zip(base_p, recon_p, golds))),
             "recon_only": int(sum(r == g and b != g for b, r, g in zip(base_p, recon_p, golds)))}
    out = {"meta": {"n": len(golds), "base_acc": round(ba, 4), "recon_acc": round(ra, 4),
                    "delta": round(ra - ba, 4), "checkpoint": str(args.checkpoint),
                    "token_bias": bool(ae.has_token_bias), "flips": flips},
           "per_subject": {s: {"n": v["n"], "base_acc": round(v["b"] / v["n"], 4),
                               "recon_acc": round(v["r"] / v["n"], 4)} for s, v in subj.items()},
           "preds": {"base": base_p, "recon": recon_p, "gold": golds}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[mmlu] base={ba:.4f}  AE-recon={ra:.4f}  delta={ra - ba:+.4f}  "
          f"(base-only correct {flips['base_only']}, recon-only {flips['recon_only']}) -> {args.out}")


if __name__ == "__main__":
    main()
