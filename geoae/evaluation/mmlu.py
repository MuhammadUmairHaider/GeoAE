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

    scripts/py -u -m geoae.evaluation.mmlu --checkpoint <ckpt> --out eval_out/mmlu_<tag>.json

With BOTH --checkpoint and --sae the two splices are scored in ONE run on the same questions against one shared base
pass (same GPU, same batches), and the output holds both plus their paired comparison (schema: meta.splices.{ae,sae},
meta.paired, preds.{base,ae,sae,gold}). With one of them the output is unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from geoae.checkpoint import load_tokenizer

from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.hooks import SplicingHook, TokenIdTap
import geoae.evaluation.interventions as nl
from geoae.seeding import seed_everything


def mmlu_prompt(ex) -> str:
    p = f"Question: {ex['question']}\n"
    for i, ch in enumerate(ex["choices"]):
        p += f"{chr(65 + i)}. {ch}\n"
    return p + "Answer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--sae", default=None, help="SAE spec instead of --checkpoint (matched:<last.pt> | "
                    "scope:<id|dir>): splices decode(encode(h)) at --layer, same positions as the AE splice")
    ap.add_argument("--model", default="google/gemma-3-4b-pt", help="LM for --sae")
    ap.add_argument("--layer", type=int, default=33, help="splice layer for --sae")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (args.checkpoint or args.sae):
        ap.error("give --checkpoint and/or --sae")
    both = bool(args.checkpoint) and bool(args.sae)
    if args.sae:
        from geoae.baselines.sae_space import load_sae_space
        sae = load_sae_space(args.sae, dev)
        model_name, layer, bypass = args.model, args.layer, sae.needs_tokens
        print(f"[mmlu] {model_name} layer {layer}  SAE {args.sae} ({sae.n_features} features)  token bypass: {bypass}")
    if args.checkpoint:
        ae, mean, std, ck = load_ae_checkpoint(args.checkpoint, dev, allow_token_bias=True)
        ae.eval()
        if both and (ck["config"]["extraction"]["model_name"], ck["config"]["data"]["target_layer"]) != (args.model, args.layer):
            ap.error("--checkpoint and --sae must splice the same model and layer (--model / --layer)")
        model_name, layer = ck["config"]["extraction"]["model_name"], ck["config"]["data"]["target_layer"]
        bypass = ae.has_token_bias
        print(f"[mmlu] {model_name} layer {layer}  token bypass: {bypass}")
    tok = load_tokenizer(model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    lm = load_lm(model_name, device_map="auto")
    hook, tap = SplicingHook(lm, layer), TokenIdTap(lm)
    recons = {}
    if args.checkpoint:
        recons["ae"] = nl.make_z_gate(np.zeros(1), np.zeros(1), np.zeros(1), ae, mean, std, dev,
                                      gate=False, tap=tap)
    if args.sae:
        @torch.no_grad()
        def recon_sae(hs):   # full SAE reconstruction at every position (chunked: 262k-wide codes)
            B, T, D = hs.shape
            tk = tap.ids_for(hs) if sae.needs_tokens else None
            return sae.reconstruct_raw(hs.reshape(B * T, D).float(), tk).reshape(B, T, D).to(hs.device, hs.dtype)
        recons["sae"] = recon_sae

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
    preds = {}
    for name, fn in recons.items():
        hook.activate(fn)
        preds[name] = run()
        hook.deactivate()
    tap.remove()
    if both:
        return write_both(args, base_p, preds, golds, subjects)
    recon_p = next(iter(preds.values()))

    ba = float(np.mean([p == g for p, g in zip(base_p, golds)]))
    ra = float(np.mean([p == g for p, g in zip(recon_p, golds)]))
    subj = {}
    for sub, b, r, g in zip(subjects, base_p, recon_p, golds):
        d = subj.setdefault(sub, {"n": 0, "b": 0, "r": 0})
        d["n"] += 1; d["b"] += (b == g); d["r"] += (r == g)
    flips = {"base_only": int(sum(b == g and r != g for b, r, g in zip(base_p, recon_p, golds))),
             "recon_only": int(sum(r == g and b != g for b, r, g in zip(base_p, recon_p, golds)))}
    out = {"meta": {"n": len(golds), "base_acc": round(ba, 4), "recon_acc": round(ra, 4),
                    "delta": round(ra - ba, 4), "checkpoint": str(args.checkpoint or args.sae),
                    "token_bias": bool(bypass), "flips": flips},
           "per_subject": {s: {"n": v["n"], "base_acc": round(v["b"] / v["n"], 4),
                               "recon_acc": round(v["r"] / v["n"], 4)} for s, v in subj.items()},
           "preds": {"base": base_p, "recon": recon_p, "gold": golds}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[mmlu] base={ba:.4f}  AE-recon={ra:.4f}  delta={ra - ba:+.4f}  "
          f"(base-only correct {flips['base_only']}, recon-only {flips['recon_only']}) -> {args.out}")


def write_both(args, base_p, preds, golds, subjects):
    """One shared base pass, AE and SAE splices on the same questions: accuracies, flips and the paired AE - SAE test."""
    from scipy.stats import binomtest
    g = np.array(golds)
    ok = {k: np.array(v) == g for k, v in {"base": base_p, **preds}.items()}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), (10000, len(g)))

    def ci(d):
        return [float(x) for x in np.quantile(d[idx].mean(1), [0.025, 0.975])]

    splices = {}
    for k in preds:
        d = ok[k].astype(float) - ok["base"].astype(float)
        splices[k] = {"recon_acc": round(float(ok[k].mean()), 4), "delta": round(float(d.mean()), 4), "delta_ci": ci(d),
                      "flips": {"base_only": int((ok["base"] & ~ok[k]).sum()), "recon_only": int((~ok["base"] & ok[k]).sum())},
                      "answers_changed": int((np.array(preds[k]) != np.array(base_p)).sum())}
    d = ok["ae"].astype(float) - ok["sae"].astype(float)
    a_only, s_only = int((ok["ae"] & ~ok["sae"]).sum()), int((~ok["ae"] & ok["sae"]).sum())
    paired = {"ae_minus_sae": round(float(d.mean()), 4), "ci": ci(d), "ae_only_correct": a_only, "sae_only_correct": s_only,
              "mcnemar_exact_p": float(binomtest(a_only, a_only + s_only, 0.5).pvalue) if a_only + s_only else 1.0}
    subj = {}
    for i, sub in enumerate(subjects):
        v = subj.setdefault(sub, {"n": 0, "base": 0, "ae": 0, "sae": 0})
        v["n"] += 1
        for k in ("base", "ae", "sae"):
            v[k] += int(ok[k][i])
    out = {"meta": {"n": len(g), "base_acc": round(float(ok["base"].mean()), 4), "splices": splices, "paired": paired,
                    "checkpoint": str(args.checkpoint), "sae": str(args.sae), "seed": args.seed},
           "per_subject": {s: {"n": v["n"], **{f"{k}_acc": round(v[k] / v["n"], 4) for k in ("base", "ae", "sae")}}
                           for s, v in subj.items()},
           "preds": {"base": base_p, **preds, "gold": golds}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[mmlu] base={out['meta']['base_acc']:.4f}  " + "  ".join(
        f"{k}={v['recon_acc']:.4f} ({v['delta']:+.4f}, flips {v['flips']['base_only']}/{v['flips']['recon_only']})"
        for k, v in splices.items()) + f"  AE-SAE={paired['ae_minus_sae']:+.4f} p={paired['mcnemar_exact_p']:.3g} -> {args.out}")


if __name__ == "__main__":
    main()
