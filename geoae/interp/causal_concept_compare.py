"""
Causal concept-control comparison: NeuronLens on BASE-model neurons (h_j) vs
AE neurons (z_j), Llama-3.2-3B @ layer 27, DBpedia-14.

Method (faithful to github.com/MuhammadUmairHaider/NeuronLens) lives in
geoae/neuronlens.py. This harness:
  1. builds the base-correct DBpedia-14 set (keep docs the base model classifies
     correctly under the 4-shot prompt), cached to JSON.
  2. fits saliency + per-concept Gaussian ranges in h-space and z-space on a
     train slice (last-token layer-27 residual of the prompt; z = AE.encode).
  3. for each concept c × substrate {h, z} × mode {range, full} applies the gate
     via evaluate.SplicingHook and measures target vs complement classification
     Acc/Conf + wiki perplexity. z also reports the no-gate AE-recon baseline.
  4. computes range separability (overlap, purity) for h vs z.

Speed: left-padded KV-cached generation; each gate is evaluated only on its
target docs + a fixed complement sample (not the whole eval set).

Usage:
    python -m geoae.interp.causal_concept_compare \
      --checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/best_val.pt \
      --layer 27 --percent 0.3 --tao 2.0 --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


from geoae.seeding import seed_everything
from geoae.checkpoint import load_lm
from geoae.evaluate import load_ae_from_checkpoint, sanity_check_splice
from geoae.hooks import SplicingHook
from geoae.interp import neuronlens as nl
from geoae.interp import _shared as shared


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="db14", choices=["db14", "emotions", "emotions_train", "ag_news"], help="Dataset to evaluate on")
    ap.add_argument("--layer", type=int, default=None,
                    help="Splice layer; defaults to the AE checkpoint's target_layer. "
                         "Must equal the AE's training layer (the AE is layer-specific).")
    ap.add_argument("--percent", type=float, default=0.3)
    ap.add_argument("--q", type=int, default=None, help="top-q salient dims (overrides --percent)")
    ap.add_argument("--ae_percent", type=float, default=None, help="top-p% salient dims for AE (defaults to --percent)")
    ap.add_argument("--ae_q", type=int, default=None, help="top-q salient dims for AE (overrides --ae_percent, defaults to --q)")
    ap.add_argument("--tao", type=float, default=2.0)
    ap.add_argument("--ae_tao", type=float, default=None,
                    help="tao for AE z-space ranges (defaults to --tao)")
    ap.add_argument("--n_build", type=int, default=600)
    ap.add_argument("--n_fit", type=int, default=80)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--n_comp", type=int, default=80)
    ap.add_argument("--n_wiki", type=int, default=100)
    ap.add_argument("--n_ppl", type=int, default=40)
    ap.add_argument("--accuracy_compare", type=int, default=0,
                    help="If >0: compare base vs AE-recon classification accuracy on this many "
                         "random docs/class from the FULL (unfiltered) test set, then exit")
    ap.add_argument("--mmlu", type=int, default=0,
                    help="If >0: compare base vs AE-recon accuracy on this many MMLU questions "
                         "(general capability), then exit")
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--correct_json", default="dbpedia/joint_correct_predictions_DB_14.json")
    ap.add_argument("--out", default="results_causal_concept_compare.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jc_batch_size", type=int, default=16,
                    help="Batch size for the joint-correct build. Peak memory is "
                         "bs*seq*mlp_intermediate; lower it if the LM OOMs.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replace_wiki_avg", action="store_true",
                    help="Replace gated dims with wiki-mean activation (reference MaskLayer uses 0)")
    args = ap.parse_args()

    dataset_cfg = shared.set_dataset(args.dataset)

    if args.correct_json == "dbpedia/joint_correct_predictions_DB_14.json":
        args.correct_json = f"dbpedia/joint_correct_predictions_{args.dataset}.json"
    if args.out == "results_causal_concept_compare.json":
        args.out = f"results_ccc_{args.dataset}.json"

    if args.smoke:
        args.n_build, args.n_fit, args.n_eval, args.n_comp, args.n_wiki, args.n_ppl = 40, 20, 12, 24, 20, 8
        if args.concepts == "all":
            if args.dataset == "db14":
                args.concepts = "9,11"
            elif args.dataset in ("emotions", "emotions_train"):
                args.concepts = "1,3"
            elif args.dataset == "ag_news":
                args.concepts = "0,2"

    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    concepts = list(range(len(shared.CLASSES))) if args.concepts == "all" else [int(x) for x in args.concepts.split(",")]

    ckpt = Path(args.checkpoint)
    ae, norm, cfg = load_ae_from_checkpoint(ckpt, device)
    mean_t, std_t = norm["mean"], norm["std"]
    model_name = cfg["extraction"]["model_name"]
    ckpt_layer = cfg["data"]["target_layer"]
    if args.layer is None:
        args.layer = ckpt_layer
    elif args.layer != ckpt_layer:
        sys.exit(f"[ccc] ERROR: --layer {args.layer} != AE checkpoint target_layer {ckpt_layer}. "
                 f"The AE (encoder/decoder + norm stats) is layer-specific; to compare layer "
                 f"{args.layer} you need an AE trained at layer {args.layer}.")
    print(f"[ccc] LM={model_name} layer={args.layer} (AE target_layer={ckpt_layer}) K={cfg['model']['n_clusters']}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    lm = load_lm(model_name, device_map="auto")
    sanity_check_splice(lm, args.layer, tokenizer, device)
    hook = SplicingHook(lm, args.layer)

    # ---- avg replacement from wiki ----
    print("[ccc] Computing avg replacement from wiki …")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wtxt = []
    for ex in wiki:
        wtxt.append(ex["text"][:1500])
        if len(wtxt) >= args.n_wiki:
            break
    h_w = shared.capture_h(lm, tokenizer, wtxt, args.layer, device)
    avg_h = h_w.mean(0)
    with torch.no_grad():
        avg_z = ae.encoder(((torch.from_numpy(h_w).to(device) - mean_t) / std_t)).mean(0).cpu().numpy()
    ppl_txt = wtxt[:args.n_ppl]

    # Reference MaskLayer is init'd with replacement_values=0; mask_range_llma never
    # overwrites it (wiki avg is computed but unused).
    if args.replace_wiki_avg:
        rep_h, rep_z = avg_h, avg_z
    else:
        rep_h = np.zeros_like(avg_h)
        rep_z = np.zeros_like(avg_z)

    # reconstruction splice baseline (reconstructs but does not gate)
    z_recon = nl.make_z_gate(np.full(len(avg_z), np.inf), np.full(len(avg_z), -np.inf),
                             avg_z, ae, mean_t, std_t, device, gate=False)

    # ---- optional: MMLU general-capability — base vs AE-recon ----
    if args.mmlu > 0:
        print(f"[ccc] MMLU: base vs AE-recon on {args.mmlu} questions …")
        mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=args.seed)
        mm = mm.select(range(min(args.mmlu, len(mm))))

        def mmlu_prompt(ex):
            p = f"Question: {ex['question']}\n"
            for i, ch in enumerate(ex["choices"]):
                p += f"{chr(65 + i)}. {ch}\n"
            return p + "Answer:"

        prompts = [mmlu_prompt(ex) for ex in mm]
        golds = [int(ex["answer"]) for ex in mm]
        subjects = [ex["subject"] for ex in mm]

        @torch.no_grad()
        def run_mmlu(bs=16):
            preds = []
            for s in tqdm(range(0, len(prompts), bs), desc="mmlu", leave=False):
                bp = prompts[s:s + bs]
                enc = tokenizer(bp, padding=True, truncation=True, max_length=1024,
                                return_tensors="pt").to(device)
                gen = lm.generate(**enc, max_new_tokens=1, do_sample=False, num_beams=1,
                                  pad_token_id=tokenizer.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                for i in range(len(bp)):
                    t = tokenizer.decode(new[i], skip_special_tokens=True).strip().upper()
                    preds.append(ord(t[0]) - 65 if t and t[0] in "ABCD" else -1)
            return preds

        base_p = run_mmlu()
        hook.activate(z_recon); recon_p = run_mmlu(); hook.deactivate()
        ba = float(np.mean([p == g for p, g in zip(base_p, golds)]))
        ra = float(np.mean([p == g for p, g in zip(recon_p, golds)]))
        # per-subject deltas (sorted by biggest movers)
        subj = {}
        for sub, bp_, rp_, g in zip(subjects, base_p, recon_p, golds):
            d = subj.setdefault(sub, {"n": 0, "b": 0, "r": 0})
            d["n"] += 1; d["b"] += (bp_ == g); d["r"] += (rp_ == g)
        movers = sorted(((s, v["n"], v["b"]/v["n"], v["r"]/v["n"], (v["r"]-v["b"])/v["n"])
                         for s, v in subj.items() if v["n"] >= 5), key=lambda x: x[4])
        print(f"\n  MMLU base={ba:.3f}  AE-recon={ra:.3f}  delta={ra - ba:+.3f}  (n={len(golds)}, random={1/4:.2f})")
        print("  biggest per-subject movers (n>=5):")
        for s, n, b, r, d in movers[:5] + movers[-5:]:
            print(f"    {s:<34} n={n:>3} base={b:.2f} recon={r:.2f} d={d:+.2f}")
        outp = "results_ccc_mmlu.json"   # MMLU is dataset-independent
        json.dump({"meta": {"n": len(golds), "base_acc": round(ba, 4), "recon_acc": round(ra, 4),
                            "delta": round(ra - ba, 4), "checkpoint": str(ckpt)},
                   "per_subject": {s: {"n": v["n"], "base_acc": round(v["b"]/v["n"], 4),
                                       "recon_acc": round(v["r"]/v["n"], 4)} for s, v in subj.items()}},
                  open(outp, "w"), indent=2)
        print(f"\n[ccc] MMLU base={ba:.3f}  AE-recon={ra:.3f}  delta={ra - ba:+.3f}  -> {outp}")
        return

    # ---- Dataset + base+recon joint-correct set ----
    print(f"[ccc] Loading {args.dataset} test split …")
    ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["dataset_split"]).shuffle(seed=args.seed)

    # ---- optional: does the AE reconstruction change base accuracy? (full set) ----
    if args.accuracy_compare > 0:
        n = args.accuracy_compare
        print(f"[ccc] Accuracy comparison: base vs AE-recon on {n} random docs/class (full, unfiltered) …")
        bucket = {c: [] for c in range(len(shared.CLASSES))}
        for ex in ds:
            c = int(ex["label"])
            if len(bucket[c]) < n:
                bucket[c].append(ex[dataset_cfg["text_column"]].strip())
            if all(len(v) >= n for v in bucket.values()):
                break
        txt, lab = [], []
        for c in range(len(shared.CLASSES)):
            txt += bucket[c]; lab += [c] * len(bucket[c])
        lab = np.array(lab)
        base = shared.predict(lm, tokenizer, txt, lab, device)
        hook.activate(z_recon)
        recon = shared.predict(lm, tokenizer, txt, lab, device)
        hook.deactivate()
        print(f"\n  {'class':<24}{'n':>5}{'base':>8}{'AE-recon':>10}{'delta':>8}")
        rows = []
        for c in range(len(shared.CLASSES)):
            idx = np.where(lab == c)[0]
            ba = float(np.mean([base["correct"][i] for i in idx]))
            ra = float(np.mean([recon["correct"][i] for i in idx]))
            rows.append({"class": shared.CLASSES[c], "n": int(len(idx)), "base_acc": round(ba, 4),
                         "recon_acc": round(ra, 4), "delta": round(ra - ba, 4)})
            print(f"  {shared.CLASSES[c]:<24}{len(idx):>5}{ba:>8.3f}{ra:>10.3f}{ra - ba:>+8.3f}")
        ob, orr = float(np.mean(base["correct"])), float(np.mean(recon["correct"]))
        print(f"  {'OVERALL':<24}{len(lab):>5}{ob:>8.3f}{orr:>10.3f}{orr - ob:>+8.3f}")
        outp = args.out.replace(".json", "_accuracy.json")
        json.dump({"meta": {"dataset": args.dataset, "n_per_class": n, "checkpoint": str(ckpt),
                            "base_acc": round(ob, 4), "recon_acc": round(orr, 4), "delta": round(orr - ob, 4)},
                   "per_class": rows}, open(outp, "w"), indent=2)
        print(f"\n[ccc] base={ob:.3f}  AE-recon={orr:.3f}  delta={orr - ob:+.3f}  -> {outp}")
        return

    corr_path = Path(args.correct_json)
    need = args.n_fit + args.n_eval
    correct = shared.build_joint_correct_set(
        ds, dataset_cfg, tokenizer, lm, device, hook, z_recon, corr_path, need,
        args.dataset, model_name=model_name,
        ae_sha=shared.ae_fingerprint(ae), batch_size=args.jc_batch_size, tag="ccc",
    )

    pool = {c: [d["text"] for d in correct if d["label"] == c] for c in range(len(shared.CLASSES))}
    fit_txt, fit_lab, eval_txt, eval_lab = [], [], [], []
    for c in range(len(shared.CLASSES)):
        K = len(pool[c])
        if K < 2:
            print(f"[ccc] WARNING: Class {c} has only {K} joint-correct docs! Ranges will be extremely noisy.")
        n_fit_c = min(args.n_fit, max(1, int(K * 0.6))) if K < need else args.n_fit
        n_eval_c = min(args.n_eval, max(1, K - n_fit_c)) if K < need else args.n_eval

        fc = pool[c][:n_fit_c]
        ec = pool[c][n_fit_c:n_fit_c + n_eval_c]

        fit_txt += fc
        fit_lab += [c] * len(fc)
        eval_txt += ec
        eval_lab += [c] * len(ec)

    fit_lab = np.array(fit_lab); eval_lab = np.array(eval_lab)

    # ---- fit pass: h, z (h at post-layer pre-norm; correct-only for range fit) ----
    print(f"[ccc] Fit pass: capturing h for {len(fit_txt)} docs …")
    h_fit = shared.capture_h(lm, tokenizer, fit_txt, args.layer, device)
    r_fit = shared.predict(lm, tokenizer, fit_txt, fit_lab, device)
    fit_correct = np.array(r_fit["correct"])
    print(f"[ccc] Fit pass: {fit_correct.sum()}/{len(fit_correct)} still shared.predict correctly")
    with torch.no_grad():
        z_fit = ae.encoder(((torch.from_numpy(h_fit).to(device) - mean_t) / std_t)).cpu().numpy()
    sal_h, sal_z = nl.saliency(h_fit, fit_lab, len(shared.CLASSES)), nl.saliency(z_fit, fit_lab, len(shared.CLASSES))

    # ---- clean + z-recon baselines (whole eval set, once) ----
    print(f"[ccc] Clean eval over {len(eval_txt)} docs …")
    base = shared.predict(lm, tokenizer, eval_txt, eval_lab, device); base_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
    hook.activate(z_recon)
    zbase = shared.predict(lm, tokenizer, eval_txt, eval_lab, device); zbase_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
    hook.deactivate()

    # per-concept eval index pools (target + fixed complement sample)
    idx_by_class = {c: np.where(eval_lab == c)[0] for c in range(len(shared.CLASSES))}

    def eval_gate(fn, c, comp_idx):
        idx = np.concatenate([idx_by_class[c], comp_idx])
        sub_txt = [eval_txt[i] for i in idx]; sub_lab = [int(eval_lab[i]) for i in idx]
        hook.activate(fn)
        r = shared.predict(lm, tokenizer, sub_txt, sub_lab, device)
        pp = shared.perplexity(lm, tokenizer, ppl_txt, device)
        hook.deactivate()
        tgt = np.mean([r["correct"][k] for k, l in enumerate(sub_lab) if l == c])
        comp = np.mean([r["correct"][k] for k, l in enumerate(sub_lab) if l != c])
        tconf = np.mean([r["gold_conf"][k] for k, l in enumerate(sub_lab) if l == c])
        return float(tgt), float(comp), float(tconf), pp

    results = {"meta": {"checkpoint": str(ckpt), "layer": args.layer, "tao": args.tao,
                        "percent": args.percent, "q": args.q,
                        "ae_percent": args.ae_percent if args.ae_percent is not None else args.percent,
                        "ae_q": args.ae_q if args.ae_q is not None else args.q,
                        "ae_tao": args.ae_tao if args.ae_tao is not None else args.tao,
                        "base_acc": round(float(np.mean(base["correct"])), 4), "base_ppl": round(base_ppl, 3),
                        "zrecon_acc": round(float(np.mean(zbase["correct"])), 4), "zrecon_ppl": round(zbase_ppl, 3)},
               "concepts": {}}
    print(f"[ccc] base acc={results['meta']['base_acc']:.3f} ppl={base_ppl:.1f} | "
          f"z-recon acc={results['meta']['zrecon_acc']:.3f} ppl={zbase_ppl:.1f}")

    ranges_h, ranges_z = {}, {}
    mus_h = np.zeros((len(shared.CLASSES), h_fit.shape[1])); sig_h = np.zeros_like(mus_h)
    mus_z = np.zeros((len(shared.CLASSES), z_fit.shape[1])); sig_z = np.zeros_like(mus_z)

    for c in concepts:
        comp_idx = rng.choice(np.where(eval_lab != c)[0], min(args.n_comp, int((eval_lab != c).sum())), replace=False)
        b_tgt = float(np.mean([base["correct"][i] for i in idx_by_class[c]]))
        b_comp = float(np.mean([base["correct"][i] for i in comp_idx]))
        zb_tgt = float(np.mean([zbase["correct"][i] for i in idx_by_class[c]]))
        zb_comp = float(np.mean([zbase["correct"][i] for i in comp_idx]))
        rec = {}
        for tag, acts, sal, rep, (bt, bc) in [
            ("h", h_fit, sal_h, rep_h, (b_tgt, b_comp)),
            ("z", z_fit, sal_z, rep_z, (zb_tgt, zb_comp)),
        ]:
            if tag == "h":
                t_kw = dict(q=args.q) if args.q is not None else dict(p=args.percent)
                # Reference: saliency + ranges from correct-only fc_vals for concept c
                acts_c = acts[(fit_lab == c) & fit_correct]
                if len(acts_c) == 0:
                    print(f"[ccc] WARNING: no correct fit activations for class {c}; skipping h gates")
                    continue
                salient = nl.select_top(np.abs(acts_c).mean(axis=0), **t_kw)
                lo, hi, mu, sigma = nl.fit_ranges(acts_c, salient, args.tao)
                lo_f, hi_f, _, _ = nl.fit_ranges(acts_c, salient, np.inf)
                ranges_h[c] = (lo, hi, salient); mus_h[c], sig_h[c] = mu, sigma
                rfn = nl.make_h_gate(lo, hi, rep, device); ffn = nl.make_h_gate(lo_f, hi_f, rep, device)
            else:
                ae_q = args.ae_q if args.ae_q is not None else args.q
                ae_p = args.ae_percent if args.ae_percent is not None else args.percent
                ae_tao = args.ae_tao if args.ae_tao is not None else args.tao
                t_kw = dict(q=ae_q) if ae_q is not None else dict(p=ae_p)
                acts_c = acts[(fit_lab == c) & fit_correct]
                if len(acts_c) == 0:
                    print(f"[ccc] WARNING: no correct fit activations for class {c}; skipping z gates")
                    continue
                salient = nl.select_top(np.abs(acts_c).mean(axis=0), **t_kw)
                lo, hi, mu, sigma = nl.fit_ranges(acts_c, salient, ae_tao)
                lo_f, hi_f, _, _ = nl.fit_ranges(acts_c, salient, np.inf)
                ranges_z[c] = (lo, hi, salient); mus_z[c], sig_z[c] = mu, sigma
                rfn = nl.make_z_gate(lo, hi, rep, ae, mean_t, std_t, device)
                ffn = nl.make_z_gate(lo_f, hi_f, rep, ae, mean_t, std_t, device)
            for mode, fn in [("range", rfn), ("full", ffn)]:
                t_a, c_a, t_cf, pp = eval_gate(fn, c, comp_idx)
                rec[f"{tag}_{mode}"] = {
                    "tgt_acc": round(t_a, 4), "tgt_conf": round(t_cf, 4), "comp_acc": round(c_a, 4),
                    "tgt_acc_drop": round(bt - t_a, 4), "comp_acc_drop": round(bc - c_a, 4),
                    "selectivity": round((bt - t_a) - (bc - c_a), 4), "ppl": round(pp, 3),
                }
        results["concepts"][str(c)] = rec
        print(f"\n=== concept {c} ({shared.CLASSES[c]})  base tgt {b_tgt:.2f}/comp {b_comp:.2f} | z-recon tgt {zb_tgt:.2f}/comp {zb_comp:.2f} ===")
        for k, v in rec.items():
            print(f"  {k:9s} tgtAcc {v['tgt_acc']:.2f} (drop {v['tgt_acc_drop']:+.2f}) "
                  f"compAcc {v['comp_acc']:.2f} (drop {v['comp_acc_drop']:+.2f}) "
                  f"sel {v['selectivity']:+.2f}  ppl {v['ppl']:.1f}")

    cidx = np.array(concepts)
    union_h = np.any([ranges_h[c][2] for c in concepts], axis=0)
    union_z = np.any([ranges_z[c][2] for c in concepts], axis=0)
    sep = {
        "range_overlap_h": nl.range_overlap(mus_h[cidx], sig_h[cidx], union_h),
        "range_overlap_z": nl.range_overlap(mus_z[cidx], sig_z[cidx], union_z),
        "range_purity_h": nl.range_purity(h_fit, fit_lab, {c: ranges_h[c] for c in concepts}),
        "range_purity_z": nl.range_purity(z_fit, fit_lab, {c: ranges_z[c] for c in concepts}),
    }
    results["separability"] = {k: (round(v, 4) if v == v else None) for k, v in sep.items()}
    print("\n=== separability (lower overlap = more separable; higher purity = cleaner) ===")
    print(f"  overlap  h={sep['range_overlap_h']:.3f}  z={sep['range_overlap_z']:.3f}")
    print(f"  purity   h={sep['range_purity_h']:.3f}  z={sep['range_purity_z']:.3f}")

    # macro selectivity summary
    for key in ["h_range", "z_range", "h_full", "z_full"]:
        sels = [results["concepts"][str(c)][key]["selectivity"] for c in concepts]
        results["meta"][f"mean_sel_{key}"] = round(float(np.mean(sels)), 4)
    print("\n=== mean selectivity over concepts ===")
    print("  " + "  ".join(f"{k}={results['meta'][f'mean_sel_{k}']:+.3f}" for k in ["h_range", "z_range", "h_full", "z_full"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[ccc] Saved -> {args.out}")


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(0)
