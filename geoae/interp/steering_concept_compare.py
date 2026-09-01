"""
Mean-diff steering comparison: base-model h-space vs AE z-space.

For each concept c:
  r_h[c] = mean(h | y=c) - mean(h | y!=c)
  r_z[c] = mean(z | y=c) - mean(z | y!=c)

Intervention convention (matching existing steering outputs):
  x' = x - alpha * r
so positive alpha erases target-class direction toward "other classes".

Data filtering mirrors causal_concept_compare.py:
  use the joint-correct set (correct under both base LM and AE-recon splice).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


from geoae.seeding import seed_everything
from geoae.checkpoint import load_lm
from geoae.evaluate import load_ae_from_checkpoint, sanity_check_splice
from geoae.hooks import SplicingHook
from geoae.interp import neuronlens as nl
from geoae.interp import _shared as shared


def _alpha_tag(alpha: float) -> str:
    return str(float(alpha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="ag_news", choices=["db14", "emotions", "emotions_train", "ag_news"])
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Splice layer; defaults to checkpoint target_layer and must match it.",
    )
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--n_fit", type=int, default=80)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--n_comp", type=int, default=80)
    ap.add_argument("--n_wiki", type=int, default=100)
    ap.add_argument("--n_ppl", type=int, default=40)
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--correct_json", default="dbpedia/joint_correct_predictions_DB_14.json")
    ap.add_argument("--out", default="results_steering_ag_news.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jc_batch_size", type=int, default=16,
                    help="Batch size for the joint-correct build. Peak memory is "
                         "bs*seq*mlp_intermediate; lower it if the LM OOMs.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dataset_cfg = shared.set_dataset(args.dataset)

    if args.correct_json == "dbpedia/joint_correct_predictions_DB_14.json":
        args.correct_json = f"dbpedia/joint_correct_predictions_{args.dataset}.json"
    if args.out == "results_steering_ag_news.json":
        args.out = f"results_steering_{args.dataset}.json"

    if args.smoke:
        args.n_fit, args.n_eval, args.n_comp, args.n_wiki, args.n_ppl = 20, 12, 24, 20, 8
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
        sys.exit(
            f"[steer] ERROR: --layer {args.layer} != AE checkpoint target_layer {ckpt_layer}. "
            f"The AE is layer-specific."
        )

    print(f"[steer] LM={model_name} layer={args.layer} (AE target_layer={ckpt_layer})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    lm = load_lm(model_name, device_map="auto")
    sanity_check_splice(lm, args.layer, tokenizer, device)
    hook = SplicingHook(lm, args.layer)

    print("[steer] Computing wiki ppl slice + z-recon baseline hook …")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wtxt = []
    for ex in wiki:
        wtxt.append(ex["text"][:1500])
        if len(wtxt) >= args.n_wiki:
            break
    h_w = shared.capture_h(lm, tokenizer, wtxt, args.layer, device)
    with torch.no_grad():
        avg_z = ae.encoder(((torch.from_numpy(h_w).to(device) - mean_t) / std_t)).mean(0).cpu().numpy()
    ppl_txt = wtxt[: args.n_ppl]
    z_recon = nl.make_z_gate(
        np.full(len(avg_z), np.inf),
        np.full(len(avg_z), -np.inf),
        avg_z,
        ae,
        mean_t,
        std_t,
        device,
        gate=False,
    )

    print(f"[steer] Loading {args.dataset} test split …")
    ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["dataset_split"]).shuffle(seed=args.seed)
    need = args.n_fit + args.n_eval
    corr_path = Path(args.correct_json)
    correct = shared.build_joint_correct_set(
        ds, dataset_cfg, tokenizer, lm, device, hook, z_recon, corr_path, need,
        args.dataset, model_name=model_name,
        ae_sha=shared.ae_fingerprint(ae), batch_size=args.jc_batch_size, tag="steer",
    )

    pool = {c: [d["text"] for d in correct if d["label"] == c] for c in range(len(shared.CLASSES))}
    fit_txt, fit_lab, eval_txt, eval_lab = [], [], [], []
    for c in range(len(shared.CLASSES)):
        K = len(pool[c])
        if K < 2:
            print(f"[steer] WARNING: class {c} has only {K} joint-correct docs; results will be noisy.")
        n_fit_c = min(args.n_fit, max(1, int(K * 0.6))) if K < need else args.n_fit
        n_eval_c = min(args.n_eval, max(1, K - n_fit_c)) if K < need else args.n_eval
        fc = pool[c][:n_fit_c]
        ec = pool[c][n_fit_c : n_fit_c + n_eval_c]
        fit_txt += fc
        fit_lab += [c] * len(fc)
        eval_txt += ec
        eval_lab += [c] * len(ec)
    fit_lab = np.array(fit_lab)
    eval_lab = np.array(eval_lab)

    print(f"[steer] Fit pass: capturing h,z for {len(fit_txt)} docs …")
    h_fit = shared.capture_h(lm, tokenizer, fit_txt, args.layer, device)
    with torch.no_grad():
        z_fit = ae.encoder(((torch.from_numpy(h_fit).to(device) - mean_t) / std_t)).cpu().numpy()

    print(f"[steer] Clean eval over {len(eval_txt)} docs …")
    base = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
    base_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
    hook.activate(z_recon)
    zbase = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
    zbase_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
    hook.deactivate()

    # Re-filter: keep only docs jointly correct at eval time (batching can flip a few)
    keep = [i for i in range(len(eval_txt))
            if base["correct"][i] and zbase["correct"][i]]
    if len(keep) < len(eval_txt):
        print(f"[steer] Re-filtered eval: {len(eval_txt)} -> {len(keep)} jointly-correct docs")
        eval_txt = [eval_txt[i] for i in keep]
        eval_lab = np.array([eval_lab[i] for i in keep])
        base = {k: [v[i] for i in keep] for k, v in base.items()}
        zbase = {k: [v[i] for i in keep] for k, v in zbase.items()}

    idx_by_class = {c: np.where(eval_lab == c)[0] for c in range(len(shared.CLASSES))}

    def eval_steer(fn, c, comp_idx):
        idx = np.concatenate([idx_by_class[c], comp_idx])
        sub_txt = [eval_txt[i] for i in idx]
        sub_lab = [int(eval_lab[i]) for i in idx]
        hook.activate(fn)
        r = shared.predict(lm, tokenizer, sub_txt, sub_lab, device)
        pp = shared.perplexity(lm, tokenizer, ppl_txt, device)
        hook.deactivate()
        tgt = np.mean([r["correct"][k] for k, lab in enumerate(sub_lab) if lab == c])
        comp = np.mean([r["correct"][k] for k, lab in enumerate(sub_lab) if lab != c])
        tconf = np.mean([r["gold_conf"][k] for k, lab in enumerate(sub_lab) if lab == c])
        return float(tgt), float(comp), float(tconf), float(pp)

    results = {
        "meta": {
            "checkpoint": str(ckpt),
            "layer": args.layer,
            "alphas": [float(a) for a in args.alphas],
            "n_fit": args.n_fit,
            "n_eval": args.n_eval,
            "base_acc": round(float(np.mean(base["correct"])), 4),
            "base_ppl": round(base_ppl, 3),
            "zrecon_acc": round(float(np.mean(zbase["correct"])), 4),
            "zrecon_ppl": round(zbase_ppl, 3),
            "summary_selectivity": {},
        },
        "concepts": {},
    }
    print(
        f"[steer] base acc={results['meta']['base_acc']:.3f} ppl={base_ppl:.1f} | "
        f"z-recon acc={results['meta']['zrecon_acc']:.3f} ppl={zbase_ppl:.1f}"
    )

    r_h, r_z = {}, {}
    for c in concepts:
        c_h = h_fit[fit_lab == c]
        o_h = h_fit[fit_lab != c]
        c_z = z_fit[fit_lab == c]
        o_z = z_fit[fit_lab != c]
        if len(c_h) == 0 or len(o_h) == 0 or len(c_z) == 0 or len(o_z) == 0:
            print(f"[steer] WARNING: concept {c} missing fit samples; skipping.")
            continue
        r_h[c] = c_h.mean(axis=0) - o_h.mean(axis=0)
        r_z[c] = c_z.mean(axis=0) - o_z.mean(axis=0)

    for c in concepts:
        if c not in r_h or c not in r_z:
            continue
        comp_pool = np.where(eval_lab != c)[0]
        comp_idx = rng.choice(comp_pool, min(args.n_comp, len(comp_pool)), replace=False)
        b_tgt = float(np.mean([base["correct"][i] for i in idx_by_class[c]]))
        b_comp = float(np.mean([base["correct"][i] for i in comp_idx]))
        zb_tgt = float(np.mean([zbase["correct"][i] for i in idx_by_class[c]]))
        zb_comp = float(np.mean([zbase["correct"][i] for i in comp_idx]))

        rec = {}
        print(
            f"\n=== concept {c} ({shared.CLASSES[c]})  base tgt {b_tgt:.2f}/comp {b_comp:.2f} | "
            f"z-recon tgt {zb_tgt:.2f}/comp {zb_comp:.2f} ==="
        )
        for alpha in args.alphas:
            atag = _alpha_tag(alpha)

            h_fn = nl.make_h_steer(r_h[c], alpha, device)
            h_tgt, h_comp, h_conf, h_ppl = eval_steer(h_fn, c, comp_idx)
            h_td = h_tgt - b_tgt
            h_cd = h_comp - b_comp
            rec[f"h_a{atag}"] = {
                "alpha": float(alpha),
                "tgt_acc": round(h_tgt, 4),
                "tgt_conf": round(h_conf, 4),
                "comp_acc": round(h_comp, 4),
                "tgt_acc_delta": round(h_td, 4),
                "comp_acc_delta": round(h_cd, 4),
                "selectivity": round(h_td - h_cd, 4),
                "ppl": round(h_ppl, 3),
            }

            z_fn = nl.make_z_steer(r_z[c], alpha, ae, mean_t, std_t, device)
            z_tgt, z_comp, z_conf, z_ppl = eval_steer(z_fn, c, comp_idx)
            z_td = z_tgt - zb_tgt
            z_cd = z_comp - zb_comp
            rec[f"z_a{atag}"] = {
                "alpha": float(alpha),
                "tgt_acc": round(z_tgt, 4),
                "tgt_conf": round(z_conf, 4),
                "comp_acc": round(z_comp, 4),
                "tgt_acc_delta": round(z_td, 4),
                "comp_acc_delta": round(z_cd, 4),
                "selectivity": round(z_td - z_cd, 4),
                "ppl": round(z_ppl, 3),
            }
            print(
                f"  a={alpha:<4} | h sel {rec[f'h_a{atag}']['selectivity']:+.3f} "
                f"(tgt {h_tgt:.2f}, comp {h_comp:.2f}) | "
                f"z sel {rec[f'z_a{atag}']['selectivity']:+.3f} "
                f"(tgt {z_tgt:.2f}, comp {z_comp:.2f})"
            )

        results["concepts"][str(c)] = rec

    for alpha in args.alphas:
        atag = _alpha_tag(alpha)
        h_sels = [
            results["concepts"][str(c)][f"h_a{atag}"]["selectivity"]
            for c in concepts
            if str(c) in results["concepts"] and f"h_a{atag}" in results["concepts"][str(c)]
        ]
        z_sels = [
            results["concepts"][str(c)][f"z_a{atag}"]["selectivity"]
            for c in concepts
            if str(c) in results["concepts"] and f"z_a{atag}" in results["concepts"][str(c)]
        ]
        if h_sels and z_sels:
            results["meta"]["summary_selectivity"][f"a{atag}"] = {
                "h_sel": round(float(np.mean(h_sels)), 4),
                "z_sel": round(float(np.mean(z_sels)), 4),
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[steer] Saved -> {args.out}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os

    os._exit(0)
