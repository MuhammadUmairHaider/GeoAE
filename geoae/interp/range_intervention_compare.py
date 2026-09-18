"""
Range-based concept REMOVAL and STEERING: base-model neurons (h) vs AE neurons (z).

NeuronLens (arXiv 2502.06809) characterises concept c on neuron j by a Gaussian
activation range AR(c,j) = [mu_c - tao*sd_c, mu_c + tao*sd_c]. This harness asks
whether interventions built on those ranges work better in the base residual
stream or in the AE latent, with every intervention applied to BOTH substrates
through one shared operator (neuronlens.make_h_edit / make_z_edit), on the same
fit docs, the same eval docs and the same complement sample per concept — so
every h-vs-z number is a paired comparison.

Per concept c and substrate s in {h, z}:

  REMOVAL  (NeuronLens gate; salient dims only)
    rm_range_<rep>   a_j in AR(c,j)          -> rep_j
    rm_full_<rep>    any value on salient j  -> rep_j        (neuron-level ablation)
    rep: zero (the reference MaskLayer), comp (mean over OTHER concepts' fit docs),
         wiki (mean last-token activation on wikipedia)

  STEERING (r = mu_c - mu_rest, the mean difference, so alpha is in class-gap units
            in both spaces — same convention as steering_concept_compare)
    st_global_a      all dims, all values: a - alpha*r       (== steering_concept_compare)
    st_salient_a     salient dims, all values
    st_range_a       salient dims, values inside AR(c,j) only
    st_transport_a   salient dims, in-range values moved along the Gaussian transport
                     AR(c,j) -> AR(rest,j):  a + alpha*(T(a) - a)
    salient vs global isolates DIM SELECTION; range vs salient isolates RANGE GATING;
    transport vs range isolates the std-based operator from the additive one.

WHY zero is not the only replacement. On L27 GELU latents 56% of dims have
|mu| > sd (30% in h): zero is a large perturbation of a shifted code, not a
neutral one. On AG News (Aug 7) zero-gating z drove target AND complement
accuracy to 0.0 for every concept — selectivity ~0 because everything broke,
not because ranges failed. `comp` is the on-distribution neutral value.

Metrics, all read against the substrate's OWN unedited baseline (base LM for h,
AE-recon splice for z):
    tgt_drop = base_tgt - tgt     comp_drop = base_comp - comp
    selectivity = tgt_drop - comp_drop           HIGHER = better
    ppl_rise    = ppl - own-baseline ppl         lower = less collateral damage
NOTE: steering_concept_compare reports tgt_delta - comp_delta, the NEGATION of this.

Also logged per concept: how often the range gate fires on target vs other-concept
eval docs (last token, salient dims). If those two rates are close, a range gate
cannot be selective whatever the substrate.

Usage (smoke, then full):
    python -m geoae.interp.range_intervention_compare \
      --checkpoint checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt \
      --dataset db14 --correct_json dbpedia/joint_correct_db14_l27_dpc.json --smoke
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


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="db14", choices=["db14", "emotions", "emotions_train", "ag_news", "biasbios"])
    ap.add_argument("--layer", type=int, default=None,
                    help="Splice layer; defaults to the checkpoint's target_layer and must match it.")
    ap.add_argument("--percent", type=float, default=0.3, help="top-p fraction of dims called salient (h)")
    ap.add_argument("--q", type=int, default=None, help="top-q salient dims for h (overrides --percent)")
    ap.add_argument("--ae_percent", type=float, default=None, help="top-p for z (defaults to --percent)")
    ap.add_argument("--ae_q", type=int, default=None, help="top-q for z (defaults to --q)")
    ap.add_argument("--tao", type=float, default=2.0, help="range half-width in sd units")
    ap.add_argument("--saliency", default="abs", choices=["abs", "dprime"],
                    help="abs: mean |a| over concept docs (NeuronLens reference). "
                         "dprime: |mu_c - mu_rest| / pooled sd (offset-invariant).")
    ap.add_argument("--removal_modes", default="range,full")
    ap.add_argument("--replace", default="zero,comp", help="comma list of zero, comp, wiki")
    ap.add_argument("--steer_modes", default="global,salient,range,transport")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--n_fit", type=int, default=80)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--n_comp", type=int, default=80)
    ap.add_argument("--n_wiki", type=int, default=100)
    ap.add_argument("--n_ppl", type=int, default=40)
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--correct_json", default="dbpedia/joint_correct_predictions_DB_14.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jc_batch_size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    removal_modes, replaces, steer_modes = _csv(args.removal_modes), _csv(args.replace), _csv(args.steer_modes)
    for m in removal_modes:
        assert m in ("range", "full"), f"unknown removal mode {m!r}"
    for m in replaces:
        assert m in ("zero", "comp", "wiki"), f"unknown replacement {m!r}"
    for m in steer_modes:
        assert m in ("global", "salient", "range", "transport"), f"unknown steer mode {m!r}"

    dataset_cfg = shared.set_dataset(args.dataset)
    if args.correct_json == "dbpedia/joint_correct_predictions_DB_14.json":
        args.correct_json = f"dbpedia/joint_correct_predictions_{args.dataset}.json"
    if args.smoke:
        args.n_fit, args.n_eval, args.n_comp, args.n_wiki, args.n_ppl = 20, 12, 24, 20, 8
        args.alphas = [1.0]
        if args.concepts == "all":
            args.concepts = {"db14": "9,11", "ag_news": "0,2"}.get(args.dataset, "1,3")
    if args.out is None:
        run = Path(args.checkpoint).parent.name.replace("k2000_bnh_b32k_lam1_d6144", "b32k") or "ae"
        tag = f"_{run}_{args.saliency}" + ("_smoke" if args.smoke else "")
        args.out = f"results/range_intervention_{args.dataset}{tag}.json"

    seed_everything(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    concepts = (list(range(len(shared.CLASSES))) if args.concepts == "all"
                else [int(x) for x in args.concepts.split(",")])

    ckpt = Path(args.checkpoint)
    ae, norm, cfg = load_ae_from_checkpoint(ckpt, device)
    mean_t, std_t = norm["mean"], norm["std"]
    model_name = cfg["extraction"]["model_name"]
    ckpt_layer = cfg["data"]["target_layer"]
    if args.layer is None:
        args.layer = ckpt_layer
    elif args.layer != ckpt_layer:
        sys.exit(f"[range] ERROR: --layer {args.layer} != AE checkpoint target_layer {ckpt_layer}. "
                 f"The AE is layer-specific.")
    print(f"[range] LM={model_name} layer={args.layer} saliency={args.saliency} tao={args.tao}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    lm = load_lm(model_name, device_map="auto")
    sanity_check_splice(lm, args.layer, tokenizer, device)
    hook = SplicingHook(lm, args.layer)

    def encode(h: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return ae.encoder((torch.from_numpy(h).to(device) - mean_t) / std_t).cpu().numpy()

    print("[range] Wiki slice (ppl texts + wiki-mean replacement) …")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wtxt = []
    for ex in wiki:
        wtxt.append(ex["text"][:1500])
        if len(wtxt) >= args.n_wiki:
            break
    h_w = shared.capture_h(lm, tokenizer, wtxt, args.layer, device)
    wiki_mean = {"h": h_w.mean(0), "z": encode(h_w).mean(0)}
    ppl_txt = wtxt[:args.n_ppl]
    z_recon = nl.make_z_edit(None, ae, mean_t, std_t, device)

    print(f"[range] Loading {args.dataset} test split …")
    ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["dataset_split"]).shuffle(seed=args.seed)
    need = args.n_fit + args.n_eval
    correct = shared.build_joint_correct_set(
        ds, dataset_cfg, tokenizer, lm, device, hook, z_recon, Path(args.correct_json), need,
        args.dataset, model_name=model_name,
        ae_sha=shared.ae_fingerprint(ae), batch_size=args.jc_batch_size, tag="range",
        classes=concepts,
    )

    # Same fit/eval split as steering_concept_compare, so st_global reproduces it.
    pool = {c: [d["text"] for d in correct if d["label"] == c] for c in range(len(shared.CLASSES))}
    fit_txt, fit_lab, eval_txt, eval_lab = [], [], [], []
    for c in range(len(shared.CLASSES)):
        K = len(pool[c])
        if K < 2:
            print(f"[range] WARNING: class {c} has only {K} joint-correct docs; results will be noisy.")
        n_fit_c = min(args.n_fit, max(1, int(K * 0.6))) if K < need else args.n_fit
        n_eval_c = min(args.n_eval, max(1, K - n_fit_c)) if K < need else args.n_eval
        fit_txt += pool[c][:n_fit_c]
        fit_lab += [c] * len(pool[c][:n_fit_c])
        eval_txt += pool[c][n_fit_c:n_fit_c + n_eval_c]
        eval_lab += [c] * len(pool[c][n_fit_c:n_fit_c + n_eval_c])
    fit_lab, eval_lab = np.array(fit_lab), np.array(eval_lab)

    print(f"[range] Fit pass: capturing h for {len(fit_txt)} docs …")
    acts_fit = {"h": shared.capture_h(lm, tokenizer, fit_txt, args.layer, device)}
    acts_fit["z"] = encode(acts_fit["h"])
    # Ranges, saliency, r and the comp mean come from docs the base model still gets
    # right at fit time (the NeuronLens reference fits on correct-only activations).
    fit_ok = np.array(shared.predict(lm, tokenizer, fit_txt, fit_lab, device)["correct"])
    print(f"[range] Fit pass: {int(fit_ok.sum())}/{len(fit_ok)} still predicted correctly")

    print(f"[range] Clean + AE-recon eval over {len(eval_txt)} docs …")
    base = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
    base_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
    hook.activate(z_recon)
    zbase = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
    zbase_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
    hook.deactivate()
    keep = [i for i in range(len(eval_txt)) if base["correct"][i] and zbase["correct"][i]]
    if len(keep) < len(eval_txt):
        print(f"[range] Re-filtered eval: {len(eval_txt)} -> {len(keep)} jointly-correct docs")
        eval_txt = [eval_txt[i] for i in keep]
        eval_lab = np.array([eval_lab[i] for i in keep])
        base = {k: [v[i] for i in keep] for k, v in base.items()}
        zbase = {k: [v[i] for i in keep] for k, v in zbase.items()}

    # Last-token eval activations, only for the gate firing-rate diagnostic.
    acts_eval = {"h": shared.capture_h(lm, tokenizer, eval_txt, args.layer, device)}
    acts_eval["z"] = encode(acts_eval["h"])
    idx_by_class = {c: np.where(eval_lab == c)[0] for c in range(len(shared.CLASSES))}
    own = {"h": (base, base_ppl), "z": (zbase, zbase_ppl)}

    def run(fn, c, comp_idx):
        idx = np.concatenate([idx_by_class[c], comp_idx])
        sub_txt = [eval_txt[i] for i in idx]
        sub_lab = [int(eval_lab[i]) for i in idx]
        hook.activate(fn)
        r = shared.predict(lm, tokenizer, sub_txt, sub_lab, device)
        pp = shared.perplexity(lm, tokenizer, ppl_txt, device)
        hook.deactivate()
        tgt = float(np.mean([r["correct"][k] for k, lab in enumerate(sub_lab) if lab == c]))
        comp = float(np.mean([r["correct"][k] for k, lab in enumerate(sub_lab) if lab != c]))
        return tgt, comp, float(pp)

    results = {
        "meta": {
            "checkpoint": str(ckpt), "layer": args.layer, "dataset": args.dataset,
            "saliency": args.saliency, "tao": args.tao, "percent": args.percent, "q": args.q,
            "ae_percent": args.ae_percent if args.ae_percent is not None else args.percent,
            "ae_q": args.ae_q if args.ae_q is not None else args.q,
            "removal_modes": removal_modes, "replace": replaces, "steer_modes": steer_modes,
            "alphas": [float(a) for a in args.alphas], "n_fit": args.n_fit, "n_eval": args.n_eval,
            "n_comp": args.n_comp, "n_ppl": args.n_ppl, "seed": args.seed,
            "base_acc": round(float(np.mean(base["correct"])), 4), "base_ppl": round(base_ppl, 3),
            "zrecon_acc": round(float(np.mean(zbase["correct"])), 4), "zrecon_ppl": round(zbase_ppl, 3),
            "selectivity_sign": "tgt_drop - comp_drop; higher is better",
        },
        "concepts": {},
    }
    print(f"[range] base acc={results['meta']['base_acc']:.3f} ppl={base_ppl:.2f} | "
          f"AE-recon acc={results['meta']['zrecon_acc']:.3f} ppl={zbase_ppl:.2f}")

    for c in concepts:
        comp_pool = np.where(eval_lab != c)[0]
        comp_idx = rng.choice(comp_pool, min(args.n_comp, len(comp_pool)), replace=False)
        print(f"\n=== concept {c} ({shared.CLASSES[c]})  n_tgt={len(idx_by_class[c])} n_comp={len(comp_idx)} ===")
        rec = {"fire": {}}
        for s in ("h", "z"):
            A = acts_fit[s][fit_ok]
            is_c = fit_lab[fit_ok] == c
            if is_c.sum() < 2 or (~is_c).sum() < 2:
                print(f"[range] WARNING: too few correct fit docs for concept {c} in {s}; skipping")
                continue
            D = A.shape[1]
            a_c, a_o = A[is_c], A[~is_c]
            mu_c, sd_c, mu_o, sd_o = a_c.mean(0), a_c.std(0), a_o.mean(0), a_o.std(0)
            score = np.abs(a_c).mean(0) if args.saliency == "abs" else nl.dprime_saliency(A, is_c)
            if s == "h":
                sal = nl.select_top(score, **(dict(q=args.q) if args.q is not None else dict(p=args.percent)))
            else:
                ae_q = args.ae_q if args.ae_q is not None else args.q
                ae_p = args.ae_percent if args.ae_percent is not None else args.percent
                sal = nl.select_top(score, **(dict(q=ae_q) if ae_q is not None else dict(p=ae_p)))
            lo_rng, hi_rng, _, _ = nl.fit_ranges(a_c, sal, args.tao)
            lo_sal, hi_sal, _, _ = nl.fit_ranges(a_c, sal, np.inf)
            inf = np.full(D, np.inf, dtype=np.float32)
            r = mu_c - mu_o
            rep = {"zero": np.zeros(D, dtype=np.float32), "comp": mu_o, "wiki": wiki_mean[s]}

            # Diagnostic: does the range gate fire more on this concept than on others?
            E = acts_eval[s][:, sal]
            inside = (E >= lo_rng[sal]) & (E <= hi_rng[sal])
            f_t, f_o = float(inside[idx_by_class[c]].mean()), float(inside[comp_idx].mean())
            rec["fire"][s] = {"tgt": round(f_t, 4), "comp": round(f_o, 4), "n_salient": int(sal.sum())}
            print(f"  [{s}] {int(sal.sum())} salient dims; range fires on target {f_t:.3f} vs others {f_o:.3f}")

            edits = {}
            for g in removal_modes:
                lo, hi = (lo_rng, hi_rng) if g == "range" else (lo_sal, hi_sal)
                for rp in replaces:
                    edits[f"rm_{g}_{rp}"] = nl.gate_edit(lo, hi, rep[rp], device)
            for alpha in args.alphas:
                a_tag = str(float(alpha))
                for m in steer_modes:
                    if m == "global":
                        e = nl.shift_edit(r, -inf, inf, alpha, device)
                    elif m == "salient":
                        e = nl.shift_edit(r, lo_sal, hi_sal, alpha, device)
                    elif m == "range":
                        e = nl.shift_edit(r, lo_rng, hi_rng, alpha, device)
                    else:
                        e = nl.transport_edit(mu_c, sd_c, mu_o, sd_o, lo_rng, hi_rng, alpha, device)
                    edits[f"st_{m}_a{a_tag}"] = e

            ref, ref_ppl = own[s]
            b_tgt = float(np.mean([ref["correct"][i] for i in idx_by_class[c]]))
            b_comp = float(np.mean([ref["correct"][i] for i in comp_idx]))
            for key, e in edits.items():
                fn = nl.make_h_edit(e, device) if s == "h" else nl.make_z_edit(e, ae, mean_t, std_t, device)
                tgt, comp, pp = run(fn, c, comp_idx)
                td, cd = b_tgt - tgt, b_comp - comp
                rec[f"{s}_{key}"] = {
                    "tgt_acc": round(tgt, 4), "comp_acc": round(comp, 4),
                    "tgt_drop": round(td, 4), "comp_drop": round(cd, 4),
                    "selectivity": round(td - cd, 4), "ppl": round(pp, 3),
                    "ppl_rise": round(pp - ref_ppl, 3),
                }
                print(f"  {s} {key:<20} tgt {tgt:.2f} comp {comp:.2f}  sel {td - cd:+.3f}  "
                      f"ppl {pp:.2f} ({pp - ref_ppl:+.2f})")
        results["concepts"][str(c)] = rec
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)      # checkpoint after every concept

    results["summary"] = summarize(results, concepts)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[range] Saved -> {args.out}")


def summarize(results: dict, concepts: list[int]) -> dict:
    """Mean over concepts per substrate, plus the PAIRED z - h difference."""
    C = results["concepts"]
    keys = sorted({k[2:] for c in concepts for k in C[str(c)] if k[:2] in ("h_", "z_")},
                  key=lambda k: (not k.startswith("rm_"), k))
    out = {}
    print("\n=== summary over concepts (selectivity: higher = better; ppl_rise vs own baseline) ===")
    print(f"{'intervention':<22} | {'h sel':>6} {'z sel':>6} | {'z-h':>7} {'+-se':>6} {'z wins':>7} | "
          f"{'h tgt/comp drop':>15} {'z tgt/comp drop':>15} | {'h ppl+':>7} {'z ppl+':>7}")
    for key in keys:
        rows = [(C[str(c)][f"h_{key}"], C[str(c)][f"z_{key}"]) for c in concepts
                if f"h_{key}" in C[str(c)] and f"z_{key}" in C[str(c)]]
        if not rows:
            continue
        h = {f: np.array([r[0][f] for r in rows]) for f in ("selectivity", "tgt_drop", "comp_drop", "ppl_rise")}
        z = {f: np.array([r[1][f] for r in rows]) for f in ("selectivity", "tgt_drop", "comp_drop", "ppl_rise")}
        d = z["selectivity"] - h["selectivity"]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        out[key] = {
            "n": len(rows),
            "h_sel": round(float(h["selectivity"].mean()), 4), "z_sel": round(float(z["selectivity"].mean()), 4),
            "z_minus_h": round(float(d.mean()), 4), "z_minus_h_se": round(se, 4),
            "z_wins": int((d > 1e-9).sum()), "h_wins": int((d < -1e-9).sum()),
            "h_tgt_drop": round(float(h["tgt_drop"].mean()), 4), "h_comp_drop": round(float(h["comp_drop"].mean()), 4),
            "z_tgt_drop": round(float(z["tgt_drop"].mean()), 4), "z_comp_drop": round(float(z["comp_drop"].mean()), 4),
            "h_ppl_rise": round(float(h["ppl_rise"].mean()), 3), "z_ppl_rise": round(float(z["ppl_rise"].mean()), 3),
        }
        v = out[key]
        print(f"{key:<22} | {v['h_sel']:+6.3f} {v['z_sel']:+6.3f} | {v['z_minus_h']:+7.3f} {se:6.3f} "
              f"{v['z_wins']:>3}/{v['h_wins']:<3} | {v['h_tgt_drop']:+7.2f}/{v['h_comp_drop']:+6.2f} "
              f"{v['z_tgt_drop']:+7.2f}/{v['z_comp_drop']:+6.2f} | {v['h_ppl_rise']:+7.2f} {v['z_ppl_rise']:+7.2f}")
    fire = {s: [C[str(c)]["fire"][s] for c in concepts if s in C[str(c)].get("fire", {})] for s in ("h", "z")}
    for s in ("h", "z"):
        if fire[s]:
            t = np.mean([f["tgt"] for f in fire[s]]); o = np.mean([f["comp"] for f in fire[s]])
            print(f"range gate firing [{s}]: target {t:.3f}  others {o:.3f}  gap {t - o:+.3f}")
            out[f"fire_{s}"] = {"tgt": round(float(t), 4), "comp": round(float(o), 4)}
    return out


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(0)
