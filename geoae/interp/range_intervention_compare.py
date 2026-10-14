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
import hashlib
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
from geoae.hooks import SplicingHook, TokenIdTap
from geoae.interp import neuronlens as nl
from geoae.interp import _shared as shared
from geoae.interp.atlas_classes import comp_by_class


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="db14", choices=["db14", "emotions", "emotions_train", "ag_news", "biasbios", *shared.ATLAS_CLASS_DATASETS, *shared.CONTROL_DATASETS])
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
    ap.add_argument("--max_try", type=int, default=None,
                    help="docs tried per (class, split) when building the joint-correct pool; on a dataset without fit/test "
                         "tags, the first max_try docs of each class (a class the splices rarely get right would otherwise "
                         "scan its whole split)")
    ap.add_argument("--extra_split", default="", help="Hub datasets: a second split appended after the configured one "
                                                      "(AG News train: the test split has too few documents every model gets right)")
    ap.add_argument("--extra_max", type=int, default=20000)
    ap.add_argument("--edit_ppl", type=int, default=1, help="0 skips the perplexity pass of every edited run (ppl fields = NaN)")
    ap.add_argument("--hb_table", default="",
                    help="token_bias .npz: enables substrate hb, the MATCHED BASE for a token-bypass AE "
                         "(edit (h-mean)/std - b[tok], add b[tok] back; h's saliency settings)")
    ap.add_argument("--sae", default="",
                    help="SAE spec of geoae/interp/sae_space.py (e.g. scope:16k): enables substrate s, edits on the SAE's "
                         "codes with the SAE reconstruction spliced in (BOS kept); h's saliency settings")
    ap.add_argument("--proj", default="",
                    help="direction-projection test: remove ONE assigned concept direction per method from the UNEDITED base "
                         "model's residual by orthogonal projection, h - (h.d)d (records h_pj_<name>; h_pjc_<name> = centred on "
                         "the class-balanced fit mean). Every method starts from the same model and documents. Comma list of: "
                         "label (class mean - other-class mean, supervised reference); km / kmt (plain k-means on the residual / "
                         "token-removed residual, --km_plain / --km_tok); geoae (the --checkpoint AE's codebook); any name in "
                         "--proj_ae; sae_km (k-means on the SAE codes, --sae_kmeans); sae_f1 / sae_f5 (the SAE's top-1 / top-5 "
                         "features by d-prime among features more active on the concept; decoder rows); h_n1 / h_n5 (top residual "
                         "dimensions by d-prime); <ae>_z1 / <ae>_z5 for geoae and every --proj_ae name (top latent dimensions "
                         "by d-prime; decoder columns). Cluster methods: the "
                         "concept's unit is the cluster holding most of its fit docs (answer position), its direction the "
                         "unit's centroid minus the mean of the assigned centroids of all classes, mapped to the residual by "
                         "the method's own (linear) decoder. Needs h in --substrates.")
    ap.add_argument("--km_plain", default="", help="--proj km: plain residual k-means .npz")
    ap.add_argument("--km_tok", default="", help="--proj kmt: plain token-removed k-means .npz (with token_bias)")
    ap.add_argument("--sae_kmeans", default="", help="--proj sae_km: k-means centroids on the SAE codes (.npz, centroids)")
    ap.add_argument("--proj_ae", default="", help="--proj <name>: extra AE codebooks, name=checkpoint,... (e.g. the no-bypass AE)")
    ap.add_argument("--tp_all", action="store_true",
                    help="documents must be TRUE POSITIVES in all three unedited models: base, AE splice AND SAE splice "
                         "(needs --sae). The pool is built with the SAE splice as an extra filter "
                         "(_shared.build_joint_correct_set extra_recons; give it its own --correct_json) and the "
                         "evaluation re-filter also requires the SAE splice to be correct.")
    ap.add_argument("--min_eval", type=int, default=1, help="a concept with fewer evaluation documents is skipped")
    ap.add_argument("--substrates", default="h,z",
                    help="comma list of h, z, hb (hb needs --hb_table). Skipping z pairs hb with an "
                         "existing z file on the same doc set.")
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
    ae, norm, cfg = load_ae_from_checkpoint(ckpt, device, allow_token_bias=True)
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
    # Token-bypass AE: splices read each position's current-token id off the input
    # embedding, and every fit-time encode uses the id captured with the activation.
    substrates = [x for x in args.substrates.split(",") if x]
    # s<k> (s1, s5, ...): SAE codes with only the k features of largest d-prime among those MORE active on the concept
    # edited (range steering: their in-range values; plain steering: all their values); reference = the SAE splice
    topk = {x: int(x[1:]) for x in substrates if len(x) > 1 and x[0] == "s" and x[1:].isdigit()}
    if (not set(substrates) - set(topk) <= {"h", "z", "hb", "s"} or ("hb" in substrates and not args.hb_table)
            or (("s" in substrates or topk) and not args.sae)):
        sys.exit(f"[range] --substrates must be from h,z,hb,s,s<k> (hb needs --hb_table, s and s<k> need --sae); got {args.substrates}")
    tap = TokenIdTap(lm) if (ae.has_token_bias or args.hb_table or args.sae) else None
    sae = None
    if args.sae:
        from geoae.interp.sae_space import load_sae_space
        sae = load_sae_space(args.sae, device, "sae")

    def encode_s(h: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return sae.encode_raw(torch.from_numpy(h).to(device).float(), None).cpu().numpy()
    tb = None
    if args.hb_table:
        from geoae.token_bias import TokenBiasLookup
        tb = TokenBiasLookup(args.hb_table, device)
        if not (np.allclose(tb.norm_mean, mean_t.cpu().numpy()) and np.allclose(tb.norm_std, std_t.cpu().numpy())):
            sys.exit(f"[range] {args.hb_table} was built under different norm stats than {ckpt}")

    def to_hb(h: np.ndarray, ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return ((torch.from_numpy(h).to(device) - mean_t) / std_t
                    - tb(torch.from_numpy(ids).to(device))).cpu().numpy()

    def encode(h: np.ndarray, ids: np.ndarray) -> np.ndarray:
        tok = torch.from_numpy(ids).to(device) if ae.has_token_bias else None
        with torch.no_grad():
            return ae.encode((torch.from_numpy(h).to(device) - mean_t) / std_t, tok).cpu().numpy()

    def capture(texts):
        return shared.capture_h(lm, tokenizer, texts, args.layer, device, return_ids=True)

    print("[range] Wiki slice (ppl texts + wiki-mean replacement) …")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wtxt = []
    for ex in wiki:
        wtxt.append(ex["text"][:1500])
        if len(wtxt) >= args.n_wiki:
            break
    h_w, ids_w = capture(wtxt)
    wiki_mean = {"h": h_w.mean(0), "z": encode(h_w, ids_w).mean(0)}
    if tb is not None:
        wiki_mean["hb"] = to_hb(h_w, ids_w).mean(0)
    if sae is not None:
        wiki_mean["s"] = encode_s(h_w).mean(0)
    ppl_txt = wtxt[:args.n_ppl]
    z_recon = nl.make_z_edit(None, ae, mean_t, std_t, device, tap=tap)

    print(f"[range] Loading {args.dataset} test split …")
    if dataset_cfg.get("local_parquet"):      # frozen local corpus (Atlas classes): order fixed at prep, not reshuffled
        ds = load_dataset("parquet", data_files=dataset_cfg["local_parquet"], split="train")
    else:
        ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["dataset_split"]).shuffle(seed=args.seed)
        if args.extra_split:
            from datasets import concatenate_datasets
            ds2 = load_dataset(dataset_cfg["dataset_name"], split=args.extra_split).shuffle(seed=args.seed)
            ds = concatenate_datasets([ds, ds2.select(range(min(len(ds2), args.extra_max)))])
    need = args.n_fit + args.n_eval
    split_kw = ({"need_by_split": {"fit": args.n_fit, "test": args.n_eval}, "max_try": args.max_try}
                if dataset_cfg.get("split_column") else {})
    cap = args.max_try if args.max_try and not split_kw else None
    if cap:
        seen, rows = {}, []
        for i, lab in enumerate(ds[dataset_cfg.get("label_column", "label")]):
            if seen.get(int(lab), 0) < cap:
                seen[int(lab)] = seen.get(int(lab), 0) + 1
                rows.append(i)
        ds = ds.select(rows)
        print(f"[range] --max_try {cap}: {len(rows)} documents kept ({min(seen.values())}-{max(seen.values())} per class)")
    if args.tp_all and sae is None:
        sys.exit("[range] --tp_all needs --sae")
    s_recon = nl.make_s_edit(None, sae, device, tap, tokenizer.bos_token_id) if sae is not None else None
    # the pool cache is keyed on everything that changes which documents can enter it
    xk = (([f"tp3|sae={args.sae}"] if args.tp_all else [])
          + ([f"extra_split={args.extra_split}:{args.extra_max}"] if args.extra_split else [])
          + ([f"max_try={cap}"] if cap else []))
    tp_kw = {**({"extra_recons": {"sae": s_recon}} if args.tp_all else {}), **({"extra_key": "|".join(xk)} if xk else {})}
    correct = shared.build_joint_correct_set(
        ds, dataset_cfg, tokenizer, lm, device, hook, z_recon, Path(args.correct_json), need,
        args.dataset, model_name=model_name,
        ae_sha=shared.ae_fingerprint(ae), batch_size=args.jc_batch_size, tag="range",
        classes=concepts, **split_kw, **tp_kw,
    )

    # Same fit/eval split as steering_concept_compare, so st_global reproduces it.
    pool = {c: [d["text"] for d in correct if d["label"] == c] for c in range(len(shared.CLASSES))}
    fit_txt, fit_lab, eval_txt, eval_lab = [], [], [], []
    if split_kw:     # pool carries document-level fit/test tags: fit docs from the fit split, eval docs from test
        for c in range(len(shared.CLASSES)):
            fd = [d["text"] for d in correct if d["label"] == c and d.get("split") == "fit"][:args.n_fit]
            ed = [d["text"] for d in correct if d["label"] == c and d.get("split") == "test"][:args.n_eval]
            fit_txt += fd; fit_lab += [c] * len(fd)
            eval_txt += ed; eval_lab += [c] * len(ed)
    for c in ([] if split_kw else range(len(shared.CLASSES))):
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
    h_fit, ids_fit = capture(fit_txt)
    acts_fit = {"h": h_fit, "z": encode(h_fit, ids_fit)}
    if tb is not None:
        acts_fit["hb"] = to_hb(h_fit, ids_fit)
    if sae is not None:
        acts_fit["s"] = encode_s(h_fit)
    # Ranges, saliency, r and the comp mean come from docs the base model still gets
    # right at fit time (the NeuronLens reference fits on correct-only activations).
    fit_ok = np.array(shared.predict(lm, tokenizer, fit_txt, fit_lab, device)["correct"])
    print(f"[range] Fit pass: {int(fit_ok.sum())}/{len(fit_ok)} still predicted correctly")

    # ---- direction-projection test: every method's units at the answer position of the fit docs
    proj = _csv(args.proj)
    proj_ae = dict(x.split("=", 1) for x in _csv(args.proj_ae))
    punits, pdir, pinfo, pclasses, pcenter, pz = {}, {}, {}, [], None, {}
    if proj:
        if "h" not in substrates:
            sys.exit("[range] --proj needs h in --substrates (the unedited base reference)")
        from geoae.interp.closest_tokens import load_baseline_kmeans
        Hf, tokf = torch.from_numpy(h_fit).to(device).float(), torch.from_numpy(ids_fit).to(device)

        def km_unit(path, token_removed):          # -> (cluster of every fit doc, centroids as RAW residual displacements)
            C, m, sd, *_ = load_baseline_kmeans(Path(path), device, allow_token_bias=True)
            x = (Hf - m) / sd
            if token_removed:
                from geoae.token_bias import TokenBiasLookup
                x = x - TokenBiasLookup(str(np.load(path, allow_pickle=True)["token_bias"]), device)(tokf)
            return torch.cdist(x, C.float()).argmin(1).cpu().numpy(), C.float() * sd

        def ae_unit(name, a, m, sd):               # decoder is linear without bias: centroid differences decode exactly
            z = a.encode((Hf - m) / sd, tokf if a.has_token_bias else None)
            pz[name] = (z.cpu().numpy(), (a.decoder.weight.float() * sd[:, None]).clone())   # latents, RAW decoder columns (D, L)
            return torch.cdist(z, a.centroids.float()).argmin(1).cpu().numpy(), a.decoder(a.centroids.float()) * sd

        with torch.no_grad():
            if "km" in proj:
                punits["km"] = km_unit(args.km_plain, False)
            if "kmt" in proj:
                punits["kmt"] = km_unit(args.km_tok, True)
            want = lambda n: bool({n, f"{n}_z1", f"{n}_z5"} & set(proj))
            if want("geoae"):
                u = ae_unit("geoae", ae, mean_t, std_t)
                if "geoae" in proj:
                    punits["geoae"] = u
            for name, path in proj_ae.items():
                if want(name):
                    a2, n2, _ = load_ae_from_checkpoint(Path(path), device, allow_token_bias=True)
                    u = ae_unit(name, a2.eval(), n2["mean"], n2["std"])
                    if name in proj:
                        punits[name] = u
                    del a2
            if "sae_km" in proj:
                Cs = torch.as_tensor(np.load(args.sae_kmeans)["centroids"], dtype=torch.float32, device=device)
                punits["sae_km"] = (torch.cdist(sae.encode_raw(Hf, None), Cs).argmin(1).cpu().numpy(),
                                    Cs @ sae.decoder_directions.float())
        unknown = (set(proj) - set(punits) - {"label", "sae_f1", "sae_f5", "h_n1", "h_n5"}
                   - {f"{n}_z{k}" for n in pz for k in (1, 5)})
        if unknown or (not sae and {"sae_km", "sae_f1", "sae_f5"} & set(proj)):
            sys.exit(f"[range] --proj: unknown or unconfigured handle(s) {sorted(unknown)} (SAE handles need --sae)")
        pclasses = [c for c in range(len(shared.CLASSES)) if (fit_ok & (fit_lab == c)).sum() >= 2]
        pcenter = torch.from_numpy(np.stack([h_fit[fit_ok & (fit_lab == c)].mean(0) for c in pclasses]).mean(0)).to(device).float()
        for name, (lab, vec) in punits.items():
            unit, pinfo[name] = {}, {}
            for c in pclasses:
                cnt = np.bincount(lab[fit_ok & (fit_lab == c)], minlength=vec.shape[0])
                k = int(cnt.argmax())
                unit[c] = k
                pinfo[name][c] = {"unit": k, "coverage": round(float(cnt[k] / cnt.sum()), 4),
                                  "purity": round(float(cnt[k] / max(int((lab[fit_ok] == k).sum()), 1)), 4)}
            mean_vec = torch.stack([vec[unit[c]] for c in pclasses]).mean(0)
            pdir[name] = {c: vec[unit[c]] - mean_vec for c in pclasses}
            print(f"[range] proj {name}: {len(set(unit.values()))} distinct units for {len(pclasses)} classes; mean coverage "
                  f"{np.mean([v['coverage'] for v in pinfo[name].values()]):.2f}, mean purity "
                  f"{np.mean([v['purity'] for v in pinfo[name].values()]):.2f}")
        del Hf

    print(f"[range] Clean + AE-recon eval over {len(eval_txt)} docs …")
    base = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
    base_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
    hook.activate(z_recon)
    zbase = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
    zbase_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
    hook.deactivate()
    keep = [i for i in range(len(eval_txt)) if base["correct"][i] and zbase["correct"][i]]
    if args.tp_all:          # also a true positive under the unedited SAE splice
        hook.activate(s_recon)
        s_pre = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
        hook.deactivate()
        keep = [i for i in keep if s_pre["correct"][i]]
    if len(keep) < len(eval_txt):
        print(f"[range] Re-filtered eval: {len(eval_txt)} -> {len(keep)} jointly-correct docs"
              + (" (base, AE splice and SAE splice)" if args.tp_all else ""))
        eval_txt = [eval_txt[i] for i in keep]
        eval_lab = np.array([eval_lab[i] for i in keep])
        base = {k: [v[i] for i in keep] for k, v in base.items()}
        zbase = {k: [v[i] for i in keep] for k, v in zbase.items()}

    # Last-token eval activations, only for the gate firing-rate diagnostic.
    h_ev, ids_ev = capture(eval_txt)
    acts_eval = {"h": h_ev, "z": encode(h_ev, ids_ev)}
    if tb is not None:
        acts_eval["hb"] = to_hb(h_ev, ids_ev)
    idx_by_class = {c: np.where(eval_lab == c)[0] for c in range(len(shared.CLASSES))}
    own = {"h": (base, base_ppl), "z": (zbase, zbase_ppl), "hb": (base, base_ppl)}   # hb unedited == identity
    if sae is not None:      # SAE-reconstruction splice on the final (jointly-correct) eval docs
        acts_eval["s"] = encode_s(h_ev)
        hook.activate(nl.make_s_edit(None, sae, device, tap, tokenizer.bos_token_id))
        sbase = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
        sbase_ppl = shared.perplexity(lm, tokenizer, ppl_txt, device)
        hook.deactivate()
        own["s"] = (sbase, sbase_ppl)
        print(f"[range] SAE-recon acc={np.mean(sbase['correct']):.3f} conf={np.mean(sbase['gold_conf']):.3f} ppl={sbase_ppl:.2f}")

    def run(fn, c, comp_idx, ppl=True):
        idx = np.concatenate([idx_by_class[c], comp_idx])
        sub_txt = [eval_txt[i] for i in idx]
        sub_lab = [int(eval_lab[i]) for i in idx]
        hook.activate(fn)
        r = shared.predict(lm, tokenizer, sub_txt, sub_lab, device)
        pp = shared.perplexity(lm, tokenizer, ppl_txt, device) if ppl else float("nan")
        hook.deactivate()
        tgt = float(np.mean([r["correct"][k] for k, lab in enumerate(sub_lab) if lab == c]))
        comp = float(np.mean([r["correct"][k] for k, lab in enumerate(sub_lab) if lab != c]))
        return tgt, comp, float(pp), r["correct"], r["gold_conf"], r["pred"]

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
            "substrates": substrates, "hb_table": args.hb_table, "tp_all": bool(args.tp_all), "extra_split": args.extra_split, "max_try": args.max_try,
            "proj": proj, "proj_files": {"km_plain": args.km_plain, "km_tok": args.km_tok, "sae_kmeans": args.sae_kmeans,
                                         "proj_ae": proj_ae},
            # additive bookkeeping
            "classes": list(shared.CLASSES),
            "n_fit_by_class": {str(c): int((fit_lab == c).sum()) for c in range(len(shared.CLASSES))},
            "n_eval_by_class": {str(c): int((eval_lab == c).sum()) for c in range(len(shared.CLASSES))},
            "eval_sha": hashlib.sha1("\x1f".join(f"{l}|{t}" for t, l in zip(eval_txt, eval_lab)).encode()).hexdigest()[:16],
            "base_conf": round(float(np.mean(base["gold_conf"])), 4),
            "zrecon_conf": round(float(np.mean(zbase["gold_conf"])), 4),
            **({"sae": args.sae, "srecon_acc": round(float(np.mean(sbase["correct"])), 4),
                "srecon_conf": round(float(np.mean(sbase["gold_conf"])), 4), "srecon_ppl": round(sbase_ppl, 3)} if sae is not None else {}),
            "confidence_def": "softmax probability of the gold label's first token at the first generated position (shared.predict gold_conf)",
        },
        "concepts": {},
    }
    print(f"[range] base acc={results['meta']['base_acc']:.3f} ppl={base_ppl:.2f} | "
          f"AE-recon acc={results['meta']['zrecon_acc']:.3f} ppl={zbase_ppl:.2f}")

    for c in concepts:
        if len(idx_by_class[c]) < max(1, args.min_eval):
            print(f"\n=== concept {c} ({shared.CLASSES[c]}) skipped: {len(idx_by_class[c])} evaluation documents ===")
            continue
        comp_pool = np.where(eval_lab != c)[0]
        comp_idx = rng.choice(comp_pool, min(args.n_comp, len(comp_pool)), replace=False)
        print(f"\n=== concept {c} ({shared.CLASSES[c]})  n_tgt={len(idx_by_class[c])} n_comp={len(comp_idx)} ===")
        rec = {"fire": {}}
        rec["eval_idx"] = {"tgt": [int(i) for i in idx_by_class[c]], "comp": [int(i) for i in comp_idx]}
        rec["ref"] = {}      # unedited per-doc reference (NOT h_/z_-prefixed: summarize() would read it as an intervention)
        for s in substrates:
            sb = "s" if s in topk else s          # where the activations and the unedited reference come from
            A = acts_fit[sb][fit_ok]
            is_c = fit_lab[fit_ok] == c
            if is_c.sum() < 2 or (~is_c).sum() < 2:
                print(f"[range] WARNING: too few correct fit docs for concept {c} in {s}; skipping")
                continue
            D = A.shape[1]
            a_c, a_o = A[is_c], A[~is_c]
            mu_c, sd_c, mu_o, sd_o = a_c.mean(0), a_c.std(0), a_o.mean(0), a_o.std(0)
            score = np.abs(a_c).mean(0) if args.saliency == "abs" else nl.dprime_saliency(A, is_c)
            if s in topk:
                sc_k = score.copy()
                sc_k[mu_c <= mu_o] = -1.0
                sal = np.zeros(D, dtype=bool)
                sal[np.argsort(-sc_k)[:topk[s]]] = True
            elif s in ("h", "hb", "s"):
                sal = nl.select_top(score, **(dict(q=args.q) if args.q is not None else dict(p=args.percent)))
            else:
                ae_q = args.ae_q if args.ae_q is not None else args.q
                ae_p = args.ae_percent if args.ae_percent is not None else args.percent
                sal = nl.select_top(score, **(dict(q=ae_q) if ae_q is not None else dict(p=ae_p)))
            lo_rng, hi_rng, _, _ = nl.fit_ranges(a_c, sal, args.tao)
            lo_sal, hi_sal, _, _ = nl.fit_ranges(a_c, sal, np.inf)
            inf = np.full(D, np.inf, dtype=np.float32)
            r = mu_c - mu_o
            rep = {"zero": np.zeros(D, dtype=np.float32), "comp": mu_o, "wiki": wiki_mean[sb]}

            # Diagnostic: does the range gate fire more on this concept than on others?
            E = acts_eval[sb][:, sal]
            inside = (E >= lo_rng[sal]) & (E <= hi_rng[sal])
            f_t, f_o = float(inside[idx_by_class[c]].mean()), float(inside[comp_idx].mean())
            rec["fire"][s] = {"tgt": round(f_t, 4), "comp": round(f_o, 4), "n_salient": int(sal.sum())}
            if s in topk:
                ids_k = np.where(sal)[0]
                rec["fire"][s].update(features=[int(i) for i in ids_k], dprime=[round(float(score[i]), 3) for i in ids_k],
                                      mean_tgt=[round(float(mu_c[i]), 3) for i in ids_k],
                                      mean_other=[round(float(mu_o[i]), 3) for i in ids_k])
            print(f"  [{s}] {int(sal.sum())} salient dims; range fires on target {f_t:.3f} vs others {f_o:.3f}")

            edits = {}
            for g in removal_modes:
                lo, hi = (lo_rng, hi_rng) if g == "range" else (lo_sal, hi_sal)
                for rp in replaces:
                    edits[f"rm_{g}_{rp}"] = nl.gate_edit(lo, hi, rep[rp], device)
            for alpha in args.alphas:
                a_tag = str(float(alpha))
                for m in steer_modes:
                    if m == "global":      # a top-k arm shifts every value of its k features, nothing else
                        e = nl.shift_edit(r, *((lo_sal, hi_sal) if s in topk else (-inf, inf)), alpha, device)
                    elif m == "salient":
                        e = nl.shift_edit(r, lo_sal, hi_sal, alpha, device)
                    elif m == "range":
                        e = nl.shift_edit(r, lo_rng, hi_rng, alpha, device)
                    else:
                        e = nl.transport_edit(mu_c, sd_c, mu_o, sd_o, lo_rng, hi_rng, alpha, device)
                    edits[f"st_{m}_a{a_tag}"] = e

            ref, ref_ppl = own[sb]
            b_tgt = float(np.mean([ref["correct"][i] for i in idx_by_class[c]]))
            b_comp = float(np.mean([ref["correct"][i] for i in comp_idx]))
            b_tgt_conf = float(np.mean([ref["gold_conf"][i] for i in idx_by_class[c]]))
            b_comp_conf = float(np.mean([ref["gold_conf"][i] for i in comp_idx]))
            order = list(idx_by_class[c]) + list(comp_idx)
            rec["ref"][s] = {"tgt_acc": round(b_tgt, 4), "comp_acc": round(b_comp, 4),
                             "tgt_conf": round(b_tgt_conf, 4), "comp_conf": round(b_comp_conf, 4),
                             "doc_conf": [round(float(ref["gold_conf"][i]), 4) for i in order],
                             "doc_correct": [int(bool(ref["correct"][i])) for i in order],
                             "doc_pred": [ref["pred"][i] for i in order]}
            for key, e in edits.items():
                fn = (nl.make_h_edit(e, device) if s == "h"
                      else nl.make_hb_edit(e, tb, mean_t, std_t, device, tap) if s == "hb"
                      else nl.make_s_edit(e, sae, device, tap, tokenizer.bos_token_id) if sb == "s"
                      else nl.make_z_edit(e, ae, mean_t, std_t, device, tap=tap))
                tgt, comp, pp, corr, conf, pred = run(fn, c, comp_idx, ppl=bool(args.edit_ppl))
                td, cd = b_tgt - tgt, b_comp - comp
                nt = len(idx_by_class[c])      # run() evaluates [target docs, complement docs] in that order
                tc, cc = float(np.mean(conf[:nt])), float(np.mean(conf[nt:]))
                tcd, ccd = b_tgt_conf - tc, b_comp_conf - cc
                rec[f"{s}_{key}"] = {
                    "tgt_acc": round(tgt, 4), "comp_acc": round(comp, 4),
                    "tgt_drop": round(td, 4), "comp_drop": round(cd, 4),
                    "selectivity": round(td - cd, 4), "ppl": round(pp, 3),
                    "ppl_rise": round(pp - ref_ppl, 3),
                    # additive: per-OTHER-class accuracy drop on the complement sample (collateral matrix row)
                    "comp_by_class": comp_by_class(eval_lab[comp_idx], [ref["correct"][i] for i in comp_idx], corr[nt:]),
                    # additive: confidence in the gold label (first-token softmax prob) and per-doc records
                    "tgt_conf": round(tc, 4), "comp_conf": round(cc, 4),
                    "tgt_conf_drop": round(tcd, 4), "comp_conf_drop": round(ccd, 4),
                    "conf_selectivity": round(tcd - ccd, 4),
                    "doc_conf": [round(float(x), 4) for x in conf], "doc_correct": [int(bool(x)) for x in corr],
                    "doc_pred": list(pred),
                }
                print(f"  {s} {key:<20} tgt {tgt:.2f} comp {comp:.2f}  sel {td - cd:+.3f}  "
                      f"ppl {pp:.2f} ({pp - ref_ppl:+.2f})  conf-sel {tcd - ccd:+.3f}")
        if proj and "h" in rec["ref"]:
            ref, ref_ppl = own["h"]
            nt = len(idx_by_class[c])
            b_tgt = float(np.mean([ref["correct"][i] for i in idx_by_class[c]]))
            b_comp = float(np.mean([ref["correct"][i] for i in comp_idx]))
            b_tc = float(np.mean([ref["gold_conf"][i] for i in idx_by_class[c]]))
            b_cc = float(np.mean([ref["gold_conf"][i] for i in comp_idx]))
            Ah, is_c = acts_fit["h"][fit_ok], fit_lab[fit_ok] == c
            r_lab = torch.from_numpy(Ah[is_c].mean(0) - Ah[~is_c].mean(0)).to(device).float()
            dirs, rec["proj"] = {}, {}
            if "label" in proj:
                dirs["label"] = r_lab[:, None]
                rec["proj"]["label"] = {}
            for name in punits:
                if c in pdir[name]:
                    dirs[name] = pdir[name][c][:, None]
                    rec["proj"][name] = dict(pinfo[name][c])
            if sae is not None and {"sae_f1", "sae_f5"} & set(proj):
                As = acts_fit["s"][fit_ok]
                sc = nl.dprime_saliency(As, is_c)
                sc[As[is_c].mean(0) <= As[~is_c].mean(0)] = -1.0       # only features MORE active on the concept
                top = np.argsort(-sc)[:5]
                for name, ids in (("sae_f1", top[:1]), ("sae_f5", top)):
                    if name in proj:
                        dirs[name] = sae.decoder_directions.float()[torch.as_tensor(ids, device=device)].T
                        rec["proj"][name] = {"features": [int(i) for i in ids], "dprime": [round(float(sc[i]), 3) for i in ids],
                                             "active_on_concept": [round(float((As[is_c][:, i] > 0).mean()), 3) for i in ids],
                                             "active_on_others": [round(float((As[~is_c][:, i] > 0).mean()), 3) for i in ids]}
            coord = {}                                # coordinate units: top-k by d-prime (the control tables' saliency)
            if {"h_n1", "h_n5"} & set(proj):
                coord["h_n"] = (nl.dprime_saliency(Ah, is_c), None)
            for name, (Zf, Wraw) in pz.items():
                if {f"{name}_z1", f"{name}_z5"} & set(proj):
                    coord[f"{name}_z"] = (nl.dprime_saliency(Zf[fit_ok], is_c), Wraw)
            for base_name, (sc, Wraw) in coord.items():
                top = np.argsort(-sc)[:5]
                for k in (1, 5):
                    name = f"{base_name}{k}"
                    if name not in proj:
                        continue
                    ids = torch.as_tensor(top[:k].copy(), device=device)
                    dirs[name] = (torch.nn.functional.one_hot(ids, Ah.shape[1]).float().T if Wraw is None else Wraw[:, ids])
                    rec["proj"][name] = {"dims": [int(i) for i in top[:k]], "dprime": [round(float(sc[i]), 3) for i in top[:k]]}
            for name, Dm in dirs.items():
                if float(Dm.norm()) < 1e-8:          # every class assigned the same unit: the method has no direction
                    rec["proj"][name]["degenerate"] = True
                    print(f"  pj {name:<10} no direction (all classes share one unit)")
                    continue
                Q = torch.linalg.qr(Dm).Q             # (D, r) orthonormal basis of the direction(s)
                if Dm.shape[1] == 1:
                    rec["proj"][name]["cos_label"] = round(float(torch.nn.functional.cosine_similarity(Dm[:, 0], r_lab, dim=0)), 4)
                    rec["proj"][name]["coef_tgt_minus_comp_sd"] = round(float(
                        ((Ah[is_c] @ Q[:, 0].cpu().numpy()).mean() - (Ah[~is_c] @ Q[:, 0].cpu().numpy()).mean())
                        / (Ah @ Q[:, 0].cpu().numpy()).std()), 3)
                for tag, cen in (("pj", None), ("pjc", pcenter)):
                    e = (lambda a, Q=Q, cen=cen: a - ((a if cen is None else a - cen) @ Q) @ Q.T)
                    tgt, comp, pp, corr, conf, pred = run(nl.make_h_edit(e, device), c, comp_idx, ppl=False)
                    tc, cc = float(np.mean(conf[:nt])), float(np.mean(conf[nt:]))
                    rec[f"h_{tag}_{name}"] = {
                        "tgt_acc": round(tgt, 4), "comp_acc": round(comp, 4),
                        "tgt_drop": round(b_tgt - tgt, 4), "comp_drop": round(b_comp - comp, 4),
                        "selectivity": round((b_tgt - tgt) - (b_comp - comp), 4),
                        "tgt_conf": round(tc, 4), "comp_conf": round(cc, 4),
                        "tgt_conf_drop": round(b_tc - tc, 4), "comp_conf_drop": round(b_cc - cc, 4),
                        "conf_selectivity": round((b_tc - tc) - (b_cc - cc), 4),
                        "doc_conf": [round(float(x), 4) for x in conf], "doc_correct": [int(bool(x)) for x in corr],
                        "doc_pred": list(pred),
                    }
                    print(f"  {tag:<3} {name:<12} tgt conf {b_tc:.2f}->{tc:.2f}  other conf {b_cc:.2f}->{cc:.2f}  "
                          f"acc {tgt:.2f}/{comp:.2f}")
        results["concepts"][str(c)] = rec
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)      # checkpoint after every concept

    results["summary"] = summarize(results, [c for c in concepts if str(c) in results["concepts"]])   # --min_eval skips
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
        if all("conf_selectivity" in r[i] for r in rows for i in (0, 1)):      # additive: confidence-based selectivity
            hc = {f: np.array([r[0][f] for r in rows]) for f in ("conf_selectivity", "tgt_conf_drop", "comp_conf_drop")}
            zc = {f: np.array([r[1][f] for r in rows]) for f in ("conf_selectivity", "tgt_conf_drop", "comp_conf_drop")}
            dc = zc["conf_selectivity"] - hc["conf_selectivity"]
            sec = float(dc.std(ddof=1) / np.sqrt(len(dc))) if len(dc) > 1 else float("nan")
            out[key].update({
                "h_conf_sel": round(float(hc["conf_selectivity"].mean()), 4),
                "z_conf_sel": round(float(zc["conf_selectivity"].mean()), 4),
                "z_minus_h_conf": round(float(dc.mean()), 4), "z_minus_h_conf_se": round(sec, 4),
                "h_tgt_conf_drop": round(float(hc["tgt_conf_drop"].mean()), 4),
                "h_comp_conf_drop": round(float(hc["comp_conf_drop"].mean()), 4),
                "z_tgt_conf_drop": round(float(zc["tgt_conf_drop"].mean()), 4),
                "z_comp_conf_drop": round(float(zc["comp_conf_drop"].mean()), 4)})
            v = out[key]
            print(f"{'  conf':<22} | {v['h_conf_sel']:+6.3f} {v['z_conf_sel']:+6.3f} | {v['z_minus_h_conf']:+7.3f} {sec:6.3f}")
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
