"""
Generation-level concept steering with EXTRA direction handles, next to concept_steer_generate.py's own.

concept_steer_generate.py steers 32-token continuations of neutral prompts toward a NAMED concept with handles built
from two codebooks (balanced k-means on the residual = base, the token-bypass AE = bypass) plus a supervised direction
(label). This script reruns the direction handles of that test (label, base_dir, bypass_dir: built by its own
run_dataset, unchanged) and adds, on the same documents, prompts, alphas and edit sizes:

  sae_dir      the SAE's OWN features: the top_k features by class-balanced precision of "feature active" over the
               concept's FIT tokens (same eligibility as clusters: >= min_count class tokens from >= min_docs documents),
               direction = their decoder rows weighted by class-token count
  sae_f1_dir   the single best such feature's decoder row
  sae_km_dir   clusters of a k-means codebook fitted on the SAE's code vectors (eval_out/sae_plain_kmeans_fit.py), built
               exactly as base_dir / bypass_dir: top_k class-informative non-hub clusters, target = their member means on
               the general reference rows (in normalised residual space), minus the mean target over concepts
  <name>_dir   the same cluster construction for any extra AE codebook (--extra_ae name=checkpoint, e.g. the no-bypass AE)

Every handle is a constant direction added at every position of the UNEDITED model, x += alpha * R_c * d / ||d||
(concept_steer_generate.make_label_edit_fn), with the same R_c per concept (the base handle's typical gap), so all
handles move the residual by the same amount at a given alpha and nothing is spliced in. Output has
concept_steer_generate's schema, so concept_steer_judge.py scores it as is.

    python -u -m geoae.interp.concept_steer_extra --gen results/delta/g4b_l33_d5120/concept/concept_steer_generate.json \
        --activations_dir activations_gemma3_4b_sampled_struct5_10M --datasets db14 --sae scope:16k \
        --sae_kmeans results/delta/g4b_l33_d5120/evals/sae_suite/controls/plain_kmeans_gs16k.npz \
        --extra_ae nobypass=<checkpoint> --out results/delta/g4b_l33_d5120/concept/concept_steer_generate_sae_db14.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.hooks import SplicingHook
from geoae.interp import concept_steer_generate as CG
from geoae.interp._context_logits import validate_layer
from geoae.interp import _shared as shared
from geoae.interp._shared import DATASET_CONFIGS
from geoae.interp.cluster_steer_generate import PROMPTS
from geoae.interp.cluster_steering import Arm

OLD = ("label", "base_dir", "bypass_dir")
_hub_loader = CG.load_concept_docs


def load_concept_docs(cfg: dict, n_fit: int, n_held: int, seed: int, smoke: bool):
    """concept_steer_generate.load_concept_docs, plus the frozen local corpora (FineWeb-Atlas draws atlas_s0..2 and the
    control datasets: cfg['local_parquet']). With a split column the fit documents come from the 'fit' split and the held
    ones from 'test'; either way the rows are shuffled with `seed` first, so the choice is deterministic. Same return
    value: ({class: fit texts followed by held texts}, kept classes)."""
    if not cfg.get("local_parquet"):
        return _hub_loader(cfg, n_fit, n_held, seed, smoke)
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=cfg["local_parquet"], split="train").shuffle(seed=seed)
    classes, label_col, text_col, split_col = cfg["classes"], cfg.get("label_column", "label"), cfg["text_column"], cfg.get("split_column")
    fit, held = {c: [] for c in range(len(classes))}, {c: [] for c in range(len(classes))}
    for ex in ds:
        c, t = int(ex[label_col]), (ex[text_col] or "").strip()
        if not t:
            continue
        if split_col:
            pool, cap = (fit, n_fit) if ex[split_col] == "fit" else (held, n_held)
        else:
            pool, cap = (fit, n_fit) if len(fit[c]) < n_fit else (held, n_held)
        if len(pool[c]) < cap:
            pool[c].append(t)
    kept = [c for c in range(len(classes)) if len(fit[c]) >= n_fit and len(held[c]) >= n_held]
    dropped = [c for c in range(len(classes)) if c not in kept]
    if dropped:
        print("[concept]   dropping classes with too few documents: "
              + ", ".join(f"{classes[c]} ({len(fit[c])} fit, {len(held[c])} held)" for c in dropped))
    return {c: fit[c] + held[c] for c in range(len(classes))}, (kept[:2] if smoke else kept)


CG.load_concept_docs = load_concept_docs          # run_dataset resolves the name at call time
INHERIT = ("base", "bypass", "n_fit", "n_held", "max_doc_tokens", "top_k", "min_count", "min_docs", "hub_x", "alphas",
           "max_new", "layer", "val_frac", "n_eval", "chunk", "seed")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="existing concept_steer_generate.json: its settings are inherited")
    ap.add_argument("--activations_dir", required=True)
    ap.add_argument("--datasets", default="db14")
    ap.add_argument("--sae", default="", help="SAE spec (geoae/interp/sae_space.py), e.g. scope:16k")
    ap.add_argument("--sae_kmeans", default="", help="k-means centroids on the SAE codes (.npz, centroids)")
    ap.add_argument("--extra_ae", default="", help="name=checkpoint,...: extra AE codebooks -> <name>_dir")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    G0 = json.load(open(cli.gen))
    args = SimpleNamespace(**{k: G0["meta"][k] for k in INHERIT}, handles=list(OLD), smoke=cli.smoke)
    if cli.smoke:
        args.n_fit, args.n_held, args.n_eval, args.alphas = 30, 10, 100_000, [0.3, 1.0]
    dev = torch.device("cuda")
    extra_ae = dict(x.split("=", 1) for x in cli.extra_ae.split(",") if x)

    act = Path(cli.activations_dir)
    meta = json.load(open(act / "meta.json"))
    tk = AutoTokenizer.from_pretrained(meta["model"])
    tk.pad_token, tk.padding_side = tk.eos_token, "left"
    lm = load_lm(meta["model"], device=dev)
    downstream_blocks = validate_layer(lm, args.layer)
    arms = {"base": Arm("base", args.base, dev), "bypass": Arm("bypass", args.bypass, dev)}
    mean, std = arms["base"].mean, arms["base"].std
    assert torch.allclose(arms["bypass"].mean, mean) and torch.allclose(arms["bypass"].std, std)
    D = mean.shape[0]

    # ---- extra codebooks: name -> (assign(h_raw, tok) -> cluster ids, K)
    sae = Cs = None
    if cli.sae:
        from geoae.interp.sae_space import load_sae_space
        sae = load_sae_space(cli.sae, dev, "sae")
    books = {}
    if cli.sae_kmeans:
        Cs = torch.as_tensor(np.load(cli.sae_kmeans)["centroids"], dtype=torch.float32, device=dev)
        books["sae_km"] = (lambda h, tok: torch.cdist(sae.encode_raw(h, None), Cs).argmin(1), Cs.shape[0])
    for name, path in extra_ae.items():
        a = Arm(name, path, dev)
        books[name] = ((lambda h, tok, a=a: a.assign((h - a.mean) / a.std, tok)), a.C.shape[0])

    # ---- general reference rows (even documents of the held-out tail), as concept_steer_generate.main
    X = np.load(str(act / f"layer_{args.layer}.npy"), mmap_mode="r")
    N = len(X)
    ev = np.arange(int(N * (1 - args.val_frac)), N)[: args.n_eval]
    doc = np.load(act / "rows_doc.npy", mmap_mode="r")[ev]
    keep = doc % 2 == 0
    tail = np.asarray(X[ev[0]: ev[-1] + 1])                     # one contiguous read (GPFS has no page cache)
    H = torch.from_numpy(tail[keep]).to(dev)
    del tail
    tok_ref = torch.from_numpy(np.asarray(np.load(act / "rows_tok.npy", mmap_mode="r")[ev][keep], dtype=np.int64)).to(dev)
    print(f"[extra] {len(H):,} reference rows (even docs of the held-out tail)", flush=True)

    M, hub, lab_ref = {}, {}, {}
    ref_books = {"base": (lambda h, tok: arms["base"].assign((h - mean) / std, tok), arms["base"].C.shape[0]),
                 "bypass": (lambda h, tok: arms["bypass"].assign((h - mean) / std, tok), arms["bypass"].C.shape[0]), **books}
    for name, (assign, K) in ref_books.items():
        lab = torch.empty(len(H), dtype=torch.long, device=dev)
        Msum = torch.zeros(K, D, device=dev)
        for i in range(0, len(H), args.chunk):
            h = H[i:i + args.chunk].float()
            l = assign(h, tok_ref[i:i + args.chunk])
            lab[i:i + len(l)] = l
            Msum.index_add_(0, l, (h - mean) / std)
        cnt = torch.bincount(lab, minlength=K).float()
        M[name], hub[name], lab_ref[name] = Msum / cnt.clamp_min(1)[:, None], cnt / cnt.sum() > args.hub_x / K, lab
        print(f"[extra] {name}: K={K} hubs {int(hub[name].sum())} ({float(cnt[hub[name]].sum() / cnt.sum()):.1%} of rows), "
              f"live {int((cnt > 0).sum())}", flush=True)
    del H, tok_ref
    torch.cuda.empty_cache()

    ds_names = cli.datasets.split(",")
    for n in ds_names:                                   # atlas draws / control corpora register their config lazily
        if n not in DATASET_CONFIGS:
            shared.set_dataset(n)
    sanity = {"abs_sum": 0.0, "n": 0, "sum_dim": torch.zeros(D, device=dev), "sumsq_dim": torch.zeros(D, device=dev)}
    results, extra_names = {}, []
    for ds_name in ds_names:
        cfg = DATASET_CONFIGS[ds_name]
        classes, n_classes = cfg["classes"], len(cfg["classes"])
        res = CG.run_dataset(ds_name, cfg, lm, tk, arms, mean, std, D, dev, args, M, hub, lab_ref, sanity)
        if res is None:
            continue
        # ---- extras: one more pass over the SAME fit documents (load_concept_docs is deterministic in the seed)
        by_class, kept = CG.load_concept_docs(cfg, args.n_fit, args.n_held, args.seed, args.smoke)
        assert kept == res["kept"]
        fit_cls, fit_doc, fit_k = [], [], {n: [] for n in books}
        F = sae.n_features if sae is not None else 0
        n_cf = torch.zeros(n_classes, F, device=dev)
        nd_cf = torch.zeros(n_classes, F, device=dev)
        off = 0
        tk.padding_side = "right"
        for c in kept:
            texts = by_class[c][: args.n_fit]
            for h_raw, tok_ids, doc_local in CG.capture_docs(lm, tk, texts, args.layer, dev, args.max_doc_tokens):
                fit_cls.append(np.full(len(h_raw), c, dtype=np.int64))
                fit_doc.append((doc_local + off).cpu().numpy())
                for n, (assign, _) in books.items():
                    fit_k[n].append(assign(h_raw, tok_ids).cpu().numpy())
                if sae is not None:
                    on = sae.encode_raw(h_raw, None) > 0
                    n_cf[c] += on.sum(0)
                    for d_ in doc_local.unique():
                        nd_cf[c] += on[doc_local == d_].any(0)
            off += len(texts)
        tk.padding_side = "left"
        fit_cls, fit_doc = np.concatenate(fit_cls), np.concatenate(fit_doc)
        handles, targets = res["handles"], res["targets"]
        R = {cn: targets[cn]["label"][1] for cn in handles if targets[cn].get("label") is not None}
        vlab = {cn: targets[cn]["label"][0] for cn in R}

        def add(name, cn, d, info):
            if name not in extra_names:
                extra_names.append(name)
            if d is None or cn not in R or float(d.norm()) < 1e-8:
                targets[cn][name], handles[cn][name] = None, None
                res["fails"].append({"dataset": ds_name, "concept": cn, "handle": name, "reason": "no eligible unit"})
                return
            targets[cn][name] = (d.clone(), R[cn])
            handles[cn][name] = {**info, "R_c": R[cn], "raw_norm": float(d.norm()), "cos_label": CG.cosine(d, vlab[cn])}

        for n, (_, K) in books.items():                      # cluster codebooks -> <n>_dir, as base_dir / bypass_dir
            n_ck, nd_ck, p_ck = CG.fit_class_cluster_stats(fit_cls, fit_doc, np.concatenate(fit_k[n]), n_classes, K)
            tgt, info = {}, {}
            for c in kept:
                order, w = CG.select_handle_clusters(n_ck[c], nd_ck[c], p_ck[c], hub[n].cpu().numpy(),
                                                     args.min_count, args.min_docs, args.top_k)
                cn = classes[c]
                if len(order) == 0:
                    tgt[cn] = None
                    continue
                tgt[cn] = (torch.as_tensor(w, device=dev, dtype=torch.float32)[:, None]
                           * M[n][torch.as_tensor(order, device=dev)]).sum(0)
                info[cn] = {"clusters": order.tolist(), "weights": w.tolist(), "p_ck": p_ck[c, order].tolist(),
                            "n_ck": n_ck[c, order].tolist()}
            dirs = CG.contrast_direction(tgt)
            for c in kept:
                add(f"{n}_dir", classes[c], dirs.get(classes[c]), info.get(classes[c], {}))
        if sae is not None:                                   # the SAE's own features -> sae_dir (top_k), sae_f1_dir (top 1)
            N_c = torch.as_tensor(np.maximum(np.bincount(fit_cls, minlength=n_classes), 1), device=dev).float()
            f_cf = n_cf / N_c[:, None]
            p_cf = f_cf / f_cf.sum(0, keepdim=True).clamp_min(1e-12)
            W = sae.decoder_directions.float()                # (F, D) raw residual units
            for c in kept:
                ok = (n_cf[c] >= args.min_count) & (nd_cf[c] >= args.min_docs)
                score = torch.where(ok, p_cf[c], torch.full_like(p_cf[c], -1.0))
                top = score.argsort(descending=True)[: args.top_k]
                top = top[score[top] >= 0]
                for name, ids in (("sae_dir", top), ("sae_f1_dir", top[:1])):
                    if len(ids) == 0:
                        add(name, classes[c], None, {})
                        continue
                    w = n_cf[c, ids] / n_cf[c, ids].sum()
                    add(name, classes[c], ((w[:, None] * W[ids]).sum(0)) / std,      # raw direction -> normalised space
                        {"features": ids.tolist(), "weights": w.tolist(), "p_cf": p_cf[c, ids].tolist(),
                         "n_cf": n_cf[c, ids].tolist(), "active_share_class": f_cf[c, ids].tolist()})
        for name in extra_names:
            cs = [handles[cn][name]["cos_label"] for cn in handles if handles[cn].get(name)]
            print(f"[extra] {ds_name}: {name}: {len(cs)}/{len(handles)} concepts"
                  + (f", cos(label) mean {np.mean(cs):.3f} [{np.min(cs):.3f}, {np.max(cs):.3f}]" if cs else ""), flush=True)
        results[ds_name] = res

    # ---- generation (concept_steer_generate.main's loop; every handle here is a constant direction)
    n_prompts = 4 if cli.smoke else 20
    prompts = PROMPTS[:n_prompts]
    assert prompts == G0["meta"]["prompts"][:n_prompts], "prompt list differs from the inherited run"
    hook = SplicingHook(lm, args.layer)
    tk.padding_side = "left"
    enc = tk(prompts, return_tensors="pt", padding=True).to(dev)
    L = enc["input_ids"].shape[1]

    def generate(fn=None):
        if fn is not None:
            hook.activate(fn)
        try:
            out = lm.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tk.pad_token_id)
        finally:
            if fn is not None:
                hook.deactivate()
        return out[:, L:]

    def ppl(cont):
        ids = torch.cat([enc["input_ids"], cont], 1)
        am = torch.cat([enc["attention_mask"], (cont != -1).long()], 1)
        logits = lm(input_ids=ids, attention_mask=am).logits[:, L - 1:-1].float()
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), cont, reduction="none").mean(1).exp().cpu().numpy()

    base_cont = generate()
    base_ppl = ppl(base_cont)
    base_ids = base_cont.cpu().numpy()
    unsteered_text = [tk.decode(r, skip_special_tokens=True) for r in base_ids]
    unsteered_d2 = [CG.distinct2(list(r)) for r in base_ids]
    same0 = np.mean([a == b for a, b in zip(unsteered_text, G0["generations"]["unsteered"]["text"])])
    print(f"[extra] unsteered: median ppl {np.median(base_ppl):.2f}; identical to the inherited run on {same0:.0%} of prompts",
          flush=True)

    all_handles = list(OLD) + extra_names
    gens = {}
    for ds_name, res in results.items():
        gens[ds_name] = {cn: {} for cn in res["handles"]}
        for hname in all_handles:
            avail = [cn for cn in res["handles"] if res["handles"][cn].get(hname) is not None]
            for cn in avail:
                gens[ds_name][cn][hname] = {"0.0": {"text": unsteered_text, "ppl": base_ppl.tolist(),
                                                    "log_ppl_ratio": [0.0] * n_prompts, "distinct2": unsteered_d2,
                                                    "edit_norm": 0.0}}
            for a in args.alphas:
                cell_lr, same = [], []
                for cn in avail:
                    log = []
                    v_c, R_c = res["targets"][cn][hname]
                    cont = generate(CG.make_label_edit_fn(v_c, R_c, mean, std, a, log))
                    pp = ppl(cont)
                    ids = cont.cpu().numpy()
                    texts = [tk.decode(r, skip_special_tokens=True) for r in ids]
                    lr = (np.log(pp) - np.log(base_ppl)).tolist()
                    gens[ds_name][cn][hname][str(a)] = {"text": texts, "ppl": pp.tolist(), "log_ppl_ratio": lr,
                                                        "distinct2": [CG.distinct2(list(r)) for r in ids],
                                                        "edit_norm": float(np.mean(log)) if log else 0.0}
                    cell_lr += lr
                    old = G0["generations"]["steered"].get(ds_name, {}).get(cn, {}).get(hname, {}).get(str(a))
                    if old is not None:
                        same += [x == y for x, y in zip(texts, old["text"])]
                print(f"[extra] {hname:12s} {ds_name:9s} a{a:<4} ppl x{np.exp(np.mean(cell_lr)):.2f} ({len(avail)} concepts)"
                      + (f"  identical to the inherited run: {np.mean(same):.0%}" if same else ""), flush=True)

    out = {
        "meta": {**vars(args), "handles": all_handles, "inherited_from": cli.gen, "sae": cli.sae, "sae_kmeans": cli.sae_kmeans,
                 "extra_ae": extra_ae, "datasets": cli.datasets, "activations_dir": cli.activations_dir, "prompts": prompts,
                 "downstream_blocks": downstream_blocks, "intervention_forward": "complete_lm_at_selected_block",
                 "concept_descriptions": {ds: {n: CG.concept_description(ds, n) for n in DATASET_CONFIGS[ds]["classes"]}
                                          for ds in ds_names}},
        "handles": {ds: r["handles"] for ds, r in results.items()},
        "coverage": {ds: r["coverage"] for ds, r in results.items()},
        "coverage_failures": [f for r in results.values() for f in r["fails"]],
        "generations": {"unsteered": {"text": unsteered_text, "ppl": base_ppl.tolist(), "distinct2": unsteered_d2},
                        "steered": gens},
    }
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[extra] wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
