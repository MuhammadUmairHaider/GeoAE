"""
Control edits on ONE shared set of documents with one or both reconstructions spliced into the model.

Orders (what the layer output passes through):
    ae       h -> AE                    ae_sae   h -> AE -> SAE
    sae      h -> SAE                   sae_ae   h -> SAE -> AE
The UNEDITED chain is the reference of its order, so every arm of an order starts from the same model, the same documents
and the same confidence. An arm then edits exactly ONE space inside the chain:
    z        the AE latent                       h_first   the residual before the chain
    s        the SAE codes                       h_last    the residual after the chain
    s_top    the SAE codes, top-K features only
In a two-stage order the space that sits second is edited closest to the model (the first space's edit still has to pass
the other reconstruction), which is why both two-stage orders are run; the single-stage orders (ae, sae) have no order at
all but different starting models. All orders share the documents.

Edits are range_intervention_compare's (same neuronlens operators). Coordinates = top --percent by d-prime (s_top: the
--top_k features with the largest d-prime among features that are MORE active on the concept), ranges = class mean +-
tao sd, r = class mean - other-class mean:
    rm_range_comp   in-range values -> other-class mean          st_range_a<A>    in-range values shifted by -A r
    st_global_a<A>  every coordinate shifted by -A r (not for s_top)
Controls (on st_range at the first alpha, key suffix): @rand = the same NUMBER of coordinates drawn at random,
@shuf = class labels permuted among the fit documents before anything is fitted (a null edit).
All statistics are fitted on the activations the arm sees INSIDE the chain (in order sae_ae the AE encodes the SAE's
reconstruction; h_last = the chain's output), at the answer position of the fit documents. Every stage is a position-wise
map of the layer output, so these come from the captured h without a forward pass.

Documents: correct under the base model and under EVERY unedited order -- in the pool
(_shared.build_joint_correct_set, extra_recons) and again in the evaluation re-filter -- with exactly --eval_exact
evaluation documents per class. A class with fewer is dropped (reported). Per run the per-document gold-label confidence,
correctness and prediction are stored.

Built-in checks, run before any edit (the job aborts if one fails; values are stored under meta.checks):
    input     the chain's input at the answer position equals the captured h the statistics are fitted on (relative L2
              error <= 5%, cosine >= 0.995: the bfloat16 model's two passes differ by rounding, a misaligned one by ~100%)
    identity  an identity edit in every (order, arm) reproduces that order's unedited confidences

    python -u -m geoae.interp.chain_control --checkpoint <ae.pt> --sae scope:16k --dataset ag_news --extra_split train \
        --correct_json <pool.json> --out results/delta/g4b_l33_d5120/interventions/chain/chain_control_ag_news.json
A finished concept is never recomputed: rerunning the same command resumes. Table: eval_out/paper_chain_control_table.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from geoae.interp import neuronlens as nl

ORDERS = {"ae": ("ae",), "sae": ("sae",), "ae_sae": ("ae", "sae"), "sae_ae": ("sae", "ae")}
ARM_STAGE = {"z": "ae", "s": "sae", "s_top": "sae"}           # the stage an arm edits; h_first / h_last sit outside
ARM_ACTS = {"h_first": "h_first", "z": "z", "s": "s", "s_top": "s", "h_last": "h_last"}
ALL_ARMS = ("h_first", "z", "s", "s_top", "h_last")
ALL_OPS = ("rm_range", "st_range", "st_global")
CONTROLS = ("rand", "shuf")


def arms_of(order, arms):
    """The arms that exist in `order` (an arm editing a stage the order does not contain is skipped)."""
    return [a for a in arms if a in ("h_first", "h_last") or ARM_STAGE[a] in order]


class Chain:
    """Position-wise composition of the AE and SAE reconstructions with at most one edit.

    ae:  .encode(x_norm, tok) / .decode(z, tok) / .has_token_bias, in the standardised space (x - mean) / std
    sae: .encode_raw(x, None) / .decode_raw(codes, None), in raw residual units; the BOS position passes through it
    """

    def __init__(self, ae, mean, std, sae, bos_id, device):
        self.ae, self.sae, self.bos, self.device = ae, sae, bos_id, device
        self.mean, self.std = mean.float().to(device), std.float().to(device)

    def ae_stage(self, x, tok, edit=None):
        t = tok if getattr(self.ae, "has_token_bias", False) else None
        z = self.ae.encode((x - self.mean) / self.std, t)
        if edit is not None:
            z = edit(z)
        return z, self.ae.decode(z, t) * self.std + self.mean

    def sae_stage(self, x, tok, edit=None):
        c = self.sae.encode_raw(x, None)
        if edit is not None:
            c = edit(c)
        y = self.sae.decode_raw(c, None)
        if self.bos is not None and tok is not None:
            keep = tok == self.bos
            y = torch.where(keep[:, None], x, y)
        return c, y

    def rows(self, x, tok, order, arm=None, edit=None):
        """(N, D) raw layer outputs -> (N, D) chain outputs; `edit` is applied where `arm` sits."""
        stage = {"ae": self.ae_stage, "sae": self.sae_stage}
        if arm == "h_first":
            x = edit(x)
        for st in order:
            _, x = stage[st](x, tok, edit if ARM_STAGE.get(arm) == st else None)
        if arm == "h_last":
            x = edit(x)
        return x

    @torch.no_grad()
    def acts(self, h, tok, order, chunk=256):
        """What each space of `order` holds for raw layer outputs h (N, D): {"h_first", "z"?, "s"?, "h_last"} (numpy)."""
        stage = {"ae": self.ae_stage, "sae": self.sae_stage}
        name = {"ae": "z", "sae": "s"}
        out = {"h_first": [], "h_last": [], **{name[st]: [] for st in order}}
        for i in range(0, len(h), chunk):
            x = torch.as_tensor(h[i:i + chunk]).to(self.device).float()
            t = torch.as_tensor(tok[i:i + chunk]).to(self.device)
            out["h_first"].append(x.cpu())
            for st in order:
                code, x = stage[st](x, t)
                out[name[st]].append(code.cpu())
            out["h_last"].append(x.cpu())
        return {a: torch.cat(v).numpy() for a, v in out.items()}

    def hook_fn(self, order, tap, arm=None, edit=None, record=None):
        """Replacement fn for SplicingHook. `record` (a list) receives the chain INPUT at the last position of the first
        full-prompt forward pass (T > 1), for the input check."""
        @torch.no_grad()
        def fn(hs):
            B, T, D = hs.shape
            x = hs.reshape(B * T, D).float().to(self.device)
            if record is not None and T > 1 and not record:
                record.append(x.reshape(B, T, D)[:, -1].cpu())
            y = self.rows(x, tap.ids_for(hs).to(self.device), order, arm, edit)
            return y.reshape(B, T, D).to(hs.device).to(hs.dtype)
        return fn


def control_rng(seed, concept, order, arm, control):
    return np.random.RandomState(zlib.crc32(f"{seed}|{concept}|{order}|{arm}|{control}".encode()))


def fit_edits(A, is_c, arm, *, percent, top_k, tao, alphas, ops, device, control=None, rng=None, control_ops=("st_range",)):
    """Edits of one arm from its fit activations A (N, d) and the concept mask is_c. -> ({op key: edit fn}, info).
    control: None | "rand" (random coordinates, same count) | "shuf" (labels permuted first); controls build only
    `control_ops` at the first alpha."""
    is_c = np.asarray(is_c, bool)
    if control == "shuf":
        is_c = rng.permutation(is_c)
    a_c, a_o = A[is_c], A[~is_c]
    mu_c, mu_o = a_c.mean(0), a_o.mean(0)
    score = nl.dprime_saliency(A, is_c)
    if arm == "s_top":
        score = np.where(mu_c > mu_o, score, -1.0)
        sal = nl.select_top(score, q=top_k) & (score > 0)
    else:
        sal = nl.select_top(score, p=percent)
    if control == "rand":
        n = int(sal.sum())
        sal = np.zeros(A.shape[1], bool)
        sal[rng.choice(A.shape[1], n, replace=False)] = True
    lo, hi, _, _ = nl.fit_ranges(a_c, sal, tao)
    inf = np.full(A.shape[1], np.inf, dtype=np.float32)
    r = mu_c - mu_o
    use = [o for o in ops if o in control_ops] if control else list(ops)
    al = list(alphas)[:1] if control else list(alphas)
    edits = {}
    if "rm_range" in use:
        edits["rm_range_comp"] = nl.gate_edit(lo, hi, mu_o, device)
    for a in al:
        if "st_range" in use:
            edits[f"st_range_a{float(a)}"] = nl.shift_edit(r, lo, hi, a, device)
        if "st_global" in use and arm != "s_top":
            edits[f"st_global_a{float(a)}"] = nl.shift_edit(r, -inf, inf, a, device)
    info = {"dim": int(A.shape[1]), "n_coords": int(sal.sum()), "coords_active_on_concept": float((a_c[:, sal] != 0).mean()) if sal.any() else 0.0,
            "in_range_concept": float(((a_c >= lo) & (a_c <= hi))[:, sal].mean()) if sal.any() else 0.0,
            "in_range_others": float(((a_o >= lo) & (a_o <= hi))[:, sal].mean()) if sal.any() else 0.0}
    return edits, info


def allocate(n_pool_fit, n_pool_eval, n_fit, n_eval, eval_exact, min_fit, split=False):
    """How many fit / evaluation-candidate documents a class gets from its pool, or None when it cannot supply
    min_fit fit documents AND eval_exact evaluation documents. Without document-level splits both come from one pool
    (n_pool_fit is its size): evaluation candidates are served first so that a short pool still yields a full
    evaluation set, the rest (up to n_fit) is fitted on."""
    if split:
        return (min(n_fit, n_pool_fit), min(n_eval, n_pool_eval)) if n_pool_fit >= min_fit and n_pool_eval >= eval_exact else None
    if n_pool_fit < min_fit + eval_exact:
        return None
    n_ev = min(n_eval, n_pool_fit - min_fit)
    return min(n_fit, n_pool_fit - n_ev), n_ev


def selectivity(ref_conf, run_conf, nt, scale="abs", ti=None, ci=None):
    """(target change, collateral, selectivity) in points. ref / run: per-document gold confidences ordered
    [nt target documents, other-class documents]; collateral = mean loss only; ti / ci: bootstrap index arrays."""
    r, e = np.asarray(ref_conf, float), np.asarray(run_conf, float)
    rt, et, rc, ec = r[:nt], e[:nt], r[nt:], e[nt:]
    if ti is not None:
        rt, et, rc, ec = rt[ti], et[ti], rc[ci], ec[ci]
    tchg, coll = et.mean() - rt.mean(), np.maximum(0, rc - ec).mean()
    if scale == "rel":
        tchg, coll = tchg / rt.mean(), coll / rc.mean()
    return 100 * tchg, 100 * coll, 100 * (-tchg - coll)


@torch.no_grad()
def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from geoae.checkpoint import load_lm
    from geoae.evaluate import load_ae_from_checkpoint, sanity_check_splice
    from geoae.hooks import SplicingHook, TokenIdTap
    from geoae.interp import _shared as shared
    from geoae.interp.sae_space import load_sae_space
    from geoae.seeding import seed_everything

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sae", required=True, help="SAE spec of geoae/interp/sae_space.py, e.g. scope:16k")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--correct_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orders", default="ae,sae,ae_sae,sae_ae")
    ap.add_argument("--arms", default="h_first,z,s,s_top,h_last")
    ap.add_argument("--ops", default="rm_range,st_range,st_global")
    ap.add_argument("--controls", default="rand,shuf", help="null controls on st_range at the first alpha ('' = none)")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5])
    ap.add_argument("--percent", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=20, help="features of the s_top arm (20 = the SAE's sparsity target)")
    ap.add_argument("--tao", type=float, default=2.0)
    ap.add_argument("--n_fit", type=int, default=80)
    ap.add_argument("--min_fit", type=int, default=40, help="a class with fewer fit documents is dropped")
    ap.add_argument("--n_eval", type=int, default=65, help="evaluation CANDIDATES per class (the re-filter drops a few)")
    ap.add_argument("--eval_exact", type=int, default=50, help="evaluation documents kept per class after the re-filter")
    ap.add_argument("--n_comp", type=int, default=80)
    ap.add_argument("--n_ppl", type=int, default=40, help="wikipedia texts for the unedited models' perplexity (0 = skip)")
    ap.add_argument("--ppl_runs", action="store_true", help="also measure perplexity for every edit (slower)")
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--extra_split", default="", help="Hub datasets: a second split appended after the configured one")
    ap.add_argument("--extra_max", type=int, default=20000)
    ap.add_argument("--max_try", type=int, default=None)
    ap.add_argument("--jc_batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    orders = {o: ORDERS[o] for o in args.orders.split(",") if o}
    arms = [a for a in args.arms.split(",") if a]
    ops = [o for o in args.ops.split(",") if o]
    controls = [c for c in args.controls.split(",") if c]
    assert set(arms) <= set(ALL_ARMS) and set(ops) <= set(ALL_OPS) and set(controls) <= set(CONTROLS), (arms, ops, controls)
    dataset_cfg = shared.set_dataset(args.dataset)
    concepts = list(range(len(shared.CLASSES))) if args.concepts == "all" else [int(x) for x in args.concepts.split(",")]
    if args.smoke:
        args.n_fit, args.min_fit, args.n_eval, args.eval_exact, args.n_comp, args.n_ppl = 20, 10, 14, 8, 12, 8
        concepts = concepts[:2]
    seed_everything(args.seed)
    device = torch.device("cuda")

    ae, norm, cfg = load_ae_from_checkpoint(Path(args.checkpoint), device, allow_token_bias=True)
    model_name, layer = cfg["extraction"]["model_name"], cfg["data"]["target_layer"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    lm = load_lm(model_name, device_map="auto")
    sanity_check_splice(lm, layer, tokenizer, device)
    hook, tap = SplicingHook(lm, layer), TokenIdTap(lm)
    sae = load_sae_space(args.sae, device, "sae")
    chain = Chain(ae, norm["mean"], norm["std"], sae, tokenizer.bos_token_id, device)
    print(f"[chain] LM={model_name} layer={layer} sae={args.sae} ({sae.n_features} features) orders={list(orders)} "
          f"arms={arms} ops={ops} controls={controls}", flush=True)

    ppl_txt = []
    if args.n_ppl:
        for ex in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True):
            ppl_txt.append(ex["text"][:1500])
            if len(ppl_txt) >= args.n_ppl:
                break

    if dataset_cfg.get("local_parquet"):
        ds = load_dataset("parquet", data_files=dataset_cfg["local_parquet"], split="train")
    else:
        ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["dataset_split"]).shuffle(seed=args.seed)
        if args.extra_split:
            from datasets import concatenate_datasets
            ds2 = load_dataset(dataset_cfg["dataset_name"], split=args.extra_split).shuffle(seed=args.seed)
            ds = concatenate_datasets([ds, ds2.select(range(min(len(ds2), args.extra_max)))])
    need = args.n_fit + args.n_eval
    has_split = bool(dataset_cfg.get("split_column"))
    split_kw = {"need_by_split": {"fit": args.n_fit, "test": args.n_eval}, "max_try": args.max_try} if has_split else {}
    extra = {o: chain.hook_fn(order, tap) for o, order in orders.items() if o != "ae"}     # "ae" is the builder's own recon
    extra_key = (f"chain:{'+'.join(orders)}|sae={args.sae}"
                 + (f"|extra_split={args.extra_split}:{args.extra_max}" if args.extra_split else "")
                 + ("|smoke" if args.smoke else ""))
    correct = shared.build_joint_correct_set(
        ds, dataset_cfg, tokenizer, lm, device, hook, chain.hook_fn(("ae",), tap), Path(args.correct_json), need,
        args.dataset, model_name=model_name, ae_sha=shared.ae_fingerprint(ae), batch_size=args.jc_batch_size,
        tag="chain", classes=concepts, extra_recons=extra, extra_key=extra_key, **split_kw)

    # ---- documents per class; a class that cannot supply min_fit fit and eval_exact evaluation documents is dropped
    fit_by, eval_by, dropped = {}, {}, {}
    for c in concepts:
        if has_split:
            pf = [d["text"] for d in correct if d["label"] == c and d.get("split") == "fit"]
            pe = [d["text"] for d in correct if d["label"] == c and d.get("split") == "test"]
        else:
            pf = pe = [d["text"] for d in correct if d["label"] == c]
        got = allocate(len(pf), len(pe), args.n_fit, args.n_eval, args.eval_exact, args.min_fit, split=has_split)
        if got is None:
            dropped[shared.CLASSES[c]] = f"pool too small ({len(pf)} fit / {len(pe)} evaluation documents)"
            continue
        if has_split:
            fit_by[c], eval_by[c] = pf[:got[0]], pe[:got[1]]
        else:                                               # evaluation candidates first, then the fit documents
            eval_by[c], fit_by[c] = pf[:got[1]], pf[got[1]:got[1] + got[0]]

    eval_txt = [t for c in eval_by for t in eval_by[c]]
    eval_lab = np.array([c for c in eval_by for _ in eval_by[c]])

    def unedited(order):
        if order is not None:
            hook.activate(chain.hook_fn(order, tap))
        r = shared.predict(lm, tokenizer, eval_txt, eval_lab, device)
        hook.deactivate()
        return r

    ref = {"base": unedited(None), **{o: unedited(order) for o, order in orders.items()}}
    for k, r in ref.items():
        print(f"[chain] unedited {k:7s} on the {len(eval_txt)} candidates: acc {np.mean(r['correct']):.3f} conf {np.mean(r['gold_conf']):.3f}")
    keep, seen = [], {}
    for i in range(len(eval_txt)):
        c = int(eval_lab[i])
        if all(r["correct"][i] for r in ref.values()) and seen.get(c, 0) < args.eval_exact:
            keep.append(i)
            seen[c] = seen.get(c, 0) + 1
    for c in list(eval_by):
        if seen.get(c, 0) < args.eval_exact:
            dropped[shared.CLASSES[c]] = f"only {seen.get(c, 0)} evaluation documents correct under every unedited model"
    kept = [c for c in eval_by if shared.CLASSES[c] not in dropped]
    for name, why in dropped.items():
        print(f"[chain] DROPPED class {name}: {why}")
    if len(kept) < 2:
        sys.exit("[chain] fewer than two classes left")
    keep = [i for i in keep if int(eval_lab[i]) in kept]
    eval_txt = [eval_txt[i] for i in keep]
    eval_lab = eval_lab[keep]
    ref = {k: {f: [v[i] for i in keep] for f, v in r.items()} for k, r in ref.items()}
    idx_by_class = {c: np.where(eval_lab == c)[0] for c in kept}
    fit_txt = [t for c in kept for t in fit_by[c]]
    fit_lab = np.array([c for c in kept for _ in fit_by[c]])
    print("[chain] final set: " + ", ".join(f"{shared.CLASSES[c]} {len(fit_by[c])} fit / {len(idx_by_class[c])} eval" for c in kept), flush=True)

    h_fit, ids_fit = shared.capture_h(lm, tokenizer, fit_txt, layer, device, return_ids=True)
    fit_ok = np.array(shared.predict(lm, tokenizer, fit_txt, fit_lab, device)["correct"])
    print(f"[chain] fit: {int(fit_ok.sum())}/{len(fit_ok)} still correct under the base model")
    acts_fit = {o: chain.acts(h_fit, ids_fit, order) for o, order in orders.items()}

    # ---- built-in checks
    checks = {}
    rec = []
    o0 = next(iter(orders))
    hook.activate(chain.hook_fn(orders[o0], tap, record=rec))
    shared.predict(lm, tokenizer, fit_txt[:16], fit_lab[:16], device)
    hook.deactivate()
    got, cap = rec[0].numpy().astype(np.float64), h_fit[:len(rec[0])].astype(np.float64)
    # The LM runs in bfloat16 (8-bit mantissa) and the two passes differ in caching, so single values differ by a unit in
    # the last place (~1%). A wrong position or document gives a relative error near 1 and a cosine far below 1.
    checks["input_rel_err"] = float((np.linalg.norm(got - cap, axis=1) / np.linalg.norm(cap, axis=1)).max())
    checks["input_min_cos"] = float(((got * cap).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(cap, axis=1))).min())
    print(f"[chain] check input: chain input at the answer position vs captured h over {len(got)} documents: "
          f"max relative L2 error {checks['input_rel_err']:.2e}, min cosine {checks['input_min_cos']:.6f}")
    n_id = min(32, len(eval_txt))
    sub_t, sub_l = eval_txt[:n_id], eval_lab[:n_id]
    checks["identity_max_abs_diff"] = {}
    for o, order in orders.items():
        hook.activate(chain.hook_fn(order, tap))
        r0 = np.array(shared.predict(lm, tokenizer, sub_t, sub_l, device)["gold_conf"])
        hook.deactivate()
        for arm in arms_of(order, arms):
            hook.activate(chain.hook_fn(order, tap, arm, lambda a: a))
            r1 = np.array(shared.predict(lm, tokenizer, sub_t, sub_l, device)["gold_conf"])
            hook.deactivate()
            checks["identity_max_abs_diff"][f"{o}|{arm}"] = float(np.abs(r1 - r0).max())
    worst = max(checks["identity_max_abs_diff"].values())
    print(f"[chain] check identity: an identity edit changes the confidences by at most {worst:.2e} over {len(checks['identity_max_abs_diff'])} (order, arm) pairs")
    if checks["input_rel_err"] > 5e-2 or checks["input_min_cos"] < 0.995 or worst > 1e-3:
        sys.exit(f"[chain] ABORT: a built-in check failed: {checks}")

    ppl0 = {}
    if ppl_txt:
        ppl0["base"] = float(shared.perplexity(lm, tokenizer, ppl_txt, device))
        for o, order in orders.items():
            hook.activate(chain.hook_fn(order, tap))
            ppl0[o] = float(shared.perplexity(lm, tokenizer, ppl_txt, device))
            hook.deactivate()

    sig = {"orders": list(orders), "arms": arms, "ops": ops, "controls": controls, "alphas": [float(a) for a in args.alphas],
           "percent": args.percent, "top_k": args.top_k, "tao": args.tao, "n_comp": args.n_comp, "seed": args.seed,
           "extra_key": extra_key, "eval_exact": args.eval_exact, "kept": kept,
           "eval_docs_sha": zlib.crc32("\x00".join(eval_txt).encode())}
    meta = {**vars(args), "model_name": model_name, "layer": layer, "classes": list(shared.CLASSES), "signature": sig,
            "order_stages": {o: list(v) for o, v in orders.items()}, "dropped_classes": dropped,
            "n_fit_by_class": {str(c): len(fit_by[c]) for c in kept}, "n_eval_by_class": {str(c): len(idx_by_class[c]) for c in kept},
            "unedited": {k: {"acc": float(np.mean(r["correct"])), "conf": float(np.mean(r["gold_conf"])), "ppl": ppl0.get(k)}
                         for k, r in ref.items()},
            "checks": checks, "confidence_def": "softmax probability of the gold label's first token at the first generated position",
            "reference": "each order's unedited chain; all arms of an order share it"}
    for k, v in meta["unedited"].items():
        print(f"[chain] unedited {k:7s} on the final set: acc {v['acc']:.3f} conf {v['conf']:.3f}" + (f" ppl {v['ppl']:.2f}" if v["ppl"] else ""))
    results = {"meta": meta, "concepts": {}}
    out = Path(args.out)
    if out.exists():
        old = json.load(open(out))
        if old.get("meta", {}).get("signature") == sig:
            results["concepts"] = old["concepts"]
            print(f"[chain] resuming: {len(results['concepts'])} concepts already in {out}")
        else:
            print(f"[chain] {out} was written under different settings or documents; starting over")
    out.parent.mkdir(parents=True, exist_ok=True)

    def run(fn, idx):
        hook.activate(fn)
        r = shared.predict(lm, tokenizer, [eval_txt[i] for i in idx], [int(eval_lab[i]) for i in idx], device)
        pp = float(shared.perplexity(lm, tokenizer, ppl_txt, device)) if (ppl_txt and args.ppl_runs) else None
        hook.deactivate()
        return r, pp

    n_runs = sum(len(fit_edits(np.zeros((4, 8), np.float32), np.array([1, 1, 0, 0], bool), a, percent=args.percent, top_k=args.top_k,
                               tao=args.tao, alphas=args.alphas, ops=ops, device="cpu")[0])
                 + len(controls) * (1 if "st_range" in ops else 0)
                 for o, order in orders.items() for a in arms_of(order, arms))
    print(f"[chain] {n_runs} edited runs per concept, {len(kept)} concepts", flush=True)
    for c in kept:
        if str(c) in results["concepts"]:
            continue
        tgt = idx_by_class[c]
        comp_pool = np.where(eval_lab != c)[0]
        comp = np.random.RandomState(args.seed * 1000 + c).choice(comp_pool, min(args.n_comp, len(comp_pool)), replace=False)
        idx = np.concatenate([tgt, comp])
        nt = len(tgt)
        rec_c = {"eval_idx": {"tgt": [int(i) for i in tgt], "comp": [int(i) for i in comp]},
                 "ref": {k: {"doc_conf": [round(float(r["gold_conf"][i]), 4) for i in idx],
                             "doc_correct": [int(bool(r["correct"][i])) for i in idx]} for k, r in ref.items()},
                 "info": {}, "runs": {}}
        print(f"\n=== concept {c} ({shared.CLASSES[c]})  n_tgt={nt} n_comp={len(comp)} ===", flush=True)
        is_c = fit_lab[fit_ok] == c
        for o, order in orders.items():
            b_t = float(np.mean([ref[o]["gold_conf"][i] for i in tgt]))
            b_c = float(np.mean([ref[o]["gold_conf"][i] for i in comp]))
            for arm in arms_of(order, arms):
                A = acts_fit[o][ARM_ACTS[arm]][fit_ok]
                todo = [(None, *fit_edits(A, is_c, arm, percent=args.percent, top_k=args.top_k, tao=args.tao, alphas=args.alphas,
                                          ops=ops, device=device))]
                for ctl in (controls if "st_range" in ops else []):
                    todo.append((ctl, *fit_edits(A, is_c, arm, percent=args.percent, top_k=args.top_k, tao=args.tao,
                                                 alphas=args.alphas, ops=ops, device=device, control=ctl,
                                                 rng=control_rng(args.seed, c, o, arm, ctl))))
                for ctl, edits, info in todo:
                    rec_c["info"][f"{o}|{arm}" + (f"@{ctl}" if ctl else "")] = info
                    for op, e in edits.items():
                        res, pp = run(chain.hook_fn(order, tap, arm, e), idx)
                        conf = [float(x) for x in res["gold_conf"]]
                        tc, cc = float(np.mean(conf[:nt])), float(np.mean(conf[nt:]))
                        key = f"{o}|{arm}|{op}" + (f"@{ctl}" if ctl else "")
                        rec_c["runs"][key] = {
                            "tgt_conf": round(tc, 4), "comp_conf": round(cc, 4),
                            "tgt_acc": float(np.mean(res["correct"][:nt])), "comp_acc": float(np.mean(res["correct"][nt:])),
                            "ppl": pp, "doc_conf": [round(x, 4) for x in conf],
                            "doc_correct": [int(bool(x)) for x in res["correct"]], "doc_pred": list(res["pred"])}
                        print(f"  {o:6s} {arm:7s} {op + ('@' + ctl if ctl else ''):20s} target conf {b_t:.3f}->{tc:.3f}  "
                              f"other {b_c:.3f}->{cc:.3f}  acc {np.mean(res['correct'][:nt]):.2f}/{np.mean(res['correct'][nt:]):.2f}",
                              flush=True)
        results["concepts"][str(c)] = rec_c
        json.dump(results, open(out, "w"))
    results["done"] = True
    json.dump(results, open(out, "w"))
    print(f"\n[chain] saved -> {out}")


if __name__ == "__main__":
    sys.exit(main())
