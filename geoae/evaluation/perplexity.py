"""
WikiText-2 perplexity under the AE / SAE reconstruction splice: the "splice cost"
that sits next to the MMLU-under-splice number (geoae.evaluation.mmlu).

Protocol: wikitext-2-raw-v1 test, "\\n\\n".join(rows) (GPTQ/AWQ convention), tokenised
once without special tokens and cut into consecutive non-overlapping slices of
seq_len-1 text tokens (the final partial slice is kept); each model input is
[BOS] + slice, so every text token is scored exactly once. Conditions on the same
chunks: clean; per item `spliced` (layer output replaced by the reconstruction at
positions 1..T-1, BOS keeps its true state) and `spliced_all` (every position,
as mmlu_splice does); and one shared `zero_ablation` (positions 1..T-1 set to 0)
for the SAEBench-style loss_recovered = (ce_zero - ce_spliced) / (ce_zero - ce_clean).
Reported: ce, ppl, bits/byte and word-level ppl (cross-tokenizer comparable),
delta_ce with a 95% chunk-bootstrap CI, KL(clean || spliced) and top-1 agreement.

    scripts/py -u -m geoae.evaluation.perplexity --ae results/.../splice_ppl.json=<ckpt> \\
        --sae out_sae.json=scope:16k --max_chunks 4
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- pure helpers

def make_chunks(ids: list[int], bos: int | None, seq_len: int, max_chunks: int | None = None) -> list[list[int]]:
    """Non-overlapping slices of seq_len-1 text tokens, each prefixed with BOS (partial tail kept).

    A tokenizer without a BOS token (bos=None, e.g. Qwen2.5) gets the text token before the slice as the prefix instead:
    the prefix is unscored context in both cases, and every text token after the very first is scored exactly once."""
    step = seq_len - 1
    assert step >= 1
    if bos is None:
        chunks = [list(ids[s - 1:s + step]) for s in range(1, len(ids), step)]
    else:
        chunks = [[bos] + list(ids[s:s + step]) for s in range(0, len(ids), step)]
    return chunks[:max_chunks] if max_chunks else chunks


def summarise(total_nll: float, n_tokens: int, n_bytes: int, n_words: int) -> dict:
    ce = total_nll / n_tokens
    return {"ce": ce, "ppl": math.exp(ce),
            "bits_per_byte": total_nll / (n_bytes * math.log(2)),
            "word_ppl": math.exp(total_nll / n_words)}


def bootstrap_ratio_ci(diff_per_chunk, tok_per_chunk, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of sum(diff)/sum(tokens) resampling chunks with replacement."""
    d, t = np.asarray(diff_per_chunk, float), np.asarray(tok_per_chunk, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    r = d[idx].sum(1) / t[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def keep_bos(recon_fn):
    """Wrap a reconstruction so position 0 keeps the original state; recon_fn sees the full tensor."""
    import torch

    def fn(hs):
        out = recon_fn(hs)
        return torch.cat([hs[:, :1], out[:, 1:].to(hs.dtype)], 1)
    return fn


def zero_non_bos(hs):
    out = hs.clone()
    out[:, 1:] = 0
    return out


def r6(x):
    return None if x is None else round(float(x), 6)

# ------------------------------------------------------------------ model side

def block_metrics(logits_c, logits_s, labels, block: int = 256):
    """(nll_c_sum, nll_s_sum, kl_sum, top1_count) over rows; logits (T-1, V), labels (T-1,). Row-blocked fp32."""
    import torch
    nc = ns = kl = 0.0
    top1 = 0
    for s in range(0, labels.shape[0], block):
        lc = torch.log_softmax(logits_c[s:s + block].float(), -1)
        ls = torch.log_softmax(logits_s[s:s + block].float(), -1)
        y = labels[s:s + block, None]
        nc += float(-lc.gather(-1, y).sum())
        ns += float(-ls.gather(-1, y).sum())
        kl += float((lc.exp() * (lc - ls)).sum())
        top1 += int((lc.argmax(-1) == ls.argmax(-1)).sum())
        del lc, ls
    return nc, ns, kl, top1


def block_nll(logits, labels, block: int = 256) -> float:
    import torch
    tot = 0.0
    for s in range(0, labels.shape[0], block):
        lp = torch.log_softmax(logits[s:s + block].float(), -1)
        tot += float(-lp.gather(-1, labels[s:s + block, None]).sum())
        del lp
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ae", action="append", default=[], metavar="OUT=CKPT", help="AE checkpoint and result JSON (repeatable)")
    ap.add_argument("--sae", action="append", default=[], metavar="OUT=SPEC",
                    help="SAE spec (scope:16k | matched:<last.pt>) and result JSON (repeatable)")
    ap.add_argument("--model", default=None, help="LM; required without --ae, must match the checkpoints otherwise")
    ap.add_argument("--layer", type=int, default=None, help="splice layer; required without --ae")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_chunks", type=int, default=None, help="only the first N chunks (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    def split(s):
        if "=" not in s:
            ap.error(f"expected OUT=SPEC, got {s!r}")
        o, sp = s.split("=", 1)
        return o, sp
    ae_items, sae_items = [split(s) for s in args.ae], [split(s) for s in args.sae]
    if not ae_items and not sae_items:
        ap.error("give at least one --ae or --sae item")

    import torch
    import numpy as np
    from datasets import load_dataset
    from geoae.checkpoint import load_tokenizer
    from geoae.checkpoint import load_ae_checkpoint, load_lm
    from geoae.hooks import SplicingHook, TokenIdTap
    import geoae.evaluation.interventions as neuronlens
    from geoae.seeding import seed_everything

    seed_everything(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name, layer = args.model, args.layer
    if ae_items:
        found = set()
        for _, ck_path in ae_items:
            _ae, _m, _s, ck = load_ae_checkpoint(ck_path, "cpu", allow_token_bias=True)
            found.add((ck["config"]["extraction"]["model_name"], ck["config"]["data"]["target_layer"]))
            del _ae, _m, _s, ck
        if len(found) != 1:
            ap.error(f"--ae checkpoints disagree on (model, layer): {sorted(found)}")
        (ck_model, ck_layer), = found
        if (args.model and args.model != ck_model) or (args.layer is not None and args.layer != ck_layer):
            ap.error(f"--model/--layer ({args.model}, {args.layer}) do not match the checkpoints ({ck_model}, {ck_layer})")
        model_name, layer = ck_model, ck_layer
    if model_name is None or layer is None:
        ap.error("--model and --layer are required when there is no --ae item")

    tok = load_tokenizer(model_name)
    bos = tok.bos_token_id      # None (Qwen2.5): each chunk is prefixed with the preceding text token, see make_chunks
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = make_chunks(ids, bos, args.seq_len, args.max_chunks)
    n_tokens = sum(len(c) - 1 for c in chunks)
    if args.max_chunks and len(chunks) * (args.seq_len - 1) < len(ids):
        scored = [t for c in chunks for t in c[1:]]
        scored_text = tok.decode(scored)
    else:
        scored_text = text if bos is not None else tok.decode(ids[1:])      # without BOS the first text token is context only
    n_bytes = len(scored_text.encode("utf-8"))
    n_words = len(re.split(r"\s+", scored_text))
    print(f"[ppl] {model_name} layer {layer}: {len(chunks)} chunks, {n_tokens} tokens, {n_bytes} bytes, {n_words} words")

    lm = load_lm(model_name, device_map="auto")
    lm_dev = lm.get_input_embeddings().weight.device
    hook, tap = SplicingHook(lm, layer), TokenIdTap(lm)
    tok_per_chunk = np.array([len(c) - 1 for c in chunks], float)
    xs = [torch.tensor([c], device=lm_dev) for c in chunks]
    labels = [x[0, 1:] for x in xs]

    @torch.no_grad()
    def logits_of(x, fn=None):
        if fn is not None:
            hook.activate(fn)
        try:
            return lm(input_ids=x).logits[0, :-1]
        finally:
            hook.deactivate()

    def meta_common(spec, kind, bypass):
        return {"dataset": "wikitext-2-raw-v1 test (Salesforce/wikitext)",
                "protocol": ("non-overlapping chunks of seq_len-1 text tokens, each prefixed with BOS; "
                             "every text token scored once; splice at the layer output" if bos is not None else
                             "no BOS token: non-overlapping chunks of seq_len-1 text tokens, each prefixed with the text "
                             "token before it (unscored context, kept unspliced in `spliced`); every text token after "
                             "the first scored once; splice at the layer output"),
                "seq_len": args.seq_len, "n_chunks": len(chunks), "n_tokens": n_tokens, "n_bytes": n_bytes,
                "n_words": n_words, "model": model_name, "layer": layer, "spec": spec, "kind": kind,
                "token_bias": bool(bypass), "max_chunks": args.max_chunks}

    try:
        # pass 1: clean + zero ablation per-chunk NLL sums
        clean_nll, zero_nll = [], []
        for i, x in enumerate(xs):
            clean_nll.append(block_nll(logits_of(x), labels[i]))
            zero_nll.append(block_nll(logits_of(x, zero_non_bos), labels[i]))
            if i % 20 == 0:
                print(f"[ppl] pass1 chunk {i + 1}/{len(xs)}", flush=True)
        clean_nll, zero_nll = np.array(clean_nll), np.array(zero_nll)
        assert np.isfinite(clean_nll).all() and np.isfinite(zero_nll).all(), "non-finite clean/zero NLL"
        clean_s = summarise(clean_nll.sum(), n_tokens, n_bytes, n_words)
        zero_s = summarise(zero_nll.sum(), n_tokens, n_bytes, n_words)
        print(f"[ppl] clean ppl {clean_s['ppl']:.4f}  zero-ablation ppl {zero_s['ppl']:.2f}")

        # identity guard on the first chunk
        ident = block_nll(logits_of(xs[0], lambda hs: hs), labels[0])
        assert abs(ident - clean_nll[0]) <= 1e-5 * abs(clean_nll[0]), \
            f"identity splice changed CE: {ident} vs {clean_nll[0]}"

        items = [("ae", o, s) for o, s in ae_items] + [("sae", o, s) for o, s in sae_items]
        first = True
        for kind, out_path, spec in items:
            if kind == "ae":
                ae, mean, std, _ck = load_ae_checkpoint(spec, dev, allow_token_bias=True)
                ae.eval()
                bypass = ae.has_token_bias
                recon = neuronlens.make_z_gate(np.zeros(1), np.zeros(1), np.zeros(1), ae, mean, std, dev,
                                               gate=False, tap=tap)
            else:
                from geoae.baselines.sae_space import load_sae_space
                sae = load_sae_space(spec, dev)
                bypass = sae.needs_tokens

                @torch.no_grad()
                def recon(hs, sae=sae):
                    B, T, D = hs.shape
                    tk = tap.ids_for(hs) if sae.needs_tokens else None
                    return sae.reconstruct_raw(hs.reshape(B * T, D).float(), tk).reshape(B, T, D).to(hs.device, hs.dtype)
            res = {}
            conds = {"spliced": (keep_bos(recon), "non-BOS"), "spliced_all": (recon, "all incl. BOS")}
            for name, (fn, posdesc) in conds.items():
                nll, kls, t1 = [], [], []
                for i, x in enumerate(xs):
                    lc = logits_of(x)
                    ls = logits_of(x, fn)
                    nc, ns, kl, top1 = block_metrics(lc, ls, labels[i])
                    del lc, ls
                    nll.append(ns); kls.append(kl); t1.append(top1)
                    if i == 0 and first and name == "spliced":
                        assert abs(ns - nc) > 1e-6 * abs(nc), "splice did not change the output"
                    if i % 20 == 0:
                        print(f"[ppl] {spec} {name} chunk {i + 1}/{len(xs)}", flush=True)
                nll = np.array(nll)
                assert np.isfinite(nll).all(), f"non-finite spliced NLL ({name})"
                s = summarise(nll.sum(), n_tokens, n_bytes, n_words)
                lo, hi = bootstrap_ratio_ci(nll - clean_nll, tok_per_chunk, 2000, args.seed)
                s.update({"delta_ce": (nll.sum() - clean_nll.sum()) / n_tokens, "delta_ce_ci95": [lo, hi],
                          "ppl_ratio": s["ppl"] / clean_s["ppl"],
                          "loss_recovered": (zero_s["ce"] - s["ce"]) / (zero_s["ce"] - clean_s["ce"]),
                          "kl_from_clean": sum(kls) / n_tokens, "top1_agreement": sum(t1) / n_tokens,
                          "positions": posdesc})
                res[name] = s
                first = False if name == "spliced" else first
            del recon
            if kind == "ae":
                del ae
            else:
                del sae
            torch.cuda.empty_cache()

            def rd(d):
                return {k: ([r6(u) for u in v] if isinstance(v, list) else r6(v) if isinstance(v, float) else v)
                        for k, v in d.items()}
            out = {"meta": meta_common(spec, kind, bypass), "clean": rd(clean_s), "zero_ablation": rd(zero_s),
                   "spliced": rd(res["spliced"]), "spliced_all": rd(res["spliced_all"])}
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(out_path, "w"), indent=1)
            sp = res["spliced"]
            print(f"[ppl] {spec}  clean ppl {clean_s['ppl']:.3f}  spliced {sp['ppl']:.3f} "
                  f"({(sp['ppl_ratio'] - 1) * 100:+.1f}%)  ΔCE {sp['delta_ce']:+.4f} "
                  f"[{sp['delta_ce_ci95'][0]:+.4f}, {sp['delta_ce_ci95'][1]:+.4f}] nats  "
                  f"bpb {clean_s['bits_per_byte']:.4f}->{sp['bits_per_byte']:.4f}  "
                  f"LR {sp['loss_recovered']:.3f}  | all-pos ppl {res['spliced_all']['ppl']:.3f}  -> {out_path}")
    finally:
        hook.deactivate()
        tap.remove()


if __name__ == "__main__":
    main()
