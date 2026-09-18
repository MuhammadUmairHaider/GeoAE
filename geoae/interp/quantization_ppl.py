"""
What does the discrete code COST the model? Perplexity under splicing.

Every other eval here asks whether clusters line up with labels a human chose.
This asks the model instead: replace each token's residual with what the
clustering says it is, run the LM forward, and measure the damage in perplexity.
It is the implicit claim of a K=2000 partition -- that ~11 bits is a sufficient
statistic for what the token is doing -- stated as something falsifiable.

Four arms, each splicing the layer-L residual and re-running the full forward:

  base        untouched                     the floor
  recon       decode(encode(h))             what the AE loses to reconstruction
  quant_ae    decode(centroid[argmin z])    AE + the hard assignment
  quant_km    centroid[argmin h_norm]       the encoder-free control, quantised

recon vs quant_ae separates the two costs: reconstruction error, and the extra
loss from collapsing onto a centroid. quant_ae vs quant_km is the comparison
that matters -- both are K=2000 discrete codes over the same tokens, one with a
learned encoder and one without.

No LLM judge, no labels, no saturated scale, and the floor is the model itself.

    python -m geoae.interp.quantization_ppl --checkpoint <ckpt.pt> \
        --baseline_kmeans <balanced.npz> --n_texts 200
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.hooks import SplicingHook


@torch.no_grad()
def ppl_under(lm, tok, texts, device, hook, fn, max_len=512):
    """Perplexity over `texts` with `fn` splicing the residual (fn=None -> base)."""
    if fn is not None:
        hook.activate(fn)
    try:
        tot_loss, tot_tok = 0.0, 0
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            n = enc["input_ids"].shape[1]
            if n < 8:
                continue
            loss = lm(**enc, labels=enc["input_ids"]).loss
            tot_loss += loss.item() * n
            tot_tok += n
    finally:
        if fn is not None:
            hook.deactivate()
    return float(np.exp(tot_loss / max(tot_tok, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--baseline_kmeans", default=None,
                    help="encoder-free control (.npz). Quantises the RAW residual "
                         "at the same K — the comparison the AE has to beat.")
    ap.add_argument("--n_texts", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--arms", default="all", choices=["all", "recon"],
                    help="'recon' skips the hard-assignment arms. Quantising EVERY "
                         "position replaces ~85%% of the residual (quant FVE ~0.28 vs "
                         "recon 0.97) and is not a realistic intervention — it measures "
                         "the paradigm, not the model.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ae, _, _, ck = load_ae_checkpoint(args.checkpoint, dev)
    ae.eval()
    layer = ck["config"]["data"]["target_layer"]
    model_name = ck["config"]["extraction"]["model_name"]
    mu = torch.tensor(ck["norm_mean"]).to(dev).float()
    sd = torch.tensor(ck["norm_std"]).to(dev).float()
    print(f"[ppl] {model_name} layer {layer}  K={ae.n_clusters} latent={ae.centroids.shape[1]}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(dev).eval()
    hook = SplicingHook(lm, layer)

    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    texts, it = [], iter(ds)
    while len(texts) < args.n_texts:
        t = next(it)["text"]
        if len(t) > 400:
            texts.append(t[:4000])
    print(f"[ppl] {len(texts)} wiki texts, max_len {args.max_len}")

    C_km = mu_km = sd_km = None
    if args.baseline_kmeans:
        d = np.load(args.baseline_kmeans)
        C_km = torch.tensor(d["centroids"]).to(dev).float()
        mu_km = torch.tensor(d["norm_mean"]).to(dev).float()
        sd_km = torch.tensor(d["norm_std"]).to(dev).float()

    # The hook hands over (B, T, D); the AE is a 2-D model and its BatchNorm1d
    # would read T as the channel dim. Flatten to (B*T, D) and restore.
    def _wrap(fn):
        def inner(hs):
            h = hs[0] if isinstance(hs, tuple) else hs
            shape = h.shape
            out = fn(h.reshape(-1, shape[-1])).reshape(shape).to(h.dtype)
            return (out,) + hs[1:] if isinstance(hs, tuple) else out
        return inner

    @_wrap
    def _recon(h):
        return ae.decoder(ae.encoder((h - mu) / sd)) * sd + mu

    @_wrap
    def _quant_ae(h):
        z = ae.encoder((h - mu) / sd)
        q = ae.centroids[torch.cdist(z, ae.centroids).argmin(1)]
        return ae.decoder(q) * sd + mu

    @_wrap
    def _quant_km(h):
        hn = (h - mu_km) / sd_km
        q = C_km[torch.cdist(hn, C_km).argmin(1)]
        return q * sd_km + mu_km

    arms = [("base", None), ("recon  (decode∘encode)", _recon)]
    if args.arms == "all":
        arms.append(("quant_ae  (AE + hard assign)", _quant_ae))
        if C_km is not None:
            arms.append(("quant_km  (no encoder)", _quant_km))

    res = {}
    base = None
    for name, fn in arms:
        p = ppl_under(lm, tok, texts, dev, hook, fn, args.max_len)
        res[name] = p
        if base is None:
            base = p
        print(f"  {name:<32} ppl {p:>10.3f}   x{p/base:>6.2f} vs base")

    if args.out:
        import json
        json.dump({"layer": layer, "model": model_name, "ppl": res},
                  open(args.out, "w"), indent=2)
        print(f"[ppl] wrote {args.out}")


if __name__ == "__main__":
    main()
