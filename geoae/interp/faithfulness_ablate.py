"""
Concept-direction ablation faithfulness, following VQLC (Yu, Garg, Ebrahimi Kahou,
Sajjad; arXiv 2602.02726).

THE CLAIM BEING TESTED. If a latent concept encodes task-relevant information in
the task-decision representation, then REMOVING that concept's direction from the
representation should change the model's output a lot. A concept that can be
deleted without consequence was not carrying the decision.

    h' = h - (h . v_hat) v_hat          orthogonal projection, v_hat = concept dir
    confidence change = | P(gold | h) - P(gold | h') |

Reported alongside the two numbers VQLC pairs it with, because a method can win
on confidence change simply by having fewer, broader concepts:

    n_concepts   distinct concepts actually used on the eval set
    active_rate  fraction of eval instances whose assigned concept is one of them

VQLC on Qwen/AG-News: confidence change 0.444, 399 concepts, active rate 0.890;
SAE top-1: 0.338, 1834 concepts, active rate 0.375. The partition wins by
activating a far larger share of examples, not by having more features.

CONCEPT VECTOR, NOT CENTROID. VQLC eq. 6 defines the concept vector as the MEAN
of encoder outputs assigned to a code, computed in a final pass with the frozen
encoder — explicitly NOT the EMA codebook vector, which reflects training history
rather than the current assignment. This follows that: v_k is recomputed from the
fit split, and the ablation direction is decoder(v_k) * norm_std, i.e. the
direction the concept injects into the RAW residual stream.

    python -m geoae.interp.faithfulness_ablate --checkpoint <ckpt.pt> \
        --dataset db14 --layer 27
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.hooks import SplicingHook
from geoae.interp import _shared as shared


class KMeansEnc:
    """Encoder-free baseline: identity encoder, centroids already in normalised space."""

    def __init__(self, npz, device):
        d = np.load(npz)
        self.centroids = torch.tensor(d["centroids"]).float().to(device)
        self.n_clusters = self.centroids.shape[0]
        self.mean = torch.tensor(d["norm_mean"]).float().to(device)
        self.std = torch.tensor(d["norm_std"]).float().to(device)

    def encode(self, h):
        return (h - self.mean) / self.std

    def to_raw_dir(self, v):
        return v * self.std


class AEEnc:
    def __init__(self, ckpt, device):
        ae, _, _, ck = load_ae_checkpoint(ckpt, device)
        ae.eval()
        self.ae = ae
        self.centroids = ae.centroids.detach()
        self.n_clusters = ae.n_clusters
        self.mean = torch.tensor(ck["norm_mean"]).float().to(device)
        self.std = torch.tensor(ck["norm_std"]).float().to(device)
        self.layer = ck["config"]["data"]["target_layer"]
        self.model_name = ck["config"]["extraction"]["model_name"]

    def encode(self, h):
        return self.ae.encoder((h - self.mean) / self.std)

    def to_raw_dir(self, v):
        return self.ae.decoder(v) * self.std


@torch.no_grad()
def concept_vectors(enc, H_fit, device, chunk=8192):
    """v_k = mean encoder output assigned to code k (VQLC eq. 6), on the fit split."""
    z, lab = [], []
    for i in range(0, len(H_fit), chunk):
        h = torch.from_numpy(H_fit[i:i + chunk]).to(device).float()
        v = enc.encode(h)
        z.append(v)
        lab.append(torch.cdist(v, enc.centroids).argmin(1))
    z, lab = torch.cat(z), torch.cat(lab)
    K = enc.n_clusters
    V = torch.zeros(K, z.shape[1], device=device)
    cnt = torch.zeros(K, device=device)
    V.index_add_(0, lab, z)
    cnt.index_add_(0, lab, torch.ones(len(lab), device=device))
    live = cnt > 0
    V[live] /= cnt[live, None]
    return V, live, lab


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="AE ckpt (also supplies model+layer)")
    ap.add_argument("--baseline_kmeans", default=None, help="score this encoder-free npz instead")
    ap.add_argument("--dataset", default="db14", choices=["db14", "ag_news"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n_fit", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = AEEnc(args.checkpoint, dev)
    enc = KMeansEnc(args.baseline_kmeans, dev) if args.baseline_kmeans else ref
    layer = args.layer if args.layer is not None else ref.layer
    print(f"[faith] {ref.model_name} layer {layer} | "
          f"{'kmeans baseline' if args.baseline_kmeans else 'AE'} K={enc.n_clusters}")

    cfg = shared.set_dataset(args.dataset)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ref.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(ref.model_name,
                                              torch_dtype=torch.float32).to(dev).eval()

    from datasets import load_dataset
    ds = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])
    ds = ds.shuffle(seed=args.seed)
    n = args.n_fit + args.n_eval
    tc = cfg["text_column"]
    texts = [ds[i][tc] for i in range(n)]
    labels = [ds[i]["label"] for i in range(n)]
    fit_t, ev_t = texts[:args.n_fit], texts[args.n_fit:]
    ev_l = labels[args.n_fit:]

    print("[faith] capturing fit-split representations …")
    H_fit = shared.capture_h(lm, tok, fit_t, layer, dev)
    V, live, _ = concept_vectors(enc, H_fit, dev)
    print(f"[faith] concept vectors from {len(H_fit):,} fit tokens: {int(live.sum())} live of {enc.n_clusters}")

    print("[faith] base predictions …")
    base = shared.predict(lm, tok, ev_t, ev_l, dev)
    H_ev = shared.capture_h(lm, tok, ev_t, layer, dev)
    with torch.no_grad():
        z_ev = enc.encode(torch.from_numpy(H_ev).to(dev).float())
        assign = torch.cdist(z_ev, enc.centroids).argmin(1)
    used = torch.unique(assign)
    n_concepts = int(live[used].sum())
    active_rate = float(live[assign].float().mean())
    print(f"[faith] eval assigned to {len(used)} concepts; {n_concepts} of them live on fit; "
          f"active_rate {active_rate:.3f}")

    # per-instance ablation direction in RAW residual space
    with torch.no_grad():
        dirs = enc.to_raw_dir(V[assign])
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)

    hook = SplicingHook(lm, layer)
    order = {"i": 0}

    def fn(hs):
        h = hs[0] if isinstance(hs, tuple) else hs
        # `predict` classifies by GREEDY GENERATION, so this hook fires once per
        # decoded token: first the prefill (T = prompt length), then T = 1 for
        # each new token under the KV cache. The concept lives at the last PROMPT
        # position — the task-decision representation — so ablate only on prefill
        # and advance the row counter only there. Advancing on every call
        # desynchronises `dirs` against the batch.
        if h.shape[1] <= 1:
            return hs
        b = h.shape[0]
        d = dirs[order["i"]:order["i"] + b].to(h.dtype)
        order["i"] += b
        last = h[:, -1, :]
        proj = (last * d).sum(-1, keepdim=True) * d
        h = h.clone()
        h[:, -1, :] = last - proj
        return (h,) + hs[1:] if isinstance(hs, tuple) else h

    print("[faith] ablated predictions …")
    hook.activate(fn)
    try:
        abl = shared.predict(lm, tok, ev_t, ev_l, dev)
    finally:
        hook.deactivate()

    bc = np.array(base["gold_conf"]); ac = np.array(abl["gold_conf"])
    res = {
        "confidence_change": float(np.abs(bc - ac).mean()),
        "confidence_drop": float((bc - ac).mean()),
        "base_acc": float(np.mean(base["correct"])),
        "ablated_acc": float(np.mean(abl["correct"])),
        "n_concepts": n_concepts,
        "active_rate": active_rate,
        "n_eval": len(ev_t),
    }
    print("\n" + "=" * 62)
    print(f"  confidence change |Δ|   {res['confidence_change']:.4f}   <- VQLC's headline")
    print(f"  confidence drop         {res['confidence_drop']:+.4f}")
    print(f"  accuracy  base -> abl   {res['base_acc']:.3f} -> {res['ablated_acc']:.3f}")
    print(f"  n_concepts              {res['n_concepts']}")
    print(f"  active_rate             {res['active_rate']:.3f}")
    print("=" * 62)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[faith] wrote {args.out}")


if __name__ == "__main__":
    main()
