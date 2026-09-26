"""
Generation-level cluster steering: does steering toward an UNSUPERVISED cluster make the
model's continuation take on that cluster's content? Base vs bypass AE.

For each codebook (base = balanced k-means on the plain residual; bypass = the token-bypass
AE's clusters), at EVERY position during generation:

    h  <-  h + alpha * (m_t - m_s(pos))

m_k = mean normalised residual of cluster k's members (held-out reference rows), s(pos) =
the position's own cluster under that codebook, t = the target. Both codebooks use this
same form, so the ONLY difference between them is the partition. (L27 is the last block,
so each step's edit acts on that step's next-token choice; steering compounds through the
generated text.) --mode latent instead edits the bypass AE's latent, z + a (c_t - c_s).

Targets: --n_targets random CONTENT clusters per codebook, one rule for both: not a hub,
>= --min_members reference members drawn from >= --min_docs documents, and most of the tokens
the cluster raises next (top-30 by lift over the average next-token distribution) are WHOLE
words: word-start tokens (leading space) of >= 3 letters. (A first version counted any
alphabetic token, which let mid-word fragment clusters — 'izioni, icularly, incible' — in
as targets that no judge can score.) Profiles list whole words only.

RANGE GATING (--gate, NeuronLens-style, tau = --tao, salient share = --percent): per cluster k,
from its reference members: mean mu_k, sd sd_k, salient coordinates S_k = top --percent by
d' (members vs everyone else), range = mu_k +- tau * sd_k.
  pos      steer a position only if it sits INSIDE its own cluster s's range: RMS z-score
           over S_s <= tau (positions the codebook describes poorly are left untouched)
  dim      move only coordinates in S_s | S_t, and only where the value is inside s's range
  pos+dim  both
  none     the ungated edit above

Scores (per prompt x target x alpha; 20 neutral prompts, greedy, --max_new tokens):
  choice   forced choice between the target and 3 other targets of the same codebook
           (chance .25), scoring the continuation by each cluster's next-token profile
           (log-lift). The LLM-judge version (cluster_steer_judge.py) is the non-circular check.
  lexical  share of continuation tokens in the target's top-100 lift tokens, minus the same
           for the unsteered continuation of that prompt
  ppl      perplexity of the continuation under the UNEDITED model (fluency cost)
  distinct share of distinct bigrams (repetition)
Codebooks are compared at matched fluency cost (log ppl ratio vs unsteered), CIs by
bootstrap over targets.

    python -u -m geoae.interp.cluster_steer_generate --base <km.npz> --bypass <ckpt> \
        --out eval_out/cluster_steer_generate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.hooks import SplicingHook, TokenIdTap
from geoae.interp.cluster_steering import Arm

PROMPTS = [
    "The", "Here is a short passage:\n\n", "Yesterday, I", "In this article, we will",
    "One thing that many people do not know is that", "She opened the door and",
    "The following is a list of", "According to recent reports,", "When I was younger, I",
    "This is an example of", "The main reason for this is", "It all started when",
    "Here are a few tips:", "In the first chapter,", "Our team decided to",
    "There are several ways to", "The results show that", "At the end of the day,",
    "He looked at the", "For more information, please",
]


def word_like(s: str, start_only: bool = True) -> bool:
    """A whole word: a word-START token (leading space) of >= 3 letters. start_only=False is
    the first version's rule (any alphabetic token), kept for reproducing it."""
    w = s.strip()
    if len(w) < 3 or not w.isalpha():
        return False
    return s.startswith(" ") if start_only else True


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base codebook: balanced k-means .npz on the plain residual")
    ap.add_argument("--bypass", required=True, help="token-bypass AE checkpoint")
    ap.add_argument("--mode", default="partition", choices=["partition", "latent"])
    ap.add_argument("--gate", default="none", choices=["none", "pos", "dim", "pos+dim"],
                    help="NeuronLens-style range gating of the edit (partition mode only)")
    ap.add_argument("--tao", type=float, default=2.0, help="range half-width in member sd units")
    ap.add_argument("--percent", type=float, default=0.3,
                    help="share of coordinates called salient per cluster (d'); 0.3 as in the range "
                         "interventions — a convention, not a tuned value")
    ap.add_argument("--activations_dir", default="activations_sampled_10M")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_eval", type=int, default=500_000)
    ap.add_argument("--n_targets", type=int, default=40)
    ap.add_argument("--min_members", type=int, default=50)
    ap.add_argument("--min_docs", type=int, default=20, help="members must come from >= this many documents")
    ap.add_argument("--word_filter", default="start", choices=["start", "alpha"],
                    help="start = whole words only (default); alpha = the first version's rule")
    ap.add_argument("--hub_x", type=float, default=3.0)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
                    help="alpha 1 moves every position the full member-mean gap; >= 1 already degenerates text")
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    act = Path(args.activations_dir)
    meta = json.load(open(act / "meta.json"))
    tk = AutoTokenizer.from_pretrained(meta["model"])
    tk.pad_token, tk.padding_side = tk.eos_token, "left"
    lm = load_lm(meta["model"], device=dev)
    from geoae.lm_arch import decoder_layers
    if args.layer != len(decoder_layers(lm)) - 1:
        raise SystemExit("--layer must be the last block (the signatures are next-token distributions)")
    head, fnorm = lm.get_output_embeddings(), lm.model.norm
    V = head.weight.shape[0]
    dec = [tk.decode([i]) for i in range(V)]
    wl = torch.tensor([word_like(s, args.word_filter == "start") for s in dec], device=dev)

    X = np.load(str(act / f"layer_{args.layer}.npy"), mmap_mode="r")
    N = len(X)
    ev = np.arange(int(N * (1 - args.val_frac)), N)[: args.n_eval]
    doc = np.load(act / "rows_doc.npy")[ev]
    ref = ev[doc % 2 == 0]                                   # same reference half as cluster_steering
    doc_ref = doc[doc % 2 == 0]
    H = torch.from_numpy(np.array(X[ref])).to(dev)
    tok_ref = torch.from_numpy(np.load(act / "rows_tok.npy")[ref].astype(np.int64)).to(dev)
    print(f"[gen] {len(ref):,} reference rows (even docs of the held-out tail)")

    arms = {"base": Arm("base", args.base, dev), "bypass": Arm("bypass", args.bypass, dev)}
    if arms["base"].kind != "km" or arms["bypass"].kind != "ae":
        raise SystemExit("--base must be a k-means .npz and --bypass an AE checkpoint")
    mean, std = arms["base"].mean, arms["base"].std
    if not (torch.allclose(arms["bypass"].mean, mean) and torch.allclose(arms["bypass"].std, std)):
        raise SystemExit("base and bypass use different norm stats")

    def probs(h_raw):
        return torch.softmax(head(fnorm(h_raw.to(head.weight.dtype))).float(), -1)

    # ---- reference statistics per codebook: member means, signatures, target choice --------
    Pbar = torch.zeros(V, device=dev)
    for i in range(0, len(ref), args.chunk):
        Pbar += probs(H[i:i + args.chunk]).sum(0)
    Pbar /= len(ref)
    stats = {}
    for name, arm in arms.items():
        K = arm.C.shape[0]
        lab = torch.empty(len(ref), dtype=torch.long, device=dev)
        M = torch.zeros(K, H.shape[1], device=dev)
        Q = torch.zeros(K, H.shape[1], device=dev)             # per-cluster sum of squares (gating)
        P = torch.zeros(K, V, device=dev)
        for i in range(0, len(ref), args.chunk):
            x = (H[i:i + args.chunk].float() - mean) / std
            l = arm.assign(x, tok_ref[i:i + args.chunk])
            lab[i:i + len(l)] = l
            M.index_add_(0, l, x)
            Q.index_add_(0, l, x * x)
            P.index_add_(0, l, probs(H[i:i + args.chunk]))
        cnt = torch.bincount(lab, minlength=K).float()
        n_tot = cnt.sum()
        tot_s, tot_q = M.sum(0), Q.sum(0)
        M /= cnt.clamp_min(1)[:, None]
        P /= cnt.clamp_min(1)[:, None]
        # member sd, and d' of each cluster vs everyone else, for the range gates
        SD = (Q / cnt.clamp_min(1)[:, None] - M * M).clamp_min(1e-8).sqrt()
        rest_n = (n_tot - cnt).clamp_min(1)[:, None]
        mu_r = (tot_s[None] - M * cnt[:, None]) / rest_n
        sd_r = ((tot_q[None] - Q) / rest_n - mu_r * mu_r).clamp_min(1e-8).sqrt()
        dprime = (M - mu_r).abs() / (0.5 * (SD * SD + sd_r * sd_r)).sqrt()
        n_sal = max(1, int(round(args.percent * H.shape[1])))
        SAL = torch.zeros_like(M, dtype=torch.bool)
        SAL.scatter_(1, dprime.topk(n_sal, dim=1).indices, True)
        del Q, mu_r, sd_r, dprime
        hub = cnt / cnt.sum() > args.hub_x / K
        lab_np = lab.cpu().numpy()
        pairs = np.unique(lab_np.astype(np.int64) * (int(doc_ref.max()) + 1) + doc_ref)
        n_docs = torch.as_tensor(np.bincount(pairs // (int(doc_ref.max()) + 1), minlength=K), device=dev)
        lift = torch.where(P >= 5e-4, P / Pbar.clamp_min(1e-9), torch.zeros_like(P))
        top30 = lift.topk(30, dim=1).indices
        content = wl[top30].float().mean(1)
        eligible = ((cnt >= args.min_members) & (n_docs >= args.min_docs) & ~hub
                    & (content >= 0.6)).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(eligible) < args.n_targets + 3:
            raise SystemExit(f"{name}: only {len(eligible)} eligible content clusters")
        targets = np.sort(rng.choice(eligible, size=args.n_targets, replace=False))
        tt = torch.as_tensor(targets, device=dev)
        top100 = lift[tt].topk(100, dim=1).indices
        mem_tok = {int(t): np.bincount(tok_ref[lab == t].cpu().numpy(), minlength=V) for t in targets}
        top_words = (lift[tt] * wl.float()).topk(20, dim=1).indices      # whole words only, for the judge
        profiles = {int(t): {
            "raises_next": [dec[int(v)] for v in top_words[k]],
            "n_docs": int(n_docs[t]),
            "members_on": [dec[int(v)] for v in np.argsort(-mem_tok[int(t)])[:12] if mem_tok[int(t)][v] > 0],
            "n_members": int(cnt[t])} for k, t in enumerate(targets)}
        stats[name] = dict(M=M, SD=SD, SAL=SAL, P_t=P[tt].clone(), top100=top100, targets=targets, profiles=profiles,
                           n_eligible=int(len(eligible)), hub_share=float((cnt[hub].sum() / cnt.sum())))
        print(f"[gen] {name}: {len(eligible)} eligible content clusters (of {K}); hubs {int(hub.sum())} "
              f"({stats[name]['hub_share']:.1%} of rows); targets e.g. "
              + " | ".join(", ".join(profiles[int(t)]["raises_next"][:5]) for t in targets[:3]))
        del P, lift
        torch.cuda.empty_cache()
    del H

    # ---- generation --------------------------------------------------------------------------
    hook, tap = SplicingHook(lm, args.layer), TokenIdTap(lm)
    enc = tk(PROMPTS, return_tensors="pt", padding=True).to(dev)
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

    gate_log = {"pos_share": [], "dim_share": []}     # share of positions steered / coords moved

    def steer_fn(name, t_idx, alpha):
        arm, st = arms[name], stats[name]
        t = int(st["targets"][t_idx])

        def fn(hs):
            B, T, D = hs.shape
            x = (hs.reshape(B * T, D).float() - mean) / std
            tok = tap.ids_for(hs)
            if args.mode == "latent" and arm.kind == "ae":
                tb = tok if arm.ae.has_token_bias else None
                z = arm.ae.encode(x, tb)
                s = torch.cdist(z, arm.C).argmin(1)
                x2 = arm.ae.decode(z + alpha * (arm.C[t] - arm.C[s]), tb)
            else:
                s = arm.assign(x, tok)
                delta = st["M"][t] - st["M"][s]
                if args.gate != "none":
                    mu_s, sd_s, sal_s = st["M"][s], st["SD"][s], st["SAL"][s]
                    zs = (x - mu_s) / sd_s
                    if "dim" in args.gate:
                        keep = (sal_s | st["SAL"][t]) & (zs.abs() <= args.tao)
                        delta = delta * keep
                        gate_log["dim_share"].append(float(keep.float().mean()))
                    if "pos" in args.gate:
                        rms = ((zs * zs) * sal_s).sum(1).div(sal_s.sum(1).clamp_min(1)).sqrt()
                        on = rms <= args.tao
                        delta = delta * on[:, None]
                        gate_log["pos_share"].append(float(on.float().mean()))
                x2 = x + alpha * delta
            return (x2 * std + mean).reshape(B, T, D).to(hs.dtype)
        return fn

    @torch.no_grad()
    def ppl(cont):
        """Per-sequence perplexity of the continuation under the unedited model, given the prompt."""
        ids = torch.cat([enc["input_ids"], cont], 1)
        am = torch.cat([enc["attention_mask"], (cont != -1).long()], 1)
        logits = lm(input_ids=ids, attention_mask=am).logits[:, L - 1:-1].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), cont, reduction="none")
        return nll.mean(1).exp().cpu().numpy()

    def distinct2(row):
        bg = list(zip(row[:-1], row[1:]))
        return len(set(bg)) / max(len(bg), 1)

    base_cont = generate()
    base_ppl = ppl(base_cont)
    base_ids = base_cont.cpu().numpy()
    gens = {"unsteered": {"text": [tk.decode(r, skip_special_tokens=True) for r in base_ids],
                          "ppl": base_ppl.tolist()}}
    print(f"[gen] unsteered: median ppl {np.median(base_ppl):.2f}; e.g. {gens['unsteered']['text'][2]!r}")

    if args.gate != "none" and args.mode != "partition":
        raise SystemExit("--gate is defined for --mode partition")
    res = {"meta": {**vars(args), "prompts": PROMPTS}, "arms": {}, "generations": gens}
    for name in arms:
        st = stats[name]
        nt, npr = len(st["targets"]), len(PROMPTS)
        # 3 fixed distractor targets per (target, prompt), shared by every alpha
        dis = np.zeros((nt, npr, 3), dtype=int)
        for i in range(nt):
            others = [j for j in range(nt) if j != i]
            for p_ in range(npr):
                dis[i, p_] = rng.choice(others, size=3, replace=False)
        logP = (0.5 * st["P_t"] + 0.5 * Pbar).log() - Pbar.clamp_min(1e-12).log()   # (nt, V) log-lift
        in100 = torch.zeros(nt, V, dtype=torch.bool, device=dev)
        in100.scatter_(1, st["top100"], True)
        base_lex = np.array([[float(in100[i][base_cont[p_]].float().mean()) for p_ in range(npr)] for i in range(nt)])
        rec = {"targets": st["targets"].tolist(), "profiles": st["profiles"], "n_eligible": st["n_eligible"],
               "hub_share": st["hub_share"], "distractors": dis.tolist(), "alphas": {}}
        # alpha 0: the unsteered continuation scored against each target (the chance anchor)
        sc0 = logP[:, base_cont].mean(-1)
        ch0 = np.array([[float(int(torch.argmax(sc0[[i] + list(dis[i, p_]), p_])) == 0) for p_ in range(npr)]
                        for i in range(nt)])
        d20 = [distinct2(list(r)) for r in base_ids]
        rec["alphas"]["0.0"] = {"choice": ch0.tolist(), "lexical": np.zeros((nt, npr)).tolist(),
                                "ppl": np.tile(base_ppl, (nt, 1)).tolist(),
                                "log_ppl_ratio": np.zeros((nt, npr)).tolist(),
                                "distinct2": np.tile(d20, (nt, 1)).tolist(), "text": []}
        for a in args.alphas:
            choice = np.zeros((nt, npr)); lex = np.zeros((nt, npr)); pp = np.zeros((nt, npr)); d2 = np.zeros((nt, npr))
            texts = []
            for i in range(nt):
                cont = generate(steer_fn(name, i, a))
                pp[i] = ppl(cont)
                sc = logP[:, cont].mean(-1)                      # (nt, npr): mean log-lift per candidate
                for p_ in range(npr):
                    cand = [i] + list(dis[i, p_])
                    choice[i, p_] = float(int(torch.argmax(sc[cand, p_])) == 0)
                    lex[i, p_] = float(in100[i][cont[p_]].float().mean()) - base_lex[i, p_]
                ids = cont.cpu().numpy()
                d2[i] = [distinct2(list(r)) for r in ids]
                texts.append([tk.decode(r, skip_special_tokens=True) for r in ids])
            lr = np.log(pp) - np.log(base_ppl)[None, :]
            gate_stats = {k: (float(np.mean(v)) if v else None) for k, v in gate_log.items()}
            gate_log["pos_share"].clear(); gate_log["dim_share"].clear()
            rec["alphas"][str(a)] = {"choice": choice.tolist(), "lexical": lex.tolist(), "ppl": pp.tolist(),
                                     "log_ppl_ratio": lr.tolist(), "distinct2": d2.tolist(), "text": texts,
                                     "gate": gate_stats}
            gs = "" if args.gate == "none" else (
                f"  steered positions {gate_stats['pos_share']:.2f}" if gate_stats["pos_share"] is not None else "") + (
                f"  moved coords {gate_stats['dim_share']:.2f}" if gate_stats["dim_share"] is not None else "")
            print(f"[gen] {name:6s} a{a}:{gs} choice {choice.mean():.3f} (chance .25)  lexical +{lex.mean():.3f}  "
                  f"ppl x{np.exp(lr.mean()):.2f}  distinct2 {d2.mean():.2f} | "
                  f"{texts[0][2][:70]!r}", flush=True)
        res["arms"][name] = rec
    tap.remove()

    # ---- summary: per alpha, and at matched fluency cost ---------------------------------------
    def boot(v, n=args.n_boot):
        """Mean and 95% CI, resampling TARGETS (rows of v)."""
        v = np.asarray(v)
        idx = rng.integers(0, len(v), size=(n, len(v)))
        b = v.mean(1)[idx].mean(1)
        return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    summ = {}
    print("\n[gen] per alpha (mean [95% CI over targets])")
    for name, rec in res["arms"].items():
        summ[name] = {}
        for a, r in rec["alphas"].items():
            r = {**r, "degenerate": (np.asarray(r["distinct2"]) < 0.6).astype(float).tolist()}
            summ[name][a] = {k: boot(r[k]) for k in ("choice", "lexical", "log_ppl_ratio", "distinct2", "degenerate")}
            s = summ[name][a]
            print(f"  {name:6s} a{a:<4} choice {s['choice'][0]:.3f} [{s['choice'][1]:.3f},{s['choice'][2]:.3f}]  "
                  f"lexical {s['lexical'][0]:+.3f}  ppl x{np.exp(s['log_ppl_ratio'][0]):.2f}  "
                  f"degenerate {s['degenerate'][0]:.2f}")
    res["summary"] = summ

    print("\n[gen] at MATCHED fluency cost (continuation perplexity x vs unsteered): choice accuracy "
          "[95% CI over targets] (chance .25)")
    matched = {}

    def curve(rec, key, rows=None):
        al = sorted(rec["alphas"], key=float)
        pick = (lambda v: np.asarray(v)) if rows is None else (lambda v: np.asarray(v)[rows])
        xs = np.array([pick(rec["alphas"][a]["log_ppl_ratio"]).mean() for a in al])
        ys = np.array([pick(rec["alphas"][a][key]).mean() for a in al])
        return xs, ys

    for lev in (1.1, 1.25, 1.5, 2.0):
        row = {}
        for name, rec in res["arms"].items():
            def at(rows=None):
                xs, ys = curve(rec, "choice", rows)
                xs = np.maximum.accumulate(xs)                    # ppl cost is monotone in alpha
                return float(np.interp(np.log(lev), xs, ys, left=np.nan, right=np.nan))
            est = at()
            nt = len(rec["targets"])
            bs = [at(rng.integers(0, nt, nt)) for _ in range(min(args.n_boot, 500))]
            row[name] = (est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))
        matched[str(lev)] = row
        print(f"  ppl x{lev:<4}  " + "   ".join(
            f"{k} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" if v[0] == v[0] else f"{k}  (out of range)"
            for k, v in row.items()))
    res["matched_fluency"] = matched
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[gen] wrote {args.out}")


if __name__ == "__main__":
    main()
