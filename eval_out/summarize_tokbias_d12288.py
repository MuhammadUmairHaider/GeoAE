"""4x-width token-bypass AE (d12288 tokbias) vs its two parents — the decision tables.

    .venv/bin/python eval_out/summarize_tokbias_d12288.py

Three columns throughout: base (balanced k-means on the plain residual) | bypass6k
(the d6144 token-bypass AE, already trained) | bypass12k (this arm), plus a "12k - 6k"
head-to-head delta wherever the two bypass widths are directly comparable, and a
"12k - base" delta where base is itself scored (probe).

Every section tolerates missing inputs — this is meant to run mid-sweep, well before
bypass12k's ~17.5 h training finishes. Reuses geoae/eval_out summarizers by import
rather than copying their logic:
  summarize_width_series     .section_steer() / .section_range() (generic over ARMS/NAMES)
  summarize_tokbias_interventions   only as a reference for the head-to-head / number-control
                             statistics it defines — NOT imported, because its head_to_head()
                             and number() hardcode the literal arm names "d6144_new"/"tokbias",
                             so calling them here would either mislabel bypass12k as "tokbias"
                             or require monkeypatching those literals; small paired-stat
                             re-implementations below (head_to_head, number, number_hb) avoid that.

Reads (whatever exists):
  checkpoints/.../k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt (+best_val.pt)
  checkpoints/.../k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias/step_0014450.pt (+best_val.pt)
  eval_out/token_structure_tokbias12k.json
  eval_out/probe_tokbias12k.json
  eval_out/mmlu_d6144_sampled_tokbias.json, eval_out/mmlu_d12288_sampled_tokbias.json
  eval_out/cq_tokbias12k.json
  results/steer_db14_d6144_tokbias.json, results/steer_db14_d12288_tokbias.json
  results/range_intervention_db14_{d6144,d12288}_tokbias_dprime.json
  results/range_intervention_biasbios_{d6144,d12288}_tokbias_dprime.json
  results/range_number_{d6144,d12288}_tokbias_dprime.json (+ _hb.json)
  eval_out/cluster_steering_tokbias12k.json (+ _allhubs.json), eval_out/cluster_steering.json (+_allhubs, 6k ref)
  eval_out/cluster_steer_judge_v2_tokbias12k.json, eval_out/cluster_steer_judge_v2.json (6k ref)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_width_series as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CK = S.CKPT
TB6 = f"{CK}/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt"
TB12 = f"{CK}/k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias/step_0014450.pt"
TAG6, TAG12 = "d6144_tokbias", "d12288_tokbias"


def load(p) -> dict | None:
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def header(t: str) -> None:
    print("\n" + "=" * 82 + f"\n{t}\n" + "=" * 82)


def fmt(v, spec=".4f", w=11) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'—':>{w}}"
    if spec == "d":
        return f"{int(v):>{w}d}"
    if spec == "s":
        return f"{str(v):>{w}}"
    return f"{v:>{w}{spec}}"


# --------------------------------------------------------------------------------- #
# 1. Checkpoints
# --------------------------------------------------------------------------------- #
def section_checkpoints() -> None:
    header("CHECKPOINTS")
    import torch

    def info(path):
        p = ROOT / path
        if not p.exists():
            return None
        c = torch.load(p, map_location="cpu", weights_only=False)
        return {"epoch": c.get("epoch"), "val_mse": c.get("val_mse"),
                "latent_dim": c["config"]["model"].get("latent_dim")}

    rows = [("step_0014450.pt (eval target)", f"{Path(TB6).parent}/step_0014450.pt",
              f"{Path(TB12).parent}/step_0014450.pt"),
             ("best_val.pt", f"{Path(TB6).parent}/best_val.pt", f"{Path(TB12).parent}/best_val.pt")]
    print(f"  {'file':<32}{'bypass6k epoch/val_mse':>28}{'bypass12k epoch/val_mse':>28}")
    for label, p6, p12 in rows:
        i6, i12 = info(p6), info(p12)
        s6 = f"{i6['epoch']}/{i6['val_mse']:.5f}" if i6 else "missing"
        s12 = f"{i12['epoch']}/{i12['val_mse']:.5f}" if i12 else "missing"
        print(f"  {label:<32}{s6:>28}{s12:>28}")
    if not (ROOT / f"{Path(TB12).parent}/step_0014450.pt").exists():
        print("  (missing: bypass12k not yet at epoch 50 — everything below reads whatever exists)")


# --------------------------------------------------------------------------------- #
# 2. Token structure
# --------------------------------------------------------------------------------- #
TOKSTRUCT_ROWS = ["fvu", "cnmi_cur", "cnmi_next", "trivial_cur_80", "top10", "live"]
ARM_ORDER = ["base", "km_tokmean", "bypass6k", "bypass12k"]


def section_token_structure() -> None:
    header("TOKEN STRUCTURE — held-out dump rows (trivial_cur_80 = share of clusters that are "
           "pure current-token detectors)")
    d = load("eval_out/token_structure_tokbias12k.json")
    if not d:
        print("  (missing: eval_out/token_structure_tokbias12k.json — run the tokstruct step)")
        return
    arms = [a for a in ARM_ORDER if a in d]
    print(f"  {'':16s}" + "".join(f"{a:>13s}" for a in arms))
    for k in TOKSTRUCT_ROWS:
        print(f"  {k:16s}" + "".join(
            f"{d[a][k]:>13.4f}" if isinstance(d[a].get(k), (int, float)) else f"{'—':>13s}"
            for a in arms))
    if "bypass6k" in arms and "bypass12k" in arms:
        print("  -- delta --")
        for k in TOKSTRUCT_ROWS:
            a6, a12 = d.get("bypass6k", {}).get(k), d.get("bypass12k", {}).get(k)
            delta = (a12 - a6) if isinstance(a6, (int, float)) and isinstance(a12, (int, float)) else None
            print(f"    Δ {k:14s}{fmt(delta, '+.4f')}")


# --------------------------------------------------------------------------------- #
# 3. Concept probe — chance-corrected NMI, paired bootstrap
# --------------------------------------------------------------------------------- #
def _centred_ci(boots_a, boots_b, m):
    bt = np.array(boots_b) - np.array(boots_a)
    lo, hi = m + np.quantile(bt - bt.mean(), [.025, .975])
    return lo, hi


def section_probe() -> None:
    header("CONCEPT PROBE — chance-corrected NMI on cache_ids/ (per-rung cNMI, then paired deltas)")
    d = load("eval_out/probe_tokbias12k.json")
    if not d:
        print("  (missing: eval_out/probe_tokbias12k.json — run the probe step)")
        return
    arms = list(next(iter(d.values()))["arms"])
    order = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]
    print(f"  {'rung':16s} {'grain':10s} {'leak':>6s}" + "".join(f"{a[:11]:>12s}" for a in order))
    grp6, grp12, grpb = {"token": [], "sequence": []}, {"token": [], "sequence": []}, {"token": [], "sequence": []}
    for rung, v in d.items():
        ar = v["arms"]
        leak = "     —" if v.get("leak") is None else f"{v['leak']:6.3f}"
        cells = "".join(f"{ar[a]['cnmi']:12.3f}" if a in ar else f"{'—':>12s}" for a in order)
        print(f"  {rung:16s} {v.get('grain', '—'):10s} {leak}{cells}")
        if "bypass6k" in ar and "bypass12k" in ar:
            grp12[v["grain"]].append(ar["bypass12k"]["cnmi"] - ar["bypass6k"]["cnmi"])
        if "base" in ar and "bypass12k" in ar:
            grpb[v["grain"]].append(ar["bypass12k"]["cnmi"] - ar["base"]["cnmi"])

    print("\n  Δ = bypass12k − bypass6k  [95% centred CI]")
    for rung, v in d.items():
        ar = v["arms"]
        if "bypass6k" in ar and "bypass12k" in ar:
            m = ar["bypass12k"]["cnmi"] - ar["bypass6k"]["cnmi"]
            lo, hi = _centred_ci(ar["bypass6k"]["boots"], ar["bypass12k"]["boots"], m)
            print(f"    {rung:16s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
    for g, v in grp12.items():
        if v:
            print(f"  MEAN Δ(12k-6k) {g:9s} {np.mean(v):+.3f} over {len(v)} rungs ({sum(x > 0 for x in v)} up)")

    print("\n  Δ = bypass12k − base  [95% centred CI]")
    for rung, v in d.items():
        ar = v["arms"]
        if "base" in ar and "bypass12k" in ar:
            m = ar["bypass12k"]["cnmi"] - ar["base"]["cnmi"]
            lo, hi = _centred_ci(ar["base"]["boots"], ar["bypass12k"]["boots"], m)
            print(f"    {rung:16s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
    for g, v in grpb.items():
        if v:
            print(f"  MEAN Δ(12k-base) {g:9s} {np.mean(v):+.3f} over {len(v)} rungs ({sum(x > 0 for x in v)} up)")


# --------------------------------------------------------------------------------- #
# 4. MMLU splice
# --------------------------------------------------------------------------------- #
def section_mmlu() -> None:
    header("MMLU UNDER THE SPLICE (n=2000) — unspliced (base) vs spliced (recon) accuracy")
    a = load("eval_out/mmlu_d6144_sampled_tokbias.json")
    b = load("eval_out/mmlu_d12288_sampled_tokbias.json")
    for n, x in (("bypass6k", a), ("bypass12k", b)):
        if not x:
            print(f"  {n:12s} (missing)")
            continue
        m = x["meta"]
        print(f"  {n:12s} base {m['base_acc']:.4f}  recon {m['recon_acc']:.4f}  delta {m['delta']:+.4f}")
    if not (a and b):
        return
    print(f"  Δ recon (12k - 6k)  {b['meta']['recon_acc'] - a['meta']['recon_acc']:+.4f}")
    if "preds" in a and "preds" in b and a["preds"]["gold"] == b["preds"]["gold"]:
        g = np.array(a["preds"]["gold"])
        ra, rb = np.array(a["preds"]["recon"]) == g, np.array(b["preds"]["recon"]) == g
        only_a, only_b = int((ra & ~rb).sum()), int((rb & ~ra).sum())
        p = stats.binomtest(only_b, only_a + only_b).pvalue if only_a + only_b else 1.0
        print(f"  recon correct only under bypass6k: {only_a}, only under bypass12k: {only_b}  "
              f"(McNemar exact p={p:.3f})")
    elif "preds" in a and "preds" in b:
        print("  (per-question gold arrays differ between the two mmlu runs — McNemar skipped)")


# --------------------------------------------------------------------------------- #
# 5. Clustering quality
# --------------------------------------------------------------------------------- #
def section_cq() -> None:
    header("CLUSTERING QUALITY — sampled dump, 1M rows, seed 0 (scale-free metrics first; ignore dunn)")
    d = load("eval_out/cq_tokbias12k.json")
    if not d:
        print("  (missing: eval_out/cq_tokbias12k.json — run the cq step)")
        return
    arms = [a for a in ("base", "bypass6k", "bypass12k") if a in d]
    print(f"  {'':<26}" + "".join(f"{a:>12}" for a in arms) + f"{'12k - 6k':>14}")
    for key, label, spec in S.CQ_ROWS:
        vals = [d.get(a, {}).get(key) for a in arms]
        d6, d12 = d.get("bypass6k", {}).get(key), d.get("bypass12k", {}).get(key)
        delta = (d12 - d6) if isinstance(d6, (int, float)) and isinstance(d12, (int, float)) else None
        print(f"  {label:<26}" + "".join(fmt(v, spec, 12) for v in vals) + fmt(delta, "+.4f", 14))


# --------------------------------------------------------------------------------- #
# 6. Interventions — head to head (12k - 6k) and number control
# --------------------------------------------------------------------------------- #
def head_to_head(path_a: str, path_b: str, title: str, sign: int) -> None:
    """Δ = (z−h)_12k − (z−h)_6k, sign-normalised so + = bypass12k better. A small
    re-implementation of summarize_tokbias_interventions.head_to_head that takes file
    paths instead of hardcoded arm names — see module docstring for why."""
    S.header(f"HEAD TO HEAD — {title}: Δ = (z−h) bypass12k − (z−h) bypass6k, + = 12k better")
    a, b = load(path_a), load(path_b)
    if not (a and b and "concepts" in a and "concepts" in b):
        print("  (missing)")
        return
    A, B = a["concepts"], b["concepts"]
    cs = [c for c in A if c in B]
    if not cs:
        print("  (no shared concepts between the two runs)")
        return
    ops = [k[2:] for k in A[cs[0]] if k.startswith("z_") and k in B[cs[0]] and "h_" + k[2:] in B[cs[0]]]
    if not ops:
        print("  (no shared operators)")
        return
    per = np.zeros((len(cs), len(ops)))
    print(f"  {'operator':22s} {'bypass6k':>9s} {'bypass12k':>9s} {'Δ':>7s} {'p':>6s} {'wins':>6s}")
    for j, op in enumerate(ops):
        zh = lambda D: sign * np.array([D[c]["z_" + op]["selectivity"] - D[c]["h_" + op]["selectivity"] for c in cs])
        za, zb = zh(A), zh(B)
        d = zb - za
        per[:, j] = d
        p = stats.ttest_1samp(d, 0).pvalue if np.any(d != 0) else 1.0
        print(f"  {op:22s} {za.mean():+9.3f} {zb.mean():+9.3f} {d.mean():+7.3f} {p:6.3f} {int((d > 0).sum()):>2d}/{len(d)}")
    pooled = per.mean(1)
    st = [j for j, o in enumerate(ops) if o.startswith("st_") or title.lower().startswith("db14 steer")]
    for name, v in (("POOLED all operators", pooled),
                    ("POOLED steering ops", per[:, st].mean(1) if st else pooled)):
        p = stats.wilcoxon(v).pvalue if np.any(v != 0) else 1.0
        print(f"  {name:22s} {'':9s} {'':9s} {v.mean():+7.3f} {p:6.3f} {int((v > 0).sum()):>2d}/{len(v)}   (Wilcoxon over concepts)")


def number_control() -> None:
    """Flip-rate comparison, bypass6k vs bypass12k — same statistic as
    summarize_tokbias_interventions.number(), re-implemented for these two arms."""
    S.header("NUMBER CONTROL — flip rate, z minus rotation controls; + = bypass12k's coordinates help more")
    a = load(f"results/range_number_{TAG6}_dprime.json")
    b = load(f"results/range_number_{TAG12}_dprime.json")
    if not (a and b):
        print("  (missing)")
        return
    ex = a["test_examples"]
    if [x["id"] for x in ex] != [x["id"] for x in b["test_examples"]]:
        print("  ! test examples differ between arms; not comparable")
        return
    lab = np.array([x["subject_number"] for x in ex])
    lem = np.array([x["subject_lemma"] for x in ex])
    nouns = sorted(set(lem))
    idx = np.random.RandomState(0).randint(0, len(nouns), size=(20000, len(nouns)))

    def flips(D, space, mode, al, direction=None):
        f = np.zeros(len(lab)); m_all = np.zeros(len(lab), bool)
        for s, dname in enumerate(("singular", "plural")):
            if direction and dname != direction:
                continue
            r = D["arms"][f"{space}:suppress_{dname}:{mode}:a{al}"]["rows"]
            m = lab == s
            f[m] = (np.array(r["gold_pair_probability"]) < .5)[m]
            m_all |= m
        return np.array([f[(lem == n) & m_all].mean() for n in nouns])

    rots = sorted({k.split(":")[0] for k in a["arms"] if k.startswith("z_rot")} &
                  {k.split(":")[0] for k in b["arms"] if k.startswith("z_rot")})
    if not rots:
        print("  (no shared rotation controls)")
        return
    print(f"  rotation controls: {rots}; {len(nouns)} subject nouns, {len(lab)} prompts")
    print(f"  {'operator':16s} {'dir':9s} | {'z−rot 6k':>9s} {'z−rot 12k':>9s} | {'Δ 12k−6k [95% noun CI]':>28s}")
    for mode in ("salient", "range", "transport"):
        for al in ("0.5", "1.0"):
            for d in (None, "singular", "plural"):
                adv = lambda D: flips(D, "z", mode, al, d) - np.mean([flips(D, r, mode, al, d) for r in rots], 0)
                va, vb = adv(a), adv(b)
                diff = vb - va
                lo, hi = np.quantile(diff[idx].mean(1), [.025, .975])
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"  {mode + ' a' + al:16s} {d or 'both':9s} | {va.mean():+9.3f} {vb.mean():+9.3f} | "
                      f"{diff.mean():+7.3f} [{lo:+.3f},{hi:+.3f}]{star:>6s}")
    for name, D in (("bypass6k", a), ("bypass12k", b)):
        bl = D["baselines"]
        print(f"  {name:10s} test base acc {bl['test_base_pair_accuracy']:.3f}  recon acc "
              f"{bl['test_recon_pair_accuracy']:.3f}  joint {bl['joint_correct_count']}")


def number_hb(name: str, path: str) -> None:
    S.header(f"NUMBER CONTROL — matched base arm hb ({name}): edit h − b[tok]; flip rates, + = first arm better")
    d = load(path)
    if not d:
        print(f"  (missing: {path})")
        return
    ex = d["test_examples"]
    lab = np.array([x["subject_number"] for x in ex])
    lem = np.array([x["subject_lemma"] for x in ex])
    nouns = sorted(set(lem))
    idx = np.random.RandomState(0).randint(0, len(nouns), size=(20000, len(nouns)))

    def flips(space, mode, al, direction=None):
        f = np.zeros(len(lab)); m_all = np.zeros(len(lab), bool)
        for s_, dname in enumerate(("singular", "plural")):
            if direction and dname != direction:
                continue
            r = d["arms"][f"{space}:suppress_{dname}:{mode}:a{al}"]["rows"]
            m = lab == s_
            f[m] = (np.array(r["gold_pair_probability"]) < .5)[m]
            m_all |= m
        return np.array([f[(lem == n) & m_all].mean() for n in nouns])

    def ci(v):
        lo, hi = np.quantile(v[idx].mean(1), [.025, .975])
        return f"{v.mean():+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}"

    print(f"  {'operator':16s} {'dir':9s} | {'h':>6s} {'hb':>6s} {'z':>6s} | {'z − hb (encoder)':>26s} | {'hb − h (token removal)':>26s}")
    for mode in ("salient", "range", "transport", "global"):
        for al in ("0.5", "1.0"):
            for dr in (None, "singular", "plural"):
                keys_present = f"h:suppress_singular:{mode}:a{al}" in d["arms"]
                if not keys_present:
                    continue
                h, hb, z = (flips(sp, mode, al, dr) for sp in ("h", "hb", "z"))
                print(f"  {mode + ' a' + al:16s} {dr or 'both':9s} | {h.mean():6.3f} {hb.mean():6.3f} {z.mean():6.3f} | "
                      f"{ci(z - hb):>26s} | {ci(hb - h):>26s}")
    kl = lambda sp: np.mean([d["arms"][k]["neutral"]["kl"] for k in d["arms"] if k.startswith(sp + ":")])
    print(f"  mean neutral-prompt KL over all arms:  h {kl('h'):.4f}   hb {kl('hb'):.4f}   z {kl('z'):.4f}")


def section_interventions() -> None:
    # Per-arm tables (z-h selectivity for bypass6k / bypass12k), reusing the generic,
    # arm-name-agnostic summarize_width_series helpers.
    S.ARMS = {
        "bypass6k": {
            "steer_db14": f"results/steer_db14_{TAG6}.json",
            "range_db14": f"results/range_intervention_db14_{TAG6}_dprime.json",
            "range_bios": f"results/range_intervention_biasbios_{TAG6}_dprime.json",
        },
        "bypass12k": {
            "steer_db14": f"results/steer_db14_{TAG12}.json",
            "range_db14": f"results/range_intervention_db14_{TAG12}_dprime.json",
            "range_bios": f"results/range_intervention_biasbios_{TAG12}_dprime.json",
        },
    }
    S.NAMES = ["bypass6k", "bypass12k"]
    S.PREFERRED, S.FALLBACK = {}, {}
    S.section_steer()
    S.section_range("range_db14", "RANGE INTERVENTIONS — DB14 (d' saliency, tao 2.0)")
    S.section_range("range_bios", "RANGE INTERVENTIONS — bias_in_bios (d' saliency, tao 2.0)")

    head_to_head(f"results/steer_db14_{TAG6}.json", f"results/steer_db14_{TAG12}.json",
                 "DB14 steering", sign=-1)   # steer: more negative = better
    head_to_head(f"results/range_intervention_db14_{TAG6}_dprime.json",
                 f"results/range_intervention_db14_{TAG12}_dprime.json", "DB14 range interventions", sign=1)
    head_to_head(f"results/range_intervention_biasbios_{TAG6}_dprime.json",
                 f"results/range_intervention_biasbios_{TAG12}_dprime.json",
                 "bias_in_bios range interventions", sign=1)
    number_control()
    number_hb("bypass6k", f"results/range_number_{TAG6}_dprime_hb.json")
    number_hb("bypass12k", f"results/range_number_{TAG12}_dprime_hb.json")


# --------------------------------------------------------------------------------- #
# 7. Next-token cluster steering
# --------------------------------------------------------------------------------- #
CSTEER_LEVELS = ["0.1", "0.5", "1.0"]


def _csteer_row(d: dict, key: str, level: str) -> str:
    v = (d or {}).get("matched_disrupt", {}).get(key, {}).get(level)
    if not v:
        return f"{'—':>26}"
    t, s = v["transfer"], v["specific"]
    return f"t {t['est']:+.3f}[{t['lo']:+.2f},{t['hi']:+.2f}] s {s['est']:.3f}"


def section_cluster_steering() -> None:
    header("NEXT-TOKEN CLUSTER STEERING — transfer (t) / specificity (s) at MATCHED DISRUPTION "
           "(KL from unedited prediction), 95% doc CI")
    for suffix, label in (("", "main (hubs excluded)"), ("_allhubs", "allhubs (no hub exclusion)")):
        d12 = load(f"eval_out/cluster_steering_tokbias12k{suffix}.json")
        d6 = load(f"eval_out/cluster_steering{suffix}.json")
        print(f"\n  -- {label} --")
        if not d12 and not d6:
            print("  (missing)")
            continue
        rows = [("base:km", "base"), ("km_tokmean:km", "km_tokmean"),
                ("bypass6k:ae", "bypass6k ae"), ("bypass6k:ae_h", "bypass6k ae_h"),
                ("bypass12k:ae", "bypass12k ae"), ("bypass12k:ae_h", "bypass12k ae_h")]
        # The 6k reference file (eval_out/cluster_steering.json) was keyed by "d6144_new"/
        # "tokbias" arm names, not "bypass6k"/"bypass12k" — fall back to those for the
        # 6k-only reference row so it still prints.
        legacy = [("km_new:km", "base"), ("km_tokmean:km", "km_tokmean"),
                  ("d6144_new:ae", "bypass6k(parent) ae"), ("d6144_new:ae_h", "bypass6k(parent) ae_h"),
                  ("tokbias:ae", "bypass6k ae"), ("tokbias:ae_h", "bypass6k ae_h")]
        for lev in CSTEER_LEVELS:
            print(f"  disrupt KL={lev}")
            for key, label2 in rows:
                v12 = _csteer_row(d12, key, lev)
                print(f"    {label2:<18} 12k-file: {v12}")
            if d6 and not d12:
                for key, label2 in legacy:
                    v6 = _csteer_row(d6, key, lev)
                    print(f"    {label2:<18}  6k-file: {v6}")


# --------------------------------------------------------------------------------- #
# 8. Generation (LLM judge)
# --------------------------------------------------------------------------------- #
def _net_fluent(judge: dict, gen: dict) -> dict:
    """cluster_steer_judge's headline score, recomputed from the per-item hits so judge files
    written before the net_fluent section existed (eval_out/cluster_steer_judge_v2.json) are
    scored the same way — no API calls. Fluent = distinct-bigram share >= 0.6; net = hit minus
    the SAME (target, prompt) cell's unsteered hit; usable = fluent share x net (per attempt)."""
    out = {}
    for arm, by_a in judge["arms"].items():
        if "0.0" not in by_a:
            continue
        h0 = np.asarray(by_a["0.0"]["hit"], dtype=float)
        for a in sorted((a for a in by_a if float(a) > 0), key=float):
            h = np.asarray(by_a[a]["hit"], dtype=float)
            ok = np.asarray(gen["arms"][arm]["alphas"][a]["distinct2"])[:, : h.shape[1]] >= 0.6
            m = float(np.where(ok, h - h0, np.nan)[ok].mean()) if ok.any() else float("nan")
            out.setdefault(a, {})[arm] = {"fluent_share": float(ok.mean()), "net_hit": m,
                                          "usable": float(ok.mean() * m) if m == m else float("nan")}
    return out


def _generation_table(judge_path: str, gen_path: str, label: str) -> None:
    j, g = load(judge_path), load(gen_path)
    if not (j and g):
        print(f"  {label}: (missing: {judge_path if not j else gen_path})")
        return
    nf = _net_fluent(j, g)
    arms = list(j["arms"])
    print(f"  {label}   [{judge_path}]")
    print(f"    {'alpha':>6}" + "".join(f"   {a + ': fluent% / net / usable':>32s}" for a in arms))
    for a in sorted(nf, key=float):
        cells = []
        for arm in arms:
            v = nf[a].get(arm)
            cells.append(f"{'—':>32s}" if v is None else
                         f"{100 * v['fluent_share']:>18.0f}% / {100 * v['net_hit']:+5.1f} / {100 * v['usable']:4.1f}")
        print(f"    {a:>6}" + "   ".join([""] + cells))
    for arm in arms:
        best = max(((a, nf[a][arm]) for a in nf if arm in nf[a] and nf[a][arm]["usable"] == nf[a][arm]["usable"]),
                   key=lambda t: t[1]["usable"], default=None)
        if best:
            a, v = best
            print(f"    BEST {arm:7s} alpha {a}: {100 * v['usable']:.1f} usable on-target per 100 attempts "
                  f"(fluent {100 * v['fluent_share']:.0f}%, net hit {100 * v['net_hit']:+.1f})")


def section_generation() -> None:
    header("GENERATION (LLM JUDGE) — fluent continuations only, net of the same cell's unsteered hit; "
           "usable = on-target per 100 attempts")
    _generation_table("eval_out/cluster_steer_judge_v2_tokbias12k.json",
                      "eval_out/cluster_steer_generate_v2_tokbias12k.json", "base vs bypass12k")
    print()
    _generation_table("eval_out/cluster_steer_judge_v2.json",
                      "eval_out/cluster_steer_generate_v2.json", "6k reference: base vs bypass6k")
    print("  (base rows should match across the two runs: same codebook, targets, prompts and greedy decoding)")


def main() -> None:
    print("=" * 82)
    print("  L27 d12288 dpc sampled TOKEN-BYPASS (4x width) — vs bypass6k parent and base")
    print("=" * 82)
    section_checkpoints()
    section_token_structure()
    section_probe()
    section_mmlu()
    section_cq()
    section_interventions()
    section_cluster_steering()
    section_generation()
    print()


if __name__ == "__main__":
    main()
