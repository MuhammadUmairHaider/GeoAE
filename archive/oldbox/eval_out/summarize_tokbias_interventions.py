"""Interventions: token-bypass AE vs its parent d6144_new (identical config otherwise).

Both arms are read as z − h (each AE builds its own joint-correct doc set, so raw
selectivities are not comparable across AEs), then compared HEAD TO HEAD paired
by concept: Δ = (z−h)_tokbias − (z−h)_d6144_new, sign-normalised so + = the
bypass AE is better. The per-arm tables come from summarize_width_series.

Number control: strict flip rate, z minus the mean of the rotation controls
(how much the AE's own coordinates help), paired by subject noun with a 20k
noun bootstrap — the same statistic as the earlier data-ablation comparison.

Reading guide: the old epoch-47 vs epoch-50 gap of one run was ~0.05 per range
operator, so single-operator |Δ| below that is not evidence; the pooled rows are.

    scripts/delta/py eval_out/summarize_tokbias_interventions.py
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
ARMS = {
    "d6144_new": {
        "ckpt": f"{CK}/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt",
        "steer_db14": "results/steer_db14_d6144_sampled.json",
        "range_db14": "results/range_intervention_db14_d6144_sampled_dprime.json",
        "range_bios": "results/range_intervention_biasbios_d6144_sampled_dprime.json",
        "number": "results/range_number_d6144_sampled_dprime.json",
    },
    "tokbias": {
        "ckpt": f"{CK}/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt",
        "steer_db14": "results/steer_db14_d6144_tokbias.json",
        "range_db14": "results/range_intervention_db14_d6144_tokbias_dprime.json",
        "range_bios": "results/range_intervention_biasbios_d6144_tokbias_dprime.json",
        "number": "results/range_number_d6144_tokbias_dprime.json",
        # same run + the matched base arm hb (edit (h-mean)/std - b[tok], add b[tok] back)
        "number_hb": "results/range_number_d6144_tokbias_dprime_hb.json",
    },
}


def _load(arm, key):
    p = ROOT / ARMS[arm][key]
    return json.loads(p.read_text()) if p.exists() else None


def head_to_head(key: str, title: str, sign: int) -> None:
    S.header(f"HEAD TO HEAD — {title}: Δ = (z−h) tokbias − (z−h) d6144_new, + = bypass better")
    a, b = _load("d6144_new", key), _load("tokbias", key)
    if not (a and b and "concepts" in a and "concepts" in b):
        print("  (missing)")
        return
    A, B = a["concepts"], b["concepts"]
    cs = [c for c in A if c in B]
    ops = [k[2:] for k in A[cs[0]] if k.startswith("z_") and k in B[cs[0]] and "h_" + k[2:] in B[cs[0]]]
    per = np.zeros((len(cs), len(ops)))
    print(f"  {'operator':22s} {'d6144_new':>9s} {'tokbias':>8s} {'Δ':>7s} {'p':>6s} {'wins':>6s}")
    for j, op in enumerate(ops):
        zh = lambda D: sign * np.array([D[c]["z_" + op]["selectivity"] - D[c]["h_" + op]["selectivity"] for c in cs])
        za, zb = zh(A), zh(B)
        d = zb - za
        per[:, j] = d
        p = stats.ttest_1samp(d, 0).pvalue if np.any(d != 0) else 1.0
        print(f"  {op:22s} {za.mean():+9.3f} {zb.mean():+8.3f} {d.mean():+7.3f} {p:6.3f} {int((d > 0).sum()):>2d}/{len(d)}")
    pooled = per.mean(1)
    st = [j for j, o in enumerate(ops) if o.startswith("st_") or key.startswith("steer")]
    for name, v in (("POOLED all operators", pooled), ("POOLED steering ops", per[:, st].mean(1))):
        p = stats.wilcoxon(v).pvalue if np.any(v != 0) else 1.0
        print(f"  {name:22s} {'':9s} {'':8s} {v.mean():+7.3f} {p:6.3f} {int((v > 0).sum()):>2d}/{len(v)}   (Wilcoxon over concepts)")


def number() -> None:
    S.header("NUMBER CONTROL — flip rate, z minus rotation controls; + = the AE's coordinates help")
    a, b = _load("d6144_new", "number"), _load("tokbias", "number")
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
    print(f"  rotation controls: {rots}; {len(nouns)} subject nouns, {len(lab)} prompts")
    print(f"  {'operator':16s} {'dir':9s} | {'z−rot old':>9s} {'z−rot new':>9s} | {'Δ new−old [95% noun CI]':>28s} | h flip")
    for mode in ("salient", "range", "transport"):
        for al in ("0.5", "1.0"):
            for d in (None, "singular", "plural"):
                adv = lambda D: flips(D, "z", mode, al, d) - np.mean([flips(D, r, mode, al, d) for r in rots], 0)
                va, vb = adv(a), adv(b)
                diff = vb - va
                lo, hi = np.quantile(diff[idx].mean(1), [.025, .975])
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"  {mode + ' a' + al:16s} {d or 'both':9s} | {va.mean():+9.3f} {vb.mean():+9.3f} | "
                      f"{diff.mean():+7.3f} [{lo:+.3f},{hi:+.3f}]{star:>6s} | {flips(a, 'h', mode, al, d).mean():.3f}")
    for name, D in (("d6144_new", a), ("tokbias", b)):
        bl = D["baselines"]
        print(f"  {name:10s} test base acc {bl['test_base_pair_accuracy']:.3f}  recon acc "
              f"{bl['test_recon_pair_accuracy']:.3f}  joint {bl['joint_correct_count']}  "
              f"neutral recon KL {np.mean(bl['neutral_reconstruction']['kl']):.4f}")


def number_hb() -> None:
    """Matched base for the number control: at the edit position (the attractor noun) the
    token varies, so editing h - b[tok] can differ from editing h. z − hb is the encoder's
    contribution; hb − h is what removing the token mean does to a plain residual edit."""
    S.header("NUMBER CONTROL — matched base arm hb (edit h − b[tok]); flip rates, + = first arm better")
    d = _load("tokbias", "number_hb")
    if not d:
        print("  (missing: run eval_out/run_tokbias_interventions.sh number_hb)")
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
                h, hb, z = (flips(sp, mode, al, dr) for sp in ("h", "hb", "z"))
                print(f"  {mode + ' a' + al:16s} {dr or 'both':9s} | {h.mean():6.3f} {hb.mean():6.3f} {z.mean():6.3f} | "
                      f"{ci(z - hb):>26s} | {ci(hb - h):>26s}")
    kl = lambda sp: np.mean([d["arms"][k]["neutral"]["kl"] for k in d["arms"] if k.startswith(sp + ":")])
    print(f"  mean neutral-prompt KL over all arms:  h {kl('h'):.4f}   hb {kl('hb'):.4f}   z {kl('z'):.4f}")


def main() -> None:
    print("=" * 78)
    print("  L27 d6144 dpc sampled — INTERVENTIONS: token-bypass AE vs parent (d6144_new)")
    print("=" * 78)
    S.ARMS = ARMS
    S.NAMES = list(ARMS)
    S.PREFERRED, S.FALLBACK = {}, {}
    S.section_steer()
    S.section_range("range_db14", "RANGE INTERVENTIONS — DB14 (d' saliency, tao 2.0)")
    S.section_range("range_bios", "RANGE INTERVENTIONS — bias_in_bios (d' saliency, tao 2.0)")
    head_to_head("steer_db14", "DB14 steering", sign=-1)     # steer: more negative = better
    head_to_head("range_db14", "DB14 range interventions", sign=1)
    head_to_head("range_bios", "bias_in_bios range interventions", sign=1)
    number()
    number_hb()


if __name__ == "__main__":
    main()
