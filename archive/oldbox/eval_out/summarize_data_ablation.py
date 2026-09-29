"""Old-dump vs new-dump summary for the d6144 dpc DATA ABLATION.

    d6144_old  trained on activations_diverse_10M  (prefix 4..255, 5-way equal mix)
    d6144_new  trained on activations_sampled_10M  (64 random positions of <=2048-token
               docs, pretraining-like mix) — identical config otherwise
    km_old / km_new  encoder-free balanced k-means (dpc init), fit on each dump

Reuses every table of eval_out/summarize_width_series.py (same statistics, same sign
conventions — see its docstring) by pointing its arm registry at these files. It reads
whatever exists, so it can be run mid-sweep.

Reading guide:
  * probe: the headline. If km_new also rises over km_old, the DATA change moved the
    raw space, and the AE comparison should be read against its own-dump control.
  * cq is printed twice, once per dump; each file holds both AEs plus that dump's km.
  * interventions: z - h per AE, each against its own joint-correct doc set.

Usage: scripts/delta/py eval_out/summarize_data_ablation.py [--with_valmse]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_width_series as S  # noqa: E402

CK = S.CKPT
E = "e2e/checkpoints/general/llama3.2-3B/layer27"
TAG = "d6144_sampled"
PROBE = f"eval_out/probe_{TAG}.json"
CQ_NEW = f"eval_out/cq_{TAG}_newdump.json"
CQ_OLD = f"eval_out/cq_{TAG}_olddump.json"

AE_ARMS = {
    "d6144_old": {
        "ckpt": f"{CK}/k2000_bnh_b32k_lam1_d6144_dpc/step_0014200.pt",
        "mmlu": "eval_out/mmlu_gelu50.json",
        "probe": PROBE,
        # epoch-50 files come from the old_ep50 step; epoch-47 (best_val) is the fallback
        "steer_db14": ["results/steer_db14_d6144_50.json", "results/steer_db14_dpc.json"],
        "range_db14": ["results/range_intervention_db14_d6144_50_dprime.json",
                       "results/range_intervention_db14_b32k_dpc_dprime.json"],
        "range_bios": ["results/range_intervention_biasbios_d6144_50_dprime.json",
                       "results/range_intervention_biasbios_b32k_dpc_dprime.json"],
    },
    "d6144_new": {
        "ckpt": f"{CK}/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt",
        "mmlu": f"eval_out/mmlu_{TAG}.json",
        "probe": PROBE,
        "steer_db14": f"results/steer_db14_{TAG}.json",
        "range_db14": f"results/range_intervention_db14_{TAG}_dprime.json",
        "range_bios": f"results/range_intervention_biasbios_{TAG}_dprime.json",
    },
}
KM_ARMS = {"km_old": {"probe": PROBE}, "km_new": {"probe": PROBE}}


def use(arms: dict, cq: str | None) -> None:
    S.ARMS = arms
    S.NAMES = list(arms)
    S.PREFERRED = {"probe": PROBE, **({"cq": cq} if cq else {})}
    S.FALLBACK = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_valmse", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("  L27 d6144 dpc — DATA ABLATION: old dump (prefix, 5-way) vs new dump (sampled)")
    print("=" * 74)
    use(AE_ARMS, None)
    S.section_provenance(args.with_valmse)
    S.section_mmlu()

    use({**KM_ARMS, **{k: {**v} for k, v in AE_ARMS.items()}}, None)
    S.section_probe()

    for path, what in ((CQ_NEW, "NEW dump (activations_sampled_10M)"),
                       (CQ_OLD, "OLD dump (activations_diverse_10M)")):
        use({**KM_ARMS, **AE_ARMS}, path)
        print(f"\n>>> clustering quality measured on the {what}: {path}")
        S.section_cq()

    use(AE_ARMS, None)
    S.section_steer()
    S.section_range("range_db14", "RANGE INTERVENTIONS — DB14 (d' saliency, tao 2.0)")
    S.section_range("range_bios", "RANGE INTERVENTIONS — bias_in_bios (d' saliency, tao 2.0)")
    S.section_reliability()
    S.section_effect_noise()
    S.section_moderation()
    print(f"\n  number control: eval_out/number_{TAG}_review.json "
          f"(old arm: eval_out/number_control_review.json)")
    print()


if __name__ == "__main__":
    main()
