"""C4: random non-sibling label pairs as a null for sibling coding.

Uses the SAME cached range/Atlas rows as the head-to-head (geoae.baselines.sae_data.load_rows with the same
rows_per_label / control_rows / seed), and the unchanged geoae.baselines.sae_coding.analyze_coding. Besides the
real sibling pairs, N seeded pairs of labels from DIFFERENT families are scored, in the same analyze_coding
call, so the real and null rates share every setting. Per SAE and kind it reports the interpretation / geometry
mix, decoder-cosine distribution, same-feature rate and the rate of top-1 decoder cos >= 0.9.

    python -u experiments/paper/analysis/sae_coding_null.py --cache_dir $DELTA_DATA/cache/gemma_scope_compare \
        --saes bt10k=matched:...,gs16k=scope:16k --out controls/coding_null.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import geoae.baselines.sae_data as D  # noqa: E402
from geoae.baselines.sae_coding import analyze_coding  # noqa: E402
from geoae.baselines.sae_controls import sample_pairs_nonsibling
from geoae.baselines.sae_controls import summarize_pairs  # noqa: E402
from geoae.baselines.sae_space import load_sae_space
from geoae.baselines.sae_space import parse_saes  # noqa: E402


def null_pairs(data, n, seed):
    fam = {}
    for p in data["pairs"]:
        for side in ("a", "b"):
            fam[p[side]] = p["family"]
    pairs, n_cand = sample_pairs_nonsibling(fam.keys(), fam, n, seed)
    return [{"a": a, "b": b, "family": f"null:{i}"} for i, (a, b) in enumerate(pairs)], n_cand


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--saes", required=True)
    ap.add_argument("--kinds", nargs="+", default=["range", "atlas"])
    ap.add_argument("--n_null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows_per_label", type=int, default=128)
    ap.add_argument("--control_rows", type=int, default=512)
    ap.add_argument("--encode_batch", type=int, default=256)
    ap.add_argument("--feature_budgets", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {}
    specs = parse_saes(args.saes)
    for kind in args.kinds:
        data = D.load_rows(Path(args.cache_dir) / kind, rows_per_label=args.rows_per_label,
                           control_rows=args.control_rows, seed=args.seed)
        h = torch.tensor(data["h"], device=dev, dtype=torch.float32)
        tok = torch.tensor(data["tok"], device=dev, dtype=torch.long)
        nulls, n_cand = null_pairs(data, args.n_null, args.seed)
        real = data["pairs"]
        print(f"[null] {kind}: rows={len(h)} real pairs={len(real)} null pairs={len(nulls)} (of {n_cand} candidates)", flush=True)
        for name, spec in specs.items():
            sae = load_sae_space(spec, dev, name)
            feats = torch.empty((len(h), sae.n_features), device=dev)
            for s in range(0, len(h), args.encode_batch):
                feats[s:s + args.encode_batch] = sae.encode_raw(h[s:s + args.encode_batch],
                                                                 tok[s:s + args.encode_batch])
            res = analyze_coding(feats, data["memberships"], list(real) + nulls, sae.decoder_directions,
                                 data["fit"], ~data["fit"], controls=data["controls"],
                                 budgets=tuple(args.feature_budgets))
            recs = res["pairs"]
            out.setdefault(kind, {})[name] = {
                "n_rows": len(h), "n_candidates_nonsibling": n_cand,
                "real_pairs": summarize_pairs(recs[:len(real)]),
                "null_pairs": summarize_pairs(recs[len(real):]),
                "null_pair_labels": [[p["a"], p["b"]] for p in nulls],
            }
            r, n = out[kind][name]["real_pairs"], out[kind][name]["null_pairs"]
            print(f"[null] {kind} {name:8s} real same={r['same_feature_rate']:.2f} cos>=.9={r['rate_top1_cos_ge_thr']:.2f} "
                  f"| null same={n['same_feature_rate']:.2f} cos>=.9={n['rate_top1_cos_ge_thr']:.2f} "
                  f"interp={n['interpretation_counts']}", flush=True)
            del feats, sae
            torch.cuda.empty_cache()
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=1))
        del h, tok


if __name__ == "__main__":
    main()
