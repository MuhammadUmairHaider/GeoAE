"""Atlas class corpus, stage `joint_correct`: the joint-correct pools (docs the base LM AND the AE-spliced LM classify
correctly) built with the existing shared.build_joint_correct_set, plus the per-class accuracy report and the kept-class
decision (rule fixed in atlas_classes.RULES: >= 40 joint-correct TEST docs).

Writes under <out_dir>/s<seed>/ : joint_correct.json (the full cache, tool format), joint_correct_kept.json (only the kept
classes' docs -- what the intervention stage reads, so dropped classes are neither targets nor complement) and
jc_report.json (per-class base-LM accuracy, AE-splice accuracy, docs tried, pool sizes, kept/dropped, --concepts string).

    python eval_out/atlas_classes_joint_correct.py --seed 0 --checkpoint <step_0014450.pt>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset  # noqa: E402

from geoae.checkpoint import load_lm  # noqa: E402
from geoae.evaluate import load_ae_from_checkpoint, sanity_check_splice  # noqa: E402
from geoae.hooks import SplicingHook, TokenIdTap  # noqa: E402
from geoae.interp import _shared as shared  # noqa: E402
from geoae.interp import atlas_classes as AC  # noqa: E402
from geoae.interp import neuronlens as nl  # noqa: E402
from geoae.seeding import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="results/delta/d6144/interventions/atlas_classes")
    ap.add_argument("--classes", default="all", help="comma list of class indices (smoke) or all")
    ap.add_argument("--n_fit", type=int, default=80)
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--max_try", type=int, default=600, help="docs tried per (class, split)")
    ap.add_argument("--min_test", type=int, default=AC.RULES["min_joint_correct_test"])
    ap.add_argument("--jc_batch_size", type=int, default=16)
    ap.add_argument("--tag", default="", help="suffix for the output files (smoke)")
    args = ap.parse_args()
    t0 = time.time()
    name = f"atlas_s{args.seed}"
    cfg = shared.set_dataset(name)
    sd = Path(args.out_dir) / f"s{args.seed}"
    full, kept_f, rep_f = (sd / f"joint_correct{args.tag}.json", sd / f"joint_correct_kept{args.tag}.json",
                           sd / f"jc_report{args.tag}.json")
    classes = list(range(len(shared.CLASSES))) if args.classes == "all" else [int(x) for x in args.classes.split(",")]
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, norm, mcfg = load_ae_from_checkpoint(Path(args.checkpoint), device, allow_token_bias=True)
    model_name, layer = mcfg["extraction"]["model_name"], mcfg["data"]["target_layer"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    lm = load_lm(model_name, device_map="auto")
    sanity_check_splice(lm, layer, tokenizer, device)
    hook = SplicingHook(lm, layer)
    tap = TokenIdTap(lm) if ae.has_token_bias else None
    z_recon = nl.make_z_edit(None, ae, norm["mean"], norm["std"], device, tap=tap)
    ds = load_dataset("parquet", data_files=str(Path(cfg["local_parquet"])), split="train")
    t1 = time.time()
    shared.build_joint_correct_set(
        ds, cfg, tokenizer, lm, device, hook, z_recon, full, args.n_fit + args.n_eval, name, model_name=model_name,
        ae_sha=shared.ae_fingerprint(ae), batch_size=args.jc_batch_size, tag="atlas-jc", classes=classes,
        need_by_split={"fit": args.n_fit, "test": args.n_eval}, max_try=args.max_try)
    t_build = time.time() - t1
    J = json.load(open(full))
    st = J["stats"]
    rows, kept = [], []
    print(f"\n{'class':<40}{'base_acc':>9}{'AE_acc':>8}{'tried':>7}{'fit_jc':>8}{'test_jc':>8}  verdict")
    for c in classes:
        f, t = st[f"{c}/fit"], st[f"{c}/test"]
        n_fit = sum(1 for d in J["docs"] if d["label"] == c and d["split"] == "fit")
        n_test = sum(1 for d in J["docs"] if d["label"] == c and d["split"] == "test")
        tried = f["n_tried"] + t["n_tried"]
        r = dict(cls=c, name=shared.CLASSES[c], n_tried_fit=f["n_tried"], n_tried_test=t["n_tried"],
                 base_acc=round((f["base_correct"] + t["base_correct"]) / max(tried, 1), 4),
                 ae_splice_acc=round((f["recon_correct"] + t["recon_correct"]) / max(tried, 1), 4),
                 joint_acc=round((f["both_correct"] + t["both_correct"]) / max(tried, 1), 4),
                 base_acc_test=round(t["base_correct"] / max(t["n_tried"], 1), 4),
                 ae_splice_acc_test=round(t["recon_correct"] / max(t["n_tried"], 1), 4),
                 n_joint_fit=n_fit, n_joint_test=n_test, kept=n_test >= args.min_test)
        rows.append(r)
        if r["kept"]:
            kept.append(c)
        print(f"{r['name']:<40}{r['base_acc']:>9.3f}{r['ae_splice_acc']:>8.3f}{tried:>7}{n_fit:>8}{n_test:>8}  "
              f"{'keep' if r['kept'] else 'DROP (<' + str(args.min_test) + ' joint-correct test docs)'}")
    K = dict(J)
    K["docs"] = [d for d in J["docs"] if d["label"] in kept]
    K["classes_built"] = kept
    json.dump(K, open(kept_f, "w"))
    rep = dict(seed=args.seed, dataset=name, classes=rows, kept=kept, dropped=[c for c in classes if c not in kept],
               concepts_arg=",".join(map(str, kept)), min_joint_correct_test=args.min_test, n_fit=args.n_fit,
               n_eval=args.n_eval, max_try=args.max_try, checkpoint=args.checkpoint, ae_sha=J["ae_sha"],
               prompt_sha=J["prompt_sha"], seconds_build=round(t_build, 1), seconds_total=round(time.time() - t0, 1))
    rep_f.write_text(json.dumps(rep, indent=1))
    print(f"[atlas-jc] kept {len(kept)}/{len(classes)}: --concepts {rep['concepts_arg']}  "
          f"(build {t_build:.0f}s, total {time.time() - t0:.0f}s) -> {rep_f}")


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    import os
    os._exit(0)
