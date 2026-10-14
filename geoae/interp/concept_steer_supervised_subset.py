"""Add the supervised AE latent direction to a frozen concept-steering subset.

Fit the original labelled documents and preserve the original full prompt batch
for generation and exact baseline/residual-label replay checks. Save only the
manifest's selected label_z outputs and reuse its existing residual-label outputs.
Every saved cell explicitly carries its original prompt indices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.hooks import SplicingHook
from geoae.interp import concept_steer_generate as CG
from geoae.interp import _shared as shared
from geoae.interp._context_logits import validate_layer
from geoae.interp.cluster_steering import Arm
from geoae.interp.concept_steer_extra import load_concept_docs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--sample_design", required=True)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    gen_path, design_path, out_path = map(Path, (cli.gen, cli.sample_design, cli.out))
    source = json.loads(gen_path.read_text())
    design = json.loads(design_path.read_text())
    meta = source["meta"]
    ds_name = design["dataset"]
    alpha = float(design["alpha"])
    alpha_key = str(alpha)
    concepts, selected = design["concepts"], design["selected_prompts"]
    source_hash = sha256(gen_path)
    if source_hash != design["generation_sha256"]:
        raise SystemExit("Generation SHA256 differs from the frozen subset manifest")
    if alpha not in meta["alphas"]:
        raise SystemExit("Subset strength is absent from the original generation run")
    if meta.get("intervention_forward") != "complete_lm_at_selected_block":
        raise SystemExit("Original run did not use complete-LM residual intervention")
    if meta["prompts"] != CG.PROMPTS[:len(meta["prompts"])]:
        raise SystemExit("Original prompt list differs from the shared generation prompts")
    if meta["max_new"] != 32 or design["prompts_per_concept"] != 2 or alpha != 0.5:
        raise SystemExit("This extension requires the frozen 2-prompt, alpha-0.5, 32-token subset")
    if len(set(concepts)) != len(concepts) or set(concepts) != set(selected):
        raise SystemExit("Manifest concept lists disagree or contain duplicates")
    for cn in concepts:
        ids = selected[cn]
        if (len(ids) != design["prompts_per_concept"] or len(set(ids)) != len(ids)
                or any(type(i) is not int or i < 0 or i >= len(meta["prompts"]) for i in ids)):
            raise SystemExit(f"Invalid selected prompt indices for {cn}: {ids}")
    if ds_name not in shared.DATASET_CONFIGS:
        shared.set_dataset(ds_name)
    cfg = shared.DATASET_CONFIGS[ds_name]
    by_class, kept = load_concept_docs(cfg, meta["n_fit"], meta["n_held"], meta["seed"], False)
    classes = cfg["classes"]
    if set(classes[c] for c in kept) != set(source["handles"][ds_name]):
        raise SystemExit("Kept classes differ from the original contrast group")
    if not set(concepts).issubset({classes[c] for c in kept}):
        raise SystemExit("Subset requests an unavailable concept")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path = out_path.with_suffix(".diagnostics.json")
    diagnostic = {"status": "running", "generation_sha256": source_hash,
                  "sample_design_sha256": sha256(design_path), "direction_fit": {},
                  "baseline_checks": {}, "label_replay_checks": {}, "failures": []}

    def save_diagnostic():
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False))

    def fail(message):
        diagnostic["status"] = "failed"
        diagnostic["failures"].append(message)
        save_diagnostic()
        raise SystemExit(message)

    save_diagnostic()
    dev = torch.device("cuda")
    act_meta = json.loads((Path(meta["activations_dir"]) / "meta.json").read_text())
    if act_meta["model"] != "google/gemma-3-4b-pt" or meta["layer"] not in act_meta["layers"]:
        fail("Activation metadata does not describe the original Gemma 3 4B generation layer")
    tk = AutoTokenizer.from_pretrained(act_meta["model"])
    tk.pad_token, tk.padding_side = tk.eos_token, "right"
    lm = load_lm(act_meta["model"], device=dev)
    downstream_blocks = validate_layer(lm, meta["layer"])
    if downstream_blocks != meta["downstream_blocks"]:
        fail("Loaded LM's downstream layer count differs from the original generation run")
    arm = Arm("bypass", meta["bypass"], dev)
    if arm.kind != "ae":
        fail("Original bypass handle is not an AE checkpoint")
    base_arm = Arm("base", meta["base"], dev)
    if base_arm.kind != "km":
        fail("Original base handle is not a k-means codebook")
    mean, std = base_arm.mean, base_arm.std
    if not (torch.allclose(arm.mean, mean) and torch.allclose(arm.std, std)):
        fail("Base and bypass normalization statistics differ")
    if len(mean) != act_meta["hidden_size"]:
        fail("AE normalization dimensions differ from the activation dump's hidden size")
    diagnostic["model"] = act_meta["model"]
    diagnostic["layer"] = meta["layer"]
    diagnostic["generation_settings"] = {"max_new_tokens": meta["max_new"], "do_sample": False,
                                         "alpha": alpha, "fit_documents_per_class": meta["n_fit"],
                                         "max_doc_tokens": meta["max_doc_tokens"],
                                         "generation_batch_size": len(meta["prompts"]),
                                         "saved_prompts_per_concept": design["prompts_per_concept"]}
    sx = torch.zeros(len(classes), len(mean), device=dev)
    sz = torch.zeros(len(classes), arm.C.shape[1], device=dev)
    count = torch.zeros(len(classes), device=dev)
    for c in kept:
        texts = by_class[c][:meta["n_fit"]]
        for h_raw, tok_ids, _ in CG.capture_docs(lm, tk, texts, meta["layer"], dev, meta["max_doc_tokens"]):
            x = (h_raw - mean) / std
            sx[c] += x.sum(0)
            sz[c] += arm.coords(x, tok_ids).sum(0)
            count[c] += len(x)
        print(f"[supervised-subset] fit {classes[c]}: {int(count[c])} tokens", flush=True)
    if bool((count[kept] <= 0).any()):
        fail("A kept class has no captured fit tokens")
    mux, muz = sx / count.clamp_min(1)[:, None], sz / count.clamp_min(1)[:, None]
    vx, vz = mux - mux[kept].mean(0), muz - muz[kept].mean(0)
    decoded_z = CG.decode_direction(arm.ae, vz)
    directions, handles = {}, {}
    for c in kept:
        cn = classes[c]
        old = source["handles"][ds_name][cn]["label"]
        error = abs(float(vx[c].norm()) - old["v_norm"])
        u = decoded_z[c]
        info = {"R_c": old["R_c"], "v_z_norm": float(vz[c].norm()),
                "u_norm": float(u.norm()), "cos_label": CG.cosine(vx[c], u),
                "label_v_norm_refit": float(vx[c].norm()), "label_v_norm_original": old["v_norm"],
                "label_v_norm_absolute_error": error, "fit_tokens": int(count[c])}
        diagnostic["direction_fit"][cn] = info
        if error > 1e-4:
            fail(f"{cn}: residual label norm changed by {error:.8g} (> 1e-4)")
        if not bool(torch.isfinite(u).all()) or float(u.norm()) < 1e-8:
            fail(f"{cn}: invalid decoded latent direction")
        directions[cn] = (u, old["R_c"])
        handles[cn] = {"label": old, "label_z": info, "cos_label_label_z": info["cos_label"]}
    direction_path = out_path.with_suffix(".directions.npz")
    np.savez_compressed(direction_path, concepts=np.asarray([classes[c] for c in kept]),
                        label=vx[kept].cpu().numpy(), label_z_latent=vz[kept].cpu().numpy(),
                        label_z=decoded_z[kept].cpu().numpy(),
                        label_unit=(vx[kept] / vx[kept].norm(dim=1, keepdim=True)).cpu().numpy(),
                        label_z_unit=(decoded_z[kept] / decoded_z[kept].norm(dim=1, keepdim=True)).cpu().numpy(),
                        R_c=np.asarray([directions[classes[c]][1] for c in kept]))
    save_diagnostic()

    tk.padding_side = "left"
    hook = SplicingHook(lm, meta["layer"])
    enc = tk(meta["prompts"], return_tensors="pt", padding=True).to(dev)
    length = enc["input_ids"].shape[1]
    batch_size = len(meta["prompts"])

    def generate(fn=None):
        if fn is not None:
            hook.activate(fn)
        try:
            out = lm.generate(**enc, max_new_tokens=meta["max_new"], do_sample=False,
                              pad_token_id=tk.pad_token_id)
        finally:
            if fn is not None:
                hook.deactivate()
        return out[:, length:]

    def ppl(cont):
        full_ids = torch.cat([enc["input_ids"], cont], 1)
        attention = torch.cat([enc["attention_mask"], torch.ones_like(cont)], 1)
        logits = lm(input_ids=full_ids, attention_mask=attention).logits[:, length - 1:-1].float()
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), cont,
                                                 reduction="none").mean(1).exp().cpu().numpy()

    base_cont = generate()
    base_text = [tk.decode(r, skip_special_tokens=True) for r in base_cont]
    expected = source["generations"]["unsteered"]["text"]
    if len(expected) != batch_size:
        fail("Original unsteered outputs do not cover the full prompt batch")
    same = [a == b for a, b in zip(base_text, expected)]
    diagnostic["baseline_full_batch"] = {"batch_size": batch_size, "matches": same,
                                          "text": base_text, "expected": expected}
    if not all(same):
        fail(f"Full original-batch baseline mismatch; inspect {diagnostic_path}")
    base_ppl = ppl(base_cont)
    base_d2 = [CG.distinct2(r.tolist()) for r in base_cont]
    save_diagnostic()

    class_index = {classes[c]: c for c in kept}
    generations, baselines = {}, {}
    for cn in concepts:
        ids = selected[cn]
        diagnostic["baseline_checks"][cn] = {"prompt_indices": ids, "matches": [same[i] for i in ids],
                                            "batch_size": batch_size,
                                            "full_batch_match": all(same)}
        baseline = {"prompt_indices": ids, "text": [base_text[i] for i in ids],
                    "ppl": base_ppl[ids].tolist(), "distinct2": [base_d2[i] for i in ids],
                    "log_ppl_ratio": [0.0] * len(ids), "edit_norm": 0.0}
        baselines[cn] = baseline
        original_label = source["generations"]["steered"][ds_name][cn]["label"][alpha_key]
        if len(original_label["text"]) != batch_size:
            fail(f"{cn}: original residual-label outputs do not cover the full prompt batch")
        u, R_c = directions[cn]
        label_log = []
        replay = generate(CG.make_label_edit_fn(vx[class_index[cn]], R_c, mean, std, alpha, label_log))
        replay_text = [tk.decode(r, skip_special_tokens=True) for r in replay]
        label_same = [a == b for a, b in zip(replay_text, original_label["text"])]
        diagnostic["label_replay_checks"][cn] = {"batch_size": batch_size, "matches": label_same,
                                                "prompt_indices": ids, "text": replay_text,
                                                "expected": original_label["text"],
                                                "edit_norm": float(np.mean(label_log))}
        if not all(label_same):
            fail(f"{cn}: original-batch residual-label replay mismatch; inspect {diagnostic_path}")
        reused_label = {key: ([value[i] for i in ids] if isinstance(value, list) else value)
                        for key, value in original_label.items()}
        reused_label.update(prompt_indices=ids, provenance="reused_exact_original_generation")
        log = []
        cont = generate(CG.make_label_edit_fn(u, R_c, mean, std, alpha, log))
        pp = ppl(cont)
        cell = {"prompt_indices": ids, "text": [tk.decode(cont[i], skip_special_tokens=True) for i in ids],
                "ppl": pp[ids].tolist(), "log_ppl_ratio": (np.log(pp) - np.log(base_ppl))[ids].tolist(),
                "distinct2": [CG.distinct2(cont[i].tolist()) for i in ids], "edit_norm": float(np.mean(log)),
                "provenance": "new_supervised_AE_latent_direction_generation"}
        generations[cn] = {"label": {"0.0": baseline, alpha_key: reused_label},
                           "label_z": {"0.0": baseline, alpha_key: cell}}
        print(f"[supervised-subset] generated {cn}: original batch {batch_size}, saved {len(ids)} prompts, alpha {alpha}, "
              f"cos(label,label_z)={handles[cn]['cos_label_label_z']:.6f}", flush=True)
        save_diagnostic()

    result = {"meta": {**meta, "handles": ["label", "label_z"], "alphas": [alpha], "datasets": ds_name,
                       "model": act_meta["model"], "do_sample": False,
                       "inherited_from": str(gen_path), "generation_sha256": source_hash,
                       "sample_design": str(design_path), "sample_design_sha256": sha256(design_path),
                       "partial_prompts": True, "prompt_indices_by_concept": selected,
                       "generation_batch_size": batch_size,
                       "batch_policy": "original full prompt batch; save frozen selected rows only",
                       "prompts_per_concept": design["prompts_per_concept"], "direction_file": str(direction_path),
                       "direction_sha256": sha256(direction_path), "downstream_blocks": downstream_blocks,
                       "intervention_forward": "complete_lm_at_selected_block",
                       "label_provenance": "reused_exact_original_generation",
                       "label_z_construction": "FIT class latent mean minus class-balanced FIT grand mean; "
                                               "decoder weight only; added in normalized residual space"},
              "handles": {ds_name: {cn: handles[cn] for cn in concepts}}, "coverage": {},
              "coverage_failures": [], "diagnostics": diagnostic,
              "generations": {"unsteered": source["generations"]["unsteered"],
                              "generated_baselines_by_concept": baselines, "steered": {ds_name: generations}}}
    diagnostic["status"] = "complete"
    save_diagnostic()
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[supervised-subset] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
