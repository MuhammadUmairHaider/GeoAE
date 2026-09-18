"""Controlled next-token number agreement with coordinate and label controls.

Fit singular/plural activation ranges on one set of subject nouns and prompt
templates; evaluate on disjoint nouns AND templates. Score the next-token choice
between ' is' and ' are'. Distractor noun number is balanced independently of
subject number. Edits operate only on the last prompt token, where ranges are fit.

Arms: original residual h, GeoAE z, and z in a fixed random orthogonal basis.
The rotated arm refits ranges in rotated coordinates and inverts the rotation
before decoding. It preserves reconstruction and Euclidean distances. A shuffled
fit-label arm is optional and leaves evaluation labels intact.

This is a constructed, forced-choice grammar task, not a natural-corpus benchmark.
Raw and reconstructed baselines, all-example and joint-correct metrics, neutral
prompt damage, per-example outcomes, and decoded edit norms are saved separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from geoae.interp import neuronlens as nl


# Explicit regular noun pairs avoid inflection heuristics and ambiguous forms.
NOUNS = [
    ("gardener", "gardeners"), ("teacher", "teachers"), ("singer", "singers"),
    ("doctor", "doctors"), ("painter", "painters"), ("farmer", "farmers"),
    ("driver", "drivers"), ("dancer", "dancers"), ("worker", "workers"),
    ("student", "students"), ("visitor", "visitors"), ("musician", "musicians"),
    ("baker", "bakers"), ("actor", "actors"), ("nurse", "nurses"),
    ("pilot", "pilots"), ("sailor", "sailors"), ("writer", "writers"),
    ("lawyer", "lawyers"), ("officer", "officers"), ("artist", "artists"),
    ("scientist", "scientists"), ("reporter", "reporters"), ("athlete", "athletes"),
    ("tourist", "tourists"), ("soldier", "soldiers"), ("chef", "chefs"),
    ("clerk", "clerks"), ("manager", "managers"), ("engineer", "engineers"),
    ("photographer", "photographers"), ("librarian", "librarians"),
]
TEMPLATES = {
    "fit": ["The {subject} near the {attractor}",
            "The {subject} behind the {attractor}",
            "The {subject} in front of the {attractor}"],
    "test": ["The {subject} beside the {attractor}",
             "The {subject} next to the {attractor}",
             "The {subject} across from the {attractor}"],
}
ATTRACTORS = [("window", "windows"), ("gate", "gates"), ("building", "buildings")]
NEUTRAL = [
    "The opposite of hot is", "The opposite of left is", "The opposite of open is",
    "The capital of France is", "The capital of Japan is", "The capital of Italy is",
    "Two plus two equals", "Three plus four equals", "Ten minus five equals",
    "Monday, Tuesday, Wednesday,", "January, February, March,",
    "The colors of the rainbow include red, orange,", "A triangle has three",
    "Water freezes at zero degrees", "A year has twelve", "A week has seven",
]


def build_examples(seed=42, n_fit_nouns=16, n_test_nouns=12):
    if min(n_fit_nouns, n_test_nouns) < 1 or n_fit_nouns + n_test_nouns > len(NOUNS):
        raise ValueError("choose positive, disjoint noun counts totaling at most 32")
    order = np.random.RandomState(seed).permutation(len(NOUNS))
    examples = []
    for split, indices in (("fit", order[:n_fit_nouns]),
                           ("test", order[n_fit_nouns:n_fit_nouns + n_test_nouns])):
        for ni in indices:
            for ti, template in enumerate(TEMPLATES[split]):
                for number in (0, 1):
                    for attractor_number in (0, 1):
                        examples.append({
                            "id": f"{split}:{NOUNS[ni][0]}:{ti}:{number}:{attractor_number}",
                            "split": split, "subject_lemma": NOUNS[ni][0],
                            "subject_number": number, "attractor_number": attractor_number,
                            "template": template,
                            "prompt": template.format(subject=NOUNS[ni][number],
                                attractor=ATTRACTORS[ti][attractor_number]),
                        })
    return examples


class OrthogonalMix:
    """Seeded permutation/pair-rotation composition; no dense D x D matrix.

    This is a structured random orthogonal transform, not a Haar sample.
    Rotating points and centroids together preserves Euclidean assignments.
    """

    def __init__(self, dim, seed=0, rounds=12, device="cpu"):
        if dim < 2 or rounds < 1:
            raise ValueError("rotation needs dim >= 2 and rounds >= 1")
        gen = torch.Generator().manual_seed(seed)
        self.steps = []
        for _ in range(rounds):
            perm = torch.randperm(dim, generator=gen)
            angle = torch.rand(dim // 2, generator=gen) * (2 * torch.pi)
            self.steps.append((perm.to(device), torch.argsort(perm).to(device),
                               angle.cos().to(device), angle.sin().to(device)))

    @staticmethod
    def _turn(x, c, s):
        n = c.numel() * 2
        pairs = x[..., :n].reshape(*x.shape[:-1], -1, 2)
        u, v = pairs[..., 0], pairs[..., 1]
        y = torch.stack((c * u + s * v, -s * u + c * v), dim=-1).flatten(-2)
        return torch.cat((y, x[..., n:]), dim=-1) if n < x.shape[-1] else y

    def forward(self, x):
        for perm, _, c, s in self.steps:
            x = self._turn(x[..., perm], c, s)
        return x

    def inverse(self, x):
        for _, inv, c, s in reversed(self.steps):
            x = self._turn(x, c, -s)[..., inv]
        return x


def fit_edits(acts, labels, source, tao, percent, saliency="abs"):
    target, other = acts[labels == source], acts[labels != source]
    if min(len(target), len(other)) < 2:
        raise ValueError("range fitting needs at least two examples of each number")
    mu_c, sd_c = target.mean(0), target.std(0)
    mu_o, sd_o = other.mean(0), other.std(0)
    score = np.abs(target).mean(0) if saliency == "abs" else nl.dprime_saliency(acts, labels == source)
    salient = nl.select_top(score, p=percent)
    lo, hi, _, _ = nl.fit_ranges(target, salient, tao)
    return dict(mu_c=mu_c, sd_c=sd_c, mu_o=mu_o, sd_o=sd_o,
                salient=salient, lo=lo, hi=hi)


def make_edit(stats, mode, alpha, device):
    lo, hi = stats["lo"], stats["hi"]
    if mode == "global":
        hi = np.full_like(hi, np.inf)
        lo = -hi
    elif mode == "salient":
        lo = np.where(stats["salient"], -np.inf, np.inf).astype(np.float32)
        hi = -lo
    if mode == "transport":
        return nl.transport_edit(stats["mu_c"], stats["sd_c"], stats["mu_o"],
                                 stats["sd_o"], lo, hi, alpha, device)
    if mode not in ("global", "salient", "range"):
        raise ValueError(f"unknown edit mode: {mode}")
    return nl.shift_edit(stats["mu_c"] - stats["mu_o"], lo, hi, alpha, device)


def make_last_patch(ae, mean, std, edit=None, rotation=None, raw=False):
    """A local-position intervention; record its coordinate-space displacement."""
    norms = []

    @torch.no_grad()
    def patch(hs):
        h = hs[:, -1, :].float()
        a = h if raw else ae.encoder((h - mean) / std)
        if rotation is not None:
            a = rotation.forward(a)
        changed = a if edit is None else edit(a)
        norms.extend((changed - a).norm(dim=-1).cpu().tolist())
        if rotation is not None:
            changed = rotation.inverse(changed)
        out_h = changed if raw else ae.decoder(changed) * std + mean
        out = hs.clone()
        out[:, -1, :] = out_h.to(hs.dtype)
        return out

    return patch, norms


@torch.inference_mode()
def forward_prompts(lm, tokenizer, prompts, device, layer, batch_size, patch=None, capture=False):
    from geoae.hooks import SplicingHook
    logits, activations = [], []
    hook = SplicingHook(lm, layer)

    def capture_fn(hs):
        activations.append(hs[:, -1, :].float().cpu())
        return hs

    if patch is not None or capture:
        hook.activate(patch if patch is not None else capture_fn)
    try:
        for start in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[start:start + batch_size], padding=True,
                            return_tensors="pt").to(device)
            output = lm(**enc, use_cache=False).logits
            logits.append(output[:, -1, :].float().cpu())
    finally:
        hook.deactivate()
    return torch.cat(logits), torch.cat(activations) if capture else None


def outcome(logits, reference, labels, token_ids):
    """Per-example measurements, including full-vocabulary KL(ref || edited)."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    choices = logits[:, token_ids]
    ref_choices = reference[:, token_ids]
    rows = torch.arange(len(labels))
    signed = (choices[:, 1] - choices[:, 0]) * (2 * labels - 1)
    ref_signed = (ref_choices[:, 1] - ref_choices[:, 0]) * (2 * labels - 1)
    logp, ref_logp = F.log_softmax(logits, -1), F.log_softmax(reference, -1)
    return {
        "correct": (signed > 0).numpy(),
        "ref_correct": (ref_signed > 0).numpy(),
        "counterpart_top1": (logits.argmax(-1) == torch.tensor(token_ids)[1 - labels]).numpy(),
        "gold_pair_probability": choices.softmax(-1)[rows, labels].numpy(),
        "gold_logit_margin_change": (signed - ref_signed).numpy(),
        "top1_change": (logits.argmax(-1) != reference.argmax(-1)).numpy(),
        "kl": (ref_logp.exp() * (ref_logp - logp)).sum(-1).numpy(),
    }


def summarize(values, labels, source, mask):
    target, comp = mask & (labels == source), mask & (labels != source)
    if not target.any() or not comp.any():
        return {"n_target": int(target.sum()), "n_complement": int(comp.sum()),
                "target_drop": None, "complement_drop": None, "selectivity": None}
    drop = values["ref_correct"].astype(float) - values["correct"].astype(float)
    td, cd = float(drop[target].mean()), float(drop[comp].mean())
    return {
        "n_target": int(target.sum()), "n_complement": int(comp.sum()),
        "target_drop": td, "complement_drop": cd, "selectivity": td - cd,
        "target_counterpart_top1": float(values["counterpart_top1"][target].mean()),
        "target_gold_pair_probability": float(values["gold_pair_probability"][target].mean()),
        "target_gold_margin_change": float(values["gold_logit_margin_change"][target].mean()),
        "complement_top1_change": float(values["top1_change"][comp].mean()),
        "target_kl": float(values["kl"][target].mean()),
        "complement_kl": float(values["kl"][comp].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_fit_nouns", type=int, default=16)
    ap.add_argument("--n_test_nouns", type=int, default=12)
    ap.add_argument("--tao", type=float, default=2.0)
    ap.add_argument("--percent", type=float, default=0.3)
    ap.add_argument("--saliency", choices=["abs", "dprime"], default="abs")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--modes", nargs="+", choices=["global", "salient", "range", "transport"],
                    default=["global", "salient", "range", "transport"])
    ap.add_argument("--rotation_seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--rotation_rounds", type=int, default=12)
    ap.add_argument("--shuffle_labels", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry_run", action="store_true", help="write the task manifest without loading models")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not np.isfinite(args.tao) or args.tao <= 0 or not 0 < args.percent <= 1:
        ap.error("tao must be positive and finite; percent must be in (0, 1]")
    if args.batch_size < 1 or any(not np.isfinite(a) or a < 0 for a in args.alphas):
        ap.error("batch_size must be positive; alphas must be finite and nonnegative")
    if args.rotation_rounds < 1 or len(set(args.rotation_seeds)) != len(args.rotation_seeds):
        ap.error("rotation_rounds must be positive and rotation seeds must be distinct")
    if args.smoke:
        args.n_fit_nouns, args.n_test_nouns = 2, 2
        args.alphas = [1.0]
        args.modes = ["global", "range", "transport"]
        args.rotation_seeds = args.rotation_seeds[:1]
    examples = build_examples(args.seed, args.n_fit_nouns, args.n_test_nouns)
    fit = [r for r in examples if r["split"] == "fit"]
    test = [r for r in examples if r["split"] == "test"]
    out_path = Path(args.out)
    if out_path.exists():
        ap.error(f"output exists: {out_path}; choose a new path to preserve existing results")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "manifest" if args.dry_run else "running",
        "meta": {**vars(args), "task": "constructed_subject_verb_number_agreement",
                 "readout": [" is", " are"], "edit_position": "last prompt token only",
                 "range_fit_position": "last prompt token only",
                 "rotation": "structured orthogonal permutation/pair rotations",
                 "selection": "report all examples and a shared base+recon correct subset",
                 "selectivity_sign": "target_drop - complement_drop; higher is better"},
        "fit_examples": fit, "test_examples": test, "neutral_prompts": NEUTRAL,
        "arms": {},
    }

    def save():
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(out_path)

    if args.dry_run:
        save()
        print(f"Manifest: {len(fit)} fit / {len(test)} test / {len(NEUTRAL)} neutral -> {out_path}")
        return

    from transformers import AutoTokenizer
    from geoae.checkpoint import load_ae_checkpoint, load_lm
    from geoae.interp._shared import ae_fingerprint
    from geoae.seeding import seed_everything
    seed_everything(args.seed)
    device = torch.device(args.device)
    ae, mean, std, ck = load_ae_checkpoint(args.checkpoint, device)
    layer = ck["config"]["data"]["target_layer"]
    model_name = ck["config"]["extraction"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokens = [tokenizer.encode(s, add_special_tokens=False) for s in result["meta"]["readout"]]
    if any(len(x) != 1 for x in tokens) or tokens[0] == tokens[1]:
        raise ValueError("this task requires distinct single-token ' is' and ' are' continuations")
    token_ids = [x[0] for x in tokens]
    lm = load_lm(model_name, device=device)
    result["meta"].update(model_name=model_name, layer=layer, ae_sha=ae_fingerprint(ae),
                          checkpoint_epoch=ck.get("epoch"), token_ids=token_ids)
    del ck
    fit_prompts = [r["prompt"] for r in fit]
    prompts = [r["prompt"] for r in test] + NEUTRAL
    fit_labels = np.array([r["subject_number"] for r in fit])
    test_labels = np.array([r["subject_number"] for r in test])
    n_test = len(test)

    def run(texts, patch=None, capture=False):
        return forward_prompts(lm, tokenizer, texts, device, layer, args.batch_size, patch, capture)

    print("Capturing fit and evaluation activations ...", flush=True)
    fit_logits, h_fit = run(fit_prompts, capture=True)
    base, h_eval = run(prompts, capture=True)
    recon_patch, _ = make_last_patch(ae, mean, std)
    recon, _ = run(prompts, recon_patch)
    with torch.inference_mode():
        z_fit = ae.encoder((h_fit.to(device) - mean) / std)
        z_eval = ae.encoder((h_eval.to(device) - mean) / std)
    correct_base = outcome(base[:n_test], base[:n_test], test_labels, token_ids)["correct"]
    correct_recon = outcome(recon[:n_test], recon[:n_test], test_labels, token_ids)["correct"]
    joint = correct_base & correct_recon
    result["baselines"] = {
        "fit_base_pair_accuracy": float(outcome(fit_logits, fit_logits, fit_labels, token_ids)["correct"].mean()),
        "test_base_pair_accuracy": float(correct_base.mean()),
        "test_recon_pair_accuracy": float(correct_recon.mean()),
        "joint_correct_count": int(joint.sum()),
        "reconstruction": {k: v.tolist() for k, v in outcome(recon[:n_test], base[:n_test], test_labels, token_ids).items()},
        "neutral_reconstruction": {
            k: v.tolist() for k, v in outcome(recon[n_test:], base[n_test:],
                                               np.zeros(len(NEUTRAL)), token_ids).items()
            if k in ("top1_change", "kl")
        },
    }
    spaces = [("h", h_fit.to(device), h_eval.to(device), None, True, False),
              ("z", z_fit, z_eval, None, False, False)]
    for seed in args.rotation_seeds:
        rotation = OrthogonalMix(z_fit.shape[1], seed, args.rotation_rounds, device)
        spaces.append((f"z_rot{seed}", rotation.forward(z_fit), rotation.forward(z_eval),
                       rotation, False, False))
    if args.shuffle_labels:
        spaces.append(("z_shuffled", z_fit, z_eval, None, False, True))
    shuffled = np.random.RandomState(args.seed + 1009).permutation(fit_labels)
    save()

    for name, a_fit, a_eval, rotation, raw, shuffle in spaces:
        reference = base if raw else recon
        fit_array = a_fit.detach().cpu().numpy()
        for source in (0, 1):
            stats = fit_edits(fit_array, shuffled if shuffle else fit_labels, source,
                              args.tao, args.percent, args.saliency)
            gate = ((a_eval.cpu().numpy() >= stats["lo"]) &
                    (a_eval.cpu().numpy() <= stats["hi"]))[:, stats["salient"]]
            for mode in args.modes:
                for alpha in args.alphas:
                    key = f"{name}:suppress_{'plural' if source else 'singular'}:{mode}:a{alpha}"
                    edit = make_edit(stats, mode, alpha, device)
                    patch, coordinate_norms = make_last_patch(ae, mean, std, edit, rotation, raw)
                    edited, _ = run(prompts, patch)
                    values = outcome(edited[:n_test], reference[:n_test], test_labels, token_ids)
                    neutral = outcome(edited[n_test:], reference[n_test:], np.zeros(len(NEUTRAL)), token_ids)
                    with torch.inference_mode():
                        delta = edit(a_eval) - a_eval
                        if rotation is not None:
                            delta = rotation.inverse(delta)
                        raw_delta = delta if raw else ae.decoder(delta) * std
                    all_summary = summarize(values, test_labels, source, np.ones(n_test, dtype=bool))
                    paired = summarize(values, test_labels, source, joint)
                    result["arms"][key] = {
                        "all": all_summary, "joint_correct": paired,
                        "matching_attractor": summarize(values, test_labels, source,
                            np.array([r["subject_number"] == r["attractor_number"] for r in test])),
                        "opposite_attractor": summarize(values, test_labels, source,
                            np.array([r["subject_number"] != r["attractor_number"] for r in test])),
                        "neutral": {"n": len(NEUTRAL), "top1_change": float(neutral["top1_change"].mean()),
                                    "kl": float(neutral["kl"].mean())},
                        "gate_fire_target": float(gate[:n_test][test_labels == source].mean()),
                        "gate_fire_complement": float(gate[:n_test][test_labels != source].mean()),
                        "n_salient": int(stats["salient"].sum()),
                        "coordinate_edit_norm": coordinate_norms,
                        "decoded_edit_norm": raw_delta.norm(dim=-1).cpu().tolist(),
                        "rows": {k: v.tolist() for k, v in values.items()},
                    }
                    print(f"{key}: all selectivity={all_summary['selectivity']:.4f}; "
                          f"neutral KL={result['arms'][key]['neutral']['kl']:.4f}", flush=True)
                    save()
    result["status"] = "complete"
    save()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
