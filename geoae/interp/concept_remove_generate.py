"""
Generation-level CONCEPT REMOVAL: base residual (h) vs bypass AE latent (z).

range_intervention_compare.py showed on DB14 (last-token classification decision) that
NeuronLens range edits done in the bypass latent remove the target class about as well as in
the base residual, with ~4-5/100 LESS collateral damage (rm_range_comp 86/20 -> 89/15;
st_range_a1 86/24 -> 90/20; st_transport_a1 86/28 -> 88/23). This test asks whether that holds
in GENERATED TEXT rather than a single forced-choice decision.

Per concept c (a DB14 class here), fit stats are built from FIT documents in BOTH substrates:
  h   normalised residual x = (h_raw - mean) / std                              (D = 3072)
  z   bypass latent z = ae.encode(x, tok)  (Arm.coords, token bias handled)      (Lz = 6144)
from accumulated per-class count/sum/sum-of-squares (streamed, not held in memory), giving
mu_c, sd_c and "rest" (the OTHER kept classes' tokens pooled): mu_rest, sd_rest. d' =
|mu_c - mu_rest| / sqrt(0.5*(sd_c^2+sd_rest^2)) picks the top --percent salient coordinates;
the NeuronLens range is [mu_c -/+ tao*sd_c] on those coordinates only (+inf/-inf, i.e. an
EMPTY interval, elsewhere -- see class_dprime_and_range). `comp` is the CLASS-BALANCED mean of
the other classes' own means (not token-weighted, unlike mu_rest -- see
class_balanced_mean_excluding); it is the NeuronLens gate's replacement value.

Ops (identical operator in both substrates; alpha in class-gap units, r = mu_c - mu_rest):
  rm_range_comp    neuronlens.gate_edit(lo, hi, comp)               -- no alpha, runs once
  st_range         neuronlens.shift_edit(r, lo, hi, alpha)          -- alpha in --alphas
  st_transport     neuronlens.transport_edit(mu_c, sd_c, mu_rest, sd_rest, lo, hi, alpha)

WHY A SECOND, POSITION-LEVEL GATE (--pos_gate llr, default). The per-COORDINATE range
[mu_c -/+ tao*sd_c] is fit and evaluated over ALL non-BOS positions of ALL fit/held documents
-- not a single last-token decision the way range_intervention_compare.py's classification
setting is. The --diag_only HELD-document check below found that this coordinate gate fires
on ~95% of positions of EVERY class, target or not: a handful of salient coordinates each
being individually "close to class c's mean" is a weak, nearly universal condition at the
token level, so the coordinate gate alone barely restricts WHERE an op fires -- ops degenerate
into an almost-everywhere edit. --pos_gate llr adds a genuinely class-selective POSITION
gate: LLR_c(pos) = sum over salient coordinates j of
    log N(a_j; mu_c_j, sd_c_j) - log N(a_j; mu_rest_j, sd_rest_j)
(the substrate's OWN coordinates: normalised x for h, the bypass latent for z) -- a position
fires only when the SALIENT coordinates, taken together, look more like class c than like the
rest. --pos_gate none recovers the old, coordinate-gate-only behaviour for comparison.

THRESHOLD: --gate_target_rate (default 0.8) CALIBRATES the LLR threshold per concept AND per
substrate, rather than using one fixed --llr_thresh (default 0.0) for both. h sums 922 salient
coordinates' log-density terms, z sums 1843 (2x the salient count, since Lz=2*D here) -- a raw
LLR score is NOT on the same scale between them, so a fixed threshold made z's gate fire on
~3x more of its COMPLEMENT positions than h's at the SAME thresh=0.0 (0.30 vs 0.10 in a
--diag_only check), i.e. the two substrates would not be tested with comparably strict gates.
calibrate_threshold(llr_of_class_c's_own_FIT_positions, rate) instead picks the (1-rate)
quantile of class c's OWN fit-time LLR scores, so the gate fires on `rate` of class c's own
positions BY CONSTRUCTION in every substrate alike -- the fair comparison is then the
COMPLEMENT fire rate at that matched target rate (and the substrate-independent AUROC).
--gate_target_rate <= 0 skips calibration and uses --llr_thresh everywhere.

Splice, h substrate: x' = x + g*(edit(x) - x) -- the edit fn is wrapped so it sees NORMALISED
x, not the raw residual (neuronlens.make_h_edit itself has no normalisation baked in); g is
the 0/1 position gate (all-1s under --pos_gate none).
Splice, z substrate, --z_mode delta (default): z = ae.encode(x, tok); x' = x +
g*ae.decoder(edit(z) - z). The decoder is linear and bias-free, and the token-bypass table
only depends on the token, not on z -- see concept_steer_generate.decode_direction / this
file's own tests for the cancellation -- so an IDENTITY edit (or g=0 throughout) leaves
x' == x exactly: the unedited text is byte-identical to the h substrate's, and the AE's
reconstruction error never enters except through the edit itself. --z_mode splice instead
uses z' = where(g, edit(z), z), x' = ae.decode(z', tok) (neuronlens.make_z_edit's own round
trip, gated); its own baseline is the unedited AE-recon splice (make_z_edit(None, ...)),
generated once per concept and scored against separately, since with splice the "unedited" z
text is NOT the same as h's. g is computed once per forward call from LLR_c (see above) and
is IN ADDITION to the coordinate range lo/hi already inside each op -- both gates apply.

Prompts: DB14 HELD documents (after the FIT documents) -- --n_target target prompts from
class c, plus --n_comp_per_class from EACH other kept class (13 for full DB14; a fixed
3-class pool under --smoke). A prompt is the document's first --prompt_tokens tokens, decoded
back to text. The SAME prompt set (and order) is reused for every substrate/op/alpha of a
concept's run, batched together (target + complement, left-padded), greedy --max_new tokens.
concept_remove_judge.py does the (independent) LLM forced-choice class judging.

--diag_only runs the FIT pass, the threshold calibration pass, and a HELD-document diagnostic
(no generation, no prompts, ~1-2 min at full size) and exits: per concept x substrate, the
coordinate-gate share inside [lo,hi] on class-c held positions vs other-class held positions,
d' of the salient coordinates (median, max), the position gate's AUROC (LLR_c score, class-c
held positions vs the rest), and its fire rate at BOTH the fixed and the calibrated threshold,
labelled -- run this FIRST to sanity-check gate selectivity before spending GPU time on
generation.

    python -u -m geoae.interp.concept_remove_generate --bypass <ckpt> \
        --out eval_out/concept_remove_generate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.hooks import SplicingHook, TokenIdTap
from geoae.interp import neuronlens as nl
from geoae.interp._shared import DATASET_CONFIGS
from geoae.interp.cluster_steering import Arm
# Reused, UNMODIFIED, from concept_steer_generate.py (per the no-modify constraint on that
# file -- these are its own module-level, importable functions).
from geoae.interp.concept_steer_generate import capture_docs, concept_description, distinct2, load_concept_docs

FIT_MAX_TOKENS = 128            # not a CLI flag (the spec's --ops/--alphas/etc. list omits one);
                                 # matches concept_steer_generate.py's own --max_doc_tokens default.


# ---------------------------------------------------------------------------
# Pure, unit-testable helpers
# ---------------------------------------------------------------------------

def class_dprime_and_range(mu_c: np.ndarray, sd_c: np.ndarray, mu_rest: np.ndarray,
                            sd_rest: np.ndarray, percent: float, tao: float):
    """d' = |mu_c - mu_rest| / sqrt(0.5*(sd_c^2+sd_rest^2)); the top --percent coordinates by
    d' are salient; the NeuronLens range lo/hi = mu_c -/+ tao*sd_c on salient coordinates,
    (+inf, -inf) -- an EMPTY interval, so untouched -- elsewhere. Mirrors
    neuronlens.dprime_saliency + select_top + fit_ranges, but built from already-accumulated
    per-class mean/std (a streamed capture) rather than a raw in-memory activation array."""
    mu_c, sd_c, mu_rest, sd_rest = map(np.asarray, (mu_c, sd_c, mu_rest, sd_rest))
    d = np.abs(mu_c - mu_rest) / np.sqrt(0.5 * (sd_c ** 2 + sd_rest ** 2) + 1e-12)
    D = len(d)
    k = max(1, min(int(round(percent * D)), D))
    salient = np.zeros(D, dtype=bool)
    salient[np.argsort(-d)[:k]] = True
    lo, hi = np.full(D, np.inf), np.full(D, -np.inf)
    lo[salient] = mu_c[salient] - tao * sd_c[salient]
    hi[salient] = mu_c[salient] + tao * sd_c[salient]
    return d, salient, lo, hi


def class_balanced_mean_excluding(class_means: dict, exclude) -> np.ndarray:
    """`comp`: the CLASS-BALANCED mean of the OTHER classes' own means (every class weighted
    equally, regardless of its token count) -- the NeuronLens gate's replacement value.
    Deliberately NOT the same as mu_rest (which pools tokens, so a token-heavy class would
    dominate it)."""
    others = [m for c, m in class_means.items() if c != exclude]
    return np.mean(np.stack(others), axis=0)


def llr_scores(a: torch.Tensor, mu_c: torch.Tensor, sd_c: torch.Tensor, mu_rest: torch.Tensor,
               sd_rest: torch.Tensor, salient: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """LLR_c per row of `a` (N, D): sum over SALIENT coordinates j of
        log N(a_j; mu_c_j, sd_c_j) - log N(a_j; mu_rest_j, sd_rest_j)
    -- the log-likelihood-ratio of class c vs the rest, restricted to the coordinates
    class_dprime_and_range already called salient (the -0.5*log(2*pi) terms cancel identically
    between the two Gaussians, so they're omitted). sd is clamped away from 0 so a near-constant
    coordinate can't produce +-inf. Returns (N,); an all-False `salient` gives all-zero scores
    (LLR_c > 0 threshold then never fires, matching an empty coordinate range)."""
    if not bool(salient.any()):
        return torch.zeros(a.shape[0], device=a.device, dtype=a.dtype)
    sd_c = sd_c.clamp_min(eps)
    sd_rest = sd_rest.clamp_min(eps)
    a_s = a[:, salient]
    mu_c_s, sd_c_s = mu_c[salient], sd_c[salient]
    mu_rest_s, sd_rest_s = mu_rest[salient], sd_rest[salient]
    ll_c = -torch.log(sd_c_s) - 0.5 * ((a_s - mu_c_s) / sd_c_s) ** 2
    ll_rest = -torch.log(sd_rest_s) - 0.5 * ((a_s - mu_rest_s) / sd_rest_s) ** 2
    return (ll_c - ll_rest).sum(dim=1)


def make_edit_splice(substrate: str, z_mode: str, edit_fn, lo: np.ndarray, hi: np.ndarray,
                      mean: torch.Tensor, std: torch.Tensor, ae, tap, device, log: dict,
                      pos_gate: str = "llr", llr_thresh: float = 0.0,
                      mu_c=None, sd_c=None, mu_rest=None, sd_rest=None, n_tgt: int = 0):
    """One splice function for either substrate. `edit_fn` operates on normalised x (h) or on
    the bypass latent z (z) -- exactly what neuronlens.gate_edit/shift_edit/transport_edit
    return. `lo`/`hi` (same units as edit_fn's input) are used ONLY for the coordinate-gate
    diagnostic below; the edit itself already has its own copy baked in via neuronlens.

    `pos_gate`: "llr" (default) computes a 0/1 POSITION gate g = (LLR_c(a) > llr_thresh) from
    `mu_c`/`sd_c`/`mu_rest`/`sd_rest` (torch tensors, substrate's own coordinates) via
    llr_scores, and applies the op only at gated positions -- "none" is g == 1 everywhere (the
    old, coordinate-gate-only behaviour). `n_tgt`: how many of the batch's rows (B) are TARGET
    prompts (the rest are complement) -- purely for the separate target/complement position-
    gate fire-rate diagnostic logged below; it does not affect the edit itself.

    h            x' = x + g*(edit_fn(x) - x)
    z (delta)    x' = x + g*ae.decoder(edit_fn(z) - z)          (identity edit or g=0 -> x'==x)
    z (splice)   z' = where(g, edit_fn(z), z); x' = ae.decode(z', tok)  (neuronlens.make_z_edit's
                 own round trip, gated; edit_fn=None there is the z baseline, generated separately)
    """
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=device)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=device)
    salient = torch.isfinite(lo_t)                        # derived: non-salient dims have lo=+inf
    if pos_gate == "llr":
        mu_c_t = torch.as_tensor(mu_c, dtype=torch.float32, device=device)
        sd_c_t = torch.as_tensor(sd_c, dtype=torch.float32, device=device)
        mu_rest_t = torch.as_tensor(mu_rest, dtype=torch.float32, device=device)
        sd_rest_t = torch.as_tensor(sd_rest, dtype=torch.float32, device=device)

    def fn(hs):
        B, T, D = hs.shape
        x = (hs.reshape(B * T, D).float() - mean) / std
        if substrate == "h":
            a = x
        else:
            tok = tap.ids_for(hs)
            a = ae.encode(x, tok if ae.has_token_bias else None)

        if pos_gate == "llr":
            llr = llr_scores(a, mu_c_t, sd_c_t, mu_rest_t, sd_rest_t, salient)
            g = (llr > llr_thresh).float()
        else:
            g = torch.ones(a.shape[0], device=device)

        if substrate == "h":
            x2 = x + g[:, None] * (edit_fn(a) - a)
        elif z_mode == "delta":
            x2 = x + g[:, None] * ae.decoder(edit_fn(a) - a)
        else:                                              # splice
            a2 = a + g[:, None] * (edit_fn(a) - a)
            x2 = ae.decode(a2, tok)

        if salient.any():
            fire = ((a[:, salient] >= lo_t[salient]) & (a[:, salient] <= hi_t[salient])).float().mean()
            log["fire"].append(float(fire))
        log["edit_norm"].append(float((x2 - x).norm(dim=1).mean()))
        g2 = g.reshape(B, T)
        if n_tgt > 0:
            log["pos_fire_tgt"].append(float(g2[:n_tgt].mean()))
        if n_tgt < B:
            log["pos_fire_comp"].append(float(g2[n_tgt:].mean()))
        return (x2 * std + mean).reshape(B, T, D).to(hs.dtype)
    return fn


def make_generate(lm, tk, enc, L: int, max_new: int, hook):
    """Copied from concept_steer_generate.py's main() (a nested closure there, not
    importable) -- identical logic, parameterised instead of closing over module globals."""
    def generate(fn=None):
        if fn is not None:
            hook.activate(fn)
        try:
            out = lm.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tk.pad_token_id)
        finally:
            if fn is not None:
                hook.deactivate()
        return out[:, L:]
    return generate


def make_ppl(lm, enc, L: int):
    """Copied from concept_steer_generate.py's main() (a nested closure there) -- identical."""
    def ppl(cont):
        ids = torch.cat([enc["input_ids"], cont], 1)
        am = torch.cat([enc["attention_mask"], (cont != -1).long()], 1)
        logits = lm(input_ids=ids, attention_mask=am).logits[:, L - 1:-1].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), cont, reduction="none")
        return nll.mean(1).exp().cpu().numpy()
    return ppl


def first_n_tokens_text(tk, doc_text: str, n: int) -> str:
    ids = tk(doc_text, truncation=True, max_length=n)["input_ids"]
    return tk.decode(ids[:n], skip_special_tokens=True)


def calibrate_threshold(llr_values, rate: float) -> float:
    """--gate_target_rate calibration: the LLR value such that `rate` (e.g. 0.8) of
    `llr_values` (class-c's OWN FIT positions) sit ABOVE it -- the (1 - rate) quantile, so
    the position gate fires on that share of class c's own fit-time positions BY
    CONSTRUCTION, in EVERY substrate alike (the fixed threshold instead depends on the
    substrate's raw LLR scale, which is not comparable across a 3072-d and a 6144-d sum of
    per-coordinate log-densities -- see the module docstring). Caller decides what `rate<=0`
    means (this function does not special-case it)."""
    vals = np.asarray(llr_values.cpu() if torch.is_tensor(llr_values) else llr_values, dtype=float)
    return float(np.quantile(vals, 1.0 - rate))


def diagnostic_cell(own: torch.Tensor, other: torch.Tensor, mu_c: torch.Tensor, sd_c: torch.Tensor,
                     mu_rest: torch.Tensor, sd_rest: torch.Tensor, salient: torch.Tensor,
                     lo: torch.Tensor, hi: torch.Tensor, thresholds: dict) -> dict:
    """The --diag_only HELD-document check for one (concept, substrate) cell: `own`/`other`
    are (N, D) position samples (all non-BOS positions of held documents) for class c and for
    the pooled rest, in that substrate's own coordinates. `thresholds`: {label: value}, e.g.
    {"fixed": 0.0, "calibrated": 1.23} -- LLR is computed ONCE and every threshold is applied
    to the same scores (calibration doesn't change the ranking, only where the cutoff sits).
      coord_gate_*    mean share of SALIENT coordinates inside [lo, hi] (the op's own range
                      gate) on class-c held positions vs other-class held positions -- this is
                      the diagnostic that motivated the position gate: if it's high and similar
                      for both, the coordinate gate alone is not class-selective.
      pos_gate_auroc  AUROC of the continuous LLR_c score for class-c vs rest positions
                      (threshold-independent; 1.0 = perfectly separates them).
      pos_gate[label] fire rate of the position gate (LLR_c > thresholds[label]) on class-c
                      held positions ("target") and on the pooled rest ("comp"), per label.
    NaN where a pool is empty (e.g. no held documents) rather than raising."""
    def coord_share(pos: torch.Tensor) -> float:
        if pos.shape[0] == 0 or not bool(salient.any()):
            return float("nan")
        inside = (pos[:, salient] >= lo[salient]) & (pos[:, salient] <= hi[salient])
        return float(inside.float().mean())

    llr_own = llr_scores(own, mu_c, sd_c, mu_rest, sd_rest, salient) if own.shape[0] else torch.zeros(0)
    llr_other = llr_scores(other, mu_c, sd_c, mu_rest, sd_rest, salient) if other.shape[0] else torch.zeros(0)
    auroc = float("nan")
    if len(llr_own) and len(llr_other):
        from sklearn.metrics import roc_auc_score
        scores = torch.cat([llr_own, llr_other]).cpu().numpy()
        labels = np.concatenate([np.ones(len(llr_own)), np.zeros(len(llr_other))])
        try:
            auroc = float(roc_auc_score(labels, scores))
        except ValueError:
            auroc = float("nan")
    pos_gate = {}
    for label, thresh in thresholds.items():
        pt = float((llr_own > thresh).float().mean()) if len(llr_own) else float("nan")
        pc = float((llr_other > thresh).float().mean()) if len(llr_other) else float("nan")
        pos_gate[label] = {"target": pt, "comp": pc, "thresh": float(thresh)}
    return {"coord_gate_target": coord_share(own), "coord_gate_comp": coord_share(other),
            "pos_gate_auroc": auroc, "pos_gate": pos_gate}


# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bypass", required=True, help="token-bypass AE checkpoint")
    ap.add_argument("--datasets", default="db14", help="one DATASET_CONFIGS key (this tool is single-dataset)")
    ap.add_argument("--n_fit", type=int, default=150)
    ap.add_argument("--n_target", type=int, default=10)
    ap.add_argument("--n_comp_per_class", type=int, default=1)
    ap.add_argument("--prompt_tokens", type=int, default=24)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--tao", type=float, default=2.0)
    ap.add_argument("--percent", type=float, default=0.3)
    ap.add_argument("--ops", default="rm_range_comp,st_range,st_transport")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--z_mode", default="delta", choices=["delta", "splice"])
    ap.add_argument("--pos_gate", default="llr", choices=["none", "llr"],
                    help="llr (default): gate each op to positions where LLR_c > threshold "
                         "(see module docstring); none: the old coordinate-gate-only behaviour.")
    ap.add_argument("--llr_thresh", type=float, default=0.0,
                    help="fixed LLR threshold, used when --gate_target_rate <= 0")
    ap.add_argument("--gate_target_rate", type=float, default=0.8,
                    help="when > 0 (default), CALIBRATE the threshold per concept/substrate so "
                         "the gate fires on this share of class-c's OWN fit positions, instead "
                         "of using the fixed --llr_thresh (see calibrate_threshold). <= 0 falls "
                         "back to --llr_thresh for every concept/substrate.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diag_only", action="store_true",
                    help="run the FIT pass + HELD-document gate diagnostic, save it, and exit "
                         "before any generation (~1-2 min at full DB14 size)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    unknown = set(ops) - {"rm_range_comp", "st_range", "st_transport"}
    if unknown:
        raise SystemExit(f"--ops: unknown op(s) {sorted(unknown)}")
    n_comp_classes = None                                # None = every other kept class
    if args.smoke:
        args.n_fit, args.n_target, args.alphas = 30, 3, [1.0]
        n_comp_classes = 3
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    arm = Arm("bypass", args.bypass, dev)
    if arm.kind != "ae":
        raise SystemExit("--bypass must be an AE checkpoint")
    ae, mean, std = arm.ae, arm.mean, arm.std
    # Arm doesn't keep the raw checkpoint dict (only the rebuilt module + norm stats), so the
    # model name / training layer are read via a second, cheap load of the same checkpoint --
    # not a modification of Arm, just an extra call to the same public loader it itself uses.
    _, _, _, ckpt = load_ae_checkpoint(args.bypass, dev, allow_token_bias=True)
    model_name = ckpt["config"]["extraction"]["model_name"]
    layer = ckpt["config"]["data"]["target_layer"]
    D, Lz = mean.shape[0], arm.C.shape[1]
    print(f"[remove] LM={model_name} layer={layer} z_mode={args.z_mode} tao={args.tao} percent={args.percent}")

    tk = AutoTokenizer.from_pretrained(model_name)
    tk.pad_token = tk.eos_token
    lm = load_lm(model_name, device=dev)
    tap = TokenIdTap(lm)
    hook = SplicingHook(lm, layer)

    cfg = DATASET_CONFIGS[args.datasets]
    classes = cfg["classes"]
    n_held = max(args.n_target, args.n_comp_per_class)
    by_class, kept_all = load_concept_docs(cfg, args.n_fit, n_held, args.seed, smoke=False)
    if args.smoke:
        # --smoke needs classes BEYOND the 2 being tested to serve as "other" complement
        # sources, so this does NOT reuse load_concept_docs's own smoke truncation (which
        # keeps only 2 classes total, leaving none spare for complements).
        working = kept_all[:2 + n_comp_classes]
        targets_list = working[:2]
        comp_pool = working[2:2 + n_comp_classes]
    else:
        working = kept_all
        targets_list = working
        comp_pool = None                                  # per concept: every OTHER working class
    if len(working) < 2 or (args.smoke and len(working) < 2 + n_comp_classes):
        raise SystemExit(f"[remove] not enough kept classes ({len(working)}) for this configuration")
    print(f"[remove] {len(working)} working classes; {len(targets_list)} target concepts: "
          f"{[classes[c] for c in targets_list]}")

    # ---- FIT pass: per-class count/sum/sum-of-squares, BOTH substrates -------------------
    stats = {c: {"n": 0, "sum_h": torch.zeros(D, device=dev), "sumsq_h": torch.zeros(D, device=dev),
                "sum_z": torch.zeros(Lz, device=dev), "sumsq_z": torch.zeros(Lz, device=dev)}
             for c in working}
    tk.padding_side = "right"                             # BOS at position 0 (see capture_docs)
    for c in working:
        for h_raw, tok_ids, _ in capture_docs(lm, tk, by_class[c][: args.n_fit], layer, dev, FIT_MAX_TOKENS):
            x = (h_raw - mean) / std
            z = arm.coords(x, tok_ids)
            s = stats[c]
            s["n"] += x.shape[0]
            s["sum_h"] += x.sum(0); s["sumsq_h"] += (x * x).sum(0)
            s["sum_z"] += z.sum(0); s["sumsq_z"] += (z * z).sum(0)
    tk.padding_side = "left"

    total_n = sum(s["n"] for s in stats.values())
    total_sum = {"h": sum(s["sum_h"] for s in stats.values()), "z": sum(s["sum_z"] for s in stats.values())}
    total_sumsq = {"h": sum(s["sumsq_h"] for s in stats.values()), "z": sum(s["sumsq_z"] for s in stats.values())}
    class_mean = {"h": {}, "z": {}}
    for c in working:
        n_c = stats[c]["n"]
        class_mean["h"][c] = (stats[c]["sum_h"] / n_c).cpu().numpy()
        class_mean["z"][c] = (stats[c]["sum_z"] / n_c).cpu().numpy()

    def class_and_rest(sub, c):
        n_c = stats[c]["n"]
        n_rest = total_n - n_c
        s_sum, s_sumsq = (stats[c]["sum_h"], stats[c]["sumsq_h"]) if sub == "h" else (stats[c]["sum_z"], stats[c]["sumsq_z"])
        mu_c = (s_sum / n_c).cpu().numpy()
        sd_c = ((s_sumsq / n_c - (s_sum / n_c) ** 2).clamp_min(0).sqrt()).cpu().numpy()
        rest_sum, rest_sumsq = total_sum[sub] - s_sum, total_sumsq[sub] - s_sumsq
        mu_rest = (rest_sum / n_rest).cpu().numpy()
        sd_rest = ((rest_sumsq / n_rest - (rest_sum / n_rest) ** 2).clamp_min(0).sqrt()).cpu().numpy()
        return mu_c, sd_c, mu_rest, sd_rest

    # ---- generation setup -----------------------------------------------------------------
    ae_for_sub = {"h": None, "z": ae}
    results = {"meta": {**vars(args), "ops": ops, "working_classes": [classes[c] for c in working],
                        "target_classes": [classes[c] for c in targets_list],
                        "class_descriptions": {n: concept_description(args.datasets, n) for n in classes}},
              "concepts": {}}

    # ---- position-gate threshold calibration (a SECOND pass over the FIT docs, now that -----
    # mu_c/sd_c/mu_rest/sd_rest are known) -- per concept AND per substrate, so h's 922-term
    # and z's 1843-term LLR sums are each calibrated on their OWN scale (see module docstring).
    gate_thresh = {classes[c]: {} for c in targets_list}
    salrange = {}                                          # (c, sub) -> (salient, lo, hi) numpy, reused below
    for c in targets_list:
        for sub in ("h", "z"):
            mu_c, sd_c, mu_rest, sd_rest = class_and_rest(sub, c)
            _, salient, lo, hi = class_dprime_and_range(mu_c, sd_c, mu_rest, sd_rest, args.percent, args.tao)
            salrange[c, sub] = (salient, lo, hi, mu_c, sd_c, mu_rest, sd_rest)
    if args.gate_target_rate > 0:
        print(f"\n[remove] calibrating position-gate thresholds (target rate {args.gate_target_rate}) ...")
        tk.padding_side = "right"
        llr_fit = {(c, sub): [] for c in targets_list for sub in ("h", "z")}
        for c in targets_list:
            for h_raw, tok_ids, _ in capture_docs(lm, tk, by_class[c][: args.n_fit], layer, dev, FIT_MAX_TOKENS):
                x = (h_raw - mean) / std
                z = arm.coords(x, tok_ids)
                for sub, a in (("h", x), ("z", z)):
                    salient, _, _, mu_c, sd_c, mu_rest, sd_rest = salrange[c, sub]
                    llr = llr_scores(a, torch.as_tensor(mu_c, dtype=torch.float32, device=dev),
                                     torch.as_tensor(sd_c, dtype=torch.float32, device=dev),
                                     torch.as_tensor(mu_rest, dtype=torch.float32, device=dev),
                                     torch.as_tensor(sd_rest, dtype=torch.float32, device=dev),
                                     torch.as_tensor(salient, device=dev))
                    llr_fit[c, sub].append(llr.cpu())
        tk.padding_side = "left"
        for c in targets_list:
            cname = classes[c]
            for sub in ("h", "z"):
                vals = torch.cat(llr_fit[c, sub]) if llr_fit[c, sub] else torch.zeros(0)
                gate_thresh[cname][sub] = calibrate_threshold(vals, args.gate_target_rate) \
                    if len(vals) else args.llr_thresh
            print(f"  {cname:<24s} thresh h={gate_thresh[cname]['h']:+.3f}  z={gate_thresh[cname]['z']:+.3f}")
    else:
        for c in targets_list:
            gate_thresh[classes[c]] = {"h": args.llr_thresh, "z": args.llr_thresh}
    results["gate_thresholds"] = gate_thresh

    # ---- HELD-document gate diagnostic (no generation): why --pos_gate llr exists ---------
    # Same held documents load_concept_docs already gave every working class (n_held =
    # max(n_target, n_comp_per_class)), captured at EVERY non-BOS position (not just the
    # first --prompt_tokens used for a prompt later) -- per the spec's "otherwise just use
    # held docs, note it": these ARE the same documents the prompts are drawn from, just used
    # here in full, so there is a modest overlap between this diagnostic and the generation
    # prompts. Kept on CPU so GPU memory stays free for generation afterwards.
    print("\n[remove] HELD-document gate diagnostic (no generation)")
    held = {"h": {}, "z": {}}
    tk.padding_side = "right"
    for c in working:
        held_docs = by_class[c][args.n_fit: args.n_fit + n_held]
        hs_list, zs_list = [], []
        for h_raw, tok_ids, _ in capture_docs(lm, tk, held_docs, layer, dev, FIT_MAX_TOKENS):
            x = (h_raw - mean) / std
            z = arm.coords(x, tok_ids)
            hs_list.append(x.cpu()); zs_list.append(z.cpu())
        held["h"][c] = torch.cat(hs_list) if hs_list else torch.zeros(0, D)
        held["z"][c] = torch.cat(zs_list) if zs_list else torch.zeros(0, Lz)
    tk.padding_side = "left"
    n_held_pos = sum(v.shape[0] for v in held["h"].values())
    print(f"[remove] {n_held} held docs/class -> {n_held_pos} total held h-positions across "
          f"{len(working)} working classes")

    diag = {"h": {}, "z": {}}
    diag_rows = {"h": [], "z": []}
    for c in targets_list:
        cname = classes[c]
        for sub in ("h", "z"):
            salient, lo, hi, mu_c, sd_c, mu_rest, sd_rest = salrange[c, sub]
            d = np.abs(mu_c - mu_rest) / np.sqrt(0.5 * (sd_c ** 2 + sd_rest ** 2) + 1e-12)
            own = held[sub][c]
            other = torch.cat([held[sub][c2] for c2 in working if c2 != c])
            thresholds = {"fixed": args.llr_thresh}
            if args.gate_target_rate > 0:
                thresholds["calibrated"] = gate_thresh[cname][sub]
            cell = diagnostic_cell(
                own, other,
                torch.as_tensor(mu_c, dtype=torch.float32), torch.as_tensor(sd_c, dtype=torch.float32),
                torch.as_tensor(mu_rest, dtype=torch.float32), torch.as_tensor(sd_rest, dtype=torch.float32),
                torch.as_tensor(salient), torch.as_tensor(lo, dtype=torch.float32),
                torch.as_tensor(hi, dtype=torch.float32), thresholds)
            cell["dprime_median"] = float(np.median(d[salient])) if salient.any() else float("nan")
            cell["dprime_max"] = float(np.max(d[salient])) if salient.any() else float("nan")
            cell["n_salient"] = int(salient.sum())
            diag[sub][cname] = cell
            diag_rows[sub].append(cell)
            pg_bits = "  ".join(f"pos-gate[{lbl}](thr {v['thresh']:+.2f}) tgt {v['target']:.3f} / comp {v['comp']:.3f}"
                                for lbl, v in cell["pos_gate"].items())
            print(f"  {sub} {cname:<24s} coord-gate tgt {cell['coord_gate_target']:.3f} / "
                  f"comp {cell['coord_gate_comp']:.3f}  d' med {cell['dprime_median']:.2f} "
                  f"max {cell['dprime_max']:.2f} (n_salient {cell['n_salient']})  AUROC {cell['pos_gate_auroc']:.3f}  "
                  f"{pg_bits}")

    def _mean_over(rows, key):
        vals = [r[key] for r in rows if r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_pos_gate(rows, label, field):
        vals = [r["pos_gate"][label][field] for r in rows
                if label in r["pos_gate"] and r["pos_gate"][label][field] == r["pos_gate"][label][field]]
        return float(np.mean(vals)) if vals else float("nan")

    summ = {}
    pg_labels = ["fixed"] + (["calibrated"] if args.gate_target_rate > 0 else [])
    for sub in ("h", "z"):
        rows = diag_rows[sub]
        summ[sub] = {"coord_gate_target": _mean_over(rows, "coord_gate_target"),
                    "coord_gate_comp": _mean_over(rows, "coord_gate_comp"),
                    "pos_gate_auroc": _mean_over(rows, "pos_gate_auroc"),
                    "pos_gate": {lbl: {"target": _mean_pos_gate(rows, lbl, "target"),
                                      "comp": _mean_pos_gate(rows, lbl, "comp")} for lbl in pg_labels}}
    for lbl in pg_labels:
        print("  [" + lbl + "] " + " | ".join(
            f"{sub}: coord-gate tgt {summ[sub]['coord_gate_target']:.2f} / comp {summ[sub]['coord_gate_comp']:.2f}; "
            f"pos-gate tgt {summ[sub]['pos_gate'][lbl]['target']:.2f} / comp {summ[sub]['pos_gate'][lbl]['comp']:.2f}, "
            f"AUROC {summ[sub]['pos_gate_auroc']:.2f}" for sub in ("h", "z")))
    results["diagnostic"] = {"per_concept": diag, "mean": summ}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    if args.diag_only:
        del held                                          # free the CPU-side held tensors
        print(f"[remove] --diag_only: wrote {args.out}, stopping before generation")
        return

    for c in targets_list:
        cname = classes[c]
        comp_classes = comp_pool if args.smoke else [c2 for c2 in working if c2 != c]
        print(f"\n=== concept {c} ({cname}) -- complement classes: {[classes[x] for x in comp_classes]} ===")

        target_docs = by_class[c][args.n_fit: args.n_fit + args.n_target]
        prompt_text = [first_n_tokens_text(tk, t, args.prompt_tokens) for t in target_docs]
        prompt_true = [c] * len(target_docs)
        for c2 in comp_classes:
            comp_docs = by_class[c2][args.n_fit: args.n_fit + args.n_comp_per_class]
            prompt_text += [first_n_tokens_text(tk, t, args.prompt_tokens) for t in comp_docs]
            prompt_true += [c2] * len(comp_docs)
        n_tgt_prompts = len(target_docs)

        tk.padding_side = "left"
        enc = tk(prompt_text, return_tensors="pt", padding=True).to(dev)
        L = enc["input_ids"].shape[1]
        generate = make_generate(lm, tk, enc, L, args.max_new, hook)
        ppl = make_ppl(lm, enc, L)

        def run(fn=None):
            cont = generate(fn)
            pp = ppl(cont)
            ids = cont.cpu().numpy()
            texts = [tk.decode(r, skip_special_tokens=True) for r in ids]
            d2 = [distinct2(list(r)) for r in ids]
            return texts, pp, d2

        uns_text, uns_ppl, uns_d2 = run()
        rec = {"class_name": cname,
              "prompts": {"text": prompt_text, "true_class": prompt_true,
                          "true_class_name": [classes[x] for x in prompt_true], "n_target": n_tgt_prompts},
              "unsteered": {"text": uns_text, "ppl": uns_ppl.tolist(), "distinct2": uns_d2},
              "edits": {"h": {}, "z": {}}}

        z_recon_ppl = uns_ppl
        if args.z_mode == "splice":
            zb_text, zb_ppl, zb_d2 = run(nl.make_z_edit(None, ae, mean, std, dev, tap=tap))
            rec["z_recon_baseline"] = {"text": zb_text, "ppl": zb_ppl.tolist(), "distinct2": zb_d2}
            z_recon_ppl = zb_ppl
        base_ppl_for = {"h": uns_ppl, "z": z_recon_ppl}

        for sub in ("h", "z"):
            mu_c, sd_c, mu_rest, sd_rest = class_and_rest(sub, c)
            _, _, lo, hi = class_dprime_and_range(mu_c, sd_c, mu_rest, sd_rest, args.percent, args.tao)
            r = mu_c - mu_rest
            comp = class_balanced_mean_excluding(class_mean[sub], c)

            cells = []                                    # (key, edit_fn)
            if "rm_range_comp" in ops:
                cells.append(("rm_range_comp", nl.gate_edit(lo, hi, comp, dev)))
            if "st_range" in ops:
                cells += [(f"st_range_a{a}", nl.shift_edit(r, lo, hi, a, dev)) for a in args.alphas]
            if "st_transport" in ops:
                cells += [(f"st_transport_a{a}", nl.transport_edit(mu_c, sd_c, mu_rest, sd_rest, lo, hi, a, dev))
                         for a in args.alphas]

            base_pp = base_ppl_for[sub]
            for key, edit_fn in cells:
                log = {"fire": [], "edit_norm": [], "pos_fire_tgt": [], "pos_fire_comp": []}
                fn = make_edit_splice(sub, args.z_mode, edit_fn, lo, hi, mean, std, ae_for_sub[sub], tap, dev, log,
                                      pos_gate=args.pos_gate, llr_thresh=gate_thresh[cname][sub],
                                      mu_c=mu_c, sd_c=sd_c, mu_rest=mu_rest, sd_rest=sd_rest, n_tgt=n_tgt_prompts)
                texts, pp, d2 = run(fn)
                lr = (np.log(pp) - np.log(base_pp)).tolist()
                rec["edits"][sub][key] = {"text": texts, "ppl": pp.tolist(), "log_ppl_ratio": lr, "distinct2": d2,
                                          "gate_fire": float(np.mean(log["fire"])) if log["fire"] else None,
                                          "edit_norm": float(np.mean(log["edit_norm"])),
                                          "pos_gate_fire_target": float(np.mean(log["pos_fire_tgt"]))
                                                                  if log["pos_fire_tgt"] else None,
                                          "pos_gate_fire_comp": float(np.mean(log["pos_fire_comp"]))
                                                                if log["pos_fire_comp"] else None}
                d2 = np.asarray(d2)
                gf = rec["edits"][sub][key]["gate_fire"]
                pgt, pgc = rec["edits"][sub][key]["pos_gate_fire_target"], rec["edits"][sub][key]["pos_gate_fire_comp"]
                gf_s = f"{gf:.3f}" if gf is not None else "n/a"
                pg_s = f"{pgt:.3f}/{pgc:.3f}" if pgt is not None and pgc is not None else "n/a"
                print(f"[remove] {sub:1s} {key:<20s} {cname:<20s} ppl x{np.exp(np.mean(lr)):.2f}  "
                      f"distinct2 tgt {np.mean(d2[:n_tgt_prompts]):.2f} comp {np.mean(d2[n_tgt_prompts:]):.2f}  "
                      f"coord_gate {gf_s}  pos_gate(tgt/comp) {pg_s}  "
                      f"edit_norm {rec['edits'][sub][key]['edit_norm']:.3f}",
                      flush=True)
        results["concepts"][str(c)] = rec
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)         # checkpoint after every concept

    tap.remove()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"[remove] wrote {args.out}")


if __name__ == "__main__":
    main()
