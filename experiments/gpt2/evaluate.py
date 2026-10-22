#!/usr/bin/env python3
"""Run the existing research evaluation pipeline for every GPT-2 transformer block.

The command recipe mirrors run_delta_prep.sh, run_delta_evals.sh and run_interp.sh.
Every activation cache, baseline, checkpoint, intervention and interpretation file
is selected for this model and layer. --dry-run needs only the Python standard
library and does not create artifacts. The actual experiment tools remain the
single implementation of the metrics.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt2"
CKDIR = "k2000_bnh_b32k_lam1_d1536_dpc_sampled_struct5_tokbias"
SEQUENCES = "sentiment,sentiment_long,subjectivity,language,topic4,topic14,topic20,formality,domain"
BIOS_CONCEPTS = ",".join(str(i) for i in range(28) if i != 26)
GEOMETRY_RUNGS = "topic14,topic4,domain,language,formality,subjectivity,pos_coarse,ner_coarse,ioi_role,ravel_continent"


@dataclass
class Step:
    name: str
    commands: list[list[str]]
    outputs: list[Path]
    requires: list[Path] = field(default_factory=list)
    last_block_only: bool = False
    api: bool = False
    complete_status: bool = False
    reset: bool = False
    alias: tuple[Path, Path] | None = None
    optional_outputs: list[Path] = field(default_factory=list)


def profile_paths(layer: int, args) -> dict[str, Path]:
    """Keep every artifact specific to its model and decoder block.

    The original GPT-2 entry point omits ``profile`` and retains its historical
    paths. Other campaigns supply a small, serializable model profile.
    """
    profile = getattr(args, "profile", None) or {}
    family = profile.get("family", profile.get("family_dir", "gpt2"))
    out = Path("results") / args.run_name
    if profile:
        out /= family
    out /= f"layer{layer}"
    if profile.get("variant"):
        out /= profile["variant"]
    cache = Path("cache") / family / f"layer{layer}"
    shared = Path("e2e/checkpoints/general") / family / f"layer{layer}"
    suffix = profile.get("baseline_suffix", "sampled_struct5")
    clusters = profile.get("n_clusters", 2000)
    model_directory = profile.get("model_name", MODEL).split("/")[-1].replace("Llama-", "llama")
    bb_root = Path("biasbios/activations") / family / f"layer{layer}"
    db_root = Path("dbpedia/activations") / family / f"layer{layer}"
    return dict(out=out, cache=cache, cache_ids=Path("cache_ids") / family / f"layer{layer}",
                shared=shared, table=shared / profile.get("token_bias_filename", "token_bias_sampled_struct5.npz"),
                km=shared / f"delta/balanced_kmeans_k{clusters}_dpc_{suffix}.npz",
                kmt=shared / f"delta/balanced_kmeans_k{clusters}_dpc_{suffix}_tokmean.npz",
                bb_root=bb_root, db_root=db_root,
                bb=bb_root / model_directory / "last", db=db_root / model_directory / "last_ids/unprompted")


def checkpoint_for(layer: int, dry_run: bool = False, profile: dict | None = None) -> Path:
    """Select the last training step, rather than a best-MSE or assumed step file."""
    profile = profile or {}
    family = profile.get("family", profile.get("family_dir", "gpt2"))
    directory = Path("checkpoints") / family / f"layer{layer}" / profile.get("ckdir", CKDIR)
    candidates = list(directory.glob("step_*.pt"))
    candidates = [p for p in candidates if p.stem.removeprefix("step_").isdigit()]
    if candidates:
        return max(candidates, key=lambda p: int(p.stem.removeprefix("step_")))
    if dry_run:
        return directory / "<latest-step>.pt"
    raise FileNotFoundError(f"No training step checkpoint in {directory}; train layer {layer} first")


def build_steps(layer: int, checkpoint: Path, dump: Path, args) -> list[Step]:
    """Construct commands without importing torch, datasets, or transformers."""
    profile = getattr(args, "profile", None) or {}
    model_name = profile.get("model_name", MODEL)
    last_layer = profile.get("n_layers", 12) - 1
    n_clusters = str(profile.get("n_clusters", 2000))
    paths = profile_paths(layer, args)
    out, cache, cache_ids, shared, table, km, kmt, bb_root, db_root, bb, db = (
        paths[name] for name in ("out", "cache", "cache_ids", "shared", "table", "km", "kmt",
                                "bb_root", "db_root", "bb", "db"))
    ev, iv, st, co = (out / p for p in ("evals", "interventions", "steering", "concept"))
    it = out / "interp"
    base_it, tok_it = out / "baseline/interp", out / "token_mean/interp"
    base_ct, tok_ct, byp_ct = (p / "closest_tokens.json" for p in (base_it, tok_it, it))
    judge_out = it / "judge_flash_lite.json"
    tok_judge_out = tok_it / "judge_flash_lite.json"
    api_cache = cache / "llm_judge"
    wsd = Path("data/wsd")
    # The metric tools retain their historical "bypass" arm key for any AE.
    # The serialized campaign profile records whether the AE has token bias.
    arms = f"base={km},base_tokmean={kmt},bypass={checkpoint}"
    layer_args = ["--layer", str(layer)]
    model_layer = ["--model_name", model_name, *layer_args]
    ck_args = ["--checkpoint", str(checkpoint)]
    py = [sys.executable, "-u"]

    def mod(name, *items):
        return [*py, "-m", f"geoae.{name}", *map(str, items)]

    def script(name, *items):
        return [*py, name, *map(str, items)]

    def single(name, command, target, **kw):
        return Step(name, [command], [target], **kw)

    def probe_files(directory):
        return [directory / f for f in ("meta.json", "source_hashes.npz", "tokens_train.npy",
                "tokens_test.npy", f"layer_{layer}.npy", f"layer_{layer}_test.npy",
                "labels_train.npy", "labels_test.npy")]

    norm = dump / f"norm_params_layer{layer}.npz"
    activation = dump / f"layer_{layer}.npy"
    km_cmd = mod("baselines.balanced_kmeans", "--norm_params", norm, *model_layer,
                 "--activations", activation, "--init", "dpc", "--reinit", "peaks",
                 "--n_clusters", n_clusters, "--seed", "42")
    concepts = [cache / f"{r}.npz" for r in ("pos", "ner", *SEQUENCES.split(","))]
    concepts.append(cache / "ner_coarse.npy")
    jc_db, jc_bb = (out / "joint_correct" / f"{name}.json" for name in ("db14", "biasbios"))
    gen = st / "cluster_steer_generate.json"
    concept_gen = co / "concept_steer_generate.json"
    judge_common = ["--provider", "openrouter", "--model", args.judge_model,
                    "--cache_dir", str(api_cache)]
    interp_judge = [*judge_common, "--n_clusters", n_clusters, "--sample", "stratified",
                    "--min_assigned", "50", "--concurrency", str(args.judge_concurrency)]
    pair_common = ["--space", f"base={base_ct}", "--space", f"token_removed={tok_ct}",
                   "--space", f"geoae={byp_ct}", "--pairs", str(args.pair_pairs),
                   "--control_pairs", str(args.pair_controls)]
    steps = [
        Step("km", [[*km_cmd, "--out", str(km)],
                    [*km_cmd, "--token_bias", str(table), "--out", str(kmt)]],
             [km, kmt], [norm, activation, table]),
        single("cache", mod("extraction.concept_suite", *model_layer, "--out", cache,
               "--only", f"pos,ner,{SEQUENCES}"), concepts[0], alias=(cache_ids, cache)),
        Step("bench", [mod("extraction.benchmark_cache", "--bench", b, *model_layer,
              "--n_rows", "6000", "--out", cache / b) for b in ("ravel", "ioi")],
             [cache / "ravel.npz", cache / "ioi.npz"]),
        Step("probe_acts", [mod("extraction.biasbios", *model_layer, "--pooling", "last", "--max_length", "256",
              "--n_train", "50000", "--n_test", "20000", "--out_base", bb_root),
              mod("extraction.dbpedia", *model_layer, "--mode", "unprompted", "--pooling", "last",
              "--max_length", "256", "--n_train", "50000", "--n_test", "10000",
              "--dir_suffix", "_ids", "--out_base", db_root)], probe_files(bb) + probe_files(db)),
        single("wsd_data", mod("analysis.word_sense", "--download_only", "--data_dir", wsd),
               wsd / "download_manifest.json"),
        single("core_eval", mod("evaluation.faithfulness", *ck_args, *layer_args, "--experiment", "all",
               "--out", ev / "core_eval.json"), ev / "core_eval.json", requires=[checkpoint]),
        single("tokstruct", script("geoae/evaluation/token_structure.py", "--arms", arms, "--activations_dir", dump,
               *layer_args, *([] if layer == last_layer else ["--skip_next"]), "--out", ev / "token_structure.json"),
               ev / "token_structure.json", requires=[checkpoint, km, kmt, activation]),
        single("probe", script("geoae/evaluation/chance_corrected_probe.py", "--cache", cache_ids,
               "--models", f"bypass={checkpoint}", "--baselines", f"base={km},base_tokmean={kmt}",
               "--atlas_last", "", "--out", ev / "probe.json"), ev / "probe.json",
               requires=[checkpoint, km, kmt, *concepts, cache / "ravel.npz", cache / "ioi.npz"]),
        single("mmlu", mod("evaluation.mmlu", *ck_args, "--out", ev / "mmlu.json"),
               ev / "mmlu.json", requires=[checkpoint]),
        single("cq", mod("evaluation.geometry", "--baseline", km, "--checkpoints", checkpoint,
               "--names", "base", "bypass", "--activations_dir", dump, *layer_args,
               "--n_sample", "1000000", "--seed", "0", "--out", ev / "cq.json"), ev / "cq.json",
               requires=[checkpoint, km, activation]),
        Step("number", [mod("evaluation.number_control", *ck_args, "--tao", "2", "--percent", "0.3",
              "--alphas", "0.5", "1", "2", "--rotation_seeds", "0", "1", "2", "--shuffle_labels",
              "--saliency", "dprime", "--out", iv / "range_number_dprime.json"),
              script("experiments/paper/analysis/review_number_control.py", "--source", iv / "range_number_dprime.json",
              "--review_out", iv / "number_review.json", "--figure_prefix", out / "figures/number_review")],
             [iv / "range_number_dprime.json", iv / "number_review.json"], [checkpoint],
             complete_status=True, reset=True),
        single("steer", mod("evaluation.steering", *ck_args, *layer_args, "--dataset", "db14",
               "--correct_json", jc_db, "--out", iv / "steer_db14.json"), iv / "steer_db14.json",
               requires=[checkpoint]),
        single("range_db14", mod("evaluation.control", *ck_args, *layer_args,
               "--dataset", "db14", "--correct_json", jc_db, "--saliency", "dprime", "--tao", "2.0",
               "--out", iv / "range_intervention_db14_dprime.json"), iv / "range_intervention_db14_dprime.json",
               requires=[checkpoint]),
        single("range_bios", mod("evaluation.control", *ck_args, *layer_args,
               "--dataset", "biasbios", "--concepts", BIOS_CONCEPTS, "--correct_json", jc_bb,
               "--saliency", "dprime", "--tao", "2.0", "--alphas", "1.0", "2.0",
               "--out", iv / "range_intervention_biasbios_dprime.json"), iv / "range_intervention_biasbios_dprime.json",
               requires=[checkpoint]),
        single("number_hb", mod("evaluation.number_control", *ck_args, "--tao", "2", "--percent", "0.3",
               "--alphas", "0.5", "1", "2", "--rotation_seeds", "0", "1", "2", "--shuffle_labels",
               "--saliency", "dprime", "--hb_table", table, "--out", iv / "range_number_dprime_hb.json"),
               iv / "range_number_dprime_hb.json", requires=[checkpoint, table], complete_status=True, reset=True),
        Step("csteer", [mod("evaluation.cluster_steering", "--arms", arms, "--activations_dir", dump, *layer_args,
              "--out", st / "cluster_steering.json"), mod("evaluation.cluster_steering", "--arms", arms,
              "--activations_dir", dump, *layer_args, "--hub_x", "1e9", "--out", st / "cluster_steering_allhubs.json")],
             [st / "cluster_steering.json", st / "cluster_steering_allhubs.json"],
             [checkpoint, km, kmt, activation], last_block_only=True),
        single("cgen", mod("analysis.cluster_steer_generate", "--base", km, "--bypass", checkpoint,
               "--activations_dir", dump, *layer_args,
               *(["--corpus_dir", args.corpus_dir] if profile else []), "--out", gen), gen,
               requires=[checkpoint, km, activation], last_block_only=not bool(profile)),
        single("cgen_concept", mod("analysis.concept_steer_generate", "--base", km, "--bypass", checkpoint,
               "--datasets", "db14,biasbios", "--handles", "label,label_z,base,bypass,base_dir,bypass_dir",
               "--activations_dir", dump, *layer_args, "--out", concept_gen), concept_gen,
               requires=[checkpoint, km, activation], last_block_only=not bool(profile)),
        single("geometry", mod("evaluation.concept_geometry", "--cache", cache_ids, "--models", f"bypass={checkpoint}",
               "--baselines", f"base={km},base_tokmean={kmt}", "--order", "base,base_tokmean,bypass",
               "--rungs", GEOMETRY_RUNGS, "--n_points", "6000", "--max_classes", "14", "--perplexity", "30",
               "--projections", "pca,tsne,umap,variance", "--outdir", out / "figures/geometry"),
               out / "figures/geometry/pca/topic14.png", requires=[checkpoint, km, kmt, *concepts]),
        single("biasprobe", mod("analysis.bias_probe", "--act_dir", bb, "--source_hashes", bb / "source_hashes.npz",
               *layer_args, "--ae_checkpoint", checkpoint, "--token_bias_table", table,
               "--spaces", "raw", "tokmean", "geoae", "--device", "cuda",
               "--plot_dir", out / "figures/probes/biasbios", "--out", out / "probes/biasbios_probe.json"),
               out / "probes/biasbios_probe.json", requires=[checkpoint, table, *probe_files(bb)],
               complete_status=True, reset=True),
        Step("tpp", [mod("analysis.probe_perturbation", "--act_dir", directory,
              "--source_hashes", directory / "source_hashes.npz", *layer_args,
              "--ae_checkpoint", checkpoint, "--token_bias_table", table,
              "--spaces", "raw", "tokmean", "geoae", "--device", "cuda",
              "--plot_dir", out / f"figures/probes/tpp_{name}", "--out", out / f"probes/tpp_{name}.json")
              for name, directory in (("dbpedia", db), ("biasbios", bb))],
             [out / "probes/tpp_dbpedia.json", out / "probes/tpp_biasbios.json"],
             [checkpoint, table, *probe_files(bb), *probe_files(db)], complete_status=True, reset=True),
        single("ct_base", mod("analysis.closest_tokens", "--baseline_kmeans", km, "--sampled_from", dump,
               "--corpus_dir", args.corpus_dir,
               "--doc_view", "--n_tokens", "500000", "--out", base_ct), base_ct, requires=[km, dump / "meta.json"]),
        single("ct_tok", mod("analysis.closest_tokens", "--baseline_kmeans", kmt, "--sampled_from", dump,
               "--corpus_dir", args.corpus_dir,
               "--doc_view", "--n_tokens", "500000", "--out", tok_ct), tok_ct, requires=[kmt, dump / "meta.json"]),
        single("ct", mod("analysis.closest_tokens", *ck_args, "--sampled_from", dump,
               "--corpus_dir", args.corpus_dir,
               "--doc_view", "--n_tokens", "500000", "--out", byp_ct), byp_ct,
               requires=[checkpoint, dump / "meta.json"]),
        single("wsd", mod("analysis.word_sense", *ck_args, "--baseline_kmeans", km,
               "--tokmean_kmeans", kmt, "--datasets", "coarsewsd20,wic", "--data_dir", wsd,
               "--act_meta", dump / "meta.json", "--out", it / "word_sense.json"), it / "word_sense.json",
               requires=[checkpoint, km, kmt, wsd / "download_manifest.json", dump / "meta.json"]),
        single("cjudge", mod("analysis.cluster_steer_judge", "--gen", gen, *judge_common, "--fluency", "judge",
               "--out", st / "cluster_steer_judge_llmfluency.json"), st / "cluster_steer_judge_llmfluency.json",
               requires=[gen], last_block_only=not bool(profile), api=True),
        single("cjudge_concept", mod("analysis.concept_steer_judge", "--gen", concept_gen, *judge_common,
               "--fluency", "judge", "--out", co / "concept_steer_judge_llmfluency.json"),
               co / "concept_steer_judge_llmfluency.json", requires=[concept_gen], last_block_only=not bool(profile), api=True),
        single("judge", mod("analysis.llm_judge", base_ct, byp_ct, *interp_judge, "--out", judge_out),
               judge_out, requires=[base_ct, byp_ct], api=True),
        single("judge_tok", mod("analysis.llm_judge", tok_ct, *interp_judge, "--out", tok_judge_out),
               tok_judge_out, requires=[tok_ct], api=True),
        single("judge3", mod("analysis.judge_agreement", it / "closest_tokens_base.json",
               it / "closest_tokens_bypass.json", "--judges", args.agreement_judges,
               "--provider", "openrouter", "--cache_dir", api_cache, "--n_clusters", "200",
               "--out", it / "judge_agreement.json"), it / "judge_agreement.json",
               requires=[base_ct, byp_ct], api=True),
        Step("atlas", [mod("analysis.atlas_summary", "--judge", judge_out, "--ct_base", base_ct,
              "--ct_bypass", byp_ct, "--out_dir", it)],
             [it / f"atlas_{name}.json" for name in ("summary", "base", "bypass")],
             [judge_out, base_ct, byp_ct]),
        Step("atlas_tok", [mod("analysis.atlas_summary", "--judge", judge_out, tok_judge_out,
              "--ct_base", tok_ct, "--ct_bypass", byp_ct, "--out_dir", it / "vs_tokmean")],
             [it / "vs_tokmean" / f"atlas_{name}.json" for name in ("summary", "base", "bypass")],
             [judge_out, tok_judge_out, tok_ct, byp_ct]),
        single("pair", mod("analysis.pairwise_judge", *pair_common, "--provider", "openrouter",
               "--model", args.pair_model, "--cache_dir", api_cache, "--concurrency", str(args.judge_concurrency),
               "--out", it / f"pairwise_{args.pair_model.split('/')[-1]}.json"),
               it / f"pairwise_{args.pair_model.split('/')[-1]}.json", requires=[base_ct, tok_ct, byp_ct], api=True),
        single("pair_export", mod("analysis.pairwise_judge", *pair_common, "--export", it / "agent_judge/batches",
               "--key", it / "agent_judge/key.json", "--batch_size", "75"), it / "agent_judge/key.json",
               requires=[base_ct, tok_ct, byp_ct]),
        single("pair_import", mod("analysis.pairwise_judge", "--import_dir", it / "agent_judge/answers",
               "--key", it / "agent_judge/key.json", "--out", it / "pairwise_agents.json"),
               it / "pairwise_agents.json", requires=[it / "agent_judge/key.json", it / "agent_judge/answers"]),
    ]
    # An interrupted cache/geometry step must not count as complete because its first file exists.
    next(s for s in steps if s.name == "cache").outputs = concepts
    next(s for s in steps if s.name == "geometry").outputs = [
        out / "figures/geometry" / projection / f"{rung}.png"
        for projection in ("pca", "tsne", "variance") for rung in GEOMETRY_RUNGS.split(",")]
    if profile:
        geometry = next(s for s in steps if s.name == "geometry")
        geometry.requires.extend([cache / "ravel.npz", cache / "ioi.npz"])
        geometry.optional_outputs = [out / "figures/geometry/umap" / f"{rung}.png"
                                     for rung in GEOMETRY_RUNGS.split(",")]
        steps.insert(next(i for i, step in enumerate(steps) if step.name == "number"),
                     single("dispersion", script("experiments/paper/analysis/paper_dispersion.py", "--dump", dump, *layer_args,
                            "--ckpt", checkpoint, "--km", km, "--km_tok", kmt, "--val_frac", "0.05",
                            "--vendi_rows", "2048", "--out", ev / "dispersion.json"),
                            ev / "dispersion.json", requires=[checkpoint, km, kmt, activation,
                            dump / "rows_doc.npy", dump / "rows_pos.npy", dump / "rows_tok.npy"]))
    return steps


GROUPS = {
    "prep": "km cache bench probe_acts wsd_data",
    "evals": "core_eval tokstruct probe mmlu cq",
    "interventions": "number steer range_db14 range_bios number_hb",
    "steering": "csteer cgen",
    "concept": "cgen_concept",
    "geometry": "geometry",
    "probes": "biasprobe tpp",
    "interp": "ct_base ct_tok ct wsd",
    "api": "cjudge cjudge_concept judge judge_tok judge3 atlas atlas_tok pair",
}


def select_steps(raw: str, all_steps: list[Step]) -> set[str]:
    atomic = {s.name for s in all_steps}
    selected = set()
    for part in raw.replace(",", " ").split():
        if part in ("all", "whole", "gpu"):
            groups = [g for g in GROUPS if part != "gpu" or g != "api"]
            selected.update(n for g in groups for n in GROUPS[g].split())
        elif part in GROUPS:
            selected.update(GROUPS[part].split())
        elif part in atomic:
            selected.add(part)
        else:
            raise ValueError(f"Unknown step/group {part!r}; groups: {', '.join(GROUPS)}, gpu, whole")
    if not selected:
        raise ValueError("At least one evaluation step is required")
    if "dispersion" in atomic and "cq" in selected and any(
            part in ("evals", "whole", "all", "gpu") for part in raw.replace(",", " ").split()):
        selected.add("dispersion")
    return selected


def guard_checkpoint(path: Path, layer: int, profile: dict | None = None) -> None:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    state = ck["model_state"]
    profile = profile or {}
    mode = profile.get("mode", "bypass")
    if mode not in ("bypass", "plain"):
        raise ValueError(f"Unknown AE mode {mode!r}")
    expected = {
        "epoch": (ck.get("epoch"), profile.get("n_epochs", 50)),
        "model": (cfg["extraction"].get("model_name"), profile.get("model_name", MODEL)),
        "layer": (cfg["data"].get("target_layer"), layer),
        "hidden_size": (cfg["model"].get("hidden_size"), profile.get("hidden_size", 768)),
        "latent_dim": (cfg["model"].get("latent_dim"), profile.get("latent_dim", 1536)),
        "n_clusters": (cfg["model"].get("n_clusters"), profile.get("n_clusters", 2000)),
        "token_bias": ("tb_table" in state, mode == "bypass"),
        "initialized": (bool(state.get("centroids_initialized", False)), True),
    }
    if profile:
        expected["token_bias_mode"] = (cfg["model"].get("token_bias_mode", "bypass"), "bypass")
        if mode == "plain":
            expected["token_bias_config"] = (bool(cfg["model"].get("token_bias")), False)
    errors = [f"{k}={v!r} (expected {want!r})" for k, (v, want) in expected.items() if v != want]
    if errors:
        raise ValueError(f"Checkpoint guard failed for {path}: {'; '.join(errors)}")
    model = profile.get("model_name", "GPT-2")
    print(f"[guard] {path}: {model} layer {layer}, {mode} AE, {expected['hidden_size'][1]} → "
          f"{expected['latent_dim'][1]}, K={expected['n_clusters'][1]}, epoch {expected['epoch'][1]}", flush=True)


def complete(step: Step) -> bool:
    if not all(p.is_file() for p in step.outputs):
        return False
    # Range experiments checkpoint their JSON after every concept. Only the
    # final write adds summary; interrupted checkpoint files must be retried.
    if step.name in ("range_db14", "range_bios", "steer"):
        try:
            value = json.loads(step.outputs[0].read_text())
            return (isinstance(value, dict) and isinstance(value.get("meta"), dict)
                    and isinstance(value.get("concepts"), dict)
                    and (step.name == "steer" or isinstance(value.get("summary"), dict)))
        except (OSError, ValueError):
            return False
    if step.complete_status:
        # Number's review file has no status; its experiment output does.
        checks = step.outputs[:1] if step.name == "number" else step.outputs
        try:
            values = [json.loads(p.read_text()) for p in checks]
            return all(isinstance(value, dict) and value.get("status") == "complete" for value in values)
        except (OSError, ValueError):
            return False
    try:
        # A killed final JSON write can leave a file whose existence alone is
        # insufficient evidence. Lists are valid for optional pair exports.
        return all(isinstance(json.loads(p.read_text()), (dict, list))
                   for p in step.outputs if p.suffix == ".json")
    except (OSError, ValueError):
        return False


def write_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def code_files(plan: list[Step], profile: dict | None = None) -> list[str]:
    """The tool entry points and their transitive local Python imports to transfer."""
    pending = [Path("experiments/gpt2/evaluate.py")]
    if profile:
        pending.append(Path("experiments/depth/evaluate.py"))
    for step in plan:
        for command in step.commands:
            if "-m" in command:
                pending.append(Path(command[command.index("-m") + 1].replace(".", "/") + ".py"))
            elif len(command) > 2 and command[2].endswith(".py"):
                pending.append(Path(command[2]))
    found = set()
    while pending:
        path = pending.pop()
        if str(path) in found or not (ROOT / path).is_file():
            continue
        found.add(str(path))
        tree = ast.parse((ROOT / path).read_text())
        package = list(path.parent.parts)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(package[:len(package) - node.level + 1] + ([base] if base else []))
                modules = [base, *[base + "." + alias.name for alias in node.names]]
            for module in modules:
                if not module.startswith(("geoae", "eval_out", "experiments.gpt2", "experiments.depth")):
                    continue
                candidate = Path(module.replace(".", "/"))
                for dependency in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                    if (ROOT / dependency).is_file():
                        pending.append(dependency)
    return sorted(found | {"experiments/gpt2/evaluate.sh", "scripts/env.sh", "scripts/py",
                           "configs/sites/local.sh", "configs/sites/delta.sh", "configs/sites/argo.sh"})


def ensure_alias(alias: Path, target: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        if alias.resolve() != target.resolve():
            raise ValueError(f"{alias} exists but does not refer to this layer's cache {target}")
    else:
        alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)


def run_layer(layer: int, args) -> bool:
    profile = getattr(args, "profile", None) or {}
    paths = profile_paths(layer, args)
    last_layer = profile.get("n_layers", 12) - 1
    model = profile.get("model_name", "GPT-2")
    # Prep-only invocations need no training checkpoint.
    try:
        checkpoint = (Path(args.checkpoint) if getattr(args, "checkpoint", None) else
                      checkpoint_for(layer, args.dry_run, profile) if profile else checkpoint_for(layer, args.dry_run))
    except FileNotFoundError:
        checkpoint = checkpoint_for(layer, dry_run=True, profile=profile)
    canonical_dump = Path(args.activations_dir)
    dump = Path(os.environ.get("GEOAE_EVAL_ACTIVATIONS_DIR", str(canonical_dump)))
    plan = build_steps(layer, checkpoint, dump, args)
    selected = select_steps(args.steps, plan)
    if any(checkpoint in s.requires for s in plan if s.name in selected):
        if not args.dry_run:
            if profile:
                guard_checkpoint(checkpoint, layer, profile)
            else:
                guard_checkpoint(checkpoint, layer)
    out = paths["out"]
    state_path = out / "pipeline_state.json"
    state = {"run_name": args.run_name, "model": profile.get("model_name", MODEL), "layer": layer, "checkpoint": str(checkpoint),
             "activations_dir": str(canonical_dump), "corpus_dir": args.corpus_dir,
             "runtime_activations_dir": str(dump),
             "requested": sorted(selected), "steps": {}}
    if profile:
        state.update(ae_mode=profile.get("mode", "bypass"),
                     layer_role=profile.get("layer_role", "mid"),
                     arm=profile.get("arm"), profile=profile)
    lock = None
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        lock = (out / ".evaluation.lock").open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(f"Another evaluation process is already using {out}")
        if state_path.exists():
            old = json.loads(state_path.read_text())
            finished_names = {name for name, record in old.get("steps", {}).items()
                              if record.get("status") in ("complete", "partial")}
            # Prep outputs use the frozen LM/dump and are independent of training.
            # They can precede the first real checkpoint without blocking evaluation.
            checkpoint_bound = bool(finished_names - set(GROUPS["prep"].split()))
            if profile and finished_names:
                changed = [name for name in ("activations_dir", "corpus_dir") if old.get(name) != state[name]]
                if checkpoint_bound and old.get("checkpoint") != str(checkpoint):
                    changed.insert(0, "checkpoint")
                if changed:
                    lock.close()
                    raise ValueError(f"{out} contains evaluations of a different {'/'.join(changed)}; "
                                     "choose a new --run-name")
            if old.get("checkpoint") == str(checkpoint) or (profile and not checkpoint_bound):
                state["steps"] = old.get("steps", {})
                state["requested"] = sorted(set(old.get("requested", [])) | selected)
        write_state(state_path, state)
        state["code_files"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in (code_files(plan, profile) if profile else code_files(plan))}
    failed = False
    try:
        print(f"[pipeline] {model} layer {layer}: {args.steps}; results {out}", flush=True)
        for step in plan:
            if step.name not in selected:
                continue
            reason = None
            if step.last_block_only and layer != last_layer:
                reason = ("This metric reads the block's residual directly through the final head; "
                          f"requires layer {last_layer}")
            elif step.api and args.judges == "skip":
                reason = "API judges disabled by --judges skip"
            elif step.api and not os.environ.get("OPENROUTER_API_KEY") and not args.dry_run and not complete(step):
                if args.judges == "required":
                    state["steps"][step.name] = {"status": "failed", "reason": "OPENROUTER_API_KEY missing"}
                    failed = True
                    write_state(state_path, state)
                    continue
                reason = "OPENROUTER_API_KEY missing; API judgment remains pending"
            elif step.name in ("atlas", "atlas_tok") and args.judges == "skip":
                reason = "Atlas summary requires API judgments"
            elif step.name in ("atlas", "atlas_tok") and not args.dry_run and any(
                    not p.is_file() for p in step.requires if "judge_flash_lite" in p.name) and any(
                    state["steps"].get(name, {}).get("status") == "skipped" for name in ("judge", "judge_tok")):
                reason = "Atlas summary remains pending until API judgments are available"
            if reason:
                print(f"[skip] {step.name}: {reason}", flush=True)
                state["steps"][step.name] = {"status": "skipped", "reason": reason}
                if not args.dry_run:
                    write_state(state_path, state)
                continue
            if args.dry_run:
                if step.alias:
                    print(f"[dry] cache alias {step.alias[0]} → {step.alias[1]}")
                for command in step.commands:
                    print(f"[dry] {step.name}: {shlex.join(command)}")
                continue
            if step.alias:
                ensure_alias(*step.alias)
            if step.name in ("probe", "geometry"):
                ensure_alias(paths["cache_ids"], paths["cache"])
            if not args.force and complete(step):
                label = "required outputs complete" if step.optional_outputs else "all outputs complete"
                print(f"[skip] {step.name}: {label}", flush=True)
                optional_missing = [str(path) for path in step.optional_outputs if not path.is_file()]
                state["steps"][step.name] = {"status": "partial" if optional_missing else "complete", "reused": True,
                                             "outputs": [str(p) for p in step.outputs]}
                if optional_missing:
                    state["steps"][step.name]["optional_missing"] = optional_missing
                    print(f"[partial] {step.name}: optional UMAP figures missing; --force retries the projection", flush=True)
                write_state(state_path, state)
                continue
            missing = [str(p) for p in step.requires if not p.exists()]
            if missing:
                reason = "Missing prerequisites: " + ", ".join(missing)
                print(f"[fail] {step.name}: {reason}", flush=True)
                state["steps"][step.name] = {"status": "failed", "reason": reason}
                failed = True
                write_state(state_path, state)
                continue
            for path in step.outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                if step.reset or (args.force and step.name == "cache"):
                    path.unlink(missing_ok=True)
                    path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
            if step.name == "cache":
                # The concept-suite NER branch produces both files together; an
                # interrupted coarse-label write needs its NER cache regenerated.
                directory = paths["cache"]
                if not (directory / "ner_coarse.npy").exists():
                    (directory / "ner.npz").unlink(missing_ok=True)
            if step.name == "judge3":
                for name, source in (("base", out / "baseline/interp/closest_tokens.json"),
                                     ("bypass", out / "interp/closest_tokens.json")):
                    ensure_alias(out / f"interp/closest_tokens_{name}.json", source)
            log_path = out / "logs" / f"{step.name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            state["steps"][step.name] = {"status": "running", "log": str(log_path),
                                         "commands": step.commands}
            write_state(state_path, state)
            code = 0
            with log_path.open("w") as logfile:
                for command in step.commands:
                    print(f"[run] {step.name}: {shlex.join(command)}", flush=True)
                    logfile.write(shlex.join(command) + "\n")
                    logfile.flush()
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, bufsize=1)
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        logfile.write(line)
                    code = process.wait()
                    if code:
                        break
            ok = code == 0 and complete(step)
            optional_missing = [str(path) for path in step.optional_outputs if not path.is_file()]
            status = "failed" if not ok else "partial" if optional_missing else "complete"
            state["steps"][step.name].update(status=status, exit_code=code,
                                              outputs=[str(p) for p in step.outputs])
            if ok and optional_missing:
                state["steps"][step.name]["optional_missing"] = optional_missing
                print(f"[partial] {step.name}: optional UMAP figures missing; --force retries the projection", flush=True)
            if not ok:
                failed = True
                print(f"[fail] {step.name}: exit {code}; expected complete outputs: {step.outputs}", flush=True)
            write_state(state_path, state)
        if not args.dry_run:
            finished = all(state["steps"].get(step.name, {}).get("status") == "complete" or (
                state["steps"].get(step.name, {}).get("status") == "skipped" and step.last_block_only
            ) for step in plan if step.name in state["requested"])
            state["status"] = "failed" if failed else "complete" if finished else "partial"
            write_state(state_path, state)
    finally:
        if lock is not None:
            lock.close()
    return not failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_layer", nargs="?", type=int, help="Compatibility spelling: evaluate.sh L all")
    parser.add_argument("legacy_steps", nargs="*", help=argparse.SUPPRESS)
    layers = parser.add_mutually_exclusive_group()
    layers.add_argument("--layer", type=int, help="One decoder block, 0 through 11")
    layers.add_argument("--layers", default="all", help="all, or comma-separated decoder blocks")
    parser.add_argument("--steps", default=None, help="Comma-separated groups or individual stages; whole/gpu/api")
    parser.add_argument("--activations-dir", default=os.environ.get("ACT_DIR", "activations_gpt2_sampled_struct5_10M"))
    parser.add_argument("--run-name", default="gpt_run", help="Campaign name under results/")
    parser.add_argument("--corpus-dir", default="corpus/gpt2_sampled_struct5_v1")
    parser.add_argument("--print-code-files", action="store_true", help="Print the exact portable evaluation code transfer list")
    parser.add_argument("--dry-run", action="store_true", default=bool(os.environ.get("DRY")))
    parser.add_argument("--force", action="store_true", default=bool(os.environ.get("FORCE")))
    parser.add_argument("--judges", choices=("auto", "skip", "required"), default="auto")
    parser.add_argument("--judge-model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--agreement-judges", default="google/gemini-2.5-flash-lite,openai/gpt-4o-mini,deepseek/deepseek-chat-v3.1")
    parser.add_argument("--pair-model", default=os.environ.get("PAIR_MODEL", "anthropic/claude-sonnet-5.5"))
    parser.add_argument("--judge-concurrency", type=int, default=int(os.environ.get("JUDGE_CONCURRENCY", "6")))
    parser.add_argument("--pair-pairs", type=int, default=int(os.environ.get("PAIR_PAIRS", "100")))
    parser.add_argument("--pair-controls", type=int, default=int(os.environ.get("PAIR_CONTROLS", "25")))
    args = parser.parse_args(argv)
    try:
        if args.legacy_layer is not None:
            if args.layer is not None or args.layers != "all":
                parser.error("Use either positional L or --layer/--layers")
            args.layer = args.legacy_layer
        if args.legacy_steps and args.steps:
            parser.error("Use either positional steps or --steps")
        args.steps = args.steps or ",".join(args.legacy_steps) or "whole"
        if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
            parser.error("--run-name must be one directory name")
        chosen = [args.layer] if args.layer is not None else list(range(12)) if args.layers == "all" else [int(v) for v in args.layers.split(",")]
        if not chosen or any(layer not in range(12) for layer in chosen):
            parser.error("GPT-2 decoder blocks must be in the range 0 through 11")
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        if args.print_code_files:
            print(json.dumps(code_files(build_steps(0, Path("<checkpoint>"), Path(args.activations_dir), args)), indent=2))
            return 0
        results = [run_layer(layer, args) for layer in dict.fromkeys(chosen)]
        return 0 if all(results) else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
