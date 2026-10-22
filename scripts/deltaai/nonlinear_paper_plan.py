"""Model-generic command plan for the nonlinear final-layer paper evaluation.

Commands run from the isolated evaluation source snapshot.  This module does
not read assets, create directories, load checkpoints, or submit jobs.  A caller
can therefore inspect and fingerprint the complete plan before allocating a
GPU, then substitute its verified node-local activation directory at launch.

Required asset keys (paths may be strings or ``Path`` objects):
  activations_dir, balanced_kmeans, balanced_kmeans_tokmean, plain_kmeans,
  plain_kmeans_tokmean, linear_checkpoint, token_bias_table, probe_cache,
  atlas_cache, atlas_concepts, atlas_draws, control_datasets_dir,
  atlas_classes_dir, corpus.
``atlas_control_classes`` maps atlas_s0/atlas_s1/atlas_s2 to the original Qwen
kept-class lists (comma-separated strings or lists of integer class indices).
These frozen lists identify the paper's concepts; each new checkpoint still
builds its own true-positive document pool.

Every returned step has a unique ``name``, a ``group``, an argv ``command``,
declared ``outputs``, ``needs_activations``, and predecessor ``dependencies``.
The runner must create output parent directories and validate completion rather
than treating the mere presence of an output file as success.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any


PAPER_RUNG_NAMES = (
    "surface", "pos_fine", "ner_coarse", "domain", "topic4", "topic14",
    "topic20", "toxicity",
)
CONTROL_DATASETS = ("db14", "biasbios", "ag_news", "news20", "atlas_s0", "atlas_s1", "atlas_s2")
REQUIRED_ASSETS = (
    "activations_dir", "balanced_kmeans", "balanced_kmeans_tokmean",
    "plain_kmeans", "plain_kmeans_tokmean", "linear_checkpoint",
    "token_bias_table", "probe_cache", "atlas_cache", "atlas_concepts",
    "atlas_draws", "control_datasets_dir", "atlas_classes_dir",
    "atlas_control_classes", "corpus", "wiki_texts",
)


def _class_argument(value: str | list[int] | tuple[int, ...]) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Atlas kept-class lists must not be empty")
        return value
    if not value:
        raise ValueError("Atlas kept-class lists must not be empty")
    return ",".join(str(int(index)) for index in value)


def build_steps(profile: dict, checkpoint: Path, output: Path, python: str) -> list[dict]:
    """Return the detailed paper evaluation commands for one trained arm.

    Geometry and token-structure diagnostics include the reused linear decoder
    directly.  Alignment additionally compares linear and nonlinear AEs on the
    same filtered rows.  MMLU and perplexity score this checkpoint against its
    own unmodified-model pass; existing linear faithfulness artifacts can be
    reused.  Controls use fresh jointly correct documents for this checkpoint,
    as in the paper, rather than borrowing the linear model's predictions.
    """
    assets = profile["assets"]
    layer, model_name = str(int(profile["target_layer"])), profile["model_name"]
    missing = [key for key in REQUIRED_ASSETS if key not in assets]
    if missing:
        raise ValueError("Missing paper evaluation assets: " + ", ".join(missing))
    classes = assets["atlas_control_classes"]
    atlas_controls = assets.get("atlas_control_datasets", ["atlas_s0", "atlas_s1", "atlas_s2"])
    if not atlas_controls or any(not re.fullmatch(r"atlas_s\d+", value) for value in atlas_controls):
        raise ValueError("Expected explicit Atlas draw dataset names")
    control_datasets = tuple(name for name in CONTROL_DATASETS if not name.startswith("atlas_")) + tuple(atlas_controls)
    missing_classes = [name for name in atlas_controls if name not in classes]
    if missing_classes:
        raise ValueError("Missing frozen Atlas kept-class lists: " + ", ".join(missing_classes))
    a = {key: str(value) for key, value in assets.items() if key != "atlas_control_classes"}
    sae = assets.get("sae")
    if "gemma" in model_name.lower() and not sae:
        raise ValueError("Gemma paper comparisons require staged SAE assets and a sae specification")
    if sae and (not assets.get("sae_kmeans") or not assets.get("sae_dir")):
        raise ValueError("SAE comparisons require immutable sae_dir weights and the original sae_kmeans codebook")
    ck, out = str(checkpoint), Path(output)
    prefix = [str(python), "-u"]
    steps: list[dict[str, Any]] = []

    def add(name: str, group: str, argv: list[str], outputs: list[Path], *,
            activations: bool = False, dependencies: tuple[str, ...] = (),
            json_keys: tuple[str, ...] = (), npz_keys: dict[str, list[str]] | None = None) -> None:
        steps.append({
            "name": name, "group": group, "command": argv,
            "outputs": [str(path) for path in outputs],
            "needs_activations": activations,
            "dependencies": list(dependencies),
            "json_required_keys": {str(outputs[0]): list(json_keys)} if json_keys else {},
            "npz_required_keys": npz_keys or {},
        })

    cq = out / "evals/cq_std.json"
    add("cq_std", "core", prefix + [
        "-m", "geoae.interp.clustering_quality", "--baseline", a["balanced_kmeans"],
        "--checkpoints", a["linear_checkpoint"], ck, "--names", "base", "linear", "bypass",
        "--activations_dir", a["activations_dir"], "--layer", layer, "--n_sample", "1000000",
        "--seed", "0", "--out", str(cq),
    ], [cq], activations=True, json_keys=("base", "linear", "bypass"))

    token = out / "evals/token_structure.json"
    token_labels = out / "evals/token_structure_labels.npz"
    add("token_structure", "core", prefix + [
        "eval_out/token_structure.py", "--arms", ",".join((
            "base=" + a["balanced_kmeans"], "base_tokmean=" + a["balanced_kmeans_tokmean"],
            "plain=" + a["plain_kmeans"], "plain_tokmean=" + a["plain_kmeans_tokmean"],
            "linear=" + a["linear_checkpoint"], "bypass=" + ck,
        )), "--activations_dir", a["activations_dir"], "--layer", layer,
        "--val_frac", "0.05", "--n_eval", "500000", "--save_labels", str(token_labels),
        "--out", str(token),
    ], [token, token_labels], activations=True, json_keys=("base", "linear", "bypass"),
        npz_keys={str(token_labels): ["cur", "next", "linear", "bypass"]})

    mmlu = out / "evals/mmlu.json"
    add("mmlu", "core", prefix + ["-m", "geoae.interp.mmlu_splice", "--checkpoint", ck,
                                   "--n", "2000", "--seed", "42", "--out", str(mmlu)], [mmlu],
        json_keys=("meta", "per_subject", "preds"))
    ppl = out / "evals/splice_ppl.json"
    add("splice_ppl", "core", prefix + ["-m", "geoae.interp.splice_ppl", "--ae", str(ppl) + "=" + ck,
                                         "--seq_len", "2048", "--seed", "0"], [ppl],
        json_keys=("meta", "clean", "spliced"))

    probe_arms = ",".join(("res_plain=km:" + a["plain_kmeans"],
                           "tok_plain=km:" + a["plain_kmeans_tokmean"],
                           "linear=ae:" + a["linear_checkpoint"], "geoae=ae:" + ck))
    pairs = "res_plain:geoae,tok_plain:geoae,linear:geoae"
    if sae:
        probe_arms += ",scope16=saekm:" + str(sae) + "@" + a["sae_kmeans"]
        pairs += ",scope16:geoae"
    probe_common = prefix + ["eval_out/probe_dense_subsets.py", "--cache", a["probe_cache"],
                             "--arms", probe_arms, "--fp_min", "6", "--pairs",
                             pairs, "--pair_min", "10",
                             "--n_token", "80000", "--n_shuffle", "10"]
    probe = out / "evals/probe_dense_subsets.json"
    add("probe_alignment", "alignment", probe_common + ["--rungs", ",".join(PAPER_RUNG_NAMES),
                                                         "--out", str(probe)], [probe], json_keys=("meta", "rungs"))
    jigsaw = out / "evals/probe_jigsaw_allrows.json"
    add("probe_jigsaw", "alignment", probe_common + ["--rungs", "toxicity,toxicity_pattern",
                                                      "--no_anchor_exclusion", "--out", str(jigsaw)], [jigsaw],
        json_keys=("meta", "rungs"))

    assign = out / "evals/atlas_lastword/assign.npz"
    assignment_command = prefix + [
        "eval_out/atlas_assign_extra.py", "--cache", a["atlas_cache"],
        "--models", "linear=" + a["linear_checkpoint"] + ",geoae=" + ck,
        "--baselines", "plain42=" + a["plain_kmeans"] + ",plain42_tokmean=" + a["plain_kmeans_tokmean"],
        "--out", str(assign)]
    assign_keys = ["linear", "geoae", "plain42", "plain42_tokmean", "row_chunk", "row_tok", "chunk_doc"]
    if sae:
        assignment_command += ["--sae_kmeans", "scope16=" + str(sae) + "@" + a["sae_kmeans"]]
        assign_keys.append("scope16")
    add("atlas_assign", "alignment", assignment_command, [assign], npz_keys={str(assign): assign_keys})
    atlas_pairs = [("atlas_pair10", "plain42_tokmean,geoae"),
                       ("atlas_pair10_linear", "linear,geoae"),
                       ("atlas_pair10_all", "plain42,plain42_tokmean,linear,geoae" + (",scope16" if sae else ""))]
    if sae:
        atlas_pairs.append(("atlas_pair10_sae", "scope16,geoae"))
    for name, arms in atlas_pairs:
        scores = out / "evals/atlas_lastword" / (name + ".json")
        add(name, "alignment", prefix + [
            "eval_out/atlas_nmi_filtered.py", "--assign", str(assign),
            "--chunks", str(Path(a["atlas_cache"]) / "chunks.npz"),
            "--concepts", a["atlas_concepts"], "--draws", a["atlas_draws"], "--arms", arms,
            "--ref", "geoae", "--min_rows", "10", "--n_boot", "300", "--seed", "0",
            "--out", str(scores),
        ], [scores], dependencies=("atlas_assign",), json_keys=("meta", "sets"))

    control_env = ["env", "GEOAE_CONTROL_DATASETS_DIR=" + a["control_datasets_dir"],
                   "GEOAE_ATLAS_CLASSES_DIR=" + a["atlas_classes_dir"]]
    for dataset in control_datasets:
        suffix = "hzss1_tp3" if sae else "hz_tp2"
        result = out / "interventions" / ("range_intervention_" + dataset + "_dprime_" + suffix + ".json")
        correct = out / "joint_correct" / (dataset + ".json")
        command = control_env + prefix + [
            "-m", "geoae.interp.range_intervention_compare", "--checkpoint", ck,
            "--dataset", dataset, "--layer", layer, "--n_fit", "80", "--n_eval", "100",
            "--min_eval", "20", "--saliency", "dprime", "--percent", "0.3", "--tao", "2.0",
            "--removal_modes", "", "--steer_modes", "global",
            "--alphas", "0.4", "0.5", "--substrates", "h,z,s,s1" if sae else "h,z", "--seed", "42",
            "--wiki_texts", a["wiki_texts"],
            "--correct_json", str(correct), "--out", str(result),
        ]
        if sae:
            command += ["--sae", str(sae), "--tp_all"]
        if dataset.startswith("atlas_"):
            command += ["--concepts", _class_argument(classes[dataset]), "--n_comp", "260", "--max_try", "600"]
        else:
            command += ["--max_try", "3000"]
            if dataset == "biasbios":
                command += ["--concepts", ",".join(str(index) for index in range(28) if index != 26)]
            if dataset == "ag_news":
                command += ["--extra_split", "train"]
        add("control_" + dataset, "controls", command, [result, correct],
            json_keys=("meta", "concepts", "summary"))

    ranges_dir = out / "ranges"
    add("range_structure", "ranges", prefix + [
        "scripts/deltaai/qwen_paper_ranges.py", "--task", "all", "--checkpoint", ck,
        "--activations_dir", a["activations_dir"], "--km", a["balanced_kmeans"],
        "--kmt", a["balanced_kmeans_tokmean"], "--token_bias_table", a["token_bias_table"],
        "--layer", layer, "--model_name", model_name, "--out_dir", str(ranges_dir),
        "--corpus_dir", a["corpus"],
    ], [ranges_dir / "ranges_along_directions.json", ranges_dir / "same_direction_removal.json"], activations=True,
        json_keys=("meta", "summary", "range_pairs"))
    return steps
