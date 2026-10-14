"""Isolated, fail-closed Gemma 3 1B extraction -> DPC AE -> base/AE evals.

The default command only prints a plan. ``run`` explicitly starts GPU work.
This is orchestration of the frozen-activation MSE AE, not end-to-end KL training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

from geoae.config import Config
from geoae.paths import PACKAGE_ROOT, resolve_path

DEFAULT_CONFIG = PACKAGE_ROOT / "configs/base/gemma3-1b_l25_k2000_d2304_dpc.yaml"
MODEL_NAME = "google/gemma-3-1b-pt"
SUITE_FILES = ["pos.npz", "ner.npz", "ner_coarse.npy", "sentiment.npz",
               "sentiment_long.npz", "subjectivity.npz", "language.npz", "topic4.npz",
               "topic14.npz", "topic20.npz", "formality.npz", "domain.npz"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


@dataclass(frozen=True)
class Stage:
    name: str
    deps: tuple[str, ...]
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    cwd: Path


class Pipeline:
    def __init__(self, config: Path = DEFAULT_CONFIG, run_dir: Path | None = None):
        self.config_path = resolve_path(config).resolve()
        self.cfg = Config.from_yaml(self.config_path)
        # YAML round-trips tuples as sequences; make the effective config and
        # checkpoint provenance compare identically after that round-trip.
        self.cfg.train.betas = list(self.cfg.train.betas)
        self.root = resolve_path(run_dir or self.cfg.train.runs_dir).resolve()
        # A run directory owns only its artifacts, never the repo/home/root itself.
        if self.root in (Path("/"), Path.home(), PACKAGE_ROOT, PACKAGE_ROOT.parent):
            raise ValueError("Choose a dedicated experiment run directory")
        if any(c in str(self.root) for c in (",", "=", "\n")):
            raise ValueError("Run path cannot contain comma, equals sign or newline (eval CLI syntax)")
        self.acts = self.root / "activations"
        self.ckpts = self.root / "checkpoints"
        self.cache = self.root / "cache"
        self.evals = self.root / "evals"
        self.baselines = self.root / "baselines"
        self.final = self.ckpts / "eval_final.pt"
        self.effective_config = self.root / "config.yaml"
        self.cfg.extraction.activations_dir = str(self.acts)
        self.cfg.data.activations_dir = str(self.acts)
        self.cfg.train.checkpoints_dir = str(self.ckpts)
        self.cfg.train.runs_dir = str(self.root)
        self.validate_config()
        self.stages = self.build_stages()

    def validate_config(self) -> None:
        c = self.cfg
        if c.extraction.model_name != MODEL_NAME:
            raise ValueError(f"This recipe targets {MODEL_NAME}")
        if (c.extraction.layers != [25] or c.extraction.target_layer != 25
                or c.data.target_layer != 25 or c.model.hidden_size != 1152
                or c.extraction.hidden_size != 1152):
            raise ValueError("Gemma 3 1B recipe requires final layer 25 and hidden size 1152")
        if c.extraction.dtype != "float32":
            raise ValueError("Gemma extraction must use float32 storage")
        if c.train.centroid_init != "dpc" or c.train.reinit_mode != "peaks":
            raise ValueError("DPC initialization and peak reseeding are required")
        if c.train.n_epochs < c.train.clustering_start_epoch:
            raise ValueError("Training must reach the clustering phase")
        if not 0 < c.data.val_frac < 1 or c.data.batch_size < c.model.n_clusters:
            raise ValueError("Invalid split or batch size smaller than K")
        if c.extraction.n_tokens * (1 - c.data.val_frac) < 2 * c.data.batch_size:
            raise ValueError("Token budget must provide at least two full training batches")
        if c.train.sup_frac or c.loss.lambda_sup:
            raise ValueError("This is an unsupervised DPC recipe; supervision must be off")

    def build_stages(self) -> dict[str, Stage]:
        stages = {}
        py = sys.executable
        ck = str(self.final)
        act = self.acts / "layer_25.npy"
        base_names = ("balanced_kmeans", "balanced_dpc", "plain_kmeans")
        base_spec = ",".join(f"{n}={self.baselines / (n + '.npz')}" for n in base_names)

        def add(name, deps, module, args, outputs, cwd=None):
            stages[name] = Stage(name, tuple(deps),
                                 tuple(map(str, [py, "-u", "-m", module, *args])),
                                 tuple(outputs), cwd or self.root)

        def action(name, deps, outputs):
            add(name, deps, "geoae.gemma_dpc",
                ["--config", self.config_path, "--run-dir", self.root, "--action", name], outputs)

        action("preflight", [], [self.root / "preflight.json"])
        add("extract", ["preflight"], "geoae.extract", ["--config", self.effective_config,
            "--seed", self.cfg.train.seed], [act, self.acts / "meta.json"])
        action("validate_activations", ["extract"], [self.root / "activation_validation.json"])
        action("train", ["validate_activations"], [self.final, self.ckpts / "selected.json"])

        for name in base_names:
            args = ["--checkpoint", ck, "--activations", act, "--n_clusters", self.cfg.model.n_clusters,
                    "--n_sample", 1000000, "--seed", self.cfg.train.seed,
                    "--out", self.baselines / f"{name}.npz"]
            if name == "plain_kmeans":
                module = "geoae.interp.fit_baseline_kmeans"
                args += ["--max_iter", 300, "--max_no_improvement", 50]
            else:
                module = "geoae.interp.fit_balanced_kmeans"
                args += ["--batch_size", 32768, "--epochs", 8, "--space", "raw",
                         "--init", "dpc" if name == "balanced_dpc" else "kmeans++",
                         "--reinit", "peaks" if name == "balanced_dpc" else "farthest",
                         "--reinit_every", self.cfg.train.reinit_every]
            add(name, ["train"], module, args, [self.baselines / f"{name}.npz"])

        add("concept_cache", ["train"], "geoae.interp.concept_suite",
            ["--checkpoint", ck, "--out", self.cache, "--n_token_sents", 6000],
            [self.cache / n for n in SUITE_FILES])
        for bench in ("ravel", "ioi"):
            add(f"{bench}_cache", ["train"], "geoae.interp.benchmark_cache",
                ["--checkpoint", ck, "--bench", bench, "--n_rows", 6000,
                 "--batch_size", 8, "--out", self.cache / bench], [self.cache / f"{bench}.npz"])
        action("atlas_cache", ["train"], [self.cache / f"atlas_{n}.npz" for n in ("doc", "tone", "content")]
               + [self.cache / "atlas_last.npz", self.cache / "atlas_manifest.json"])
        action("validate_cache", ["concept_cache", "ravel_cache", "ioi_cache", "atlas_cache"],
               [self.cache / "validated.json"])

        # The legacy MMLU CLI ignores --out. Isolate its fixed filename via cwd.
        add("mmlu", ["train"], "geoae.interp.causal_concept_compare",
            ["--checkpoint", ck, "--layer", 25, "--mmlu", 2000, "--seed", 42],
            [self.evals / "results_ccc_mmlu.json"], cwd=self.evals)
        for name in base_names:
            args = ["--baseline", self.baselines / f"{name}.npz",
                    "--activations_dir", self.acts, "--layer", 25, "--n_sample", 1000000,
                    "--seed", 0, "--out", self.evals / f"cq_{name}.json", "--names", name]
            if name == "balanced_kmeans":
                args += ["ae_dpc", "--checkpoints", ck]
            add(f"cq_{name}", [name], "geoae.interp.clustering_quality", args,
                [self.evals / f"cq_{name}.json"])
        add("probe", ["validate_cache", *base_names], "geoae.interp.concept_probe",
            ["--cache", self.cache, "--models", f"ae_dpc={ck}", "--baselines", base_spec,
             "--out", self.evals / "concept_probe.json"], [self.evals / "concept_probe.json"])
        rungs = "pos_coarse,pos_fine,ravel_country,topic14,language"
        add("geometry", ["validate_cache", "balanced_kmeans"], "geoae.interp.concept_geometry",
            ["--cache", self.cache, "--models", f"ae_dpc={ck}", "--baselines",
             f"base_balanced={self.baselines / 'balanced_kmeans.npz'}", "--rungs", rungs,
             "--projections", "pca,tsne", "--n_points", 6000, "--max_classes", 8,
             "--outdir", self.root / "figures"],
            [self.root / "figures" / proj / f"{r}.png" for proj in ("pca", "tsne") for r in rungs.split(",")])
        for dataset in ("db14", "ag_news", "biasbios"):
            args = ["--checkpoint", ck, "--dataset", dataset,
                    "--correct_json", self.cache / f"joint_correct_{dataset}.json",
                    "--saliency", "dprime", "--percent", .3, "--tao", 2,
                    "--jc_batch_size", 8, "--out", self.evals / f"range_{dataset}.json"]
            if dataset == "biasbios":
                # Do not carry over Llama's teacher exclusion to another model.
                args += ["--alphas", 1, 2]
            add(f"range_{dataset}", ["train"], "geoae.interp.range_intervention_compare", args,
                [self.evals / f"range_{dataset}.json"])
        add("steer_db14", ["range_db14"], "geoae.interp.steering_concept_compare",
            ["--checkpoint", ck, "--dataset", "db14", "--jc_batch_size", 8,
             "--correct_json", self.cache / "joint_correct_db14.json",
             "--out", self.evals / "steer_db14.json"], [self.evals / "steer_db14.json"])
        add("number", ["train"], "geoae.interp.range_number_control",
            ["--checkpoint", ck, "--out", self.evals / "number_dprime.json",
             "--saliency", "dprime", "--tao", 2, "--percent", .3,
             "--alphas", .5, 1, 2, "--rotation_seeds", 0, 1, 2, "--shuffle_labels",
             "--batch_size", 8], [self.evals / "number_dprime.json"])
        action("summary", list(stages), [self.root / "summary.json", self.root / "summary.md"])
        return stages

    def selected(self, names: list[str] | None) -> list[Stage]:
        wanted = set()

        def visit(name):
            if name not in self.stages:
                raise ValueError(f"Unknown stage {name!r}; choices: {', '.join(self.stages)}")
            if name not in wanted:
                wanted.add(name)
                for dep in self.stages[name].deps:
                    visit(dep)

        for name in names or self.stages:
            visit(name)
        return [s for n, s in self.stages.items() if n in wanted]


def check_resources(p: Pipeline, *, check_model: bool = True) -> dict:
    """Metadata/tokenizer checks only: never loads model weights or runs inference."""
    import torch
    existing = p.root
    while not existing.exists():
        existing = existing.parent
    act_bytes = p.cfg.extraction.n_tokens * 1152 * 4
    remaining = 0 if (p.acts / "meta.json").exists() else act_bytes
    free = shutil.disk_usage(existing).free
    reserve = 20 * 1024**3  # model download, checkpoints, caches and temporary headroom
    if free < remaining + reserve:
        raise RuntimeError(f"Need {(remaining + reserve)/1e9:.1f} GB free; have {free/1e9:.1f} GB")
    if not torch.cuda.is_available():
        raise RuntimeError("This full-run recipe requires a CUDA GPU; plan/tests can run on CPU")
    info = {"model": MODEL_NAME, "layer": 25, "hidden_size": 1152,
            "latent_dim": p.cfg.model.latent_dim, "activation_budget_bytes": act_bytes,
            "free_disk_bytes": free, "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    if info["gpu_memory_bytes"] < 24 * 1024**3:
        raise RuntimeError("Full batch/cache recipe expects >=24 GiB GPU memory; use a reviewed smaller config")
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        available = next(int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        info["available_ram_bytes"] = available
        if available < 20 * 1024**3:
            raise RuntimeError("One-million-token geometry/baseline stages need >=20 GiB available host RAM")
    if check_model:
        from transformers import AutoConfig, AutoTokenizer
        from geoae.hf_auth import ensure_hf_login
        ensure_hf_login(verbose=False)
        cfg = AutoConfig.from_pretrained(MODEL_NAME)
        text = getattr(cfg, "text_config", cfg)
        if text.hidden_size != 1152 or text.num_hidden_layers != 26:
            raise RuntimeError("Downloaded model config no longer matches the recipe")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        ids = [tok.encode(x, add_special_tokens=False) for x in (" is", " are")]
        if any(len(x) != 1 for x in ids) or ids[0] == ids[1]:
            raise RuntimeError("Number control requires distinct single-token ' is' / ' are'")
        info.update(model_revision=getattr(cfg, "_commit_hash", None), number_token_ids=ids)
    return info


def validate_activations(p: Pipeline) -> dict:
    meta = json.loads((p.acts / "meta.json").read_text())
    if meta.get("model") != MODEL_NAME or meta.get("layers") != [25]:
        raise ValueError("Activation model/layer provenance mismatch")
    x = np.load(p.acts / "layer_25.npy", mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != 1152 or x.dtype != np.float32:
        raise ValueError("Activations must have shape (N,1152) and dtype float32")
    if len(x) != meta.get("n_tokens") or len(x) < .95 * p.cfg.extraction.n_tokens:
        raise ValueError("Extraction incomplete or metadata/row count mismatch (require >=95% budget)")
    if len([v for v in meta.get("domain_tokens", {}).values() if v > 0]) < 3:
        raise ValueError("Fewer than three source domains survived extraction")
    max_abs = 0.0
    for start in range(0, len(x), 16384):
        block = x[start:start + 16384]
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite activations at rows {start}:{start + len(block)}")
        max_abs = max(max_abs, float(np.abs(block).max()))
    return {"rows": len(x), "dtype": str(x.dtype), "max_abs": max_abs,
            "domains": meta["domain_tokens"], "all_rows_finite": True}


def inspect_checkpoint(path: Path, cfg: Config, *, require_final: bool) -> dict:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved = ck["config"]
    for section in ("extraction", "data", "model", "loss", "train"):
        if saved[section] != cfg.to_dict()[section]:
            raise ValueError(f"Checkpoint {section} config differs; refusing mixed-run resume/evaluation")
    if require_final and ck["epoch"] != cfg.train.n_epochs:
        raise ValueError(f"Need final epoch {cfg.train.n_epochs}, found {ck['epoch']}")
    initialized = bool(ck["model_state"]["centroids_initialized"])
    if require_final and not initialized:
        raise ValueError("Centroids not initialized: cannot evaluate a reconstruction-only checkpoint")
    lrs = [g["lr"] for g in ck["opt_state"]["param_groups"]]
    if any(not math.isclose(lr, cfg.train.lr, rel_tol=1e-9) for lr in lrs):
        raise ValueError("Optimizer learning rate differs from the configured learning rate")
    for name, value in ck["model_state"].items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    for key in ("norm_mean", "norm_std"):
        v = np.asarray(ck[key])
        if v.shape != (cfg.model.hidden_size,) or not np.isfinite(v).all():
            raise ValueError(f"Invalid checkpoint {key}")
        if key == "norm_std" and not (v > 0).all():
            raise ValueError("Nonpositive normalization standard deviation")
    if not math.isfinite(ck["val_mse"]):
        raise ValueError("Nonfinite validation MSE")
    return {"source": str(path), "epoch": ck["epoch"], "step": ck["step"],
            "val_mse": ck["val_mse"], "centroids_initialized": initialized,
            "optimizer_lrs": lrs, "sha256": digest(path)}


def train_and_seal(p: Pipeline, resume: bool) -> None:
    candidates = sorted(p.ckpts.glob("step_*.pt"))
    cmd = [sys.executable, "-u", "-m", "geoae.train", "--config", str(p.effective_config), "--no_wandb"]
    if candidates:
        if not resume:
            raise RuntimeError("Training checkpoints exist; use --resume-training to resume the same recipe")
        info = inspect_checkpoint(candidates[-1], p.cfg, require_final=False)
        if info["epoch"] > p.cfg.train.n_epochs:
            raise ValueError("Checkpoint is beyond the configured final epoch")
        if info["epoch"] < p.cfg.train.n_epochs:
            subprocess.run(cmd + ["--resume", "latest"], check=True, cwd=PACKAGE_ROOT)
    else:
        subprocess.run(cmd, check=True, cwd=PACKAGE_ROOT)
    candidates = sorted(p.ckpts.glob("step_*.pt"))
    if not candidates:
        raise RuntimeError("Trainer returned without any step checkpoint")
    info = inspect_checkpoint(candidates[-1], p.cfg, require_final=True)
    if p.final.exists():
        if digest(p.final) != info["sha256"]:
            raise RuntimeError("Existing sealed evaluation checkpoint differs; refusing overwrite")
    else:
        # Hard link keeps the selected weights stable without duplicating disk use.
        os.link(candidates[-1], p.final)
    write_json(p.ckpts / "selected.json", info)


def reduce_atlas(labels: list[list[int]], mode: str, min_support: int = 25):
    """Deterministic reductions matching the documented cluster-probe protocol."""
    counts = Counter(x for row in labels for x in set(row))
    chosen = []
    for row in labels:
        row = sorted(set(row))
        if not row or (mode == "single" and len(row) != 1):
            chosen.append(None)
        elif mode == "single":
            chosen.append(row[0])
        elif mode == "rarest":
            chosen.append(min(row, key=lambda x: (counts[x], x)))
        elif mode == "commonest":
            chosen.append(min(row, key=lambda x: (-counts[x], x)))
        else:
            raise ValueError(mode)
    support = Counter(x for x in chosen if x is not None)
    idx = np.array([i for i, x in enumerate(chosen) if x is not None and support[x] >= min_support])
    return idx.astype(np.int64), np.array([chosen[i] for i in idx], dtype=np.int64)


def build_atlas(p: Pipeline) -> None:
    # Last-token caching avoids storing ~1M unnecessary token states. Use the
    # same BF16 frozen LM as extraction, not atlas_cache.py's FP32 LM override.
    import torch
    from datasets import load_dataset
    from geoae.interp.concept_suite import Encoder
    from geoae.seeding import seed_everything
    seed_everything(p.cfg.train.seed)
    enc = Encoder(MODEL_NAME, 25, torch.device("cuda"))
    ds = load_dataset("guidelabs/fineweb-atlas", "chunks", split="train", streaming=True)
    texts, labels = [], {f: [] for f in ("document_ids", "tone_ids", "content_ids")}
    for i, row in enumerate(ds):
        if i >= 8000:
            break
        if row.get("chunk_status") != "ok":
            continue
        text = row.get("chunk_text", row.get("text", ""))
        if not text:
            continue
        texts.append(text)
        for field in labels:
            labels[field].append(list(row.get(field) or []))
    del ds
    if not texts:
        raise ValueError("No valid Atlas chunks")
    parts = []
    try:
        for start in range(0, len(texts), 8):
            seqs = [enc.tok.encode(t, add_special_tokens=True)[:160] for t in texts[start:start + 8]]
            hs = enc.run(seqs)
            parts.extend(hs[i, len(ids) - 1].float().cpu().numpy() for i, ids in enumerate(seqs))
            if start % 400 == 0:
                print(f"[atlas] {start}/{len(texts)} chunks", flush=True)
    finally:
        enc.hook.deactivate()
    hidden = np.stack(parts)
    if not np.isfinite(hidden).all():
        raise ValueError("Nonfinite Atlas cache")
    np.savez(p.cache / "atlas_last.npz", H_last=hidden,
             **{k: np.array(v, dtype=object) for k, v in labels.items()})
    reductions = {}
    for name, field, mode in (("doc", "document_ids", "single"),
                              ("tone", "tone_ids", "rarest"),
                              ("content", "content_ids", "commonest")):
        idx, y = reduce_atlas(labels[field], mode)
        if len(set(y)) < 2:
            raise ValueError(f"Atlas {name} has fewer than two eligible classes")
        np.savez(p.cache / f"atlas_{name}.npz", H_last=hidden[idx], label=y, source_rows=idx)
        reductions[name] = {"mode": mode, "min_support": 25, "rows": len(idx), "classes": len(set(y))}
    write_json(p.cache / "atlas_manifest.json", {"model": MODEL_NAME, "layer": 25,
               "n_stream": 8000, "n_chunks": len(hidden), "dtype": "float32",
               "compute_dtype": "bfloat16", "max_len": 160, "reductions": reductions})


def validate_cache(p: Pipeline) -> dict:
    from geoae.interp.concept_probe import LADDER, load_rung
    rungs = {}
    for name, filename, key, *_ in LADDER:
        h, y = load_rung(p.cache, filename, key, 0)
        if h.ndim != 2 or h.shape[1] != 1152 or len(h) != len(y) or not np.isfinite(h).all():
            raise ValueError(f"Malformed/nonfinite concept cache: {name}")
        if len(set(y.tolist())) < 2:
            raise ValueError(f"Concept cache {name} contains fewer than two classes")
        rungs[name] = {"rows": len(h), "classes": len(set(y.tolist()))}
    return {"model": MODEL_NAME, "layer": 25, "rungs": rungs}


def summarize(p: Pipeline) -> None:
    # Keep all machine-readable results, including coverage; never assume all
    # Gemma baseline number prompts are correct (the old Llama reviewer does).
    results = {f.stem: json.loads(f.read_text()) for f in sorted(p.evals.glob("*.json"))}
    selected = json.loads((p.ckpts / "selected.json").read_text())
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value

    write_json(p.root / "summary.json", {"model": MODEL_NAME, "layer": 25,
               "checkpoint": selected, "results": finite_json(results),
               "nonfinite_policy": "Undefined legacy metric values are represented as null, never as zero."})
    lines = ["# Gemma 3 1B: DPC AE evaluation", "", "Base columns are the original model/raw-space controls.", "",
             "| Metric | Base | DPC AE |", "|---|---:|---:|",
             f"| Reconstruction MSE | 0 (identity) | {selected['val_mse']:.6f} |"]
    mm = results["results_ccc_mmlu"]["meta"]
    lines.append(f"| MMLU accuracy | {mm['base_acc']:.4f} | {mm['recon_acc']:.4f} |")
    cq = results["cq_balanced_kmeans"]
    for metric in ("silhouette", "davies_bouldin", "calinski_harabasz", "effective_k"):
        lines.append(f"| {metric} (base: balanced KM) | {cq['balanced_kmeans'][metric]} | {cq['ae_dpc'][metric]} |")
    lines += ["", "## Semantic cluster agreement", "",
              "| Task | Base balanced KM | Base balanced DPC | Base plain KM | AE |",
              "|---|---:|---:|---:|---:|"]
    for task, d in results["concept_probe"].items():
        lines.append("| " + task + " | " + " | ".join(f"{d[k]['nmi']:.4f}" for k in
            ("balanced_kmeans", "balanced_dpc", "plain_kmeans", "ae_dpc")) + " |")
    lines += ["", "## Paired range interventions", "", "S = target drop − collateral drop; higher is better.", "",
              "| Dataset / operator | Base S | AE S | Concepts evaluated |", "|---|---:|---:|---:|"]
    for dataset in ("db14", "ag_news", "biasbios"):
        d = results[f"range_{dataset}"]
        for mode in d["summary"]:
            pairs = [(c[f"h_{mode}"]["selectivity"], c[f"z_{mode}"]["selectivity"])
                     for c in d["concepts"].values() if f"h_{mode}" in c and f"z_{mode}" in c]
            if pairs:
                h, z = np.mean(pairs, axis=0)
                lines.append(f"| {dataset} / {mode} | {h:.4f} | {z:.4f} | {len(pairs)} |")
    num = results["number_dprime"]["baselines"]
    lines += ["", "## Number control", "", "| Metric | Base | AE reconstruction |", "|---|---:|---:|",
              f"| Unedited test pair accuracy | {num['test_base_pair_accuracy']:.4f} | {num['test_recon_pair_accuracy']:.4f} |",
              "", f"Joint-correct population: {num['joint_correct_count']} prompts. Full arm-level all-example and joint-correct results are in `evals/number_dprime.json`.",
              "Do not equate pair-restricted suppression with factual editing or full-vocabulary generation success.",
              "", "## Artifacts", ""]
    lines += [f"- [{f.name}](evals/{f.name})" for f in sorted(p.evals.glob("*.json"))]
    lines += ["", "Caveats: one AE seed; cluster geometry uses sampled states; MMLU uses the legacy one-token generated-answer scorer; range coverage depends on joint-correct examples. No LLM-judge API calls, audited TPP, or downstream classifier probes are included in this launch recipe.", ""]
    (p.root / "summary.md").write_text("\n".join(lines))


def action(p: Pipeline, name: str, resume: bool) -> None:
    if name == "preflight":
        write_json(p.root / "preflight.json", check_resources(p))
    elif name == "validate_activations":
        write_json(p.root / "activation_validation.json", validate_activations(p))
    elif name == "train":
        train_and_seal(p, resume)
    elif name == "atlas_cache":
        build_atlas(p)
    elif name == "validate_cache":
        write_json(p.cache / "validated.json", validate_cache(p))
    elif name == "summary":
        summarize(p)
    else:
        raise ValueError(f"Unknown internal action: {name}")


def output_signature(paths: tuple[Path, ...]) -> list[dict]:
    return [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]


def validate_outputs(stage: Stage) -> None:
    for path in stage.outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Stage {stage.name} did not produce {path}")
        if path.suffix == ".json":
            d = json.loads(path.read_text())
            if stage.name == "number" and (d.get("status") != "complete" or len(d.get("arms", {})) != 144):
                raise RuntimeError("Number-control output is incomplete")
            if stage.name.startswith("range_") and (not d.get("summary") or not d.get("concepts")):
                raise RuntimeError("Range output has no completed concepts/summary")
            if stage.name == "probe" and len(d) != 22:
                raise RuntimeError(f"Expected 22 concept rungs, got {len(d)}")


def manifest(p: Pipeline) -> dict:
    # Include source hashes so code changes cannot silently reuse old stage results.
    source = {str(f.relative_to(PACKAGE_ROOT)): digest(f) for f in sorted((PACKAGE_ROOT / "geoae").rglob("*.py"))}
    return {"version": 1, "config": p.cfg.to_dict(), "source_sha256": source,
            "stages": {n: {"command": list(s.command), "deps": list(s.deps), "cwd": str(s.cwd),
                            "outputs": [str(x) for x in s.outputs]} for n, s in p.stages.items()}}


def completed(p: Pipeline, stage: Stage) -> bool:
    marker = p.root / "state" / f"{stage.name}.json"
    if not marker.exists():
        return False
    d = json.loads(marker.read_text())
    validate_outputs(stage)
    if d.get("outputs") != output_signature(stage.outputs):
        raise RuntimeError(f"Completed outputs changed for {stage.name}; use a new run directory")
    return True


def run(p: Pipeline, stages: list[Stage], resume: bool) -> None:
    import fcntl
    import yaml
    p.root.mkdir(parents=True, exist_ok=True)
    with (p.root / ".pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline process owns this run directory") from None
        fp = p.root / "manifest.json"
        spec = manifest(p)
        if fp.exists():
            if json.loads(fp.read_text()) != spec:
                raise RuntimeError("Recipe/source changed since this run started; choose a new --run-dir")
        else:
            if any(x.name != ".pipeline.lock" for x in p.root.iterdir()):
                raise RuntimeError("Nonempty run directory has no manifest; refusing to adopt unrelated artifacts")
            write_json(fp, spec)
            p.effective_config.write_text(yaml.safe_dump(p.cfg.to_dict(), sort_keys=False))
        # Verify effective config instead of trusting an editable on-disk copy.
        if Config.from_yaml(p.effective_config).to_dict() != p.cfg.to_dict():
            raise RuntimeError("Effective training config changed")
        for path in (p.acts, p.ckpts, p.cache, p.evals, p.baselines, p.root / "logs", p.root / "state"):
            path.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("WANDB_MODE", "disabled")
        from geoae.hf_auth import ensure_hf_login
        ensure_hf_login(verbose=False)
        # Auth may have been found in a repo token file; pass it to child CLIs too.
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if key in os.environ:
                env[key] = os.environ[key]
        for stage in stages:
            if completed(p, stage):
                print(f"[skip] {stage.name}: verified complete", flush=True)
                continue
            for dep in stage.deps:
                if not completed(p, p.stages[dep]):
                    raise RuntimeError(f"Missing completed dependency {dep} for {stage.name}")
            existing = [x for x in stage.outputs if x.exists()]
            if existing and not (stage.name == "train" and resume):
                raise RuntimeError(f"Unmarked/partial output for {stage.name}: {existing[0]}. "
                                   "Preserve/move aside that stage's partial outputs before retrying, or use a new run directory.")
            cmd = list(stage.command)
            if stage.name == "train" and resume:
                cmd.append("--resume-training")
            print(f"[run] {stage.name}: {shlex.join(cmd)}", flush=True)
            started = time.time()
            log = p.root / "logs" / f"{stage.name}.log"
            with log.open("a") as output:
                output.write(f"\n# {time.ctime()} {shlex.join(cmd)}\n")
                output.flush()
                result = subprocess.run(cmd, cwd=stage.cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Stage {stage.name} failed ({result.returncode}); see {log}")
            validate_outputs(stage)
            write_json(p.root / "state" / f"{stage.name}.json", {
                "outputs": output_signature(stage.outputs), "seconds": time.time() - started,
                "command": cmd, "completed_at": time.time()})
            print(f"[done] {stage.name}; log: {log}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="plan", choices=["plan", "check", "run", "status"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--run-dir", type=Path, help="Isolated alternative artifact directory")
    ap.add_argument("--stages", nargs="+", help="Select stages; prerequisites are included automatically")
    ap.add_argument("--resume-training", action="store_true")
    ap.add_argument("--action", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    p = Pipeline(args.config, args.run_dir)
    if args.action:
        action(p, args.action, args.resume_training)
        return
    stages = p.selected(args.stages)
    if args.mode == "check":
        print(json.dumps(check_resources(p), indent=2))
    elif args.mode == "run":
        run(p, stages, args.resume_training)
    elif args.mode == "status":
        for stage in stages:
            print(f"{'done' if completed(p, stage) else 'pending':7s} {stage.name}")
    else:
        print(f"PLAN ONLY: {MODEL_NAME}, layer 25, d={p.cfg.model.latent_dim}, "
              f"K={p.cfg.model.n_clusters}, {p.cfg.extraction.n_tokens:,} tokens, "
              f"{p.cfg.train.n_epochs} epochs\nArtifacts: {p.root}\n")
        for stage in stages:
            print(f"{stage.name}: {shlex.join(stage.command)}")
        print("\nNothing launched. Use 'check' for metadata/tokenizer/resource checks; 'run' to execute.")


if __name__ == "__main__":
    main()
