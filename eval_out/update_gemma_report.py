"""Refresh the Gemma report from saved artifacts; never starts an experiment.

Run on each requested status/report update:
    scripts/delta/py eval_out/update_gemma_report.py
Outputs live outside geoae/ and the run's completion records, so refreshing the
report does not change the active pipeline's source manifest or artifacts.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
from statistics import mean

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/gemma3_1b_l25_d2304_dpc_seed42"
DEFAULT_OUT = ROOT / "docs/gemma3_1b_dpc_results.md"
MODELS = ("balanced_kmeans", "balanced_dpc", "plain_kmeans", "ae_dpc")
MODEL_LABELS = ("Base balanced KM", "Base balanced DPC", "Base plain KM", "DPC AE")
TOKEN_TASKS = {"surface", "pos_coarse", "pos_fine", "ner_coarse", "ner_fine",
               "ravel_country", "ravel_continent", "ravel_language", "ioi_role", "ioi_name"}


def read_json(path):
    return json.loads(path.read_text())


def safe_json(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def signature(paths):
    return [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths]


def verified_stage(root, name, stage):
    """Partial files are never sufficient evidence of a completed result."""
    marker = safe_json(root / "state" / f"{name}.json")
    if marker is None:
        return False
    paths = [Path(p) for p in stage["outputs"]]
    try:
        return bool(paths) and all(p.is_file() and p.stat().st_size for p in paths) and marker.get("outputs") == signature(paths)
    except OSError:
        return False


def active_stages(stages):
    commands = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except (OSError, UnicodeError):
            continue
        for name, spec in stages.items():
            expected = spec["command"]
            if argv[1:len(expected)] == expected[1:] and len(argv) >= len(expected):
                commands[name] = int(proc.name)
    return commands


def last_attempt(path):
    if not path.exists():
        return ""
    return path.read_text(errors="replace").split("\n# ")[-1].replace("\r", "\n")


def fmt(value, digits=4):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "NR"
    return f"{value:.{digits}f}"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |",
                       *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def semantic_chart(probe, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    tasks = list(probe)
    fig, ax = plt.subplots(figsize=(9.2, 9.0))
    y = np.arange(len(tasks))
    for offset, baseline, color, label in [(-.18, "balanced_kmeans", "#007c83", "AE − balanced raw KM"),
                                            (.18, "balanced_dpc", "#c76b23", "AE − balanced raw DPC")]:
        delta = [probe[t]["ae_dpc"]["nmi"] - probe[t][baseline]["nmi"] for t in tasks]
        ax.barh(y + offset, delta, height=.34, color=color, label=label)
    ax.set_yticks(y, tasks)
    ax.invert_yaxis()
    ax.axvline(0, color="#374151", lw=.8)
    ax.set_xlabel("Difference in NMI; positive favors the AE")
    ax.set_title("Gemma 3 1B: semantic cluster agreement across 22 tasks", loc="left", pad=14)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def render_html(markdown, output, timestamp):
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(markdown)
    toc = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            title = tokens[i + 1].content
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            token.attrSet("id", anchor)
            if token.tag == "h2":
                toc.append(f'<a href="#{anchor}">{html.escape(title)}</a>')
    body = md.renderer.render(tokens, md.options, {})
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    # Embed plots so the HTML remains viewable when copied without the run directory.
    def embed(match):
        path = (output.parent / html.unescape(match[1])).resolve()
        return 'src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '"'
    body = re.sub(r'src="([^"]+\.png)"', embed, body)
    style = """
    *{box-sizing:border-box}body{margin:0;color:#192d3b;background:#f1f5f7;font:16px/1.65 system-ui,sans-serif}
    aside{position:fixed;inset:0 auto 0 0;width:245px;background:#142a38;color:white;padding:28px 22px;overflow:auto}
    aside a{display:block;color:#cbe6e8;font-size:13px;margin:12px 0;text-decoration:none}
    main{margin-left:245px;padding:30px}article{max-width:1250px;margin:auto;background:white;padding:40px;border-radius:10px}
    h1{font-size:36px;line-height:1.2}h2{margin-top:46px;padding-top:15px;border-top:2px solid #dce5e9}
    h3{margin-top:28px}a{color:#007c83}img{max-width:100%;height:auto}code{font-size:12px;overflow-wrap:anywhere;background:#eef3f5}
    pre{padding:14px;background:#eef3f5;overflow:auto}.table-wrap{overflow-x:auto;margin:22px 0}
    table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
    th,td{text-align:left;padding:9px;border-bottom:1px solid #dce5e9}th{background:#e9f1f4}tr:nth-child(even){background:#f8fafb}
    button{padding:8px 12px;border:0;border-radius:4px;cursor:pointer}footer{font-size:12px;color:#587080;margin-top:35px}
    @media(max-width:850px){aside{position:static;width:auto}aside nav{columns:2}main{margin:0;padding:12px}article{padding:22px}}
    @media print{aside{display:none}main{margin:0;padding:0}article{padding:0}body{background:white;font-size:10pt}table{font-size:8pt}tr,img{break-inside:avoid}thead{display:table-header-group}}
    """
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Gemma 3 1B — DPC AE results</title><style>{style}</style></head><body>
    <aside><h2>GeoAE / Gemma 1B</h2><p>Results snapshot</p><nav>{''.join(toc)}</nav>
    <button onclick="window.print()">Print / save PDF</button><a href="{output.name}">Markdown source</a></aside>
    <main><article>{body}<footer>Updated {timestamp}. Figures are embedded; source links refer to local artifacts.
    This report refreshes on request, not continuously.</footer></article></main></body></html>'''


def build_report(root, output):
    root, output = root.resolve(), output.resolve()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    manifest = read_json(root / "manifest.json")
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    stages = manifest["stages"]
    active = active_stages(stages)
    done = [name for name, spec in stages.items() if verified_stage(root, name, spec)]
    status = {}
    for name in stages:
        if name in done:
            status[name] = "Complete"
        elif name in active:
            status[name] = "Running"
        elif (root / "state" / f"{name}.json").exists():
            status[name] = "Unverified: output changed/missing"
        elif "Traceback (most recent call last)" in last_attempt(root / "logs" / f"{name}.log"):
            status[name] = "Stopped with error"
        else:
            status[name] = "Pending"
    results, hashes = {}, {}
    for name in done:
        for raw in stages[name]["outputs"]:
            path = Path(raw)
            if path.parent == root / "evals" and path.suffix == ".json":
                results[path.stem] = read_json(path)
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot_path = output.with_suffix(".json")
    previous = safe_json(snapshot_path) or {}
    new_stages = [n for n in done if n not in previous.get("completed_stages", [])]
    updates = previous.get("updates", [])
    if not previous or new_stages or hashes != previous.get("result_sha256") or status != previous.get("stage_status"):
        updates.append({"timestamp": now, "new_completed_stages": new_stages,
                        "running_stages": list(active), "verified_result_files": sorted(hashes)})

    def link(path, label=None):
        path = Path(path)
        return f"[{label or path.name}]({Path(os.path.relpath(path, output.parent)).as_posix()})"

    def source(stem):
        return "Source: " + link(root / "evals" / f"{stem}.json") + "."

    def result(stage, stem):
        return results.get(stem) if stage in done else None

    selected = read_json(root / "checkpoints/selected.json")
    acts = read_json(root / "activation_validation.json")
    probe = result("probe", "concept_probe")
    mmlu = result("mmlu", "results_ccc_mmlu")
    cluster = {}
    for name in ("balanced_kmeans", "balanced_dpc", "plain_kmeans"):
        cluster.update(result(f"cq_{name}", f"cq_{name}") or {})
    assessment = []
    if mmlu:
        m = mmlu["meta"]
        assessment.append(f"AE reconstruction changes MMLU from **{m['base_acc']:.2%} to {m['recon_acc']:.2%}** ({100*m['delta']:+.2f} percentage points; {m['n']:,} examples).")
    if "ae_dpc" in cluster and "balanced_kmeans" in cluster:
        a, b = cluster["ae_dpc"], cluster["balanced_kmeans"]
        assessment.append(f"The AE has stronger geometric separation than balanced raw k-means: silhouette **{a['silhouette']:.4f} vs {b['silhouette']:.4f}**, Davies–Bouldin **{a['davies_bouldin']:.3f} vs {b['davies_bouldin']:.3f}** (lower is better), and **{a['effective_k']:,}/{cfg['model']['n_clusters']:,}** clusters above the effective-usage threshold.")
        assessment.append(f"The AE centroid matrix is also more concentrated: effective rank **{a['effective_rank']:.1f}**, versus **{b['effective_rank']:.1f}** for balanced raw KM. Better separation does not imply more independent semantic directions.")
    if probe:
        nmi = {model: mean(row[model]["nmi"] for row in probe.values()) for model in MODELS}
        f1 = {model: mean(row[model]["f1"] for row in probe.values()) for model in MODELS}
        assessment.append(f"Semantic cluster agreement is mixed. Across {len(probe)} tasks, AE mean NMI is **{nmi['ae_dpc']:.4f}**, versus **{nmi['balanced_kmeans']:.4f}** for balanced raw KM and **{nmi['balanced_dpc']:.4f}** for balanced raw DPC. Mean cluster-label F1 is **{f1['ae_dpc']:.4f}**, versus **{f1['balanced_kmeans']:.4f}** and **{f1['balanced_dpc']:.4f}**, respectively.")
        assessment.append("The current evidence supports improved cluster geometry, not a general improvement in semantic clustering or preservation of model behavior. Token syntax and sequence-level semantics can move differently; the full task tables below show those differences.")
        if all(t in probe for t in ("pos_coarse", "pos_fine", "ravel_country")):
            assessment.append("A clear task contrast is POS versus RAVEL: AE/base-balanced-KM NMI is "
                              + "; ".join(f"**{probe[t]['ae_dpc']['nmi']:.4f}/{probe[t]['balanced_kmeans']['nmi']:.4f}** for {t}"
                                          for t in ("pos_coarse", "pos_fine", "ravel_country"))
                              + ". This measures cluster alignment with the labels, not loss or gain of all information available to a trained decoder.")
    causal = ["range_db14", "range_ag_news", "range_biasbios", "steer_db14", "number"]
    missing = [name for name in causal if name not in done]
    if missing:
        assessment.append("Causal conclusions remain provisional. Still awaiting completed results for: " + ", ".join(f"`{x}`" for x in missing) + ".")
    for dataset in ("db14", "ag_news", "biasbios"):
        stage = f"range_{dataset}"
        data = result(stage, stage)
        values = (data or {}).get("summary", {}).get("rm_range_zero")
        if values:
            assessment.append(f"For {dataset}'s fixed range-gated zero-removal arm, base/AE selectivity is "
                              f"**{fmt(values['h_sel'])}/{fmt(values['z_sel'])}**, over {values['n']} paired concepts. "
                              "The intervention tables include target suppression, collateral damage, and perplexity changes.")
    text = ["# Gemma 3 1B: DPC AE results", f"**Last updated: {now}**  \n**Run:** `{root.name}`  \n**Verified stages:** {len(done)}/{len(stages)}. **Running:** {', '.join(active) or 'none detected'}.",
            "This is a saved snapshot, refreshed when requested. A result enters the main tables only after its completion record and output metadata agree. Partial files and collection logs are reported separately; missing scores are not treated as zero.",
            "## Current findings", *assessment,
            "## Run and checkpoint", table(["Setting", "Value"], [
                ("Frozen model / decoder layer", f"{cfg['extraction']['model_name']} / {cfg['data']['target_layer']}"),
                ("Residual → latent dimensions", f"{cfg['model']['hidden_size']:,} → {cfg['model']['latent_dim']:,}"),
                ("AE / clusters", f"Dense GELU + BatchNorm; DPC initialization/reseeding; K={cfg['model']['n_clusters']:,}"),
                ("Extraction budget / retained rows", f"{cfg['extraction']['n_tokens']:,} / {acts['rows']:,}"),
                ("Activations", f"{acts['dtype']}; all finite={acts['all_rows_finite']}; maximum absolute value {acts['max_abs']:,.0f}; no clipping"),
                ("Final checkpoint", f"Epoch {selected['epoch']}; step {selected['step']:,}; centroids initialized={selected['centroids_initialized']}"),
                ("Training", f"Seed {cfg['train']['seed']}; batch {cfg['data']['batch_size']:,}; optimizer LR {selected['optimizer_lrs']}"),
                ("Validation reconstruction MSE", fmt(selected['val_mse'], 6)),
                ("Checkpoint SHA-256", f"`{selected['sha256']}`"),
            ]), "The MSE is measured on per-channel normalized activations. The raw identity reference has zero reconstruction error by definition. The 5% validation tail is split by row, not guaranteed disjoint by document.",
            "Sources: " + ", ".join(link(root / p) for p in ["config.yaml", "activation_validation.json", "checkpoints/selected.json"]) + ".",
            "## Model preservation"]
    if mmlu:
        m = mmlu["meta"]
        text += [table(["Metric", "Original Gemma", "AE reconstruction", "AE − base"], [
            (f"MMLU accuracy ({m['n']:,} examples)", f"{m['base_acc']:.2%}", f"{m['recon_acc']:.2%}", f"{100*m['delta']:+.2f} pp"),
            ("Normalized validation MSE", "0 (identity)", fmt(selected["val_mse"], 6), "—")]),
            "This is the legacy one-token generated-answer scorer: the model generates one token, decoded as A/B/C/D when possible. It is not constrained-choice likelihood scoring or an official benchmark reproduction. Both accuracies are near or below the 25% uniform-choice reference. No paired confidence interval or multi-seed uncertainty estimate is available.", source("results_ccc_mmlu")]
    else:
        text.append("MMLU result pending.")
    text += ["## Cluster geometry", "All codebooks use K=2,000 and normalization fitted from the same Gemma corpus. The main geometry sample contains 1M states; silhouette and Dunn use smaller subsamples. Base codebooks operate on normalized raw activations; the AE codebook operates in its learned latent representation."]
    metrics = [("silhouette", "Silhouette ↑"), ("davies_bouldin", "Davies–Bouldin ↓"),
               ("calinski_harabasz", "Calinski–Harabasz ↑"), ("dunn_index", "Dunn ↑, subsampling-sensitive"),
               ("separability_ratio", "Separation ratio"), ("effective_k", "Effective K"),
               ("n_empty_clusters", "Empty clusters"), ("cluster_balance", "Normalized occupancy entropy"),
               ("effective_rank", "Centered centroid effective rank"), ("intra_cluster_var", "Intra-cluster variance"),
               ("inter_centroid_dist_mean", "Mean centroid distance"), ("inter_centroid_dist_min", "Minimum centroid distance")]
    if cluster:
        text += [table(["Metric", *MODEL_LABELS], [[label, *(fmt(cluster.get(model, {}).get(key), 0 if key in ("effective_k", "n_empty_clusters") else 4) for model in MODELS)] for key, label in metrics]),
                 "Effective K counts clusters used by more than 0.1/K of sampled tokens; it is not the entropy-based effective count. Rank describes the centered centroid matrix, not semantic dimensionality. Distances and variances depend on the representation's scale. Plain KM's concentrated occupancy makes it a weak sole comparator.",
                 "The balanced baselines use uniform Sinkhorn assignments; the AE uses Zipf balancing. Their comparison does not isolate the encoder alone. Baseline fitting and geometry samples may overlap. Small stochastic/subsample metric differences should not be overinterpreted."]
        text += [source(f"cq_{name}") for name in ("balanced_kmeans", "balanced_dpc", "plain_kmeans") if f"cq_{name}" in results]
    text += ["## Semantic cluster agreement", "NMI measures agreement between cluster membership and labels; it is not chance-adjusted mutual information. F1 is the mean over eligible labels of the best single cluster's F1 (minimum support 20). These are descriptive cluster-label scores, not supervised classifier accuracy. Macro means weight each heterogeneous task equally and use the rounded values in the saved artifact. Token tasks are capped at 80,000 rows by the probe CLI; sequence tasks use their full caches."]
    if probe:
        rows = []
        for group, tasks in [("All tasks", list(probe)), ("Token tasks", [t for t in probe if t in TOKEN_TASKS]), ("Sequence tasks", [t for t in probe if t not in TOKEN_TASKS])]:
            for metric in ("nmi", "f1"):
                rows.append([f"{group} ({len(tasks)}): {metric.upper()}", *(fmt(mean(probe[t][model][metric] for t in tasks)) for model in MODELS)])
        text.append(table(["Macro average", *MODEL_LABELS], rows))
        text.append(table(["Comparator", "AE task wins: NMI", "AE task wins: F1"], [[MODEL_LABELS[i], *(f"{sum(row['ae_dpc'][metric] > row[model][metric] for row in probe.values())}/{len(probe)}" for metric in ("nmi", "f1"))] for i, model in enumerate(MODELS[:-1])]))
        chart = output.parent / "assets" / "gemma3_1b_semantic_delta.png"
        semantic_chart(probe, chart)
        text.append("!" + link(chart, "Per-task NMI difference between AE and balanced raw baselines"))
        text.append("Exportable chart: " + link(chart.with_suffix(".svg"), "SVG") + ".")
        for metric in ("nmi", "f1"):
            text += [f"### Per-task {metric.upper()}", table(["Task", *MODEL_LABELS], [[task, *(fmt(values[model][metric]) for model in MODELS)] for task, values in probe.items()])]
        text += [source("concept_probe"), "Atlas document labels use single-label chunks, tone uses a rarest-label reduction, and content uses a commonest-label reduction. Those derived labels are not equivalent to original single-label ground truth."]
    else:
        text.append("Concept-probe result pending.")
    text += ["## Local geometry and plots"]
    if "geometry" in done:
        log = last_attempt(root / "logs/geometry.log")
        rows, rung = [], None
        for line in log.splitlines():
            match = re.match(r"\[geo\] (\w+): ([\d,]+) points, (\d+) classes", line)
            if match:
                rung, points, classes = match.groups()
            if rung and "kNN-10 label agreement:" in line:
                scores = dict(re.findall(r"(ae_dpc|base_balanced) ([\d.]+)", line))
                if len(scores) == 2:
                    base, ae = float(scores["base_balanced"]), float(scores["ae_dpc"])
                    rows.append([rung, points, classes, f"{base:.1%}", f"{ae:.1%}", f"{100*(ae-base):+.1f} pp"])
        text += [table(["Task", "Points", "Classes", "Raw h kNN", "AE latent kNN", "AE − raw"], rows),
                 "kNN-10 uses a 60/40 stratified split and PCA to 50 components in each representation. PCA is fitted before the split, making this an exploratory, transductive neighborhood diagnostic. The eight most frequent classes are shown, not the full task. Scores above reproduce the rounded log values. PCA/t-SNE layouts are separate per representation; t-SNE distances between panels have no common scale.",
                 "Source: " + link(root / "logs/geometry.log") + ".",
                 table(["Task", "PCA", "t-SNE"], [[rung, *(link(root / "figures" / kind / f"{rung}.png", kind) for kind in ("pca", "tsne"))] for rung in ("pos_coarse", "pos_fine", "ravel_country", "topic14", "language")])]
        for rung in ("pos_fine", "topic14"):
            text.append("!" + link(root / "figures/pca" / f"{rung}.png", f"PCA: {rung}, raw and AE representations"))
    else:
        text.append("Complete geometry output pending; partial figures are not presented as a finished suite.")
    text += ["## Causal evaluations and coverage", "Selectivity S = target accuracy drop − complement accuracy drop; positive AE − base favors the AE. Compare h and z on paired concepts, keeping collateral damage and perplexity changes visible. Limited jointly correct examples can exclude classes. These results do not establish general factual editing ability."]
    for dataset in ("db14", "ag_news", "biasbios"):
        stage = f"range_{dataset}"
        text.append(f"### Range interventions: {dataset}")
        data = result(stage, stage)
        if data:
            rows = []
            for op, values in data.get("summary", {}).items():
                if "h_sel" in values:
                    rows.append([op, values["n"], fmt(values["h_sel"]), fmt(values["z_sel"]), fmt(values["z_minus_h"]),
                                 fmt(values.get("h_tgt_drop")), fmt(values.get("z_tgt_drop")),
                                 fmt(values.get("h_comp_drop")), fmt(values.get("z_comp_drop")),
                                 fmt(values.get("h_ppl_rise"), 2), fmt(values.get("z_ppl_rise"), 2)])
            text += [table(["Operator", "Paired concepts", "Base S", "AE S", "ΔS", "Base target drop", "AE target drop", "Base collateral", "AE collateral", "Base ΔPPL", "AE ΔPPL"], rows), source(stage)]
        else:
            text.append(f"**{status[stage]}.** No verified completed score yet.")
            partial = safe_json(root / "evals" / f"{stage}.json")
            if partial:
                text.append(f"Partial artifact currently contains {len(partial.get('concepts', {}))} concept records; these are excluded from the result tables until the stage finishes.")
        log = last_attempt(root / "logs" / f"{stage}.log")
        counts = re.findall(r"Found (\d+) joint-correct docs for class (\d+)", log)
        finding = re.findall(r"Finding joint-correct predictions for class (\d+) \(([^)]+)\)", log)
        if counts:
            names = dict(finding)
            text += ["Joint-correct collection diagnostics from the latest log (candidate counts, not evaluated sample sizes):",
                     table(["Class", "Label", "Joint-correct candidates found"], [[c, names.get(c, "—"), n] for n, c in counts])]
        if finding and not data:
            text.append(f"Latest collection message: class {finding[-1][0]} ({finding[-1][1]}).")
        if log:
            text.append("Collection/progress source: " + link(root / "logs" / f"{stage}.log") + ".")
    text.append("### Dense DB14 steering")
    steering = result("steer_db14", "steer_db14")
    if steering:
        rows = []
        # Recompute over paired concepts rather than separately averaging h and z populations.
        for arm in steering["meta"].get("summary_selectivity", {}):
            pairs = [(c[f"h_{arm}"]["selectivity"], c[f"z_{arm}"]["selectivity"]) for c in steering["concepts"].values() if f"h_{arm}" in c and f"z_{arm}" in c]
            if pairs:
                b, a = mean(x[0] for x in pairs), mean(x[1] for x in pairs)
                rows.append([arm, len(pairs), fmt(b), fmt(a), fmt(a-b)])
        text += [table(["Alpha", "Paired concepts", "Base S", "AE S", "ΔS"], rows), source("steer_db14")]
    else:
        text.append(f"**{status['steer_db14']}.** No verified completed score yet.")
    text.append("### Grammatical number control")
    number = result("number", "number_dprime")
    if number:
        b = number["baselines"]
        text += [f"Completed {len(number['arms'])} arms. Joint-correct test population: {b['joint_correct_count']} prompts.",
                 table(["Metric", "Original Gemma", "AE reconstruction"], [["Unedited is/are pair accuracy", f"{b['test_base_pair_accuracy']:.2%}", f"{b['test_recon_pair_accuracy']:.2%}"]])]
        rows = []
        for key, arm in number["arms"].items():
            rows.append([key, fmt(arm["all"].get("selectivity")), fmt(arm["joint_correct"].get("selectivity")),
                         fmt(arm["joint_correct"].get("target_drop")), fmt(arm["joint_correct"].get("complement_drop")),
                         fmt(arm["joint_correct"].get("target_counterpart_top1")), fmt(arm["neutral"].get("kl"))])
        text += [table(["Arm", "All-example S", "Joint-correct S", "Joint target drop", "Joint collateral", "Joint counterpart top-1", "Neutral KL"], rows),
                 "All-example and joint-correct populations remain separate. Pair-restricted is/are flips are not full-vocabulary generation success. Rotation and shuffled-label arms are retained as controls; no best-alpha selection is performed.", source("number_dprime")]
    else:
        text.append(f"**{status['number']}.** Awaiting the 144-arm run; no verified completed score yet.")
    text += ["## Pipeline status", "Completion is checked against saved file size and modification time, following the launcher's stage records. This is not a fresh rerun or a full content hash of the multi-GB activation dump."]
    stage_rows = []
    for name in stages:
        marker = safe_json(root / "state" / f"{name}.json") or {}
        duration = f"{marker['seconds']/60:.1f} min" if name in done and "seconds" in marker else "—"
        log = root / "logs" / f"{name}.log"
        stage_rows.append([name, status[name], duration, link(log, "log") if log.exists() else "—"])
    text += [table(["Stage", "Status", "Completed duration", "Log"], stage_rows), "## Provenance, limitations, and updates",
             "This is one AE training seed and one width (2×). No multi-seed confidence estimates or causal superiority claims follow from the current geometric improvements. The Gemma scores are not a matched comparison to earlier Llama experiments. The frozen model uses BF16 compute and float32 stored activations; no clipping was used.",
             "The initial sequence-cache sampling defect was repaired using full-split shuffling; seven sequence caches were regenerated. Geometry's integer-object labels were encoded as categorical IDs for kNN while preserving plot labels. Earlier partial caches/plots were backed up; this report uses the verified completed replacements."]
    for name in ("concept_sequence_shuffle_v1/audit.json", "geometry_label_dtype_v1/repair.json"):
        path = root / "repairs" / name
        if path.exists():
            text.append("Repair provenance: " + link(path) + ".")
    text += ["Refresh on the next requested update:", "```bash\nscripts/delta/py eval_out/update_gemma_report.py\n```",
             "The refresh reads artifacts and regenerates this report, its embedded-figure HTML, and a JSON snapshot. It does not launch jobs or change the experiment manifest.",
             table(["UTC snapshot", "Newly completed stages", "Running at snapshot"], [[u["timestamp"], ", ".join(u["new_completed_stages"]) or "No new completion", ", ".join(u["running_stages"]) or "None detected"] for u in updates[-10:]])]
    text += ["### Verified result artifacts", *["- " + link(root / "evals" / name) for name in sorted(hashes)]]
    markdown = "\n\n".join(text) + "\n"
    for raw in re.findall(r"\]\(([^)]+)\)", markdown):
        if not raw.startswith(("http:", "https:", "#")) and not (output.parent / raw).exists():
            raise FileNotFoundError(f"Broken report source link: {raw}")
    atomic_write(output, markdown)
    atomic_write(output.with_suffix(".html"), render_html(markdown, output, now))
    snapshot = {"generated_at": now, "run_dir": str(root), "checkpoint": selected,
                "completed_stages": done, "stage_status": status, "active_pids": active,
                "result_sha256": hashes, "results": results, "updates": updates}
    atomic_write(snapshot_path, json.dumps(clean_json(snapshot), indent=2, allow_nan=False) + "\n")
    print(f"Updated {output}\nUpdated {output.with_suffix('.html')}\nVerified {len(done)}/{len(stages)} stages; running: {', '.join(active) or 'none detected'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_report(args.run_dir, args.output)


if __name__ == "__main__":
    main()
