"""Small command registry; target modules load only after a command is selected."""
from __future__ import annotations

import argparse
import runpy
import sys

COMMANDS = {
    "train": "geoae.training.mse",
    "train-kl": "geoae.training.kl_cached",
    "train-stream": "geoae.training.kl_stream",
    "extract": "geoae.extraction.prefix",
    "extract-sampled": "geoae.extraction.sampled",
    "extract-logits": "geoae.extraction.logits",
    "corpus": "geoae.extraction.corpus",
    "token-bias": "geoae.token_bias",
    "pipeline": "geoae.workflow.pipeline"
}

GROUPS = {
    "evaluate": {
        "alignment": "geoae.evaluation.alignment",
        "atlas-alignment": "geoae.evaluation.atlas_alignment",
        "atlas-assign": "geoae.evaluation.atlas_assign",
        "atlas-chance-corrected": "geoae.evaluation.atlas_chance_corrected",
        "atlas-nmi": "geoae.evaluation.atlas_nmi",
        "causal-control": "geoae.evaluation.causal_control",
        "chain-control": "geoae.evaluation.chain_control",
        "chance-corrected-probe": "geoae.evaluation.chance_corrected_probe",
        "cluster-steering": "geoae.evaluation.cluster_steering",
        "concept-geometry": "geoae.evaluation.concept_geometry",
        "concept-separability": "geoae.evaluation.concept_separability",
        "control": "geoae.evaluation.control",
        "dbpedia": "geoae.evaluation.dbpedia",
        "faithfulness": "geoae.evaluation.faithfulness",
        "faithfulness-ablation": "geoae.evaluation.faithfulness_ablation",
        "filtered-probe": "geoae.evaluation.filtered_probe",
        "geometry": "geoae.evaluation.geometry",
        "inference-benchmark": "geoae.evaluation.inference_benchmark",
        "mmlu": "geoae.evaluation.mmlu",
        "morphology-control": "geoae.evaluation.morphology_control",
        "number-control": "geoae.evaluation.number_control",
        "perplexity": "geoae.evaluation.perplexity",
        "quantization": "geoae.evaluation.quantization",
        "sae-compare": "geoae.evaluation.sae_compare",
        "sae-geometry": "geoae.evaluation.sae_geometry",
        "sae-matched": "geoae.evaluation.sae_matched",
        "sae-report": "geoae.evaluation.sae_report",
        "ssl-control": "geoae.evaluation.ssl_control",
        "steering": "geoae.evaluation.steering",
        "token-structure": "geoae.evaluation.token_structure",
        "dispersion": "experiments.paper.analysis.paper_dispersion"
    },
    "analyze": {
        "atlas-cluster-knn-agreement": "geoae.analysis.atlas_cluster_knn_agreement",
        "atlas-concept-stats": "geoae.analysis.atlas_concept_stats",
        "atlas-knn-graph": "geoae.analysis.atlas_knn_graph",
        "atlas-knn-precision": "geoae.analysis.atlas_knn_precision",
        "atlas-lift-by-word": "geoae.analysis.atlas_lift_by_word",
        "atlas-summary": "geoae.analysis.atlas_summary",
        "atlas-symmetric-kmeans": "geoae.analysis.atlas_symmetric_kmeans",
        "atlas-token-detail": "geoae.analysis.atlas_token_detail",
        "atlas-token-detail-analyze": "geoae.analysis.atlas_token_detail_analyze",
        "audit-bias-cache": "geoae.analysis.audit_bias_cache",
        "bias-probe": "geoae.analysis.bias_probe",
        "closest-tokens": "geoae.analysis.closest_tokens",
        "cluster-generality": "geoae.analysis.cluster_generality",
        "cluster-steer-generate": "geoae.analysis.cluster_steer_generate",
        "cluster-steer-judge": "geoae.analysis.cluster_steer_judge",
        "concept-probe": "geoae.analysis.concept_probe",
        "concept-remove-generate": "geoae.analysis.concept_remove_generate",
        "concept-remove-judge": "geoae.analysis.concept_remove_judge",
        "concept-steer-extra": "geoae.analysis.concept_steer_extra",
        "concept-steer-generate": "geoae.analysis.concept_steer_generate",
        "concept-steer-judge": "geoae.analysis.concept_steer_judge",
        "concept-steer-supervised-subset": "geoae.analysis.concept_steer_supervised_subset",
        "concept-trajectory": "geoae.analysis.concept_trajectory",
        "dbpedia-merge": "geoae.analysis.dbpedia_merge",
        "dbpedia-pca": "geoae.analysis.dbpedia_pca",
        "dla-inventory": "geoae.analysis.dla_inventory",
        "eraser-movies": "geoae.analysis.eraser_movies",
        "grad-audit": "geoae.analysis.grad_audit",
        "judge-agreement": "geoae.analysis.judge_agreement",
        "lambda-balance": "geoae.analysis.lambda_balance",
        "lambda-sweep": "geoae.analysis.lambda_sweep",
        "linear-probe-compare": "geoae.analysis.linear_probe_compare",
        "llm-judge": "geoae.analysis.llm_judge",
        "logit-attribution": "geoae.analysis.logit_attribution",
        "manifold-map": "geoae.analysis.manifold_map",
        "pairwise-judge": "geoae.analysis.pairwise_judge",
        "pca-plot": "geoae.analysis.pca_plot",
        "plot-probe-comparison": "geoae.analysis.plot_probe_comparison",
        "probe-perturbation": "geoae.analysis.probe_perturbation",
        "rank-clusters": "geoae.analysis.rank_clusters",
        "training-dynamics": "geoae.analysis.training_dynamics",
        "word-sense": "geoae.analysis.word_sense"
    },
    "baseline": {
        "atlas-latent-kmeans": "geoae.baselines.atlas_latent_kmeans",
        "balanced-kmeans": "geoae.baselines.balanced_kmeans",
        "causal-kmeans": "geoae.baselines.causal_kmeans",
        "kmeans": "geoae.baselines.kmeans",
        "latent-kmeans": "geoae.baselines.latent_kmeans",
        "matched-benchmark": "geoae.baselines.matched_topk.benchmark",
        "matched-evaluate": "geoae.baselines.matched_topk.evaluate",
        "matched-report": "geoae.baselines.matched_topk.report",
        "matched-stage": "geoae.baselines.matched_topk.stage",
        "matched-train": "geoae.baselines.matched_topk.train",
        "sae-controls-fit": "geoae.baselines.sae_controls_fit",
        "sae-kmeans": "geoae.baselines.sae_kmeans",
        "token-erasure": "geoae.baselines.token_erasure"
    },
    "prepare": {
        "atlas-cache": "geoae.extraction.atlas_cache",
        "atlas-classes-prep": "geoae.extraction.atlas_classes_prep",
        "atlas-joint-correct": "geoae.extraction.atlas_joint_correct",
        "atlas-probe": "geoae.extraction.atlas_probe",
        "atlas-siblings-extract": "geoae.extraction.atlas_siblings_extract",
        "atlas-siblings-pairs": "geoae.extraction.atlas_siblings_pairs",
        "benchmark-cache": "geoae.extraction.benchmark_cache",
        "biasbios": "geoae.extraction.biasbios",
        "concept-cache": "geoae.extraction.concept_cache",
        "concept-suite": "geoae.extraction.concept_suite",
        "control-redraw": "geoae.extraction.control_redraw",
        "code-quantity": "geoae.extraction.datasets.code_quantity",
        "fewnerd": "geoae.extraction.datasets.fewnerd",
        "med-streusle": "geoae.extraction.datasets.med_streusle",
        "neuronlens": "geoae.extraction.datasets.neuronlens",
        "neuronlens-report": "geoae.extraction.datasets.neuronlens_report",
        "dbpedia": "geoae.extraction.dbpedia",
        "goemotions": "geoae.extraction.goemotions"
    }
}


def command_routes() -> dict[str, str]:
    """Stable command-to-module routes without importing research dependencies."""
    routes = dict(COMMANDS)
    for group, commands in GROUPS.items():
        routes.update({f"{group} {name}": module for name, module in commands.items()})
    return routes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="geoae", description="Train GeoAE, extract activations, and run evaluation or analysis tools.",
        epilog="Use 'geoae commands' to list routes or '<command> --help' for a tool's options.",
    )
    parser.add_argument("command", choices=[*COMMANDS, *GROUPS, "commands"], nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if args.command == "commands":
        if args.arguments:
            parser.error("commands takes no arguments")
        for command, module in sorted(command_routes().items()):
            print(f"geoae {command} -> python -m {module}")
        return
    command = args.command
    forwarded = args.arguments
    if command in GROUPS:
        group = argparse.ArgumentParser(prog=f"geoae {command}", description=f"Available {command} tools.")
        group.add_argument("tool", choices=list(GROUPS[command]))
        group.add_argument("arguments", nargs=argparse.REMAINDER)
        selected = group.parse_args(forwarded)
        module = GROUPS[command][selected.tool]
        command += f" {selected.tool}"
        forwarded = selected.arguments
    else:
        module = COMMANDS[command]
    original_argv = sys.argv
    try:
        sys.argv = [f"geoae {command}", *forwarded]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = original_argv
