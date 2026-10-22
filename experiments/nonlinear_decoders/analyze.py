"""Reproduce the review tables and figure from completed local result copies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from geoae.paths import artifact_path



def analyze(review_dir: Path, output_dir: Path) -> None:
    """Rebuild the saved Qwen comparison without running models or judges."""
    review_dir = review_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == review_dir or review_dir in output_dir.parents:
        raise ValueError("Choose an output directory outside the preserved review inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    ROOT = review_dir
    OUTPUT = output_dir
    RESULTS = ROOT.parent / 'evaluation/results'
    ARMS = ('linear', 'mlp', 'residual_mlp')
    LABELS = ('Linear', 'MLP', 'Residual MLP')
    COLORS = ('#667085', '#2563eb', '#d97706')
    TOPIC_DRAWS = ('atlas14_s0', 'atlas14_s1', 'atlas14_s2')


    def read(relative):
        return json.loads((ROOT / relative).read_text())


    def write_csv(name, rows):
        with (OUTPUT / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


    core = read('core_ranges/summary.json')
    alignment = read('alignment/findings.json')
    atlas = read('alignment/atlas_matched_all_decoders.json')
    controls = read('controls/comparison.json')
    paired_mmlu = read('mmlu_paired_comparisons.json')
    values = {'linear': core['linear'], **core['arms']}
    pooled = next(row for row in controls['complete_alpha05_comparison'] if row['dataset'] == 'All concepts')
    atlas_means = {
        arm: {metric: float(np.mean([atlas['sets'][draw]['point'][arm][metric] for draw in TOPIC_DRAWS]))
              for metric in ('homogeneity', 'completeness', 'nmi')}
        for arm in ARMS
    }

    core_rows = []
    for arm in ARMS:
        value = values[arm]
        core_rows.append({
            'decoder': arm,
            'calinski_harabasz': value['geometry']['calinski_harabasz'],
            'davies_bouldin': value['geometry']['davies_bouldin'],
            'sampled_silhouette': value['geometry']['silhouette'],
            'within_cluster_variance_percent': 100 * value['geometry']['within_var_share'],
            'reconstruction_fvu': value['fvu'],
            'variance_explained_percent': 100 * (1 - value['fvu']),
            'mmlu_accuracy_percent': 100 * value['mmlu']['recon_acc'],
            'wikitext2_perplexity': value['ppl']['ppl'],
            'excess_cross_entropy_nats': value['ppl']['delta_ce'],
            'atlas_three_way_matched_homogeneity': atlas_means[arm]['homogeneity'],
            'atlas_three_way_matched_nmi': atlas_means[arm]['nmi'],
            'control_selectivity_gain_pp': pooled['paper' if arm == 'linear' else arm]['alphas']['0.5']['delta_selectivity'],
        })
    write_csv('core_comparison.csv', core_rows)

    alignment_rows = []
    for task, point in alignment['paper_qwen_bypass'].items():
        alignment_rows.append({'task': task, 'decoder': 'paper_linear', **point})
        for arm in ARMS[1:]:
            if task == 'atlas':
                result = json.loads((RESULTS / arm / 'evals/atlas_lastword/atlas_pair10.json').read_text())
                point = {metric: float(np.mean([result['sets'][draw]['point']['geoae'][metric] for draw in TOPIC_DRAWS]))
                         for metric in ('homogeneity', 'completeness', 'nmi')}
            else:
                raw = alignment['arms'][arm]['probes'][task]['paper_protocol_geoae']
                point = {metric: raw[metric] for metric in ('homogeneity', 'completeness', 'nmi')}
            alignment_rows.append({'task': task, 'decoder': arm, **point})
    write_csv('alignment_comparison.csv', alignment_rows)

    control_rows = []
    for comparison in controls['complete_alpha05_comparison']:
        for arm in ARMS:
            result = comparison['paper' if arm == 'linear' else arm]
            point = result['alphas']['0.5']
            lo, hi = point['delta_selectivity_ci95']
            control_rows.append({
                'dataset': comparison['dataset'], 'decoder': arm, 'alpha': 0.5,
                'concepts': result['n_concepts'], 'target_documents': result['n_target_documents'],
                'base_target_drop_pp': point['h']['target_drop'],
                'latent_target_drop_pp': point['z']['target_drop'],
                'base_collateral_pp': point['h']['collateral'],
                'latent_collateral_pp': point['z']['collateral'],
                'selectivity_gain_pp': point['delta_selectivity'],
                'within_run_ci95_low': lo, 'within_run_ci95_high': hi,
            })
    write_csv('control_comparison.csv', control_rows)

    summary = {
        'paper_sha256': hashlib.sha256((ROOT / 'paper_source/main.tex').read_bytes()).hexdigest(),
        'paper_source': str(ROOT / 'paper_source/main.tex'),
        'core': core_rows,
        'alignment_paper_protocol': alignment_rows,
        'atlas_exactly_matched_all_three': atlas_means,
        'controls_alpha05': control_rows,
        'mmlu_paired_tests': paired_mmlu,
        'limitations': [
            'One training seed; decoder parameter counts, biases and normalization constraints differ.',
            'Paper-protocol alignment masks vary by decoder; the extra Atlas three-way comparison uses identical rows.',
            'Controls use architecture-specific fresh jointly correct documents and fitted directions.',
            'Control confidence intervals are recomputed within-run concept-bootstrap intervals, not tests between decoders.',
            'The original range appendix is Llama; the new range extension is Qwen without a matched linear Qwen range arm.',
        ],
    }
    (OUTPUT / 'comparison_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.7))
    panels = (
        ('Cluster separation: CH ↑', 'calinski_harabasz', '{:.1f}'),
        ('Reconstruction error: FVU ↓', 'reconstruction_fvu', '{:.4f}'),
        ('WikiText-2 perplexity ↓', 'wikitext2_perplexity', '{:.3f}'),
        ('MMLU accuracy (%) ↑', 'mmlu_accuracy_percent', '{:.2f}'),
        ('Atlas homogeneity, identical rows ↑', 'atlas_three_way_matched_homogeneity', '{:.3f}'),
        ('Control selectivity gain (pp) ↑\nα = 0.5', 'control_selectivity_gain_pp', '{:.2f}'),
    )
    for ax, (title, key, fmt) in zip(axes.flat, panels):
        ys = np.array([row[key] for row in core_rows])
        if key == 'control_selectivity_gain_pp':
            bounds = [pooled['paper' if arm == 'linear' else arm]['alphas']['0.5']['delta_selectivity_ci95'] for arm in ARMS]
            errors = np.array([[ys[i] - bounds[i][0] for i in range(3)],
                               [bounds[i][1] - ys[i] for i in range(3)]])
            for i, color in enumerate(COLORS):
                ax.errorbar(i, ys[i], yerr=errors[:, i:i+1], fmt='o', color=color, capsize=5, markersize=8)
            ax.axhline(0, color='#cbd5e1', lw=1)
            ax.set_ylim(-1.2, 12.2)
            label_y = ys + errors[1] + .3
        else:
            for i, color in enumerate(COLORS):
                ax.plot(i, ys[i], 'o', color=color, markersize=9)
            spread = max(float(np.ptp(ys)), abs(float(ys.mean())) * .03)
            bottom = float(ys.min()) - spread * .45
            top = float(ys.max()) + spread * .60
            if key in ('mmlu_accuracy_percent', 'wikitext2_perplexity'):
                clean = 65.1 if key == 'mmlu_accuracy_percent' else 8.028514
                ax.axhline(clean, color='#94a3b8', ls='--', lw=1)
                ax.text(2.28, clean, 'Clean', color='#64748b', va='center', ha='right', fontsize=9,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
                bottom = min(bottom, clean - spread * .2)
                top = max(top, clean + spread * .35)
            ax.set_ylim(bottom, top)
            label_y = ys + (top - bottom) * .06
        for i, value in enumerate(ys):
            ax.text(i, label_y[i], fmt.format(value), ha='center', fontsize=11, color=COLORS[i])
        ax.set_title(title, pad=14)
        ax.set_xticks(range(3), LABELS)
        ax.set_xlim(-.4, 2.4)
        ax.grid(axis='y', color='#e2e8f0', lw=.7)
        ax.set_axisbelow(True)

    fig.suptitle('Qwen2.5-3B, final layer: nonlinear decoder tradeoffs', fontsize=17, fontweight='bold', y=.98)
    fig.text(.5, .935, '50 epochs · K = 2,000 · latent width 4,096 · one training seed', ha='center', color='#475569')
    fig.text(.06, .052, 'Linear values recovered from the paper’s result artifacts. Atlas uses identical rows across all decoders.', fontsize=10, color='#475569')
    fig.text(.06, .027, 'Control pools differ by decoder; error bars show within-run 95% concept-bootstrap intervals.', fontsize=10, color='#475569')
    fig.subplots_adjust(top=.84, bottom=.13, left=.075, right=.98, hspace=.55, wspace=.35)
    fig.savefig(OUTPUT / 'decoder_comparison.png', dpi=180, facecolor='white')
    fig.savefig(OUTPUT / 'decoder_comparison.pdf', facecolor='white')
    plt.close(fig)
    print('Wrote comparison_summary.json, three CSV tables, and decoder_comparison.png/.pdf')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, default=artifact_path(
        'runs/nonlinear_decoder_validation/deltaai/review'), help='Saved review JSON and evaluation results')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for tables and figures')
    args = parser.parse_args()
    analyze(args.review_dir, args.output_dir)


if __name__ == '__main__':
    main()
