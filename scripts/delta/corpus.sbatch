#!/bin/bash
# Build the frozen text corpus for a sampled-position dump, on a CPU node.
#
#   sbatch -A <cpu-account> scripts/delta/corpus.sbatch [extract_config.yaml]
#
# WHY A SEPARATE JOB. geoae.build_corpus streams documents from the HF hub and
# tokenises them; it never touches a GPU. Inside extract_sampled.sbatch it burns
# ~40 min of an A100's allocation doing nothing but network and CPU work. GPU
# hours are this project's binding constraint (201 on bbyl-delta-gpu on
# 2026-09-26), so it is charged to a *-delta-cpu allocation instead.
#
# Run this first, then extract_sampled.sbatch skips its own corpus step (that
# step is incremental and finds the corpus already complete).
#
# The corpus is SHARED by the no-struct and struct5 configs (both name
# corpus/sampled_v1). Building the struct5 config's corpus after this one adds
# only its structured source; the per-source token quotas differ between the two
# configs, but those are applied at EXTRACT time, not here.
#
# Needs outbound internet to huggingface.co — verified reachable from Delta on
# 2026-09-26, and the sources (fineweb, fineweb-edu, wikipedia, pile, codeparrot,
# finemath) are all public, so no HF token is required. The Llama tokenizer is
# gated but loads from the cache under $HF_HOME.
#
# ~40 min and ~0.3 GB for the 10M-row config, plus the structured source.
#SBATCH --job-name=geoae-corpus
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
CFG="${1:-configs/base/llama3.2-3b_extract_sampled.yaml}"
source scripts/delta/env.sh
mkdir -p logs

# --workers 3: each open streaming dataset holds Parquet row groups, ~3-5 GB RSS
# apiece, so this is a memory bound rather than a CPU one.
uv run --no-sync python -u -m geoae.build_corpus --config "$CFG" --workers 3 \
  2>&1 | tee -a "logs/corpus_$(basename "$CFG" .yaml).log"

echo
echo "Corpus step done for $CFG. Next (GPU):"
echo "  sbatch scripts/delta/extract_sampled.sbatch $CFG"
