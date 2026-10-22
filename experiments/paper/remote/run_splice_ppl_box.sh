#!/usr/bin/env bash
# WikiText-2 splice perplexity for every Llama and Gemma 1B arm (box). Logs: logs/splice_ppl_*.log
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"; source configs/sites/local.sh >/dev/null 2>&1
P=".venv/bin/python -u -m geoae.evaluation.perplexity"
L=checkpoints/llama3.2-3B/layer27; R=results/delta; S=step_0014450.pt
$P --ae $R/d6144/evals/splice_ppl.json=$L/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/$S \
   --ae $R/d12288/evals/splice_ppl.json=$L/k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias/$S \
   --ae $R/d3072/evals/splice_ppl.json=$L/k2000_bnh_b32k_lam1_d3072_dpc_sampled_tokbias/$S \
   --ae $R/d768/evals/splice_ppl.json=$L/k2000_bnh_b32k_lam1_d768_dpc_sampled_tokbias/$S \
   --ae $R/d6144k4000/evals/splice_ppl.json=$L/k4000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/$S \
   --ae $R/d12288k4000/evals/splice_ppl.json=$L/k4000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias/$S \
   --ae $R/d6144/evals/splice_ppl_nobypass.json=$L/k2000_bnh_b32k_lam1_d6144_dpc_sampled/$S \
   --ae $R/d6144/evals/splice_ppl_decoderonly.json=$L/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly/$S \
   --ae $R/d6144/evals/splice_ppl_alpha0p5.json=$L/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_alpha0p5/$S \
   > logs/splice_ppl_llama.log 2>&1
for LY in 13 25; do
  G=checkpoints/gemma3-1B/layer$LY; args=""
  for W in 1152 2304 4608; do args="$args --ae $R/g1b_l${LY}_d$W/evals/splice_ppl.json=$G/k2000_bnh_b32k_lam1_d${W}_dpc_sampled_struct5_tokbias/$S"; done
  $P $args > logs/splice_ppl_g1b_l$LY.log 2>&1
done
echo SPLICE_PPL_BOX_DONE >> logs/splice_ppl_llama.log
