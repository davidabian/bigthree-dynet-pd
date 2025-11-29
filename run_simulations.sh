#!/bin/bash
# bigthree-dynet-pd: full simulation grid used in the primary article.
#
# This script reproduces the main grid of simulations reported in:
#   Abián, D., Bernad, J., Ilarri, S. & Trillo-Lado, R. (2025).
#   "Individual and collective gains from cooperation and reciprocity in a
#    dynamic-network Prisoner’s Dilemma driven by extraversion, openness,
#    and agreeableness."
#
# If you use these simulations or any derived data, please make sure you
# cite the article and software as described in README.md.

python3 bigthree_dynet_pd.py \
  --n-agents 30,100,200 \
  --n-turns 300 \
  --beta 0.2,0.5,0.8 \
  --seed 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 \
  --min-ideal-degree 1 \
  --max-ideal-degree 10 \
  --init-graph bounded12 \
  --init-degree2-fraction 0.5 \
  --init-edge-prob 0.05 \
  --personality-source beta \
  --trait-scenarios baseline,agreeableness_hi,agreeableness_lo,openness_hi,openness_lo,extraversion_hi,extraversion_lo \
  --payoff-sets "3,1,5,0" \
  --exchange-cost 2.0 \
  --leniency-delta 0.20 \
  --damping-lambda 0.5 \
  --outdir grid_bigthree_dynet_pd \
  2>&1 | tee run_simulations_output.log
