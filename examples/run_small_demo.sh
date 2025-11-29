#!/bin/bash
python3 bigthree_dynet_pd.py \
  --n-agents 30 \
  --n-turns 50 \
  --beta 0.5 \
  --seed 101,102,103,104,105 \
  --min-ideal-degree 1 \
  --max-ideal-degree 10 \
  --init-graph bounded12 \
  --init-degree2-fraction 0.5 \
  --init-edge-prob 0.05 \
  --personality-source beta \
  --trait-scenarios baseline \
  --payoff-sets "3,1,5,0" \
  --exchange-cost 2.0 \
  --leniency-delta 0.20 \
  --damping-lambda 0.5 \
  --outdir grid_small_demo
