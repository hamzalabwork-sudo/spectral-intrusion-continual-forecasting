#!/bin/bash
# Sensitivity of IFR to kappa and the replay weight lambda on the validation split of a cross-domain order.
# Usage: bash scripts/run_sensitivity.sh A      ->  results/tuning/crossdomain_A/{er_w*, ifr_w*_k*}/
source "$(dirname "$0")/common.sh"
o=${1:-A}; out=results/tuning/crossdomain_$o
for w in 0.5 1.0 2.0; do
  $PY -u run_continual.py --config "configs/crossdomain/crossdomain_$o.yaml" --data data --holdout val \
    --methods ifr_uniform --epochs 20 --seeds 0 1 2 --set ifr.weight=$w --out_dir "$out/er_w$w"
  for k in 1.0 4.0 16.0; do
    $PY -u run_continual.py --config "configs/crossdomain/crossdomain_$o.yaml" --data data --holdout val \
      --methods ifr --epochs 20 --seeds 0 1 2 --set ifr.weight=$w ifr.kappa=$k --out_dir "$out/ifr_w${w}_k$k"
  done
done
