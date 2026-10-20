#!/bin/bash
# Controlled studies with AR(2) regimes of known spectra, and the numerical check of Proposition 1.
# Usage: bash scripts/run_synthetic.sh
source "$(dirname "$0")/common.sh"

# Overlap study (Figs. 2-3): forgetting vs. spectral overlap on three architectures.
$PY -u experiments/synthetic_overlap.py --backbones linear --out results/synthetic_overlap_linear.json
$PY -u experiments/synthetic_overlap.py --backbones mlp --out results/synthetic_overlap_mlp.json
$PY -u experiments/synthetic_overlap.py --backbones mamba --seeds 0 1 2 --out results/synthetic_overlap_mamba_a.json
$PY -u experiments/synthetic_overlap.py --backbones mamba --seeds 3 4 --out results/synthetic_overlap_mamba_b.json

# Intervention study (Fig. 5, Table 4): replay losses with a 1,024-window buffer, ten seeds.
for r in 0.0 0.5; do
  tag=$([ "$r" = 0.0 ] && echo r0 || echo r05)
  $PY -u experiments/synthetic_intervention.py --rhos $r --buffer 1024 --seeds 0 1 2 3 4 \
    --out results/int1024_${tag}_a.json
  $PY -u experiments/synthetic_intervention.py --rhos $r --buffer 1024 --seeds 5 6 7 8 9 \
    --out results/int1024_${tag}_b.json
  # the 128-window buffer that over-fits (reported in the text)
  $PY -u experiments/synthetic_intervention.py --rhos $r --buffer 128 --out results/synthetic_intervention_${tag}.json
done

# Proposition 1: measured vs. predicted increase of the old regime's error (Table 2).
$PY -u experiments/verify_proposition.py --seeds 10 --out_json results/proposition_check.json \
  --out_tex outputs/tables/tab_prop.tex
