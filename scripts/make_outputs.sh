#!/bin/bash
# Regenerate the paper's tables and figures from the result files in results/ (no training needed).
# Output: outputs/tables/*.tex and outputs/figs/*.pdf
source "$(dirname "$0")/common.sh"
mkdir -p outputs/tables outputs/figs
$PY experiments/make_tables.py --out outputs/tables                 # benchmark, ablation, efficiency
$PY experiments/make_paper_figures.py --out outputs/figs --tables outputs/tables
$PY experiments/analyze_sensitivity.py --order A --tex outputs/tables/tab_sensitivity.tex
$PY experiments/analyze_synthetic.py results/synthetic_overlap_*.json
$PY experiments/analyze_intervention.py results/int1024_*.json
for d in real_band real_band_linear real_band_h96; do
  echo "== $d"; $PY experiments/analyze_real_band.py results/$d/*.json
done
$PY experiments/pooled_test.py
# needs the CSV files in data/:
if [ -f data/electricity.csv ]; then
  $PY experiments/make_regime_table.py --out outputs/tables/tab_regimes.tex
  # IFR's gain over ER per old task vs. the task's intrusion ("Benefit versus intrusion")
  $PY experiments/gain_vs_discrepancy.py \
    --bench electricity results/pilot/electricity configs/within/electricity.yaml data/electricity.csv \
    --bench traffic results/pilot/traffic configs/within/traffic.yaml data/traffic.csv \
    --bench ettm1 results/main/ettm1 configs/within/ettm1.yaml data/ETTm1.csv \
    --bench ettm2 results/pilot/ettm2 configs/within/ettm2.yaml data/ETTm2.csv \
    --bench weather results/main/weather configs/within/weather.yaml data/weather.csv \
    --bench order_A results/crossdomain/order_A configs/crossdomain/crossdomain_A.yaml data \
    --bench order_B results/crossdomain/order_B configs/crossdomain/crossdomain_B.yaml data \
    --bench order_C results/crossdomain/order_C configs/crossdomain/crossdomain_C.yaml data \
    --out outputs/gain_vs_discrepancy.json
fi
