#!/bin/bash
# Frequency-resolved forgetting on the five real datasets (sequential fine-tuning, seeds 0-2).
# Usage: bash scripts/run_real_band.sh <mamba48|linear48|mamba96>
#   mamba48  -> results/real_band/         (Mamba, H=48)
#   linear48 -> results/real_band_linear/  (linear forecaster, H=48)
#   mamba96  -> results/real_band_h96/     (Mamba, H=96)
# Then: python experiments/analyze_real_band.py results/real_band/*.json
source "$(dirname "$0")/common.sh"
case ${1:-mamba48} in
  mamba48)  extra=(); out=results/real_band ;;
  linear48) extra=(--backbone linear); out=results/real_band_linear ;;
  mamba96)  extra=(--pred_len 96); out=results/real_band_h96 ;;
  *) echo "unknown setting: $1" >&2; exit 1 ;;
esac
for b in electricity ettm1 ettm2 weather traffic; do
  read -r cfg data <<< "$(bench_config "$b")"
  $PY -u experiments/real_band_forgetting.py --name "$b" --config "$cfg" --data "$data" --seeds 0 1 2 \
    --out_dir "$out" "${extra[@]}" 2>&1 | tee "logs/band_${1:-mamba48}_$b.log"
done
