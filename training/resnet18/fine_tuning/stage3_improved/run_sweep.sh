#!/usr/bin/env bash
# Stage 3 (improved) — class-balanced fine-tuning sweep.
#
# Run the strength=0.0 SANITY run FIRST and verify reproduction before the sweep:
#   conda run -n chess python training_scripts/train.py --run_name s00 --balance_strength 0.0
#   conda run -n chess python check_reproduction.py --run_name s00
#
# Then this script runs the post-sanity sweep (0.3 / 0.5 / 0.7 / 1.0).
# Everything else (data, aug, optimizer, schedule, checkpoint metric) is identical.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/.conda/envs/chess/bin/python}"
TRAIN="$HERE/training_scripts/train.py"

declare -a STRENGTHS=("0.3" "0.5" "0.7" "1.0")
declare -a NAMES=("s03" "s05" "s07" "s10")
FORMULA="${FORMULA:-inv_sqrt}"

echo "python: $PY"
echo "formula: $FORMULA   strengths: ${STRENGTHS[*]}"

for i in "${!STRENGTHS[@]}"; do
    s="${STRENGTHS[$i]}"; name="${NAMES[$i]}"
    echo ""
    echo "=================================================================="
    echo ">>> RUN $name  --balance_strength $s  --balance_formula $FORMULA"
    echo "=================================================================="
    "$PY" "$TRAIN" --run_name "$name" --balance_strength "$s" --balance_formula "$FORMULA"
done

echo ""
echo ">>> Sweep complete. Building report ..."
"$PY" "$HERE/build_report.py"
