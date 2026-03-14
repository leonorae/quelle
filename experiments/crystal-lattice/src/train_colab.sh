#!/usr/bin/env bash
# Train crystal-lattice on Colab.
#
# Usage:
#   bash experiments/crystal-lattice/src/train_colab.sh
#
# Override defaults with env vars:
#   PHASE=1 bash train_colab.sh   # run only Phase 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/.."
PHASE="${PHASE:-all}"
LOG_FILE="$EXP_DIR/outputs/train.log"

mkdir -p "$EXP_DIR/outputs"

echo "=== crystal-lattice training ==="
echo "Phase: $PHASE"
echo "Log: $LOG_FILE"
echo ""

cd "$EXP_DIR"
python -m src.curiosity_loop \
    --phase "$PHASE" \
    --output-dir outputs \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=== Training complete. ==="
