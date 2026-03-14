#!/usr/bin/env bash
# Train variable-bitrate-reasoning on Colab.
#
# Usage:
#   bash experiments/variable-bitrate-reasoning/src/train_colab.sh
#
# The model generates its own arithmetic data, so no setup script is needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/.."
CONFIG="${CONFIG:-$EXP_DIR/configs/default.yaml}"
LOG_FILE="$EXP_DIR/outputs/train.log"

mkdir -p "$EXP_DIR/outputs"

echo "=== variable-bitrate-reasoning training ==="
echo "Config: $CONFIG"
echo "Log: $LOG_FILE"
echo ""

cd "$EXP_DIR"
python -m src.train --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=== Training complete. ==="
echo "Run evaluation next:  cd $EXP_DIR && python -m src.evaluate --config $CONFIG"
