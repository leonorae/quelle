#!/usr/bin/env bash
# Run Phase 0 diagnostic probes on the trained d12 checkpoint.
#
# Usage:
#   bash experiments/VVVVVV/src/run_phase0.sh [extra args passed to run_phase0.py]
#
# Override checkpoint step:
#   bash run_phase0.sh --step 6000
#
# Override NANOCHAT_BASE_DIR:
#   NANOCHAT_BASE_DIR=/my/path bash run_phase0.sh
#
# Results: experiments/VVVVVV/outputs/phase0_results.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/.."
NANOCHAT_DIR="$SCRIPT_DIR/../../../nanochat"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$EXP_DIR/outputs/nanochat_base}"
export PYTHONPATH="$NANOCHAT_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d12"

mkdir -p "$EXP_DIR/outputs"

echo "=== VVVVVV Phase 0 diagnostics ==="
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

cd "$NANOCHAT_DIR"

uv run python "$SCRIPT_DIR/run_phase0.py" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "$@"
