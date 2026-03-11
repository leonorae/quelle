#!/usr/bin/env bash
# Train a d12 nanochat model for Phase 0 baseline.
#
# Usage:
#   bash experiments/VVVVVV/src/train_d12.sh
#
# Override defaults with env vars:
#   N_ITERATIONS=3000 bash train_d12.sh
#   NANOCHAT_BASE_DIR=/my/path bash train_d12.sh
#   RESUME_FROM_STEP=5000 bash train_d12.sh   # resume from checkpoint at step 5000
#
# Requires setup.sh to have been run first.
# Logs to experiments/VVVVVV/outputs/train_d12.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/.."
NANOCHAT_DIR="$SCRIPT_DIR/../../../nanochat"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$EXP_DIR/outputs/nanochat_base}"
N_ITERATIONS="${N_ITERATIONS:-6000}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:--1}"
LOG_FILE="$EXP_DIR/outputs/train_d12.log"

mkdir -p "$EXP_DIR/outputs"

echo "=== VVVVVV d12 training ==="
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "N_ITERATIONS=$N_ITERATIONS"
echo "RESUME_FROM_STEP=$RESUME_FROM_STEP"
echo "Checkpoints: $NANOCHAT_BASE_DIR/base_checkpoints/d12/"
echo "Log: $LOG_FILE"
echo ""

cd "$NANOCHAT_DIR"

# Disable torch.compile: hangs indefinitely on Tesla T4 during triton kernel
# compilation (no progress after 50+ minutes). Eager mode is slightly slower
# per step but actually runs.
export TORCHDYNAMO_DISABLE=1

# Reduce CUDA allocator fragmentation. The logits tensor [B, 2048, 32768] fp32
# requires a large contiguous allocation; without expandable segments the
# allocator holds reserved-but-unallocated fragmented blocks it cannot reuse.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python -m scripts.base_train \
    --depth=12 \
    --num-iterations="$N_ITERATIONS" \
    --resume-from-step="$RESUME_FROM_STEP" \
    --save-every=1000 \
    --eval-every=250 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --run=dummy \
    --model-tag=d12 \
    --window-pattern=L \
    --device-batch-size=16 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=== Training complete. ==="
echo "Run Phase 0 probes next:  bash experiments/VVVVVV/src/run_phase0.sh"
