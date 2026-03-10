#!/usr/bin/env bash
# One-time setup: download ClimbMix data shards and train the tokenizer.
#
# Usage:
#   bash experiments/VVVVVV/src/setup.sh
#
# Override defaults with env vars:
#   NANOCHAT_BASE_DIR=/my/path N_SHARDS=50 bash setup.sh
#   TORCH_EXTRA=gpu bash setup.sh   # install GPU torch (cuda 12.8); default: cpu
#
# N_SHARDS: number of ClimbMix train shards to download (default 30).
# "~170 shards, enough for GPT-2" per nanochat docs; 30 is ample for d12 Phase 0.
# The validation shard is always downloaded automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/.."
NANOCHAT_DIR="$SCRIPT_DIR/../../../nanochat"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$EXP_DIR/outputs/nanochat_base}"
N_SHARDS="${N_SHARDS:-30}"
TORCH_EXTRA="${TORCH_EXTRA:-cpu}"

echo "=== VVVVVV Phase 0 setup ==="
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "NANOCHAT_DIR=$NANOCHAT_DIR"
echo "N_SHARDS=$N_SHARDS"
echo "TORCH_EXTRA=$TORCH_EXTRA"
echo ""

# Install nanochat dependencies via uv
echo "--- Installing dependencies (--extra $TORCH_EXTRA) ---"
cd "$NANOCHAT_DIR"
uv sync --extra "$TORCH_EXTRA"

# Download data shards (+ val shard is always included)
echo ""
echo "--- Downloading $N_SHARDS ClimbMix train shards ---"
uv run python -m nanochat.dataset -n "$N_SHARDS"

# Train tokenizer on downloaded shards
echo ""
echo "--- Training tokenizer ---"
uv run python -m scripts.tok_train

echo ""
echo "=== Setup complete. ==="
echo "Run training next:  bash experiments/VVVVVV/src/train_d12.sh"
