#!/bin/bash
# Babbelaar smoke test — exercises dataset loading, tokenizer training, base
# pretraining, and SFT on a CPU or MPS (Apple Silicon) box using a tiny model.
#
# Usage:
#   bash runs/runbabbelaar.sh
#
# Optional env vars:
#   NANOCHAT_DATA_DIR  — path to a directory already containing Babbelaar
#                        Parquet shards.  If set and the directory has files,
#                        the download step is skipped entirely.
#   SFT_TRAIN_FILE     — path to sft_train.jsonl produced by step 150.
#   SFT_VAL_FILE       — path to sft_val.jsonl produced by step 150.
#   WANDB_RUN          — wandb run name ('dummy' disables logging, the default).
#
# This run is NOT expected to produce a capable model — it just verifies that
# all code paths work end-to-end.  On an M3 MacBook Pro it takes ~15 minutes.

set -e

# ── Python environment ────────────────────────────────────────────────────────
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# ── Dataset ───────────────────────────────────────────────────────────────────
# If NANOCHAT_DATA_DIR is already set and populated, skip the download.
# Otherwise grab 3 train shards + all 4 validation shards (~600 MB).
if [ -z "$NANOCHAT_DATA_DIR" ] || [ -z "$(ls -A "$NANOCHAT_DATA_DIR" 2>/dev/null)" ]; then
    python -m nanochat.dataset --num-train-shards 3
    export NANOCHAT_DATA_DIR="$(pwd)/data/all"
    echo "NANOCHAT_DATA_DIR set to $NANOCHAT_DATA_DIR"
else
    echo "Using existing dataset at $NANOCHAT_DATA_DIR"
fi

# ── Tokenizer ─────────────────────────────────────────────────────────────────
# Train on at most 200 M characters (fast; enough to verify the pipeline).
python -m scripts.tok_train --max-chars 200000000 --vocab-size 32768
python -m scripts.tok_eval

# ── Base pretraining ──────────────────────────────────────────────────────────
# Tiny 4-layer model, short context, 300 iterations (~5 min on M3).
python -m scripts.base_train \
    --depth=4 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=256 \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --eval-every=100 \
    --eval-tokens=65536 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=300 \
    --run=$WANDB_RUN

# ── SFT ───────────────────────────────────────────────────────────────────────
if [ -z "$SFT_TRAIN_FILE" ]; then
    echo ""
    echo "SFT_TRAIN_FILE is not set — skipping SFT step."
    echo "To run SFT, set SFT_TRAIN_FILE (and optionally SFT_VAL_FILE) and re-run."
    echo "  export SFT_TRAIN_FILE=/path/to/sft_train.jsonl"
    echo "  export SFT_VAL_FILE=/path/to/sft_val.jsonl"
    exit 0
fi

SFT_VAL_ARGS=""
if [ -n "$SFT_VAL_FILE" ]; then
    SFT_VAL_ARGS="--sft-val-file $SFT_VAL_FILE"
fi

python -m scripts.chat_sft \
    --sft-file "$SFT_TRAIN_FILE" \
    $SFT_VAL_ARGS \
    --max-seq-len=256 \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --eval-every=100 \
    --eval-tokens=65536 \
    --chatcore-every=-1 \
    --num-iterations=100 \
    --run=$WANDB_RUN

echo ""
echo "Smoke test complete."
echo "To chat with the model: python -m scripts.chat_cli"
