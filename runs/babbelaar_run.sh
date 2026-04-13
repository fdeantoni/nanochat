#!/bin/bash
# Babbelaar full GPU run — pretraining + SFT for Max Babbelaar on 8×H100.
# Designed for a blank RunPod/Vast.ai node; takes approximately 3–4 hours.
#
# Usage:
#   bash runs/babbelaar_run.sh
# With screen + wandb logging (recommended):
#   WANDB_RUN=babbelaar screen -L -Logfile runs/babbelaar_run.log -S babbelaar bash runs/babbelaar_run.sh
#
# Optional env vars:
#   NANOCHAT_DATA_DIR  — path to an existing directory of Babbelaar Parquet shards.
#                        If set and populated, the corpus download is skipped.
#   SFT_TRAIN_FILE     — path to sft_train.jsonl.  If unset, the script
#                        downloads sft_train.jsonl + sft_val.jsonl from
#                        fdeantoni/max-babbelaar-sft automatically.
#   SFT_VAL_FILE       — path to sft_val.jsonl (auto-set alongside SFT_TRAIN_FILE
#                        when downloaded).
#   WANDB_RUN          — wandb run name ('dummy' disables logging, the default).
#                        Set this before running to enable wandb: WANDB_RUN=babbelaar

set -e
export OMP_NUM_THREADS=1

# ── Python environment ────────────────────────────────────────────────────────
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# ── Report ────────────────────────────────────────────────────────────────────
python -m nanochat.report reset

# ── Corpus download ───────────────────────────────────────────────────────────
# Download all 66 train shards + 4 validation shards from fdeantoni/max-babbelaar-corpus.
# Already-present files are skipped.  First batch of 8 shards is enough to start
# tokenizer training; download the rest in the background.
if [ -z "$NANOCHAT_DATA_DIR" ] || [ -z "$(ls -A "$NANOCHAT_DATA_DIR" 2>/dev/null)" ]; then
    python -m nanochat.dataset --num-train-shards 8
    export NANOCHAT_DATA_DIR="$(pwd)/data/all"
    echo "NANOCHAT_DATA_DIR set to $NANOCHAT_DATA_DIR"
    # Download remaining shards in background while tokenizer trains
    python -m nanochat.dataset &
    DATASET_DOWNLOAD_PID=$!
else
    echo "Using existing corpus at $NANOCHAT_DATA_DIR"
    DATASET_DOWNLOAD_PID=""
fi

# ── Tokenizer ─────────────────────────────────────────────────────────────────
python -m scripts.tok_train
python -m scripts.tok_eval

# ── Wait for full corpus ──────────────────────────────────────────────────────
if [ -n "$DATASET_DOWNLOAD_PID" ]; then
    echo "Waiting for full corpus download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi

# ── Base pretraining ──────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN

# ── Base evaluation ───────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size=16

# ── SFT files ─────────────────────────────────────────────────────────────────
# Auto-download from fdeantoni/max-babbelaar-sft if SFT_TRAIN_FILE is not set.
SFT_DIR="./data/sft"
if [ -z "$SFT_TRAIN_FILE" ]; then
    python -m nanochat.dataset --sft --sft-dir "$SFT_DIR"
    export SFT_TRAIN_FILE="$(pwd)/${SFT_DIR#./}/sft_train.jsonl"
    export SFT_VAL_FILE="$(pwd)/${SFT_DIR#./}/sft_val.jsonl"
    echo "SFT_TRAIN_FILE set to $SFT_TRAIN_FILE"
    echo "SFT_VAL_FILE set to $SFT_VAL_FILE"
fi

SFT_VAL_ARGS=""
if [ -n "$SFT_VAL_FILE" ]; then
    SFT_VAL_ARGS="--sft-val-file $SFT_VAL_FILE"
fi

# ── SFT ───────────────────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --sft-file "$SFT_TRAIN_FILE" \
    $SFT_VAL_ARGS \
    --device-batch-size=16 \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# ── Report ────────────────────────────────────────────────────────────────────
python -m nanochat.report generate

echo ""
echo "Run complete."
echo "To chat with Max Babbelaar: python -m scripts.chat_cli"
