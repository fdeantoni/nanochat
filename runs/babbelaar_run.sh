#!/bin/bash
# Babbelaar full GPU run — pretraining + SFT for Max Babbelaar.
# Designed for RunPod/Vast.ai spot instances; resilient to preemption.
# All state persists on NANOCHAT_BASE_DIR (network volume) so runs resume automatically.
#
# Usage:
#   bash runs/babbelaar_run.sh
# With screen + wandb logging (recommended):
#   WANDB_RUN=babbelaar screen -L -Logfile runs/babbelaar_run.log -S babbelaar bash runs/babbelaar_run.sh
#
# Optional env vars:
#   NANOCHAT_BASE_DIR  — base directory for all nanochat artifacts (default: ~/.cache/nanochat).
#                        Set to a network volume path for spot instance persistence.
#   NANOCHAT_DATA_DIR  — path to an existing directory of Babbelaar Parquet shards.
#                        If set and populated, the corpus download is skipped.
#   NANOCHAT_SFT_DIR   — override the directory where SFT files are downloaded
#                        (default: ~/.cache/nanochat/sft_data_babbelaar).
#   SFT_TRAIN_FILE     — path to sft_train.jsonl.  If unset, the script
#                        downloads sft_train.jsonl + sft_val.jsonl from
#                        fdeantoni/max-babbelaar-sft automatically.
#   SFT_VAL_FILE       — path to sft_val.jsonl (auto-set alongside SFT_TRAIN_FILE
#                        when downloaded).
#   NPROC_PER_NODE     — number of GPUs to use (default: auto-detect via nvidia-smi).
#                        Set to 1 for single-GPU runs.
#   WANDB_RUN          — wandb run name ('dummy' disables logging, the default).
#                        Set this before running to enable wandb: WANDB_RUN=babbelaar
#   MODEL_TAG          — model tag override (default: empty, uses d${DEPTH}).

export OMP_NUM_THREADS=1
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
echo "Using $NPROC_PER_NODE GPU(s) for training"

# ── Training config (single source of truth) ─────────────────────────
DEPTH=24
MODEL_TAG="${MODEL_TAG:-}"
TARGET_PARAM_DATA_RATIO=8
DEVICE_BATCH_SIZE=16
SAVE_EVERY=500

# Derive checkpoint subdir name exactly as base_train.py does:
#   output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
if [ -n "$MODEL_TAG" ]; then
    CKPT_DIRNAME="$MODEL_TAG"
else
    CKPT_DIRNAME="d${DEPTH}"
fi

NANOCHAT_BASE="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
PRETRAIN_CKPT_DIR="${NANOCHAT_BASE}/base_checkpoints/${CKPT_DIRNAME}"
SFT_CKPT_DIR="${NANOCHAT_BASE}/chatsft_checkpoints/${CKPT_DIRNAME}"

MODEL_TAG_ARG=""
if [ -n "$MODEL_TAG" ]; then
    MODEL_TAG_ARG="--model-tag=$MODEL_TAG"
fi

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# ── Python environment ────────────────────────────────────────────────────────
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# ── Stage markers (on network volume for persistence) ─────────────────
MARKER_DIR="${NANOCHAT_BASE}/babbelaar_markers"
mkdir -p "$MARKER_DIR"

# ── Report ────────────────────────────────────────────────────────────────────
python -m nanochat.report reset

# ── Corpus download ───────────────────────────────────────────────────────────
if [ ! -f "$MARKER_DIR/corpus_done" ]; then
    if [ -z "$NANOCHAT_DATA_DIR" ] || [ -z "$(ls -A "$NANOCHAT_DATA_DIR" 2>/dev/null)" ]; then
        python -m nanochat.dataset --num-train-shards 8
        python -m nanochat.dataset &
        DATASET_DOWNLOAD_PID=$!
    else
        echo "Using existing corpus at $NANOCHAT_DATA_DIR"
        DATASET_DOWNLOAD_PID=""
    fi
else
    echo "Corpus already downloaded (marker exists), skipping."
    DATASET_DOWNLOAD_PID=""
fi

# ── Tokenizer ─────────────────────────────────────────────────────────────────
if [ ! -f "$MARKER_DIR/tokenizer_done" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
    touch "$MARKER_DIR/tokenizer_done"
else
    echo "Tokenizer already trained (marker exists), skipping."
fi

# ── Wait for full corpus ──────────────────────────────────────────────────────
if [ -n "$DATASET_DOWNLOAD_PID" ]; then
    echo "Waiting for full corpus download to complete..."
    wait $DATASET_DOWNLOAD_PID
    touch "$MARKER_DIR/corpus_done"
fi

# ── Base pretraining (with resume support) ────────────────────────────────────
if [ ! -f "$MARKER_DIR/pretrain_done" ]; then
    RESUME_STEP=-1

    if [ -d "$PRETRAIN_CKPT_DIR" ]; then
        LATEST_META=$(ls "$PRETRAIN_CKPT_DIR"/meta_*.json 2>/dev/null | sort -V | tail -1)
        if [ -n "$LATEST_META" ]; then
            # Extract step number from filename like meta_003500.json
            RESUME_STEP=$(basename "$LATEST_META" | grep -oP '\d+' | sed 's/^0*//')
            RESUME_STEP=${RESUME_STEP:-0}
            echo "Found pretrain checkpoint at step $RESUME_STEP in $PRETRAIN_CKPT_DIR, resuming..."
        fi
    fi

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$DEPTH \
        --target-param-data-ratio=$TARGET_PARAM_DATA_RATIO \
        --device-batch-size=$DEVICE_BATCH_SIZE \
        --fp8 \
        --save-every=$SAVE_EVERY \
        --resume-from-step=$RESUME_STEP \
        $MODEL_TAG_ARG \
        --run=$WANDB_RUN || {
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 143 ]; then
            echo "Pretraining interrupted by signal (exit code $EXIT_CODE). Safe to restart."
            exit 0
        else
            echo "Pretraining failed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
    }
    touch "$MARKER_DIR/pretrain_done"
else
    echo "Pretraining already complete (marker exists), skipping."
fi

# ── Base evaluation ───────────────────────────────────────────────────────────
if [ ! -f "$MARKER_DIR/base_eval_done" ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
        --device-batch-size=$DEVICE_BATCH_SIZE || {
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 143 ]; then
            echo "Base eval interrupted by signal (exit code $EXIT_CODE). Safe to restart."
            exit 0
        else
            echo "Base eval failed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
    }
    touch "$MARKER_DIR/base_eval_done"
else
    echo "Base eval already complete (marker exists), skipping."
fi

# ── SFT files ─────────────────────────────────────────────────────────────────
if [ -z "$SFT_TRAIN_FILE" ]; then
    python -m nanochat.dataset --sft
    SFT_DIR=$(python -c "from nanochat.dataset import DEFAULT_SFT_DIR; print(DEFAULT_SFT_DIR)")
    export SFT_TRAIN_FILE="${SFT_DIR}/sft_train.jsonl"
    export SFT_VAL_FILE="${SFT_DIR}/sft_val.jsonl"
    echo "SFT_TRAIN_FILE set to $SFT_TRAIN_FILE"
    echo "SFT_VAL_FILE set to $SFT_VAL_FILE"
fi

SFT_VAL_ARGS=""
if [ -n "$SFT_VAL_FILE" ]; then
    SFT_VAL_ARGS="--sft-val-file $SFT_VAL_FILE"
fi

# ── SFT ───────────────────────────────────────────────────────────────────────
if [ ! -f "$MARKER_DIR/sft_done" ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
        --sft-file "$SFT_TRAIN_FILE" \
        $SFT_VAL_ARGS \
        --device-batch-size=$DEVICE_BATCH_SIZE \
        $MODEL_TAG_ARG \
        --run=$WANDB_RUN || {
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 143 ]; then
            echo "SFT interrupted by signal (exit code $EXIT_CODE). Safe to restart."
            exit 0
        else
            echo "SFT failed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
    }
    touch "$MARKER_DIR/sft_done"
else
    echo "SFT already complete (marker exists), skipping."
fi

# ── SFT evaluation ────────────────────────────────────────────────────────────
if [ ! -f "$MARKER_DIR/sft_eval_done" ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft || {
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 143 ]; then
            echo "SFT eval interrupted by signal (exit code $EXIT_CODE). Safe to restart."
            exit 0
        else
            echo "SFT eval failed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
    }
    touch "$MARKER_DIR/sft_eval_done"
else
    echo "SFT eval already complete (marker exists), skipping."
fi

# ── Report ────────────────────────────────────────────────────────────────────
python -m nanochat.report generate

echo ""
echo "Run complete."
echo "To chat with Max Babbelaar: python -m scripts.chat_cli"
