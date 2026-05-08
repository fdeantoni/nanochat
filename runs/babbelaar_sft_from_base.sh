#!/bin/bash
# SFT-only run for Max Babbelaar — downloads the base model from HuggingFace and runs SFT.
# Skips pretraining entirely; designed for spot instances where the base checkpoint lives on HF.
#
# Usage:
#   bash runs/babbelaar_sft_from_base.sh
# With screen + wandb logging (recommended):
#   WANDB_RUN=babbelaar screen -L -Logfile runs/babbelaar_sft_run.log -S babbelaar bash runs/babbelaar_sft_from_base.sh
#
# Optional env vars:
#   NANOCHAT_BASE_DIR     — base directory for all nanochat artifacts (default: ~/.cache/nanochat).
#                           Set to a network volume path for spot instance persistence.
#   NANOCHAT_SFT_DIR      — override the directory where SFT files are downloaded
#                           (default: $NANOCHAT_BASE_DIR/sft_data_babbelaar).
#   SFT_TRAIN_FILE        — path to sft_train.jsonl.  If unset, downloaded from
#                           fdeantoni/max-babbelaar-sft automatically.
#   SFT_VAL_FILE          — path to sft_val.jsonl (auto-set alongside SFT_TRAIN_FILE).
#   NPROC_PER_NODE        — number of GPUs to use (default: auto-detect via nvidia-smi).
#                           Set to 1 for single-GPU runs.
#   DEPTH                 — model depth (default: 18). Must match the uploaded base checkpoint.
#   MODEL_TAG             — model tag override (default: d${DEPTH}).
#   MODEL_STEP            — specific base checkpoint step to start SFT from (default: latest).
#   SFT_NUM_ITERATIONS    — override SFT iteration count (default: auto-scaled for ~131M tokens).
#   SAVE_EVERY_SFT        — checkpoint cadence in steps (default: 500).
#   WANDB_RUN             — wandb run name ('dummy' disables logging, the default).
#   HF_BASE_REPO          — HuggingFace repo to download the base model from
#                           (default: fdeantoni/max-babbelaar-base).
#   RESUME_SFT_STEP       — resume SFT from an existing chatsft checkpoint at this step.
#                           Clears sft_done/sft_eval_done markers automatically and passes
#                           --resume-sft-step to chat_sft.  SFT_NUM_ITERATIONS then controls
#                           how many additional steps to run (default: auto-scaled as usual).
#   CLEAN                 — set to "true" to wipe SFT stage markers and checkpoints
#                           before starting, forcing a completely fresh SFT run.
#                           Does NOT remove base_checkpoints (download phase is always re-used).

export OMP_NUM_THREADS=1
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$NPROC_PER_NODE" -eq 0 ] && NPROC_PER_NODE=1
echo "Using $NPROC_PER_NODE GPU(s) for training"

# ── Training config ───────────────────────────────────────────────────
DEPTH="${DEPTH:-18}"
MODEL_TAG="${MODEL_TAG:-}"
SAVE_EVERY_SFT="${SAVE_EVERY_SFT:-500}"
HF_BASE_REPO="${HF_BASE_REPO:-fdeantoni/max-babbelaar-base}"

# Derive checkpoint subdir name exactly as base_train.py does
if [ -n "$MODEL_TAG" ]; then
    CKPT_DIRNAME="$MODEL_TAG"
else
    CKPT_DIRNAME="d${DEPTH}"
fi
MODEL_TAG_ARG="--model-tag=${MODEL_TAG:-$CKPT_DIRNAME}"

MODEL_STEP="${MODEL_STEP:-}"
MODEL_STEP_ARG=""
if [ -n "$MODEL_STEP" ]; then
    MODEL_STEP_ARG="--model-step=$MODEL_STEP"
    echo "SFT will start from base checkpoint step $MODEL_STEP"
fi

RESUME_SFT_STEP="${RESUME_SFT_STEP:-}"
RESUME_SFT_STEP_ARG=""
if [ -n "$RESUME_SFT_STEP" ]; then
    RESUME_SFT_STEP_ARG="--resume-sft-step=$RESUME_SFT_STEP"
    echo "Resuming SFT from chatsft checkpoint step $RESUME_SFT_STEP"
fi

# SFT batch size and iteration count (same formula as babbelaar_run.sh)
DEVICE_BATCH_SIZE=16
SFT_TOTAL_BATCH_SIZE=$((DEVICE_BATCH_SIZE * 2048 * NPROC_PER_NODE))
SFT_NUM_ITERATIONS_DEFAULT=$((4000 / NPROC_PER_NODE))
[ $SFT_NUM_ITERATIONS_DEFAULT -lt 200 ] && SFT_NUM_ITERATIONS_DEFAULT=200
SFT_NUM_ITERATIONS="${SFT_NUM_ITERATIONS:-$SFT_NUM_ITERATIONS_DEFAULT}"

NANOCHAT_BASE="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
SFT_CKPT_DIR="${NANOCHAT_BASE}/chatsft_checkpoints/${CKPT_DIRNAME}"

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# ── Clean start (CLEAN=true) ──────────────────────────────────────────
if [ "${CLEAN}" = "true" ]; then
    echo "[CLEAN] Removing SFT stage markers and SFT checkpoints for a fresh run..."
    rm -f "${NANOCHAT_BASE}/babbelaar_markers/sft_done" \
          "${NANOCHAT_BASE}/babbelaar_markers/sft_eval_done"
    rm -rf "$SFT_CKPT_DIR"
    echo "[CLEAN] Done."
fi

# ── Resume: clear done markers so SFT reruns from the given checkpoint ─
if [ -n "$RESUME_SFT_STEP" ]; then
    rm -f "${NANOCHAT_BASE}/babbelaar_markers/sft_done" \
          "${NANOCHAT_BASE}/babbelaar_markers/sft_eval_done"
    echo "[RESUME] Cleared sft_done/sft_eval_done markers — will resume from step $RESUME_SFT_STEP."
fi

# ── System deps ───────────────────────────────────────────────────────
command -v gcc &> /dev/null || apt-get install -y build-essential

# ── Python environment ────────────────────────────────────────────────
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
VENV_DIR="${NANOCHAT_BASE}/.venv"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
[ -d "$VENV_DIR" ] || uv venv "$VENV_DIR"
uv sync --extra gpu
source "$VENV_DIR/bin/activate"

# ── Stage markers ─────────────────────────────────────────────────────
MARKER_DIR="${NANOCHAT_BASE}/babbelaar_markers"
mkdir -p "$MARKER_DIR"

# ── Download base model from HuggingFace ─────────────────────────────
if [ ! -f "$MARKER_DIR/base_download_done" ]; then
    echo "Downloading base model (${CKPT_DIRNAME}) + tokenizer from ${HF_BASE_REPO}..."
    python3 - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="${HF_BASE_REPO}",
    repo_type="model",
    allow_patterns=["base_checkpoints/${CKPT_DIRNAME}/**", "tokenizer/**"],
    local_dir="${NANOCHAT_BASE}",
    local_dir_use_symlinks=False,
)
print("Base model downloaded to ${NANOCHAT_BASE}")
PY
    touch "$MARKER_DIR/base_download_done"
else
    echo "Base model already downloaded (marker exists), skipping."
fi

# Verify the download succeeded
BASE_CKPT_DIR="${NANOCHAT_BASE}/base_checkpoints/${CKPT_DIRNAME}"
if [ ! -d "$BASE_CKPT_DIR" ] || [ -z "$(ls "$BASE_CKPT_DIR"/model_*.pt 2>/dev/null)" ]; then
    echo "ERROR: No model checkpoint found in ${BASE_CKPT_DIR} after download."
    echo "       Check that ${HF_BASE_REPO} contains base_checkpoints/${CKPT_DIRNAME}/model_*.pt"
    exit 1
fi
echo "Base checkpoint directory: ${BASE_CKPT_DIR}"

# ── SFT files ─────────────────────────────────────────────────────────
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

# ── SFT training ─────────────────────────────────────────────────────
if [ ! -f "$MARKER_DIR/sft_done" ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
        --sft-file "$SFT_TRAIN_FILE" \
        $SFT_VAL_ARGS \
        --device-batch-size=$DEVICE_BATCH_SIZE \
        --total-batch-size=$SFT_TOTAL_BATCH_SIZE \
        --num-iterations=$SFT_NUM_ITERATIONS \
        --save-every=$SAVE_EVERY_SFT \
        --eval-every=$SAVE_EVERY_SFT \
        --chatcore-every=-1 \
        $MODEL_TAG_ARG \
        $MODEL_STEP_ARG \
        $RESUME_SFT_STEP_ARG \
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

# ── SFT evaluation ────────────────────────────────────────────────────
# Pick the checkpoint with the lowest val_bpb (saturation typically before final step)
SFT_EVAL_STEP_ARG=""
if [ -d "$SFT_CKPT_DIR" ]; then
    BEST_SFT_STEP=$(python3 - "$SFT_CKPT_DIR" <<'PY' 2>/dev/null
import json, glob, os, sys
ckpt_dir = sys.argv[1]
best = None
for p in sorted(glob.glob(os.path.join(ckpt_dir, 'meta_*.json'))):
    try:
        with open(p) as fh:
            d = json.load(fh)
        bpb = d.get('val_bpb')
        if bpb is None:
            continue
        step = int(os.path.basename(p).split('_')[-1].split('.')[0])
        if best is None or bpb < best[0]:
            best = (bpb, step)
    except Exception:
        pass
if best:
    print(best[1])
PY
)
    if [ -n "$BEST_SFT_STEP" ]; then
        SFT_EVAL_STEP_ARG="--step=$BEST_SFT_STEP"
        echo "Using best SFT checkpoint by val_bpb: step $BEST_SFT_STEP"
    else
        echo "No val_bpb metadata in $SFT_CKPT_DIR; chat_eval will use latest step."
    fi
fi

if [ ! -f "$MARKER_DIR/sft_eval_done" ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft \
        -a "ARC-Easy|ARC-Challenge|MMLU|BabbelaarPersonaProbe|BabbelaarTemporalBoundary|BabbelaarDutchResponse" \
        $MODEL_TAG_ARG $SFT_EVAL_STEP_ARG || {
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

echo ""
echo "SFT run complete."
echo "To chat with Max Babbelaar: python -m scripts.chat_cli"
