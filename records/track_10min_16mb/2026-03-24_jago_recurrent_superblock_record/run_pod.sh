#!/usr/bin/env bash
# Single entrypoint for RunPod / SSH: avoids broken copy-paste of long torchrun lines.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCRIPT="$HERE/train_gpt.py"

cd "$ROOT"

export REPO_ROOT="${REPO_ROOT:-$ROOT}"
export DATA_PATH="${DATA_PATH:-$ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/data/tokenizers/fineweb_1024_bpe.model}"

export SEED="${SEED:-1337}"
export RECURRENT_SUPERBLOCK="${RECURRENT_SUPERBLOCK:-1}"
export PHYSICAL_LAYERS="${PHYSICAL_LAYERS:-6}"
export VIRTUAL_LAYERS="${VIRTUAL_LAYERS:-12}"
export DEPTH_ADAPTER_RANK="${DEPTH_ADAPTER_RANK:-8}"
export VIRTUAL_DEPTH_DROPOUT="${VIRTUAL_DEPTH_DROPOUT:-1}"
export VDD_START_KEEP="${VDD_START_KEEP:-0.67}"
export VDD_END_FULL_FRAC="${VDD_END_FULL_FRAC:-0.25}"
export XSA_LAST_N="${XSA_LAST_N:-12}"
export VR_ENABLED="${VR_ENABLED:-1}"
export GA_ENABLED="${GA_ENABLED:-1}"
export BIGRAM_HASH="${BIGRAM_HASH:-2048}"
export PARTIAL_ROPE_DIM="${PARTIAL_ROPE_DIM:-16}"
export EMA_ENABLED="${EMA_ENABLED:-1}"
export EMA_DECAY="${EMA_DECAY:-0.997}"
export QAT_START_FRAC="${QAT_START_FRAC:-0.18}"
export MIXED_INT5_INT6="${MIXED_INT5_INT6:-1}"
export PRUNE_FRAC="${PRUNE_FRAC:-0.00}"
export USE_LZMA="${USE_LZMA:-1}"
export MODEL_DIM="${MODEL_DIM:-576}"
export MLP_MULT="${MLP_MULT:-3.5}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export ITERATIONS="${ITERATIONS:-9000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export SLIDING_EVAL_BATCH_SEQS="${SLIDING_EVAL_BATCH_SEQS:-32}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
export MUON_WD="${MUON_WD:-0.04}"
export ADAM_WD="${ADAM_WD:-0.04}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"

NPROC="${NPROC:-8}"
if ! [[ "$NPROC" =~ ^[0-9]+$ ]]; then
  echo "NPROC must be an integer, got: $NPROC" >&2
  exit 1
fi

echo "RUN_POD: ROOT=$ROOT REPO_ROOT=$REPO_ROOT SCRIPT=$SCRIPT NPROC=$NPROC" >&2
test -f "$SCRIPT" || { echo "Missing $SCRIPT" >&2; exit 1; }

exec torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT"
