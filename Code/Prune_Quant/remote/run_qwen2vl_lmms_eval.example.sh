#!/usr/bin/env bash
set -euo pipefail

# Copy this file to a local run script, edit the variables below, then run it.
# This script evaluates Qwen2-VL with lmms-eval on MMMU, OCRBench, VizWiz,
# ScienceQA, and TextVQA through the Prune_Quant wrapper.

# ---- Paths ----
export PROJECT_ROOT=/home/aistudio/LLMQuant-Learning/Code/Prune_Quant
export MODEL_PATH=/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct
export LMMS_EVAL_ROOT="$PROJECT_ROOT/third_party/lmms-eval"
export WORK_DIR=/home/aistudio/datasets/output/qwen2vl_lmms_eval

# Use python3 on machines where `python` is not available.
export PYTHON="${PYTHON:-python3}"

# ---- Model ----
export QWEN2VL_MODEL="$MODEL_PATH"
export PQ_MODEL_TYPE=qwen2vl
export PQ_ATTN_IMPLEMENTATION=eager
export PQ_MAX_NEW_TOKENS=16

# Optional image resolution controls. Keep this fixed across VLMEvalKit and
# lmms-eval when comparing pruning/quantization variants.
export PQ_MIN_VISUAL_TOKENS="${PQ_MIN_VISUAL_TOKENS:-}"
export PQ_MAX_VISUAL_TOKENS="${PQ_MAX_VISUAL_TOKENS:-1500}"
export PQ_MIN_PIXELS=
export PQ_MAX_PIXELS=

# ---- GAE pruning ----
# 1.0 disables pruning for a vanilla baseline. Use 0.5 for GAE 50%.
export PQ_RETENTION_RATIO=0.5
export PQ_MIN_KEEP=1
export PQ_GAE_ANSWER_SOURCE=generated
export PQ_GAE_PER_TOKEN=false

# ---- lmms-eval ----
export LMMS_EVAL_TASKS="${LMMS_EVAL_TASKS:-mmmu_val ocrbench vizwiz_vqa_val scienceqa_img textvqa_val}"
export LMMS_EVAL_OUTPUT_PATH="$WORK_DIR/lmms_eval_qwen2vl_gae_r${PQ_RETENTION_RATIO}"
export LMMS_EVAL_CACHE="$WORK_DIR/lmms_eval_cache"
export LMMS_EVAL_LIMIT=
export LMMS_EVAL_LOG_SAMPLES=1
export LMMS_EVAL_DISABLE_OPENAI=1

require_path() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "Missing required variable: $name" >&2
    exit 1
  fi
  if [[ ! -e "$value" ]]; then
    echo "$name does not exist: $value" >&2
    exit 1
  fi
}

append_bool_flag() {
  local array_name="$1"
  local flag="$2"
  local value="${3:-0}"
  if [[ "$value" == "1" || "$value" == "true" || "$value" == "TRUE" ]]; then
    local quoted_flag
    printf -v quoted_flag "%q" "$flag"
    eval "$array_name+=( $quoted_flag )"
  fi
}

require_path PROJECT_ROOT
require_path MODEL_PATH
require_path LMMS_EVAL_ROOT
mkdir -p "$WORK_DIR"

export PYTHONPATH="$PROJECT_ROOT/src:$LMMS_EVAL_ROOT:${PYTHONPATH:-}"

read -r -a lmms_eval_tasks <<< "$LMMS_EVAL_TASKS"
lmms_eval_cmd=(
  "$PYTHON" "$PROJECT_ROOT/remote/run_lmms_eval_smart.py"
  --lmms-eval-root "$LMMS_EVAL_ROOT"
  --tasks "${lmms_eval_tasks[@]}"
  --model prune_quant_qwen2vl
  --model-path "$QWEN2VL_MODEL"
  --output-path "$LMMS_EVAL_OUTPUT_PATH"
  --cache "$LMMS_EVAL_CACHE"
  --python "$PYTHON"
  --batch-size 1
  --verbosity "${LMMS_EVAL_VERBOSITY:-INFO}"
)
if [[ -n "$LMMS_EVAL_LIMIT" ]]; then
  lmms_eval_cmd+=(--limit "$LMMS_EVAL_LIMIT")
fi
append_bool_flag lmms_eval_cmd --log-samples "$LMMS_EVAL_LOG_SAMPLES"
"${lmms_eval_cmd[@]}"
