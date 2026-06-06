#!/usr/bin/env bash
set -euo pipefail

# Copy this file to a local run script, edit the variables below, then run it.
# This script evaluates vanilla Qwen2-VL through the Prune_Quant VLMEvalKit
# wrapper with pruning disabled: no pruning, no MASQuant, no CMC, no TensorRT.

# ---- Paths ----
export PROJECT_ROOT=/home/aistudio/LLMQuant-Learning/Code/Prune_Quant
export EXT_ROOT=/home/aistudio/EXT
export MODEL_PATH=/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct
export VLMEVALKIT_ROOT="$EXT_ROOT/VLMEvalKit"
export WORK_DIR=/home/aistudio/datasets/output/qwen2vl_vanilla

# Use python3 on machines where `python` is not available.
export PYTHON="${PYTHON:-python3}"

# ---- Model ----
export QWEN2VL_MODEL="$MODEL_PATH"
# Full-token evaluation does not request attention scores, so use accelerated
# SDPA by default while still allowing callers to override it.
export PQ_ATTN_IMPLEMENTATION="${PQ_ATTN_IMPLEMENTATION:-sdpa}"
export PQ_MAX_NEW_TOKENS=16

# Optional image resolution controls.
# Keep this fixed so vanilla, pure pruning, and MASQuant runs are comparable.
export PQ_MIN_VISUAL_TOKENS=1500
export PQ_MAX_VISUAL_TOKENS=1500
export PQ_MIN_PIXELS=
export PQ_MAX_PIXELS=

# ---- Pruning disabled ----
export PQ_RETENTION_RATIO=1.0
export PQ_MIN_KEEP=1
export PQ_GAE_ANSWER_SOURCE=generated
export PQ_GAE_PER_TOKEN=false

# ---- VLMEvalKit ----
export VLMEVAL_DATASETS="MME MMStar"
export VLMEVAL_MODEL_NAME=Qwen2VL_PrunedGAE
export VLMEVAL_WORK_DIR="$WORK_DIR/vlmeval_mme_mmstar_qwen2vl_vanilla"
export VLMEVAL_VERBOSE=1
export VLMEVAL_MODE=auto
export VLMEVAL_REUSE=1
export VLMEVAL_REUSE_AUX=1
export VLMEVAL_EXACT_MATCH_DATASETS="MME MMStar"

# MME/MMStar do not need GPT-as-judge. Keep OpenAI env vars out of exact-matching
# scoring so VLMEvalKit cannot accidentally call an external judge.
export VLMEVAL_DISABLE_OPENAI=1

# ---- Stage switches ----
export RUN_INSTALL_VLMEVAL=1
export RUN_VLMEVAL=1

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
require_path VLMEVALKIT_ROOT
mkdir -p "$WORK_DIR"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

if [[ "$RUN_INSTALL_VLMEVAL" == "1" ]]; then
  "$PYTHON" "$PROJECT_ROOT/remote/install_vlmeval_pruned_gae.py" --vlmeval-root "$VLMEVALKIT_ROOT"
fi

if [[ "$RUN_VLMEVAL" == "1" ]]; then
  if [[ "$VLMEVAL_DISABLE_OPENAI" == "1" ]]; then
    unset OPENAI_API_KEY
    unset OPENAI_API_BASE
    unset OPENAI_API_MODEL
    unset OPENAI_API_TYPE
    unset OPENAI_API_VERSION
    unset AZURE_OPENAI_API_KEY
    unset LOCAL_LLM
  fi

  read -r -a vlmeval_datasets <<< "$VLMEVAL_DATASETS"
  vlmeval_cmd=(
    "$PYTHON" "$PROJECT_ROOT/remote/run_vlmeval_smart.py"
    --vlmeval-root "$VLMEVALKIT_ROOT"
    --data "${vlmeval_datasets[@]}"
    --model "$VLMEVAL_MODEL_NAME"
    --work-dir "$VLMEVAL_WORK_DIR"
    --python "$PYTHON"
    --mode "$VLMEVAL_MODE"
    --exact-match-datasets "$VLMEVAL_EXACT_MATCH_DATASETS"
  )
  append_bool_flag vlmeval_cmd --verbose "$VLMEVAL_VERBOSE"
  if [[ "$VLMEVAL_REUSE" != "1" ]]; then
    vlmeval_cmd+=(--no-reuse)
  fi
  if [[ "$VLMEVAL_REUSE_AUX" != "1" ]]; then
    vlmeval_cmd+=(--no-reuse-aux)
  fi
  "${vlmeval_cmd[@]}"
fi
