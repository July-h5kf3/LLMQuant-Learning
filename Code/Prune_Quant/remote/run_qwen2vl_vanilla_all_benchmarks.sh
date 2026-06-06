#!/usr/bin/env bash
set -euo pipefail

# Qwen2-VL original-model benchmark suite: no pruning, no quantization, no CMC.
# Runs VLMEvalKit MME first, then lmms-eval tasks:
#   MMMU, OCRBench, VizWiz, ScienceQA, TextVQA.
#
# Vision-token policy: cap only large images at 1500 visual tokens. Images that
# naturally produce fewer than 1500 visual tokens are left unchanged by keeping
# PQ_MIN_VISUAL_TOKENS empty.

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

# ---- Paths ----
export PROJECT_ROOT="${PROJECT_ROOT:-/home/aistudio/LLMQuant-Learning/Code/Prune_Quant}"
export EXT_ROOT="${EXT_ROOT:-/home/aistudio/EXT}"
export MODEL_PATH="${MODEL_PATH:-/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct}"
export VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-$EXT_ROOT/VLMEvalKit}"
export LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-$PROJECT_ROOT/third_party/lmms-eval}"
export WORK_DIR="${WORK_DIR:-/home/aistudio/datasets/output/qwen2vl_vanilla_all_benchmarks}"

require_path PROJECT_ROOT
require_path MODEL_PATH
require_path VLMEVALKIT_ROOT
require_path LMMS_EVAL_ROOT
mkdir -p "$WORK_DIR"

# ---- Runtime ----
export PYTHON="${PYTHON:-python3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$PROJECT_ROOT/src:$LMMS_EVAL_ROOT:${PYTHONPATH:-}"

# ---- Model ----
export QWEN2VL_MODEL="$MODEL_PATH"
export PQ_MODEL_TYPE="${PQ_MODEL_TYPE:-qwen2vl}"
export PQ_QUANT_METHOD="${PQ_QUANT_METHOD:-none}"
export PQ_ATTN_IMPLEMENTATION="${PQ_ATTN_IMPLEMENTATION:-sdpa}"
export PQ_MAX_NEW_TOKENS="${PQ_MAX_NEW_TOKENS:-16}"
export PQ_DTYPE="${PQ_DTYPE:-auto}"
export PQ_DEVICE_MAP="${PQ_DEVICE_MAP:-auto}"

# ---- No pruning ----
export PQ_RETENTION_RATIO="${PQ_RETENTION_RATIO:-1.0}"
export PQ_MIN_KEEP="${PQ_MIN_KEEP:-1}"
export PQ_PRUNER="${PQ_PRUNER:-gae_oracle}"
export PQ_GAE_ANSWER_SOURCE="${PQ_GAE_ANSWER_SOURCE:-generated}"
export PQ_GAE_PER_TOKEN="${PQ_GAE_PER_TOKEN:-false}"

# ---- Vision-token cap ----
export PQ_MIN_PIXELS="${PQ_MIN_PIXELS:-}"
export PQ_MAX_PIXELS="${PQ_MAX_PIXELS:-}"
export PQ_MIN_VISUAL_TOKENS="${PQ_MIN_VISUAL_TOKENS:-}"
export PQ_MAX_VISUAL_TOKENS="${PQ_MAX_VISUAL_TOKENS:-1500}"

# ---- VLMEvalKit: MME ----
export VLMEVAL_DATASETS="${VLMEVAL_DATASETS:-MME}"
export VLMEVAL_MODEL_NAME="${VLMEVAL_MODEL_NAME:-Qwen2VL_PrunedGAE}"
export VLMEVAL_WORK_DIR="${VLMEVAL_WORK_DIR:-$WORK_DIR/vlmeval_mme_qwen2vl_vanilla}"
export VLMEVAL_VERBOSE="${VLMEVAL_VERBOSE:-1}"
export VLMEVAL_MODE="${VLMEVAL_MODE:-auto}"
export VLMEVAL_REUSE="${VLMEVAL_REUSE:-1}"
export VLMEVAL_REUSE_AUX="${VLMEVAL_REUSE_AUX:-1}"
export VLMEVAL_EXACT_MATCH_DATASETS="${VLMEVAL_EXACT_MATCH_DATASETS:-MME}"
export VLMEVAL_DISABLE_OPENAI="${VLMEVAL_DISABLE_OPENAI:-1}"

# ---- lmms-eval: MMMU/OCRBench/VizWiz/ScienceQA/TextVQA ----
export LMMS_EVAL_TASKS="${LMMS_EVAL_TASKS:-mmmu_val ocrbench vizwiz_vqa_val scienceqa_img textvqa_val}"
export LMMS_EVAL_OUTPUT_PATH="${LMMS_EVAL_OUTPUT_PATH:-$WORK_DIR/lmms_eval_qwen2vl_vanilla}"
export LMMS_EVAL_CACHE="${LMMS_EVAL_CACHE:-$WORK_DIR/lmms_eval_cache}"
export LMMS_EVAL_LIMIT="${LMMS_EVAL_LIMIT:-}"
export LMMS_EVAL_LOG_SAMPLES="${LMMS_EVAL_LOG_SAMPLES:-1}"
export LMMS_EVAL_VERBOSITY="${LMMS_EVAL_VERBOSITY:-INFO}"
export LMMS_EVAL_DISABLE_OPENAI="${LMMS_EVAL_DISABLE_OPENAI:-1}"

# ---- Stage switches ----
export RUN_INSTALL_VLMEVAL="${RUN_INSTALL_VLMEVAL:-1}"
export RUN_VLMEVAL="${RUN_VLMEVAL:-1}"
export RUN_LMMS_EVAL="${RUN_LMMS_EVAL:-1}"

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

if [[ "$RUN_LMMS_EVAL" == "1" ]]; then
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
    --verbosity "$LMMS_EVAL_VERBOSITY"
  )
  if [[ -n "$LMMS_EVAL_LIMIT" ]]; then
    lmms_eval_cmd+=(--limit "$LMMS_EVAL_LIMIT")
  fi
  append_bool_flag lmms_eval_cmd --log-samples "$LMMS_EVAL_LOG_SAMPLES"
  "${lmms_eval_cmd[@]}"
fi
