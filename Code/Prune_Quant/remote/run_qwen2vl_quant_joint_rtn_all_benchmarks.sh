#!/usr/bin/env bash
set -euo pipefail

# Qwen2-VL + MASQuant pseudo + quant-joint GAE pruning benchmark suite.
# Runs VLMEvalKit MME first, then lmms-eval tasks:
#   MMMU, OCRBench, VizWiz, ScienceQA, TextVQA.
#
# Vision-token policy: cap only large images at 1500 visual tokens. Images that
# naturally produce fewer than 1500 visual tokens are left unchanged by keeping
# PROCESSOR_MIN_VISUAL_TOKENS empty.

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

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

append_if_set() {
  local array_name="$1"
  local flag="$2"
  local value="${3:-}"
  if [[ -n "$value" ]]; then
    local quoted_flag
    local quoted_value
    printf -v quoted_flag "%q" "$flag"
    printf -v quoted_value "%q" "$value"
    eval "$array_name+=( $quoted_flag $quoted_value )"
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

run_cmd() {
  log "$*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi
  "$@"
}

# ---- Paths ----
export PROJECT_ROOT="${PROJECT_ROOT:-/home/aistudio/LLMQuant-Learning/Code/Prune_Quant}"
export EXT_ROOT="${EXT_ROOT:-/home/aistudio/EXT}"
export MODEL_PATH="${MODEL_PATH:-/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct}"
export MASQUANT_ROOT="${MASQUANT_ROOT:-$EXT_ROOT/EfficientAI/masquant}"
export VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-$EXT_ROOT/VLMEvalKit}"
export LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-$PROJECT_ROOT/third_party/lmms-eval}"
export WORK_DIR="${WORK_DIR:-/home/aistudio/datasets/output/qwen2vl_quant_joint_rtn_all_benchmarks}"

# Calibration data used by MASQuant phase 1.
export CALIB_JSONL="${CALIB_JSONL:-/home/aistudio/datasets/data/calibration/qig_coco_train2017_caption.jsonl}"

# CMC calibration data. CMC reads the original ShareGPT4V-style JSON and image directory.
export CMC_VISION_JSON="${CMC_VISION_JSON:-/home/aistudio/datasets/data/calibration/qig_coco_train2017_caption.json}"
export CMC_VISION_PREFIX="${CMC_VISION_PREFIX:-/home/aistudio/datasets/data/calibration/qig_coco_train2017_images}"

require_path PROJECT_ROOT
require_path MODEL_PATH
require_path MASQUANT_ROOT
require_path VLMEVALKIT_ROOT
require_path LMMS_EVAL_ROOT

# ---- Runtime ----
export PYTHON="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---- Model and quantization ----
export MODEL_TYPE="${MODEL_TYPE:-qwen2vl}"
export DATASET_TYPE="${DATASET_TYPE:-text-vision}"
export NSAMPLES="${NSAMPLES:-128}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export WBITS="${WBITS:-4}"
export ABITS="${ABITS:-8}"
export GROUP_SIZE="${GROUP_SIZE:-0}"
export EPOCHS="${EPOCHS:-2}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
export DTYPE="${DTYPE:-bfloat16}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"

export PROCESSOR_USE_FAST="${PROCESSOR_USE_FAST:-}"
export PROCESSOR_MIN_PIXELS="${PROCESSOR_MIN_PIXELS:-}"
export PROCESSOR_MAX_PIXELS="${PROCESSOR_MAX_PIXELS:-}"
export PROCESSOR_MIN_VISUAL_TOKENS="${PROCESSOR_MIN_VISUAL_TOKENS:-}"
export PROCESSOR_MAX_VISUAL_TOKENS="${PROCESSOR_MAX_VISUAL_TOKENS:-1500}"

# ---- Joint pruning behavior ----
export PRUNER="${PRUNER:-gae_quant_joint}"
export CALIB_RETENTION_RATIO="${CALIB_RETENTION_RATIO:-0.5}"
export EVAL_RETENTION_RATIO="${EVAL_RETENTION_RATIO:-0.5}"
export MIN_KEEP="${MIN_KEEP:-1}"
export GAE_ANSWER_SOURCE="${GAE_ANSWER_SOURCE:-generated}"
export GAE_PER_TOKEN="${GAE_PER_TOKEN:-false}"
export GAE_QUANT_LAMBDA="${GAE_QUANT_LAMBDA:-0.5}"
export GAE_QUANT_METHOD="${GAE_QUANT_METHOD:-rtn}"
export RTN_BITS="${RTN_BITS:-4}"
export RTN_GROUP_SIZE="${RTN_GROUP_SIZE:-0}"
export GAE_DISABLE_MASQUANT_FAKE_QUANT="${GAE_DISABLE_MASQUANT_FAKE_QUANT:-1}"
export ALLOW_VANILLA_FALLBACK="${ALLOW_VANILLA_FALLBACK:-0}"
export PATCH_MASQUANT_INPUTS_EMBEDS_MASK="${PATCH_MASQUANT_INPUTS_EMBEDS_MASK:-0}"

# ---- CMC ----
export USE_CMC="${USE_CMC:-1}"
export CMC_NET="${CMC_NET:-qwen2-vl-7b}"
export CMC_CALI_DATA_TYPE="${CMC_CALI_DATA_TYPE:-vision-audio-only}"
export CMC_RANK="${CMC_RANK:-0.2}"
export CMC_QUANT_CMC="${CMC_QUANT_CMC:-0}"
export CMC_N_CALI_SAMPLES="${CMC_N_CALI_SAMPLES:-128}"

# ---- VLMEvalKit: MME ----
export VLMEVAL_DATASETS="${VLMEVAL_DATASETS:-MME}"
export VLMEVAL_MODEL_NAME="${VLMEVAL_MODEL_NAME:-Qwen2VL_MASQuant_Pseudo}"
export VLMEVAL_WORK_DIR="${VLMEVAL_WORK_DIR:-$WORK_DIR/vlmeval_mme_quant_joint_rtn}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export VLMEVAL_VERBOSE="${VLMEVAL_VERBOSE:-1}"
export VLMEVAL_SMART_RUNNER="${VLMEVAL_SMART_RUNNER:-1}"
export VLMEVAL_MODE="${VLMEVAL_MODE:-auto}"
export VLMEVAL_REUSE="${VLMEVAL_REUSE:-1}"
export VLMEVAL_REUSE_AUX="${VLMEVAL_REUSE_AUX:-1}"
export VLMEVAL_EXACT_MATCH_DATASETS="${VLMEVAL_EXACT_MATCH_DATASETS:-MME}"
export VLMEVAL_DISABLE_OPENAI="${VLMEVAL_DISABLE_OPENAI:-1}"

# ---- lmms-eval: MMMU/OCRBench/VizWiz/ScienceQA/TextVQA ----
export LMMS_EVAL_HF_HOME="${LMMS_EVAL_HF_HOME:-/home/aistudio/data/datasets/387822/abcd/hf_home}"
export LMMS_EVAL_TASKS="${LMMS_EVAL_TASKS:-mmmu_val ocrbench vizwiz_vqa_val scienceqa_img textvqa_val}"
export LMMS_EVAL_OUTPUT_PATH="${LMMS_EVAL_OUTPUT_PATH:-$WORK_DIR/lmms_eval_quant_joint_rtn}"
export LMMS_EVAL_CACHE="${LMMS_EVAL_CACHE:-$WORK_DIR/lmms_eval_cache}"
export LMMS_EVAL_LIMIT="${LMMS_EVAL_LIMIT:-}"
export LMMS_EVAL_LOG_SAMPLES="${LMMS_EVAL_LOG_SAMPLES:-1}"
export LMMS_EVAL_VERBOSITY="${LMMS_EVAL_VERBOSITY:-INFO}"
export LMMS_EVAL_DISABLE_OPENAI="${LMMS_EVAL_DISABLE_OPENAI:-1}"
export RUN_LMMS_EVAL="${RUN_LMMS_EVAL:-1}"

# ---- Resume / stage switches ----
model_name="${MODEL_PATH%/}"
model_name="${model_name##*/}"

export MASQUANT_ACT_SCALES="${MASQUANT_ACT_SCALES:-$WORK_DIR/act_scales/${model_name}-${DATASET_TYPE}-${NSAMPLES}.pt}"
export CMC_LOW_RANK="${CMC_LOW_RANK:-$WORK_DIR/cmc/low_rank_adapters_quantcmc${CMC_QUANT_CMC}_rank${CMC_RANK}_${CMC_CALI_DATA_TYPE}.pt}"
export CMC_WHITE="${CMC_WHITE:-$WORK_DIR/cmc/white_matrix_${CMC_CALI_DATA_TYPE}.pt}"

run_calibrate_was_set=0
run_cmc_was_set=0
if [[ -n "${RUN_CALIBRATE+x}" ]]; then
  run_calibrate_was_set=1
fi
if [[ -n "${RUN_CMC+x}" ]]; then
  run_cmc_was_set=1
fi

if [[ -z "${MASQUANT_RESUME:-}" && -d "$WORK_DIR/masquant_outputs" ]]; then
  found_resume="$(find "$WORK_DIR/masquant_outputs" -name mas_parameters.pth | sort | tail -n 1 || true)"
  if [[ -n "$found_resume" ]]; then
    export MASQUANT_RESUME="$found_resume"
  fi
fi

if [[ "$run_calibrate_was_set" == "0" ]]; then
  if [[ -n "${MASQUANT_RESUME:-}" && -f "$MASQUANT_RESUME" && -f "$MASQUANT_ACT_SCALES" ]]; then
    export RUN_CALIBRATE=0
  else
    export RUN_CALIBRATE=1
  fi
fi

if [[ "${RUN_CALIBRATE:-1}" == "1" ]]; then
  unset MASQUANT_RESUME
fi

if [[ "$run_cmc_was_set" == "0" ]]; then
  if [[ -f "$CMC_LOW_RANK" && -f "$CMC_WHITE" ]]; then
    export RUN_CMC=0
  else
    export RUN_CMC=1
  fi
fi

export RUN_INSTALL_VLMEVAL="${RUN_INSTALL_VLMEVAL:-1}"
export RUN_VLMEVAL="${RUN_VLMEVAL:-1}"

mkdir -p "$WORK_DIR"

log "vision_token_cap max=$PROCESSOR_MAX_VISUAL_TOKENS min=${PROCESSOR_MIN_VISUAL_TOKENS:-<unset>}"
log "lambda=$GAE_QUANT_LAMBDA rtn_bits=$RTN_BITS retention=$EVAL_RETENTION_RATIO"
log "VLMEvalKit datasets=$VLMEVAL_DATASETS"
log "lmms-eval tasks=$LMMS_EVAL_TASKS"

run_cmd bash "$PROJECT_ROOT/remote/run_masquant_pseudo_pipeline.sh"

if [[ "$RUN_LMMS_EVAL" == "1" ]]; then
  if [[ -z "${MASQUANT_RESUME:-}" ]]; then
    found_resume="$(find "$WORK_DIR/masquant_outputs" -name mas_parameters.pth | sort | tail -n 1 || true)"
    if [[ -n "$found_resume" ]]; then
      export MASQUANT_RESUME="$found_resume"
    fi
  fi
  if [[ -z "${MASQUANT_RESUME:-}" ]]; then
    echo "Could not find MASQuant resume under $WORK_DIR/masquant_outputs for lmms-eval." >&2
    echo "Set MASQUANT_RESUME or run calibration first." >&2
    exit 1
  fi
  if [[ ! -f "$MASQUANT_RESUME" ]]; then
    echo "MASQUANT_RESUME does not exist: $MASQUANT_RESUME" >&2
    exit 1
  fi

  export PYTHONPATH="$PROJECT_ROOT/src:$LMMS_EVAL_ROOT:${PYTHONPATH:-}"
  export QWEN2VL_MODEL="$MODEL_PATH"
  export QWEN25VL_MODEL="$MODEL_PATH"
  export MASQUANT_ROOT
  export MASQUANT_RESUME
  export MASQUANT_ACT_SCALES
  export CMC_LOW_RANK
  export CMC_WHITE
  export PQ_MODEL_TYPE="$MODEL_TYPE"
  export PQ_QUANT_METHOD=masquant
  export PQ_DTYPE="$DTYPE"
  export PQ_DEVICE_MAP="$DEVICE_MAP"
  export PQ_ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION"
  export PQ_MASQUANT_WBITS="$WBITS"
  export PQ_MASQUANT_ABITS="$ABITS"
  export PQ_MASQUANT_GROUP_SIZE="$GROUP_SIZE"
  export PQ_MASQUANT_INFERENCE_MODE="${PQ_MASQUANT_INFERENCE_MODE:-split_scales}"
  export PQ_MASQUANT_BATCH_SIZE="$BATCH_SIZE"
  export PQ_MASQUANT_SYMMETRIC="${PQ_MASQUANT_SYMMETRIC:-true}"
  export PQ_CMC_RANK="$CMC_RANK"
  export PQ_CMC_QUANT_CMC="$CMC_QUANT_CMC"
  export PQ_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
  export PQ_RETENTION_RATIO="$EVAL_RETENTION_RATIO"
  export PQ_MIN_KEEP="$MIN_KEEP"
  export PQ_PRUNER="$PRUNER"
  export PQ_GAE_ANSWER_SOURCE="$GAE_ANSWER_SOURCE"
  export PQ_GAE_PER_TOKEN="$GAE_PER_TOKEN"
  export PQ_GAE_QUANT_LAMBDA="$GAE_QUANT_LAMBDA"
  export PQ_GAE_QUANT_METHOD="$GAE_QUANT_METHOD"
  export PQ_RTN_BITS="$RTN_BITS"
  export PQ_RTN_GROUP_SIZE="$RTN_GROUP_SIZE"
  export PQ_GAE_DISABLE_MASQUANT_FAKE_QUANT="$GAE_DISABLE_MASQUANT_FAKE_QUANT"
  export PQ_ALLOW_VANILLA_FALLBACK="$ALLOW_VANILLA_FALLBACK"
  export PQ_MIN_PIXELS="$PROCESSOR_MIN_PIXELS"
  export PQ_MAX_PIXELS="$PROCESSOR_MAX_PIXELS"
  export PQ_MIN_VISUAL_TOKENS="$PROCESSOR_MIN_VISUAL_TOKENS"
  export PQ_MAX_VISUAL_TOKENS="$PROCESSOR_MAX_VISUAL_TOKENS"

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
    --batch-size "$BATCH_SIZE"
    --verbosity "$LMMS_EVAL_VERBOSITY"
  )
  append_if_set lmms_eval_cmd --limit "$LMMS_EVAL_LIMIT"
  append_bool_flag lmms_eval_cmd --log-samples "$LMMS_EVAL_LOG_SAMPLES"
  run_cmd "${lmms_eval_cmd[@]}"
else
  log "Skipping lmms-eval because RUN_LMMS_EVAL=$RUN_LMMS_EVAL"
fi

log "All requested benchmarks finished."
