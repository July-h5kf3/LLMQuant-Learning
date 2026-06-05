#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required variable: $name" >&2
    exit 1
  fi
}

require_path() {
  local name="$1"
  local value="${!name:-}"
  require_var "$name"
  if [[ ! -e "$value" ]]; then
    echo "$name does not exist: $value" >&2
    exit 1
  fi
}

run_cmd() {
  log "$*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi
  "$@"
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

PROJECT_ROOT="${PROJECT_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-}"
MASQUANT_ROOT="${MASQUANT_ROOT:-}"
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-}"
WORK_DIR="${WORK_DIR:-}"
CALIB_JSONL="${CALIB_JSONL:-}"

require_path PROJECT_ROOT
require_path MODEL_PATH
require_path MASQUANT_ROOT
require_path VLMEVALKIT_ROOT
require_var WORK_DIR

PYTHON="${PYTHON:-python}"
MODEL_TYPE="${MODEL_TYPE:-qwen2vl}"
DATASET_TYPE="${DATASET_TYPE:-text-vision}"
NSAMPLES="${NSAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WBITS="${WBITS:-4}"
ABITS="${ABITS:-8}"
GROUP_SIZE="${GROUP_SIZE:-0}"
EPOCHS="${EPOCHS:-2}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
PROCESSOR_USE_FAST="${PROCESSOR_USE_FAST:-}"
PROCESSOR_MIN_PIXELS="${PROCESSOR_MIN_PIXELS:-}"
PROCESSOR_MAX_PIXELS="${PROCESSOR_MAX_PIXELS:-}"
PROCESSOR_MIN_VISUAL_TOKENS="${PROCESSOR_MIN_VISUAL_TOKENS:-}"
PROCESSOR_MAX_VISUAL_TOKENS="${PROCESSOR_MAX_VISUAL_TOKENS:-}"

CALIB_RETENTION_RATIO="${CALIB_RETENTION_RATIO:-1.0}"
EVAL_RETENTION_RATIO="${EVAL_RETENTION_RATIO:-1.0}"
MIN_KEEP="${MIN_KEEP:-1}"
GAE_ANSWER_SOURCE="${GAE_ANSWER_SOURCE:-generated}"
GAE_PER_TOKEN="${GAE_PER_TOKEN:-false}"
PRUNER="${PRUNER:-gae_oracle}"
GAE_QUANT_LAMBDA="${GAE_QUANT_LAMBDA:-0.5}"
GAE_QUANT_METHOD="${GAE_QUANT_METHOD:-rtn}"
RTN_BITS="${RTN_BITS:-4}"
RTN_GROUP_SIZE="${RTN_GROUP_SIZE:-0}"
GAE_DISABLE_MASQUANT_FAKE_QUANT="${GAE_DISABLE_MASQUANT_FAKE_QUANT:-1}"
ALLOW_VANILLA_FALLBACK="${ALLOW_VANILLA_FALLBACK:-0}"
PATCH_MASQUANT_INPUTS_EMBEDS_MASK="${PATCH_MASQUANT_INPUTS_EMBEDS_MASK:-0}"

CMC_CALI_DATA_TYPE="${CMC_CALI_DATA_TYPE:-vision-audio-only}"
CMC_RANK="${CMC_RANK:-0.2}"
CMC_QUANT_CMC="${CMC_QUANT_CMC:-0}"
CMC_N_CALI_SAMPLES="${CMC_N_CALI_SAMPLES:-128}"
CMC_SCRIPT_NAME="${CMC_SCRIPT_NAME:-infer_mas.py}"
CMC_NET="${CMC_NET:-}"
CMC_VISION_JSON="${CMC_VISION_JSON:-}"
CMC_VISION_PREFIX="${CMC_VISION_PREFIX:-}"
CMC_AUDIO_JSON="${CMC_AUDIO_JSON:-}"
CMC_AUDIO_PREFIX="${CMC_AUDIO_PREFIX:-}"
CMC_EXTRA_ARGS="${CMC_EXTRA_ARGS:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
VLMEVAL_DATASETS="${VLMEVAL_DATASETS:-MME}"
VLMEVAL_MODEL_NAME="${VLMEVAL_MODEL_NAME:-Qwen2VL_MASQuant_Pseudo}"
VLMEVAL_VERBOSE="${VLMEVAL_VERBOSE:-1}"
VLMEVAL_DISABLE_OPENAI="${VLMEVAL_DISABLE_OPENAI:-1}"
VLMEVAL_SMART_RUNNER="${VLMEVAL_SMART_RUNNER:-1}"
VLMEVAL_MODE="${VLMEVAL_MODE:-auto}"
VLMEVAL_REUSE="${VLMEVAL_REUSE:-1}"
VLMEVAL_REUSE_AUX="${VLMEVAL_REUSE_AUX:-1}"
VLMEVAL_JUDGE="${VLMEVAL_JUDGE:-}"
VLMEVAL_EXACT_MATCH_DATASETS="${VLMEVAL_EXACT_MATCH_DATASETS:-MME MMStar}"
VLMEVAL_FORCE_EVAL="${VLMEVAL_FORCE_EVAL:-0}"

RUN_CALIBRATE="${RUN_CALIBRATE:-1}"
RUN_CMC="${RUN_CMC:-1}"
RUN_INSTALL_VLMEVAL="${RUN_INSTALL_VLMEVAL:-1}"
RUN_VLMEVAL="${RUN_VLMEVAL:-1}"
USE_CMC="${USE_CMC:-1}"

mkdir -p "$WORK_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

model_name="${MODEL_PATH%/}"
model_name="${model_name##*/}"

MASQUANT_ACT_SCALES="${MASQUANT_ACT_SCALES:-$WORK_DIR/act_scales/${model_name}-${DATASET_TYPE}-${NSAMPLES}.pt}"
MASQUANT_RESUME="${MASQUANT_RESUME:-}"
CMC_WHITE="${CMC_WHITE:-$WORK_DIR/cmc/white_matrix_${CMC_CALI_DATA_TYPE}.pt}"
CMC_LOW_RANK="${CMC_LOW_RANK:-$WORK_DIR/cmc/low_rank_adapters_quantcmc${CMC_QUANT_CMC}_rank${CMC_RANK}_${CMC_CALI_DATA_TYPE}.pt}"
VLMEVAL_WORK_DIR="${VLMEVAL_WORK_DIR:-$WORK_DIR/vlmeval_mme_masquant_pseudo}"

if [[ -z "$CMC_NET" ]]; then
  if [[ "$MODEL_TYPE" == "qwen2_5_vl" ]]; then
    CMC_NET="qwen2.5-vl-7b"
  else
    CMC_NET="qwen2-vl-7b"
  fi
fi

cd "$PROJECT_ROOT"

if [[ "$RUN_CALIBRATE" == "1" ]]; then
  require_path CALIB_JSONL
  calibrate_cmd=(
    "$PYTHON" -m prune_quant_baseline.scripts.run_prune_then_quant_masquant
    --stage calibrate
    --model-type "$MODEL_TYPE"
    --model-path "$MODEL_PATH"
    --masquant-root "$MASQUANT_ROOT"
    --work-dir "$WORK_DIR"
    --calib-jsonl "$CALIB_JSONL"
    --dtype "$DTYPE"
    --device-map "$DEVICE_MAP"
    --local-files-only "$LOCAL_FILES_ONLY"
    --trust-remote-code "$TRUST_REMOTE_CODE"
    --attn-implementation "$ATTN_IMPLEMENTATION"
    --retention-ratio "$CALIB_RETENTION_RATIO"
    --min-keep "$MIN_KEEP"
    --pruner "$PRUNER"
    --gae-answer-source "$GAE_ANSWER_SOURCE"
    --gae-per-token "$GAE_PER_TOKEN"
    --gae-quant-lambda "$GAE_QUANT_LAMBDA"
    --gae-quant-method "$GAE_QUANT_METHOD"
    --rtn-bits "$RTN_BITS"
    --rtn-group-size "$RTN_GROUP_SIZE"
    --dataset-type "$DATASET_TYPE"
    --nsamples "$NSAMPLES"
    --batch-size "$BATCH_SIZE"
    --wbits "$WBITS"
    --abits "$ABITS"
    --epochs "$EPOCHS"
    --group-size "$GROUP_SIZE"
    --masquant-act-scales "$MASQUANT_ACT_SCALES"
  )
  append_if_set calibrate_cmd --processor-use-fast "$PROCESSOR_USE_FAST"
  append_if_set calibrate_cmd --processor-min-pixels "$PROCESSOR_MIN_PIXELS"
  append_if_set calibrate_cmd --processor-max-pixels "$PROCESSOR_MAX_PIXELS"
  append_if_set calibrate_cmd --processor-min-visual-tokens "$PROCESSOR_MIN_VISUAL_TOKENS"
  append_if_set calibrate_cmd --processor-max-visual-tokens "$PROCESSOR_MAX_VISUAL_TOKENS"
  append_bool_flag calibrate_cmd --patch-masquant-inputs-embeds-mask "$PATCH_MASQUANT_INPUTS_EMBEDS_MASK"
  run_cmd "${calibrate_cmd[@]}"
else
  log "Skipping MASQuant calibration because RUN_CALIBRATE=$RUN_CALIBRATE"
fi

if [[ -z "$MASQUANT_RESUME" ]]; then
  MASQUANT_RESUME="$(find "$WORK_DIR/masquant_outputs" -name mas_parameters.pth | sort | tail -n 1 || true)"
fi
if [[ -z "$MASQUANT_RESUME" ]]; then
  echo "Could not find MASQuant resume under $WORK_DIR/masquant_outputs." >&2
  echo "Set MASQUANT_RESUME or run calibration first." >&2
  exit 1
fi
if [[ ! -f "$MASQUANT_RESUME" ]]; then
  echo "MASQUANT_RESUME does not exist: $MASQUANT_RESUME" >&2
  exit 1
fi
log "Using MASQUANT_RESUME=$MASQUANT_RESUME"

if [[ "$USE_CMC" == "1" && "$RUN_CMC" == "1" ]]; then
  if [[ "$CMC_CALI_DATA_TYPE" != "no-white" ]]; then
    require_path CMC_VISION_JSON
    require_path CMC_VISION_PREFIX
  fi
  cmc_cmd=(
    "$PYTHON" -m prune_quant_baseline.scripts.run_prune_then_quant_masquant
    --stage cmc
    --model-type "$MODEL_TYPE"
    --model-path "$MODEL_PATH"
    --masquant-root "$MASQUANT_ROOT"
    --work-dir "$WORK_DIR"
    --masquant-resume "$MASQUANT_RESUME"
    --masquant-act-scales "$MASQUANT_ACT_SCALES"
    --wbits "$WBITS"
    --abits "$ABITS"
    --group-size "$GROUP_SIZE"
    --batch-size "$BATCH_SIZE"
    --attn-implementation "$ATTN_IMPLEMENTATION"
    --cmc-script-name "$CMC_SCRIPT_NAME"
    --cmc-net "$CMC_NET"
    --cmc-cali-data-type "$CMC_CALI_DATA_TYPE"
    --cmc-rank "$CMC_RANK"
    --cmc-quant-cmc "$CMC_QUANT_CMC"
    --cmc-n-cali-samples "$CMC_N_CALI_SAMPLES"
    --cmc-white-matrix-path "$CMC_WHITE"
    --cmc-low-rank-adapters "$CMC_LOW_RANK"
  )
  append_if_set cmc_cmd --cmc-vision-json "$CMC_VISION_JSON"
  append_if_set cmc_cmd --cmc-vision-prefix "$CMC_VISION_PREFIX"
  append_if_set cmc_cmd --cmc-audio-json "$CMC_AUDIO_JSON"
  append_if_set cmc_cmd --cmc-audio-prefix "$CMC_AUDIO_PREFIX"
  if [[ -n "$CMC_EXTRA_ARGS" ]]; then
    read -r -a cmc_extra_array <<< "$CMC_EXTRA_ARGS"
    cmc_cmd+=("${cmc_extra_array[@]}")
  fi
  run_cmd "${cmc_cmd[@]}"
elif [[ "$USE_CMC" == "1" ]]; then
  log "Skipping CMC because RUN_CMC=$RUN_CMC"
else
  log "CMC disabled because USE_CMC=$USE_CMC"
fi

if [[ "$RUN_INSTALL_VLMEVAL" == "1" ]]; then
  run_cmd "$PYTHON" "$PROJECT_ROOT/remote/install_vlmeval_pruned_gae.py" --vlmeval-root "$VLMEVALKIT_ROOT"
else
  log "Skipping VLMEvalKit wrapper install because RUN_INSTALL_VLMEVAL=$RUN_INSTALL_VLMEVAL"
fi

if [[ "$RUN_VLMEVAL" == "1" ]]; then
  if [[ "$USE_CMC" == "1" ]]; then
    if [[ ! -f "$CMC_LOW_RANK" ]]; then
      echo "CMC_LOW_RANK does not exist: $CMC_LOW_RANK" >&2
      exit 1
    fi
    if [[ ! -f "$CMC_WHITE" ]]; then
      echo "CMC_WHITE does not exist: $CMC_WHITE" >&2
      exit 1
    fi
  else
    CMC_LOW_RANK=""
    CMC_WHITE=""
  fi

  cd "$VLMEVALKIT_ROOT"
  export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
  export QWEN2VL_MODEL="$MODEL_PATH"
  export QWEN25VL_MODEL="$MODEL_PATH"
  export MASQUANT_ROOT
  export MASQUANT_RESUME
  export MASQUANT_ACT_SCALES
  export CMC_LOW_RANK
  export CMC_WHITE
  export PQ_MODEL_TYPE="$MODEL_TYPE"
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
  export PQ_ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION"
  export PQ_MIN_PIXELS="$PROCESSOR_MIN_PIXELS"
  export PQ_MAX_PIXELS="$PROCESSOR_MAX_PIXELS"
  export PQ_MIN_VISUAL_TOKENS="$PROCESSOR_MIN_VISUAL_TOKENS"
  export PQ_MAX_VISUAL_TOKENS="$PROCESSOR_MAX_VISUAL_TOKENS"

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
  if [[ "$VLMEVAL_SMART_RUNNER" == "1" ]]; then
    cd "$PROJECT_ROOT"
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
    append_if_set vlmeval_cmd --judge "$VLMEVAL_JUDGE"
    append_bool_flag vlmeval_cmd --verbose "$VLMEVAL_VERBOSE"
    append_bool_flag vlmeval_cmd --force-eval "$VLMEVAL_FORCE_EVAL"
    if [[ "$VLMEVAL_REUSE" != "1" ]]; then
      vlmeval_cmd+=(--no-reuse)
    fi
    if [[ "$VLMEVAL_REUSE_AUX" != "1" ]]; then
      vlmeval_cmd+=(--no-reuse-aux)
    fi
  else
    vlmeval_cmd=("$PYTHON" run.py --data "${vlmeval_datasets[@]}" --model "$VLMEVAL_MODEL_NAME" --work-dir "$VLMEVAL_WORK_DIR")
    append_bool_flag vlmeval_cmd --verbose "$VLMEVAL_VERBOSE"
  fi
  run_cmd "${vlmeval_cmd[@]}"
else
  log "Skipping VLMEvalKit evaluation because RUN_VLMEVAL=$RUN_VLMEVAL"
fi

log "Pipeline finished."
