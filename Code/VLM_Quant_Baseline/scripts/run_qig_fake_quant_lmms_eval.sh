#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-QIG}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-/root/autodl-tmp/QIG/3rdparty/lmms-eval}"
INFERENCE_DATA_ROOT="${INFERENCE_DATA_ROOT:-/root/autodl-tmp/dataset/inferecne}"
NETWORK_TURBO="${NETWORK_TURBO:-1}"

FP16_CHECKPOINT="${FP16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct}"
EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/eval/QIG/fake_quant}"
LOG_ROOT="${LOG_ROOT:-${EVAL_ROOT}/logs}"

MODEL="${MODEL:-qwen2_vl}"
TASKS="${TASKS:-mmmu_val,vizwiz_vqa_val,chartqa,ai2d,scienceqa_img}"
LOG_SUFFIX="${LOG_SUFFIX:-${TASKS//,/_}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0,max_new_tokens=64}"

W_BIT="${W_BIT:-4}"
A_BIT="${A_BIT:-8}"
PSEUDO_METHOD="${PSEUDO_METHOD:-rtn}"
SCALE_PATH="${SCALE_PATH:-}"
RUN_PROCESS="${RUN_PROCESS:-0}"
PSEUDO_MODEL_ARGS="${PSEUDO_MODEL_ARGS:-pretrained=${FP16_CHECKPOINT},use_flash_attention_2=False}"

DRY_RUN="${DRY_RUN:-0}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

run_cmd() {
  log "$*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

setup_env() {
  if [[ "$DRY_RUN" != "1" ]]; then
    require_path "$CONDA_SH"
    require_path "$LMMS_EVAL_ROOT"
    require_path "$INFERENCE_DATA_ROOT"
    require_path "${FP16_CHECKPOINT}/config.json"
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    if [[ "$NETWORK_TURBO" == "1" && -f /etc/network_turbo ]]; then
      # shellcheck disable=SC1091
      source /etc/network_turbo || true
    fi
  fi
  cd "$REPO_ROOT"

  export PYTHONPATH="${LMMS_EVAL_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-${INFERENCE_DATA_ROOT}/hf-home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${INFERENCE_DATA_ROOT}/datasets}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${INFERENCE_DATA_ROOT}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${INFERENCE_DATA_ROOT}/hub}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${INFERENCE_DATA_ROOT}/xdg}"

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
  fi
}

main() {
  setup_env

  local cmd=(
    python -W ignore main.py
    --model "$MODEL"
    --model_args "$PSEUDO_MODEL_ARGS"
    --tasks "$TASKS"
    --batch_size "$BATCH_SIZE"
    --log_samples
    --log_samples_suffix "$LOG_SUFFIX"
    --output_path "$EVAL_ROOT"
    --method "$PSEUDO_METHOD"
    --pseudo_quant
    --w_bit "$W_BIT"
    --a_bit "$A_BIT"
    --gen_kwargs "$GEN_KWARGS"
  )

  if [[ -n "$SCALE_PATH" ]]; then
    cmd+=(--scale_path "$SCALE_PATH")
  fi
  if [[ "$RUN_PROCESS" == "1" ]]; then
    cmd+=(--run_process)
  fi
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  local start end elapsed status log_path
  log_path="${LOG_ROOT}/${MODEL}_${PSEUDO_METHOD}_w${W_BIT}a${A_BIT}_fake_quant.log"
  start="$(date +%s)"
  if [[ "$DRY_RUN" == "1" ]]; then
    run_cmd "${cmd[@]}"
  else
    set +e
    "${cmd[@]}" 2>&1 | tee "$log_path"
    status="${PIPESTATUS[0]}"
    set -e
    end="$(date +%s)"
    elapsed="$((end - start))"
    {
      printf 'elapsed_sec=%s\n' "$elapsed"
      printf 'status=%s\n' "$status"
    } > "${log_path}.time"
    return "$status"
  fi
}

main "$@"
