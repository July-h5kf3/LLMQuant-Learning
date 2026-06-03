#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-QIG}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-/root/autodl-tmp/QIG/3rdparty/lmms-eval}"
INFERENCE_DATA_ROOT="${INFERENCE_DATA_ROOT:-/root/autodl-tmp/dataset/inferecne}"

FP16_CHECKPOINT="${FP16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct}"
REAL_CHECKPOINT="${REAL_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W3A16-autogptq-smoke}"
EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/eval/QIG/real_w3a16_vllm}"
LOG_ROOT="${LOG_ROOT:-${EVAL_ROOT}/logs}"

TASKS="${TASKS:-mmmu_val,vizwiz_vqa_val,chartqa,ai2d,scienceqa_img}"
LOG_SUFFIX="${LOG_SUFFIX:-${TASKS//,/_}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0,max_new_tokens=64}"

VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-gptq}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"

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
  require_path "$CONDA_SH"
  require_path "$LMMS_EVAL_ROOT"
  require_path "$INFERENCE_DATA_ROOT"
  require_path "${FP16_CHECKPOINT}/config.json"
  require_path "${REAL_CHECKPOINT}/config.json"

  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  cd "$REPO_ROOT"

  export PYTHONPATH="${LMMS_EVAL_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-${INFERENCE_DATA_ROOT}/hf-home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${INFERENCE_DATA_ROOT}/datasets}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${INFERENCE_DATA_ROOT}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${INFERENCE_DATA_ROOT}/hub}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${INFERENCE_DATA_ROOT}/xdg}"

  mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
}

main() {
  setup_env

  local cmd=(
    python -W ignore main.py
    --model qwen2_vl
    --tasks "$TASKS"
    --batch_size "$BATCH_SIZE"
    --log_samples
    --log_samples_suffix "$LOG_SUFFIX"
    --output_path "$EVAL_ROOT"
    --real_quant
    --inference_engine vllm
    --vllm_model_path "$REAL_CHECKPOINT"
    --vllm_processor_path "$FP16_CHECKPOINT"
    --vllm_quantization "$VLLM_QUANTIZATION"
    --vllm_dtype "$VLLM_DTYPE"
    --vllm_trust_remote_code
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS"
    --w_bit 3
    --a_bit 16
    --gen_kwargs "$GEN_KWARGS"
  )

  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  local start end elapsed status log_path
  log_path="${LOG_ROOT}/qwen2_vl_w3a16_real_vllm.log"
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
