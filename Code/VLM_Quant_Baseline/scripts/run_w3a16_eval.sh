#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-QIG}"

INFERENCE_DATA_ROOT="${INFERENCE_DATA_ROOT:-/root/autodl-tmp/dataset/inferecne}"

EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/eval/QIG/w3a16_speed}"
LOG_ROOT="${LOG_ROOT:-${EVAL_ROOT}/logs}"
INFER_PAIRS="${INFER_PAIRS:-${REPO_ROOT}/inference/question.json}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
REAL_CHECKPOINT="${REAL_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W3A16-autogptq-smoke}"
FP16_CHECKPOINT="${FP16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct}"
PSEUDO_MODEL_ARGS="${PSEUDO_MODEL_ARGS:-pretrained=${FP16_CHECKPOINT},use_flash_attention_2=False}"
SCALE_PATH="${SCALE_PATH:-/root/autodl-tmp/scale/QIG/qig/qwen2_vl_7b_w3a16.pt}"

DRY_RUN="${DRY_RUN:-0}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

run_cmd() {
  log "$*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

setup_env() {
  require_path "$CONDA_SH"
  require_path "$INFERENCE_DATA_ROOT"
  require_path "${FP16_CHECKPOINT}/config.json"
  require_path "${REAL_CHECKPOINT}/config.json"
  require_path "$INFER_PAIRS"

  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  cd "$REPO_ROOT"

  export HF_HOME="${HF_HOME:-${INFERENCE_DATA_ROOT}/hf-home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${INFERENCE_DATA_ROOT}/datasets}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${INFERENCE_DATA_ROOT}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${INFERENCE_DATA_ROOT}/hub}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${INFERENCE_DATA_ROOT}/xdg}"

  mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
}

run_timed() {
  local label="$1"
  local output_path="$2"
  local log_path="$3"
  shift 3

  mkdir -p "$(dirname "$output_path")" "$(dirname "$log_path")"
  log "Running ${label}"

  if [[ "$DRY_RUN" == "1" ]]; then
    run_cmd "$@"
  else
    local start end elapsed status
    start="$(date +%s)"
    set +e
    "$@" 2>&1 | tee "$log_path"
    status="${PIPESTATUS[0]}"
    set -e
    end="$(date +%s)"
    elapsed="$((end - start))"
    printf 'elapsed_sec=%s\n' "$elapsed" > "${output_path}.time"
    return "$status"
  fi
}

main() {
  setup_env

  run_timed \
    "pseudo QIG W3A16 HF inference" \
    "${EVAL_ROOT}/pseudo_qig_w3a16.json" \
    "${LOG_ROOT}/pseudo_qig_w3a16.log" \
    python -W ignore inference.py \
      --model qwen2_vl \
      --model_args "$PSEUDO_MODEL_ARGS" \
      --method qig \
      --pseudo_quant \
      --w_bit 3 \
      --a_bit 16 \
      --scale_path "$SCALE_PATH" \
      --infer_pairs "$INFER_PAIRS" \
      --save_path "${EVAL_ROOT}/pseudo_qig_w3a16.json" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE"

  run_timed \
    "real GPTQ W3A16 vLLM inference" \
    "${EVAL_ROOT}/real_gptq_w3a16.json" \
    "${LOG_ROOT}/real_gptq_w3a16.log" \
    python -W ignore inference.py \
      --model qwen2_vl \
      --model_args "pretrained=${FP16_CHECKPOINT}" \
      --inference_engine vllm \
      --vllm_model_path "$REAL_CHECKPOINT" \
      --vllm_quantization gptq \
      --vllm_dtype float16 \
      --vllm_trust_remote_code \
      --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
      --w_bit 3 \
      --a_bit 16 \
      --infer_pairs "$INFER_PAIRS" \
      --save_path "${EVAL_ROOT}/real_gptq_w3a16.json" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE"

  log "W3A16 speed comparison finished."
}

main "$@"
