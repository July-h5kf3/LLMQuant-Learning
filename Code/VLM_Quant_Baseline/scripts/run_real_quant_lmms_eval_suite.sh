#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
SUITE_ROOT="${SUITE_ROOT:-/root/autodl-tmp/eval/QIG/real_quant_lmms_eval_${RUN_ID}}"
TASKS="${TASKS:-mmmu_val,vizwiz_vqa_val,chartqa,ai2d,scienceqa_img}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0,max_new_tokens=64}"

FP16_CHECKPOINT="${FP16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct}"
W3A16_CHECKPOINT="${W3A16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W3A16-autogptq}"
W4A16_CHECKPOINT="${W4A16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-vllm}"
W4A8_CHECKPOINT="${W4A8_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A8-vllm-smoke}"

QIG_CONDA_ENV="${QIG_CONDA_ENV:-QIG}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

require_checkpoint() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}/config.json" ]]; then
    echo "Missing ${label} checkpoint config: ${path}/config.json" >&2
    exit 1
  fi
}

run_variant() {
  local variant="$1"
  shift
  local out_dir="${SUITE_ROOT}/${variant}"
  mkdir -p "$out_dir"
  log "Running ${variant}"
  EVAL_ROOT="$out_dir" TASKS="$TASKS" BATCH_SIZE="$BATCH_SIZE" GEN_KWARGS="$GEN_KWARGS" LIMIT="$LIMIT" "$@"
}

mkdir -p "$SUITE_ROOT"
cd "$REPO_ROOT"

require_checkpoint "FP16 processor" "$FP16_CHECKPOINT"
require_checkpoint "W3A16 vLLM" "$W3A16_CHECKPOINT"
require_checkpoint "W4A16 vLLM" "$W4A16_CHECKPOINT"
require_checkpoint "W4A8 vLLM" "$W4A8_CHECKPOINT"

run_variant \
  w3a16_vllm \
  env \
    CONDA_ENV="$QIG_CONDA_ENV" \
    FP16_CHECKPOINT="$FP16_CHECKPOINT" \
    REAL_CHECKPOINT="$W3A16_CHECKPOINT" \
    VLLM_QUANTIZATION="${W3A16_VLLM_QUANTIZATION:-gptq}" \
    bash scripts/run_qig_w3a16_real_vllm_eval.sh

run_variant \
  w4a16_vllm \
  env \
    CONDA_ENV="$QIG_CONDA_ENV" \
    FP16_CHECKPOINT="$FP16_CHECKPOINT" \
    REAL_CHECKPOINT="$W4A16_CHECKPOINT" \
    W_BIT=4 \
    A_BIT=16 \
    VLLM_QUANTIZATION="${W4A16_VLLM_QUANTIZATION:-compressed-tensors}" \
    bash scripts/run_qig_real_vllm_eval.sh

run_variant \
  w4a8_vllm \
  env \
    CONDA_ENV="$QIG_CONDA_ENV" \
    FP16_CHECKPOINT="$FP16_CHECKPOINT" \
    REAL_CHECKPOINT="$W4A8_CHECKPOINT" \
    W_BIT=4 \
    A_BIT=8 \
    VLLM_QUANTIZATION="${W4A8_VLLM_QUANTIZATION:-compressed-tensors}" \
    bash scripts/run_qig_real_vllm_eval.sh

python scripts/summarize_lmms_eval_speed.py \
  --run_root "$SUITE_ROOT" \
  --output_csv "${SUITE_ROOT}/summary.csv"

log "Summary: ${SUITE_ROOT}/summary.csv"
