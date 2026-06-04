#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
SUITE_ROOT="${SUITE_ROOT:-/root/autodl-tmp/eval/QIG/real_quant_lmms_eval_${RUN_ID}}"
TASKS="${TASKS:-mmmu_val,vizwiz_vqa_val,chartqa,ai2d,scienceqa_img}"
W3A16_BATCH_SIZE="${W3A16_BATCH_SIZE:-${BATCH_SIZE:-1}}"
W3A16_VLLM_MAX_NUM_SEQS="${W3A16_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-$W3A16_BATCH_SIZE}}"
TRTLLM_BATCH_SIZE="${TRTLLM_BATCH_SIZE:-1}"
TRTLLM_CONCURRENCY="${TRTLLM_CONCURRENCY:-8}"
TRTLLM_MAX_BATCH_SIZE="${TRTLLM_MAX_BATCH_SIZE:-$TRTLLM_CONCURRENCY}"
LIMIT="${LIMIT:-}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0,max_new_tokens=64}"

FP16_CHECKPOINT="${FP16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct}"
W3A16_CHECKPOINT="${W3A16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W3A16-autogptq/Qwen2-VL-7B-Instruct-w3g128}"
W4A16_CHECKPOINT="${W4A16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-trtllm}"
W4A8_CHECKPOINT="${W4A8_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A8-trtllm}"
W4A16_ENGINE_DIR="${W4A16_ENGINE_DIR:-${TRTLLM_ENGINE_DIR:-}}"
W4A8_ENGINE_DIR="${W4A8_ENGINE_DIR:-${TRTLLM_ENGINE_DIR:-}}"
W4A16_FAKE_METHOD="${W4A16_FAKE_METHOD:-rtn}"
W4A16_FAKE_SCALE_PATH="${W4A16_FAKE_SCALE_PATH:-}"
W4A16_FAKE_RUN_PROCESS="${W4A16_FAKE_RUN_PROCESS:-0}"
W4A8_FAKE_METHOD="${W4A8_FAKE_METHOD:-rtn}"
W4A8_FAKE_SCALE_PATH="${W4A8_FAKE_SCALE_PATH:-}"
W4A8_FAKE_RUN_PROCESS="${W4A8_FAKE_RUN_PROCESS:-0}"
TRTLLM_CONDA_ENV="${TRTLLM_CONDA_ENV:-/root/autodl-tmp/envs/QIG_TRTLLM}"

QIG_CONDA_ENV="${QIG_CONDA_ENV:-QIG}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${QIG_CONDA_ENV}/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

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
  local batch_size="$2"
  shift 2
  local out_dir="${SUITE_ROOT}/${variant}"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$out_dir"
  fi
  log "Running ${variant}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN command: EVAL_ROOT=${out_dir} TASKS=${TASKS} BATCH_SIZE=${batch_size} GEN_KWARGS=${GEN_KWARGS} LIMIT=${LIMIT} $*"
    return 0
  fi
  EVAL_ROOT="$out_dir" TASKS="$TASKS" BATCH_SIZE="$batch_size" GEN_KWARGS="$GEN_KWARGS" LIMIT="$LIMIT" "$@"
}

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$SUITE_ROOT"
fi
cd "$REPO_ROOT"

if [[ "$DRY_RUN" != "1" ]]; then
  require_checkpoint "FP16 processor" "$FP16_CHECKPOINT"
  require_checkpoint "W3A16 vLLM" "$W3A16_CHECKPOINT"
  require_checkpoint "W4A16 TensorRT-LLM" "$W4A16_CHECKPOINT"
  require_checkpoint "W4A8 TensorRT-LLM" "$W4A8_CHECKPOINT"
  if [[ -z "$W4A16_ENGINE_DIR" || ! -f "${W4A16_ENGINE_DIR}/llm/config.json" ]]; then
    echo "Missing W4A16 TensorRT-LLM engine dir. Set W4A16_ENGINE_DIR=/path/to/qwen2vl-engine." >&2
    exit 1
  fi
  if [[ -z "$W4A8_ENGINE_DIR" || ! -f "${W4A8_ENGINE_DIR}/llm/config.json" ]]; then
    echo "Missing W4A8 TensorRT-LLM engine dir. Set W4A8_ENGINE_DIR=/path/to/qwen2vl-engine." >&2
    exit 1
  fi
fi

run_variant \
  w3a16_vllm \
  "$W3A16_BATCH_SIZE" \
  env \
    CONDA_ENV="$QIG_CONDA_ENV" \
    FP16_CHECKPOINT="$FP16_CHECKPOINT" \
    REAL_CHECKPOINT="$W3A16_CHECKPOINT" \
    VLLM_QUANTIZATION="${W3A16_VLLM_QUANTIZATION:-gptq}" \
    VLLM_MAX_NUM_SEQS="$W3A16_VLLM_MAX_NUM_SEQS" \
    bash scripts/run_qig_w3a16_real_vllm_eval.sh

run_variant \
  w4a16_trtllm \
  "$TRTLLM_BATCH_SIZE" \
  env \
    CONDA_ENV="$TRTLLM_CONDA_ENV" \
    FP16_CHECKPOINT="$FP16_CHECKPOINT" \
    REAL_CHECKPOINT="$W4A16_CHECKPOINT" \
    TRTLLM_ENGINE_DIR="$W4A16_ENGINE_DIR" \
    TRTLLM_CONCURRENCY="$TRTLLM_CONCURRENCY" \
    TRTLLM_MAX_BATCH_SIZE="$TRTLLM_MAX_BATCH_SIZE" \
    W_BIT=4 \
    A_BIT=16 \
    bash scripts/run_qig_real_trtllm_eval.sh

run_variant \
  w4a8_trtllm \
  "$TRTLLM_BATCH_SIZE" \
  env \
    CONDA_ENV="$TRTLLM_CONDA_ENV" \
    FP16_CHECKPOINT="$FP16_CHECKPOINT" \
    REAL_CHECKPOINT="$W4A8_CHECKPOINT" \
    TRTLLM_ENGINE_DIR="$W4A8_ENGINE_DIR" \
    TRTLLM_CONCURRENCY="$TRTLLM_CONCURRENCY" \
    TRTLLM_MAX_BATCH_SIZE="$TRTLLM_MAX_BATCH_SIZE" \
    W_BIT=4 \
    A_BIT=8 \
    bash scripts/run_qig_real_trtllm_eval.sh

run_variant \
  w4a16_fake_quant \
  1 \
  env \
    CONDA_ENV="$QIG_CONDA_ENV" \
    FP16_CHECKPOINT="$FP16_CHECKPOINT" \
    PSEUDO_METHOD="$W4A16_FAKE_METHOD" \
    SCALE_PATH="$W4A16_FAKE_SCALE_PATH" \
    RUN_PROCESS="$W4A16_FAKE_RUN_PROCESS" \
    W_BIT=4 \
    A_BIT=16 \
    bash scripts/run_qig_fake_quant_lmms_eval.sh

run_variant \
  w4a8_fake_quant \
  1 \
  env \
    CONDA_ENV="$QIG_CONDA_ENV" \
    FP16_CHECKPOINT="$FP16_CHECKPOINT" \
    PSEUDO_METHOD="$W4A8_FAKE_METHOD" \
    SCALE_PATH="$W4A8_FAKE_SCALE_PATH" \
    RUN_PROCESS="$W4A8_FAKE_RUN_PROCESS" \
    W_BIT=4 \
    A_BIT=8 \
    bash scripts/run_qig_fake_quant_lmms_eval.sh

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" scripts/summarize_lmms_eval_speed.py \
    --run_root "$SUITE_ROOT" \
    --output_csv "${SUITE_ROOT}/summary.csv"

  log "Summary: ${SUITE_ROOT}/summary.csv"
fi
