#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/root/autodl-tmp/envs/QIG_TRTLLM}"
NETWORK_TURBO="${NETWORK_TURBO:-1}"

FP16_CHECKPOINT="${FP16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct}"
REAL_CHECKPOINT="${REAL_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-trtllm}"
TRTLLM_ENGINE_DIR="${TRTLLM_ENGINE_DIR:-/root/autodl-fs/trtllm_workspace/Qwen2-VL-7B-Instruct-W4A16-engine}"
TRTLLM_TIMING_CACHE="${TRTLLM_TIMING_CACHE:-${TRTLLM_ENGINE_DIR}/model.cache}"

TRTLLM_MAX_BATCH_SIZE="${TRTLLM_MAX_BATCH_SIZE:-8}"
TRTLLM_MAX_INPUT_LEN="${TRTLLM_MAX_INPUT_LEN:-2048}"
TRTLLM_MAX_SEQ_LEN="${TRTLLM_MAX_SEQ_LEN:-2112}"
TRTLLM_MAX_NUM_TOKENS="${TRTLLM_MAX_NUM_TOKENS:-8192}"
TRTLLM_MAX_MULTIMODAL_LEN="${TRTLLM_MAX_MULTIMODAL_LEN:-$((TRTLLM_MAX_BATCH_SIZE * 324))}"
TRTLLM_OPT_NUM_TOKENS="${TRTLLM_OPT_NUM_TOKENS:-}"
TRTLLM_FAST_BUILD="${TRTLLM_FAST_BUILD:-0}"
TRTLLM_WORKERS="${TRTLLM_WORKERS:-1}"

VISION_MIN_HW_DIMS="${VISION_MIN_HW_DIMS:-128}"
VISION_MAX_HW_DIMS="${VISION_MAX_HW_DIMS:-5184}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_VISION_BUILD="${SKIP_VISION_BUILD:-0}"
SKIP_LLM_BUILD="${SKIP_LLM_BUILD:-0}"

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
    require_path "${FP16_CHECKPOINT}/config.json"
    require_path "${REAL_CHECKPOINT}/config.json"
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    if [[ "$NETWORK_TURBO" == "1" && -f /etc/network_turbo ]]; then
      # shellcheck disable=SC1091
      source /etc/network_turbo || true
    fi
    mkdir -p "${TRTLLM_ENGINE_DIR}"
  fi

  export TRT_LLM_NO_LIB_INIT="${TRT_LLM_NO_LIB_INIT:-1}"
  export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.9}"
  export TLLM_WORKER_USE_SINGLE_PROCESS="${TLLM_WORKER_USE_SINGLE_PROCESS:-1}"
  export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"

  local conda_prefix="${CONDA_PREFIX:-}"
  if [[ -n "$conda_prefix" && -d "${conda_prefix}/lib/python3.12/site-packages/nvidia/cu13/lib" ]]; then
    export LD_LIBRARY_PATH="${conda_prefix}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -n "$conda_prefix" && -d "${conda_prefix}/lib/python3.12/site-packages/tensorrt_llm/libs" ]]; then
    export LD_LIBRARY_PATH="${conda_prefix}/lib/python3.12/site-packages/tensorrt_llm/libs:${LD_LIBRARY_PATH:-}"
  fi
}

build_vision() {
  if [[ "$SKIP_VISION_BUILD" == "1" ]]; then
    log "Skipping vision engine build"
    return 0
  fi
  run_cmd python -m tensorrt_llm.tools.multimodal_builder \
    --model_type qwen2_vl \
    --model_path "$FP16_CHECKPOINT" \
    --output_dir "${TRTLLM_ENGINE_DIR}/vision" \
    --max_batch_size "$TRTLLM_MAX_BATCH_SIZE" \
    --min_hw_dims "$VISION_MIN_HW_DIMS" \
    --max_hw_dims "$VISION_MAX_HW_DIMS"
}

build_llm() {
  if [[ "$SKIP_LLM_BUILD" == "1" ]]; then
    log "Skipping LLM engine build"
    return 0
  fi

  local cmd=(
    trtllm-build
    --checkpoint_dir "$REAL_CHECKPOINT"
    --output_dir "${TRTLLM_ENGINE_DIR}/llm"
    --max_batch_size "$TRTLLM_MAX_BATCH_SIZE"
    --max_input_len "$TRTLLM_MAX_INPUT_LEN"
    --max_seq_len "$TRTLLM_MAX_SEQ_LEN"
    --max_num_tokens "$TRTLLM_MAX_NUM_TOKENS"
    --max_prompt_embedding_table_size "$TRTLLM_MAX_MULTIMODAL_LEN"
    --input_timing_cache "$TRTLLM_TIMING_CACHE"
    --output_timing_cache "$TRTLLM_TIMING_CACHE"
    --workers "$TRTLLM_WORKERS"
  )
  if [[ -n "$TRTLLM_OPT_NUM_TOKENS" ]]; then
    cmd+=(--opt_num_tokens "$TRTLLM_OPT_NUM_TOKENS")
  fi
  if [[ "$TRTLLM_FAST_BUILD" == "1" ]]; then
    cmd+=(--fast_build)
  fi
  run_cmd "${cmd[@]}"
}

main() {
  setup_env
  build_vision
  build_llm
  log "TensorRT-LLM Qwen2-VL engines ready at ${TRTLLM_ENGINE_DIR}"
}

main "$@"
