#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-QIG_TRTLLM}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-/root/autodl-tmp/QIG/3rdparty/lmms-eval}"
INFERENCE_DATA_ROOT="${INFERENCE_DATA_ROOT:-/root/autodl-tmp/dataset/inferecne}"

FP16_CHECKPOINT="${FP16_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct}"
REAL_CHECKPOINT="${REAL_CHECKPOINT:-/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-trtllm}"
EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/eval/QIG/real_trtllm}"
LOG_ROOT="${LOG_ROOT:-${EVAL_ROOT}/logs}"

MODEL="${MODEL:-qwen2_vl}"
TASKS="${TASKS:-mmmu_val,vizwiz_vqa_val,chartqa,ai2d,scienceqa_img}"
LOG_SUFFIX="${LOG_SUFFIX:-${TASKS//,/_}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0,max_new_tokens=64}"

W_BIT="${W_BIT:-4}"
A_BIT="${A_BIT:-16}"
TRTLLM_BACKEND="${TRTLLM_BACKEND:-engine}"
TRTLLM_DTYPE="${TRTLLM_DTYPE:-auto}"
TRTLLM_TP_SIZE="${TRTLLM_TP_SIZE:-1}"
TRTLLM_PP_SIZE="${TRTLLM_PP_SIZE:-1}"
TRTLLM_MAX_BATCH_SIZE="${TRTLLM_MAX_BATCH_SIZE:-8}"
TRTLLM_MAX_NUM_TOKENS="${TRTLLM_MAX_NUM_TOKENS:-8192}"
TRTLLM_MAX_MULTIMODAL_LEN="${TRTLLM_MAX_MULTIMODAL_LEN:-1296}"
TRTLLM_KV_CACHE_FRACTION="${TRTLLM_KV_CACHE_FRACTION:-0.9}"
TRTLLM_MODEL_TYPE="${TRTLLM_MODEL_TYPE:-}"
TRTLLM_ENGINE_DIR="${TRTLLM_ENGINE_DIR:-}"
TRTLLM_WORKSPACE="${TRTLLM_WORKSPACE:-}"
TRTLLM_ENABLE_BUILD_CACHE="${TRTLLM_ENABLE_BUILD_CACHE:-0}"
TRTLLM_FAST_BUILD="${TRTLLM_FAST_BUILD:-0}"

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
    require_path "${REAL_CHECKPOINT}/config.json"
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
  fi
  cd "$REPO_ROOT"

  export PYTHONPATH="${LMMS_EVAL_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-${INFERENCE_DATA_ROOT}/hf-home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${INFERENCE_DATA_ROOT}/datasets}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${INFERENCE_DATA_ROOT}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${INFERENCE_DATA_ROOT}/hub}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${INFERENCE_DATA_ROOT}/xdg}"
  export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
  local conda_prefix="${CONDA_PREFIX:-}"
  if [[ -n "$conda_prefix" && -d "${conda_prefix}/lib/python3.12/site-packages/nvidia/cu13/lib" ]]; then
    export LD_LIBRARY_PATH="${conda_prefix}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
  fi
}

main() {
  setup_env

  local cmd=(
    python -W ignore main.py
    --model "$MODEL"
    --tasks "$TASKS"
    --batch_size "$BATCH_SIZE"
    --log_samples
    --log_samples_suffix "$LOG_SUFFIX"
    --output_path "$EVAL_ROOT"
    --real_quant
    --inference_engine trtllm
    --trtllm_model_path "$REAL_CHECKPOINT"
    --trtllm_tokenizer_path "$FP16_CHECKPOINT"
    --trtllm_backend "$TRTLLM_BACKEND"
    --trtllm_dtype "$TRTLLM_DTYPE"
    --trtllm_tensor_parallel_size "$TRTLLM_TP_SIZE"
    --trtllm_pipeline_parallel_size "$TRTLLM_PP_SIZE"
    --trtllm_max_batch_size "$TRTLLM_MAX_BATCH_SIZE"
    --trtllm_max_num_tokens "$TRTLLM_MAX_NUM_TOKENS"
    --trtllm_max_multimodal_len "$TRTLLM_MAX_MULTIMODAL_LEN"
    --trtllm_kv_cache_free_gpu_memory_fraction "$TRTLLM_KV_CACHE_FRACTION"
    --trtllm_trust_remote_code
    --w_bit "$W_BIT"
    --a_bit "$A_BIT"
    --gen_kwargs "$GEN_KWARGS"
  )

  if [[ -n "$TRTLLM_MODEL_TYPE" ]]; then
    cmd+=(--trtllm_model_type "$TRTLLM_MODEL_TYPE")
  fi
  if [[ -n "$TRTLLM_ENGINE_DIR" ]]; then
    cmd+=(--trtllm_engine_dir "$TRTLLM_ENGINE_DIR")
  fi
  if [[ -n "$TRTLLM_WORKSPACE" ]]; then
    cmd+=(--trtllm_workspace "$TRTLLM_WORKSPACE")
  fi
  if [[ "$TRTLLM_ENABLE_BUILD_CACHE" == "1" ]]; then
    cmd+=(--trtllm_enable_build_cache)
  fi
  if [[ "$TRTLLM_FAST_BUILD" == "1" ]]; then
    cmd+=(--trtllm_fast_build)
  fi
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  local start end elapsed status log_path
  log_path="${LOG_ROOT}/${MODEL}_w${W_BIT}a${A_BIT}_real_trtllm.log"
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
