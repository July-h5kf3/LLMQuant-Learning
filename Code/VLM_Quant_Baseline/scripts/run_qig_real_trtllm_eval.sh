#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-QIG_TRTLLM}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-/root/autodl-tmp/QIG/3rdparty/lmms-eval}"
INFERENCE_DATA_ROOT="${INFERENCE_DATA_ROOT:-/root/autodl-tmp/dataset/inferecne}"
NETWORK_TURBO="${NETWORK_TURBO:-1}"

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
TRTLLM_CONTEXT_CHUNKING_POLICY="${TRTLLM_CONTEXT_CHUNKING_POLICY:-}"
TRTLLM_USE_SINGLE_RANK_MPI_STUB="${TRTLLM_USE_SINGLE_RANK_MPI_STUB:-1}"
TRTLLM_MPI_STUB_DIR="${TRTLLM_MPI_STUB_DIR:-/tmp/mpi_stub}"

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

setup_single_rank_mpi_stub() {
  local stub_dir="$1"
  local src_path="${stub_dir}/mpi_stub.c"
  local lib_path="${stub_dir}/libmpi.so.40"
  local cc_bin="${CC:-cc}"

  mkdir -p "$stub_dir"
  if [[ ! -f "$lib_path" ]]; then
    if ! command -v "$cc_bin" >/dev/null 2>&1; then
      echo "Cannot build TensorRT-LLM single-rank MPI stub: compiler not found: $cc_bin" >&2
      exit 1
    fi
    cat > "$src_path" <<'EOF'
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef void* MPI_Comm;
typedef void* MPI_Group;
typedef void* MPI_Info;
typedef void* MPI_Datatype;
typedef void* MPI_Op;
typedef void* MPI_Request;
typedef void* MPI_Message;
typedef int MPI_Fint;

char ompi_mpi_comm_world[8];
char ompi_mpi_comm_self[8];
char ompi_mpi_comm_null[8];
char ompi_mpi_group_null[8];
char ompi_mpi_info_null[8];
char ompi_mpi_op_null[8];
char ompi_mpi_op_sum[8];
char ompi_mpi_op_prod[8];
char ompi_mpi_op_max[8];
char ompi_mpi_op_min[8];
char ompi_mpi_op_maxloc[8];
char ompi_mpi_op_minloc[8];
char ompi_mpi_op_band[8];
char ompi_mpi_op_bor[8];
char ompi_mpi_op_bxor[8];
char ompi_mpi_op_land[8];
char ompi_mpi_op_lor[8];
char ompi_mpi_op_lxor[8];
char ompi_mpi_op_replace[8];
char ompi_mpi_byte[8];
char ompi_mpi_char[8];
char ompi_mpi_c_bool[8];
char ompi_mpi_int[8];
char ompi_mpi_float[8];
char ompi_mpi_double[8];
char ompi_mpi_int8_t[8];
char ompi_mpi_uint8_t[8];
char ompi_mpi_int32_t[8];
char ompi_mpi_uint16_t[8];
char ompi_mpi_uint32_t[8];
char ompi_mpi_int64_t[8];
char ompi_mpi_uint64_t[8];

static int g_initialized = 0;
static char g_request[8];

static int dtype_size(MPI_Datatype datatype) {
    if (datatype == (void*)ompi_mpi_byte || datatype == (void*)ompi_mpi_char ||
        datatype == (void*)ompi_mpi_int8_t || datatype == (void*)ompi_mpi_uint8_t ||
        datatype == (void*)ompi_mpi_c_bool) return 1;
    if (datatype == (void*)ompi_mpi_uint16_t) return 2;
    if (datatype == (void*)ompi_mpi_int || datatype == (void*)ompi_mpi_int32_t ||
        datatype == (void*)ompi_mpi_uint32_t || datatype == (void*)ompi_mpi_float) return 4;
    if (datatype == (void*)ompi_mpi_int64_t || datatype == (void*)ompi_mpi_uint64_t ||
        datatype == (void*)ompi_mpi_double) return 8;
    return 1;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) { g_initialized = 1; if (provided) *provided = required; return 0; }
int MPI_Finalize(void) { g_initialized = 0; return 0; }
int MPI_Initialized(int *flag) { if (flag) *flag = g_initialized; return 0; }
int MPI_Abort(MPI_Comm comm, int errorcode) { return errorcode ? errorcode : 1; }
int MPI_Comm_rank(MPI_Comm comm, int *rank) { if (rank) *rank = 0; return 0; }
int MPI_Comm_size(MPI_Comm comm, int *size) { if (size) *size = 1; return 0; }
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) { if (newcomm) *newcomm = comm ? comm : (void*)ompi_mpi_comm_world; return 0; }
int MPI_Comm_free(MPI_Comm *comm) { if (comm) *comm = (void*)ompi_mpi_comm_null; return 0; }
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group) { if (group) *group = (void*)ompi_mpi_group_null; return 0; }
int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm) { if (newcomm) *newcomm = comm ? comm : (void*)ompi_mpi_comm_world; return 0; }
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) { if (newcomm) *newcomm = comm ? comm : (void*)ompi_mpi_comm_world; return 0; }
int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm) { if (newcomm) *newcomm = comm ? comm : (void*)ompi_mpi_comm_world; return 0; }
MPI_Comm MPI_Comm_f2c(MPI_Fint comm) { return (void*)ompi_mpi_comm_world; }
int MPI_Group_free(MPI_Group *group) { if (group) *group = (void*)ompi_mpi_group_null; return 0; }
int MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup) { if (newgroup) *newgroup = group ? group : (void*)ompi_mpi_group_null; return 0; }
int MPI_Group_rank(MPI_Group group, int *rank) { if (rank) *rank = 0; return 0; }
int MPI_Group_size(MPI_Group group, int *size) { if (size) *size = 1; return 0; }
int MPI_Group_translate_ranks(MPI_Group g1, int n, const int ranks1[], MPI_Group g2, int ranks2[]) { for (int i=0;i<n;i++) ranks2[i]=ranks1? ranks1[i] : 0; return 0; }
int MPI_Barrier(MPI_Comm comm) { return 0; }
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) { return 0; }
int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request *request) { if (request) *request = (void*)g_request; return 0; }
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) { if (sendbuf && recvbuf && sendbuf != recvbuf) memcpy(recvbuf, sendbuf, (size_t)sendcount * dtype_size(sendtype)); return 0; }
int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm) { if (sendbuf && recvbuf && sendbuf != recvbuf) memcpy((char*)recvbuf + (displs? displs[0] : 0) * dtype_size(recvtype), sendbuf, (size_t)sendcount * dtype_size(sendtype)); return 0; }
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) { if (sendbuf && recvbuf && sendbuf != recvbuf) memcpy(recvbuf, sendbuf, (size_t)count * dtype_size(datatype)); return 0; }
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) { return 0; }
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) { if (request) *request = (void*)g_request; return 0; }
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, void *status) { return 0; }
int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, void *status) { if (flag) *flag = 0; return 0; }
int MPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, void *status) { if (flag) *flag = 0; if (message) *message = NULL; return 0; }
int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, void *status) { if (message) *message = NULL; return 0; }
int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, void *status) { return 0; }
int MPI_Wait(MPI_Request *request, void *status) { if (request) *request = NULL; return 0; }
int MPI_Get_count(const void *status, MPI_Datatype datatype, int *count) { if (count) *count = 0; return 0; }
int MPI_Type_size(MPI_Datatype datatype, int *size) { if (size) *size = dtype_size(datatype); return 0; }
int MPI_Info_create(MPI_Info *info) { if (info) *info = (void*)ompi_mpi_info_null; return 0; }
int MPI_Info_set(MPI_Info info, const char *key, const char *value) { return 0; }
int MPI_Comm_spawn(const char *command, char *argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]) { if (intercomm) *intercomm = comm ? comm : (void*)ompi_mpi_comm_world; if (array_of_errcodes) for (int i=0;i<maxprocs;i++) array_of_errcodes[i]=0; return 0; }
EOF
    "$cc_bin" -shared -fPIC -O2 "$src_path" -Wl,-soname,libmpi.so.40 -o "$lib_path"
    ln -sf libmpi.so.40 "${stub_dir}/libmpi.so"
  fi
  export LD_LIBRARY_PATH="${stub_dir}:${LD_LIBRARY_PATH:-}"
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
  export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
  export TRT_LLM_NO_LIB_INIT="${TRT_LLM_NO_LIB_INIT:-1}"
  export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"
  export TLLM_WORKER_USE_SINGLE_PROCESS="${TLLM_WORKER_USE_SINGLE_PROCESS:-1}"
  local conda_prefix="${CONDA_PREFIX:-}"
  if [[ -n "$conda_prefix" && -d "${conda_prefix}/lib/python3.12/site-packages/nvidia/cu13/lib" ]]; then
    export LD_LIBRARY_PATH="${conda_prefix}/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -n "$conda_prefix" && -d "${conda_prefix}/lib/python3.12/site-packages/tensorrt_llm/libs" ]]; then
    export LD_LIBRARY_PATH="${conda_prefix}/lib/python3.12/site-packages/tensorrt_llm/libs:${LD_LIBRARY_PATH:-}"
  fi
  if [[ "$TRTLLM_USE_SINGLE_RANK_MPI_STUB" == "1" ]]; then
    if [[ "$TRTLLM_TP_SIZE" != "1" || "$TRTLLM_PP_SIZE" != "1" ]]; then
      echo "TRTLLM_USE_SINGLE_RANK_MPI_STUB=1 is only valid for TP=1 and PP=1." >&2
      exit 1
    fi
    if [[ "$DRY_RUN" != "1" ]]; then
      setup_single_rank_mpi_stub "$TRTLLM_MPI_STUB_DIR"
    else
      export LD_LIBRARY_PATH="${TRTLLM_MPI_STUB_DIR}:${LD_LIBRARY_PATH:-}"
    fi
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
    if [[ -n "$TRTLLM_WORKSPACE" ]]; then
      mkdir -p "$TRTLLM_WORKSPACE"
    fi
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
  if [[ -n "$TRTLLM_CONTEXT_CHUNKING_POLICY" ]]; then
    cmd+=(--trtllm_scheduler_context_chunking_policy "$TRTLLM_CONTEXT_CHUNKING_POLICY")
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
