#!/usr/bin/env bash
set -euo pipefail

# Qwen2-VL + quantization-aware GAE pruning:
#   D_i = lambda * C_i^quant - C_i^drop
# where C_i^drop uses rollout(grad_Attention * Attention_RTN) and
# C_i^quant uses input-token activation quantization difficulty.

# ---- Paths ----
export PROJECT_ROOT=/home/aistudio/LLMQuant-Learning/Code/Prune_Quant
export EXT_ROOT=/home/aistudio/EXT
export MODEL_PATH=/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct
export MASQUANT_ROOT="$EXT_ROOT/EfficientAI/masquant"
export VLMEVALKIT_ROOT="$EXT_ROOT/VLMEvalKit"
export WORK_DIR=/home/aistudio/datasets/output/qwen2vl_quant_joint_rtn

# Calibration data used by MASQuant phase 1.
export CALIB_JSONL=/home/aistudio/datasets/data/calibration/qig_coco_train2017_caption.jsonl

# CMC calibration data. CMC reads the original ShareGPT4V-style JSON and image directory.
export CMC_VISION_JSON=/home/aistudio/datasets/data/calibration/qig_coco_train2017_caption.json
export CMC_VISION_PREFIX=/home/aistudio/datasets/data/calibration/qig_coco_train2017_images

# ---- Model and quantization ----
export MODEL_TYPE=qwen2vl
export DATASET_TYPE=text-vision
export NSAMPLES=128
export BATCH_SIZE=1
export WBITS=4
export ABITS=8
export GROUP_SIZE=0
export EPOCHS=2
export ATTN_IMPLEMENTATION=eager
export DTYPE=bfloat16
export DEVICE_MAP=auto
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PROCESSOR_USE_FAST=
export PROCESSOR_MIN_PIXELS=
export PROCESSOR_MAX_PIXELS=
export PROCESSOR_MIN_VISUAL_TOKENS=1500
export PROCESSOR_MAX_VISUAL_TOKENS=1500

# ---- Joint pruning behavior ----
export PRUNER=gae_quant_joint
export CALIB_RETENTION_RATIO=0.5
export EVAL_RETENTION_RATIO="${EVAL_RETENTION_RATIO:-0.5}"
export MIN_KEEP=1
export GAE_ANSWER_SOURCE=generated
export GAE_PER_TOKEN=false
export GAE_QUANT_LAMBDA="${GAE_QUANT_LAMBDA:-1.0}"
export GAE_QUANT_METHOD=rtn
export RTN_BITS="${RTN_BITS:-4}"
export RTN_GROUP_SIZE="${RTN_GROUP_SIZE:-0}"
export GAE_DISABLE_MASQUANT_FAKE_QUANT="${GAE_DISABLE_MASQUANT_FAKE_QUANT:-1}"
export ALLOW_VANILLA_FALLBACK="${ALLOW_VANILLA_FALLBACK:-0}"

# Qwen2-VL path builds multimodal masks in the MASQuant bridge.
export PATCH_MASQUANT_INPUTS_EMBEDS_MASK=0

# ---- CMC ----
export USE_CMC=1
export CMC_NET=qwen2-vl-7b
export CMC_CALI_DATA_TYPE=vision-audio-only
export CMC_RANK=0.2
export CMC_QUANT_CMC=0
export CMC_N_CALI_SAMPLES=128

# ---- VLMEvalKit ----
export VLMEVAL_DATASETS="MME MMStar"
export VLMEVAL_MODEL_NAME=Qwen2VL_MASQuant_Pseudo
export VLMEVAL_WORK_DIR="$WORK_DIR/vlmeval_mme_mmstar_quant_joint_rtn"
export MAX_NEW_TOKENS=16
export VLMEVAL_VERBOSE=1
export VLMEVAL_SMART_RUNNER=1
export VLMEVAL_MODE=auto
export VLMEVAL_REUSE=1
export VLMEVAL_REUSE_AUX=1
export VLMEVAL_EXACT_MATCH_DATASETS="MME MMStar"
export VLMEVAL_DISABLE_OPENAI=1

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

echo "[quant-joint-rtn] RUN_CALIBRATE=${RUN_CALIBRATE:-1} MASQUANT_RESUME=${MASQUANT_RESUME:-<auto-after-calibrate>}"
echo "[quant-joint-rtn] RUN_CMC=${RUN_CMC:-1} CMC_LOW_RANK=$CMC_LOW_RANK CMC_WHITE=$CMC_WHITE"
echo "[quant-joint-rtn] lambda=$GAE_QUANT_LAMBDA rtn_bits=$RTN_BITS retention=$EVAL_RETENTION_RATIO"

exec bash "$PROJECT_ROOT/remote/run_masquant_pseudo_pipeline.sh"
