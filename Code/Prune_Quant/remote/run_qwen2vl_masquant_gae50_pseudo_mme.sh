#!/usr/bin/env bash
set -euo pipefail

# Qwen2-VL + GAE 50% visual-token pruning + MASQuant pseudo quantization.
# This script keeps its artifacts separate from the no-pruning MASQuant run.

# ---- Paths ----
export PROJECT_ROOT=/home/aistudio/LLMQuant-Learning/Code/Prune_Quant
export EXT_ROOT=/home/aistudio/EXT
export MODEL_PATH=/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct
export MASQUANT_ROOT="$EXT_ROOT/EfficientAI/masquant"
export VLMEVALKIT_ROOT="$EXT_ROOT/VLMEvalKit"
export WORK_DIR=/home/aistudio/datasets/output/qwen2vl_masquant_gae50

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

# Reduce allocator fragmentation for MASQuant's temporary weight tensors.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Qwen2-VL is noisy with the new default fast processor. Keep this empty to use
# the Transformers default, or set it to false for the slow processor.
export PROCESSOR_USE_FAST=

# Keep visual-token resolution fixed across vanilla, pure pruning, and MASQuant.
export PROCESSOR_MIN_PIXELS=
export PROCESSOR_MAX_PIXELS=
export PROCESSOR_MIN_VISUAL_TOKENS=1500
export PROCESSOR_MAX_VISUAL_TOKENS=1500

# ---- Pruning behavior ----
# 0.5 means keep the top 50% visual tokens during MASQuant calibration.
export CALIB_RETENTION_RATIO=0.5
# 0.5 means evaluate MASQuant with the same GAE 50% pruning policy.
export EVAL_RETENTION_RATIO=0.5
export MIN_KEEP=1
export GAE_ANSWER_SOURCE=generated
export GAE_PER_TOKEN=false

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
export VLMEVAL_WORK_DIR="$WORK_DIR/vlmeval_mme_mmstar_masquant_gae50"
export MAX_NEW_TOKENS=16
export VLMEVAL_VERBOSE=1
export VLMEVAL_SMART_RUNNER=1
export VLMEVAL_MODE=auto
export VLMEVAL_REUSE=1
export VLMEVAL_REUSE_AUX=1
export VLMEVAL_EXACT_MATCH_DATASETS="MME MMStar"
export VLMEVAL_DISABLE_OPENAI=1

# ---- Stage switches ----
export RUN_CALIBRATE=1
export RUN_CMC=1
export RUN_INSTALL_VLMEVAL=1
export RUN_VLMEVAL=1

# Resume examples:
# export MASQUANT_RESUME="$WORK_DIR/masquant_outputs/.../mas_parameters.pth"
# export MASQUANT_ACT_SCALES="$WORK_DIR/act_scales/Qwen2-VL-7B-Instruct-text-vision-128.pt"
# export CMC_LOW_RANK="$WORK_DIR/cmc/low_rank_adapters_quantcmc0_rank0.2_vision-audio-only.pt"
# export CMC_WHITE="$WORK_DIR/cmc/white_matrix_vision-audio-only.pt"

exec bash "$PROJECT_ROOT/remote/run_masquant_pseudo_pipeline.sh"
