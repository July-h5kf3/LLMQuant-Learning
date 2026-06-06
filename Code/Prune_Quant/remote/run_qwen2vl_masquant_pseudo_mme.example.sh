#!/usr/bin/env bash
set -euo pipefail

# Copy this file to a local run script, edit the variables below, then run it.
# This script intentionally contains configuration only; the full pipeline lives
# in remote/run_masquant_pseudo_pipeline.sh.

# ---- Paths ----
export PROJECT_ROOT=/home/aistudio/LLMQuant-Learning/Code/Prune_Quant
export EXT_ROOT=/home/aistudio/EXT
export MODEL_PATH=/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct
export MASQUANT_ROOT="$EXT_ROOT/EfficientAI/masquant"
export VLMEVALKIT_ROOT="$EXT_ROOT/VLMEvalKit"
export WORK_DIR=/home/aistudio/datasets/output/qwen2vl_masquant

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
# Pure MASQuant does not compute GAE attention scores, so default to accelerated
# SDPA. Set ATTN_IMPLEMENTATION=eager for debugging or attention-map workflows.
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export DTYPE=bfloat16
export DEVICE_MAP=auto

# Qwen2-VL is noisy with the new default fast processor. Keep this empty to use
# the Transformers default, or set it to false for the slow processor.
export PROCESSOR_USE_FAST=

# Optional image resolution controls. Keep this fixed so vanilla, pure pruning,
# and MASQuant runs are comparable.
export PROCESSOR_MIN_PIXELS=
export PROCESSOR_MAX_PIXELS=
export PROCESSOR_MIN_VISUAL_TOKENS="${PROCESSOR_MIN_VISUAL_TOKENS:-}"
export PROCESSOR_MAX_VISUAL_TOKENS="${PROCESSOR_MAX_VISUAL_TOKENS:-1500}"

# ---- Pruning behavior ----
# 1.0 means no GAE pruning. Set to 0.5 to run prune-then-quant calibration.
export CALIB_RETENTION_RATIO=1.0
# 1.0 evaluates MASQuant only. Set to 0.5 for MASQuant + GAE pruning at eval time.
export EVAL_RETENTION_RATIO=1.0
export MIN_KEEP=1
export GAE_ANSWER_SOURCE=generated
export GAE_PER_TOKEN=false

# Qwen2.5-VL with pruned inputs_embeds usually needs this. Qwen2-VL usually does not.
export PATCH_MASQUANT_INPUTS_EMBEDS_MASK=0

# ---- CMC ----
export USE_CMC=1
export CMC_NET=qwen2-vl-7b
export CMC_CALI_DATA_TYPE=vision-audio-only
export CMC_RANK=0.2
export CMC_QUANT_CMC=0
export CMC_N_CALI_SAMPLES=128

# ---- VLMEvalKit ----
export VLMEVAL_DATASETS="${VLMEVAL_DATASETS:-MME}"
export VLMEVAL_MODEL_NAME=Qwen2VL_MASQuant_Pseudo
export MAX_NEW_TOKENS=16
export VLMEVAL_VERBOSE=1
# The smart runner reuses existing xlsx files and forces exact-matching judges
# for datasets like MME, so scoring will not call GPT for those benchmarks.
export VLMEVAL_SMART_RUNNER=1
export VLMEVAL_MODE=auto
export VLMEVAL_REUSE=1
export VLMEVAL_REUSE_AUX=1
export VLMEVAL_EXACT_MATCH_DATASETS="${VLMEVAL_EXACT_MATCH_DATASETS:-MME}"
# Keep OpenAI env vars out of exact-matching runs to avoid slow or failing API
# calls during scoring.
export VLMEVAL_DISABLE_OPENAI=1

# ---- Stage switches ----
# Set a stage to 0 if it has already been completed and you only want to resume later stages.
export RUN_CALIBRATE=1
export RUN_CMC=1
export RUN_INSTALL_VLMEVAL=1
export RUN_VLMEVAL=1
export RUN_LMMS_EVAL="${RUN_LMMS_EVAL:-1}"

# ---- lmms-eval ----
export LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-$PROJECT_ROOT/third_party/lmms-eval}"
export LMMS_EVAL_TASKS="${LMMS_EVAL_TASKS:-mmmu_val ocrbench vizwiz_vqa_val scienceqa_img textvqa_val}"
export LMMS_EVAL_OUTPUT_PATH="${LMMS_EVAL_OUTPUT_PATH:-$WORK_DIR/lmms_eval_masquant_pseudo}"
export LMMS_EVAL_CACHE="${LMMS_EVAL_CACHE:-$WORK_DIR/lmms_eval_cache}"
export LMMS_EVAL_LIMIT="${LMMS_EVAL_LIMIT:-}"
export LMMS_EVAL_LOG_SAMPLES="${LMMS_EVAL_LOG_SAMPLES:-1}"
export LMMS_EVAL_VERBOSITY="${LMMS_EVAL_VERBOSITY:-INFO}"
export LMMS_EVAL_DISABLE_OPENAI="${LMMS_EVAL_DISABLE_OPENAI:-1}"

# If you already have artifacts, uncomment and set them explicitly.
# export MASQUANT_RESUME="$WORK_DIR/masquant_outputs/.../mas_parameters.pth"
# export MASQUANT_ACT_SCALES="$WORK_DIR/act_scales/Qwen2-VL-7B-Instruct-text-vision-128.pt"
# export CMC_LOW_RANK="$WORK_DIR/cmc/low_rank_adapters_quantcmc0_rank0.2_vision-audio-only.pt"
# export CMC_WHITE="$WORK_DIR/cmc/white_matrix_vision-audio-only.pt"

exec bash "$PROJECT_ROOT/remote/run_masquant_pseudo_pipeline.sh"
