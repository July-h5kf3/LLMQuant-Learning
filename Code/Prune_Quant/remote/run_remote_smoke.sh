#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_TYPE:=qwen2vl}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${INPUT_JSONL:?INPUT_JSONL is required}"
: "${OUTPUT_JSONL:?OUTPUT_JSONL is required}"

python -m prune_quant_baseline.scripts.run_infer_pruned \
  --model-type "$MODEL_TYPE" \
  --model-path "$MODEL_PATH" \
  --input-jsonl "$INPUT_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --pruner attention_proxy \
  --retention-ratio "${RETENTION_RATIO:-0.5}" \
  --quant-method "${QUANT_METHOD:-none}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-32}"
