# Remote Testing Guide

This repository is designed so that local development only writes code.
Real model loading, dataset access, and GPU tests must be executed on the remote machine.

## Required information to be provided later

- SSH host
- SSH user
- SSH password or key
- Remote project path
- Remote model path
- Remote data path
- CUDA / Python environment information

## Do not commit

- SSH password
- Hugging Face token
- private model paths if sensitive
- benchmark outputs if too large

## Example remote smoke command

```bash
python -m prune_quant_baseline.scripts.run_infer_pruned \
  --model-type qwen2vl \
  --model-path "$MODEL_PATH" \
  --input-jsonl "$INPUT_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --pruner attention_proxy \
  --retention-ratio 0.5 \
  --quant-method none \
  --max-new-tokens 32
```
