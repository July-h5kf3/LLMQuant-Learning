# Prune + Quant Baseline

Baseline scaffolding for multimodal LLM visual-token pruning and quantized inference.

This repository is intentionally local-safe:

- no model weights are downloaded locally;
- no datasets are downloaded locally;
- real model inference is intended for a remote machine with existing model/data paths;
- model, data, and output paths are provided through CLI arguments, YAML config, or environment variables.

The first-stage implementation focuses on stable interfaces for attention-proxy pruning, physical token gather, quantized loader setup, model adapters, CLI skeletons, and synthetic tests.

## Prune Then MASQuant

This baseline keeps pruning before quantization in both phases:

1. Calibration: GAE scores are computed first, visual tokens are physically gathered, and MASQuant activation scales/cache are collected from the pruned prompts.
2. Inference: the MASQuant model is loaded first, then the same GAE pruning path builds pruned prompt embeddings for generation.

Example calibration on the remote model machine:

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage calibrate \
  --model-type qwen2_5_vl \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --masquant-root /path/to/EfficientAI/masquant \
  --work-dir /path/to/prune_then_masquant_work \
  --calib-jsonl /path/to/calib.jsonl \
  --retention-ratio 0.5 \
  --nsamples 128 \
  --patch-masquant-inputs-embeds-mask
```

Example pruned quantized inference:

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage infer \
  --model-type qwen2_5_vl \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --masquant-root /path/to/EfficientAI/masquant \
  --work-dir /path/to/prune_then_masquant_work \
  --eval-jsonl /path/to/eval.jsonl \
  --output-jsonl /path/to/output.jsonl \
  --masquant-resume /path/to/mas_parameters.pth \
  --retention-ratio 0.5
```
