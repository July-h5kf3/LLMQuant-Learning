# Prune + Quant Baseline

Baseline scaffolding for multimodal LLM visual-token pruning and quantized inference.

This repository is intentionally local-safe:

- no model weights are downloaded locally;
- no datasets are downloaded locally;
- real model inference is intended for a remote machine with existing model/data paths;
- model, data, and output paths are provided through CLI arguments, YAML config, or environment variables.

The first-stage implementation focuses on stable interfaces for attention-proxy pruning, physical token gather, quantized loader setup, model adapters, CLI skeletons, and synthetic tests.
