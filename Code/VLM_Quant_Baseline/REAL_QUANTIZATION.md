# Real Quantization Workflow

This baseline keeps the original pseudo-quant search and eval path, and adds a
real W4A16 path for vLLM-loadable checkpoints.

## Export a W4A16 Checkpoint

Use `quantize_w4a16_vllm.py` to run GPTQ-style one-shot calibration through
`llm-compressor` and save compressed weights:

```bash
python quantize_w4a16_vllm.py \
  --model qwen2_vl \
  --model_id /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
  --output_dir /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16 \
  --calib_pairs /root/autodl-tmp/dataset/inferecne/question.json \
  --n_samples 8 \
  --max_seq_length 2048 \
  --trust_remote_code \
  --skip_sample_generation
```

Notes:

- The current exporter supports Qwen2-VL and Qwen2.5-VL.
- The scheme is W4A16 with group size 128, matching vLLM's compressed-tensors
  loading path.
- Vision modules and `lm_head` are excluded from weight packing by default.
- For a full run, increase `--n_samples` once the smoke test is healthy.

## Run Real-Quant Inference

Use `inference.py` with the vLLM backend and point it at the exported
checkpoint:

```bash
python inference.py \
  --model qwen2_vl \
  --model_args pretrained=/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
  --inference_engine vllm \
  --vllm_model_path /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16 \
  --vllm_quantization compressed-tensors \
  --vllm_dtype auto \
  --vllm_trust_remote_code \
  --w_bit 4 \
  --a_bit 16 \
  --infer_pairs inference/question.json \
  --save_path /root/autodl-tmp/eval/QIG/qwen2_vl_7b_w4a16_vllm/results.json \
  --max_new_tokens 64
```

This path does not call the pseudo-quant wrapper. It loads the packed checkpoint
directly through vLLM and writes a JSON file with metadata and generated answers.
