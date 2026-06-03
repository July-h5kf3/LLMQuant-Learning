# Real Quantization With TensorRT-LLM

This repo keeps the fake-quant path for method research and adds a real-quant path for `lmms-eval` throughput measurement. The main flow is:

1. Export a TensorRT-LLM checkpoint with ModelOpt.
2. Run `main.py --real_quant --inference_engine trtllm`.
3. Compare metric and wall-clock time against the existing fake-quant run.

## Support Matrix

| Format | Export path | `lmms-eval` path | Status |
| --- | --- | --- | --- |
| W4A16 | TensorRT-LLM ModelOpt `int4_awq` | TensorRT-LLM engine | Main supported path. |
| W4A8 | TensorRT-LLM ModelOpt `w4a8_awq` | TensorRT-LLM engine | Main target for RTX 4090 INT acceleration. |
| W3A16 | AutoRound `W3A16` | vLLM | Keep the existing working path unchanged; TensorRT-LLM does not list INT3/W3A16 engine support. |

The first target should be `qwen2_vl`. The eval adapter can also route TensorRT-LLM-supported Qwen2.5-VL/Qwen3-VL/LLaVA/VILA input processors, but `quantize_trtllm.py` keeps the W4 ModelOpt export path conservative because the upstream TensorRT-LLM exporter currently has an explicit Qwen2-VL branch. `internvl2` is intentionally blocked until TensorRT-LLM has a compatible multimodal input processor for it.

For RTX 4090, use the Ada compute capability target (`FLASHINFER_CUDA_ARCH_LIST=8.9`) and verify the exact CUDA driver, TensorRT-LLM release, and GPU compute capability on the target machine before treating W4A16/W4A8 numbers as final. The previous FP4 fallback branch has been removed from this baseline.

## Installation

Use a Linux NVIDIA environment. TensorRT-LLM is not a macOS package. NVIDIA's current Linux pip guide installs the prebuilt wheel with `pip3 install tensorrt_llm` after CUDA/PyTorch prerequisites are aligned; if your target image already comes from a TensorRT-LLM NGC container, prefer the wheel and PyTorch version shipped in that container.

```bash
conda create -n QIG_TRTLLM python=3.12 -y
conda activate QIG_TRTLLM

cd /root/autodl-tmp/LLMQuant-Learning/Code/VLM_Quant_Baseline
pip install -r requirements.txt
pip install -r requirements-trtllm.txt
pip install -e .

cd /root/autodl-tmp/QIG/3rdparty/lmms-eval
pip install -e .
```

If pip resolves a TensorRT-LLM build that expects a different CUDA/PyTorch stack, recreate the environment from the matching NVIDIA container or follow the exact PyTorch command in the TensorRT-LLM install page for that release. Do not mix a random PyTorch wheel with a TensorRT-LLM wheel built against another CUDA major/minor version.

For the RTX 4090 target, the eval/export scripts default to:

```bash
export TRT_LLM_NO_LIB_INIT=1
export FLASHINFER_CUDA_ARCH_LIST=8.9
```

The eval/export scripts set these by default and still allow overrides through the environment.

Sanity check:

```bash
python - <<'PY'
import torch
import tensorrt_llm
print("cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("tensorrt_llm", getattr(tensorrt_llm, "__version__", "unknown"))
PY
```

## Export W4

W4A16:

```bash
python quantize_trtllm.py \
  --model qwen2_vl \
  --model_dir /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
  --output_dir /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-trtllm \
  --quant_format w4a16 \
  --calib_dataset scienceqa \
  --calib_size 128 \
  --calib_max_seq_length 512 \
  --batch_size 1 \
  --tp_size 1 \
  --awq_block_size 128 \
  --force
```

W4A8:

```bash
python quantize_trtllm.py \
  --model qwen2_vl \
  --model_dir /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
  --output_dir /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A8-trtllm \
  --quant_format w4a8 \
  --calib_dataset scienceqa \
  --calib_size 128 \
  --calib_max_seq_length 512 \
  --batch_size 1 \
  --tp_size 1 \
  --awq_block_size 128 \
  --force
```

TensorRT-LLM AWQ internally caps `calib_size` to 32 in current releases when `qformat` contains `awq`, so `--calib_size 128` is a request for enough source examples rather than a guarantee that all 128 are used by the AWQ search.

For large models, increase `--tp_size` during export and use the same `TRTLLM_TP_SIZE` during evaluation.

The W4 exporter currently enables `--model qwen2_vl`, `--model llava_onevision`, `--model vila`, and `--model llava`. If you want to try Qwen2.5-VL/Qwen3-VL, first check the TensorRT-LLM version on the target machine for a matching ModelOpt export branch; otherwise use the PyTorch backend only for functional debugging, not speedup numbers.

## W3A16 Fallback

W3A16 is exported through AutoRound because TensorRT-LLM does not currently advertise W3A16/INT3 engine support:

```bash
python quantize_trtllm.py \
  --model qwen2_vl \
  --model_dir /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
  --output_dir /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W3A16-autoround \
  --quant_format w3a16 \
  --autoround_dataset NeelNanda/pile-10k \
  --autoround_nsamples 128 \
  --autoround_iters 200 \
  --autoround_batch_size 1 \
  --force
```

Use W3A16 for accuracy exploration or a non-TRT backend. It is not suitable for TensorRT-LLM speedup claims yet.

## lmms-eval

Smoke test one task:

```bash
LIMIT=1 \
TASKS=mmmu_val \
FP16_CHECKPOINT=/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
REAL_CHECKPOINT=/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-trtllm \
W_BIT=4 A_BIT=16 \
bash scripts/run_qig_real_trtllm_eval.sh
```

Full run:

```bash
TASKS=mmmu_val,vizwiz_vqa_val,chartqa,ai2d,scienceqa_img \
FP16_CHECKPOINT=/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
REAL_CHECKPOINT=/root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-trtllm \
W_BIT=4 A_BIT=16 \
bash scripts/run_qig_real_trtllm_eval.sh
```

Switch to W4A8 by changing `REAL_CHECKPOINT` to the W4A8 export directory and setting `W_BIT=4 A_BIT=8`.

Useful script knobs:

- `MODEL=qwen2_vl` selects the lmms-eval model adapter name. The script defaults to `qwen2_vl`.
- `TRTLLM_ENGINE_DIR=/path/to/engine-cache` saves a built engine after the first run.
- `TRTLLM_WORKSPACE=/path/to/workspace` controls TensorRT-LLM temporary build files.
- `TRTLLM_ENABLE_BUILD_CACHE=1` enables TensorRT-LLM LLM API build cache.
- `TRTLLM_FAST_BUILD=1` requests TensorRT-LLM fast build mode.
- `DRY_RUN=1` prints the command without touching conda, checkpoint paths, or output folders.

Equivalent direct command:

```bash
python -W ignore main.py \
  --model qwen2_vl \
  --tasks mmmu_val \
  --batch_size 1 \
  --log_samples \
  --output_path /root/autodl-tmp/eval/QIG/real_trtllm \
  --real_quant \
  --inference_engine trtllm \
  --trtllm_model_path /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-trtllm \
  --trtllm_tokenizer_path /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
  --trtllm_backend engine \
  --trtllm_max_multimodal_len 1296 \
  --trtllm_trust_remote_code \
  --w_bit 4 \
  --a_bit 16 \
  --gen_kwargs temperature=0,max_new_tokens=64
```

`--trtllm_model_path` points to the real-quant TensorRT-LLM checkpoint. `--trtllm_tokenizer_path` must point to the original HF checkpoint because the multimodal processor and chat template live there.

`--trtllm_max_multimodal_len` maps to TensorRT-LLM `max_prompt_embedding_table_size`. The default `1296` follows the TensorRT-LLM Qwen2-VL example for `max_batch_size=4` and `image_shape=[504,504]`. Increase it if a task uses larger images, more images per prompt, or larger eval batch sizes.

## Inference Smoke Test

`inference.py` also accepts `--inference_engine trtllm` for one-off checks, but `lmms-eval` is the primary path.

```bash
python inference.py \
  --model qwen2_vl \
  --inference_engine trtllm \
  --trtllm_model_path /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct-W4A16-trtllm \
  --trtllm_tokenizer_path /root/autodl-tmp/weights/Qwen/Qwen2-VL-7B-Instruct \
  --trtllm_trust_remote_code \
  --w_bit 4 \
  --a_bit 16 \
  --infer_pairs inference/question.json \
  --save_path /root/autodl-tmp/eval/QIG/trtllm_smoke.json \
  --max_new_tokens 64 \
  --temperature 0
```
