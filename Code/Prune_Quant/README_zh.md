# Prune + Quant Baseline

[English](README.md) | [简体中文](README_zh.md)

用于多模态大语言模型视觉 token 剪枝与量化推理的 baseline 脚手架。

本仓库默认保持本地安全：

- 不提交模型权重；
- 不提交数据集；
- 真实模型推理应在已有模型/数据路径的 GPU 机器上运行；
- 模型、数据和输出路径通过 CLI 参数、YAML 配置或环境变量传入。

当前实现支持：

- 纯 GAE 指导的视觉 token 剪枝；
- 先剪枝再量化的实验，即 GAE 剪枝后接 MASQuant 校准/推理；
- 面向图像任务的 Qwen2-VL 风格 Hugging Face 输入。

## 推荐目录结构

建议把代码、模型权重、数据集和实验输出放在不同目录下。

```bash
export PROJECT_ROOT=/path/to/Prune_Quant
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export WORK_ROOT=/path/to/prune_quant_runs

mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$WORK_ROOT"
```

对于纯 GAE 剪枝，`Qwen/Qwen2-VL-7B-Instruct` 与当前复现路径一致。
对于 MASQuant，请使用 MASQuant 支持的模型，例如 `Qwen/Qwen2.5-VL-7B-Instruct`。

## 环境安装

在 GPU 机器上创建 Python 环境。

```bash
conda create -n prune-quant python=3.10 -y
conda activate prune-quant

cd "$PROJECT_ROOT"
pip install -U pip setuptools wheel
```

先检查可见 GPU 和驱动：

```bash
nvidia-smi
```

本项目应能同时运行在 A800 和 RTX PRO 6000 Blackwell 机器上。由于 RTX PRO 6000 Blackwell 架构更新，任何可能运行在 Blackwell 上的环境都建议使用 CUDA 12.8 PyTorch wheel。

RTX PRO 6000 Blackwell，以及 A800/Blackwell 共用环境的推荐安装方式：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

当机器驱动足够新、能够支持 CUDA 12.8 时使用上面的安装方式。A800 属于 Ampere，兼容较新的 CUDA runtime；RTX PRO 6000 Blackwell 则应使用较新的 CUDA/PyTorch 软件栈。

不要把共享环境统一到 CUDA 12.6：它对 A800 没问题，但 PyTorch `cu126` wheel 并不是 RTX PRO 6000 Blackwell / SM120 的稳妥目标。Blackwell 使用 `cu128`。

仅 A800 且驱动较旧的机器可以使用下面的 fallback：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

不要在 RTX PRO 6000 Blackwell 上使用这个 A800-only fallback。如果驱动支持，优先使用一个共享的 `cu128` 环境；如果 A800 机器驱动较旧，则使用独立环境。

验证安装的 wheel 能否看到 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

安装本项目和常用运行依赖：

```bash
pip install -e ".[quant,test]"
pip install accelerate datasets huggingface_hub qwen-vl-utils sentencepiece protobuf
```

安装 VLMEvalKit，并注册本仓库的 pruned Qwen2-VL 模型 wrapper。VLMEvalKit 使用 `run.py` 做标准推理和评测，模型名通过 `vlmeval/config.py` 中的 `supported_VLM` 解析。

```bash
export EXT_ROOT=/path/to/external
git clone https://github.com/open-compass/VLMEvalKit.git "$EXT_ROOT/VLMEvalKit"
export VLMEVALKIT_ROOT="$EXT_ROOT/VLMEvalKit"

cd "$VLMEVALKIT_ROOT"
pip install -e .

cd "$PROJECT_ROOT"
python remote/install_vlmeval_pruned_gae.py --vlmeval-root "$VLMEVALKIT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
```

如果修改了 `src/prune_quant_baseline/vlmeval/` 下的 wrapper 文件，需要重新运行 `remote/install_vlmeval_pruned_gae.py`，因为安装脚本会把 wrapper 复制到 VLMEvalKit checkout 中。

GAE 剪枝需要 attention tensor 和 attention gradient。下面的命令都使用 eager attention：

```bash
export TOKENIZERS_PARALLELISM=false
```

## 模型下载

如果目标模型需要认证，先登录 Hugging Face。

```bash
huggingface-cli login
```

下载用于纯 GAE 剪枝的 Qwen2-VL 模型。

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen2-VL-7B-Instruct"

export QWEN2VL_MODEL="$MODEL_ROOT/Qwen2-VL-7B-Instruct"
```

下载用于 MASQuant 实验的 Qwen2.5-VL 模型。

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"

export QWEN25VL_MODEL="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"
```

脚本默认从本地加载模型。请让 `--model-path` 指向下载好的本地目录。

## 数据集下载

JSONL 推理/校准脚本要求每行一个样本，包含图片路径、prompt，以及可选 answer：

```json
{"id": "0", "image": "/abs/path/to/image.jpg", "prompt": "Describe the image.", "answer": "A short reference answer."}
```

对于 GAE oracle 剪枝，需要 answer 来定义目标。如果 JSONL 中没有 `answer`，传入 `--gae-answer-source generated`，让模型先生成 replay answer。

### MASQuant 校准数据

本仓库的 prune-then-MASQuant 路径通过 `--calib-jsonl` 控制校准数据。你不需要使用 MASQuant 代码中硬编码的 `/nas/...` 路径。一个实用的 text-vision 校准集是 MASQuant 过滤后的 ShareGPT4V metadata 加 COCO `train2017` 图片。

下载 MASQuant 过滤后的 ShareGPT4V metadata：

```bash
mkdir -p "$DATA_ROOT/masquant" "$DATA_ROOT/coco"

wget -O "$DATA_ROOT/masquant/sharegpt4v_filtered_coco.json" \
  https://raw.githubusercontent.com/alibaba/EfficientAI/main/masquant/dataset/sharegpt4v_instruct_gpt4-vision_cap100k_filtered_coco_image.json
```

下载 COCO `train2017` 图片。这个文件较大，压缩包约 18 GB：

```bash
wget -c -P "$DATA_ROOT/coco" http://images.cocodataset.org/zips/train2017.zip
unzip -q -n "$DATA_ROOT/coco/train2017.zip" -d "$DATA_ROOT/coco"
```

把 metadata 转换成本项目的 JSONL 格式：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
src = data_root / "masquant" / "sharegpt4v_filtered_coco.json"
out = data_root / "calib_sharegpt4v_coco.jsonl"
rows = json.loads(src.read_text())

with out.open("w", encoding="utf-8") as f:
    for row in rows:
        prompt = ""
        answer = ""
        for turn in row["conversations"]:
            if turn["from"] == "human":
                prompt = turn["value"].replace("<image>", "").strip()
            elif turn["from"] == "gpt":
                answer = turn["value"].strip()
                break
        image = data_root / row["image"]
        if not image.exists():
            image = data_root / "coco" / row["image"].replace("coco/", "")
        if not image.exists():
            continue
        f.write(json.dumps({
            "id": row.get("id", image.stem),
            "image": str(image.resolve()),
            "prompt": prompt,
            "answer": answer,
        }, ensure_ascii=False) + "\n")

print(out)
PY

export CALIB_JSONL="$DATA_ROOT/calib_sharegpt4v_coco.jsonl"
```

小规模 smoke run 可以创建一个短校准子集：

```bash
head -n 16 "$CALIB_JSONL" > "$DATA_ROOT/calib_smoke_16.jsonl"
```

### 测试数据集

VLMEvalKit 是默认 benchmark 路径。它负责数据集加载、预测文件生成和指标计算。把 VLMEvalKit 的数据缓存放到 `DATA_ROOT` 下：

```bash
export LMUData="$DATA_ROOT/vlmeval"
mkdir -p "$LMUData"
```

不剪枝的 VLMEvalKit smoke run：

```bash
cd "$VLMEVALKIT_ROOT"
export PQ_RETENTION_RATIO=1.0
python run.py --data MME --model Qwen2VL_PrunedGAE \
  --work-dir "$WORK_ROOT/vlmeval_smoke" \
  --verbose
```

常用 benchmark 运行方式：

```bash
cd "$VLMEVALKIT_ROOT"
python run.py --data MME MMStar MMVet --model Qwen2VL_PrunedGAE \
  --work-dir "$WORK_ROOT/vlmeval_qwen2vl_gae50" \
  --verbose
```

如果只是调试低层脚本，也可以创建或下载自定义评测 JSONL：

```bash
export EVAL_JSONL="$DATA_ROOT/eval.jsonl"
```

## 纯 GAE 剪枝

默认评测路径使用 VLMEvalKit。安装脚本会注册 `Qwen2VL_PrunedGAE`，运行参数通过环境变量控制：

```bash
export QWEN2VL_MODEL="$MODEL_ROOT/Qwen2-VL-7B-Instruct"
export PQ_RETENTION_RATIO=0.5
export PQ_MIN_KEEP=1
export PQ_MAX_NEW_TOKENS=16
export PQ_GAE_ANSWER_SOURCE=generated
export PQ_GAE_PER_TOKEN=false
export PQ_ATTN_IMPLEMENTATION=eager

# 对齐论文中的 Qwen2-VL 图像设置，约 1500 个 visual language token。
export PQ_MIN_VISUAL_TOKENS=1500
export PQ_MAX_VISUAL_TOKENS=1500

cd "$VLMEVALKIT_ROOT"
python run.py --data MME MMStar MMVet --model Qwen2VL_PrunedGAE \
  --work-dir "$WORK_ROOT/vlmeval_qwen2vl_gae50" \
  --verbose
```

如果 A800 显存紧张，先降低图像 token budget：

```bash
export PQ_MIN_VISUAL_TOKENS=
export PQ_MAX_VISUAL_TOKENS=1024
```

然后重新运行同一个 `python run.py ...` 命令。

底层 JSONL 脚本仍然适合不用 VLMEvalKit 时调试小规模自定义文件：

```bash
python -m prune_quant_baseline.scripts.run_infer_pruned \
  --model-type qwen2vl \
  --model-path "$QWEN2VL_MODEL" \
  --input-jsonl "$EVAL_JSONL" \
  --output-jsonl "$WORK_ROOT/qwen2vl_gae50.jsonl" \
  --pruner gae_oracle \
  --retention-ratio 0.5 \
  --min-keep 1 \
  --quant-method none \
  --dtype bfloat16 \
  --device-map auto \
  --attn-implementation eager \
  --gae-answer-source generated \
  --max-new-tokens 128
```

可选的本地 benchmark helper，主要用于 TSV/debug：

```bash
python -m prune_quant_baseline.scripts.run_image_benchmark \
  --dataset MME \
  --dataset-source hf \
  --hf-dataset lmms-lab/MME \
  --hf-split test \
  --hf-cache-dir "$DATA_ROOT/hf_cache" \
  --model-path "$QWEN2VL_MODEL" \
  --model-type qwen2vl \
  --output-jsonl "$WORK_ROOT/mme_qwen2vl_gae50.jsonl" \
  --metrics-json "$WORK_ROOT/mme_qwen2vl_gae50_metrics.json" \
  --pruner gae_oracle \
  --retention-ratio 0.5 \
  --attn-implementation eager \
  --gae-answer-source sample \
  --gae-per-token false \
  --processor-max-pixels 401408 \
  --max-new-tokens 16
```

如果 MME 数据已经是本地 VLMEvalKit 风格 TSV，helper 也可以直接读取。这个路径用于调试，不作为默认报告结果：

```bash
python -m prune_quant_baseline.scripts.run_image_benchmark \
  --dataset MME \
  --dataset-source tsv \
  --tsv "$DATA_ROOT/MME.tsv" \
  --model-path "$QWEN2VL_MODEL" \
  --model-type qwen2vl \
  --output-jsonl "$WORK_ROOT/mme_qwen2vl_gae50.jsonl" \
  --metrics-json "$WORK_ROOT/mme_qwen2vl_gae50_metrics.json" \
  --pruner gae_oracle \
  --retention-ratio 0.5 \
  --attn-implementation eager \
  --gae-answer-source sample \
  --gae-per-token false \
  --processor-max-pixels 401408 \
  --max-new-tokens 16
```

TSV 应包含常见的 VLMEvalKit 列，包括作为 base64 图片数据的 `image`，以及 `question` 和 `answer`。`category`、`question_id`、`image_path` 等可选列会保留到输出中。

对于 MME/MMStar 风格评测，优先使用 `--gae-answer-source sample`，因为 TSV 已经包含短标签答案（`A/B/C/D` 或 `Yes/No`）。`--gae-answer-source generated` 会先生成 replay answer，再对这段 answer 运行 GAE，这会增加显存开销，也可能把一个 token 的标签变成多个 answer token。`--gae-per-token false` 对整段 answer 只做一次 backward，而不是每个 answer token 做一次 backward；这是 A800/RTX PRO 6000 上更稳的默认选择。Qwen2-VL 报告的是 spatial merge 之后的视觉 token；`216` 个 language-side visual token 通常对应约 `864` 个 merge 前 ViT patch。输出 JSONL 现在会同时记录 `num_visual_language_tokens` 和 `num_image_patches_before_merge`。

`--processor-max-pixels` 控制 Qwen2-VL 动态图像分辨率。A800 上可以先从 `401408` (`512 * 28 * 28`) 开始；如果显存稳定，可以尝试更高质量的 `802816` (`1024 * 28 * 28`)。如果要对齐论文中 Qwen2-VL 图像约 1500 个 visual token 的设置，可以使用 token budget 参数，而不是手算 pixels：

```bash
--processor-min-visual-tokens 1500 \
--processor-max-visual-tokens 1500
```

这大约等价于 `1176000` max pixels (`1500 * 28 * 28`)。这个设置更贴近论文，但比低显存设置重很多。

使用 `--retention-ratio 1.0` 可以在相同数据/模型设置下运行 no-pruning baseline。

## MASQuant 安装

MASQuant 作为外部代码仓库使用。

```bash
export EXT_ROOT=/path/to/external
mkdir -p "$EXT_ROOT"

git clone https://github.com/alibaba/EfficientAI.git "$EXT_ROOT/EfficientAI"
export MASQUANT_ROOT="$EXT_ROOT/EfficientAI/masquant"

cd "$MASQUANT_ROOT"
pip install -e .
pip install lmms-eval
```

`flash-attn` 对这个 baseline 是可选的。GAE 需要 eager attention，prune-then-MASQuant 脚本可以 patch MASQuant 的 Qwen2.5 loader，让它遵循 `--attn-implementation eager`。

如果要用 VLMEvalKit 跑 MASQuant 量化后的 TensorRT 模型，还需要在远端 GPU 环境中安装 TensorRT / TensorRT-LLM，并准备一个能够把 MASQuant 参数导入 TensorRT engine 的构建脚本。MASQuant 上游当前保存的是 `mas_parameters.pth`，本仓库不会假设某个固定的 TensorRT 导出入口，而是通过 `--tensorrt-builder-command` 接入你的构建命令。

## GAE + MASQuant

prune-then-quant baseline 分为两个阶段。

阶段 1：在已经剪枝的 prompt 上校准 MASQuant：

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage calibrate \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --masquant-root "$MASQUANT_ROOT" \
  --work-dir "$WORK_ROOT/qwen25vl_gae50_masquant" \
  --calib-jsonl "$CALIB_JSONL" \
  --retention-ratio 0.5 \
  --min-keep 1 \
  --nsamples 128 \
  --batch-size 1 \
  --wbits 4 \
  --abits 8 \
  --epochs 2 \
  --attn-implementation eager \
  --gae-answer-source generated \
  --patch-masquant-inputs-embeds-mask
```

该命令会写出：

- 位于 `$WORK_ROOT/qwen25vl_gae50_masquant/cache` 下的 pruned MASQuant cache；
- 位于 `$WORK_ROOT/qwen25vl_gae50_masquant/act_scales` 下的 pruned activation scales；
- 位于 `$WORK_ROOT/qwen25vl_gae50_masquant/masquant_outputs` 下的 MASQuant 输出。

查找训练得到的 MASQuant 参数：

```bash
export MASQUANT_RESUME=$(find "$WORK_ROOT/qwen25vl_gae50_masquant/masquant_outputs" \
  -name mas_parameters.pth | sort | tail -n 1)

echo "$MASQUANT_RESUME"
```

阶段 2：加载 MASQuant 模型进行推理，并再次应用 GAE 剪枝：

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage infer \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --masquant-root "$MASQUANT_ROOT" \
  --work-dir "$WORK_ROOT/qwen25vl_gae50_masquant" \
  --eval-jsonl "$EVAL_JSONL" \
  --output-jsonl "$WORK_ROOT/qwen25vl_gae50_masquant/eval_gae50_masquant.jsonl" \
  --masquant-resume "$MASQUANT_RESUME" \
  --retention-ratio 0.5 \
  --min-keep 1 \
  --wbits 4 \
  --abits 8 \
  --attn-implementation eager \
  --gae-answer-source generated \
  --max-new-tokens 128
```

这里最重要的不变量是：

- 校准阶段先使用 GAE-pruned prompts，再让 MASQuant 学习量化参数；
- 推理阶段先加载 MASQuant，再在 generation 前对 prompt 应用 GAE 剪枝。

### CMC 补偿

MASQuant 完整流程还包含 CMC（Cross-Modal Compensation）。在阶段 1 得到 `mas_parameters.pth` 后，可以额外运行 CMC，生成 white matrix 和 low-rank adapters：

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage cmc \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --masquant-root "$MASQUANT_ROOT" \
  --work-dir "$WORK_ROOT/qwen25vl_gae50_masquant" \
  --masquant-resume "$MASQUANT_RESUME" \
  --wbits 4 \
  --abits 8 \
  --cmc-net qwen2.5-vl-7b \
  --cmc-cali-data-type vision-audio-only \
  --cmc-rank 0.2 \
  --cmc-quant-cmc 0 \
  --cmc-n-cali-samples 128 \
  --cmc-vision-json "$DATA_ROOT/masquant/sharegpt4v_filtered_coco.json" \
  --cmc-vision-prefix "$DATA_ROOT/coco/train2017"
```

该命令默认使用 MASQuant 上游的 `infer_mas.py`，并把 CMC 产物保存到：

- `$WORK_ROOT/qwen25vl_gae50_masquant/cmc/white_matrix_vision-audio-only.pt`
- `$WORK_ROOT/qwen25vl_gae50_masquant/cmc/low_rank_adapters_quantcmc0_rank0.2_vision-audio-only.pt`

注意：CMC 的 `--scales_path` 对应的是 MAS 训练后的 `mas_parameters.pth`，不是 raw activation scales。脚本默认使用 `--masquant-resume` 作为 CMC 的 scales path；只有需要覆盖时才传 `--cmc-scales-path`。

如果你的 MASQuant checkout 里有自定义的 VL 专用入口，可以用 `--cmc-script-name infer_mas_vl.py` 覆盖。`vision-audio-only` 校准数据读取逻辑来自 MASQuant 上游脚本，原始代码硬编码了 `/nas/yuehu/...` 路径；这里的 `--cmc-vision-json` 和 `--cmc-vision-prefix` 会在运行前 patch 到你的本地 ShareGPT4V/COCO 路径。如果只想先验证流程，可以用 `--cmc-cali-data-type no-white`，它不读取 COCO 白化数据。

### 保存 MASQuant TensorRT 产物并用 VLMEvalKit 评测

如果目标是“量化一次，评测时直接加载”，在阶段 1 和 CMC 后，先构建 TensorRT engine，并把 engine、MASQuant 参数、CMC 产物和 processor 打包成 artifact：

```bash
export MASQUANT_TRT_ARTIFACT="$WORK_ROOT/qwen25vl_gae50_masquant/masquant_trt_artifact"
export TRT_ENGINE_DIR="$MASQUANT_TRT_ARTIFACT/engine"
export CMC_LOW_RANK="$WORK_ROOT/qwen25vl_gae50_masquant/cmc/low_rank_adapters_quantcmc0_rank0.2_vision-audio-only.pt"
export CMC_WHITE="$WORK_ROOT/qwen25vl_gae50_masquant/cmc/white_matrix_vision-audio-only.pt"

python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage export-tensorrt \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --work-dir "$WORK_ROOT/qwen25vl_gae50_masquant" \
  --masquant-resume "$MASQUANT_RESUME" \
  --masquant-act-scales "$WORK_ROOT/qwen25vl_gae50_masquant/act_scales/Qwen2.5-VL-7B-Instruct-text-vision-128.pt" \
  --wbits 4 \
  --abits 8 \
  --cmc-low-rank-adapters "$CMC_LOW_RANK" \
  --cmc-white-matrix-path "$CMC_WHITE" \
  --tensorrt-engine-dir "$TRT_ENGINE_DIR" \
  --tensorrt-artifact-dir "$MASQUANT_TRT_ARTIFACT" \
  --tensorrt-builder-command "python -m prune_quant_baseline.scripts.build_masquant_tensorrt --model {model_path} --model-type qwen2vl --masquant-root $MASQUANT_ROOT --masquant-resume {masquant_resume} --act-scales {masquant_act_scales} --cmc-low-rank {cmc_low_rank_adapters} --cmc-white-matrix {cmc_white_matrix} --output {engine_dir} --wbits {wbits} --abits {abits} --convert-command 'python /path/to/masquant_trt_convert.py --model {hf_export_dir} --state {torch_export_dir}/masquant_state.pt --out {checkpoint_dir}' --llm-build-command 'trtllm-build --checkpoint_dir {checkpoint_dir} --output_dir {llm_engine_dir} --gemm_plugin=float16 --gpt_attention_plugin=float16 --max_batch_size 1 --max_input_len 2048 --max_seq_len 3072 --max_multimodal_len 1296' --vision-build-command 'python /path/to/build_qwen2vl_vision_engine.py --model {hf_export_dir} --output_dir {vision_engine_dir}'"
```

`build_masquant_tensorrt` 会先把 MASQuant + CMC 权重量化结果写到 `$TRT_ENGINE_DIR/.build/masquant_export`，再执行你传入的 TensorRT 转换/构建命令。这里的 `/path/to/masquant_trt_convert.py` 和 `/path/to/build_qwen2vl_vision_engine.py` 需要替换成你环境里真正支持 MASQuant `QuantLinear`、分模态 scale 和 CMC 低秩分支的转换脚本。TensorRT-LLM 自带 Qwen2-VL 示例 converter 通常不认识 MASQuant 的自定义模块；若只想做 TensorRT-LLM 连通性检查，可以在 builder 上传 `--tensorrt-llm-root "$TENSORRT_LLM_ROOT" --allow-stock-trtllm-example`。若 `TRT_ENGINE_DIR` 已经由别的流程构建好，可以省略 `--tensorrt-builder-command`，只写 artifact manifest。非 dry-run 模式会检查 engine 目录非空，避免注册空产物。

然后用 VLMEvalKit 直接加载这个保存好的 TensorRT artifact：

```bash
cd "$PROJECT_ROOT"
python remote/install_vlmeval_pruned_gae.py --vlmeval-root "$VLMEVALKIT_ROOT"

cd "$VLMEVALKIT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export PQ_MASQUANT_TRT_ARTIFACT="$MASQUANT_TRT_ARTIFACT"
export PQ_MAX_NEW_TOKENS=16

python run.py --data MME MMStar MMVet --model Qwen2VL_MASQuant_TensorRT \
  --work-dir "$WORK_ROOT/vlmeval_qwen25vl_masquant_trt" \
  --verbose
```

`Qwen2VL_MASQuant_TensorRT` 只读取保存好的 artifact，不会在 VLMEvalKit 推理阶段重新跑 MASQuant 校准或重新构建 engine。默认 runtime 使用 TensorRT-LLM 的 `ModelRunner`；如果你的 engine 需要自定义多模态输入约定，可以把自定义 runtime 类路径写进 manifest，或通过环境变量覆盖：

```bash
export PQ_TRT_RUNTIME_CLASS="your_package.your_module.YourTensorRTRuntime"
```

自定义 runtime 只需要实现 `generate(inputs, processor, max_new_tokens) -> str`。

## 常用 Smoke Check

语法和轻量检查：

```bash
python -m compileall src tests
pytest
```

不加载模型，只 dry-run MASQuant 命令构造：

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage calibrate \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --masquant-root "$MASQUANT_ROOT" \
  --work-dir "$WORK_ROOT/dry_run" \
  --calib-jsonl "$CALIB_JSONL" \
  --dry-run
```

## 注意事项

- 不要提交模型权重、数据集、Hugging Face token 或大型 benchmark 输出。
- 如果 CUDA 显存紧张，降低图像分辨率、减少校准样本数，或减小 `--max-new-tokens`。
- GAE 比 attention-proxy 剪枝慢，因为它需要基于梯度进行打分。
