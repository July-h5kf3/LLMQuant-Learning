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

VLMEvalKit 评测默认可以直接使用 MASQuant pseudo quant：评测进程启动时加载保存好的 `mas_parameters.pth`，如果提供 CMC 的 `low_rank_adapters*.pt`，也会一起装回模型。不使用 TensorRT 时不需要安装 TensorRT / TensorRT-LLM。

## GAE + MASQuant

prune-then-quant baseline 分为两个阶段。

如果只想跑完整的 Qwen2-VL MASQuant pseudo quant + CMC + VLMEvalKit MME 流程，推荐使用脚本入口：

```bash
cd "$PROJECT_ROOT"
cp remote/run_qwen2vl_masquant_pseudo_mme.example.sh remote/run_qwen2vl_masquant_pseudo_mme.local.sh
```

编辑 `remote/run_qwen2vl_masquant_pseudo_mme.local.sh` 顶部的路径和参数，例如：

- `PROJECT_ROOT`：本项目路径；
- `EXT_ROOT`：外部仓库根目录；
- `MODEL_PATH`：Qwen2-VL 模型路径；
- `WORK_DIR`：MASQuant 中间产物和 VLMEvalKit 输出目录；
- `MASQUANT_ROOT`：`EfficientAI/masquant` 路径；
- `VLMEVALKIT_ROOT`：VLMEvalKit 路径；
- `CALIB_JSONL`：阶段 1 MASQuant 校准用 JSONL；
- `CMC_VISION_JSON` / `CMC_VISION_PREFIX`：CMC 校准用 ShareGPT4V-style JSON 和图片目录；
- `WBITS` / `ABITS` / `NSAMPLES` / `EPOCHS` / `CMC_RANK` / `MAX_NEW_TOKENS` 等模型和实验参数。

然后运行：

```bash
bash remote/run_qwen2vl_masquant_pseudo_mme.local.sh
```

真正的流程逻辑在 `remote/run_masquant_pseudo_pipeline.sh` 中；用户脚本只负责填写地址和参数。若某个阶段已经跑完，可以在用户脚本里设置 `RUN_CALIBRATE=0`、`RUN_CMC=0`、`RUN_INSTALL_VLMEVAL=0` 或 `RUN_VLMEVAL=0` 跳过。默认 `CALIB_RETENTION_RATIO=1.0`、`EVAL_RETENTION_RATIO=1.0` 表示不做 GAE 剪枝，只评测 MASQuant pseudo quant；需要剪枝时把对应值改成例如 `0.5`。

VLMEvalKit 阶段默认使用 `remote/run_vlmeval_smart.py`（`VLMEVAL_SMART_RUNNER=1`）。它会按数据集单独运行，复用已有 prediction xlsx；如果 xlsx 已经存在，会切到 `--mode eval` 只补评分；如果 score 文件已经存在，会直接跳过。MME 和 MMStar 默认使用 `--judge exact_matching`，因此不会调用 GPT-as-judge。`VLMEVAL_DISABLE_OPENAI=1` 仍会在 exact-matching 运行中清理 OpenAI 相关环境变量，避免评分阶段意外访问外部 API。

常用覆盖参数：

```bash
export VLMEVAL_MODE=auto          # auto, all, infer, 或 eval
export VLMEVAL_REUSE=1            # 传给 VLMEvalKit --reuse
export VLMEVAL_FORCE_EVAL=1       # 即使找到 score，也重新补评分
export VLMEVAL_JUDGE=deepseek-v4-pro
export VLMEVAL_EXACT_MATCH_DATASETS="MME MMStar"
```

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

### 使用 pseudo quant 在 VLMEvalKit 上评测

如果目标是“量化一次，评测时直接加载”，在阶段 1 和 CMC 后，直接让 VLMEvalKit 加载保存好的 MASQuant 参数和 CMC 产物：

```bash
export WORK_DIR="$WORK_ROOT/qwen2vl_masquant"
export QWEN2VL_MODEL=/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct
export MASQUANT_ROOT=/home/aistudio/EXT/EfficientAI/masquant
export MASQUANT_RESUME=$(find "$WORK_DIR/masquant_outputs" -name mas_parameters.pth | sort | tail -n 1)
export MASQUANT_ACT_SCALES="$WORK_DIR/act_scales/Qwen2-VL-7B-Instruct-text-vision-128.pt"
export CMC_LOW_RANK="$WORK_DIR/cmc/low_rank_adapters_quantcmc0_rank0.2_vision-audio-only.pt"
export CMC_WHITE="$WORK_DIR/cmc/white_matrix_vision-audio-only.pt"

cd "$PROJECT_ROOT"
python remote/install_vlmeval_pruned_gae.py --vlmeval-root "$VLMEVALKIT_ROOT"

cd "$VLMEVALKIT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export PQ_MODEL_TYPE=qwen2vl
export PQ_MASQUANT_WBITS=4
export PQ_MASQUANT_ABITS=8
export PQ_MAX_NEW_TOKENS=16
export PQ_RETENTION_RATIO=1.0

python run.py --data MME --model Qwen2VL_MASQuant_Pseudo \
  --work-dir "$WORK_DIR/vlmeval_mme_masquant_pseudo" \
  --verbose
```

`Qwen2VL_MASQuant_Pseudo` 不会重新跑 MASQuant 校准，也不会构建 TensorRT engine；它只在评测进程启动时加载 `MASQUANT_RESUME`，并在设置了 `CMC_LOW_RANK` 时加载 CMC 低秩补偿。`PQ_RETENTION_RATIO=1.0` 表示只评测 MASQuant 量化效果，不额外做 GAE 剪枝；如果要评测“MASQuant + GAE 剪枝”，把它改成例如 `0.5`。

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
