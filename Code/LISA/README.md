# LISA / LISA++ 复现与量化实验

本目录是 LISA / LISA++ 相关实验的主要工作区。当前代码已经覆盖基线训练、分割任务评测、ReasonSeg-Inst 风格数据构造，以及多种 LLM / MLLM 量化方法在 LISA backbone 上的接入与评测。

## 当前状态

- 基线模型：以 LISA++ / LISA 系列模型为主要实验对象，代码中保留了 LLaVA、LLaVA-1.5 和 SAM 相关实现。
- 训练入口：`train_ds.py`，通过 `configs/train_ds.yaml` 控制训练数据、LoRA、损失权重、日志和 checkpoint。
- 测试入口：`test_ds.py`，通过 `configs/test_ds.yaml` 控制测试数据、精度、checkpoint 和量化方法。
- 数据集：支持 ReasonSeg、ReasonSeg-Inst、语义分割、指代表达分割和 VQA 相关数据封装。
- 量化方法：当前 `test_ds.py` 支持 `none`、`bnb_4bit`、`bnb_8bit`、`awq`、`gptq`、`hqq`、`quanto`、`smoothquant`、`mbq`、`masquant`。
- 兼容处理：`utils/tokenizer_compat.py` 和 `model/compat_transformers_431/` 用于处理新版 Transformers 与原始 LISA / LLaVA 代码之间的 tokenizer、LLaMA 配置和 special tokens 差异。

## 目录说明

- `configs/`：训练、测试和量化实验配置。
- `configs/quant/`：各量化方法的参数配置，例如 AWQ、GPTQ、HQQ、SmoothQuant、MBQ、MASQuant。
- `model/`：LISA 主体模型、LLaVA / LLaVA-1.5 代码、SAM 代码和 Transformers 兼容补丁记录。
- `quantization/`：量化方法实现、校准数据构造、量化 backbone 加载、权重导出和合并工具。
- `utils/`：数据集、tokenizer、conversation、指标和通用工具。
- `train_ds.py`：训练脚本。
- `test_ds.py`：测试 / 推理 / 量化评测脚本。
- `prepare_reasonseg_inst_val.py`：基于 COCO val2017 和多模态模型生成 ReasonSeg-Inst 风格验证数据的辅助脚本。
- `exp_utils.py`：将评测结果写成 Markdown 表格，默认输出到 `results/`。

## 环境安装

本目录使用 `uv` 管理 Python 环境，Python 版本为 3.11。

```bash
cd Code/LISA
uv sync
```

如需进入虚拟环境：

```bash
source .venv/bin/activate
```

当前 `pyproject.toml` 中固定了部分关键依赖版本，例如：

- `torch==2.9.1`
- `torchvision==0.24.1`
- `transformers==5.3.0`
- `triton==3.5.1`
- `peft==0.7.1`

AWQ 相关依赖放在 optional dependency 中，需要时可额外安装：

```bash
uv sync --extra awq
```

## 本地文件准备

权重、数据集、日志和结果文件默认不进入版本管理，需要在本地自行准备或生成：

- `weights/`：LISA++ / LISA 权重、SAM 权重、量化后 backbone 等。
- `dataset/`：ReasonSeg、ReasonSeg-Inst、COCO、ADE20K、RefCOCO 等数据。
- `runs/`：训练日志和 checkpoint。
- `results/`：测试脚本生成的 Markdown 结果表。

配置文件中目前使用的是服务器绝对路径，例如 `/root/autodl-tmp/...`。在新机器上运行前，需要先修改：

- `configs/train_ds.yaml` 中的 `version`、`vision_tower`、`vision_pretrained`、`dataset_dir`、`log_base_dir`；
- `configs/test_ds.yaml` 中的 `version`、`vision_tower`、`vision_pretrained`、`dataset_dir`、`resume`；
- `configs/quant/*.yaml` 中的 `model_path`、量化输出路径、校准数据路径和权重路径。

## 训练

默认训练配置：

```bash
uv run python train_ds.py --config configs/train_ds.yaml
```

训练脚本会：

- 加载 LISA 模型、tokenizer、vision tower 和 SAM；
- 按配置构造 `HybridDataset`；
- 使用 LoRA 配置训练语言模型相关模块，并可训练 mask decoder；
- 在开启验证时计算 gIoU / cIoU；
- 将最优 checkpoint 保存到 `log_base_dir/exp_name/ckpt_model/checkpoint.pt`。

常用配置项在 `configs/train_ds.yaml` 中，包括：

- `dataset`、`sample_rates`、`reason_seg_data`、`val_dataset`；
- `epochs`、`steps_per_epoch`、`batch_size`、`grad_accumulation_steps`；
- `precision`、`lr`、`gradient_checkpointing`；
- `lora_r`、`lora_alpha`、`lora_target_modules`；
- `ce_loss_weight`、`dice_loss_weight`、`bce_loss_weight`。

## 测试与评测

默认测试配置：

```bash
uv run python test_ds.py --config configs/test_ds.yaml
```

可以在命令行覆盖 checkpoint、测试集和随机种子：

```bash
uv run python test_ds.py \
  --config configs/test_ds.yaml \
  --resume runs/your_exp/ckpt_model/checkpoint.pt \
  --test_dataset "ReasonSeg|val" \
  --seed 3407
```

支持的测试集写法：

- `ReasonSeg|val`
- `ReasonSeg|test`
- `ReasonSegInst|val`

对于 ReasonSeg 类任务，脚本会输出 cIoU、gIoU、峰值显存和平均 forward 时间。对于 ReasonSeg-Inst，脚本会按 COCO segm 评测方式输出 mAP、AP50、AP75、AP-small、AP-medium、AP-large，并同样记录显存和耗时。

评测结果默认写入：

```text
results/{model_name}_{dataset_name}.md
```

## 量化评测

量化入口统一在 `configs/test_ds.yaml` 中配置：

```yaml
quant_method: "hqq"
quant_config: "quant/hqq.yaml"
```

当前支持的方法和配置文件包括：

| quant_method | 配置文件 | 说明 |
| --- | --- | --- |
| `none` | 空 | 不启用量化 |
| `bnb_4bit` | `quant/bnb_4bit.yaml` | BitsAndBytes 4-bit |
| `bnb_8bit` | `quant/bnb_8bit.yaml` | BitsAndBytes LLM.int8 |
| `awq` | `quant/awq.yaml` | AWQ 校准、导出并回填 LISA backbone |
| `gptq` | `quant/gptq.yaml` | GPTQ 校准、导出并回填 LISA backbone |
| `hqq` | `quant/hqq.yaml` | HQQ backbone 量化路径 |
| `quanto` | `quant/quanto.yaml` | Quanto runtime quantization |
| `smoothquant` | `quant/smoothquant.yaml` / `quant/smoothquant_w4a16.yaml` | SmoothQuant 风格平滑与伪量化 |
| `mbq` | `quant/mbq.yaml` / `quant/mbq_w4a8.yaml` | Modality-Balanced Quantization |
| `masquant` | `quant/masquant.yaml` / `quant/masquant_w4a16.yaml` | Modality-Aware Smoothing Quantization |

AWQ、GPTQ、SmoothQuant、MBQ、MASQuant 等方法会根据配置检查已有量化产物；缺失时会先构造校准数据并生成所需权重或 scale 文件，再进入评测流程。

部分方法当前会强制使用 `fp16` 执行：

- `hqq`
- `quanto`
- `smoothquant`
- 开启 `wa_quant` 的 `mbq`
- 开启 `wa_quant` 的 `masquant`

## ReasonSeg-Inst 数据构造

`prepare_reasonseg_inst_val.py` 用于从 COCO val2017 构造 ReasonSeg-Inst 风格验证集。默认会读取：

- `dataset/COCO2017/annotations/instances_val2017.json`
- `dataset/COCO2017/val2017`

示例：

```bash
export AUTODL_API_KEY=your_api_key

uv run python prepare_reasonseg_inst_val.py \
  --dataset-root dataset \
  --output-root dataset/ReasonSeg-inst \
  --target-pairs 1800 \
  --split val \
  --resume
```

脚本默认使用 `qwen3.6-plus`，通过 `AUTODL_API_KEY` 读取 API key。`ReasonSegInstDataset` 会读取 `dataset/ReasonSeg-inst/val.json` 和 `dataset/ReasonSeg-inst/val/`，因此建议像上面的示例一样显式指定 `--output-root dataset/ReasonSeg-inst`。调试时可使用 `--dry-run`、`--limit-images` 或 `--max-images`。

## Transformers 兼容记录

原始 LISA / LLaVA 代码对 Transformers 版本比较敏感。当前仓库已经针对新版 Transformers 做过兼容处理，重点包括：

- 显式处理 LISA 所需的 `[SEG]` token；
- 兼容不同版本 tokenizer 对 added tokens 的注册方式；
- 保留 `model/compat_transformers_431/LISA_transformers_upgrade_notes.md` 作为后续排查参考。

## TODO

- [x] LISA++ baseline 训练与 ReasonSeg 评测流程整理。
- [x] 修复新版 Transformers 下 tokenizer、LLaMA config 和 special tokens 兼容问题。
- [x] 接入 4-bit / 8-bit 基础量化评测路径。
- [x] 接入 AWQ、GPTQ、HQQ、Quanto、SmoothQuant、MBQ、MASQuant 相关实验配置与加载路径。
- [x] 增加 ReasonSeg-Inst 风格验证集构造与 COCO segm 指标评测。
- [ ] 系统整理不同量化方法在 ReasonSeg / ReasonSeg-Inst 上的结果表。
- [ ] 继续完善 MASQuant、MBQ 等多模态量化方法的实验对齐和消融配置。
- [ ] 接入大语言模型推理框架 vLLM-Omini，方便后续方法测评。

## 参考链接

- LISA 论文: https://arxiv.org/abs/2308.00692
- LISA++ 论文: https://arxiv.org/abs/2312.17240
- LISA 原始仓库: https://github.com/dvlab-research/LISA
- MASQuant 论文: https://arxiv.org/abs/2603.04800
- MASQuant 仓库: https://github.com/alibaba/EfficientAI/tree/main/masquant
