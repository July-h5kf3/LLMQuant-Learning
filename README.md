<center>
    <h1>LLM Quant Learning</h1>
</center>

本仓库用于记录大模型量化方向的学习、论文阅读、专题整理和代码实验。当前内容已经从 LLM PTQ 方法阅读扩展到多模态大模型量化、视觉 token 剪枝与量化协同等实验实践。

仓库整体分为三类内容：

- `paper/`：论文阅读清单、阅读笔记、方法总结、配图和阶段性表格。
- `blog/`：从阅读和复现中整理出的专题长文草稿。
- `Code/`：量化、剪枝和多模态模型实验代码。

## 仓库结构

- `paper/`：论文阅读主目录。当前维护 `README.md`、`Note.md`、`Data.md`、`figure/` 和少量输出表格；PDF 原文按 `.gitignore` 规则本地保存，不作为主要同步内容。
- `blog/`：专题化笔记，目前包括 Hessian 矩阵、Transformer 位置编码、Triton、低精度浮点数等内容。
- `Code/`：代码实践区，包含 LISA 量化实验、VLM 量化 baseline、视觉 token 剪枝与量化协同 baseline。
- `Week/`：周报、阶段性汇报和临时材料，本目录按 `.gitignore` 规则作为本地资料区。

## 代码实践

当前 `Code/` 下包含三个主要子项目，各自环境和运行方式相对独立：

### `Code/LISA/`

LISA / LISA++ 复现与量化实验工作区。当前覆盖：

- LISA 主体模型，以及 LLaVA、LLaVA-1.5、SAM 相关代码；
- `train_ds.py` 和 `test_ds.py` 两个训练、测试入口；
- ReasonSeg、ReasonSeg-Inst、语义分割、指代表达分割、VQA 等数据封装；
- 新版 Transformers 下 tokenizer、LLaMA config、special tokens 的兼容处理；
- `none`、`bnb_4bit`、`bnb_8bit`、`awq`、`gptq`、`hqq`、`quanto`、`smoothquant`、`mbq`、`masquant` 等量化评测路径；
- ReasonSeg-Inst 风格验证集构造和 COCO segm 指标评测。

详细说明见 [`Code/LISA/README.md`](Code/LISA/README.md)。

### `Code/VLM_Quant_Baseline/`

VLM / LVLM 量化 baseline，基于 QIG 相关开源实现整理。当前主要用于横向对比和复现实验，**Real Quant 已搭建完成，支持在 Qwen2-VL-7B 上做真实推理加速**，包含：

- `main_quant.py` 量化搜索入口、`main.py` 评测入口、`inference.py` 推理示例；
- Qwen2-VL、Qwen2.5-VL、LLaVA-OneVision、LLaVA-1.5、InternVL2、VILA 等模型封装；
- AWQ、GPTQ、MBQ、QIG、RTN、SmoothQuant 等方法实现或接入；
- 校准数据处理、pseudo quant 评测；
- 基于 TensorRT-LLM 的 real quant 导出与推理（Qwen2-VL-7B W4A16 / W4A8），以及 real GPTQ W3A16 / vLLM 评测脚本，使用方法见 `REAL_QUANTIZATION.md`。

详细说明见 [`Code/VLM_Quant_Baseline/README.md`](Code/VLM_Quant_Baseline/README.md)。

### `Code/Prune_Quant/`

多模态大模型视觉 token 剪枝与量化协同 baseline，**整体 baseline 已搭建完成**。当前支持：

- GAE-guided visual token pruning；
- GAE 剪枝后接 MASQuant 的 prune-then-quant 实验；
- Qwen2-VL / Qwen2.5-VL 风格 Hugging Face 输入；
- VLMEvalKit wrapper、远端运行脚本、smoke test 和单元测试；
- RTN、GPTQ、AWQ、BitsAndBytes、MASQuant、TensorRT 相关量化接口或桥接代码。

详细说明见 [`Code/Prune_Quant/README_zh.md`](Code/Prune_Quant/README_zh.md) 和 [`Code/Prune_Quant/README.md`](Code/Prune_Quant/README.md)。

## 论文与笔记

论文阅读路线和阶段性清单见 [`paper/README.md`](paper/README.md)，更细的阅读记录见 [`paper/Note.md`](paper/Note.md)。

当前重点包括：

- LLM PTQ：AdaRound、GPTQ、SmoothQuant、OWQ、SpinQuant、FlatQuant、SliderQuant、OSAQ、SERQ 等；
- 低精度浮点量化：ARCQuant 等围绕 NVFP4 / MXFP4 等 FP4 格式的工作；
- VLM / MLLM 量化：MASQuant、MBQ、QIG、VLMQ、VEQ 等；
- 剪枝与量化协同：QAPruner、Joint Quantization and Token Pruning、GAE 相关视觉 token 压缩方法；
- Kernel 与工程基础：Triton、Hessian 近似、量化误差重建、推理后端适配。

专题文章见 [`blog/README.md`](blog/README.md)。

## 使用入口

- 阅读论文路线：[`paper/README.md`](paper/README.md)
- 查看详细笔记：[`paper/Note.md`](paper/Note.md)
- 查看专题文章：[`blog/README.md`](blog/README.md)
- 查看代码区总览：[`Code/README.md`](Code/README.md)
- 运行 LISA 实验：[`Code/LISA/README.md`](Code/LISA/README.md)
- 运行 VLM 量化 baseline：[`Code/VLM_Quant_Baseline/README.md`](Code/VLM_Quant_Baseline/README.md)
- 运行剪枝量化 baseline：[`Code/Prune_Quant/README_zh.md`](Code/Prune_Quant/README_zh.md)

## 本地资源说明

模型权重、数据集、训练日志、实验输出和周报材料通常不进入版本管理，需要在本地或服务器上自行准备。尤其是：

- `Code/LISA/weights/`
- `Code/LISA/dataset/`
- `Code/LISA/runs/`
- `Code/LISA/results/`
- `Week/`

不同代码子项目的依赖并不完全一致，建议进入对应子目录后按其 README 单独安装环境和准备路径。

## 后续方向

- 持续补充 LLM / VLM 量化论文阅读和方法对比；
- 系统整理 LISA、QIG baseline、Prune_Quant 的实验结果；
- 对 MASQuant、MBQ、QIG、SmoothQuant、GPTQ 等方法做更统一的配置和横向比较；
- 继续推进视觉 token 剪枝与量化协同实验。
