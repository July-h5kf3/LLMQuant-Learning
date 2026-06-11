<center>
    <h1>代码实践</h1>
</center>

这个目录用于存放本项目中的代码复现、实验实现与工程化整理，主要承接论文阅读之后的动手验证部分。当前代码实践集中在多模态大模型量化、VLM 量化 baseline、视觉 token 剪枝与量化协同三个方向。

各子目录相对独立，依赖、模型路径、数据路径和运行脚本请以各自 README 为准。

## 目录总览

| 目录 | 角色 | 当前状态 |
| --- | --- | --- |
| [`LISA/`](LISA/) | LISA / LISA++ 复现、训练、测试与量化实验 | 主实验工作区，已接入多种量化方法和 ReasonSeg / ReasonSeg-Inst 评测 |
| [`VLM_Quant_Baseline/`](VLM_Quant_Baseline/) | VLM / LVLM 量化 baseline | Real Quant 已搭建完成，支持 Qwen2-VL-7B 真实推理加速；覆盖量化搜索、校准、评测、推理和横向对比 |
| [`Prune_Quant/`](Prune_Quant/) | 视觉 token 剪枝与量化协同 baseline | baseline 已搭建完成，支持 GAE 剪枝、prune-then-MASQuant、VLMEvalKit / lmms-eval 集成和单元测试 |

## `LISA/`

`LISA/` 是当前最完整的实验工作区，用于 LISA / LISA++ 复现、训练、测试和量化方法接入。当前包含：

- LISA 主体模型，以及 LLaVA、LLaVA-1.5、SAM 相关依赖代码；
- 基于 YAML 配置的训练脚本 `train_ds.py` 和测试脚本 `test_ds.py`；
- ReasonSeg、ReasonSeg-Inst、语义分割、指代表达分割、VQA 等数据集封装；
- 新版 Transformers 下 tokenizer、LLaMA config、special tokens 的兼容处理；
- `none`、`bnb_4bit`、`bnb_8bit`、`awq`、`gptq`、`hqq`、`quanto`、`smoothquant`、`mbq`、`masquant` 等量化评测路径；
- 量化校准数据构造、量化权重导出 / 合并、实验结果 Markdown 表格输出；
- ReasonSeg-Inst 风格验证数据构造脚本和 COCO segm 指标评测流程。

详细运行方式、环境配置、数据准备和量化配置见 [`LISA/README.md`](LISA/README.md)。

## `VLM_Quant_Baseline/`

`VLM_Quant_Baseline/` 是 VLM / LVLM 量化 baseline，基于 QIG 相关开源实现整理而来，主要用于复现实验行为和后续横向比较。其中 Real Quant 已搭建完成，支持在 Qwen2-VL-7B 上做真实推理加速（TensorRT-LLM W4A16 / W4A8），使用方法见 `REAL_QUANTIZATION.md`。当前包含：

- `main_quant.py`：量化搜索入口；
- `main.py`：评测入口；
- `inference.py`：推理示例入口；
- `configs/`：Qwen2-VL、LLaVA-OneVision、InternVL2 等模型的评测和搜索配置；
- `qmllm/models/`：Qwen2-VL、Qwen2.5-VL、LLaVA-OneVision、LLaVA-1.5、InternVL2、VILA 等模型封装；
- `qmllm/methods/`：AWQ、GPTQ、MBQ、QIG、RTN、SmoothQuant 等方法实现或接入；
- `scripts/`：pseudo W3A16、real GPTQ W3A16、vLLM / lmms-eval 相关评测脚本；
- `REAL_QUANTIZATION.md`：从 pseudo quant baseline 迁移到真实推理加速后端的设计说明。

详细说明见 [`VLM_Quant_Baseline/README.md`](VLM_Quant_Baseline/README.md)。

## `Prune_Quant/`

`Prune_Quant/` 是多模态大模型视觉 token 剪枝与量化推理 baseline，重点关注剪枝和量化的协同实验，整体 baseline 已搭建完成。当前包含：

- GAE-guided visual token pruning；
- GAE 剪枝后接 MASQuant 校准 / 推理的 prune-then-quant 路径；
- 面向 Qwen2-VL / Qwen2.5-VL 的 Hugging Face 模型适配；
- `src/prune_quant_baseline/pruners/`：attention proxy、GAE oracle、token gather、learned compressor 等剪枝模块；
- `src/prune_quant_baseline/quant/`：RTN、GPTQ、AWQ、BitsAndBytes、MASQuant、TensorRT 等量化接口或桥接代码；
- `src/prune_quant_baseline/vlmeval/`：VLMEvalKit 评测 wrapper；
- `remote/`：远端 GPU 机器上的 smoke test、VLMEvalKit 安装和批量运行脚本；
- `tests/`：配置加载、剪枝、量化桥接、compressor training 等单元测试。

中文说明见 [`Prune_Quant/README_zh.md`](Prune_Quant/README_zh.md)，英文说明见 [`Prune_Quant/README.md`](Prune_Quant/README.md)。

## 本地资源说明

代码目录中不保存大体积模型权重、数据集和实验输出。运行实验前需要按对应子项目 README 准备本地路径。

`LISA/` 下这些目录默认不会进入版本管理：

- `LISA/weights/`：模型权重、SAM 权重、量化后 backbone 等；
- `LISA/dataset/`：ReasonSeg、ReasonSeg-Inst、COCO、ADE20K、RefCOCO 等数据；
- `LISA/runs/`：训练日志和 checkpoint；
- `LISA/results/`：测试脚本生成的 Markdown 结果表。

`VLM_Quant_Baseline/` 和 `Prune_Quant/` 中的模型、数据和输出路径通常通过配置文件、命令行参数或环境变量指定，建议放在代码目录之外的服务器工作目录中。

## 后续规划

- 整理 LISA 上不同量化方法的 ReasonSeg / ReasonSeg-Inst 结果表；
- 用 `VLM_Quant_Baseline/` 对 QIG、MBQ、AWQ、GPTQ、SmoothQuant 等方法做横向比较；
- 完善 `Prune_Quant/` 中 GAE 剪枝、MASQuant、VLMEvalKit 评测路径的自动化脚本；
- 逐步统一校准数据格式、实验记录格式和结果汇总方式。
