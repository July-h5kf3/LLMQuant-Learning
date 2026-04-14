<center>
    <h1>LLM Quant Learning</h1>
</center>

本仓库用于记录我围绕大模型量化方向的学习、论文阅读、阶段性汇报和实验复现。当前重点已经从单纯阅读扩展到多模态大模型量化实践，代码侧主要围绕 LISA / LISA++ 的复现、评测和量化方法接入展开。

## 仓库结构

- `paper/`：论文阅读清单、阅读笔记和配图资料。仓库中主要维护 `Note.md`、`README.md` 与 `figure/` 下的配图；PDF 原文按 `.gitignore` 规则本地保存，不作为主要同步内容。
- `blog/`：专题化整理后的博客或长文草稿，目前包含 Hessian 矩阵、Transformer 位置编码、Triton 等内容。
- `Code/`：代码实践区。当前主要内容是 `Code/LISA/`，用于 LISA / LISA++ 复现、训练、测试以及多种量化方法实验。
- `Week/`：周报、阶段性汇报和临时思考材料。本目录按 `.gitignore` 规则作为本地资料区，不要求全部进入版本管理。

## 当前代码进展

`Code/LISA/` 目前已经形成一个相对完整的 LISA 实验工作区，包含：

- LISA / LISA++ 相关模型代码，以及 LLaVA、LLaVA-1.5、SAM 依赖代码；
- 基于 YAML 配置的训练脚本 `train_ds.py` 和测试脚本 `test_ds.py`；
- ReasonSeg、ReasonSeg-Inst、语义分割、指代表达分割、VQA 等数据集封装；
- 适配较新 Transformers 版本的 tokenizer 与 LLaMA 兼容处理；
- 多种量化评测路径，包括 `none`、`bnb_4bit`、`bnb_8bit`、`awq`、`gptq`、`hqq`、`quanto`、`smoothquant`、`mbq`、`masquant`；
- 量化校准数据构造、量化权重导出 / 合并、实验结果 Markdown 表格输出。

详细运行方式见 [`Code/LISA/README.md`](Code/LISA/README.md)。

## 当前学习重点

- 梳理 PTQ、QAT、混合精度、权重量化、激活量化、KV Cache 量化等核心方法；
- 关注多模态大模型量化中的模态差异、校准样本构造和跨模态计算一致性问题；
- 以 LISA / LISA++ 为实验对象，对齐基线精度并接入 MASQuant、MBQ、SmoothQuant 等方法；
- 记录复现过程中遇到的工程问题，例如 Transformers 版本兼容、特殊 token 注册、量化后权重回填等。

## 使用说明

阅读资料优先从以下文件进入：

- [`paper/README.md`](paper/README.md)：论文阅读路线和阶段性清单；
- [`paper/Note.md`](paper/Note.md)：更细的论文笔记；
- [`blog/`](blog)：专题笔记；
- [`Code/README.md`](Code/README.md)：代码实践区说明；
- [`Code/LISA/README.md`](Code/LISA/README.md)：LISA 复现与量化实验说明。

代码实验需要额外准备本地数据和权重。以下目录不会纳入版本管理：

- `Code/LISA/weights/`
- `Code/LISA/dataset/`
- `Code/LISA/runs/`
- `Code/LISA/results/`

## 更新方式

仓库会继续按学习和实验推进同步：

- 本周阅读或复现的内容；
- 关键结论、踩坑记录与个人理解；
- LISA 相关实验配置、量化方法接入和评测结果；
- 后续待补的方向与计划。
