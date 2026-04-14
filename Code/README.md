<center>
    <h1>代码实践</h1>
</center>

这个目录用于存放本项目中的代码复现、实验实现与工程化整理，主要承接论文阅读之后的动手验证部分。当前代码实践重点是多模态大模型量化，核心工作集中在 `LISA/`。

## 当前内容

### `LISA/`

`LISA/` 是当前主要实验工作区，用于 LISA / LISA++ 复现、训练、测试和量化方法接入。当前已经包含：

- LISA 主体模型实现，以及 LLaVA、LLaVA-1.5、SAM 相关依赖代码；
- 基于 YAML 配置的训练脚本 `train_ds.py` 和测试脚本 `test_ds.py`；
- ReasonSeg、ReasonSeg-Inst、语义分割、指代表达分割、VQA 等数据集封装；
- 新版 Transformers 下 tokenizer、LLaMA config、special tokens 的兼容处理；
- 多种量化评测路径，包括 `none`、`bnb_4bit`、`bnb_8bit`、`awq`、`gptq`、`hqq`、`quanto`、`smoothquant`、`mbq`、`masquant`；
- 量化校准数据构造、量化权重导出 / 合并、实验结果 Markdown 表格输出；
- ReasonSeg-Inst 风格验证数据构造脚本和 COCO segm 指标评测流程。

详细运行方式、环境配置、数据准备和量化配置见 [`LISA/README.md`](LISA/README.md)。

## 本地文件说明

`LISA/` 下的部分实验资源体积较大，默认不会进入版本管理：

- `LISA/weights/`：模型权重、SAM 权重、量化后 backbone 等；
- `LISA/dataset/`：ReasonSeg、ReasonSeg-Inst、COCO、ADE20K、RefCOCO 等数据；
- `LISA/runs/`：训练日志和 checkpoint；
- `LISA/results/`：测试脚本生成的 Markdown 结果表。

## 后续规划

- 持续补充 LISA 相关量化实验结果；
- 对 MASQuant、MBQ、SmoothQuant 等多模态量化方法做更系统的对齐和消融；
- 根据后续论文阅读进展，在 `Code/` 下新增独立复现或实验子目录。
