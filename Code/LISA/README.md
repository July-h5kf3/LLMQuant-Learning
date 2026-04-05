# LISA / LISA++ 量化实验仓库

## 仓库介绍

本仓库基于 LISA++ 思路整理与扩展，当前主要用于搭建多种量化方法在 LISA 系列模型上的实现与实验基线。

当前代码以 `LISA++` 为 baseline，围绕训练、测试与后续量化实验进行实现和整理。

## 安装

本项目使用 `uv` 管理环境与依赖，在 `Code/LISA` 目录下直接安装即可：

```bash
uv sync
```

如需进入虚拟环境，可执行：

```bash
source .venv/bin/activate
```

## 运行

训练：

```bash
uv run python train_ds.py --config configs/train_ds.yaml
```

测试 / 推理：

```bash
uv run python test_ds.py --config configs/test_ds.yaml
```

训练与测试的主要参数均通过 `configs/train_ds.yaml` 和 `configs/test_ds.yaml` 配置。

## TODO

- [x] 目前模型性能依旧高度绑定Transformers版本，会导致很多可以调库的方法无法被使用。测试发现若使用版本5.4.0则无法复现出论文中的效果
  
  出现这个现象的主要原因是新版的Transformers库中llava的configs与4.31.0中的存在差异，新版的AutoTokenizer不兼容，需要显式调用LlamaTokenizer，另外就是目前`[SEG]`等added_tokens在不同版本下注册方式不一致。修复这三个问题就可以了，目前仓库已经修复了这个问题，实现了在ReasonSeg-Sem数据集上的指标对齐。另外我在/Code/LISA/model/compat_transformers_431中放了AI总结的经验，以后再遇到类似的问题可以作为一个参考。
- [ ] 在 LISA++ baseline 上实现与论文结果的精度对齐:
  1. 在Reasoning instance segmentation下的结果
  2. 在4-bit/8-bit量化下的对齐
- [ ] 复现并实现 MASQuant

## 参考链接

- LISA 论文: https://arxiv.org/abs/2308.00692
- LISA++ 论文: https://arxiv.org/abs/2312.17240
- LISA 原始仓库: https://github.com/dvlab-research/LISA
- MASQuant 论文: https://arxiv.org/abs/2603.04800
- MASQuant 仓库: https://github.com/alibaba/EfficientAI/tree/main/masquant
