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

- 在 LISA++ baseline 上实现与论文结果的精度对齐
- 复现并实现 MASQuant

## 参考链接

- LISA 论文: https://arxiv.org/abs/2308.00692
- LISA++ 论文: https://arxiv.org/abs/2312.17240
- LISA 原始仓库: https://github.com/dvlab-research/LISA
- MASQuant 论文: https://arxiv.org/abs/2603.04800
- MASQuant 仓库: https://github.com/alibaba/EfficientAI/tree/main/masquant
