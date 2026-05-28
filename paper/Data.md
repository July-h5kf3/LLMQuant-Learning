下面按前面论文统计的口径整理 **QIG**：只统计原文中的**非 ablation 主结果/补充实验/效率表**，不放 Table 1、Table 4、Table A1 这类 sensitivity/IG 配置 ablation。QIG 原文主要评估 **W3A16 weight-only** 和 **W4A8 weight-activation** 两类量化设置；主结果表的指标都是 Accuracy (%)，越高越好。([arXiv][1])

## Table 2：三类 LVLM 主结果，Acc↑

| 测试模型               | 量化精度  | 测试方法 | VizWiz↑ | MMMU↑ | ChartQA↑ | AI2D↑ | ScienceQA↑ |  Avg↑ |
| ------------------ | ----- | ---- | ------: | ----: | -------: | ----: | ---------: | ----: |
| LLaVA-OneVision-7B | FP16  | -    |   60.41 | 49.22 |    80.04 | 81.31 |      95.88 | 73.37 |
| LLaVA-OneVision-7B | W3A16 | RTN  |   59.12 | 43.67 |    68.88 | 78.92 |      94.55 | 69.03 |
| LLaVA-OneVision-7B | W3A16 | GPTQ |   54.87 | 42.33 |    73.72 | 76.81 |      92.12 | 67.97 |
| LLaVA-OneVision-7B | W3A16 | AWQ  |   58.65 | 42.89 |    74.08 | 77.92 |      82.20 | 67.15 |
| LLaVA-OneVision-7B | W3A16 | MBQ  |   57.99 | 44.00 |    76.84 | 78.47 |      94.89 | 70.44 |
| LLaVA-OneVision-7B | W3A16 | QIG  |   62.82 | 45.78 |    77.20 | 79.11 |      95.29 | 72.04 |
| LLaVA-OneVision-7B | W4A8  | RTN  |   58.10 | 42.89 |    71.00 | 77.82 |      94.10 | 68.78 |
| LLaVA-OneVision-7B | W4A8  | SQ   |   55.67 | 42.00 |    66.28 | 77.20 |      93.51 | 66.93 |
| LLaVA-OneVision-7B | W4A8  | MBQ  |   58.13 | 44.78 |    74.92 | 78.27 |      94.70 | 70.16 |
| LLaVA-OneVision-7B | W4A8  | QIG  |   59.10 | 45.00 |    74.52 | 78.30 |      94.25 | 70.23 |
| InternVL2-8B       | FP16  | -    |   60.86 | 48.56 |    82.64 | 82.42 |      97.07 | 74.31 |
| InternVL2-8B       | W3A16 | RTN  |   55.95 | 43.89 |    79.24 | 80.51 |      96.28 | 71.17 |
| InternVL2-8B       | W3A16 | GPTQ |   59.79 | 43.11 |    76.40 | 76.65 |      94.30 | 70.05 |
| InternVL2-8B       | W3A16 | AWQ  |   58.14 | 45.56 |    74.42 | 79.47 |      95.88 | 70.70 |
| InternVL2-8B       | W3A16 | MBQ  |   59.33 | 46.02 |    80.04 | 79.66 |      95.93 | 72.20 |
| InternVL2-8B       | W3A16 | QIG  |   59.55 | 46.22 |    80.04 | 79.73 |      96.03 | 72.31 |
| InternVL2-8B       | W4A8  | RTN  |   56.68 | 43.00 |    78.96 | 79.02 |      96.22 | 70.80 |
| InternVL2-8B       | W4A8  | SQ   |   55.56 | 44.78 |    77.96 | 76.59 |      95.88 | 70.15 |
| InternVL2-8B       | W4A8  | MBQ  |   57.36 | 45.67 |    78.00 | 79.47 |      96.38 | 71.38 |
| InternVL2-8B       | W4A8  | QIG  |   58.33 | 47.33 |    78.16 | 79.63 |      96.73 | 72.04 |
| Qwen2-VL-7B        | FP16  | -    |   68.34 | 51.22 |    81.40 | 80.12 |      85.03 | 73.22 |
| Qwen2-VL-7B        | W3A16 | RTN  |   65.02 | 44.67 |    73.64 | 76.33 |      81.06 | 68.14 |
| Qwen2-VL-7B        | W3A16 | GPTQ |   67.73 | 44.44 |    76.20 | 74.87 |      81.76 | 69.00 |
| Qwen2-VL-7B        | W3A16 | AWQ  |   66.24 | 45.89 |    77.08 | 77.53 |      81.01 | 69.56 |
| Qwen2-VL-7B        | W3A16 | MBQ  |   66.62 | 46.48 |    79.18 | 77.81 |      81.85 | 70.15 |
| Qwen2-VL-7B        | W3A16 | QIG  |   67.12 | 47.11 |    77.76 | 77.88 |      81.61 | 70.30 |
| Qwen2-VL-7B        | W4A8  | RTN  |   58.71 | 45.44 |    74.16 | 77.01 |      79.62 | 66.99 |
| Qwen2-VL-7B        | W4A8  | SQ   |   47.60 | 43.78 |    70.88 | 76.07 |      78.98 | 63.46 |
| Qwen2-VL-7B        | W4A8  | MBQ  |   60.17 | 44.89 |    76.92 | 76.49 |      78.93 | 67.48 |
| Qwen2-VL-7B        | W4A8  | QIG  |   58.85 | 46.00 |    76.68 | 77.17 |      80.17 | 67.77 |

## Table 3：InternVL2-26B 大模型结果，Acc↑

| 测试模型          | 量化精度  | 测试方法 | ChartQA↑ | MMMU↑ | VizWiz↑ |
| ------------- | ----- | ---- | -------: | ----: | ------: |
| InternVL2-26B | FP16  | -    |    86.44 | 52.78 |   65.65 |
| InternVL2-26B | W4A8  | MBQ  |    84.44 | 49.78 |   63.51 |
| InternVL2-26B | W4A8  | QIG  |    85.24 | 50.22 |   63.91 |
| InternVL2-26B | W3A16 | MBQ  |    84.48 | 51.67 |   63.33 |
| InternVL2-26B | W3A16 | QIG  |    85.12 | 50.89 |   64.14 |

这张表是为了验证 QIG 是否能扩展到更大的 InternVL2-26B；原文说明 QIG 在 ChartQA 和 VizWiz 上超过 MBQ，在 MMMU 上保持可比表现。([arXiv][1])

[1]: https://arxiv.org/pdf/2603.17809 "Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients"
