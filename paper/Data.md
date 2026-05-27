下面按 **SplitQ 原文的非 ablation 主结果表**整理：只取 **Table 1、Table 2、Table 7**，不取 Table 3–6 和 Appendix 的 ablation 表。论文实验说明：Qwen2.5-VL 使用 MMMU、OCRBench、TextVQA、SEED-Bench、VizWiz、ScienceQA；LLaVA-v1.5 使用 SEED-I、VizWiz、SciQA；`-` 表示原文写明“no meaningful results are obtained”。([arXiv][1])

## Table 1：Qwen2.5-VL-3B / 7B 主结果

| 测试模型          | 量化精度   | 测试方法   | MMMU↑ | SEED↑ | OCRBench↑ | VizWiz↑ | ScienceQA↑ | TextVQA↑ | Avg↑ |
| ------------- | ------ | ------ | ----: | ----: | --------: | ------: | ---------: | -------: | ---: |
| Qwen2.5-VL-3B | W16A16 | FP16   |  42.2 |  69.9 |      79.3 |    69.1 |       81.9 |     77.9 | 70.0 |
| Qwen2.5-VL-3B | W4A8   | SQ     |  25.6 |  55.7 |      66.9 |    57.5 |       72.1 |     63.9 | 56.9 |
| Qwen2.5-VL-3B | W4A8   | MBQ    |  41.2 |  58.2 |      66.9 |    65.0 |       76.7 |     73.4 | 63.5 |
| Qwen2.5-VL-3B | W4A8   | MASQ   |  46.7 |  59.7 |      67.2 |    62.7 |       77.9 |     69.2 | 63.9 |
| Qwen2.5-VL-3B | W4A8   | SplitQ |  46.3 |  69.7 |      79.1 |    67.7 |       82.4 |     77.4 | 70.4 |
| Qwen2.5-VL-3B | W4A4   | SQ     |  23.3 |   0.0 |       0.0 |     0.0 |        0.0 |      0.0 |  3.9 |
| Qwen2.5-VL-3B | W4A4   | MBQ    |  25.0 |   0.0 |       0.0 |     0.0 |        0.0 |      0.0 |  4.2 |
| Qwen2.5-VL-3B | W4A4   | MASQ   |  26.7 |   0.0 |       7.7 |     0.0 |        0.0 |      0.0 |  5.7 |
| Qwen2.5-VL-3B | W4A4   | SplitQ |  43.7 |  69.3 |      78.8 |    67.6 |       81.5 |     77.0 | 69.6 |
| Qwen2.5-VL-3B | W3A3   | SQ     |     - |     - |         - |       - |          - |        - |    - |
| Qwen2.5-VL-3B | W3A3   | MBQ    |     - |     - |         - |       - |          - |        - |    - |
| Qwen2.5-VL-3B | W3A3   | MASQ   |     - |     - |         - |       - |          - |        - |    - |
| Qwen2.5-VL-3B | W3A3   | SplitQ |  39.8 |  67.2 |      74.5 |    63.4 |       70.6 |     71.2 | 64.5 |
| Qwen2.5-VL-3B | W3A2   | SplitQ |  33.1 |  51.3 |      57.0 |    54.5 |       49.1 |     50.2 | 49.2 |
| Qwen2.5-VL-7B | W16A16 | FP16   |  46.7 |  73.0 |      83.8 |    70.8 |       88.4 |     82.9 | 74.3 |
| Qwen2.5-VL-7B | W4A8   | SQ     |  37.8 |  62.7 |      70.2 |    61.5 |       83.3 |     71.1 | 64.4 |
| Qwen2.5-VL-7B | W4A8   | MBQ    |  43.3 |  67.7 |      74.1 |    64.3 |       86.0 |     74.8 | 68.3 |
| Qwen2.5-VL-7B | W4A8   | MASQ   |  43.4 |  69.5 |      72.8 |    66.4 |       85.7 |     77.0 | 69.1 |
| Qwen2.5-VL-7B | W4A8   | SplitQ |  49.1 |  73.2 |      83.5 |    68.7 |       88.1 |     82.6 | 74.2 |
| Qwen2.5-VL-7B | W4A4   | SQ     |  24.8 |   0.0 |       0.2 |     0.0 |        0.7 |      0.0 |  4.3 |
| Qwen2.5-VL-7B | W4A4   | MBQ    |  26.7 |   3.0 |       0.5 |     0.0 |        0.9 |      0.0 |  5.2 |
| Qwen2.5-VL-7B | W4A4   | MASQ   |  25.0 |   0.6 |      13.2 |     0.0 |        7.1 |      0.4 |  7.7 |
| Qwen2.5-VL-7B | W4A4   | SplitQ |  46.9 |  72.6 |      83.0 |    68.4 |       87.9 |     82.5 | 73.5 |
| Qwen2.5-VL-7B | W3A3   | SQ     |     - |     - |         - |       - |          - |        - |    - |
| Qwen2.5-VL-7B | W3A3   | MBQ    |     - |     - |         - |       - |          - |        - |    - |
| Qwen2.5-VL-7B | W3A3   | MASQ   |     - |     - |         - |       - |          - |        - |    - |
| Qwen2.5-VL-7B | W3A3   | SplitQ |  43.9 |  71.1 |      79.6 |    63.7 |       80.3 |     78.5 | 69.5 |
| Qwen2.5-VL-7B | W3A2   | SplitQ |  35.6 |  57.3 |      61.5 |    50.1 |       55.6 |     62.6 | 53.7 |

数据来自原文 Table 1；原文说明 Qwen2.5-VL 上比较了 SmoothQuant/SQ、MBQ、MASQuant/MASQ 与 SplitQ，并且 W3A3/W3A2 下多数 baseline 无法得到有效输出。([arXiv][1])

## Table 2：LLaVA-v1.5-7B / 13B 主结果

| 测试模型           | 量化精度 | 测试方法    | SEED-I↑ | VizWiz↑ | SciQA↑ | Avg↑ |
| -------------- | ---- | ------- | ------: | ------: | -----: | ---: |
| LLaVA-v1.5-7B  | FP16 | DuQuant |    66.2 |    54.3 |   70.0 | 63.5 |
| LLaVA-v1.5-7B  | FP16 | QVLM    |    66.2 |    54.3 |   70.0 | 63.5 |
| LLaVA-v1.5-7B  | FP16 | QSVD    |    66.2 |    54.3 |   70.0 | 63.5 |
| LLaVA-v1.5-7B  | FP16 | SplitQ  |    66.2 |    54.3 |   70.0 | 63.5 |
| LLaVA-v1.5-7B  | W4A8 | DuQuant |    54.4 |    50.6 |   55.3 | 53.4 |
| LLaVA-v1.5-7B  | W4A8 | QVLM    |    46.1 |    48.7 |   53.2 | 49.3 |
| LLaVA-v1.5-7B  | W4A8 | QSVD    |    57.8 |    53.5 |   63.6 | 58.3 |
| LLaVA-v1.5-7B  | W4A8 | SplitQ  |    65.3 |    56.6 |   70.3 | 64.1 |
| LLaVA-v1.5-7B  | W4A4 | DuQuant |    51.5 |    49.8 |   54.8 | 52.0 |
| LLaVA-v1.5-7B  | W4A4 | QVLM    |    37.2 |    48.9 |   53.1 | 46.4 |
| LLaVA-v1.5-7B  | W4A4 | QSVD    |    55.1 |    53.6 |   57.7 | 55.5 |
| LLaVA-v1.5-7B  | W4A4 | SplitQ  |    64.8 |    54.6 |   69.4 | 62.9 |
| LLaVA-v1.5-7B  | W3A3 | DuQuant |       - |       - |      - |    - |
| LLaVA-v1.5-7B  | W3A3 | QVLM    |       - |       - |      - |    - |
| LLaVA-v1.5-7B  | W3A3 | QSVD    |       - |       - |      - |    - |
| LLaVA-v1.5-7B  | W3A3 | SplitQ  |    61.5 |    57.2 |   64.9 | 61.2 |
| LLaVA-v1.5-7B  | W3A2 | DuQuant |       - |       - |      - |    - |
| LLaVA-v1.5-7B  | W3A2 | QVLM    |       - |       - |      - |    - |
| LLaVA-v1.5-7B  | W3A2 | QSVD    |       - |       - |      - |    - |
| LLaVA-v1.5-7B  | W3A2 | SplitQ  |    54.4 |    55.7 |   52.3 | 54.1 |
| LLaVA-v1.5-13B | FP16 | DuQuant |    68.3 |    57.3 |   74.7 | 66.8 |
| LLaVA-v1.5-13B | FP16 | QVLM    |    68.3 |    57.3 |   74.7 | 66.8 |
| LLaVA-v1.5-13B | FP16 | QSVD    |    68.3 |    57.3 |   74.7 | 66.8 |
| LLaVA-v1.5-13B | FP16 | SplitQ  |    68.3 |    57.3 |   74.7 | 66.8 |
| LLaVA-v1.5-13B | W4A8 | DuQuant |    66.1 |    56.5 |   72.3 | 65.0 |
| LLaVA-v1.5-13B | W4A8 | QVLM    |    64.2 |    55.7 |   68.4 | 62.8 |
| LLaVA-v1.5-13B | W4A8 | QSVD    |    66.8 |    56.9 |   75.0 | 66.2 |
| LLaVA-v1.5-13B | W4A8 | SplitQ  |    68.3 |    57.7 |   74.7 | 66.9 |
| LLaVA-v1.5-13B | W4A4 | DuQuant |    64.6 |    55.3 |   67.2 | 62.4 |
| LLaVA-v1.5-13B | W4A4 | QVLM    |    48.3 |    50.9 |   65.0 | 54.7 |
| LLaVA-v1.5-13B | W4A4 | QSVD    |    67.0 |    56.8 |   67.8 | 63.9 |
| LLaVA-v1.5-13B | W4A4 | SplitQ  |    67.9 |    57.2 |   74.0 | 66.4 |
| LLaVA-v1.5-13B | W3A3 | DuQuant |       - |       - |      - |    - |
| LLaVA-v1.5-13B | W3A3 | QVLM    |       - |       - |      - |    - |
| LLaVA-v1.5-13B | W3A3 | QSVD    |       - |       - |      - |    - |
| LLaVA-v1.5-13B | W3A3 | SplitQ  |    64.6 |    56.6 |   69.5 | 63.6 |
| LLaVA-v1.5-13B | W3A2 | DuQuant |       - |       - |      - |    - |
| LLaVA-v1.5-13B | W3A2 | QVLM    |       - |       - |      - |    - |
| LLaVA-v1.5-13B | W3A2 | QSVD    |       - |       - |      - |    - |
| LLaVA-v1.5-13B | W3A2 | SplitQ  |    59.2 |    57.0 |   61.8 | 59.3 |

数据来自原文 Table 2；该表进一步在 LLaVA-v1.5 7B/13B 上比较 DuQuant、Q-VLM、QSVD 和 SplitQ。([arXiv][1])


[1]: https://arxiv.org/html/2605.19929v1 "Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models"
