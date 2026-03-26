<center>
    <h1>论文阅读笔记综述</h1>
</center>

这里主要保留论文阅读清单、阅读笔记与配图资料。

PDF 原文不再直接存放在仓库中，后续统一在本地管理；仓库内仅维护阅读进度、摘要与思考，便于持续同步与版本管理。

更加详细的论文阅读笔记以及思考见 `Note.md` 以及博客。

### 博客填坑
- [x] Hessian矩阵系列串讲

- [ ] 线性代数基础(进阶)

- [ ] 常见网络层量化处理方法及处理理由

- [ ] (初阶汇报)基于Hessian矩阵的权重重要性分析的量化方法

### 入门

这部分论文无法选作文献汇报

- [x] [模型量化,PTQ,基于Hessian矩阵的权重感知]**AdaRound**:Up or Down? Adaptive Rounding for Post-Training Quantization (ICML 2020)

​	本篇文章作者首先从数学角度证明了在模型量化过程中，直接将浮点数进行四舍五入round到最近定点数的方法并不是精度最优的。并且通过了一个简单的实验验证了猜想，随后基于此作者进行一系列的数学推导和数学近似推导除了最终的优化目标:最小化由于量化在预激活值中引入的均方误差，从而提出了自适应的Round方法:AdaRound.这种方法在进行量化时，自适应地决定将浮点值转到最近右定点还是左定点值。AdaRound可以在不需要QAT or finetune的情况下仅使用少量无标签的校准数据在精度上达到SOTA，甚至4bit量化也可以保留较好的精度。

- [x] [大语言模型量化,PTQ，硬件适配]**ZeroQuant**:Zeroquant: Efficient and affordable post-training quantization for large-scale transformers (NeurIPS 2022）

  ​	这篇文章指出，低比特量化在大型 Transformer 架构模型中精度受限的主要原因是激活值和权重矩阵的值分布方差较大。针对这一问题，提出了 ZeroQuant 方案。该方案主要包括：对权重采用 Group-wise 量化、对激活值采用 Token-wise 量化，这种方法既能适配硬件架构，又能保持较高的精度；同时，通过 Layer-wise 知识蒸馏方法来减少量化带来的精度损失。但是存在实验模型规模较小的问题。

- [x] [大语言模型量化,PTQ,基于Hessian矩阵的权重感知]**GPTQ**:Gptq: Accurate post-training quantization for generative pre-trained transformers (ICLR 2023)

神作。本篇文章对OBC中提出的OBQ方法进行了优化，提出了贪心地对权重进行量化不能带来明显增益且加大了计算存储开销，因此采用了顺序量化地方法，并针对这个方法采用了更加高效的Hessian矩阵求逆方法，并采用Cholesky分解稳定数值。并提出了lazy batch-update方法解决IO带来的瓶颈问题。

- [x] [模型量化,PTQ]**AdaQuant**:Accurate post training quantization with small calibration sets (ICML 2021)

​	本篇文章的主要贡献在于提出了一个基于小数据集（校验集）的训练后量化方法AdaQuant，AdaQuant通过提出一个block/layer-wise的损失函数，通过在校验集上的训练学习量化参数(重点包括了一个最优的权重扰动，类似于AdaRound来避免四舍五入的不足),实现了减少量化的精度损失；提出了基于PI(整数规划)的bit精度分配方案，但是并没有解释精确损失的累加合理性；提出量化对BN融合造成的统计量偏移问题，并提出了PN(Para-Normalization)来解决这个问题。并在Bert-base网络上实现了不到1%的损失(4-8bit)

- [ ] **Smoothquant**:Accurate and efficient post-training quantization for large language models (ICML 2023)

- [ ] **SpinQuant**: Spinquant: Llm quantization with learned rotations (ICLR 2025)

- [ ] **Q-dit:Q-dit**: Accurate post-training quantization for diffusion transformers (CVPR 2025)

- [ ] **SVDQuant:Svdquant**: Absorbing outliers by low-rank components for 4-bit diffusion models (ICLR 2025)
- [ ] **Mpq-dm:Mpq-dm**: Mixed precision quantization for extremely low bit diffusion models (AAAI 2025)

### 进阶

这部分更多收集一些arXiv上比较好的工作以及一些会议的Spotlight和Oral以及Best Paper

- [ ] [视频生成模型的量化]ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformer(ICLR'2025)

- [ ] [视觉生成模型的量化]PAROAttention: Pattern-Aware ReOrdering for Efficient Sparse and Quantized Attention in Visual Generation Models

- [x] [大语言模型的量化,PTQ，基于Hessian矩阵的权重感知]OWQ: Outlier-Aware Quantization for Efficient Fine-tuning and Inference of Large Language Models(AAAI 2024 Oral)

本文提出了一个异常感知的权重量化方法OWQ，利用LLMs中的异常激活值挑选出Weak Column，对其采用全精度的方式在牺牲很小的性能的情况下提升了巨大的精度。此外为进一步提升其性能做了一定的硬件适配并提出了一个基于OWQ的WTC方案，简单来说就是在OWQ量化模型上微调只更新Weak Column的参数。

- [ ] [大语言模型的量化，硬件适配问题，PTQ]ZeroQuant-HERO: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers
- [ ] [大语言模型的量化,量化框架,PTQ]OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models(ICLR'2024 Spotlight)
- [ ] [大语言模型的量化,量化框架,3Bit量化]TEQUILA: TRAPPING-FREE TERNARY QUANTIZA TION FOR LARGE LANGUAGE MODEL
- [ ] [RWKV架构的量化] RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization(ICML'2025)
- [ ] [MoE架构的量化] MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance(ICML'2025)
- [ ] [大语言模型量化] OSTQuant: Refining Large Language Models via Optimizing Data Distribution(ICLR'2025)
- [ ] [大语言模型量化,基于Hessian矩阵分析的权重感知]HAWQV3: Dyadic Neural Network Quantization(ICML'2021)
- [ ] [大语言模型量化,华为盘古"小模型"]CBQ: Cross-Block Quantization for Large Language Models(ICLR'2025)


### 前沿

这部分更多是收集一些大模型厂商的Technical Report，个人认为在资本趋利性下，对显存的”压榨“会做到极致

- [ ] **DeepSeekV3**：DeepSeek-V3 Technical Report
- [ ] **HunYuan**: AngelSlim

### 综述

这部分是一些综述，从综述入手一个领域是一个很Nice的选择，不一定要是paper可以是中文期刊甚至博客~

- [ ] A White Paper on Neural Network Quantization(这个是Begining of Begining 是高通写的量化过程描述，从这个开始is best)

- [ ] Scaling Laws for Precision

- [ ] Scaling Laws for Floating Point Quantization Training(腾讯混元)





Awesome系类：[pprp/Awesome-LLM-Quantization: Awesome list for LLM quantization](https://github.com/pprp/Awesome-LLM-Quantization)

[混合精度量化的paper\ List](https://zhuanlan.zhihu.com/p/365272572)(年代比较久远，挑一些顶会的来看吧)


