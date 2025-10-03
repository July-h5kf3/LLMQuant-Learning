<center>
    <h1>论文阅读笔记综述</h1>
</center>

这里是论文阅读的List，分为入门，进阶，前沿，综述三个板块,每篇文章阅读后，我会简单做一些小的介绍和笔记~

后续会更加细粒度的对文章内容进行区分，类似于Awesome系列~

更加详细的论文阅读笔记以及思考见Note.md

### 入门

这部分论文无法选作文献汇报

- [ ] **AdaRound**:Up or Down? Adaptive Rounding for Post-Training Quantization (ICML 2020)

- [ ] **ZeroQuant**:Zeroquant: Efficient and affordable post-training quantization for large-scale transformers (NeurIPS 2022）

- [ ] **GPTQ**:Gptq: Accurate post-training quantization for generative pre-trained transformers (ICLR 2023)

- [x] **AdaQuant**:Accurate post training quantization with small calibration sets (ICML 2021)

​	本篇文章的主要贡献在于提出了一个基于小数据集（校验集）的训练后量化方法AdaQuant，AdaQuant通过提出一个block/layer-wise的损失函数，通过在校验集上的训练学习量化参数(重点包括了一个最优的权重扰动，类似于AdaRound来避免四舍五入的不足),实现了减少量化的精度损失；提出了基于PI(整数规划)的bit精度分配方案，但是并没有解释精确损失的累加合理性；提出量化对BN融合造成的统计量偏移问题，并提出了PN(Para-Normalization)来解决这个问题。并在Bert-base网络上实现了不到1%的损失(4-8bit)

- [ ] **Smoothquant**:Accurate and efficient post-training quantization for large language models (ICML 2023)

- [ ] **SpinQuant**: Spinquant: Llm quantization with learned rotations (ICLR 2025)

- [ ] **Q-dit:Q-dit**: Accurate post-training quantization for diffusion transformers (CVPR 2025)
- [ ] **SVDQuant:Svdquant**: Absorbing outliers by low-rank components for 4-bit diffusion models (ICLR 2025)
- [ ] **Mpq-dm:Mpq-dm**: Mixed precision quantization for extremely low bit diffusion models (AAAI 2025)

### 进阶

这部分更多收集一些arXiv上比较好的工作以及一些会议的Spotlight和Oral以及Best Paper

### 前沿

这部分更多是收集一些大模型厂商的Technical Report，个人认为在资本趋利性下，对显存的”压榨“会做到极致

- [ ] **DeepSeekV3**：DeepSeek-V3 Technical Report

### 综述

这部分是一些综述，从综述入手一个领域是一个很Nice的选择，不一定要是paper可以是中文期刊甚至博客~

- [ ] A Survey on Model Compression for Large Language Model(TACL 2023)





Awesome系类：[pprp/Awesome-LLM-Quantization: Awesome list for LLM quantization](https://github.com/pprp/Awesome-LLM-Quantization)

[混合精度量化的paper\ List](https://zhuanlan.zhihu.com/p/365272572)(年代比较久远，挑一些顶会的来看吧)



