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

- [x] [大语言模型量化,PTQ，激活值异常值处理，硬件适配]**Smoothquant**:Accurate and efficient post-training quantization for large language models (ICML 2023)

  本文指出LLMs中激活值Outlier是W8A8量化精度下降的主要原因，而权重相对更容易量化。基于这一观察，作者提出了SmoothQuant，通过一个数学等价的通道缩放变换，将激活值中的量化困难迁移到权重中，从而平滑激活值的异常通道。其核心在于利用$s$对输入激活做除法、对权重做乘法，使得线性层输出保持不变，同时通过迁移强度$\alpha$控制量化难度在权重和激活之间的分配。SmoothQuant不需要训练，只需要少量校准数据统计激活范围，就可以实现较稳定的W8A8量化，并且适配INT8 GEMM硬件。但是本质上它更适合8bit量化，在更低比特下仅靠平滑很难完全解决量化误差问题。

- [x] [大语言模型量化,PTQ，旋转变换，低比特量化]**SpinQuant**: Spinquant: Llm quantization with learned rotations (ICLR 2025)

  这篇文章可以看作是QuaRot这类旋转量化方法的进一步优化。其出发点在于：对权重、激活值和KV-Cache施加正交旋转不会改变Transformer中线性计算的数学等价性，但是可以改变数值分布，使Outlier被扩散到更多维度，从而降低低比特量化难度。不同于直接使用随机Hadamard旋转，SpinQuant提出使用少量校准数据学习旋转矩阵，并通过Cayley Transform等方式保证旋转矩阵的正交性。简单来说，就是把“找一个能让量化后分布更好看的坐标系”也纳入校准优化中。实验上SpinQuant在W4A4KV4等极低比特设置下相比随机旋转和SmoothQuant都有明显提升，说明旋转矩阵本身的选择非常重要。但是它引入了学习旋转和推理时旋转变换的额外成本，工程落地时需要继续考虑算子融合以及硬件友好性。

- [x] [大语言模型量化,PTQ，可学习等价变换，低比特量化]**FlatQuant**: Flatquant: Flatness matters for llm quantization (ICML 2025)

  本文延续了SmoothQuant和QuaRot/SpinQuant中的一个核心思路：在量化之前先通过数学等价变换改变权重和激活值的分布，使其更加适合均匀量化。作者认为仅仅依靠Per-channel Scaling或Hadamard旋转仍然可能留下陡峭且离散的分布，因此提出了FlatQuant(Fast and Learnable Affine Transformation)，为每个线性层学习一个仿射变换来同时拉平权重和激活值分布，从而减少Outlier对量化区间的浪费。为了降低这类变换带来的推理开销，FlatQuant进一步使用Kronecker分解压缩变换矩阵，并将相关操作融合到单个Kernel中。实验上它在W4A4等低比特设置下取得了很强的效果，可以看作是“平滑/旋转”这一类预量化等价变换方法的增强版。不过个人感觉这类方法的关键并不只在理论上的等价变换，而在于最终能不能把额外变换真正融合进高效算子，否则精度收益可能会被推理开销吃掉。

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

- [x] [KV-Cache量化，Google Research]TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate(ICLR' 2026)

  本文主要介绍了一种针对高维向量量化的创新方法，旨在通过大幅度压缩数据规模来优化AI模型推理，KV Cache管理以及向量数据库检索的效率。其核心在于结合了随机旋转技术和最优标量量化器，能在极低的比特位宽下实现接近理论极限的MSE。针对内积检索中的偏置问题，作者设计了一个两阶段架构，利用 1-比特 QJL 变换补偿余数，从而确保了内积估算的无偏性。实验数据表明，该算法在 Llama-3.1 等大语言模型的长文本测试中，仅需 2.5 至 3.5 比特即可保持与全精度近乎一致的性能。此外，相较于传统的乘积量化 (PQ) 技术，TurboQuant在保持高召回率的同时，将索引构建时间降低至接近于零，展现出卓越的加速器友好性。
  
- [x] [多模态大模型的量化，Alibaba]MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models(CVPR' 2026)

  本文旨在解决基于通道级平滑的PTQ方法应用于多模态大模型时面临的一个核心挑战：Smoothing Misalignment。论文通过MAS为每个模态确定一个平滑因子来解决这个问题，并通过CMC方法来解决与之伴随而来的Cross-Modal Computation Invariance问题。
  
- [x] [多模态大模型的量化]MBQ: Modality-Balanced Quantization for Large Vision-Language Models(CVPR'2025)

  这篇文章注意到了将最先进的LLM量化方法直接应用到视觉语言模型时，性能下降显著，并通过实验探明了一个很有可能的原因在于对于不同模态Token的同质化处理，针对这个发现在校准时的优化误差对不同模态按梯度进行加权，从而在一定程度上克服了这个问题。

- [x] [多模态大模型的量化] Fine-Grained Post-Training Quantization for Large Vision Language Modelswith Quantization-Aware Integrated Gradients(CVPR' 2026)

  这篇文章在MBQ的基础上，进一步加深了思考，在MBQ中是对不同模态之间的Token进行加权，但是事实上同一模态内的Token也存在差异，因此可以也进行加权，作者选用了按照量化感知的梯度积分作为加权的依据，在一定程度上取得了不错的效果，但是缺乏理论论证。

- [x] [多模态大模型的量化 teleAI] VLMQ: Efficient Post-Training Quantization for Large Vision-Language Models via Hessian Augmentation

  这篇文章将不同Token的重要性引入到Hessian矩阵的计算中，从而实现了对GPTQ以及GPTAQ等基于Hessian的方法改进，理论分析也较为扎实，但是实验量较少。

- [x] [多模态大模型剪枝量化协同] QAPruner: Quantization-Aware Vision Token Pruning for Multimodal Large Language Models

  这篇文章提到，将基于语义感知的视觉Token剪枝方法直接应用到PTQ后的模型上时，会带来严重的性能下降问题，提出的解决方法是在给Per-Token打分的时候考虑量化的影响

- [x] [多模态大模型剪枝量化协同] Towards Joint Quantization and Token Pruning of Vision-Language Models

  这篇文章提出了一种协作式量化与剪枝框架，和QAPruner的思路类似，就是在剪枝的时候考虑量化带来的影响，且Token分数在量化下给出。实际上本质上还是先量化再剪枝

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
