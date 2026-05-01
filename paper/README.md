<center>
    <h1>论文阅读笔记综述</h1>
</center>

这里主要保留论文阅读清单、阅读笔记与配图资料。

PDF 原文不再直接存放在仓库中，后续统一在本地管理；仓库内仅维护阅读进度、摘要与思考，便于持续同步与版本管理。

更加详细的论文阅读笔记以及思考见 `Note.md` 以及博客。

## Papers

统一整理为 Awesome 风格表格，不再区分入门、进阶、前沿与综述；保留阅读状态、个人评价与简要总结。

| Tags | Paper | Venue | Read | 评价 | 总结 |
| --- | --- | --- | --- | --- | --- |
| 模型量化, PTQ, Hessian 感知 | [**AdaRound**: Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568) | ICML 2020 | Yes | 很经典 | 证明直接 round 并非最优，转而最小化预激活 MSE，自适应决定向上/向下舍入；少量无标签校准数据下也能显著提升 PTQ 精度。 |
| 大语言模型量化, PTQ, 硬件适配 | [**ZeroQuant**: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861) | NeurIPS 2022 | Yes | 工程导向强 | 通过权重 group-wise 和激活 token-wise 量化兼顾硬件友好与精度，并用 layer-wise distillation 补偿损失；问题是实验模型规模偏小。 |
| 大语言模型量化, PTQ, Hessian 感知 | [**GPTQ**: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | ICLR 2023 | Yes | 神作 | 对 OBQ 做了系统工程化优化：顺序量化、稳定的 Hessian 逆近似、Cholesky 分解、lazy batch-update，一举解决低比特 LLM PTQ 的效果和效率问题。 |
| 模型量化, PTQ | [**AdaQuant**: Accurate Post Training Quantization with Small Calibration Sets](https://proceedings.mlr.press/v139/hubara21a/hubara21a.pdf) | ICML 2021 | Yes | 很扎实 | 用小校准集学习量化参数和权重扰动，结合 block/layer-wise 损失降低 PTQ 误差；还讨论了 bit 分配与 BN 融合统计偏移问题。 |
| 大语言模型量化, PTQ, 激活异常值, 硬件适配 | [**SmoothQuant**: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) | ICML 2023 | Yes | 很实用 | 将激活中的 outlier 通过等价缩放迁移到权重侧，显著提升 W8A8 稳定性；更适合 8bit，进一步降比特时能力有限。 |
| 大语言模型量化, PTQ, 旋转变换, 低比特 | [**SpinQuant**: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406) | ICLR 2025 | Yes | 很有启发 | 在 QuaRot 类方法上更进一步，用校准数据学习正交旋转矩阵以扩散 outlier，在 W4A4KV4 等极低比特设置下效果突出，但额外变换成本需要工程消化。 |
| 大语言模型量化, PTQ, 可学习等价变换, 低比特 | [**FlatQuant**: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426) | ICML 2025 | Yes | 值得关注 | 学习仿射变换同时拉平权重和激活分布，并用 Kronecker 分解与 kernel 融合降低额外开销；核心挑战仍是能否真正高效落地。 |
| Diffusion Transformer 量化 | [**Q-DiT**: Accurate Post-Training Quantization for Diffusion Transformers](https://arxiv.org/abs/2406.17343) | CVPR 2025 | No | - | - |
| Diffusion 模型量化, 低秩分解 | [**SVDQuant**: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007) | ICLR 2025 | No | - | - |
| Diffusion 模型量化, 混合精度 | [**MPQ-DM**: Mixed Precision Quantization for Extremely Low Bit Diffusion Models](https://arxiv.org/abs/2412.11549) | AAAI 2025 | No | - | - |
| 视频生成模型量化 | [**ViDiT-Q**: Efficient and Accurate Quantization of Diffusion Transformer](https://arxiv.org/abs/2406.02540) | ICLR 2025 | No | - | - |
| 视觉生成模型量化, 稀疏注意力 | [**PAROAttention**: Pattern-Aware ReOrdering for Efficient Sparse and Quantized Attention in Visual Generation Models](https://arxiv.org/abs/2506.16054) | arXiv | No | - | - |
| 大语言模型量化, PTQ, Hessian 感知 | [**OWQ**: Outlier-Aware Quantization for Efficient Fine-tuning and Inference of Large Language Models](https://arxiv.org/abs/2306.02272) | AAAI 2024 Oral | Yes | 思路直接有效 | 抓住异常激活对应的 weak column，对少量关键列保留全精度，大幅缓解量化误差；进一步还延伸到只更新 weak column 的微调方案。 |
| 大语言模型量化, PTQ, 硬件适配 | [**ZeroQuant-HERO**: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers](https://arxiv.org/abs/2310.17723) | - | No | - | - |
| 大语言模型量化, 量化框架, PTQ | [**OmniQuant**: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137) | ICLR 2024 Spotlight | No | - | - |
| 大语言模型量化, 量化框架, 3Bit | [**TEQUILA**: Trapping-Free Ternary Quantization for Large Language Model](https://arxiv.org/abs/2509.23809) | - | No | - | - |
| RWKV 架构量化 | [**RWKVQuant**: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization](https://arxiv.org/abs/2505.03803) | ICML 2025 | No | - | - |
| MoE 架构量化 | [**MoEQuant**: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance](https://arxiv.org/abs/2505.03804) | ICML 2025 | No | - | - |
| 大语言模型量化 | [**OSTQuant**: Refining Large Language Models via Optimizing Data Distribution](https://arxiv.org/abs/2501.13987) | ICLR 2025 | No | - | - |
| 大语言模型量化, Hessian 分析 | [**HAWQV3**: Dyadic Neural Network Quantization](https://arxiv.org/abs/2011.10680) | ICML 2021 | No | - | - |
| 大语言模型量化, 华为盘古 | [**CBQ**: Cross-Block Quantization for Large Language Models](https://arxiv.org/abs/2312.07950) | ICLR 2025 | No | - | - |
| KV-Cache 量化, Google Research | [**TurboQuant**: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) | ICLR 2026 | Yes | 很强 | 面向高维向量在线量化，结合随机旋转与近最优标量量化器，在极低比特下接近理论 MSE 极限；对 KV Cache 和向量检索都很有价值。 |
| 多模态大模型量化, Alibaba | [**MASQuant**: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](https://arxiv.org/abs/2603.04800) | CVPR 2026 | Yes | 问题定义清晰 | 识别出 MLLM 中 smoothing misalignment 问题，为不同模态设置独立平滑因子，并处理跨模态计算等价性。 |
| 多模态大模型量化 | [**MBQ**: Modality-Balanced Quantization for Large Vision-Language Models](https://arxiv.org/abs/2412.19509) | CVPR 2025 | Yes | 很自然的改进 | 发现直接套用 LLM 量化到 VLM 时，不同模态 token 被同质对待会伤害性能；通过模态加权校准误差缓解该问题。 |
| 多模态大模型量化 | [**Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients**](https://arxiv.org/abs/2603.17809) | CVPR 2026 | Yes | 想法进一步细化 | 在 MBQ 基础上继续细分到模态内 token 级重要性，用量化感知积分梯度做加权，效果不错，但理论支撑偏弱。 |
| 多模态大模型量化, teleAI | [**VLMQ**: Efficient Post-Training Quantization for Large Vision-Language Models via Hessian Augmentation](https://arxiv.org/abs/2508.03351) | - | Yes | 理论比较扎实 | 将不同 token 的重要性显式引入 Hessian 估计，对 GPTQ/GPTAQ 一类方法做了自然扩展，但实验规模还不算充分。 |
| 多模态大模型剪枝量化协同 | [**QAPruner**: Quantization-Aware Vision Token Pruning for Multimodal Large Language Models](https://arxiv.org/abs/2604.02816) | - | Yes | 很有现实意义 | 指出先 PTQ 再直接做语义剪枝会明显掉点，因此在 token 打分阶段显式考虑量化影响。 |
| 多模态大模型剪枝量化协同 | [**Towards Joint Quantization and Token Pruning of Vision-Language Models**](https://arxiv.org/abs/2604.17320) | - | Yes | 与 QAPruner 同脉络 | 强调剪枝与量化协同建模，不过本质上依然更接近“先量化、再在量化约束下做剪枝”。 |
| Technical Report | [**DeepSeek-V3 Technical Report**](https://arxiv.org/abs/2412.19437) | Technical Report | No | - | - |
| Technical Report | [**AngelSlim**](https://arxiv.org/abs/2602.21233) | Technical Report | No | - | - |
| 综述, 白皮书 | [**A White Paper on Neural Network Quantization**](https://arxiv.org/abs/2106.08295) | White Paper | No | 起点很好 | 高通写的量化过程综述，适合作为入门中的入门。 |
| 综述 | [**Scaling Laws for Precision**](https://arxiv.org/abs/2411.04330) | - | No | - | - |
| 综述, 腾讯混元 | [**Scaling Laws for Floating Point Quantization Training**](https://arxiv.org/abs/2501.02423) | - | No | - | - |





Awesome系类：[pprp/Awesome-LLM-Quantization: Awesome list for LLM quantization](https://github.com/pprp/Awesome-LLM-Quantization)

[混合精度量化的paper\ List](https://zhuanlan.zhihu.com/p/365272572)(年代比较久远，挑一些顶会的来看吧)
