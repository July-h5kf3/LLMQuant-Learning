# QIG Baseline

本目录是 `LLMQuant-Learning` 中用于研究 VLM/LVLM 量化的一个 baseline 仓库，基于 QIG 原始开源实现整理而来。

该 baseline 主要用于对比和复现实验中的视觉语言模型量化流程，包含模型封装、校准数据处理、量化搜索、伪量化评测与简单推理等代码。当前目录保留原项目的核心代码结构，方便后续在统一实验框架中接入、修改和横向比较。

## Repository Role

- 作为 VLM/LVLM 量化方向的 baseline 实现纳入本仓库。
- 用于和其他量化方法、实验设置或自研模块进行对比。
- 保留原始实现中的配置、脚本和示例资源，便于复现实验行为。

## Structure

- `main_quant.py`: 量化搜索入口。
- `main.py`: 评测入口。
- `inference.py`: 推理示例入口。
- `REAL_QUANTIZATION.md`: 将 pseudo quant baseline 迁移到真实推理加速后端的设计说明。
- `configs/`: 不同模型与量化设置的配置文件。
- `scripts/run_w3a16_eval.sh`: 使用本机已准备好的权重和样本对比 pseudo W3A16 与 real GPTQ W3A16 推理速度。
- `scripts/run_qig_w3a16_real_vllm_eval.sh`: 使用 real GPTQ W3A16 checkpoint 和 vLLM 后端运行正式 lmms-eval/QIG 任务。
- `qmllm/`: 模型、数据、校准与量化方法实现。
- `3rdparty/`: 第三方依赖项目说明。

## W3A16 Speed Check

在 AutoDL 服务器的实际工作目录中运行：

```bash
cd /root/autodl-tmp/LLMQuant-Learning/Code/VLM_Quant_Baseline
bash scripts/run_w3a16_eval.sh
```

脚本会用同一份 `inference/question.json` 样本分别测试：

- pseudo QIG W3A16：FP16 权重 + QIG scale + `--pseudo_quant`
- real GPTQ W3A16：AutoGPTQ W3A16 checkpoint + vLLM `quantization=gptq`

输出位于 `/root/autodl-tmp/eval/QIG/w3a16_speed/`，每条结果旁边会生成 `.time` 文件记录总耗时。

## Real W3A16 Eval

在 AutoDL 服务器运行正式 lmms-eval 任务：

```bash
cd /root/autodl-tmp/LLMQuant-Learning/Code/VLM_Quant_Baseline
bash scripts/run_qig_w3a16_real_vllm_eval.sh
```

默认任务为 `mmmu_val,vizwiz_vqa_val,chartqa,ai2d,scienceqa_img`。可先做 smoke test：

```bash
LIMIT=1 TASKS=mmmu_val bash scripts/run_qig_w3a16_real_vllm_eval.sh
```

## Upstream Reference

该 baseline 来源于以下论文和开源实现：

**Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients**  
Ziwei Xiang, Fanhu Zeng, Hongjian Fang, Rui-Qi Wang, Renxing Chen, Yanan Zhu, Yi Chen, Peipei Yang, Xu-Yao Zhang

- Paper: https://arxiv.org/abs/2603.17809
- Original method name: QIG
