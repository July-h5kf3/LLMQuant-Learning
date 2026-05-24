# Codex 实现任务说明：Prune + Quant Baseline for MLLM Visual Token Compression

> 目标：先搭建一个**可扩展、可远程测试**的 Prune + Quant baseline 代码框架。  
> 约束：**本地只进行代码编写，不下载模型，不准备数据，不跑真实模型测试**。模型、数据、评测脚本都在远程机器上，后续会补充远程 SSH 与密码。当前阶段只完成最基础代码与清晰接口。

---

## 1. 背景与目标

本项目希望基于论文 `arXiv:2506.01097`：

- Paper URL: <https://arxiv.org/abs/2506.01097>
- 方向：Multimodal LLM 的 task-related visual token pruning / token compression。
- 目标方法：在视觉 token 进入 LLM 之前进行 token selection / pruning，减少 prefill、decode、KV cache 负担。
- 本项目目标：搭建一个后续可扩展为论文方法的 baseline，包括：
  1. Vanilla inference baseline；
  2. Quant-only baseline；
  3. Prune-only baseline；
  4. Prune + Quant baseline。

因为论文没有开源代码，所以第一阶段不要求完整复现论文全部训练细节，而是先搭建一个稳定、清晰、可测试的基础框架。

---

## 2. 最重要的工程约束

Codex 在本地实现时必须遵守以下约束：

1. **不要在本地下载任何模型权重。**
   - 不要调用 `from_pretrained()` 下载远程模型。
   - 不要执行任何会触发 Hugging Face 下载的命令。
   - 不要假设本地有模型文件。

2. **不要在本地准备或下载数据集。**
   - 不要下载 benchmark。
   - 不要下载图片、视频、JSONL 数据。
   - 数据路径全部通过 CLI 参数或环境变量传入。

3. **不要在本地跑真实模型测试。**
   - 可以写 unit test / smoke test 文件，但不要执行。
   - 可以写 synthetic tensor test，用于远程或 CI 后续运行。
   - 不要运行需要 GPU、模型权重或真实数据的脚本。

4. **不要写死远程路径。**
   - 所有模型路径、数据路径、输出路径都必须通过配置文件或命令行参数注入。
   - 后续远程 SSH 信息会另行提供。

5. **不要把远程密码写进代码、配置或日志。**
   - 后续测试时如果需要 SSH 密码，只在交互式命令或临时环境中使用。
   - 仓库内不保存任何密钥、token、密码。

6. **优先保证代码结构清晰，而不是一次性复现全部论文细节。**
   - 第一阶段重点是 attention-proxy pruning、token gather、quant loader、模型 adapter 接口。
   - GAE oracle 和 learned compressor 可以先写 skeleton + TODO，但接口要稳定。

---

## 3. 推荐实现范围：第一阶段 MVP

第一阶段只需要完成以下能力：

### 3.1 必须完成

- Python package 结构；
- 通用 pruning 接口；
- Attention-proxy visual token scoring；
- Top-K visual token selection；
- 保序 token pruning / gather；
- 对 `inputs_embeds`、`attention_mask`、`position_ids` 的通用裁剪函数；
- Quantized model loader 的基础封装；
- LLaVA-OneVision / Qwen2-VL adapter skeleton；
- CLI 脚本 skeleton；
- YAML 配置文件模板；
- synthetic tensor tests 文件，但当前本地不运行。

### 3.2 可以先写 skeleton，不要求完整可用

- GAE / Grad-Attention oracle relevance；
- Learned 1D Conv compressor；
- Compressor training script；
- VLMEvalKit 集成；
- latency profiling；
- video-specific adapter 细节。

---

## 4. 预期仓库结构

请创建或调整为如下结构：

```text
prune_quant_baseline/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore

  configs/
    llava_onevision_7b_image.yaml
    qwen2vl_7b_image.yaml
    qwen2vl_7b_video.yaml

  src/
    prune_quant_baseline/
      __init__.py

      core/
        __init__.py
        datatypes.py
        logging_utils.py
        tensor_utils.py
        config.py

      pruners/
        __init__.py
        base.py
        attention_proxy.py
        token_gather.py
        gae_oracle.py
        learned_compressor.py

      compressors/
        __init__.py
        conv1d_compressor.py
        train_compressor.py

      quant/
        __init__.py
        loaders.py
        bnb.py
        gptq.py

      models/
        __init__.py
        base_adapter.py
        llava_onevision_hf.py
        qwen2vl_hf.py

      scripts/
        __init__.py
        run_infer_pruned.py
        run_generate_labels.py
        run_train_compressor.py
        run_quantize.py
        profile_latency.py

  tests/
    test_token_gather.py
    test_attention_proxy.py
    test_config_loading.py

  remote/
    README_REMOTE_TESTING.md
    run_remote_smoke.sh
    run_remote_eval.sh
```

备注：如果当前仓库已有结构，尽量兼容现有结构，不要强行大规模重构。没有现有结构时按上面新建。

---

## 5. 核心设计原则

### 5.1 Pruning 必须物理删除 token

不能只是把视觉 token 的 `attention_mask` 置零。正式测速时，必须真的从序列中 gather 出保留 token：

- 删除部分 visual token；
- 保留全部 text / system / instruction / answer prompt tokens；
- Top-K 后必须恢复原始 sequence order；
- 不要按 score 从高到低重排 token。

### 5.2 所有 pruning 方法共用一个 token gather 后端

`attention_proxy`、`gae_oracle`、`learned_compressor` 只负责输出每个 visual token 的 score。  
真正的 token selection 和 sequence gather 应该由统一函数实现，避免每个 pruner 重复写逻辑。

### 5.3 模型相关逻辑放到 adapter

不同 MLLM 找 visual token 位置的方式不同。因此：

- 通用 pruning 逻辑不要依赖某一个模型类；
- LLaVA-OneVision、Qwen2-VL 的特殊逻辑放到 `models/*_hf.py`；
- 如果 adapter 暂时无法完整实现，先写明确的 TODO 和报错信息。

### 5.4 Quantization 先只量化 LLM 主体

第一阶段先支持：

- bitsandbytes 4-bit / 8-bit loader；
- GPTQ / AWQ loader skeleton；
- 通过 ignore modules 跳过 vision tower / projector 的配置接口。

不要在第一阶段尝试复杂的 vision tower quantization。

---

## 6. 关键数据结构

请在 `src/prune_quant_baseline/core/datatypes.py` 中实现类似结构。

```python
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class VisualTokenMeta:
    """Sequence-level metadata for visual token pruning."""

    visual_indices: torch.LongTensor
    # shape: [num_visual_tokens]
    # global sequence positions of visual tokens before pruning

    text_indices: Optional[torch.LongTensor] = None
    # optional: global sequence positions of instruction/text tokens used for scoring

    keep_indices: Optional[torch.LongTensor] = None
    # optional: global sequence positions kept after pruning

    image_grid_thw: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    rope_deltas: Optional[torch.Tensor] = None


@dataclass
class PruneResult:
    """Output of sequence pruning."""

    inputs_embeds: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    keep_indices: torch.LongTensor
    kept_visual_indices: torch.LongTensor
    visual_scores: torch.Tensor
```

可以根据实际需要扩展字段，但不要让 dataclass 依赖具体模型权重。

---

## 7. Pruner 基类接口

在 `src/prune_quant_baseline/pruners/base.py` 中定义：

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
import torch

from prune_quant_baseline.core.datatypes import VisualTokenMeta


class VisualTokenPruner(ABC):
    """Base interface for visual token scoring."""

    @abstractmethod
    def score(
        self,
        *,
        attentions: Optional[Any] = None,
        hidden_states: Optional[Any] = None,
        meta: VisualTokenMeta,
        **kwargs,
    ) -> torch.Tensor:
        """
        Return one scalar score per visual token.

        Returns:
            scores: torch.Tensor, shape [num_visual_tokens]
        """
        raise NotImplementedError
```

---

## 8. Attention-proxy pruning

### 8.1 目标

实现一个不需要训练、不需要 backward 的 baseline：

- 从第一层 attention 中取 instruction/text tokens 对 visual tokens 的 attention；
- 对 head 和 text query positions 求平均；
- 得到每个 visual token 的 score；
- Top-K 保留。

### 8.2 输入假设

`attentions[0]` 常见 shape：

```text
[B, num_heads, seq_len, seq_len]
```

第一阶段只要求支持 `B=1`。如果 `B>1`，可以抛出清晰错误，后续再扩展。

### 8.3 参考实现逻辑

在 `src/prune_quant_baseline/pruners/attention_proxy.py` 中实现：

```python
score = attn_first_layer[0, :, text_indices, :][:, :, visual_indices].mean(dim=(0, 1))
```

注意 PyTorch 高级索引时 shape 可能不符合直觉，建议实现时写得更稳：

```python
# attn: [B, H, S, S]
# query_idx: [Nt]
# visual_idx: [Nv]
sub = attn[0].index_select(dim=1, index=query_idx).index_select(dim=2, index=visual_idx)
# sub: [H, Nt, Nv]
scores = sub.mean(dim=(0, 1))
# scores: [Nv]
```

### 8.4 边界处理

必须处理：

- `attentions is None`；
- `attentions` 为空；
- `visual_indices` 为空；
- `text_indices` 未提供；
- attention shape 不是 4D；
- batch size 不是 1。

错误信息要清楚，方便远程测试时定位。

---

## 9. Token selection 与 gather

请在 `src/prune_quant_baseline/pruners/token_gather.py` 中实现通用函数。

### 9.1 `select_topk_visual_tokens`

```python
def select_topk_visual_tokens(
    visual_indices: torch.LongTensor,
    scores: torch.Tensor,
    retention_ratio: float,
    min_keep: int = 1,
) -> torch.LongTensor:
    """
    Select top-K visual token global indices by score.

    Important:
    - top-k is based on score;
    - returned indices must be sorted by original sequence order;
    - do not return indices sorted by score.
    """
```

要求：

- `retention_ratio` 范围 `(0, 1]`；
- `k = max(min_keep, ceil(num_visual_tokens * retention_ratio))`；
- `k <= num_visual_tokens`；
- 返回 global sequence positions；
- 返回结果按原始 sequence 顺序升序排列。

### 9.2 `build_keep_indices`

```python
def build_keep_indices(
    seq_len: int,
    visual_indices: torch.LongTensor,
    kept_visual_indices: torch.LongTensor,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """
    Return global sequence indices to keep.

    Keep all non-visual tokens and only selected visual tokens.
    Return indices sorted in original sequence order.
    """
```

逻辑：

- 创建 `[0, 1, ..., seq_len-1]`；
- 标记 visual positions；
- 对 visual positions 只保留 kept；
- 对 non-visual positions 全保留；
- 返回排序后的 `keep_indices`。

### 9.3 `gather_sequence_tensors`

```python
def gather_sequence_tensors(
    *,
    inputs_embeds: torch.Tensor,
    keep_indices: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Physically gather sequence tensors according to keep_indices.

    inputs_embeds: [B, S, D]
    attention_mask: optional [B, S]
    position_ids: optional [B, S] or [B, 3, S] or model-specific shape
    """
```

要求：

- 第一阶段只要求 `B=1`，但代码尽量写成支持 batch；
- `inputs_embeds` 沿 dim=1 gather；
- `attention_mask` 沿最后一个 sequence dim gather；
- `position_ids` 支持以下常见 shape：
  - `[B, S]`；
  - `[B, 3, S]`；
  - `[3, B, S]`；
- 对未知 shape 抛出清楚错误；
- 对 Qwen2-VL，优先保留原始 position_ids 的 selected positions，不要简单重新编号。

---

## 10. GAE oracle skeleton

在 `src/prune_quant_baseline/pruners/gae_oracle.py` 中先实现 skeleton。

目标：后续用于两阶段 oracle pruning。

预期流程：

1. 第一次 no-grad generate 得到答案；
2. 第二次 teacher-forcing 回放 prompt + answer；
3. 开启 `output_attentions=True`、`use_cache=False`；
4. 使用 eager attention，避免 flash-attention / sdpa 导致拿不到完整 attention map 或 gradient；
5. 对 answer token log-prob 做 backward；
6. 使用 attention × gradient 进行 rollout，得到 visual token relevance；
7. 用 relevance 做 Top-K pruning；
8. 再次 compressed inference。

第一阶段要求：

- 写清楚类与方法；
- 参数、返回值、TODO 完整；
- 如果调用未完成逻辑，抛出 `NotImplementedError`；
- 不要写半可用但静默错误的实现。

建议类名：

```python
class GAEOraclePruner(VisualTokenPruner):
    ...
```

---

## 11. Learned compressor skeleton

论文部署版使用轻量 compressor 预测 visual relevance。第一阶段先实现最小 PyTorch module，不要求训练可复现论文结果。

### 11.1 Conv1D compressor

在 `src/prune_quant_baseline/compressors/conv1d_compressor.py` 中实现：

```python
import torch
import torch.nn as nn


class DWConvBlock(nn.Module):
    ...


class RelevanceCompressor(nn.Module):
    """
    Input:
        x: [B, Nv], first-layer text-to-visual attention proxy
        mask: optional [B, Nv]
    Output:
        probability distribution over visual tokens, [B, Nv]
    """
```

要求：

- depthwise separable conv1d；
- 5 层左右即可；
- 输出 `[B, Nv]`；
- 支持 `mask`；
- masked positions logits 设为极小值；
- 输出 softmax distribution。

### 11.2 LearnedCompressorPruner

在 `src/prune_quant_baseline/pruners/learned_compressor.py` 中实现 pruner wrapper：

- 输入 first-layer attention proxy；
- 调用 compressor；
- 输出 visual token scores；
- 支持从 checkpoint 加载 compressor；
- 如果 checkpoint 不存在，抛出清楚错误。

### 11.3 训练脚本 skeleton

在 `src/prune_quant_baseline/compressors/train_compressor.py` 中实现训练入口 skeleton：

- 读取 label JSONL / PT 文件；
- 支持 variable length padding；
- 使用 KL loss；
- 保存 checkpoint；
- 第一阶段可以只写结构和 TODO，不要求真实训练。

---

## 12. Quantization loader

在 `src/prune_quant_baseline/quant/loaders.py` 中实现统一入口。

建议接口：

```python
def load_model_and_processor(
    *,
    model_id_or_path: str,
    model_type: str,
    quant_method: str = "none",
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    **kwargs,
):
    """
    Load HF model and processor.

    Important:
    - local_files_only defaults to True to avoid accidental local download.
    - This function should only be executed on remote machine where model files exist.
    """
```

支持值：

```text
quant_method = none | bnb4 | bnb8 | gptq | awq
model_type   = llava_onevision | qwen2vl
```

第一阶段要求：

- `none`、`bnb4`、`bnb8` 可以写实际 loader 代码；
- `gptq`、`awq` 可以先写 skeleton；
- 所有 loader 必须默认 `local_files_only=True`；
- 如果本地没有模型，错误信息必须提示：需要在远程机器上运行。

注意：不要在 import 顶层强制导入很重的依赖。可以在函数内部 lazy import：

```python
def load_xxx(...):
    from transformers import AutoProcessor
    ...
```

---

## 13. Model adapter 设计

### 13.1 Base adapter

在 `src/prune_quant_baseline/models/base_adapter.py`：

```python
from abc import ABC, abstractmethod
from typing import Any

from prune_quant_baseline.core.datatypes import VisualTokenMeta


class MLLMAdapter(ABC):
    """Adapter for model-specific multimodal preprocessing and token metadata."""

    @abstractmethod
    def prepare_inputs(self, processor: Any, sample: dict, device: str | None = None) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_visual_token_meta(self, model: Any, inputs: dict) -> VisualTokenMeta:
        raise NotImplementedError

    @abstractmethod
    def build_inputs_embeds(self, model: Any, inputs: dict):
        raise NotImplementedError
```

### 13.2 LLaVA-OneVision adapter skeleton

在 `models/llava_onevision_hf.py` 中：

- 写 `LlavaOneVisionHFAdapter`；
- 支持 image sample 的 prepare；
- video 先 TODO；
- 明确如何从 input ids / special tokens / image features 找 visual token positions；
- 如果暂时无法可靠定位，先抛出 `NotImplementedError`，但保留接口。

### 13.3 Qwen2-VL adapter skeleton

在 `models/qwen2vl_hf.py` 中：

- 写 `Qwen2VLHFAdapter`；
- 支持 image sample 的 prepare；
- video 先 TODO；
- 注意 Qwen2-VL 的 `image_grid_thw`、`video_grid_thw`、`rope_deltas`；
- pruning 后 position_ids 应优先 gather 原始 position_ids，不要重算简化 position ids。

---

## 14. CLI 脚本

### 14.1 `run_infer_pruned.py`

目标：远程机器上运行单条或 JSONL 推理。

建议参数：

```bash
python -m prune_quant_baseline.scripts.run_infer_pruned \
  --model-type qwen2vl \
  --model-path /remote/path/to/model \
  --input-jsonl /remote/path/to/eval.jsonl \
  --output-jsonl /remote/path/to/output.jsonl \
  --pruner attention_proxy \
  --retention-ratio 0.5 \
  --quant-method none \
  --dtype bfloat16 \
  --max-new-tokens 128
```

第一阶段要求：

- 写 argparse；
- 加载 config；
- 根据 `model_type` 创建 adapter；
- 根据 `quant_method` 加载模型；
- 根据 `pruner` 创建 pruner；
- 对每条 sample 执行推理流程；
- 如果 adapter 未完成，抛出清楚错误；
- 输出 JSONL 字段至少包括：
  - `id`；
  - `prompt`；
  - `prediction`；
  - `retention_ratio`；
  - `num_visual_tokens_before`；
  - `num_visual_tokens_after`；
  - `quant_method`；
  - `model_type`。

### 14.2 `run_generate_labels.py`

目标：后续为 learned compressor 生成 GAE labels。

第一阶段只需 skeleton：

```bash
python -m prune_quant_baseline.scripts.run_generate_labels \
  --model-type qwen2vl \
  --model-path /remote/path/to/model \
  --input-jsonl /remote/path/to/calib.jsonl \
  --output-path /remote/path/to/gae_labels.pt
```

### 14.3 `profile_latency.py`

目标：后续远程测速。

字段建议：

- wall-clock latency；
- prefill latency；
- decode latency；
- peak GPU memory；
- visual tokens before / after；
- generated tokens；
- quant method；
- dtype。

第一阶段可以只写 CLI 和 TODO。

---

## 15. 配置文件模板

请在 `configs/` 下写 YAML 模板。

示例：`configs/qwen2vl_7b_image.yaml`

```yaml
model:
  model_type: qwen2vl
  model_path: ${MODEL_PATH}
  dtype: bfloat16
  device_map: auto
  trust_remote_code: true
  local_files_only: true

quant:
  method: none
  ignore_modules:
    - visual
    - vision_tower
    - multi_modal_projector

pruning:
  method: attention_proxy
  retention_ratio: 0.5
  min_keep: 1
  physical_delete: true

inference:
  max_new_tokens: 128
  temperature: 0.0
  do_sample: false
  output_attentions: true
  use_cache: true

data:
  input_jsonl: ${INPUT_JSONL}
  output_jsonl: ${OUTPUT_JSONL}
```

注意：`${MODEL_PATH}` 这类变量由远程环境提供，本地不要替换成真实路径。

---

## 16. Synthetic tests：只写，不在本地运行

请创建以下 tests，方便远程或 CI 运行。

### 16.1 `tests/test_token_gather.py`

测试点：

- Top-K 返回原始顺序；
- retention ratio 边界；
- 全部 non-visual token 保留；
- visual token 只保留 selected；
- `inputs_embeds` gather shape 正确；
- `attention_mask` gather shape 正确；
- `position_ids` 支持 `[B, S]` 和 `[B, 3, S]`。

### 16.2 `tests/test_attention_proxy.py`

测试点：

- synthetic attention shape `[1, H, S, S]`；
- text-to-visual 平均 score 正确；
- visual_indices 为空时报错；
- batch size > 1 报错或清晰处理。

### 16.3 `tests/test_config_loading.py`

测试点：

- YAML 加载；
- 环境变量替换；
- 默认值补齐；
- 缺失必要字段时报错。

重要：当前本地阶段不要执行这些测试。

---

## 17. 远程测试占位说明

请创建 `remote/README_REMOTE_TESTING.md`，写入以下内容：

```markdown
# Remote Testing Guide

This repository is designed so that local development only writes code.
Real model loading, dataset access, and GPU tests must be executed on the remote machine.

## Required information to be provided later

- SSH host
- SSH user
- SSH password or key
- Remote project path
- Remote model path
- Remote data path
- CUDA / Python environment information

## Do not commit

- SSH password
- Hugging Face token
- private model paths if sensitive
- benchmark outputs if too large

## Example remote smoke command

```bash
python -m prune_quant_baseline.scripts.run_infer_pruned \
  --model-type qwen2vl \
  --model-path "$MODEL_PATH" \
  --input-jsonl "$INPUT_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --pruner attention_proxy \
  --retention-ratio 0.5 \
  --quant-method none \
  --max-new-tokens 32
```
```

也可以创建 `remote/run_remote_smoke.sh`，但里面只能使用环境变量，不要写死路径。

---

## 18. 推理流程：第一阶段目标逻辑

远程正式运行时，attention-proxy pruning 的流程应该是：

```text
1. Load model + processor on remote.
2. Processor prepares multimodal input.
3. Adapter obtains or builds inputs_embeds.
4. Run a prefill forward with output_attentions=True to get first-layer attention.
5. Adapter identifies visual token positions and text query positions.
6. AttentionProxyPruner scores each visual token.
7. select_topk_visual_tokens chooses retained visual tokens.
8. build_keep_indices keeps all non-visual tokens plus selected visual tokens.
9. gather_sequence_tensors physically shortens inputs_embeds / attention_mask / position_ids.
10. Run generation with compressed sequence.
11. Save prediction and metadata to JSONL.
```

如果某些 HF 模型的 `generate()` 不接受直接 compressed `inputs_embeds`，可以先在 adapter 中留 TODO，并实现较保守的 forward/generate wrapper。不要为了临时跑通而写不可维护的 hack。

---

## 19. 代码质量要求

请遵守：

- Python 3.10+；
- type hints；
- dataclass / Protocol / ABC 合理使用；
- 函数要有 docstring；
- 错误信息要清楚；
- 不要吞异常；
- 不要在 import 时加载模型；
- 不要在 import 时访问 GPU；
- 不要在 import 时访问网络；
- logging 使用标准库 `logging`；
- CLI 使用 `argparse` 即可，不强制 Hydra；
- 配置文件用 `yaml.safe_load`；
- 环境变量替换要可控，不要 eval 任意字符串。

---

## 20. 第一阶段完成标准

Codex 完成后，应该满足：

1. 仓库结构完整；
2. `pip install -e .` 理论上可安装；
3. 所有核心模块可 import，且 import 不触发模型下载；
4. token selection / gather 逻辑有独立函数；
5. attention-proxy pruner 可基于 synthetic attention 返回分数；
6. quant loader 默认 `local_files_only=True`；
7. adapter 未完成处有明确 TODO / NotImplementedError；
8. CLI 参数完整，不写死路径；
9. 远程测试说明已写好；
10. 本地不运行真实模型测试，不下载模型，不下载数据。

---

## 21. 后续阶段计划，不属于当前必须完成范围

### Stage 2：远程 smoke test

- 使用远程已有模型和一小批样本；
- 验证 vanilla inference；
- 验证 attention-proxy pruning；
- 检查输出 shape、token count、预测 JSONL。

### Stage 3：GAE oracle

- 生成 oracle relevance label；
- 比较 random pruning / attention-proxy / GAE oracle；
- 确认 pruning ratio 对性能的影响。

### Stage 4：Learned compressor

- 使用 GAE label 训练 Conv1D compressor；
- 用 compressor 替代 backward-based oracle；
- 进行真实 latency / memory profile。

### Stage 5：Prune + Quant

- 加载 bnb4 / bnb8 / GPTQ / AWQ 模型；
- 对比：
  - FP16/BF16 vanilla；
  - quant-only；
  - prune-only；
  - prune + quant；
- 分析 quantization 是否改变 visual token importance 分布。

---

## 22. 给 Codex 的执行建议

请按以下顺序实现：

1. 创建 package skeleton 和配置文件；
2. 实现 `datatypes.py`、`config.py`、`logging_utils.py`；
3. 实现 `token_gather.py`；
4. 实现 `attention_proxy.py`；
5. 写 synthetic tests 文件，但不要运行；
6. 实现 quant loader skeleton；
7. 实现 model adapter skeleton；
8. 实现 CLI skeleton；
9. 实现 compressor module skeleton；
10. 写远程测试说明；
11. 检查是否有任何本地下载模型/数据的行为，如有必须删除。

当前最重要的是把基础代码写干净，不追求一次性真实跑通远程模型。

