# Token Pruning Visualization

这个目录用于可视化比较两种剪枝策略删掉的 visual tokens：

- 原始 GAE：`gae_scores` 是保留分数，分数最低的一批 visual tokens 会被标红并删除。
- 量化-剪枝协同指标：`quant_joint_scores = lambda * C_quant - C_drop` 是删除分数，分数最高的一批 visual tokens 会被标红并删除。

输出图片会保存到 `vis/outputs/`。四宫格含义如下：

- 左上：原始 token 分布，其中 GAE 删除的 visual tokens 用红色标注。
- 右上：原始 token 分布，其中量化-剪枝协同指标删除的 visual tokens 用红色标注。
- 左下：GAE 删除后剩余 token 的分布。
- 右下：量化-剪枝协同指标删除后剩余 token 的分布。

每个 token 的纵向线段由该 token 的 `D_model` 通道最小值和最大值构成。粗黑线表示 Vision Token 与 Text Token 的边界。

## 快速预览

没有真实中间量时，可以先跑一个合成样本检查图的样式：

```bash
python vis/visualize_token_pruning.py --demo --retention-ratio 0.5
```

如果本机没有 `python` 命令，请改用 `python3`。脚本优先使用 `matplotlib` 出图；如果环境里没有 `matplotlib`，会自动使用 Pillow 兜底生成 PNG。

生成结果示例：

```text
vis/outputs/demo_sample_token_pruning.png
```

## 真实样本格式

脚本支持 `.pt` / `.pth` / `.npz` 文件。每次运行展示一个样本，推荐字段如下：

```python
{
    "id": "sample_0001",
    "inputs_embeds": Tensor_or_ndarray,        # [S, D] 或 [1, S, D]
    "visual_indices": Tensor_or_ndarray,      # [N_visual]，visual token 在完整序列中的位置
    "text_indices": Tensor_or_ndarray,        # 可选
    "vision_text_boundary": int,              # 可选，text token 开始的位置
    "gae_scores": Tensor_or_ndarray,          # [N_visual]
    "quant_joint_scores": Tensor_or_ndarray,  # [N_visual]
}
```

如果样本没有 `visual_indices`，可以用 `--visual-count N` 表示前 `N` 个 token 是 visual tokens。若没有 `vision_text_boundary`，默认使用 `max(visual_indices) + 1` 作为黑色边界线位置。

## 运行真实样本

推荐方式是直接传入 YAML，让脚本按仓库现有运行逻辑从校准集样本生成可视化所需的 `inputs_embeds`、原始 GAE 分数和量化-剪枝协同分数：

```bash
python3 vis/visualize_token_pruning.py --config vis/example_visualization_config.yaml
```

YAML 至少需要包含三类信息：

```yaml
model:
  model_type: qwen2_5_vl
  model_path: ${MODEL_PATH}
  dtype: bfloat16
  device_map: auto
  trust_remote_code: true
  local_files_only: true
  attn_implementation: eager

calibration:
  path: ${CALIB_JSONL}
  image_root: ${IMAGE_ROOT}

quant_joint:
  quant_lambda: 1.0
  quant_method: rtn
  rtn_bits: 4
  rtn_group_size: 0

pruning:
  retention_ratio: 0.5
  min_keep: 1
```

可选字段：

```yaml
scoring:
  answer_source: sample   # sample 或 generated
  per_token: true
  max_new_tokens: 16

visualization:
  limit: 1
  sample_offset: 0
  output_dir: outputs
  save_sample_artifacts: true
  sample_artifact_dir: samples
```

说明：

- `calibration.path` 指向校准集 JSONL；其中每一行就是一个可视化样本。
- 也兼容 `calibration.calib_jsonl`、`calibration.input_jsonl`、`data.calib_jsonl`、`data.input_jsonl`。
- 校准样本沿用仓库 adapter 逻辑，需要包含 `prompt` / `question` / `text` 之一，以及 `image` / `image_path` / `images` 之一。
- `answer_source: sample` 时优先使用样本里的 `answer`；没有 answer 会自动生成 replay answer。
- 量化-剪枝协同分数通过仓库里的 `_score_gae_quant_joint` 计算，目前对应 RTN scoring forward。
- 协同剪枝参数优先读取 `quant_joint.*`，也兼容已有配置中的 `pruning.quant_lambda`、`pruning.quant_method`、`pruning.rtn_bits`、`pruning.rtn_group_size`。
- 原始 GAE 分数通过 `_score_gae_oracle` 计算。
- `save_sample_artifacts: true` 会额外保存 `.pt` 样本包，后续可以不加载模型直接重画。
- YAML 内的相对路径按 YAML 文件所在目录解析；`vis/example_visualization_config.yaml` 里的 `outputs` 会解析到 `vis/outputs`。

也可以对已经保存好的样本包重画：

```bash
python vis/visualize_token_pruning.py \
  --sample /path/to/sample.pt \
  --retention-ratio 0.5 \
  --output-name sample_0001_pruning.png
```

如果字段名不同，可以通过参数指定：

```bash
python vis/visualize_token_pruning.py \
  --sample /path/to/sample.npz \
  --embeds-key hidden_states \
  --gae-key c_drop \
  --quant-key joint \
  --retention-ratio 0.5
```

## 生成样本包示例

下面是一个最小 `.pt` 样本包示例。实际使用时，把 `inputs_embeds`、`visual_indices`、`gae_scores` 和 `quant_joint_scores` 替换成模型前向与打分逻辑导出的真实张量即可。

```python
from pathlib import Path
import torch

sample = {
    "id": "sample_0001",
    "inputs_embeds": inputs_embeds.detach().cpu(),          # [1, S, D] 或 [S, D]
    "visual_indices": visual_indices.detach().cpu(),        # [N_visual]
    "vision_text_boundary": int(visual_indices.max()) + 1,
    "gae_scores": gae_scores.detach().cpu(),                # 原始 GAE keep scores
    "quant_joint_scores": joint_scores.detach().cpu(),      # 协同指标 drop scores
}

out = Path("vis/samples/sample_0001.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(sample, out)
```

随后运行：

```bash
python vis/visualize_token_pruning.py --sample vis/samples/sample_0001.pt --retention-ratio 0.5
```
