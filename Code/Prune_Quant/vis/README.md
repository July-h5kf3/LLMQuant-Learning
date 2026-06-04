# Token Pruning Visualization

这个目录用于可视化比较两种剪枝策略删掉的 visual tokens：

- 原始 GAE：`gae_scores` 是保留分数，分数最低的一批 visual tokens 会被标红并删除。
- 量化-剪枝协同指标：`quant_joint_scores = lambda * C_quant - C_drop` 是删除分数，分数最高的一批 visual tokens 会被标红并删除。

输出图片会保存到 `vis/outputs/`。现在图中只展示 visual tokens，横轴是 visual token 的局部编号 `0 ... N_visual-1`。四宫格含义如下：

- 左上：原始 visual token 分布，其中 GAE 删除的 visual tokens 用红色标注。
- 右上：原始 visual token 分布，其中量化-剪枝协同指标删除的 visual tokens 用红色标注。
- 左下：GAE 删除后剩余 visual tokens 的分布。
- 右下：量化-剪枝协同指标删除后剩余 visual tokens 的分布。

每个 visual token 的纵向线段由该 token 的 `D_model` 通道最小值和最大值构成。

脚本还会额外生成一张 `_image_overlay.png`，把被删除的 visual tokens 投回原始图片：

- 左列：原图。
- 中列：GAE 删除位置。
- 右列：量化-剪枝协同指标删除位置。

这个映射基于 Qwen 的 `image_grid_thw` 和 `spatial_merge_size`，适用于单图样本；多图样本会先使用第一张图并打印提示。

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
    "vision_text_boundary": int,              # 可选，仅在没有 visual_indices 时用于推断前 N 个 token 为视觉 token
    "gae_scores": Tensor_or_ndarray,          # [N_visual]
    "quant_joint_scores": Tensor_or_ndarray,  # [N_visual]
}
```

如果样本没有 `visual_indices`，可以用 `--visual-count N` 表示前 `N` 个 token 是 visual tokens。

## 运行真实样本

推荐方式是直接传入 YAML，让脚本按仓库现有运行逻辑从题目集合生成可视化所需的 `inputs_embeds`、原始 GAE 分数和量化-剪枝协同分数：

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
  # 可选：控制 processor 的视觉 token budget。
  # 如果希望 Qwen2-VL 图像样本接近 1500 个 visual tokens，可以先设置这个值。
  # 实际 token 数仍会受原图尺寸和 processor resize 规则影响。
  processor_max_visual_tokens: 1500

questions:
  source: tsv        # tsv / jsonl / hf
  dataset: MME
  path: ${MME_TSV}
  image_root: ${IMAGE_ROOT}
  mme_prompt_style: default

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
  random_sample: true
  seed: null           # 需要复现同一张样本时设置整数
  image_overlay: true
  score_bars: true
  output_dir: outputs
  save_sample_artifacts: true
  sample_artifact_dir: samples
```

说明：

- `questions.source: tsv` 支持 VLMEvalKit 风格的 MME TSV；每一行题目都会转换成一个可视化样本。
- MME TSV 通常包含 base64 `image`、`question`、`answer`，以及可选 `category`、`question_id`、`image_path`。
- `questions.source: jsonl` 仍要求每行包含 `prompt` / `question` / `text` 之一，以及 `image` / `image_path` / `images` 之一。
- `questions.source: hf` 可从 Hugging Face dataset 抽题，例如 `hf_dataset: lmms-lab/MME`、`hf_split: test`。
- 旧入口仍兼容：如果没有 `questions`，脚本会读取 `calibration.path`、`calibration.calib_jsonl`、`calibration.input_jsonl`、`data.calib_jsonl` 或 `data.input_jsonl`。
- `answer_source: sample` 时优先使用样本里的 `answer`；没有 answer 会自动生成 replay answer。
- 量化-剪枝协同分数通过仓库里的 `_score_gae_quant_joint` 计算，目前对应 RTN scoring forward。
- 协同剪枝参数优先读取 `quant_joint.*`，也兼容已有配置中的 `pruning.quant_lambda`、`pruning.quant_method`、`pruning.rtn_bits`、`pruning.rtn_group_size`。
- 原始 GAE 分数通过 `_score_gae_oracle` 计算。
- `score_bars: true` 会额外生成 `_score_bars.png`，用柱状图展示每个 visual token 的通道范围代理值 `max(channel) - min(channel)`、GAE score、`C_i^{quant}` 和 `D_i = lambda * C_i^{quant} - C_i^{drop}`；红色 bar 表示该行对应策略删除的 token，其中 GAE 行删除低分 top-k，`D_i` 行删除高分 top-k。
- `save_sample_artifacts: true` 会额外保存 `.pt` 样本包，后续可以不加载模型直接重画。
- YAML 内的相对路径按 YAML 文件所在目录解析；`vis/example_visualization_config.yaml` 里的 `outputs` 会解析到 `vis/outputs`。
- YAML 模式运行时会打印 `seq_len`、`visual_tokens`、`image_grid_thw` 和 processor pixel budget；如果 visual token 数不是预期的约 1500，先看这行诊断。
- Qwen2-VL/Qwen2.5-VL 的 visual token 数由 processor resize 后的 `image_grid_thw` 决定，不是固定 1500。若没有设置 `model.processor_max_visual_tokens` / `model.processor_max_pixels`，或者原图本身较小，实际 visual tokens 可能明显少于 1500。
- `visualization.random_sample: true` 时，每次运行会从当前样本源中随机抽样；设置 `seed` 可以固定抽到同一批样本。
- 当使用 `questions` 时，随机抽样单位就是题目，例如 MME TSV 中的一行问题。

也可以对已经保存好的样本包重画：

```bash
python vis/visualize_token_pruning.py \
  --sample /path/to/sample.pt \
  --retention-ratio 0.5 \
  --image-overlay \
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
