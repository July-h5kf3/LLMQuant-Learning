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
