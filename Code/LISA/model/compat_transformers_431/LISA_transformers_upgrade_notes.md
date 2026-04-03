# LISA 升级到新版本 Transformers 的实验记录

日期: 2026-04-03

实验仓库: `/root/autodl-tmp/LISA`

只读参考仓库: `/root/autodl-tmp/LLMQuant-Learning/Code/LISA`

## 1. 目标

目标是让 LISA / LISA++ 在高版本 `transformers` 下保持原有精度，并为后续接入 `vllm` 等依赖新版 `transformers` 的工具打基础。

截至 2026-04-03，我确认 PyPI 上 `transformers` 的最新正式版本是 `5.3.0`。

来源: https://pypi.org/project/transformers/

## 2. 已确认的根因

这次实验里，已经确认了三类问题，不是单一问题。

### 2.1 `model_type: "llava"` 会和新版 Transformers 的官方 LlavaConfig 冲突

LISA 权重目录 `/root/autodl-tmp/LISA/weights/LISA_PLUS_7B/config.json` 里写的是:

```json
"model_type": "llava"
```

在 `transformers==5.3.0` 下:

```python
from transformers import AutoConfig
AutoConfig.from_pretrained("/root/autodl-tmp/LISA/weights/LISA_PLUS_7B")
```

会解析成官方的:

```python
transformers.models.llava.configuration_llava.LlavaConfig
```

而不是当前仓库里的自定义 `LlavaConfig`。

这会导致:

- `AutoConfig`
- `AutoModel`
- `AutoTokenizer`

都可能走错分支，出现静默错配。

### 2.2 `AutoTokenizer` 在新版下会给出不同的 token ids

同一句文本:

```text
Describe the image briefly.
```

在 `transformers==4.31.0` 下，`AutoTokenizer` 给出的 ids 是:

```python
[[1, 20355, 915, 278, 1967, 23359, 29889]]
```

在 `transformers==5.3.0` 下，`AutoTokenizer` 给出的 ids 是:

```python
[[1, 4002, 29581, 278, 1967, 23359, 29889]]
```

但是在 `transformers==5.3.0` 下，如果显式使用:

```python
from transformers import LlamaTokenizer
```

则 token ids 与旧版一致:

```python
[[1, 20355, 915, 278, 1967, 23359, 29889]]
```

结论:

- 新版里不能继续依赖 `AutoTokenizer.from_pretrained(...)`
- 需要改成显式 `LlamaTokenizer.from_pretrained(...)`

### 2.3 即使输入 token ids 固定一致，新版 `from_pretrained` 也会把权重装偏

我做了同一组固定 token ids 的对比:

```python
[[1, 20355, 915, 278, 1967, 23359, 29889]]
```

结果:

- `4.31.0` 下 logits 基线为 `sum = -72.65625`
- `5.3.0` 下即使使用相同 token ids，logits 仍明显偏移

进一步检查参数本身，发现同一个参数在新旧环境下装载后的值不同:

参数:

```text
model.layers.0.self_attn.q_proj.weight
```

旧环境下:

```text
sum = -90.21685028076172
```

新版 `transformers==5.3.0` 直接 `from_pretrained` 后:

```text
sum = -22.047840118408203
```

这说明问题不只是 tokenizer，而是新版 `from_pretrained` 对这套老权重的加载落位本身也发生了偏移。

## 3. 这轮实验已经做过的代码修改

我在实验仓库里做了两类修改。

### 3.1 vendoring 了 4.31 的 Llama 实现

新增目录:

`/root/autodl-tmp/LISA/model/compat_transformers_431/`

里面包含:

- `configuration_llama.py`
- `modeling_llama.py`
- `__init__.py`

用途:

- 把 LLaMA 核心实现固定在 `4.31.0` 行为
- 避免新版 `transformers` 内部实现变化直接影响数值行为

### 3.2 修改了 `llava_llama.py`

文件:

`/root/autodl-tmp/LISA/model/llava1p5/model/language_model/llava_llama.py`

主要改动:

- 不再直接依赖新版 `transformers` 自带的 `LlamaConfig / LlamaModel / LlamaForCausalLM`
- 改为依赖 vendored 的 `compat_transformers_431`
- 将自定义配置类型从 `"llava"` 改为 `"lisa_llava"`
- 增加了一个“新版环境下优先走 legacy 加载逻辑”的 `from_pretrained`

## 4. 建议你在正式仓库里照着改的内容

如果你要把这些改动正式迁到只读参考仓库的代码体系里，建议按下面顺序做。

### 4.1 先解决配置类型冲突

把自定义配置改成:

```python
class LlavaConfig(LlamaConfig):
    model_type = "lisa_llava"
```

同时注册:

```python
AutoConfig.register("lisa_llava", LlavaConfig)
```

不要再继续和官方 `"llava"` 复用同一个名字。

### 4.2 不要再用 `AutoTokenizer`

原来类似:

```python
tokenizer = transformers.AutoTokenizer.from_pretrained(...)
```

需要改成:

```python
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(
    model_path,
    model_max_length=args.model_max_length,
    padding_side="right",
    use_fast=False,
)
```

至少下面这些入口要改:

- 训练入口
- 测试入口
- chat / app 入口
- 量化入口

原因不是风格问题，而是新版 `AutoTokenizer` 在这个权重目录上会走偏。

### 4.3 不要直接使用新版 `PreTrainedModel.from_pretrained`

对当前这套老权重，建议自己接管加载流程:

1. 先用自定义 `config_class.from_pretrained(...)` 读配置
2. 直接实例化模型
3. 根据 `pytorch_model.bin.index.json` 逐 shard 手动 `torch.load`
4. 对每个 shard 执行 `model.load_state_dict(shard, strict=False)`
5. 最后再 `model.to(dtype=..., device=...)`

这一步非常关键。因为本次实验已经确认，新版直接 `from_pretrained` 会把至少一部分参数装偏。

### 4.4 对 RoPE 配置做一次新旧字段归一化

新版 `transformers` 会把 RoPE 相关字段组织成:

```python
rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}
```

而旧版 `4.31.0` 的 Llama 实现只认识:

```python
rope_scaling = {"type": "...", "factor": ...}
```

如果你 vendoring 旧版 Llama 实现，就需要在配置类里把新版字段归一化:

- `rope_type == "default"` 时转成 `None`
- `rope_type in {"linear", "dynamic"}` 时转成旧格式

## 5. 目前的验证结论

已经确认的:

- `model_type` 冲突是真实存在的
- `AutoTokenizer` 在新版下会漂
- 新版默认 `from_pretrained` 会把至少部分参数装偏
- 上面三类问题都足以导致 `test_ds.py` 指标下降
- 在实验仓库 `/root/autodl-tmp/LISA` 中，`train_ds.py --eval_only` 已经可以在 `transformers==5.3.0` 下稳定跑通
- 新版兼容加载后，语言模型主干参数已经能和旧版对齐
  - 例如 `model.layers.0.self_attn.q_proj.weight` 在新旧环境下都为 `-90.21685028076172`
- `[SEG]` 新增 token 的 embedding / lm_head 行也已经对齐
- 但整体验证指标仍未完全追平旧版基线

### 5.1 已复现的基线与新版指标

旧环境:

- Python: `/root/autodl-tmp/LLMQuant-Learning/Code/LISA/.venv/bin/python`
- `transformers==4.31.0`
- 命令: `train_ds.py --eval_only ... --val_dataset 'ReasonSeg|val'`
- 结果:

```text
giou: 0.6327
ciou: 0.6976
```

新版环境:

- Python: 仍使用旧虚拟环境里的 Python，但通过 `PYTHONPATH=/root/autodl-tmp/LISA/.venv-diag/lib/python3.11/site-packages` 注入 `transformers==5.3.0`
- 命令同上
- 当前结果:

```text
giou: 0.6231
ciou: 0.6678
```

在加入一次从 `4.31.0` 环境抽取出来的视觉分支 patch 之后，再跑 `train_ds.py --eval_only`，结果仍基本一致:

```text
giou: 0.6232
ciou: 0.6679
```

结论:

- 当前新版兼容方案已经把“无法加载 / 大幅错位 / tokenizer 漂移”这几个大问题解决掉了
- 但距离旧版基线仍有约 `0.0096 giou / 0.0298 ciou` 的差距
- 这个差距不是语言模型主干加载错误导致的，更像是视觉或多模态链路里仍有一处更细的行为差异

### 5.2 当前最可疑但尚未完全证实的剩余问题

这轮继续深入后，发现:

- 新旧环境对比时，明显漂移的参数主要集中在 `model.visual_model.*`
- `CLIPImageProcessor` 的输出和 `CLIPVisionModel` 选层特征在单样本上的 sum 基本一致
- 说明剩余误差更可能在:
  - SAM / mask decoder 分支
  - 或者视觉分支在旧版 `from_pretrained` 下的某些隐式加载行为
  - 或者评测链路里某个新版依赖的细粒度数值变化

当前还不能诚实地宣称“新版性能已完全对齐旧版”。

还没有完整跑完的:

- 使用“显式 `LlamaTokenizer` + 手动 shard 加载”后的整套 `test_ds.py` 全量指标

原因:

- 这轮主要时间花在根因定位上
- 实验仓库当前没有完全同步只读参考仓库里的测试入口
- 最后一轮整体验证需要把只读仓库的测试脚本/配置完整迁到实验仓库或在正式仓库里落地后再跑

## 6. 推荐的下一步执行顺序

1. 先把测试入口 `test_ds.py` 与 `configs/test_ds.yaml` 迁入实验仓库。
2. 把 tokenizer 全部替换为显式 `LlamaTokenizer`。
3. 把模型加载入口统一替换为“手动 shard 加载”的自定义 `from_pretrained`。
4. 把权重目录里的 `config.json` 复制一份，改成:
   - `"model_type": "lisa_llava"`
   - `"transformers_version": "5.3.0"` 或你最终落地的目标版本
5. 用同一个 checkpoint，分别在:
   - `transformers==4.31.0`
   - `transformers==5.3.0`
   跑 `test_ds.py`
6. 先看 tokenizer 对齐，再看参数对齐，再看最终 giou / ciou。

## 7. 这轮实验最重要的结论

不要把问题理解成“权重只需要改一改 config 版本号”。

真正的问题至少有三层:

1. `model_type` 冲突
2. `AutoTokenizer` 自动解析漂移
3. 新版 `from_pretrained` 对老权重的加载落位偏移

所以正确修复路径不是只改 `config.json`，而是:

- 显式 tokenizer
- 显式配置类型
- 显式权重加载

这三步要一起做。

## 8. 2026-04-03 最终验证结论

这轮实验已经完成了新版 `transformers` 下的性能对齐验证。

最终可复现结果:

- 旧环境基线
  - 环境: `transformers==4.31.0`
  - 入口: `train_ds.py --eval_only`
  - 指标: `giou=0.6327`, `ciou=0.6976`
- 新环境最终结果
  - 环境: `transformers==5.3.0`
  - 入口: `train_ds.py --eval_only`
  - 指标: `giou=0.6324`, `ciou=0.6977`
  - 日志: `/root/autodl-tmp/LISA/eval_tf530_segfix.log`
- 新环境去掉诊断 patch 后的复验
  - 环境: `transformers==5.3.0`
  - 不加载 `legacy_transformers_431_patch.bin`
  - 指标: `giou=0.6324`, `ciou=0.6977`
  - 日志: `/root/autodl-tmp/LISA/eval_tf530_segfix_no_patch.log`

结论:

- 新版 `transformers` 适配已经成功
- 当前实验仓库中的兼容方案已经把性能拉回到旧环境同一水平
- `legacy_transformers_431_patch.bin` 不是最终必需品，它只是之前定位视觉分支差异时生成的诊断产物

## 9. 最终根因

最后确认下来，真正影响指标的关键根因有四个:

1. `config.json` 里的 `model_type: "llava"` 会在新版 `transformers` 中和官方 Llava 配置冲突
2. 不能再用 `AutoTokenizer`，必须显式使用 `LlamaTokenizer`
3. 不能依赖新版默认的 `from_pretrained` 去加载这套旧权重，需要手动 shard 加载
4. 最关键但最隐蔽的一点:
   checkpoint 自带了 `added_tokens.json`，其中 `[SEG]` 已经存在，但它在新版 `transformers` 下会被当成普通 added token 处理，导致 `"ASSISTANT: [SEG]."` 被错误切成:
   - 旧版: `... 29901, 32000, 29889`
   - 新版错误行为: `... 29901, 29871, 32000, 29889`

这会让整条输入序列长度变化，继而让最后一个 `[SEG]` 对应到错误的位置，最终拖低 mask 预测指标。

## 10. 最终有效修复

### 10.1 配置与模型类兼容

文件:

- `/root/autodl-tmp/LISA/model/llava1p5/model/language_model/llava_llama.py`
- `/root/autodl-tmp/LISA/model/compat_transformers_431/__init__.py`
- `/root/autodl-tmp/LISA/model/compat_transformers_431/configuration_llama.py`
- `/root/autodl-tmp/LISA/model/compat_transformers_431/modeling_llama.py`

处理方式:

- 把内部配置类型改为 `lisa_llava`
- 对旧 checkpoint 的 `config.json` 做兼容读取，如果发现 `model_type == "llava"`，就在加载时改写成 `lisa_llava`
- 对 `transformers > 4.31` 走自定义 `_legacy_from_pretrained`
- 自定义 loader 中使用 `no_init_weights()` + 手动遍历 `pytorch_model.bin.index.json` / shard 文件并 `load_state_dict(strict=False)`

这样做的目的不是“改权重值”，而是保证新版库对旧结构的加载顺序和落位行为与 4.31 更接近。

### 10.2 tokenizer 兼容

文件:

- `/root/autodl-tmp/LISA/utils/tokenizer_compat.py`
- `/root/autodl-tmp/LISA/train_ds.py`
- `/root/autodl-tmp/LISA/test_ds.py`
- `/root/autodl-tmp/LISA/merge_lora_weights_and_save_hf_model.py`
- `/root/autodl-tmp/LISA/export_lisa_modules.py`

最终关键逻辑是:

```python
from transformers import AddedToken, LlamaTokenizer


def add_lisa_seg_token(tokenizer):
    num_added_tokens = tokenizer.add_tokens(
        [AddedToken("[SEG]", lstrip=True, normalized=False)]
    )
    seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    return num_added_tokens, seg_token_id
```

注意:

- 这里必须“总是重新注册一次 `[SEG]`”
- 即使 checkpoint 目录里已经有 `added_tokens.json`
- 因为新版 `transformers` 会把已存在的 `[SEG]` 按旧元信息加载成 `lstrip=False`
- 再次调用 `add_tokens(AddedToken(..., lstrip=True))` 不会扩大 vocab，但会刷新它的切分行为

这一步是最后把指标拉回来的决定性修复。

### 10.3 已验证的中间结论

修复前:

- 单样本 `input_ids` 长度:
  - 旧环境: `93`
  - 新环境: `94`
- 差异点正好出现在结尾 `ASSISTANT: [SEG].`

修复后:

- `input_ids` 完全一致
- `attention_masks` 完全一致
- `image_embeddings` 完全一致
- `pred_embeddings` 的单样本差异从之前的 `max_abs≈4.19` 降到 `0.125`
- 最终 `giou/ciou` 回到旧环境水平

## 11. 如果要在正式仓库里落地，应该怎么改

你可以按下面顺序在正式仓库中迁移:

1. 复制 `compat_transformers_431` 目录
2. 替换 `llava_llama.py`，保留:
   - `model_type = "lisa_llava"`
   - `_load_compat_config`
   - `_legacy_from_pretrained`
   - 手动 shard 加载逻辑
3. 新增 `utils/tokenizer_compat.py`
4. 在所有 tokenizer 初始化入口统一替换成:
   - `load_lisa_tokenizer(...)`
   - `add_lisa_seg_token(tokenizer)`
5. 把直接 `tokenizer.add_tokens("[SEG]")` 的地方全部换掉
6. 再跑一次:
   - `train_ds.py --eval_only`
   - 如果你正式仓库还保留 `test_ds.py` 流程，也建议补跑一遍

## 12. 当前实验仓库里真正需要保留的改动

建议保留:

- `model/compat_transformers_431/`
- `model/llava1p5/model/language_model/llava_llama.py`
- `utils/tokenizer_compat.py`
- `train_ds.py` 中对 tokenizer helper 的调用
- `test_ds.py` 中对 tokenizer helper 的调用
- 其他脚本中对 tokenizer helper 的调用

可以不依赖:

- `weights/LISA_PLUS_7B/legacy_transformers_431_patch.bin`
- `legacy_431_patch_keys.txt`

这两者保留作排障记录可以，但不是最终适配方案的一部分。
