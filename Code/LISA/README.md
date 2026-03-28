# LISA 推理复现说明

这个目录用于整理原始 LISA 的推理复现任务。当前目标不是直接运行旧仓库，而是基于原论文与原实现，梳理出一套更适合较新环境的 PyTorch 推理版本，并补齐推理前所需的 fine-tuning / 权重适配流程。

## 任务边界

- 以推理为主，但要把推理所需的 fine-tuning / adapter 适配过程算入范围
- 不复现完整训练流水线，不做大规模数据准备、训练采样与训练期数据处理复刻
- 优先对齐原始 LISA，再考虑 LISA++
- 优先兼容较新的 `torch`、`transformers` 与 CUDA 环境

## 总体拆解

LISA 的整体复现链路可以拆成 5 个大模块：

1. `LLaVA 多模态输入与生成`
2. `推理前的 fine-tuning / 权重适配`
3. `LISA 特有的 [SEG] 特征桥接`
4. `SAM 分割解码`
5. `推理脚本与结果后处理`

下面的 TODO 按“先打通依赖底座，再确定需要被 fine-tune 的部件，再实现上层能力”的顺序排列。也就是说，先解决 LLaVA 如何吃图文并输出 hidden states，再确认 fine-tuning 的对象、权重组织和合并方式，然后再做 `[SEG]` 到 mask 的桥接，最后处理脚本、可视化和兼容性。

## TODO

### 0. 边界确认

- [ ] 确认原始 LISA 以 `main` 分支为准，LISA++ 对应 `origin/lisa_plus`
- [ ] 明确当前不是复现完整训练，但要把推理前所需的 fine-tuning / adapter 流程纳入范围
- [ ] 明确不把大规模 dataset、训练采样、loss 对齐和完整训练框架纳入首要实现范围
- [ ] 确认复现目标是“单图单轮推理先跑通”，而不是一步到位支持全部 demo 能力

### 1. LLaVA 模块

这一部分是整个推理链路的底座，负责把图像和文本组织成 LLM 可以处理的多模态输入，并产出生成结果与 hidden states。

#### 1.1 文件职责梳理

- [ ] 阅读 `chat.py`
  作用：原始 CLI 推理入口，展示了 prompt 组装、图像预处理、模型加载、`evaluate()` 调用和结果保存的最小闭环
- [ ] 阅读 `model/llava/mm_utils.py`
  作用：处理 `<image>` 占位符与 `tokenizer_image_token()`，是图文 token 组织的起点
- [ ] 阅读 `model/llava/conversation.py`
  作用：定义对话模板，决定用户输入如何被包装成 LLaVA 风格 prompt
- [ ] 阅读 `model/llava/model/llava_arch.py`
  作用：核心多模态拼接逻辑，负责把图像特征插入文本 embedding 序列
- [ ] 阅读 `model/llava/model/language_model/llava_llama.py`
  作用：LLaVA 的语言模型封装，forward/generate 以及 hidden states 返回逻辑都在这里
- [ ] 阅读 `model/llava/model/multimodal_encoder/clip_encoder.py`
  作用：CLIP vision tower 封装，负责将图像转成视觉特征
- [ ] 阅读 `model/llava/model/multimodal_encoder/builder.py`
  作用：vision tower 的构建入口

#### 1.2 复现顺序

- [ ] 先复现 prompt 组织
  目标：明确 `<image>`、`<im_start>`、`<im_end>` 的插入方式
- [ ] 再复现 tokenizer 与 image token 注入
  目标：得到与原始 LLaVA 兼容的 `input_ids`
- [ ] 再复现 vision tower 前向
  目标：把图像变成可插入 LLM 的视觉 token
- [ ] 再复现多模态 embedding 拼接
  目标：理清图像 token 如何替换文本中的 `<image>` 占位
- [ ] 再复现生成接口
  目标：确认新版 `transformers` 下如何稳定拿到 `sequences` 和最后层 hidden states
- [ ] 最后再决定哪些旧版 LLaVA 封装保留，哪些需要重写

#### 1.3 本阶段产出

- [ ] 一份独立的 LLaVA 推理最小闭环说明
- [ ] 一份“旧实现 vs 新环境”的兼容性判断
- [ ] 一套不依赖训练代码的图文输入组织方案

### 2. fine-tuning / 权重适配模块

这一部分虽然不属于最终推理前向本身，但它直接决定你最后拿来推理的权重是不是正确的，因此也必须算在复现范围内。

#### 2.1 文件职责梳理

- [ ] 阅读 `merge_lora_weights_and_save_hf_model.py`
  作用：展示 LoRA 权重如何与 base model 合并，并导出可直接推理的 HF 格式模型
- [ ] 阅读 `model/LISA.py` 中与可训练参数相关的部分
  作用：确认哪些层在原始方案中需要 fine-tune，例如 `text_hidden_fcs`、是否包含 mask decoder 等
- [ ] 阅读 `train_ds.py` 中与模型构建相关的部分
  作用：只抽取“哪些模块参与 fine-tuning、参数如何初始化、权重如何保存”这些信息，不复刻完整训练流水线

#### 2.2 复现顺序

- [ ] 先确认 base model 由哪些部分组成
  目标：LLM、vision tower、SAM、额外桥接层分别来自哪里
- [ ] 再确认 fine-tuning 时哪些参数会更新
  目标：区分冻结模块和需要训练的模块
- [ ] 再确认最终推理需要加载哪几类权重
  目标：base 权重、fine-tuned 权重、是否需要 merge
- [ ] 再确认权重导出格式
  目标：最后得到能直接 `from_pretrained()` 加载的推理模型
- [ ] 最后判断在新环境下是否保留 LoRA merge 路线，还是改成更直接的权重组织方式

#### 2.3 本阶段产出

- [ ] 一份“推理前需要哪些权重”的清单
- [ ] 一份“哪些模块需要 fine-tune / 哪些保持冻结”的说明
- [ ] 一份面向推理的权重加载与合并方案

### 3. LISA 特有桥接模块

这一部分是 LISA 相比普通 LLaVA 的关键增量，负责把文本生成结果中的 `[SEG]` token 变成 SAM 可理解的 prompt embedding。

#### 3.1 文件职责梳理

- [ ] 阅读 `model/LISA.py`
  作用：LISA 的主模型封装，连接 LLaVA 和 SAM
- [ ] 重点阅读 `LisaMetaModel.initialize_lisa_modules()`
  作用：创建 `text_hidden_fcs` 和 SAM 相关组件
- [ ] 重点阅读 `LISAForCausalLM.evaluate()`
  作用：推理阶段主流程，包含生成、`[SEG]` 定位、mask 解码

#### 3.2 复现顺序

- [ ] 先确认 `[SEG]` token 在 tokenizer 中的表示方式
- [ ] 再确认生成结果中如何定位 `[SEG]`
- [ ] 再确认如何抽取 `[SEG]` 对应 hidden states
- [ ] 再复现 `text_hidden_fcs`
  目标：把 LLM hidden state 映射到 256 维 segmentation prompt embedding
- [ ] 最后确认这部分输出如何送给 SAM prompt encoder

#### 3.3 本阶段产出

- [ ] 一份 `[SEG]` token 到 prompt embedding 的流程图
- [ ] 一份不依赖训练逻辑的 LISA bridge 最小实现方案

### 4. SAM 模块

这一部分负责真正输出分割 mask，整体与 `transformers` 解耦较多，是当前最可能直接复用的部分。

#### 4.1 文件职责梳理

- [ ] 阅读 `model/segment_anything/build_sam.py`
  作用：SAM 模型构建入口
- [ ] 阅读 `model/segment_anything/modeling/`
  作用：SAM 的 image encoder、prompt encoder、mask decoder 等主体实现
- [ ] 阅读 `model/segment_anything/utils/transforms.py`
  作用：`ResizeLongestSide`，是 SAM 侧输入预处理的关键

#### 4.2 复现顺序

- [ ] 先复现 SAM 图像预处理
- [ ] 再复现 image encoder 前向
- [ ] 再复现 prompt encoder 接收 text prompt embedding 的方式
- [ ] 再复现 mask decoder 与 `postprocess_masks`
- [ ] 最后验证输出 mask 是否能恢复到原图尺寸

#### 4.3 本阶段产出

- [ ] 一份可独立运行的 SAM 推理子模块
- [ ] 一份 SAM 可直接复用范围说明

### 5. 推理入口与后处理

这一部分不是算法核心，但决定你最终复现版本的可运行性与可验证性。

#### 5.1 文件职责梳理

- [ ] 阅读 `chat.py`
  作用：最直接的命令行推理流程参考
- [ ] 阅读 `app.py`
  作用：Gradio demo 版本，可参考但不建议直接沿用

#### 5.2 复现顺序

- [ ] 先实现最小 CLI 推理入口
- [ ] 再实现文本输出与 mask 保存
- [ ] 再实现 mask 叠图可视化
- [ ] 最后决定是否保留多轮对话、网页 demo 等非核心功能

#### 5.3 本阶段产出

- [ ] 一份最小可运行推理脚本
- [ ] 一套基础样例图与输出结果

### 6. 兼容性与重写决策

这一部分不是最后做，而是贯穿全过程持续判断。

- [ ] 判断哪些模块对 `torch 2.8.0 + CUDA 12.8` 风险低，可以直接复用
- [ ] 判断哪些模块强耦合旧版 `transformers==4.31.0`，应视为参考实现
- [ ] 判断 fine-tuning 相关脚本中哪些部分只是“权重组织参考”，哪些部分能直接沿用
- [ ] 排查原始代码中的硬编码
  例如固定 `255` 个 image token 的写法
- [ ] 排查旧版量化、`deepspeed`、`bitsandbytes` 相关逻辑是否需要彻底剥离
- [ ] 为后续 LISA++ 迁移保留差异记录

## 当前阅读顺序建议

如果你现在开始真正看代码，建议顺序如下：

1. `chat.py`
2. `model/LISA.py`
3. `model/llava/mm_utils.py`
4. `model/llava/conversation.py`
5. `model/llava/model/llava_arch.py`
6. `model/llava/model/language_model/llava_llama.py`
7. `model/llava/model/multimodal_encoder/clip_encoder.py`
8. `merge_lora_weights_and_save_hf_model.py`
9. `train_ds.py` 中模型构建相关部分
10. `model/segment_anything/build_sam.py`
11. `model/segment_anything/modeling/`

## 说明文档

更细的文件职责、复现顺序和“可直接复用 / 只能参考”的判断见 [reuse_notes.txt](/Users/lorn/Documents/Playground/周汇报/LLMQuant-Learning/Code/LISA/reuse_notes.txt)。
