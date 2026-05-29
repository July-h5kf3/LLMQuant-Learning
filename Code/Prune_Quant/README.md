# Prune + Quant Baseline

[English](README.md) | [简体中文](README_zh.md)

Baseline scaffolding for multimodal LLM visual-token pruning and quantized inference.

This repository is intentionally local-safe:

- no model weights are committed;
- no datasets are committed;
- real model inference is intended for a GPU machine with existing model/data paths;
- model, data, and output paths are provided through CLI arguments, YAML config, or environment variables.

The current implementation supports:

- pure GAE-guided visual token pruning;
- prune-then-quant experiments with GAE pruning followed by MASQuant calibration/inference;
- Qwen2-VL style Hugging Face inputs for image tasks.

## Recommended Layout

Use separate directories for code, model weights, datasets, and experiment outputs.

```bash
export PROJECT_ROOT=/path/to/Prune_Quant
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
export WORK_ROOT=/path/to/prune_quant_runs

mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$WORK_ROOT"
```

For pure GAE pruning, `Qwen/Qwen2-VL-7B-Instruct` matches the current reproduction path.
For MASQuant, use a model supported by MASQuant, such as `Qwen/Qwen2.5-VL-7B-Instruct`.

## Environment Installation

Create a Python environment on the GPU machine.

```bash
conda create -n prune-quant python=3.10 -y
conda activate prune-quant

cd "$PROJECT_ROOT"
pip install -U pip setuptools wheel
```

Check the visible GPU and driver first:

```bash
nvidia-smi
```

This project should run on both A800 and RTX PRO 6000 Blackwell machines. Because RTX PRO 6000 Blackwell is the newer architecture, use a CUDA 12.8 PyTorch wheel for any environment that may run on it.

Recommended install for RTX PRO 6000 Blackwell, and for a shared A800/Blackwell environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Use this when the machine driver is new enough for CUDA 12.8. A800 is Ampere and is compatible with this newer CUDA runtime, while RTX PRO 6000 Blackwell should use a recent CUDA/PyTorch stack.

Do not unify on CUDA 12.6 for the shared environment: it is fine for A800, but PyTorch `cu126` wheels are not a safe target for RTX PRO 6000 Blackwell / SM120. Use `cu128` for Blackwell.

Fallback for A800-only machines with older drivers:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Do not use the A800-only fallback on RTX PRO 6000 Blackwell. Prefer one shared `cu128` environment when the driver supports it, or separate environments if the A800 machine has an older driver.

Verify the installed wheel can see the GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

Install this project and common runtime dependencies.

```bash
pip install -e ".[quant,test]"
pip install accelerate datasets huggingface_hub qwen-vl-utils sentencepiece protobuf
```

Install VLMEvalKit and register this repository's pruned Qwen2-VL model wrapper. VLMEvalKit uses `run.py` for standard inference and evaluation, and model names are resolved through `supported_VLM` in `vlmeval/config.py`.

```bash
export EXT_ROOT=/path/to/external
git clone https://github.com/open-compass/VLMEvalKit.git "$EXT_ROOT/VLMEvalKit"
export VLMEVALKIT_ROOT="$EXT_ROOT/VLMEvalKit"

cd "$VLMEVALKIT_ROOT"
pip install -e .

cd "$PROJECT_ROOT"
python remote/install_vlmeval_pruned_gae.py --vlmeval-root "$VLMEVALKIT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
```

Re-run `remote/install_vlmeval_pruned_gae.py` after editing files under `src/prune_quant_baseline/vlmeval/`, because the installer copies the wrapper into the VLMEvalKit checkout.

GAE pruning requires attention tensors and attention gradients. Use eager attention in the commands below:

```bash
export TOKENIZERS_PARALLELISM=false
```

## Model Download

Login to Hugging Face if the target model requires authentication.

```bash
huggingface-cli login
```

Download the Qwen2-VL model for pure GAE pruning.

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen2-VL-7B-Instruct"

export QWEN2VL_MODEL="$MODEL_ROOT/Qwen2-VL-7B-Instruct"
```

Download the Qwen2.5-VL model for MASQuant experiments.

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"

export QWEN25VL_MODEL="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"
```

The scripts default to local model loading. Keep `--model-path` pointed at the downloaded local directory.

## Dataset Download

The JSONL inference/calibration scripts expect one sample per line with an image path, prompt, and optional answer:

```json
{"id": "0", "image": "/abs/path/to/image.jpg", "prompt": "Describe the image.", "answer": "A short reference answer."}
```

For GAE oracle pruning, an answer is needed to define the target. If the JSONL has no `answer`, pass `--gae-answer-source generated` so the model first generates a replay answer.

### MASQuant Calibration Data

For this repository's prune-then-MASQuant path, calibration is controlled by `--calib-jsonl`. You do not need to use MASQuant's hard-coded `/nas/...` paths. A practical text-vision calibration set is MASQuant's filtered ShareGPT4V metadata plus COCO `train2017` images.

Download the MASQuant-filtered ShareGPT4V metadata:

```bash
mkdir -p "$DATA_ROOT/masquant" "$DATA_ROOT/coco"

wget -O "$DATA_ROOT/masquant/sharegpt4v_filtered_coco.json" \
  https://raw.githubusercontent.com/alibaba/EfficientAI/main/masquant/dataset/sharegpt4v_instruct_gpt4-vision_cap100k_filtered_coco_image.json
```

Download COCO `train2017` images. This is large, roughly 18 GB compressed:

```bash
wget -c -P "$DATA_ROOT/coco" http://images.cocodataset.org/zips/train2017.zip
unzip -q -n "$DATA_ROOT/coco/train2017.zip" -d "$DATA_ROOT/coco"
```

Convert the metadata to this project's JSONL format:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
src = data_root / "masquant" / "sharegpt4v_filtered_coco.json"
out = data_root / "calib_sharegpt4v_coco.jsonl"
rows = json.loads(src.read_text())

with out.open("w", encoding="utf-8") as f:
    for row in rows:
        prompt = ""
        answer = ""
        for turn in row["conversations"]:
            if turn["from"] == "human":
                prompt = turn["value"].replace("<image>", "").strip()
            elif turn["from"] == "gpt":
                answer = turn["value"].strip()
                break
        image = data_root / row["image"]
        if not image.exists():
            image = data_root / "coco" / row["image"].replace("coco/", "")
        if not image.exists():
            continue
        f.write(json.dumps({
            "id": row.get("id", image.stem),
            "image": str(image.resolve()),
            "prompt": prompt,
            "answer": answer,
        }, ensure_ascii=False) + "\n")

print(out)
PY

export CALIB_JSONL="$DATA_ROOT/calib_sharegpt4v_coco.jsonl"
```

For small smoke runs, create a short calibration subset:

```bash
head -n 16 "$CALIB_JSONL" > "$DATA_ROOT/calib_smoke_16.jsonl"
```

### Test Datasets

VLMEvalKit is the default benchmark path. It handles dataset loading, prediction files, and metric computation. Put its dataset cache under `DATA_ROOT`:

```bash
export LMUData="$DATA_ROOT/vlmeval"
mkdir -p "$LMUData"
```

Quick VLMEvalKit smoke run without pruning:

```bash
cd "$VLMEVALKIT_ROOT"
export PQ_RETENTION_RATIO=1.0
python run.py --data MME --model Qwen2VL_PrunedGAE \
  --work-dir "$WORK_ROOT/vlmeval_smoke" \
  --verbose
```

Common benchmark run:

```bash
cd "$VLMEVALKIT_ROOT"
python run.py --data MME MMStar MMVet --model Qwen2VL_PrunedGAE \
  --work-dir "$WORK_ROOT/vlmeval_qwen2vl_gae50" \
  --verbose
```

You can still create or download a custom evaluation JSONL for low-level script debugging:

```bash
export EVAL_JSONL="$DATA_ROOT/eval.jsonl"
```

## Pure GAE Prune

Default evaluation uses VLMEvalKit. The installer registers `Qwen2VL_PrunedGAE`, and its runtime settings are controlled by environment variables:

```bash
export QWEN2VL_MODEL="$MODEL_ROOT/Qwen2-VL-7B-Instruct"
export PQ_RETENTION_RATIO=0.5
export PQ_MIN_KEEP=1
export PQ_MAX_NEW_TOKENS=16
export PQ_GAE_ANSWER_SOURCE=generated
export PQ_GAE_PER_TOKEN=false
export PQ_ATTN_IMPLEMENTATION=eager

# Paper-faithful Qwen2-VL image setting, roughly 1500 visual language tokens.
export PQ_MIN_VISUAL_TOKENS=1500
export PQ_MAX_VISUAL_TOKENS=1500

cd "$VLMEVALKIT_ROOT"
python run.py --data MME MMStar MMVet --model Qwen2VL_PrunedGAE \
  --work-dir "$WORK_ROOT/vlmeval_qwen2vl_gae50" \
  --verbose
```

If A800 memory is tight, first lower the image budget:

```bash
export PQ_MIN_VISUAL_TOKENS=
export PQ_MAX_VISUAL_TOKENS=1024
```

Then re-run the same `python run.py ...` command.

The lower-level JSONL script remains useful for debugging a small custom file without VLMEvalKit:

```bash
python -m prune_quant_baseline.scripts.run_infer_pruned \
  --model-type qwen2vl \
  --model-path "$QWEN2VL_MODEL" \
  --input-jsonl "$EVAL_JSONL" \
  --output-jsonl "$WORK_ROOT/qwen2vl_gae50.jsonl" \
  --pruner gae_oracle \
  --retention-ratio 0.5 \
  --min-keep 1 \
  --quant-method none \
  --dtype bfloat16 \
  --device-map auto \
  --attn-implementation eager \
  --gae-answer-source generated \
  --max-new-tokens 128
```

Optional local benchmark helper for TSV/debug runs:

```bash
python -m prune_quant_baseline.scripts.run_image_benchmark \
  --dataset MME \
  --dataset-source hf \
  --hf-dataset lmms-lab/MME \
  --hf-split test \
  --hf-cache-dir "$DATA_ROOT/hf_cache" \
  --model-path "$QWEN2VL_MODEL" \
  --model-type qwen2vl \
  --output-jsonl "$WORK_ROOT/mme_qwen2vl_gae50.jsonl" \
  --metrics-json "$WORK_ROOT/mme_qwen2vl_gae50_metrics.json" \
  --pruner gae_oracle \
  --retention-ratio 0.5 \
  --attn-implementation eager \
  --gae-answer-source sample \
  --gae-per-token false \
  --processor-max-pixels 401408 \
  --max-new-tokens 16
```

If the MME data is already available locally as a VLMEvalKit-style TSV, the helper can also read it directly. This path is for debugging and is not the default reported result:

```bash
python -m prune_quant_baseline.scripts.run_image_benchmark \
  --dataset MME \
  --dataset-source tsv \
  --tsv "$DATA_ROOT/MME.tsv" \
  --model-path "$QWEN2VL_MODEL" \
  --model-type qwen2vl \
  --output-jsonl "$WORK_ROOT/mme_qwen2vl_gae50.jsonl" \
  --metrics-json "$WORK_ROOT/mme_qwen2vl_gae50_metrics.json" \
  --pruner gae_oracle \
  --retention-ratio 0.5 \
  --attn-implementation eager \
  --gae-answer-source sample \
  --gae-per-token false \
  --processor-max-pixels 401408 \
  --max-new-tokens 16
```

The TSV should contain the usual VLMEvalKit columns, including `image` as base64 image data plus `question` and `answer`. Optional columns such as `category`, `question_id`, and `image_path` are preserved in the output.

For MME/MMStar style evaluation, prefer `--gae-answer-source sample` because the TSV already has short labels (`A/B/C/D` or `Yes/No`). `--gae-answer-source generated` first generates a replay answer and then runs GAE on that answer, which costs extra memory and may turn a one-token label into many answer tokens. `--gae-per-token false` uses one backward pass over the answer instead of one backward pass per answer token; it is the safer default for A800/RTX PRO 6000 runs. Qwen2-VL reports visual tokens after its spatial merge; `216` language-side visual tokens normally means about `864` ViT patches before merge. The output JSONL now records both `num_visual_language_tokens` and `num_image_patches_before_merge`.

`--processor-max-pixels` controls Qwen2-VL dynamic image resolution. A useful A800 starting point is `401408` (`512 * 28 * 28`), and a higher-quality setting is `802816` (`1024 * 28 * 28`) if memory is stable. To match the paper's Qwen2-VL image setting of roughly 1500 visual tokens, use token-budget arguments instead of hand-computing pixels:

```bash
--processor-min-visual-tokens 1500 \
--processor-max-visual-tokens 1500
```

This is equivalent to about `1176000` max pixels (`1500 * 28 * 28`). It is more paper-faithful but much heavier than the low-memory setting.

Use `--retention-ratio 1.0` for the no-pruning baseline while keeping the same data/model settings.

## MASQuant Installation

MASQuant is kept as an external checkout.

```bash
export EXT_ROOT=/path/to/external
mkdir -p "$EXT_ROOT"

git clone https://github.com/alibaba/EfficientAI.git "$EXT_ROOT/EfficientAI"
export MASQUANT_ROOT="$EXT_ROOT/EfficientAI/masquant"

cd "$MASQUANT_ROOT"
pip install -e .
pip install lmms-eval
```

`flash-attn` is optional for this baseline. GAE needs eager attention, and the prune-then-MASQuant script can patch MASQuant's Qwen2.5 loader to honor `--attn-implementation eager`.

To evaluate a MASQuant-quantized TensorRT model through VLMEvalKit, also install TensorRT / TensorRT-LLM on the remote GPU machine and provide a builder script that imports MASQuant parameters into a TensorRT engine. Upstream MASQuant currently saves `mas_parameters.pth`; this repository does not assume a fixed TensorRT export entrypoint, and instead wires your builder through `--tensorrt-builder-command`.

## GAE + MASQuant

The prune-then-quant baseline has two phases.

Phase 1 calibrates MASQuant on already-pruned prompts:

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage calibrate \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --masquant-root "$MASQUANT_ROOT" \
  --work-dir "$WORK_ROOT/qwen25vl_gae50_masquant" \
  --calib-jsonl "$CALIB_JSONL" \
  --retention-ratio 0.5 \
  --min-keep 1 \
  --nsamples 128 \
  --batch-size 1 \
  --wbits 4 \
  --abits 8 \
  --epochs 2 \
  --attn-implementation eager \
  --gae-answer-source generated \
  --patch-masquant-inputs-embeds-mask
```

This writes:

- a pruned MASQuant cache under `$WORK_ROOT/qwen25vl_gae50_masquant/cache`;
- pruned activation scales under `$WORK_ROOT/qwen25vl_gae50_masquant/act_scales`;
- MASQuant outputs under `$WORK_ROOT/qwen25vl_gae50_masquant/masquant_outputs`.

Find the trained MASQuant parameters:

```bash
export MASQUANT_RESUME=$(find "$WORK_ROOT/qwen25vl_gae50_masquant/masquant_outputs" \
  -name mas_parameters.pth | sort | tail -n 1)

echo "$MASQUANT_RESUME"
```

Phase 2 runs inference with the MASQuant model and applies GAE pruning again:

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage infer \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --masquant-root "$MASQUANT_ROOT" \
  --work-dir "$WORK_ROOT/qwen25vl_gae50_masquant" \
  --eval-jsonl "$EVAL_JSONL" \
  --output-jsonl "$WORK_ROOT/qwen25vl_gae50_masquant/eval_gae50_masquant.jsonl" \
  --masquant-resume "$MASQUANT_RESUME" \
  --retention-ratio 0.5 \
  --min-keep 1 \
  --wbits 4 \
  --abits 8 \
  --attn-implementation eager \
  --gae-answer-source generated \
  --max-new-tokens 128
```

The important invariant is:

- calibration uses GAE-pruned prompts before MASQuant learns quantization parameters;
- inference loads MASQuant first, then applies GAE pruning to the prompt before generation.

### CMC Compensation

The full MASQuant workflow also includes CMC (Cross-Modal Compensation). After Phase 1 has produced `mas_parameters.pth`, run the CMC stage to generate a white matrix and low-rank adapters:

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage cmc \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --masquant-root "$MASQUANT_ROOT" \
  --work-dir "$WORK_ROOT/qwen25vl_gae50_masquant" \
  --masquant-resume "$MASQUANT_RESUME" \
  --wbits 4 \
  --abits 8 \
  --cmc-net qwen2.5-vl-7b \
  --cmc-cali-data-type vision-audio-only \
  --cmc-rank 0.2 \
  --cmc-quant-cmc 0 \
  --cmc-n-cali-samples 128 \
  --cmc-vision-json "$DATA_ROOT/masquant/sharegpt4v_filtered_coco.json" \
  --cmc-vision-prefix "$DATA_ROOT/coco/train2017"
```

By default this calls upstream MASQuant `infer_mas.py` and saves CMC artifacts to:

- `$WORK_ROOT/qwen25vl_gae50_masquant/cmc/white_matrix_vision-audio-only.pt`
- `$WORK_ROOT/qwen25vl_gae50_masquant/cmc/low_rank_adapters_quantcmc0_rank0.2_vision-audio-only.pt`

Note: CMC `--scales_path` means MAS-trained `mas_parameters.pth`, not raw activation scales. This script uses `--masquant-resume` as the CMC scales path by default; only pass `--cmc-scales-path` when you need to override it.

If your MASQuant checkout has a custom VL entrypoint, override it with `--cmc-script-name infer_mas_vl.py`. The `vision-audio-only` calibration data loading comes from upstream MASQuant, whose source hard-codes `/nas/yuehu/...` paths; `--cmc-vision-json` and `--cmc-vision-prefix` patch those to your local ShareGPT4V/COCO paths before running. For a quick plumbing check, use `--cmc-cali-data-type no-white`, which does not read COCO whitening data.

### Save A MASQuant TensorRT Artifact And Evaluate With VLMEvalKit

For the “quantize once, load directly at evaluation time” path, build the TensorRT engine after Phase 1 and CMC, then package the engine, MASQuant parameters, CMC artifacts, and processor into an artifact:

```bash
export MASQUANT_TRT_ARTIFACT="$WORK_ROOT/qwen25vl_gae50_masquant/masquant_trt_artifact"
export TRT_ENGINE_DIR="$MASQUANT_TRT_ARTIFACT/engine"
export CMC_LOW_RANK="$WORK_ROOT/qwen25vl_gae50_masquant/cmc/low_rank_adapters_quantcmc0_rank0.2_vision-audio-only.pt"
export CMC_WHITE="$WORK_ROOT/qwen25vl_gae50_masquant/cmc/white_matrix_vision-audio-only.pt"

python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage export-tensorrt \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --work-dir "$WORK_ROOT/qwen25vl_gae50_masquant" \
  --masquant-resume "$MASQUANT_RESUME" \
  --masquant-act-scales "$WORK_ROOT/qwen25vl_gae50_masquant/act_scales/Qwen2.5-VL-7B-Instruct-text-vision-128.pt" \
  --wbits 4 \
  --abits 8 \
  --cmc-low-rank-adapters "$CMC_LOW_RANK" \
  --cmc-white-matrix-path "$CMC_WHITE" \
  --tensorrt-engine-dir "$TRT_ENGINE_DIR" \
  --tensorrt-artifact-dir "$MASQUANT_TRT_ARTIFACT" \
  --tensorrt-builder-command "python -m prune_quant_baseline.scripts.build_masquant_tensorrt --model {model_path} --model-type qwen2vl --masquant-root $MASQUANT_ROOT --masquant-resume {masquant_resume} --act-scales {masquant_act_scales} --cmc-low-rank {cmc_low_rank_adapters} --cmc-white-matrix {cmc_white_matrix} --output {engine_dir} --wbits {wbits} --abits {abits} --convert-command 'python /path/to/masquant_trt_convert.py --model {hf_export_dir} --state {torch_export_dir}/masquant_state.pt --out {checkpoint_dir}' --llm-build-command 'trtllm-build --checkpoint_dir {checkpoint_dir} --output_dir {llm_engine_dir} --gemm_plugin=float16 --gpt_attention_plugin=float16 --max_batch_size 1 --max_input_len 2048 --max_seq_len 3072 --max_multimodal_len 1296' --vision-build-command 'python /path/to/build_qwen2vl_vision_engine.py --model {hf_export_dir} --output_dir {vision_engine_dir}'"
```

`build_masquant_tensorrt` first writes the MASQuant + CMC materialized model to `$TRT_ENGINE_DIR/.build/masquant_export`, then runs the TensorRT conversion/build commands you provide. Replace `/path/to/masquant_trt_convert.py` and `/path/to/build_qwen2vl_vision_engine.py` with scripts that really understand MASQuant `QuantLinear`, split modality scales, and CMC low-rank branches in your environment. TensorRT-LLM's stock Qwen2-VL converter usually does not understand MASQuant custom modules; for a plumbing-only TensorRT-LLM check, pass `--tensorrt-llm-root "$TENSORRT_LLM_ROOT" --allow-stock-trtllm-example` to the builder. If `TRT_ENGINE_DIR` was already built by another workflow, omit `--tensorrt-builder-command` and only write the artifact manifest. Non-dry-run mode checks that the engine directory is non-empty before registering it.

Then run VLMEvalKit against the saved TensorRT artifact:

```bash
cd "$PROJECT_ROOT"
python remote/install_vlmeval_pruned_gae.py --vlmeval-root "$VLMEVALKIT_ROOT"

cd "$VLMEVALKIT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export PQ_MASQUANT_TRT_ARTIFACT="$MASQUANT_TRT_ARTIFACT"
export PQ_MAX_NEW_TOKENS=16

python run.py --data MME MMStar MMVet --model Qwen2VL_MASQuant_TensorRT \
  --work-dir "$WORK_ROOT/vlmeval_qwen25vl_masquant_trt" \
  --verbose
```

`Qwen2VL_MASQuant_TensorRT` only reads the saved artifact. It does not rerun MASQuant calibration or rebuild the engine during VLMEvalKit inference. The default runtime uses TensorRT-LLM `ModelRunner`; if your engine needs a custom multimodal calling convention, set a custom runtime class in the manifest or override it with:

```bash
export PQ_TRT_RUNTIME_CLASS="your_package.your_module.YourTensorRTRuntime"
```

The custom runtime only needs to implement `generate(inputs, processor, max_new_tokens) -> str`.

## Useful Smoke Checks

Syntax and lightweight checks:

```bash
python -m compileall src tests
pytest
```

Dry-run the MASQuant command construction without loading the model:

```bash
python -m prune_quant_baseline.scripts.run_prune_then_quant_masquant \
  --stage calibrate \
  --model-type qwen2_5_vl \
  --model-path "$QWEN25VL_MODEL" \
  --masquant-root "$MASQUANT_ROOT" \
  --work-dir "$WORK_ROOT/dry_run" \
  --calib-jsonl "$CALIB_JSONL" \
  --dry-run
```

## Notes

- Do not commit model weights, datasets, Hugging Face tokens, or large benchmark outputs.
- If CUDA memory is tight, lower image resolution, use fewer calibration samples, or reduce `--max-new-tokens`.
- GAE is slower than attention-proxy pruning because it performs gradient-based scoring.
