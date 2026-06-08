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

Install the evaluation toolkits. VLMEvalKit remains the default path for MME/MMStar/MMVet-style runs, and `lmms-eval` is now tracked as this repository's `third_party/lmms-eval` submodule for OCRBench, VizWiz, ScienceQA, and TextVQA metrics.

```bash
export EXT_ROOT=/path/to/external
git clone https://github.com/open-compass/VLMEvalKit.git "$EXT_ROOT/VLMEvalKit"
export VLMEVALKIT_ROOT="$EXT_ROOT/VLMEvalKit"

cd "$VLMEVALKIT_ROOT"
pip install -e .

cd "$PROJECT_ROOT"
python remote/install_vlmeval_pruned_gae.py --vlmeval-root "$VLMEVALKIT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

git submodule update --init --recursive third_party/lmms-eval
cd "$PROJECT_ROOT/third_party/lmms-eval"
pip install -e ".[qwen]"
cd "$PROJECT_ROOT"
pip install -e .
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

VLMEvalKit handles dataset loading, prediction files, and metric computation for the existing MME/MMStar/MMVet path. Put its dataset cache under `DATA_ROOT`:

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

The lmms-eval task set used by this repository is:

| Capability | Benchmark | lmms-eval task |
| --- | --- | --- |
| OCR | OCRBench | `ocrbench` |
| Viz | VizWiz | `vizwiz_vqa_val` |
| S-QA | ScienceQA | `scienceqa_img` |
| T-VQA | TextVQA | `textvqa_val` |

Run the default lmms-eval suite through this repository's `prune_quant_qwen2vl` wrapper:

```bash
cd "$PROJECT_ROOT"
export QWEN2VL_MODEL="$MODEL_ROOT/Qwen2-VL-7B-Instruct"
export PQ_MODEL_TYPE=qwen2vl
export PQ_RETENTION_RATIO=0.5
export PQ_GAE_ANSWER_SOURCE=generated
export PQ_GAE_PER_TOKEN=false
export PQ_ATTN_IMPLEMENTATION=eager
export PQ_MIN_VISUAL_TOKENS=1500
export PQ_MAX_VISUAL_TOKENS=1500

python remote/run_lmms_eval_smart.py \
  --lmms-eval-root "$PROJECT_ROOT/third_party/lmms-eval" \
  --tasks ocrbench vizwiz_vqa_val scienceqa_img textvqa_val \
  --model prune_quant_qwen2vl \
  --model-path "$QWEN2VL_MODEL" \
  --output-path "$WORK_ROOT/lmms_eval_qwen2vl_gae50" \
  --cache "$WORK_ROOT/lmms_eval_cache" \
  --log-samples
```

For a quick plumbing check, add `--limit 8`. Use `PQ_RETENTION_RATIO=1.0` for the no-pruning lmms-eval baseline while keeping model and image-resolution settings fixed.

The lmms-eval smart runner enables response-cache resume whenever `--cache`
or `LMMS_EVAL_CACHE` is set. It derives a stable `LMMS_CACHE_RUN_ID` from the
model, model args, task list, batch size, limit, and output path, so rerunning
the same command after an interruption reopens the same lmms-eval run cache and
skips samples that already produced deterministic responses. Set
`LMMS_CACHE_RUN_ID` yourself only when you intentionally want multiple commands
to share one resume namespace.

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
```

`flash-attn` is optional for this baseline. GAE needs eager attention, and the prune-then-MASQuant script can patch MASQuant's Qwen2.5 loader to honor `--attn-implementation eager`.

VLMEvalKit evaluation can use MASQuant pseudo quant directly: the evaluation process loads the saved `mas_parameters.pth`, and when CMC `low_rank_adapters*.pt` is provided, applies it to the model. TensorRT / TensorRT-LLM is not required for this path.

## GAE + MASQuant

The prune-then-quant baseline has two phases.

For the complete Qwen2-VL MASQuant pseudo quant + CMC + VLMEvalKit MME workflow, use the script entrypoint:

```bash
cd "$PROJECT_ROOT"
cp remote/run_qwen2vl_masquant_pseudo_mme.example.sh remote/run_qwen2vl_masquant_pseudo_mme.local.sh
```

Edit paths and parameters at the top of `remote/run_qwen2vl_masquant_pseudo_mme.local.sh`, for example:

- `PROJECT_ROOT`: this repository path;
- `EXT_ROOT`: external checkout root;
- `MODEL_PATH`: Qwen2-VL model path;
- `WORK_DIR`: MASQuant intermediate artifact and VLMEvalKit output directory;
- `MASQUANT_ROOT`: `EfficientAI/masquant` path;
- `VLMEVALKIT_ROOT`: VLMEvalKit path;
- `CALIB_JSONL`: JSONL used by phase 1 MASQuant calibration;
- `CMC_VISION_JSON` / `CMC_VISION_PREFIX`: ShareGPT4V-style JSON and image directory used by CMC;
- `WBITS` / `ABITS` / `NSAMPLES` / `EPOCHS` / `CMC_RANK` / `MAX_NEW_TOKENS` and other experiment parameters.

Then run:

```bash
bash remote/run_qwen2vl_masquant_pseudo_mme.local.sh
```

The full pipeline logic is in `remote/run_masquant_pseudo_pipeline.sh`; the local script only stores paths and parameters. Set `RUN_CALIBRATE=0`, `RUN_CMC=0`, `RUN_INSTALL_VLMEVAL=0`, or `RUN_VLMEVAL=0` to skip completed stages. By default `CALIB_RETENTION_RATIO=1.0` and `EVAL_RETENTION_RATIO=1.0` mean no GAE pruning, evaluating MASQuant pseudo quant only; set either to a value such as `0.5` when you want pruning.

The VLMEvalKit stage uses `remote/run_vlmeval_smart.py` by default (`VLMEVAL_SMART_RUNNER=1`). It runs each dataset separately, reuses existing prediction xlsx files, switches to `--mode eval` when an xlsx already exists, and skips reruns when score files are already present. MME and MMStar default to `--judge exact_matching`, so they do not call GPT-as-judge. `VLMEVAL_DISABLE_OPENAI=1` still clears OpenAI-related environment variables for exact-matching runs to avoid accidental external API calls during scoring.

Useful overrides:

```bash
export VLMEVAL_MODE=auto          # auto, all, infer, or eval
export VLMEVAL_REUSE=1            # pass VLMEvalKit --reuse
export VLMEVAL_FORCE_EVAL=1       # rebuild score files even if one is found
export VLMEVAL_JUDGE=deepseek-v4-pro
export VLMEVAL_EXACT_MATCH_DATASETS="MME MMStar"
```

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

### Evaluate With VLMEvalKit Pseudo Quant

For the “quantize once, load directly at evaluation time” path, run VLMEvalKit against the saved MASQuant parameters and CMC artifacts after Phase 1 and CMC:

```bash
export WORK_DIR="$WORK_ROOT/qwen2vl_masquant"
export QWEN2VL_MODEL=/home/aistudio/datasets/models/Qwen2-VL-7B-Instruct
export MASQUANT_ROOT=/home/aistudio/EXT/EfficientAI/masquant
export MASQUANT_RESUME=$(find "$WORK_DIR/masquant_outputs" -name mas_parameters.pth | sort | tail -n 1)
export MASQUANT_ACT_SCALES="$WORK_DIR/act_scales/Qwen2-VL-7B-Instruct-text-vision-128.pt"
export CMC_LOW_RANK="$WORK_DIR/cmc/low_rank_adapters_quantcmc0_rank0.2_vision-audio-only.pt"
export CMC_WHITE="$WORK_DIR/cmc/white_matrix_vision-audio-only.pt"

cd "$PROJECT_ROOT"
python remote/install_vlmeval_pruned_gae.py --vlmeval-root "$VLMEVALKIT_ROOT"

cd "$VLMEVALKIT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export PQ_MODEL_TYPE=qwen2vl
export PQ_MASQUANT_WBITS=4
export PQ_MASQUANT_ABITS=8
export PQ_MAX_NEW_TOKENS=16
export PQ_RETENTION_RATIO=1.0

python run.py --data MME --model Qwen2VL_MASQuant_Pseudo \
  --work-dir "$WORK_DIR/vlmeval_mme_masquant_pseudo" \
  --verbose
```

`Qwen2VL_MASQuant_Pseudo` does not rerun MASQuant calibration and does not build a TensorRT engine. It loads `MASQUANT_RESUME` when the evaluation process starts, and loads CMC compensation when `CMC_LOW_RANK` is set. `PQ_RETENTION_RATIO=1.0` evaluates only MASQuant quantization; set it to a value such as `0.5` to evaluate MASQuant plus GAE pruning.

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
