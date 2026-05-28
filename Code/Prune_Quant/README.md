# Prune + Quant Baseline

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

Create or download your evaluation JSONL file:

```bash
export EVAL_JSONL="$DATA_ROOT/eval.jsonl"
```

For quick MME evaluation, the benchmark helper can download MME through `datasets`:

```bash
python -m prune_quant_baseline.scripts.run_image_benchmark \
  --dataset MME \
  --dataset-source hf \
  --hf-dataset lmms-lab/MME \
  --hf-split test \
  --hf-cache-dir "$DATA_ROOT/hf_cache" \
  --model-path "$QWEN2VL_MODEL" \
  --model-type qwen2vl \
  --output-jsonl "$WORK_ROOT/mme_smoke.jsonl" \
  --pruner none \
  --limit 4
```

You can also pre-download common test datasets into the Hugging Face cache:

```bash
huggingface-cli download lmms-lab/MME --repo-type dataset \
  --local-dir "$DATA_ROOT/hf_datasets/MME"

huggingface-cli download zli12321/mmstar --repo-type dataset \
  --local-dir "$DATA_ROOT/hf_datasets/MMStar"

huggingface-cli download lmms-lab/MMVet --repo-type dataset \
  --local-dir "$DATA_ROOT/hf_datasets/MMVet"
```

Current `run_image_benchmark.py` supports direct HF loading for MME. For MMStar/MMVet/MME TSV files produced by VLMEvalKit, use `run_image_benchmark.py` with `--dataset-source tsv --tsv /path/to/file.tsv`. If you want to evaluate MMStar/MMVet directly from Hugging Face rows, convert them to the JSONL format shown above and run `run_infer_pruned.py`.

## Pure GAE Prune

Run pure GAE-guided pruning without quantization on a JSONL file:

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

Run pure GAE pruning on MME through the benchmark helper:

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
  --gae-answer-source generated \
  --max-new-tokens 16
```

If the MME data is already available locally as a VLMEvalKit-style TSV, do not use the Hugging Face options. Use `--dataset-source tsv` and pass the TSV path:

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
  --gae-answer-source generated \
  --max-new-tokens 16
```

The TSV should contain the usual VLMEvalKit columns, including `image` as base64 image data plus `question` and `answer`. Optional columns such as `category`, `question_id`, and `image_path` are preserved in the output.

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
