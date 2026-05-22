import argparse
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from transformers import AutoProcessor


DEFAULT_IGNORE = ["lm_head", "re:visual.*", "re:model.visual.*", "re:.*vision_tower.*"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a vLLM-loadable W4A16 checkpoint with llm-compressor."
    )
    parser.add_argument("--model", default="qwen2_vl", choices=["qwen2_vl", "qwen2_5_vl"])
    parser.add_argument("--model_id", required=True, help="HF model id or local fp16/bf16 checkpoint path.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calib_pairs", default=None,
                        help="JSON/JSONL file with {images/image, question, answer/caption}. "
                             "If omitted, --dataset_id is used.")
    parser.add_argument("--dataset_id", default="lmms-lab/flickr30k")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--question", default="What does this image show?")
    parser.add_argument("--w_group", type=int, default=128)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_sample_generation", action="store_true")
    parser.add_argument("--sample_max_new_tokens", type=int, default=80)
    return parser.parse_args()


def load_model_class(model_name: str):
    if model_name == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration, "Qwen2VLDecoderLayer"
    if model_name == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration, "Qwen2_5_VLDecoderLayer"
    raise ValueError(f"Unsupported model: {model_name}")


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    def normalize(item, idx):
        if "images" not in item and "image" in item:
            item["images"] = item.pop("image")
        images = item.get("images") or []
        if isinstance(images, (str, os.PathLike)):
            images = [str(images)]
        item["images"] = list(images)
        item["question"] = re.sub(r"\s*<image>\s*", " ", item.get("question", ""), flags=re.I).strip()
        item["answer"] = item.get("answer") or item.get("caption") or ""
        item.setdefault("id", idx)
        return item

    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    items.append(normalize(json.loads(line), i))
        return items

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_items = data if isinstance(data, list) else data.get("data", [])
    return [normalize(item, i) for i, item in enumerate(raw_items)]


def pil_to_data_uri(image: Image.Image) -> str:
    with BytesIO() as buffered:
        image.convert("RGB").save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded}"


def local_pairs_dataset(path: str, n_samples: int, seed: int) -> Dataset:
    items = load_json_or_jsonl(path)
    ds = Dataset.from_list(items)
    if len(ds) > n_samples:
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds


def hf_vision_dataset(dataset_id: str, split: str, n_samples: int, seed: int) -> Dataset:
    ds = load_dataset(dataset_id, split=split)
    ds = ds.shuffle(seed=seed)
    if len(ds) > n_samples:
        ds = ds.select(range(n_samples))
    return ds


def first_caption(example: Dict[str, Any]) -> str:
    for key in ("answer", "caption", "captions", "text"):
        value = example.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            return " ".join(str(x) for x in value)
    return ""


def first_image(example: Dict[str, Any]) -> Image.Image:
    if "image" in example and isinstance(example["image"], Image.Image):
        return example["image"]
    images = example.get("images") or []
    if isinstance(images, str):
        images = [images]
    if not images:
        raise ValueError("Calibration sample has no image.")
    with Image.open(images[0]) as image:
        return image.convert("RGB")


def build_messages(example: Dict[str, Any], default_question: str):
    image = first_image(example)
    question = example.get("question") or default_question
    answer = first_caption(example)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_to_data_uri(image)},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


def make_tokenizer_fn(processor, max_seq_length: int, default_question: str):
    from qwen_vl_utils import process_vision_info

    def preprocess_and_tokenize(example):
        messages = build_messages(example, default_question)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=max_seq_length,
            truncation=True,
        )

    return preprocess_and_tokenize


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def main():
    args = parse_args()
    if args.w_group != 128:
        raise ValueError("llm-compressor scheme='W4A16' uses group size 128; keep --w_group 128 for now.")

    try:
        from llmcompressor import oneshot
        try:
            from llmcompressor.modifiers.gptq import GPTQModifier
        except ImportError:
            from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    except ImportError as exc:
        raise ImportError("Install llm-compressor first: `pip install llmcompressor`.") from exc

    model_cls, sequential_target = load_model_class(args.model)
    model_kwargs = {
        "torch_dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    model = model_cls.from_pretrained(
        args.model_id,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    if args.calib_pairs:
        ds = local_pairs_dataset(args.calib_pairs, args.n_samples, args.seed)
    else:
        ds = hf_vision_dataset(args.dataset_id, args.dataset_split, args.n_samples, args.seed)

    tokenize = make_tokenizer_fn(processor, args.max_seq_length, args.question)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    recipe = [
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            sequential_targets=[sequential_target],
            ignore=DEFAULT_IGNORE,
        )
    ]

    oneshot(
        model=model,
        tokenizer=args.model_id,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=min(args.n_samples, len(ds)),
        trust_remote_code_model=args.trust_remote_code,
        data_collator=data_collator,
    )

    if not args.skip_sample_generation and args.calib_pairs:
        from compressed_tensors.offload import dispatch_model

        dispatch_model(model)
        sample = load_json_or_jsonl(args.calib_pairs)[0]
        messages = build_messages(sample, args.question)[:1]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=args.sample_max_new_tokens)
        print(processor.decode(output[0], skip_special_tokens=True))

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, save_compressed=True)
    processor.save_pretrained(args.output_dir)
    print(f"[OK] Saved vLLM W4A16 checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
