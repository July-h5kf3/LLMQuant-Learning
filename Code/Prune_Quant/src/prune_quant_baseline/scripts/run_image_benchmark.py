import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image

from prune_quant_baseline.core.logging_utils import configure_logging, get_logger
from prune_quant_baseline.scripts.run_infer_pruned import (
    _build_pruned_generation_inputs,
    _generate_from_pruned_inputs,
    _generate_vanilla,
    _make_adapter,
    _make_pruner,
    _move_inputs_to_model_device,
    _score_attention_proxy,
    _score_gae_oracle,
)
from prune_quant_baseline.pruners.attention_proxy import AttentionProxyPruner
from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner
from prune_quant_baseline.quant.loaders import load_model_and_processor


LOGGER = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen2-VL image benchmark TSV files.")
    parser.add_argument("--dataset", required=True, choices=["MME", "MMStar", "MMVet"])
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metrics-json")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", default="qwen2vl", choices=["qwen2vl"])
    parser.add_argument("--pruner", default="none", choices=["none", "attention_proxy", "gae_oracle"])
    parser.add_argument("--retention-ratio", type=float, default=1.0)
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--quant-method", default="none", choices=["none", "bnb4", "bnb8", "gptq", "awq"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--processor-use-fast", choices=["true", "false"])
    parser.add_argument(
        "--mme-prompt-style",
        default="default",
        choices=["default", "qwen_vl", "gpt4v", "original"],
    )
    parser.add_argument("--jpeg-reencode", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gae-answer-source", choices=["sample", "generated"], default="sample")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-vanilla-fallback", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _decode_image(value: Any, cache_key: str, image_cache: dict[str, Image.Image]) -> Image.Image:
    raw = "" if pd.isna(value) else str(value)
    if len(raw) > 16:
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        image_cache[cache_key] = image
        return image.copy()
    if cache_key in image_cache:
        return image_cache[cache_key].copy()
    raise ValueError(f"Missing base64 image and no cached image for {cache_key!r}.")


def _format_question(dataset: str, row: dict[str, Any], *, mme_prompt_style: str = "default") -> str:
    question = str(row["question"])
    if dataset == "MMStar":
        options = "\n".join(f"{opt}. {row[opt]}" for opt in ["A", "B", "C", "D"])
        return (
            f"{question}\n{options}\n"
            "Answer with the option letter only, one of A, B, C, or D."
        )
    if dataset == "MME":
        if mme_prompt_style == "original":
            return question
        question = question.replace(" Please answer yes or no.", "")
        if mme_prompt_style == "qwen_vl":
            return f"{question} Answer:"
        if mme_prompt_style == "gpt4v":
            return f"{question}\nAnswer the question with Yes or No."
        return f"{question}\nAnswer the question using a single word or phrase."
    return question


def _extract_yes_no(text: str) -> str:
    pred = text.lower().strip().replace(".", "")
    if pred in ["yes", "no"]:
        return pred.capitalize()
    if len(pred) == 1:
        if pred == "y":
            return "Yes"
        if pred == "n":
            return "No"
        return ""
    prefix = pred[:4]
    if "yes" in prefix:
        return "Yes"
    if "no" in prefix:
        return "No"
    return ""


def _extract_option(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else ""


def _normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9.\/-]+", " ", text.lower()).strip()


def _mmvet_proxy_correct(prediction: str, answer: str) -> bool:
    pred = _normalize_answer(prediction)
    and_parts = str(answer).split("<AND>")
    for and_part in and_parts:
        alternatives = [_normalize_answer(part) for part in and_part.split("<OR>")]
        if not any(alt and alt in pred for alt in alternatives):
            return False
    return True


def _score_predictions(dataset: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"dataset": dataset, "num_samples": 0}
    if dataset == "MMStar":
        correct = 0
        for record in records:
            pred = _extract_option(record["prediction"])
            record["parsed_prediction"] = pred
            record["correct"] = pred == str(record["answer"]).strip().upper()
            correct += int(record["correct"])
        return {"dataset": dataset, "num_samples": len(records), "accuracy": correct / len(records) * 100}
    if dataset == "MME":
        correct = 0
        by_category: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            pred = _extract_yes_no(record["prediction"])
            gold = str(record["answer"]).strip().capitalize()
            record["parsed_prediction"] = pred
            record["correct"] = pred == gold
            correct += int(record["correct"])
            by_category.setdefault(str(record.get("category", "unknown")), []).append(record)
        category_scores = {}
        total_score = 0.0
        for category, items in by_category.items():
            acc = sum(int(item["correct"]) for item in items) / len(items) * 100
            grouped: dict[str, list[dict[str, Any]]] = {}
            for item in items:
                grouped.setdefault(str(item.get("image_path", item.get("index"))), []).append(item)
            acc_plus = sum(all(x["correct"] for x in group) for group in grouped.values()) / len(grouped) * 100
            score = acc + acc_plus
            category_scores[category] = {"accuracy": acc, "accuracy_plus": acc_plus, "score": score}
            total_score += score
        return {
            "dataset": dataset,
            "num_samples": len(records),
            "accuracy": correct / len(records) * 100,
            "mme_score": total_score,
            "category_scores": category_scores,
        }
    if dataset == "MMVet":
        correct = 0
        for record in records:
            record["correct_proxy"] = _mmvet_proxy_correct(record["prediction"], str(record["answer"]))
            correct += int(record["correct_proxy"])
        return {
            "dataset": dataset,
            "num_samples": len(records),
            "exact_match_proxy": correct / len(records) * 100,
            "note": "MMVet official score requires GPT/LLM judging; this is a strict local proxy.",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def _jpeg_reencode(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _load_done(output_jsonl: Path) -> tuple[set[str], list[dict[str, Any]]]:
    done: set[str] = set()
    records: list[dict[str, Any]] = []
    if not output_jsonl.exists():
        return done, records
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            done.add(str(record["index"]))
            records.append(record)
    return done, records


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done, records = _load_done(output_jsonl) if args.resume else (set(), [])

    df = pd.read_csv(args.tsv, sep="\t")
    if args.limit is not None:
        df = df.head(args.limit)

    model, processor = load_model_and_processor(
        model_id_or_path=args.model_path,
        model_type=args.model_type,
        quant_method=args.quant_method,
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        processor_use_fast=None if args.processor_use_fast is None else args.processor_use_fast == "true",
    )
    model.eval()
    adapter = _make_adapter(args.model_type)
    pruner = None if args.pruner == "none" else _make_pruner(args.pruner)
    image_cache: dict[str, Image.Image] = {}

    mode = "a" if args.resume else "w"
    with output_jsonl.open(mode, encoding="utf-8") as out_f:
        for row_idx, row in df.iterrows():
            row_dict = row.to_dict()
            index = str(row_dict.get("index", row_idx))
            if index in done:
                continue
            image_key = str(row_dict.get("image_path", index))
            sample = {
                "id": index,
                "image": _decode_image(row_dict.get("image"), image_key, image_cache),
                "prompt": _format_question(args.dataset, row_dict, mme_prompt_style=args.mme_prompt_style),
                "answer": str(row_dict.get("answer", "")),
            }
            if args.jpeg_reencode:
                sample["image"] = _jpeg_reencode(sample["image"])
            inputs = _move_inputs_to_model_device(model, adapter.prepare_inputs(processor, sample))
            pruning_applied = False
            num_visual_tokens_before = None
            num_visual_tokens_after = None
            try:
                if pruner is None or args.retention_ratio >= 1.0:
                    prediction = _generate_vanilla(model, processor, inputs, args.max_new_tokens)
                else:
                    meta = adapter.get_visual_token_meta(model, inputs)
                    num_visual_tokens_before = int(meta.visual_indices.numel())
                    if isinstance(pruner, AttentionProxyPruner):
                        scores = _score_attention_proxy(model, pruner, inputs, meta)
                    elif isinstance(pruner, GAEOraclePruner):
                        answer = sample["answer"]
                        if args.gae_answer_source == "generated" or not answer:
                            LOGGER.info("Generating oracle replay answer for sample %s.", index)
                            answer = _generate_vanilla(model, processor, inputs, args.max_new_tokens)
                        if not answer:
                            raise ValueError("GAE oracle requires a non-empty sample answer or generated answer.")
                        scores = _score_gae_oracle(
                            model=model,
                            processor=processor,
                            adapter=adapter,
                            pruner=pruner,
                            sample=sample,
                            answer=answer,
                        )
                    else:
                        raise NotImplementedError(f"Unsupported pruner: {type(pruner)!r}")
                    pruned_inputs, num_visual_tokens_before, num_visual_tokens_after = _build_pruned_generation_inputs(
                        model=model,
                        adapter=adapter,
                        inputs=inputs,
                        scores=scores,
                        retention_ratio=args.retention_ratio,
                        min_keep=args.min_keep,
                    )
                    prediction = _generate_from_pruned_inputs(
                        model=model,
                        processor=processor,
                        pruned_inputs=pruned_inputs,
                        max_new_tokens=args.max_new_tokens,
                    )
                    pruning_applied = True
            except Exception as exc:
                if not args.allow_vanilla_fallback:
                    raise
                LOGGER.warning("Sample %s failed under pruning; using vanilla fallback: %s", index, exc)
                prediction = _generate_vanilla(model, processor, inputs, args.max_new_tokens)

            record = {
                "index": index,
                "dataset": args.dataset,
                "question": str(row_dict.get("question", "")),
                "answer": str(row_dict.get("answer", "")),
                "prediction": prediction,
                "category": str(row_dict.get("category", "")),
                "image_path": str(row_dict.get("image_path", "")),
                "pruner": args.pruner,
                "retention_ratio": args.retention_ratio,
                "mme_prompt_style": args.mme_prompt_style if args.dataset == "MME" else None,
                "jpeg_reencode": args.jpeg_reencode,
                "num_visual_tokens_before": num_visual_tokens_before,
                "num_visual_tokens_after": num_visual_tokens_after,
                "pruning_applied": pruning_applied,
            }
            for key in ["l2_category", "bench", "A", "B", "C", "D"]:
                if key in row_dict:
                    record[key] = str(row_dict[key])
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            records.append(record)
            if (len(records) % 20) == 0:
                LOGGER.info("Processed %d records for %s.", len(records), args.dataset)

    metrics = _score_predictions(args.dataset, records)
    metrics["pruner"] = args.pruner
    metrics["retention_ratio"] = args.retention_ratio
    if args.metrics_json:
        Path(args.metrics_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
