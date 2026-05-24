import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from prune_quant_baseline.core.config import load_config
from prune_quant_baseline.core.logging_utils import configure_logging, get_logger
from prune_quant_baseline.models.llava_onevision_hf import LlavaOneVisionHFAdapter
from prune_quant_baseline.models.qwen2vl_hf import Qwen2VLHFAdapter
from prune_quant_baseline.pruners.attention_proxy import AttentionProxyPruner
from prune_quant_baseline.quant.loaders import load_model_and_processor


LOGGER = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run remote pruned inference for JSONL samples.")
    parser.add_argument("--config", help="Optional YAML config path.")
    parser.add_argument("--model-type", choices=["llava_onevision", "qwen2vl"])
    parser.add_argument("--model-path")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--pruner", choices=["attention_proxy", "gae_oracle", "learned_compressor"])
    parser.add_argument("--retention-ratio", type=float)
    parser.add_argument("--min-keep", type=int)
    parser.add_argument("--quant-method", choices=["none", "bnb4", "bnb8", "gptq", "awq"])
    parser.add_argument("--dtype")
    parser.add_argument("--device-map")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _merge_args_with_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config:
        config = load_config(args.config)
    model = config.get("model", {})
    pruning = config.get("pruning", {})
    quant = config.get("quant", {})
    inference = config.get("inference", {})
    data = config.get("data", {})
    return {
        "model_type": args.model_type or model.get("model_type"),
        "model_path": args.model_path or model.get("model_path"),
        "input_jsonl": args.input_jsonl or data.get("input_jsonl"),
        "output_jsonl": args.output_jsonl or data.get("output_jsonl"),
        "pruner": args.pruner or pruning.get("method", "attention_proxy"),
        "retention_ratio": args.retention_ratio if args.retention_ratio is not None else pruning.get("retention_ratio", 0.5),
        "min_keep": args.min_keep if args.min_keep is not None else pruning.get("min_keep", 1),
        "quant_method": args.quant_method or quant.get("method", "none"),
        "dtype": args.dtype or model.get("dtype", "bfloat16"),
        "device_map": args.device_map or model.get("device_map", "auto"),
        "max_new_tokens": args.max_new_tokens or inference.get("max_new_tokens", 128),
        "trust_remote_code": model.get("trust_remote_code", True),
        "local_files_only": model.get("local_files_only", True),
    }


def _make_adapter(model_type: str):
    if model_type == "llava_onevision":
        return LlavaOneVisionHFAdapter()
    if model_type == "qwen2vl":
        return Qwen2VLHFAdapter()
    raise ValueError("model_type must be one of: llava_onevision, qwen2vl.")


def _make_pruner(name: str):
    if name == "attention_proxy":
        return AttentionProxyPruner()
    raise NotImplementedError(f"Pruner {name!r} is not implemented for first-stage remote inference.")


def _read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)
    cfg = _merge_args_with_config(args)
    required = [key for key in ("model_type", "model_path", "input_jsonl", "output_jsonl") if not cfg.get(key)]
    if required:
        raise ValueError(f"Missing required argument/config value(s): {', '.join(required)}.")

    model, processor = load_model_and_processor(
        model_id_or_path=cfg["model_path"],
        model_type=cfg["model_type"],
        quant_method=cfg["quant_method"],
        dtype=cfg["dtype"],
        device_map=cfg["device_map"],
        trust_remote_code=cfg["trust_remote_code"],
        local_files_only=cfg["local_files_only"],
    )
    adapter = _make_adapter(cfg["model_type"])
    _make_pruner(cfg["pruner"])

    output_path = Path(cfg["output_jsonl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for sample in _read_jsonl(cfg["input_jsonl"]):
            prompt = sample.get("prompt", "")
            inputs = adapter.prepare_inputs(processor, sample)
            meta = adapter.get_visual_token_meta(model, inputs)
            raise NotImplementedError(
                "End-to-end compressed generate is not implemented until the adapter can reliably "
                "locate visual token positions and the target HF model accepts compressed inputs_embeds."
            )
            record = {
                "id": sample.get("id"),
                "prompt": prompt,
                "prediction": "",
                "retention_ratio": cfg["retention_ratio"],
                "num_visual_tokens_before": int(meta.visual_indices.numel()),
                "num_visual_tokens_after": None,
                "quant_method": cfg["quant_method"],
                "model_type": cfg["model_type"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
