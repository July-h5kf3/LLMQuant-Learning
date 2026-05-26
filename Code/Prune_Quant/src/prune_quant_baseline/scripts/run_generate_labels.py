import argparse
import json
from pathlib import Path
from typing import Sequence

from prune_quant_baseline.core.logging_utils import configure_logging
from prune_quant_baseline.models.qwen2vl_hf import Qwen2VLHFAdapter
from prune_quant_baseline.pruners.attention_proxy import AttentionProxyPruner
from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner
from prune_quant_baseline.quant.loaders import load_model_and_processor
from prune_quant_baseline.scripts.run_infer_pruned import (
    _generate_vanilla,
    _move_inputs_to_model_device,
    _read_jsonl,
    _score_attention_proxy,
    _score_gae_oracle,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GAE oracle labels on a remote machine.")
    parser.add_argument("--model-type", default="qwen2vl", choices=["qwen2vl"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--image-root", help="Optional root used to resolve relative image/image_path fields.")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gae-answer-source", choices=["sample", "generated"], default="sample")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--processor-use-fast", choices=["true", "false"])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)
    processor_use_fast = None if args.processor_use_fast is None else args.processor_use_fast == "true"
    model, processor = load_model_and_processor(
        model_id_or_path=args.model_path,
        model_type=args.model_type,
        quant_method="none",
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        processor_use_fast=processor_use_fast,
    )
    model.eval()
    adapter = Qwen2VLHFAdapter()
    proxy_pruner = AttentionProxyPruner()
    gae_pruner = GAEOraclePruner()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for row_idx, sample in enumerate(_read_jsonl(args.input_jsonl)):
            if args.limit is not None and row_idx >= args.limit:
                break
            if args.image_root:
                for key in ("image", "image_path"):
                    value = sample.get(key)
                    if isinstance(value, str) and not Path(value).is_absolute():
                        sample[key] = str(Path(args.image_root) / value)
            inputs = adapter.prepare_inputs(processor, sample)
            inputs = _move_inputs_to_model_device(model, inputs)
            meta = adapter.get_visual_token_meta(model, inputs)
            proxy_scores = _score_attention_proxy(model, proxy_pruner, inputs, meta)
            answer = str(sample.get("answer") or "").strip()
            if args.gae_answer_source == "generated" or not answer:
                answer = _generate_vanilla(model, processor, inputs, args.max_new_tokens)
            gae_scores = _score_gae_oracle(
                model=model,
                processor=processor,
                adapter=adapter,
                pruner=gae_pruner,
                sample=sample,
                answer=answer or "Yes",
            )
            record = {
                "id": sample.get("id", str(row_idx)),
                "num_visual_tokens": int(meta.visual_indices.numel()),
                "proxy_attention": proxy_scores.detach().float().cpu().tolist(),
                "gae_scores": gae_scores.detach().float().cpu().tolist(),
                "answer": answer,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
