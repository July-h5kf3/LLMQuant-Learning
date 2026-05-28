import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from prune_quant_baseline.core.config import load_config
from prune_quant_baseline.core.logging_utils import configure_logging, get_logger
from prune_quant_baseline.models.llava_onevision_hf import LlavaOneVisionHFAdapter
from prune_quant_baseline.models.qwen2vl_hf import Qwen2VLHFAdapter
from prune_quant_baseline.pruners.gae_oracle import (
    GAEOraclePruner,
    compute_answer_logprob_target,
    compute_answer_token_logprobs,
    normalize_relevance_scores,
)
from prune_quant_baseline.pruners.attention_proxy import AttentionProxyPruner
from prune_quant_baseline.pruners.learned_compressor import LearnedCompressorPruner
from prune_quant_baseline.pruners.token_gather import (
    build_keep_indices,
    gather_sequence_tensors,
    select_topk_visual_tokens,
)
from prune_quant_baseline.quant.loaders import load_model_and_processor


LOGGER = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run remote pruned inference for JSONL samples.")
    parser.add_argument("--config", help="Optional YAML config path.")
    parser.add_argument("--model-type", choices=["llava_onevision", "qwen2vl", "qwen2_5_vl"])
    parser.add_argument("--model-path")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--pruner", choices=["attention_proxy", "gae_oracle", "learned_compressor"])
    parser.add_argument("--retention-ratio", type=float)
    parser.add_argument("--min-keep", type=int)
    parser.add_argument("--quant-method", choices=["none", "bnb4", "bnb8", "gptq", "awq", "masquant"])
    parser.add_argument("--masquant-root")
    parser.add_argument("--masquant-resume")
    parser.add_argument("--masquant-act-scales")
    parser.add_argument("--masquant-wbits", type=int)
    parser.add_argument("--masquant-abits", type=int)
    parser.add_argument("--masquant-group-size", type=int)
    parser.add_argument("--masquant-inference-mode", choices=["split_scales", "merged_scales"])
    parser.add_argument("--masquant-symmetric", choices=["true", "false"])
    parser.add_argument("--masquant-batch-size", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--device-map")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, help="Optional max number of samples to process.")
    parser.add_argument("--attn-implementation", default="eager", help="Use 'none' to leave the HF default unchanged.")
    parser.add_argument("--processor-use-fast", choices=["true", "false"])
    parser.add_argument("--gae-answer-source", choices=["sample", "generated"], default="sample")
    parser.add_argument("--gae-per-token", choices=["true", "false"], default="true")
    parser.add_argument("--log-oracle-replay", action="store_true")
    parser.add_argument("--compressor-checkpoint")
    parser.add_argument("--allow-vanilla-fallback", action="store_true")
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
        "masquant_root": args.masquant_root or quant.get("masquant_root"),
        "masquant_resume": args.masquant_resume or quant.get("masquant_resume"),
        "masquant_act_scales": args.masquant_act_scales or quant.get("masquant_act_scales"),
        "masquant_wbits": args.masquant_wbits if args.masquant_wbits is not None else quant.get("wbits", 4),
        "masquant_abits": args.masquant_abits if args.masquant_abits is not None else quant.get("abits", 8),
        "masquant_group_size": (
            args.masquant_group_size if args.masquant_group_size is not None else quant.get("group_size", 0)
        ),
        "masquant_inference_mode": args.masquant_inference_mode or quant.get("inference_mode", "split_scales"),
        "masquant_symmetric": (
            args.masquant_symmetric == "true"
            if args.masquant_symmetric is not None
            else bool(quant.get("symmetric", True))
        ),
        "masquant_batch_size": args.masquant_batch_size if args.masquant_batch_size is not None else quant.get("batch_size", 1),
        "dtype": args.dtype or model.get("dtype", "bfloat16"),
        "device_map": args.device_map or model.get("device_map", "auto"),
        "max_new_tokens": args.max_new_tokens or inference.get("max_new_tokens", 128),
        "limit": args.limit,
        "attn_implementation": None if args.attn_implementation == "none" else args.attn_implementation,
        "processor_use_fast": None if args.processor_use_fast is None else args.processor_use_fast == "true",
        "gae_answer_source": args.gae_answer_source,
        "gae_per_token": args.gae_per_token == "true",
        "log_oracle_replay": args.log_oracle_replay,
        "compressor_checkpoint": args.compressor_checkpoint or pruning.get("compressor_checkpoint"),
        "allow_vanilla_fallback": args.allow_vanilla_fallback,
        "trust_remote_code": model.get("trust_remote_code", True),
        "local_files_only": model.get("local_files_only", True),
    }


def _make_adapter(model_type: str):
    if model_type == "llava_onevision":
        return LlavaOneVisionHFAdapter()
    if model_type in {"qwen2vl", "qwen2_5_vl"}:
        return Qwen2VLHFAdapter()
    raise ValueError("model_type must be one of: llava_onevision, qwen2vl, qwen2_5_vl.")


def _make_pruner(name: str, *, checkpoint_path: str | None = None):
    if name == "attention_proxy":
        return AttentionProxyPruner()
    if name == "gae_oracle":
        return GAEOraclePruner()
    if name == "learned_compressor":
        if not checkpoint_path:
            raise ValueError("learned_compressor requires --compressor-checkpoint.")
        return LearnedCompressorPruner(checkpoint_path)
    raise NotImplementedError(f"Pruner {name!r} is not implemented.")


def _read_jsonl(path: str | Path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return
    if text.startswith("["):
        for item in json.loads(text):
            yield item
        return
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def _move_inputs_to_model_device(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return inputs
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _decode_prediction(processor: Any, generated_ids: Any, input_len: int | None) -> str:
    if input_len is not None:
        generated_ids = generated_ids[:, input_len:]
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if hasattr(processor, "decode"):
        return processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    raise ValueError("Processor does not provide decode or batch_decode.")


def _decode_token_ids(processor: Any, token_ids: Any) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    if hasattr(processor, "decode"):
        return processor.decode(token_ids[0], skip_special_tokens=True).strip()
    raise ValueError("Processor does not provide decode or batch_decode.")


def _generate_vanilla(model: Any, processor: Any, inputs: dict[str, Any], max_new_tokens: int) -> str:
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else None
    with __import__("torch").no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    return _decode_prediction(processor, generated_ids, input_len)


def _generate_from_pruned_inputs(
    *,
    model: Any,
    processor: Any,
    pruned_inputs: dict[str, Any],
    max_new_tokens: int,
) -> str:
    """Greedy decode from a compressed prompt represented by inputs_embeds."""

    import torch

    generated: list[torch.Tensor] = []
    attention_mask = pruned_inputs.get("attention_mask")
    position_ids = pruned_inputs.get("position_ids")
    if attention_mask is None:
        attention_mask = torch.ones(
            pruned_inputs["inputs_embeds"].shape[:2],
            dtype=torch.long,
            device=pruned_inputs["inputs_embeds"].device,
        )
    cache_position = torch.arange(
        pruned_inputs["inputs_embeds"].shape[1],
        device=pruned_inputs["inputs_embeds"].device,
        dtype=torch.long,
    )
    with torch.no_grad():
        outputs = model(
            inputs_embeds=pruned_inputs["inputs_embeds"],
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_token_ids = {eos_token_id}
        elif eos_token_id is None:
            eos_token_ids = set()
        else:
            eos_token_ids = {int(item) for item in eos_token_id}
        next_position_ids = None
        if position_ids is not None:
            if position_ids.dim() == 3:
                next_position_ids = position_ids[:, :, -1:] + 1
            elif position_ids.dim() == 2:
                next_position_ids = position_ids[:, -1:] + 1
        for _ in range(max_new_tokens):
            generated.append(next_token)
            if eos_token_ids and int(next_token.item()) in eos_token_ids:
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)],
                dim=-1,
            )
            step_kwargs: dict[str, Any] = {
                "input_ids": next_token,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "cache_position": cache_position[-1:] + 1,
                "use_cache": True,
            }
            if next_position_ids is not None:
                step_kwargs["position_ids"] = next_position_ids
            outputs = model(**step_kwargs)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cache_position = torch.cat([cache_position, step_kwargs["cache_position"]])
            if next_position_ids is not None:
                next_position_ids = next_position_ids + 1
    if not generated:
        return ""
    return _decode_token_ids(processor, torch.cat(generated, dim=-1))


def _build_pruned_generation_inputs(
    *,
    model: Any,
    adapter: Any,
    inputs: dict[str, Any],
    scores: Any,
    retention_ratio: float,
    min_keep: int,
) -> tuple[dict[str, Any], int, int]:
    import torch

    meta = adapter.get_visual_token_meta(model, inputs)
    if scores.numel() != meta.visual_indices.numel():
        raise ValueError(
            f"Score count {scores.numel()} does not match visual token count {meta.visual_indices.numel()}."
        )
    kept_visual = select_topk_visual_tokens(
        meta.visual_indices,
        scores.to(meta.visual_indices.device),
        retention_ratio=retention_ratio,
        min_keep=min_keep,
    )
    if kept_visual.numel() == meta.visual_indices.numel():
        return inputs, int(meta.visual_indices.numel()), int(kept_visual.numel())
    keep_indices = build_keep_indices(
        seq_len=inputs["input_ids"].shape[-1],
        visual_indices=meta.visual_indices,
        kept_visual_indices=kept_visual,
        device=inputs["input_ids"].device,
    )
    with torch.no_grad():
        inputs_embeds = adapter.build_inputs_embeds(model, inputs)
        position_ids = adapter.build_position_ids(model, inputs)
    pruned_embeds, pruned_mask, pruned_pos = gather_sequence_tensors(
        inputs_embeds=inputs_embeds,
        keep_indices=keep_indices,
        attention_mask=inputs.get("attention_mask"),
        position_ids=position_ids,
    )
    pruned_input_ids = inputs["input_ids"].index_select(dim=1, index=keep_indices.to(inputs["input_ids"].device))
    gen_inputs = {
        "input_ids": pruned_input_ids,
        "inputs_embeds": pruned_embeds,
        "attention_mask": pruned_mask,
        "position_ids": pruned_pos,
    }
    gen_inputs = {key: value for key, value in gen_inputs.items() if value is not None}
    return gen_inputs, int(meta.visual_indices.numel()), int(kept_visual.numel())


def _score_attention_proxy(model: Any, pruner: AttentionProxyPruner, inputs: dict[str, Any], meta: Any):
    import torch

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    return pruner.score(attentions=outputs.attentions, meta=meta)


def _freeze_parameters(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _score_gae_oracle(
    *,
    model: Any,
    processor: Any,
    adapter: Any,
    pruner: GAEOraclePruner,
    sample: dict[str, Any],
    answer: str,
    per_token: bool = True,
) -> Any:
    import torch

    _freeze_parameters(model)
    device = next(model.parameters()).device
    teacher_inputs, answer_start = adapter.prepare_teacher_forcing_inputs(processor, sample, answer, device=device)
    teacher_meta = adapter.get_visual_token_meta(model, teacher_inputs)

    def forward_with_retained_attentions():
        with torch.no_grad():
            inputs_embeds = adapter.build_inputs_embeds(model, teacher_inputs)
            position_ids = adapter.build_position_ids(model, teacher_inputs)
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=teacher_inputs.get("attention_mask"),
            position_ids=position_ids,
            output_attentions=True,
            use_cache=False,
        )
        if outputs.attentions is None:
            raise ValueError("Model did not return attentions; use attn_implementation='eager'.")
        for attn in outputs.attentions:
            if attn is not None and attn.requires_grad:
                attn.retain_grad()
        return outputs

    def clear_attention_grads(attentions: Any) -> None:
        for attn in attentions:
            if attn is not None:
                attn.grad = None

    if per_token:
        outputs = forward_with_retained_attentions()
        token_logprobs, query_indices = compute_answer_token_logprobs(
            logits=outputs.logits,
            input_ids=teacher_inputs["input_ids"],
            answer_start=answer_start,
        )
        if token_logprobs.numel() == 0:
            raise ValueError("GAE oracle requires at least one answer token.")
        token_scores = []
        for token_logprob, query_idx in zip(token_logprobs, query_indices):
            model.zero_grad(set_to_none=True)
            clear_attention_grads(outputs.attentions)
            token_logprob.backward(retain_graph=True)
            token_scores.append(
                pruner.score(
                    attentions=outputs.attentions,
                    meta=teacher_meta,
                    query_indices=query_idx.view(1),
                    normalize=False,
                )
            )
        scores = normalize_relevance_scores(torch.stack(token_scores, dim=0).mean(dim=0))
    else:
        outputs = forward_with_retained_attentions()
        target, query_indices = compute_answer_logprob_target(
            logits=outputs.logits,
            input_ids=teacher_inputs["input_ids"],
            answer_start=answer_start,
        )
        target.backward()
        scores = pruner.score(attentions=outputs.attentions, meta=teacher_meta, query_indices=query_indices)
    model.zero_grad(set_to_none=True)
    return scores


def _generate_from_inputs(
    *,
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    max_new_tokens: int,
) -> str:
    if "inputs_embeds" not in inputs and "input_ids" in inputs:
        return _generate_vanilla(model, processor, inputs, max_new_tokens)
    return _generate_from_pruned_inputs(
        model=model,
        processor=processor,
        pruned_inputs=inputs,
        max_new_tokens=max_new_tokens,
    )


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
        attn_implementation=cfg["attn_implementation"],
        processor_use_fast=cfg["processor_use_fast"],
        masquant_root=cfg["masquant_root"],
        masquant_resume=cfg["masquant_resume"],
        masquant_act_scales=cfg["masquant_act_scales"],
        masquant_wbits=cfg["masquant_wbits"],
        masquant_abits=cfg["masquant_abits"],
        masquant_group_size=cfg["masquant_group_size"],
        masquant_inference_mode=cfg["masquant_inference_mode"],
        masquant_symmetric=cfg["masquant_symmetric"],
        masquant_batch_size=cfg["masquant_batch_size"],
    )
    model.eval()
    adapter = _make_adapter(cfg["model_type"])
    pruner = _make_pruner(cfg["pruner"], checkpoint_path=cfg["compressor_checkpoint"])

    output_path = Path(cfg["output_jsonl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for row_idx, sample in enumerate(_read_jsonl(cfg["input_jsonl"])):
            if cfg["limit"] is not None and row_idx >= cfg["limit"]:
                break
            prompt = sample.get("prompt") or sample.get("question") or sample.get("text") or ""
            inputs = adapter.prepare_inputs(processor, sample)
            inputs = _move_inputs_to_model_device(model, inputs)

            pruning_applied = False
            num_visual_tokens_before = None
            num_visual_tokens_after = None
            try:
                meta = adapter.get_visual_token_meta(model, inputs)
                num_visual_tokens_before = int(meta.visual_indices.numel())
                if isinstance(pruner, (AttentionProxyPruner, LearnedCompressorPruner)):
                    scores = _score_attention_proxy(model, pruner, inputs, meta)
                elif isinstance(pruner, GAEOraclePruner):
                    answer = str(sample.get("answer") or "").strip()
                    if cfg["gae_answer_source"] == "generated" or not answer:
                        if cfg["log_oracle_replay"]:
                            LOGGER.info("Generating oracle replay answer for sample %s.", sample.get("id", row_idx))
                        answer = _generate_vanilla(model, processor, inputs, cfg["max_new_tokens"])
                    if not answer:
                        raise ValueError("GAE oracle requires a non-empty sample answer or generated answer.")
                    scores = _score_gae_oracle(
                        model=model,
                        processor=processor,
                        adapter=adapter,
                        pruner=pruner,
                        sample=sample,
                        answer=answer,
                        per_token=bool(cfg["gae_per_token"]),
                    )
                else:
                    raise NotImplementedError(f"Unsupported pruner type: {type(pruner)!r}")
                pruned_inputs, num_visual_tokens_before, num_visual_tokens_after = _build_pruned_generation_inputs(
                    model=model,
                    adapter=adapter,
                    inputs=inputs,
                    scores=scores,
                    retention_ratio=float(cfg["retention_ratio"]),
                    min_keep=int(cfg["min_keep"]),
                )
                prediction = _generate_from_inputs(
                    model=model,
                    processor=processor,
                    inputs=pruned_inputs,
                    max_new_tokens=cfg["max_new_tokens"],
                )
                pruning_applied = True
            except Exception as exc:
                if not cfg["allow_vanilla_fallback"]:
                    raise
                LOGGER.warning("Pruned generation failed; falling back to vanilla generation: %s", exc)
                prediction = _generate_vanilla(model, processor, inputs, cfg["max_new_tokens"])
            record = {
                **sample,
                "id": sample.get("id", str(row_idx)),
                "prompt": prompt,
                "prediction": prediction,
                "retention_ratio": cfg["retention_ratio"],
                "num_visual_tokens_before": num_visual_tokens_before,
                "num_visual_tokens_after": num_visual_tokens_after,
                "pruning_applied": pruning_applied,
                "quant_method": cfg["quant_method"],
                "model_type": cfg["model_type"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
