from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from prune_quant_baseline.core.logging_utils import configure_logging, get_logger
from prune_quant_baseline.quant.loaders import load_model_and_processor
from prune_quant_baseline.quant.masquant import (
    MASQuantRunConfig,
    build_cmc_command,
    build_train_command,
    format_command,
    masquant_env,
    patch_lmclass_attention_implementation,
    patch_lmclass_qwen2_vl_support,
    patch_qwen25_vl_inputs_embeds_masks,
    run_command,
)
from prune_quant_baseline.quant.tensorrt import (
    DEFAULT_RUNTIME_CLASS,
    format_tensorrt_builder_command,
    run_tensorrt_builder_command,
    write_masquant_tensorrt_artifact,
)
from prune_quant_baseline.scripts.run_infer_pruned import (
    _build_pruned_generation_inputs,
    _generate_vanilla,
    _make_adapter,
    _move_inputs_to_model_device,
    _read_jsonl,
    _resolve_processor_pixels,
    _score_gae_oracle,
)
from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner


LOGGER = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Prune-then-MASQuant baseline with GAE pruning in calibration and inference."
    )
    parser.add_argument(
        "--stage",
        choices=["prepare-cache", "calibrate", "cmc", "infer", "export-tensorrt"],
        default="calibrate",
    )
    parser.add_argument("--model-type", default="qwen2_5_vl", choices=["qwen2vl", "qwen2_5_vl"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--masquant-root", help="Path to alibaba/EfficientAI/masquant checkout.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--calib-jsonl", help="Calibration JSONL used for GAE-pruned MASQuant calibration.")
    parser.add_argument("--eval-jsonl", help="Evaluation JSONL for pruned quantized inference.")
    parser.add_argument("--output-jsonl", help="Output JSONL for pruned quantized inference.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", choices=["true", "false"], default="true")
    parser.add_argument("--trust-remote-code", choices=["true", "false"], default="true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--processor-use-fast", choices=["true", "false"])
    parser.add_argument("--processor-min-pixels", type=int)
    parser.add_argument("--processor-max-pixels", type=int)
    parser.add_argument("--processor-min-visual-tokens", type=int)
    parser.add_argument("--processor-max-visual-tokens", type=int)
    parser.add_argument("--retention-ratio", type=float, default=0.5)
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gae-answer-source", choices=["sample", "generated"], default="sample")
    parser.add_argument("--gae-per-token", choices=["true", "false"], default="true")
    parser.add_argument("--dataset-type", default="text-vision")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--inference-mode", choices=["split_scales", "merged_scales"], default="split_scales")
    parser.add_argument("--masquant-act-scales")
    parser.add_argument("--masquant-resume")
    parser.add_argument("--masquant-output-dir")
    parser.add_argument("--cmc-script-name", default="infer_mas.py")
    parser.add_argument("--cmc-net", default="qwen2.5-vl-7b")
    parser.add_argument("--cmc-output-dir")
    parser.add_argument("--cmc-n-cali-samples", type=int, default=128)
    parser.add_argument(
        "--cmc-cali-data-type",
        choices=["vision-audio-only", "text-audio-vision", "no-white"],
        default="vision-audio-only",
    )
    parser.add_argument("--cmc-rank", type=float, default=0.2)
    parser.add_argument("--cmc-quant-cmc", type=int, default=0)
    parser.add_argument("--cmc-white-matrix-path")
    parser.add_argument("--cmc-low-rank-adapters")
    parser.add_argument("--cmc-no-lr", action="store_true")
    parser.add_argument("--cmc-no-quantize", action="store_true")
    parser.add_argument("--cmc-tasks-multimodal", default="")
    parser.add_argument("--cmc-limit-multimodal", type=float)
    parser.add_argument("--cmc-eval-ppl", action="store_true")
    parser.add_argument("--cmc-eval-sqnr", action="store_true")
    parser.add_argument("--cmc-eval-omni-task", action="store_true")
    parser.add_argument("--cmc-extra-arg", action="append", default=[])
    parser.add_argument("--tensorrt-artifact-dir", help="Directory that stores the reusable MASQuant TensorRT artifact.")
    parser.add_argument("--tensorrt-engine-dir", help="Directory containing the built TensorRT engine files.")
    parser.add_argument(
        "--tensorrt-builder-command",
        help=(
            "Optional TensorRT build command template. Available placeholders: "
            "{model_path}, {masquant_resume}, {masquant_act_scales}, {engine_dir}, "
            "{artifact_dir}, {wbits}, {abits}, {group_size}, {inference_mode}, "
            "{cmc_low_rank_adapters}, {cmc_white_matrix}."
        ),
    )
    parser.add_argument("--tensorrt-builder-cwd", help="Working directory for --tensorrt-builder-command.")
    parser.add_argument("--tensorrt-runtime-class", default=DEFAULT_RUNTIME_CLASS)
    parser.add_argument("--skip-act-scale-collection", action="store_true")
    parser.add_argument(
        "--patch-masquant-inputs-embeds-mask",
        action="store_true",
        help="Patch MASQuant Qwen2.5-VL so pruned inputs_embeds still produce modality masks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _bool_arg(value: str) -> bool:
    return value == "true"


def _detach_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _cache_record(inputs: dict[str, Any]) -> dict[str, Any]:
    keep_keys = {
        "input_ids",
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "rope_deltas",
    }
    return {key: _detach_cpu(value) for key, value in inputs.items() if key in keep_keys}


class ActivationScaleCollector:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.act_scales: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self._current_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._enabled = False

    def install(self) -> None:
        import torch.nn as nn

        filter_modules = ("visual", "vision_tower", "multi_modal_projector", "lm_head", "audio")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not any(part in name for part in filter_modules):
                self._hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _masks_from_input_ids(self, input_ids: torch.Tensor, hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = getattr(self.model, "config", None)
        thinker_config = getattr(config, "thinker_config", None)
        image_token_id = getattr(config, "image_token_id", None)
        video_token_id = getattr(config, "video_token_id", None)
        audio_token_id = getattr(thinker_config, "audio_token_index", None)
        image_mask_2d = torch.zeros_like(input_ids, dtype=torch.bool)
        if image_token_id is not None:
            image_mask_2d |= input_ids == int(image_token_id)
        if video_token_id is not None:
            image_mask_2d |= input_ids == int(video_token_id)
        audio_mask_2d = torch.zeros_like(input_ids, dtype=torch.bool)
        if audio_token_id is not None:
            audio_mask_2d |= input_ids == int(audio_token_id)
        text_mask_2d = ~(image_mask_2d | audio_mask_2d)
        image_mask = image_mask_2d.unsqueeze(-1).expand(*input_ids.shape, hidden_dim)
        audio_mask = audio_mask_2d.unsqueeze(-1).expand(*input_ids.shape, hidden_dim)
        text_mask = text_mask_2d.unsqueeze(-1).expand(*input_ids.shape, hidden_dim)
        return audio_mask, image_mask, text_mask

    def _merge_scale(self, name: str, suffix: str, tensor: torch.Tensor) -> None:
        key = f"{name}.{suffix}"
        tensor = tensor.detach().float().cpu()
        if key in self.act_scales:
            self.act_scales[key] = torch.maximum(self.act_scales[key], tensor)
        else:
            self.act_scales[key] = tensor

    def _masked_abs_max(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden_dim = tensor.shape[-1]
        flat_tensor = tensor.reshape(-1, hidden_dim).abs().detach()
        flat_mask = mask.reshape(-1, hidden_dim)[:, 0].to(flat_tensor.device)
        if not bool(flat_mask.any()):
            return torch.zeros(hidden_dim, dtype=torch.float32, device=flat_tensor.device)
        return torch.max(flat_tensor[flat_mask], dim=0)[0].float()

    def _hook(self, name: str):
        def stat_input_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            del module, output
            if not self._enabled or self._current_masks is None or not inputs:
                return
            tensor = inputs[0]
            if not torch.is_tensor(tensor) or tensor.dim() < 3:
                return
            audio_mask, image_mask, text_mask = self._current_masks
            hidden_dim = tensor.shape[-1]
            if image_mask.shape[-1] != hidden_dim:
                audio_mask, image_mask, text_mask = self._masks_from_input_ids(
                    self._current_input_ids.to(tensor.device), hidden_dim
                )
            flat = tensor.reshape(-1, hidden_dim).abs().detach()
            self._merge_scale(name, "all_in_one_scale", torch.max(flat, dim=0)[0].float())
            self._merge_scale(name, "text_scale", self._masked_abs_max(tensor, text_mask))
            self._merge_scale(name, "vision_scale", self._masked_abs_max(tensor, image_mask))
            self._merge_scale(name, "audio_scale", self._masked_abs_max(tensor, audio_mask))

        return stat_input_hook

    def collect(self, input_ids: torch.Tensor, forward_inputs: dict[str, Any]) -> None:
        self._current_input_ids = input_ids
        hidden_dim = int(forward_inputs["inputs_embeds"].shape[-1]) if "inputs_embeds" in forward_inputs else 1
        self._current_masks = self._masks_from_input_ids(input_ids, hidden_dim)
        self._enabled = True
        try:
            with torch.no_grad():
                self.model(**forward_inputs, use_cache=False)
        finally:
            self._enabled = False
            self._current_masks = None


def _forward_inputs_for_scale_collection(inputs: dict[str, Any]) -> dict[str, Any]:
    if "inputs_embeds" not in inputs:
        return dict(inputs)
    return {
        key: value
        for key, value in inputs.items()
        if key in {"inputs_embeds", "attention_mask", "position_ids", "cache_position"}
    }


def _prepare_masquant_config(args: argparse.Namespace) -> MASQuantRunConfig:
    work_dir = Path(args.work_dir).expanduser().resolve()
    act_scales = args.masquant_act_scales or work_dir / "act_scales" / (
        f"{Path(args.model_path.rstrip('/')).name}-{args.dataset_type}-{args.nsamples}.pt"
    )
    return MASQuantRunConfig(
        masquant_root=args.masquant_root,
        model_path=args.model_path,
        output_dir=args.masquant_output_dir or work_dir / "masquant_outputs",
        cache_dir=work_dir / "cache",
        dataset_type=args.dataset_type,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        wbits=args.wbits,
        abits=args.abits,
        epochs=args.epochs,
        group_size=args.group_size,
        inference_mode=args.inference_mode,
        attn_implementation=args.attn_implementation,
        act_scales_path=act_scales,
        resume=args.masquant_resume,
    )


def prepare_pruned_calibration_artifacts(args: argparse.Namespace, config: MASQuantRunConfig) -> None:
    if not args.calib_jsonl:
        raise ValueError("--calib-jsonl is required for prepare-cache/calibrate.")

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    config.resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    config.resolved_act_scales_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = work_dir / "pruned_calibration_metadata.jsonl"
    processor_min_pixels = _resolve_processor_pixels(
        pixel_value=args.processor_min_pixels,
        token_value=args.processor_min_visual_tokens,
        name="min",
    )
    processor_max_pixels = _resolve_processor_pixels(
        pixel_value=args.processor_max_pixels,
        token_value=args.processor_max_visual_tokens,
        name="max",
    )

    model, processor = load_model_and_processor(
        model_id_or_path=args.model_path,
        model_type=args.model_type,
        quant_method="none",
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=_bool_arg(args.trust_remote_code),
        local_files_only=_bool_arg(args.local_files_only),
        attn_implementation=None if args.attn_implementation == "none" else args.attn_implementation,
        processor_use_fast=None if args.processor_use_fast is None else _bool_arg(args.processor_use_fast),
        processor_min_pixels=processor_min_pixels,
        processor_max_pixels=processor_max_pixels,
    )
    model.eval()
    adapter = _make_adapter(args.model_type)
    pruner = GAEOraclePruner()
    collector = None if args.skip_act_scale_collection else ActivationScaleCollector(model)
    if collector is not None:
        collector.install()

    cache_entries: list[dict[str, Any]] = []
    metadata_records: list[dict[str, Any]] = []
    try:
        for row_idx, sample in enumerate(_read_jsonl(args.calib_jsonl)):
            if row_idx >= args.nsamples:
                break
            inputs = adapter.prepare_inputs(processor, sample)
            inputs = _move_inputs_to_model_device(model, inputs)
            if args.retention_ratio >= 1.0:
                meta = adapter.get_visual_token_meta(model, inputs)
                pruned_inputs = inputs
                before = after = int(meta.visual_indices.numel())
            else:
                answer = str(sample.get("answer") or "").strip()
                if args.gae_answer_source == "generated" or not answer:
                    LOGGER.info("Generating calibration replay answer for sample %s.", sample.get("id", row_idx))
                    answer = _generate_vanilla(model, processor, inputs, args.max_new_tokens)
                if not answer:
                    raise ValueError("GAE calibration requires sample answers or --gae-answer-source generated.")

                with torch.enable_grad():
                    scores = _score_gae_oracle(
                        model=model,
                        processor=processor,
                        adapter=adapter,
                        pruner=pruner,
                        sample=sample,
                        answer=answer,
                        per_token=args.gae_per_token == "true",
                    )
                pruned_inputs, before, after = _build_pruned_generation_inputs(
                    model=model,
                    adapter=adapter,
                    inputs=inputs,
                    scores=scores,
                    retention_ratio=args.retention_ratio,
                    min_keep=args.min_keep,
                )
            if collector is not None:
                forward_inputs = _forward_inputs_for_scale_collection(pruned_inputs)
                collector.collect(pruned_inputs["input_ids"], forward_inputs)

            cache_entries.append(_cache_record(pruned_inputs))
            metadata_records.append(
                {
                    "id": sample.get("id", str(row_idx)),
                    "retention_ratio": args.retention_ratio,
                    "num_visual_tokens_before": before,
                    "num_visual_tokens_after": after,
                    "cache_file": str(config.cache_file),
                    "act_scales": str(config.resolved_act_scales_path),
                    "input_ids": _jsonable(pruned_inputs.get("input_ids")),
                }
            )
            if (row_idx + 1) % 8 == 0:
                LOGGER.info("Prepared %d pruned calibration samples.", row_idx + 1)
    finally:
        if collector is not None:
            collector.remove()

    torch.save(cache_entries, config.cache_file)
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if collector is not None:
        torch.save(collector.act_scales, config.resolved_act_scales_path)
    LOGGER.info("Wrote MASQuant pruned cache: %s", config.cache_file)
    if collector is not None:
        LOGGER.info("Wrote pruned activation scales: %s", config.resolved_act_scales_path)
    LOGGER.info("Wrote calibration metadata: %s", metadata_path)


def run_pruned_masquant_inference(args: argparse.Namespace, config: MASQuantRunConfig) -> None:
    if not args.eval_jsonl or not args.output_jsonl:
        raise ValueError("--eval-jsonl and --output-jsonl are required for --stage infer.")
    from prune_quant_baseline.scripts.run_infer_pruned import main as run_infer_main

    resume = args.masquant_resume
    act_scales = args.masquant_act_scales
    if resume is None and act_scales is None:
        raise ValueError("--stage infer requires --masquant-resume or --masquant-act-scales.")

    argv = [
        "--model-type",
        args.model_type,
        "--model-path",
        args.model_path,
        "--input-jsonl",
        args.eval_jsonl,
        "--output-jsonl",
        args.output_jsonl,
        "--pruner",
        "gae_oracle",
        "--retention-ratio",
        str(args.retention_ratio),
        "--min-keep",
        str(args.min_keep),
        "--quant-method",
        "masquant",
        "--masquant-root",
        str(config.root),
        "--masquant-wbits",
        str(args.wbits),
        "--masquant-abits",
        str(args.abits),
        "--masquant-group-size",
        str(args.group_size),
        "--masquant-inference-mode",
        args.inference_mode,
        "--masquant-batch-size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
        "--attn-implementation",
        args.attn_implementation,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gae-answer-source",
        args.gae_answer_source,
        "--gae-per-token",
        args.gae_per_token,
    ]
    if resume is not None:
        argv.extend(["--masquant-resume", resume])
    if act_scales is not None:
        argv.extend(["--masquant-act-scales", act_scales])
    if args.processor_use_fast is not None:
        argv.extend(["--processor-use-fast", args.processor_use_fast])
    if args.processor_min_pixels is not None:
        argv.extend(["--processor-min-pixels", str(args.processor_min_pixels)])
    if args.processor_max_pixels is not None:
        argv.extend(["--processor-max-pixels", str(args.processor_max_pixels)])
    if args.processor_min_visual_tokens is not None:
        argv.extend(["--processor-min-visual-tokens", str(args.processor_min_visual_tokens)])
    if args.processor_max_visual_tokens is not None:
        argv.extend(["--processor-max-visual-tokens", str(args.processor_max_visual_tokens)])
    LOGGER.info("Running pruned MASQuant inference via run_infer_pruned.")
    if args.dry_run:
        LOGGER.info("python -m prune_quant_baseline.scripts.run_infer_pruned %s", format_command(argv))
        return
    run_infer_main(argv)


def _default_cmc_white_matrix_path(args: argparse.Namespace) -> Path:
    return Path(args.work_dir).expanduser().resolve() / "cmc" / (
        f"white_matrix_{args.cmc_cali_data_type}.pt"
    )


def _default_cmc_low_rank_adapters_path(args: argparse.Namespace) -> Path:
    return Path(args.work_dir).expanduser().resolve() / "cmc" / (
        f"low_rank_adapters_quantcmc{args.cmc_quant_cmc}_rank{args.cmc_rank}_{args.cmc_cali_data_type}.pt"
    )


def _cmc_white_matrix_path(args: argparse.Namespace) -> Path:
    if args.cmc_white_matrix_path:
        return Path(args.cmc_white_matrix_path).expanduser().resolve()
    return _default_cmc_white_matrix_path(args)


def _cmc_low_rank_adapters_path(args: argparse.Namespace) -> Path:
    if args.cmc_low_rank_adapters:
        return Path(args.cmc_low_rank_adapters).expanduser().resolve()
    return _default_cmc_low_rank_adapters_path(args)


def run_masquant_cmc(args: argparse.Namespace, config: MASQuantRunConfig) -> None:
    act_scales = args.masquant_act_scales or str(config.resolved_act_scales_path)
    white_matrix_path = _cmc_white_matrix_path(args)
    low_rank_adapters_path = _cmc_low_rank_adapters_path(args)
    white_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    low_rank_adapters_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = args.cmc_output_dir or Path(args.work_dir).expanduser().resolve() / "cmc_outputs"
    command = build_cmc_command(
        config,
        script_name=args.cmc_script_name,
        net=args.cmc_net,
        scales_path=act_scales,
        output_dir=output_dir,
        n_cali_samples=args.cmc_n_cali_samples,
        cali_data_type=args.cmc_cali_data_type,
        rank=args.cmc_rank,
        quant_cmc=args.cmc_quant_cmc,
        save_white_matrix_path=white_matrix_path,
        save_low_rank_adapters=low_rank_adapters_path,
        quantize=not args.cmc_no_quantize,
        lr=not args.cmc_no_lr,
        tasks_multimodal=args.cmc_tasks_multimodal,
        limit_multimodal=args.cmc_limit_multimodal,
        eval_ppl=args.cmc_eval_ppl,
        eval_sqnr=args.cmc_eval_sqnr,
        eval_omni_task=args.cmc_eval_omni_task,
        extra_args=tuple(args.cmc_extra_arg),
    )
    LOGGER.info("Running MASQuant CMC.")
    run_command(command, cwd=config.root, env=masquant_env(config), dry_run=args.dry_run)
    if args.dry_run:
        LOGGER.info("CMC white matrix path: %s", white_matrix_path)
        LOGGER.info("CMC low-rank adapters path: %s", low_rank_adapters_path)


def export_masquant_tensorrt_artifact(args: argparse.Namespace, config: MASQuantRunConfig) -> None:
    if not args.tensorrt_artifact_dir or not args.tensorrt_engine_dir:
        raise ValueError("--stage export-tensorrt requires --tensorrt-artifact-dir and --tensorrt-engine-dir.")
    if not args.masquant_resume:
        raise ValueError("--stage export-tensorrt requires --masquant-resume.")

    artifact_dir = Path(args.tensorrt_artifact_dir).expanduser().resolve()
    engine_dir = Path(args.tensorrt_engine_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.parent.mkdir(parents=True, exist_ok=True)
    cmc_low_rank_adapters = args.cmc_low_rank_adapters
    if cmc_low_rank_adapters is None and _default_cmc_low_rank_adapters_path(args).exists():
        cmc_low_rank_adapters = str(_default_cmc_low_rank_adapters_path(args))
    cmc_white_matrix = args.cmc_white_matrix_path
    if cmc_white_matrix is None and _default_cmc_white_matrix_path(args).exists():
        cmc_white_matrix = str(_default_cmc_white_matrix_path(args))
    builder_command = None
    if args.tensorrt_builder_command:
        builder_command = format_tensorrt_builder_command(
            args.tensorrt_builder_command,
            {
                "model_path": args.model_path,
                "masquant_resume": args.masquant_resume,
                "masquant_act_scales": args.masquant_act_scales or "",
                "engine_dir": engine_dir,
                "artifact_dir": artifact_dir,
                "wbits": args.wbits,
                "abits": args.abits,
                "group_size": args.group_size,
                "inference_mode": args.inference_mode,
                "cmc_low_rank_adapters": cmc_low_rank_adapters or "",
                "cmc_white_matrix": cmc_white_matrix or "",
            },
        )
        LOGGER.info("TensorRT builder command: %s", format_command(builder_command))
        run_tensorrt_builder_command(
            builder_command,
            cwd=args.tensorrt_builder_cwd,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        LOGGER.info("MASQuant TensorRT artifact dir: %s", artifact_dir)
        LOGGER.info("TensorRT engine dir: %s", engine_dir)
        return

    artifact = write_masquant_tensorrt_artifact(
        artifact_dir=artifact_dir,
        model_path=args.model_path,
        model_type=args.model_type,
        engine_dir=engine_dir,
        masquant_resume=args.masquant_resume,
        masquant_act_scales=args.masquant_act_scales,
        cmc_low_rank_adapters=cmc_low_rank_adapters,
        cmc_white_matrix=cmc_white_matrix,
        wbits=args.wbits,
        abits=args.abits,
        group_size=args.group_size,
        inference_mode=args.inference_mode,
        symmetric=True,
        runtime_class=args.tensorrt_runtime_class,
        builder_command=builder_command,
    )
    LOGGER.info("Wrote MASQuant TensorRT artifact manifest: %s", artifact.root / "manifest.json")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)
    config = _prepare_masquant_config(args)

    if args.stage in {"prepare-cache", "calibrate"}:
        if args.dry_run:
            LOGGER.info("Pruned cache path: %s", config.cache_file)
            LOGGER.info("Pruned activation scales path: %s", config.resolved_act_scales_path)
        else:
            prepare_pruned_calibration_artifacts(args, config)

    if args.stage == "calibrate":
        if args.epochs > 0 and not args.patch_masquant_inputs_embeds_mask:
            LOGGER.warning(
                "MASQuant epochs > 0 with a pruned inputs_embeds cache usually needs "
                "--patch-masquant-inputs-embeds-mask for Qwen2.5-VL modality masks."
            )
        if args.patch_masquant_inputs_embeds_mask and not args.dry_run:
            patched = patch_qwen25_vl_inputs_embeds_masks(config.root)
            LOGGER.info("Patched MASQuant inputs_embeds modality masks at %s", patched)
        if args.model_type == "qwen2vl" and not args.dry_run:
            patched = patch_lmclass_qwen2_vl_support(config.root)
            LOGGER.info("Patched MASQuant Qwen2-VL support at %s", patched)
        if args.attn_implementation != "flash_attention_2" and not args.dry_run:
            patched = patch_lmclass_attention_implementation(config.root)
            LOGGER.info("Patched MASQuant attention implementation at %s", patched)
        command = build_train_command(config)
        run_command(command, cwd=config.root, env=masquant_env(config), dry_run=args.dry_run)

    if args.stage == "cmc":
        run_masquant_cmc(args, config)

    if args.stage == "infer":
        run_pruned_masquant_inference(args, config)

    if args.stage == "export-tensorrt":
        export_masquant_tensorrt_artifact(args, config)


if __name__ == "__main__":
    main()
