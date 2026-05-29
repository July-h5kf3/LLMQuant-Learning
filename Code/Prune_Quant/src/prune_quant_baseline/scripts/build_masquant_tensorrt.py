from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from prune_quant_baseline.quant.masquant import (
    _masquant_namespace,
    _prepend_sys_path,
    _set_quant_params,
    patch_lmclass_attention_implementation,
    patch_lmclass_qwen2_vl_support,
    patch_masquant_qwen2_vl_quant_support,
    validate_masquant_root,
)
from prune_quant_baseline.quant.tensorrt import format_tensorrt_builder_command


LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a MASQuant+CMC Qwen2-VL model and build a TensorRT-LLM engine. "
            "The built-in TensorRT-LLM commands follow the upstream Qwen2-VL example; "
            "override them if your TensorRT-LLM checkout uses a different workflow."
        )
    )
    parser.add_argument("--model", required=True, help="HF Qwen2-VL model path.")
    parser.add_argument("--model-type", default="qwen2vl", choices=["qwen2vl", "qwen2_5_vl"])
    parser.add_argument("--masquant-root", required=True, help="Path to EfficientAI/masquant.")
    parser.add_argument("--masquant-resume", required=True, help="Path to MASQuant mas_parameters.pth.")
    parser.add_argument("--act-scales", help="Raw activation scales, recorded for reproducibility.")
    parser.add_argument("--cmc-low-rank", help="Path to CMC low_rank_adapters*.pt.")
    parser.add_argument("--cmc-white-matrix", help="Path to CMC white_matrix*.pt, recorded for reproducibility.")
    parser.add_argument("--output", required=True, help="TensorRT engine root directory.")
    parser.add_argument("--work-dir", help="Intermediate build directory. Defaults to <output>/.build.")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--inference-mode", choices=["split_scales", "merged_scales"], default="split_scales")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--cmc-rank", type=float, default=0.2)
    parser.add_argument("--cmc-quant-cmc", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--vision-max-batch-size", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=3072)
    parser.add_argument("--max-multimodal-len", type=int, default=1296)
    parser.add_argument(
        "--tensorrt-llm-root",
        help="TensorRT-LLM checkout root. Used by the optional stock Qwen2-VL example workflow.",
    )
    parser.add_argument(
        "--allow-stock-trtllm-example",
        action="store_true",
        help=(
            "Allow the stock TensorRT-LLM Qwen2-VL example commands. This is useful for plumbing checks, "
            "but it may not preserve MASQuant custom QuantLinear/CMC semantics without a custom converter."
        ),
    )
    parser.add_argument(
        "--convert-command",
        help=(
            "Optional checkpoint conversion command template. Placeholders include "
            "{hf_export_dir}, {torch_export_dir}, {checkpoint_dir}, {engine_dir}, {dtype}, "
            "{model}, {masquant_resume}, {cmc_low_rank}, {cmc_white_matrix}."
        ),
    )
    parser.add_argument("--llm-build-command", help="Optional TensorRT-LLM decoder build command template.")
    parser.add_argument("--vision-build-command", help="Optional TensorRT vision encoder build command template.")
    parser.add_argument("--skip-materialize", action="store_true", help="Skip MASQuant materialization step.")
    parser.add_argument("--materialize-only", action="store_true", help="Only write the MASQuant export, do not build engines.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _path_or_none(value: str | None) -> str | None:
    if not value:
        return None
    return str(Path(value).expanduser().resolve())


def _require_file(value: str | None, name: str) -> Path:
    if not value:
        raise ValueError(f"{name} is required.")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _language_model_layers(model: Any) -> Any:
    for candidate in (
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "model", None),
    ):
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise AttributeError("Could not locate Qwen2-VL language model layers.")


def _masquant_model_type(model_type: str) -> str:
    return "qwen2_vl" if model_type == "qwen2vl" else "vl"


def materialize_masquant_export(args: argparse.Namespace, export_dir: Path) -> None:
    masquant_root = validate_masquant_root(args.masquant_root)
    patch_lmclass_qwen2_vl_support(masquant_root)
    patch_masquant_qwen2_vl_quant_support(masquant_root)
    if args.attn_implementation != "flash_attention_2":
        patch_lmclass_attention_implementation(masquant_root)

    resume_path = _require_file(args.masquant_resume, "--masquant-resume")
    low_rank_path = _require_file(args.cmc_low_rank, "--cmc-low-rank") if args.cmc_rank and args.cmc_rank > 0 else None
    if args.act_scales:
        _require_file(args.act_scales, "--act-scales")
    if args.cmc_white_matrix:
        _require_file(args.cmc_white_matrix, "--cmc-white-matrix")

    export_dir.mkdir(parents=True, exist_ok=True)
    previous_mode = os.environ.get("inference_mode")
    os.environ["inference_mode"] = args.inference_mode
    try:
        with _prepend_sys_path(masquant_root):
            import torch
            from models.LMClass import LMClass
            from quantize.infer_quant import mas_quantize_model
            from quantize.svd_utils import trans_scales
            from transformers import AutoProcessor

            mas_args = _masquant_namespace(
                model_id_or_path=args.model,
                resume=args.masquant_resume,
                act_scales=args.act_scales,
                wbits=args.wbits,
                abits=args.abits,
                group_size=args.group_size,
                symmetric=True,
                attn_implementation=args.attn_implementation,
                batch_size=args.batch_size,
            )
            mas_args.mode = "infer"
            mas_args.rank = args.cmc_rank
            mas_args.quant_cmc = args.cmc_quant_cmc
            mas_args.scales_path = str(resume_path)
            _set_quant_params(mas_args)

            llm = LMClass(mas_args)
            llm.seqlen = 2048
            llm.model.eval()
            for param in llm.model.parameters():
                param.requires_grad_(False)
            llm.model.to("cuda")

            scales = torch.load(resume_path, weights_only=False)
            layers = _language_model_layers(llm.model)
            down_shape = layers[0].mlp.down_proj.weight.shape[1]
            text_scales, vision_scales, audio_scales = trans_scales(
                scales,
                down_shape,
                _masquant_model_type(args.model_type),
            )
            low_rank_adapters = torch.load(low_rank_path, weights_only=False) if low_rank_path is not None else {}
            model = mas_quantize_model(
                llm.model,
                low_rank_adapters=low_rank_adapters,
                text_scales=text_scales,
                vision_scales=vision_scales,
                audio_scales=audio_scales,
                args=mas_args,
            )
            model.eval()

            hf_dir = export_dir / "hf_model"
            processor_dir = export_dir / "processor"
            state_path = export_dir / "masquant_state.pt"
            model.save_pretrained(hf_dir, safe_serialization=False)
            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
            processor.save_pretrained(processor_dir)
            torch.save(model.state_dict(), state_path)

            metadata = {
                "model": str(Path(args.model).expanduser().resolve()),
                "model_type": args.model_type,
                "masquant_resume": str(resume_path),
                "act_scales": _path_or_none(args.act_scales),
                "cmc_low_rank": str(low_rank_path) if low_rank_path is not None else None,
                "cmc_white_matrix": _path_or_none(args.cmc_white_matrix),
                "wbits": args.wbits,
                "abits": args.abits,
                "group_size": args.group_size,
                "inference_mode": args.inference_mode,
                "hf_export_dir": str(hf_dir),
                "processor_dir": str(processor_dir),
                "state_path": str(state_path),
            }
            (export_dir / "masquant_export.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    finally:
        if previous_mode is None:
            os.environ.pop("inference_mode", None)
        else:
            os.environ["inference_mode"] = previous_mode


def _format_optional_command(template: str | None, values: Mapping[str, Any]) -> list[str] | None:
    if not template:
        return None
    return format_tensorrt_builder_command(template, values)


def build_default_tensorrt_llm_commands(args: argparse.Namespace, values: Mapping[str, Any]) -> list[list[str]]:
    if not args.tensorrt_llm_root:
        return []
    if not args.allow_stock_trtllm_example:
        raise ValueError(
            "Stock TensorRT-LLM Qwen2-VL converters do not consume MASQuant custom QuantLinear/CMC state. "
            "Pass custom --convert-command/--llm-build-command/--vision-build-command for a real MASQuant engine, "
            "or pass --allow-stock-trtllm-example for a plumbing-only stock TensorRT-LLM build."
        )
    if args.model_type != "qwen2vl":
        raise ValueError("The built-in TensorRT-LLM workflow currently targets Qwen2-VL. Use custom commands for Qwen2.5-VL.")
    root = Path(args.tensorrt_llm_root).expanduser().resolve()
    qwen_convert = root / "examples" / "models" / "core" / "qwen" / "convert_checkpoint.py"
    multimodal_build = root / "examples" / "models" / "core" / "multimodal" / "build_multimodal_engine.py"
    if not qwen_convert.exists():
        raise FileNotFoundError(qwen_convert)
    if not multimodal_build.exists():
        raise FileNotFoundError(multimodal_build)
    plugin_dtype = "float16" if args.dtype == "float16" else "auto"
    return [
        [
            sys.executable,
            str(qwen_convert),
            "--model_dir",
            str(values["hf_export_dir"]),
            "--output_dir",
            str(values["checkpoint_dir"]),
            "--dtype",
            args.dtype,
        ],
        [
            "trtllm-build",
            "--checkpoint_dir",
            str(values["checkpoint_dir"]),
            "--output_dir",
            str(values["llm_engine_dir"]),
            f"--gemm_plugin={plugin_dtype}",
            f"--gpt_attention_plugin={plugin_dtype}",
            f"--max_batch_size={args.max_batch_size}",
            f"--max_input_len={args.max_input_len}",
            f"--max_seq_len={args.max_seq_len}",
            f"--max_multimodal_len={args.max_multimodal_len}",
        ],
        [
            sys.executable,
            str(multimodal_build),
            "--model_type",
            "qwen2_vl",
            "--model_path",
            str(values["hf_export_dir"]),
            "--output_dir",
            str(values["vision_engine_dir"]),
        ],
    ]


def build_commands(args: argparse.Namespace, values: Mapping[str, Any]) -> list[list[str]]:
    commands = [
        command
        for command in (
            _format_optional_command(args.convert_command, values),
            _format_optional_command(args.llm_build_command, values),
            _format_optional_command(args.vision_build_command, values),
        )
        if command is not None
    ]
    if commands:
        return commands
    return build_default_tensorrt_llm_commands(args, values)


def _run_command(command: Sequence[str], *, dry_run: bool) -> None:
    LOGGER.info("%s", " ".join(str(part) for part in command))
    if dry_run:
        return
    subprocess.run(list(command), check=True)


def _engine_dir_has_files(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    engine_dir = Path(args.output).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else engine_dir / ".build"
    torch_export_dir = work_dir / "masquant_export"
    hf_export_dir = torch_export_dir / "hf_model"
    checkpoint_dir = work_dir / "trtllm_checkpoint"
    llm_engine_dir = engine_dir / "llm"
    vision_engine_dir = engine_dir / "vision"
    engine_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    values = {
        "model": args.model,
        "masquant_resume": args.masquant_resume,
        "act_scales": args.act_scales or "",
        "cmc_low_rank": args.cmc_low_rank or "",
        "cmc_white_matrix": args.cmc_white_matrix or "",
        "engine_dir": engine_dir,
        "llm_engine_dir": llm_engine_dir,
        "vision_engine_dir": vision_engine_dir,
        "work_dir": work_dir,
        "torch_export_dir": torch_export_dir,
        "hf_export_dir": hf_export_dir,
        "checkpoint_dir": checkpoint_dir,
        "dtype": args.dtype,
        "wbits": args.wbits,
        "abits": args.abits,
        "group_size": args.group_size,
        "inference_mode": args.inference_mode,
        "max_batch_size": args.max_batch_size,
        "max_input_len": args.max_input_len,
        "max_seq_len": args.max_seq_len,
        "max_multimodal_len": args.max_multimodal_len,
    }

    plan_path = work_dir / "build_plan.json"
    commands = build_commands(args, values)
    plan_path.write_text(
        json.dumps(
            {
                "materialize": not args.skip_materialize,
                "materialize_only": args.materialize_only,
                "commands": commands,
                "values": {key: str(value) for key, value in values.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote build plan: %s", plan_path)

    if not args.skip_materialize:
        LOGGER.warning(
            "Materializing MASQuant custom modules for TensorRT conversion. If the TensorRT-LLM stock converter "
            "rejects the exported custom QuantLinear state, provide custom convert/build commands that consume "
            "%s directly.",
            torch_export_dir,
        )
        if not args.dry_run:
            materialize_masquant_export(args, torch_export_dir)

    if args.materialize_only:
        return
    if not commands:
        raise ValueError(
            "No TensorRT build commands configured. Pass --tensorrt-llm-root for the built-in Qwen2-VL workflow, "
            "or pass --convert-command/--llm-build-command/--vision-build-command."
        )
    for command in commands:
        _run_command(command, dry_run=args.dry_run)
    if not args.dry_run and not _engine_dir_has_files(engine_dir):
        raise FileNotFoundError(f"TensorRT engine directory is empty after build: {engine_dir}")


if __name__ == "__main__":
    main()
