#!/usr/bin/env python3
"""Run lmms-eval with the Prune_Quant Qwen2-VL wrapper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = ("ocrbench", "vizwiz_vqa_val", "scienceqa_img", "textvqa_val")
MODEL_PLUGIN_MODULE = "prune_quant_baseline.lmms_eval"
DEFAULT_HF_HOME = Path("/home/aistudio/data/datasets/387822/abcd/hf_home")


def _bool_env(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _append_model_arg(model_args: list[str], key: str, value: str | None) -> None:
    if value is not None and value != "":
        model_args.append(f"{key}={value}")


def _build_default_model_args(args: argparse.Namespace) -> str:
    if args.model_args:
        return args.model_args

    env = os.environ
    model_path = args.model_path or env.get("QWEN2VL_MODEL") or env.get("QWEN25VL_MODEL") or ""
    items: list[str] = []
    _append_model_arg(items, "pretrained", model_path)
    _append_model_arg(items, "model_type", env.get("PQ_MODEL_TYPE", "qwen2vl"))
    _append_model_arg(items, "quant_method", env.get("PQ_QUANT_METHOD", "none"))
    _append_model_arg(items, "dtype", env.get("PQ_DTYPE", "auto"))
    _append_model_arg(items, "device_map", env.get("PQ_DEVICE_MAP", "auto"))
    _append_model_arg(items, "attn_implementation", env.get("PQ_ATTN_IMPLEMENTATION", "eager"))
    _append_model_arg(items, "retention_ratio", env.get("PQ_RETENTION_RATIO", "0.5"))
    _append_model_arg(items, "min_keep", env.get("PQ_MIN_KEEP", "1"))
    _append_model_arg(items, "max_new_tokens", env.get("PQ_MAX_NEW_TOKENS", "16"))
    _append_model_arg(items, "gae_answer_source", env.get("PQ_GAE_ANSWER_SOURCE", "generated"))
    _append_model_arg(items, "gae_per_token", env.get("PQ_GAE_PER_TOKEN", "false"))
    _append_model_arg(items, "pruner", env.get("PQ_PRUNER", "gae_oracle"))
    _append_model_arg(items, "gae_quant_lambda", env.get("PQ_GAE_QUANT_LAMBDA", "0.5"))
    _append_model_arg(items, "gae_quant_method", env.get("PQ_GAE_QUANT_METHOD", "rtn"))
    _append_model_arg(items, "rtn_bits", env.get("PQ_RTN_BITS", "4"))
    _append_model_arg(items, "rtn_group_size", env.get("PQ_RTN_GROUP_SIZE", "0"))
    _append_model_arg(items, "allow_vanilla_fallback", env.get("PQ_ALLOW_VANILLA_FALLBACK", "false"))
    _append_model_arg(items, "gae_score_disable_masquant_fake_quant", env.get("PQ_GAE_DISABLE_MASQUANT_FAKE_QUANT", "false"))
    _append_model_arg(items, "min_pixels", env.get("PQ_MIN_PIXELS"))
    _append_model_arg(items, "max_pixels", env.get("PQ_MAX_PIXELS"))
    _append_model_arg(items, "min_visual_tokens", env.get("PQ_MIN_VISUAL_TOKENS"))
    _append_model_arg(items, "max_visual_tokens", env.get("PQ_MAX_VISUAL_TOKENS"))
    _append_model_arg(items, "masquant_root", env.get("MASQUANT_ROOT"))
    _append_model_arg(items, "masquant_resume", env.get("MASQUANT_RESUME"))
    _append_model_arg(items, "masquant_act_scales", env.get("MASQUANT_ACT_SCALES"))
    _append_model_arg(items, "masquant_wbits", env.get("PQ_MASQUANT_WBITS"))
    _append_model_arg(items, "masquant_abits", env.get("PQ_MASQUANT_ABITS"))
    _append_model_arg(items, "masquant_group_size", env.get("PQ_MASQUANT_GROUP_SIZE"))
    _append_model_arg(items, "masquant_inference_mode", env.get("PQ_MASQUANT_INFERENCE_MODE"))
    _append_model_arg(items, "masquant_symmetric", env.get("PQ_MASQUANT_SYMMETRIC"))
    _append_model_arg(items, "masquant_batch_size", env.get("PQ_MASQUANT_BATCH_SIZE"))
    _append_model_arg(items, "masquant_cmc_low_rank_adapters", env.get("CMC_LOW_RANK"))
    _append_model_arg(items, "masquant_cmc_white_matrix", env.get("CMC_WHITE"))
    _append_model_arg(items, "masquant_cmc_rank", env.get("PQ_CMC_RANK"))
    _append_model_arg(items, "masquant_cmc_quant_cmc", env.get("PQ_CMC_QUANT_CMC"))
    return ",".join(items)


def _build_subprocess_env(project_root: str | Path, lmms_eval_root: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(Path(project_root) / "src"), str(lmms_eval_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    plugins = [
        item
        for item in env.get("LMMS_EVAL_PLUGINS", "").split(",")
        if item
    ]
    if MODEL_PLUGIN_MODULE not in plugins:
        plugins.append(MODEL_PLUGIN_MODULE)
    env["LMMS_EVAL_PLUGINS"] = ",".join(plugins)

    env.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    env.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")

    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if _bool_env("LMMS_EVAL_DISABLE_OPENAI", "1"):
        for name in (
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            "OPENAI_API_MODEL",
            "OPENAI_API_TYPE",
            "OPENAI_API_VERSION",
            "AZURE_OPENAI_API_KEY",
            "LOCAL_LLM",
        ):
            env.pop(name, None)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmms-eval-root", default="")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--model", default="prune_quant_qwen2vl")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-args", default="")
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--cache", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--limit", default="")
    parser.add_argument("--verbosity", default="INFO")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    lmms_eval_root = Path(args.lmms_eval_root or project_root / "third_party" / "lmms-eval").resolve()
    if not lmms_eval_root.exists():
        raise FileNotFoundError(f"lmms-eval root not found: {lmms_eval_root}")

    model_args = _build_default_model_args(args)
    cmd = [
        args.python,
        "-m",
        "lmms_eval",
        "--model",
        args.model,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(args.tasks),
        "--batch_size",
        args.batch_size,
        "--output_path",
        str(args.output_path),
        "--verbosity",
        args.verbosity,
    ]
    if args.limit:
        cmd += ["--limit", args.limit]
    if args.cache:
        cmd += ["--use_cache", args.cache]
    if args.log_samples:
        cmd.append("--log_samples")

    env = _build_subprocess_env(project_root, lmms_eval_root)

    print("[lmms-eval-smart] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=lmms_eval_root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
