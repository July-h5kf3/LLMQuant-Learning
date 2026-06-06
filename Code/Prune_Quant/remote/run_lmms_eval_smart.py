#!/usr/bin/env python3
"""Run lmms-eval with the Prune_Quant Qwen2-VL wrapper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = ("mmmu_val", "ocrbench", "vizwiz_vqa_val", "scienceqa_img", "textvqa_val")
MODEL_PLUGIN_MODULE = "prune_quant_baseline.lmms_eval"
DEFAULT_HF_HOME = Path("/home/aistudio/data/datasets/387822/abcd/hf_home")
LOCAL_DATASET_TASKS = {
    "mmmu_val": {
        "repo_id": "lmms-lab/MMMU",
        "source_yaml": ("mmmu", "mmmu_val.yaml"),
        "data_files": {"validation": ("data", "validation-*")},
    },
    "ocrbench": {
        "repo_id": "echo840/OCRBench",
        "source_yaml": ("ocrbench", "ocrbench.yaml"),
        "data_files": {"test": ("data", "test-*")},
    },
    "vizwiz_vqa_val": {
        "repo_id": "lmms-lab/VizWiz-VQA",
        "source_yaml": ("vizwiz_vqa", "vizwiz_vqa_val.yaml"),
        "data_files": {"val": ("data", "val-*")},
    },
    "scienceqa_img": {
        "repo_id": "lmms-lab/ScienceQA",
        "source_yaml": ("scienceqa", "scienceqa_img.yaml"),
        "data_files": {"test": ("ScienceQA-IMG", "test-*")},
    },
    "textvqa_val": {
        "repo_id": "lmms-lab/textvqa",
        "source_yaml": ("textvqa", "textvqa_val.yaml"),
        "data_files": {"validation": ("data", "validation-*")},
    },
}


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

    lmms_eval_hf_home = env.get("LMMS_EVAL_HF_HOME", "").strip()
    if lmms_eval_hf_home:
        hf_home = Path(lmms_eval_hf_home)
        env["HF_HOME"] = str(hf_home)
        env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
        env["HF_HUB_CACHE"] = str(hf_home / "hub")
        env["HF_MODULES_CACHE"] = str(hf_home / "modules")
        env["LMMS_EVAL_DATASETS_CACHE"] = str(hf_home / "datasets")
    else:
        env.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
        env.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
        env.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
        env.setdefault("HF_MODULES_CACHE", str(DEFAULT_HF_HOME / "modules"))
        env.setdefault("LMMS_EVAL_DATASETS_CACHE", env["HF_DATASETS_CACHE"])
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


def _hf_dataset_cache_name(repo_id: str) -> str:
    return f"datasets--{repo_id.replace('/', '--')}"


def _resolve_local_hf_dataset_snapshot(hf_home: Path, repo_id: str) -> Path | None:
    repo_cache = hf_home / "hub" / _hf_dataset_cache_name(repo_id)
    snapshots = repo_cache / "snapshots"
    ref_main = repo_cache / "refs" / "main"
    if ref_main.exists():
        revision = ref_main.read_text(encoding="utf-8").strip()
        snapshot = snapshots / revision
        if snapshot.exists():
            return snapshot
    if snapshots.exists():
        snapshot_dirs = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if snapshot_dirs:
            return snapshot_dirs[-1]
    return None


def _prepare_local_task_overlays(
    tasks: list[str],
    lmms_eval_root: Path,
    hf_home: Path,
    output_path: Path,
) -> tuple[list[str], Path | None, list[tuple[str, str]]]:
    overlay_root = output_path / "_local_lmms_tasks"
    rewritten_tasks: list[str] = []
    missing_snapshots: list[tuple[str, str]] = []
    wrote_overlay = False
    task_root = lmms_eval_root / "lmms_eval" / "tasks"

    for task in tasks:
        task_config = LOCAL_DATASET_TASKS.get(task)
        if task_config is None:
            rewritten_tasks.append(task)
            continue

        snapshot = _resolve_local_hf_dataset_snapshot(hf_home, task_config["repo_id"])
        if snapshot is None:
            missing_snapshots.append((task, task_config["repo_id"]))
            rewritten_tasks.append(task)
            continue

        overlay_root.mkdir(parents=True, exist_ok=True)
        local_task = f"pq_local_{task}"
        source_yaml = task_root.joinpath(*task_config["source_yaml"])
        overlay_path = overlay_root / f"{task}.yaml"
        data_files = task_config.get("data_files", {})
        dataset_kwargs = ["dataset_kwargs:"]
        if data_files:
            dataset_kwargs.append("  data_files:")
        for split, parts in data_files.items():
            dataset_kwargs.append(f"    {split}: {snapshot.joinpath(*parts)}")
        dataset_kwargs.append("  local_files_only: true")
        overlay_path.write_text(
            "\n".join(
                [
                    f"include: {source_yaml}",
                    f"task: {local_task}",
                    f"dataset_path: {snapshot}",
                    *dataset_kwargs,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        rewritten_tasks.append(local_task)
        wrote_overlay = True

    return rewritten_tasks, overlay_root if wrote_overlay else None, missing_snapshots


def _require_local_snapshots(env: dict[str, str]) -> bool:
    allow_hub_fallback = os.environ.get("LMMS_EVAL_ALLOW_HUB_FALLBACK", "").strip().lower()
    if allow_hub_fallback in {"1", "true", "yes", "y", "on"}:
        return False
    return env.get("HF_HUB_OFFLINE", "1").strip().lower() in {"1", "true", "yes", "y", "on"}


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

    env = _build_subprocess_env(project_root, lmms_eval_root)

    hf_home = Path(env.get("HF_HOME", str(DEFAULT_HF_HOME))).expanduser()
    tasks, include_path, missing_snapshots = _prepare_local_task_overlays(
        tasks=list(args.tasks),
        lmms_eval_root=lmms_eval_root,
        hf_home=hf_home,
        output_path=args.output_path,
    )
    if missing_snapshots and _require_local_snapshots(env):
        missing = ", ".join(f"{task} -> {repo_id}" for task, repo_id in missing_snapshots)
        raise FileNotFoundError(
            "Missing local Hugging Face dataset snapshots for offline lmms-eval: "
            f"{missing}. Expected each repo under {hf_home / 'hub'} as datasets--ORG--NAME/snapshots/<revision>. "
            "Upload the missing repo snapshots or set LMMS_EVAL_ALLOW_HUB_FALLBACK=1 with network access."
        )

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
        ",".join(tasks),
        "--batch_size",
        args.batch_size,
        "--output_path",
        str(args.output_path),
        "--verbosity",
        args.verbosity,
    ]
    if include_path is not None:
        cmd += ["--include_path", str(include_path)]
    if args.limit:
        cmd += ["--limit", args.limit]
    if args.cache:
        cmd += ["--use_cache", args.cache]
    if args.log_samples:
        cmd.append("--log_samples")

    print("[lmms-eval-smart] " + " ".join(cmd), flush=True)
    print(
        "[lmms-eval-smart] "
        f"HF_HOME={env.get('HF_HOME', '')} "
        f"HF_DATASETS_CACHE={env.get('HF_DATASETS_CACHE', '')} "
        f"HF_HUB_CACHE={env.get('HF_HUB_CACHE', '')} "
        f"HF_MODULES_CACHE={env.get('HF_MODULES_CACHE', '')} "
        f"LMMS_EVAL_DATASETS_CACHE={env.get('LMMS_EVAL_DATASETS_CACHE', '')} "
        f"HF_DATASETS_OFFLINE={env.get('HF_DATASETS_OFFLINE', '')} "
        f"HF_HUB_OFFLINE={env.get('HF_HUB_OFFLINE', '')}",
        flush=True,
    )
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=lmms_eval_root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
