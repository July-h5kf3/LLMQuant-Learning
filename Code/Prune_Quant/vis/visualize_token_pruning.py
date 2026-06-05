#!/usr/bin/env python3
"""Visualize token ranges removed by GAE and quantization-aware pruning.

The script expects one sample per run. A sample contains token embeddings
(`inputs_embeds`) and two visual-token score vectors:

* GAE scores are keep scores, so low-scoring visual tokens are removed.
* Quant-joint scores are drop scores, so high-scoring visual tokens are removed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import math
import random
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class GreenHighlightSpec(NamedTuple):
    mode: str
    label: str
    values: np.ndarray | None
    tokens: np.ndarray


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        del exc
        return None, None
    return plt, Line2D


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required field: {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if getattr(value, "dtype", None) is not None and str(value.dtype) in {
            "torch.bfloat16",
            "torch.float16",
        }:
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _load_sample(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("torch is required to load .pt/.pth samples.") from exc
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported sample file suffix {suffix!r}; use .pt, .pth, or .npz.")


def _read_jsonl_file(path: str | Path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return
    if text.startswith("["):
        for item in __import__("json").loads(text):
            yield item
        return
    for line in text.splitlines():
        if line.strip():
            yield __import__("json").loads(line)


def _resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _make_demo_sample(num_visual: int, num_text: int, dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vision = rng.normal(loc=0.0, scale=0.8, size=(num_visual, dim))
    text = rng.normal(loc=0.2, scale=0.45, size=(num_text, dim))
    token_axis = np.linspace(0.0, 1.0, num_visual)
    vision += 0.8 * np.sin(2.0 * np.pi * token_axis)[:, None]
    inputs_embeds = np.concatenate([vision, text], axis=0).astype(np.float32)

    gae_scores = (
        0.7 * np.exp(-((token_axis - 0.35) ** 2) / 0.03)
        + 0.45 * np.exp(-((token_axis - 0.78) ** 2) / 0.012)
        + 0.05 * rng.random(num_visual)
    )
    c_quant = (
        0.55 * np.exp(-((token_axis - 0.62) ** 2) / 0.02)
        + 0.25 * np.exp(-((token_axis - 0.18) ** 2) / 0.018)
        + 0.03 * rng.random(num_visual)
    )
    c_drop = gae_scores.copy()
    quant_joint_scores = c_quant - c_drop
    return {
        "id": np.asarray("demo_sample"),
        "inputs_embeds": inputs_embeds,
        "visual_indices": np.arange(num_visual, dtype=np.int64),
        "text_indices": np.arange(num_visual, num_visual + num_text, dtype=np.int64),
        "gae_scores": gae_scores.astype(np.float32),
        "c_quant": c_quant.astype(np.float32),
        "c_drop": c_drop.astype(np.float32),
        "quant_joint_scores": quant_joint_scores.astype(np.float32),
    }


def _load_visualization_config(path: Path) -> dict[str, Any]:
    from prune_quant_baseline.core.config import _deep_merge, _expand_env
    import yaml

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = {
        "model": {
            "dtype": "bfloat16",
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True,
            "attn_implementation": "eager",
        },
        "calibration": {},
        "questions": {},
        "quant_joint": {
            "quant_lambda": 0.5,
            "quant_method": "rtn",
            "rtn_bits": 4,
            "rtn_group_size": 0,
        },
        "pruning": {
            "retention_ratio": 0.5,
            "min_keep": 1,
        },
        "scoring": {
            "answer_source": "sample",
            "per_token": True,
            "gae_normalizer": "none",
            "max_new_tokens": 16,
        },
        "visualization": {
            "limit": 1,
            "random_sample": True,
            "seed": None,
            "image_overlay": True,
            "green_highlight": "proxy",
            "show_predictions": True,
            "output_dir": str(Path(__file__).resolve().parent / "outputs"),
            "save_sample_artifacts": False,
            "sample_artifact_dir": str(Path(__file__).resolve().parent / "samples"),
        },
    }
    return _deep_merge(defaults, _expand_env(raw, strict=False))


def _required(config: dict[str, Any], path: str) -> Any:
    cursor: Any = config
    for key in path.split("."):
        if not isinstance(cursor, dict) or key not in cursor or cursor[key] in (None, ""):
            raise ValueError(f"Missing required config field: {path}")
        cursor = cursor[key]
    return cursor


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except (ImportError, TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def _first_nonempty(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        if key in row and not _is_missing(row[key]):
            return str(row[key])
    return default


def _decode_base64_image(value: Any, cache_key: str, image_cache: dict[str, Any]) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        image = value.convert("RGB")
        image_cache[cache_key] = image
        return image.copy()
    if _is_missing(value):
        if cache_key in image_cache:
            return image_cache[cache_key].copy()
        raise ValueError(f"Missing image data for {cache_key!r}; this question source requires an image per sample.")
    raw = str(value).strip()
    if len(raw) > 16:
        try:
            image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to decode image data for {cache_key!r}.") from exc
        image_cache[cache_key] = image
        return image.copy()
    if cache_key in image_cache:
        return image_cache[cache_key].copy()
    raise ValueError(f"Image data for {cache_key!r} is too short to be a valid base64 image.")


def _format_question(dataset: str, row: dict[str, Any], *, mme_prompt_style: str = "default") -> str:
    question = str(row["question"])
    if dataset == "MMStar":
        option_values = []
        for opt in ["A", "B", "C", "D"]:
            if opt not in row or _is_missing(row[opt]):
                option_values = []
                break
            option_values.append(f"{opt}. {row[opt]}")
        options = "\n".join(option_values)
        if not options:
            return f"{question}\nAnswer with the option letter only, one of A, B, C, or D."
        return f"{question}\n{options}\nAnswer with the option letter only, one of A, B, C, or D."
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


def _sample_from_question_row(
    row: dict[str, Any],
    *,
    row_idx: int,
    dataset: str,
    image_cache: dict[str, Any],
    mme_prompt_style: str,
    image_root: Path | None,
) -> dict[str, Any]:
    question_id = _first_nonempty(row, "question_id")
    index = _first_nonempty(row, "index", "id", default=f"{question_id or 'row'}::{row_idx}")
    image_path = _first_nonempty(row, "image_path", "image_file", "path")
    image_key = _first_nonempty(row, "question_id", "image_path", "image_file", "path", default=index)
    image_value = row.get("image")

    sample: dict[str, Any] = {
        "id": index,
        "prompt": _format_question(dataset, row, mme_prompt_style=mme_prompt_style),
        "answer": _first_nonempty(row, "answer"),
        "question": _first_nonempty(row, "question"),
        "category": _first_nonempty(row, "category"),
        "question_id": question_id,
    }
    if not _is_missing(image_value):
        sample["image"] = _decode_base64_image(image_value, image_key, image_cache)
        if image_path:
            sample["image_path"] = str((image_root / image_path).resolve()) if image_root and not Path(image_path).is_absolute() else image_path
    elif image_path:
        path = Path(image_path).expanduser()
        if image_root is not None and not path.is_absolute():
            path = image_root / path
        sample["image"] = str(path)
        sample["image_path"] = str(path)
    else:
        raise ValueError(f"Question row {index!r} does not contain image or image_path.")
    return sample


def _load_question_samples(
    *,
    source_cfg: dict[str, Any],
    base_dir: Path,
    fallback_image_root: Path | None,
) -> list[dict[str, Any]]:
    source = str(source_cfg.get("source", "jsonl"))
    dataset = str(source_cfg.get("dataset", "MME"))
    image_root = _resolve_path(source_cfg.get("image_root"), base_dir) or fallback_image_root
    image_cache: dict[str, Any] = {}
    if source == "jsonl":
        path = _resolve_path(source_cfg.get("path") or source_cfg.get("input_jsonl"), base_dir)
        if path is None:
            raise ValueError("questions.path is required when questions.source=jsonl.")
        return [_prepare_sample_paths(dict(item), image_root) for item in _read_jsonl_file(path)]
    if source == "tsv":
        try:
            import pandas as pd
        except ImportError:
            pd = None
        path = _resolve_path(source_cfg.get("path") or source_cfg.get("tsv"), base_dir)
        if path is None:
            raise ValueError("questions.path or questions.tsv is required when questions.source=tsv.")
        mme_prompt_style = str(source_cfg.get("mme_prompt_style", "default"))
        if pd is None:
            with path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            return [
                _sample_from_question_row(
                    row,
                    row_idx=row_idx,
                    dataset=dataset,
                    image_cache=image_cache,
                    mme_prompt_style=mme_prompt_style,
                    image_root=image_root,
                )
                for row_idx, row in enumerate(rows)
            ]
        df = pd.read_csv(path, sep="\t")
        return [
            _sample_from_question_row(
                row.to_dict(),
                row_idx=int(row_idx),
                dataset=dataset,
                image_cache=image_cache,
                mme_prompt_style=mme_prompt_style,
                image_root=image_root,
            )
            for row_idx, row in df.iterrows()
        ]
    if source == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("datasets is required to read questions.source=hf.") from exc
        hf_dataset = str(source_cfg.get("hf_dataset", "lmms-lab/MME"))
        hf_split = str(source_cfg.get("hf_split", "test"))
        hf_cache_dir = source_cfg.get("hf_cache_dir")
        ds = load_dataset(hf_dataset, split=hf_split, cache_dir=hf_cache_dir)
        mme_prompt_style = str(source_cfg.get("mme_prompt_style", "default"))
        return [
            _sample_from_question_row(
                dict(row),
                row_idx=row_idx,
                dataset=dataset,
                image_cache=image_cache,
                mme_prompt_style=mme_prompt_style,
                image_root=image_root,
            )
            for row_idx, row in enumerate(ds)
        ]
    raise ValueError("questions.source must be one of: jsonl, tsv, hf.")


def _resolve_processor_pixels(config: dict[str, Any], name: str) -> int | None:
    model_cfg = config.get("model", {})
    pixel_value = model_cfg.get(f"processor_{name}_pixels")
    token_value = model_cfg.get(f"processor_{name}_visual_tokens")
    if pixel_value is not None and token_value is not None:
        raise ValueError(f"Use either model.processor_{name}_pixels or model.processor_{name}_visual_tokens, not both.")
    if pixel_value is not None:
        return int(pixel_value)
    if token_value is not None:
        return int(token_value) * 28 * 28
    return None


def _prepare_sample_paths(sample: dict[str, Any], image_root: Path | None) -> dict[str, Any]:
    if image_root is None:
        return sample
    sample = dict(sample)
    for key in ("image", "image_path"):
        value = sample.get(key)
        if isinstance(value, str) and not Path(value).expanduser().is_absolute():
            sample[key] = str(image_root / value)
    if isinstance(sample.get("images"), list):
        images = []
        for value in sample["images"]:
            if isinstance(value, str) and not Path(value).expanduser().is_absolute():
                images.append(str(image_root / value))
            else:
                images.append(value)
        sample["images"] = images
    return sample


def _sample_image_path(sample: dict[str, Any]) -> str | None:
    for key in ("image", "image_path"):
        value = sample.get(key)
        if isinstance(value, str):
            return value
    images = sample.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    return None


def _spatial_merge_size(model: Any) -> int:
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    value = getattr(vision_config, "spatial_merge_size", None)
    if value is None:
        value = getattr(getattr(model, "config", None), "spatial_merge_size", None)
    return int(value or 2)


def _build_sample_artifact(
    *,
    model: Any,
    processor: Any,
    adapter: Any,
    sample: dict[str, Any],
    inputs: dict[str, Any],
    meta: Any,
    gae_scores: Any,
    quant_joint_scores: Any,
    quant_components: dict[str, Any] | None = None,
    answer: str | None = None,
    predictions: dict[str, str] | None = None,
) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        inputs_embeds = adapter.build_inputs_embeds(model, inputs)
    boundary = int(meta.visual_indices.max().item()) + 1

    def to_cpu_tensor(value: Any, name: str) -> Any:
        if hasattr(value, "detach"):
            return value.detach().float().cpu()
        return torch.as_tensor(_as_numpy(value, name=name), dtype=torch.float32).cpu()

    artifact = {
        "id": sample.get("id", "sample"),
        "inputs_embeds": inputs_embeds.detach().float().cpu(),
        "visual_indices": meta.visual_indices.detach().cpu(),
        "text_indices": None if meta.text_indices is None else meta.text_indices.detach().cpu(),
        "vision_text_boundary": boundary,
        "image_grid_thw": None if meta.image_grid_thw is None else meta.image_grid_thw.detach().cpu(),
        "image_path": _sample_image_path(sample),
        "image": sample.get("image"),
        "prompt": sample.get("prompt"),
        "question": sample.get("question") or sample.get("prompt") or sample.get("text"),
        "answer": answer if answer is not None else sample.get("answer"),
        "reference_answer": sample.get("answer"),
        "predictions": dict(predictions or sample.get("predictions") or {}),
        "category": sample.get("category"),
        "question_id": sample.get("question_id"),
        "spatial_merge_size": _spatial_merge_size(model),
        "num_visual_tokens": int(meta.visual_indices.numel()),
        "seq_len": int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else int(inputs_embeds.shape[1]),
        "gae_scores": to_cpu_tensor(gae_scores, "gae_scores"),
        "quant_joint_scores": to_cpu_tensor(quant_joint_scores, "quant_joint_scores"),
    }
    if quant_components:
        for src_key, dst_key in (("c_quant", "c_quant"), ("c_drop", "c_drop"), ("joint", "quant_joint_scores")):
            value = quant_components.get(src_key)
            if value is not None:
                artifact[dst_key] = value.detach().float().cpu()
    return artifact


def _sample_answer(
    *,
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    sample: dict[str, Any],
    answer_source: str,
    max_new_tokens: int,
) -> str:
    from prune_quant_baseline.scripts.run_infer_pruned import _generate_vanilla

    answer = str(sample.get("answer") or "").strip()
    if answer_source == "generated" or not answer:
        answer = _generate_vanilla(model, processor, inputs, max_new_tokens)
    if not answer:
        raise ValueError("GAE scoring requires a non-empty sample answer or scoring.answer_source: generated.")
    return answer


def _prediction_entry(sample: dict[str, Any], key: str) -> str:
    predictions = sample.get("predictions")
    if not isinstance(predictions, dict):
        return ""
    return _metadata_string(predictions.get(key))


def _build_prediction_variants(
    *,
    model: Any,
    processor: Any,
    adapter: Any,
    inputs: dict[str, Any],
    gae_scores: Any,
    quant_scores: Any,
    retention_ratio: float,
    min_keep: int,
    max_new_tokens: int,
) -> dict[str, str]:
    import torch
    from prune_quant_baseline.scripts.run_infer_pruned import (
        _build_pruned_generation_inputs,
        _generate_from_pruned_inputs,
        _generate_vanilla,
    )

    device = inputs["input_ids"].device if "input_ids" in inputs else next(model.parameters()).device
    gae_scores = torch.as_tensor(_as_numpy(gae_scores, name="gae_scores"), dtype=torch.float32, device=device)
    quant_scores = torch.as_tensor(_as_numpy(quant_scores, name="quant_joint_scores"), dtype=torch.float32, device=device)
    original = _generate_vanilla(model, processor, inputs, max_new_tokens)
    gae_inputs, _, _ = _build_pruned_generation_inputs(
        model=model,
        adapter=adapter,
        inputs=inputs,
        scores=gae_scores,
        retention_ratio=retention_ratio,
        min_keep=min_keep,
        score_mode="keep",
    )
    quant_inputs, _, _ = _build_pruned_generation_inputs(
        model=model,
        adapter=adapter,
        inputs=inputs,
        scores=quant_scores,
        retention_ratio=retention_ratio,
        min_keep=min_keep,
        score_mode="drop",
    )
    return {
        "original": original,
        "gae_pruned": _generate_from_pruned_inputs(
            model=model,
            processor=processor,
            pruned_inputs=gae_inputs,
            max_new_tokens=max_new_tokens,
        ),
        "quant_joint_pruned": _generate_from_pruned_inputs(
            model=model,
            processor=processor,
            pruned_inputs=quant_inputs,
            max_new_tokens=max_new_tokens,
        ),
    }


def _iter_config_artifacts(config_path: Path, cli_limit: int | None = None):
    from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner, QuantJointGAEPruner
    from prune_quant_baseline.quant.loaders import load_model_and_processor
    from prune_quant_baseline.scripts.run_infer_pruned import (
        _make_adapter,
        _move_inputs_to_model_device,
        _read_jsonl,
        _score_gae_oracle,
        _score_gae_quant_joint,
    )

    config = _load_visualization_config(config_path)
    base_dir = config_path.parent
    model_cfg = config["model"]
    calibration_cfg = config["calibration"]
    questions_cfg = config.get("questions") or {}
    data_cfg = config.get("data", {})
    scoring_cfg = config["scoring"]
    quant_cfg = config["quant_joint"]
    pruning_cfg = config["pruning"]
    vis_cfg = config["visualization"]
    retention_ratio = float(pruning_cfg.get("retention_ratio", 0.5))
    min_keep = int(pruning_cfg.get("min_keep", 1))
    max_new_tokens = int(scoring_cfg.get("max_new_tokens", 16))
    show_predictions = _bool_value(vis_cfg.get("show_predictions", True))

    image_root = _resolve_path(calibration_cfg.get("image_root") or data_cfg.get("image_root"), base_dir)
    limit = cli_limit if cli_limit is not None else int(vis_cfg.get("limit", 1))
    sample_offset = int(vis_cfg.get("sample_offset", 0))
    if questions_cfg:
        raw_samples = _load_question_samples(
            source_cfg=questions_cfg,
            base_dir=base_dir,
            fallback_image_root=image_root,
        )
        sample_source_name = f"questions.{questions_cfg.get('source', 'jsonl')}"
    else:
        calib_path = _resolve_path(
            calibration_cfg.get("path")
            or calibration_cfg.get("calib_jsonl")
            or calibration_cfg.get("input_jsonl")
            or data_cfg.get("calib_jsonl")
            or data_cfg.get("input_jsonl"),
            base_dir,
        )
        if calib_path is None:
            raise ValueError("Missing required config field: questions.path or calibration.path")
        raw_samples = [_prepare_sample_paths(dict(item), image_root) for item in _read_jsonl(calib_path)]
        sample_source_name = "calibration"
    candidate_indices = list(range(sample_offset, len(raw_samples)))
    if not candidate_indices:
        raise ValueError(f"No calibration samples available after sample_offset={sample_offset}.")
    if _bool_value(vis_cfg.get("random_sample", True)):
        rng = random.Random(vis_cfg.get("seed"))
        selected_indices = rng.sample(candidate_indices, k=min(limit, len(candidate_indices)))
    else:
        selected_indices = candidate_indices[:limit]

    model, processor = load_model_and_processor(
        model_id_or_path=str(_required(config, "model.model_path")),
        model_type=str(_required(config, "model.model_type")),
        quant_method="none",
        dtype=str(model_cfg.get("dtype", "bfloat16")),
        device_map=str(model_cfg.get("device_map", "auto")),
        trust_remote_code=_bool_value(model_cfg.get("trust_remote_code", True)),
        local_files_only=_bool_value(model_cfg.get("local_files_only", True)),
        attn_implementation=None
        if model_cfg.get("attn_implementation") in (None, "none")
        else str(model_cfg.get("attn_implementation", "eager")),
        processor_use_fast=None
        if model_cfg.get("processor_use_fast") is None
        else _bool_value(model_cfg.get("processor_use_fast")),
        processor_min_pixels=_resolve_processor_pixels(config, "min"),
        processor_max_pixels=_resolve_processor_pixels(config, "max"),
    )
    model.eval()
    adapter = _make_adapter(str(model_cfg["model_type"]))
    gae_pruner = GAEOraclePruner()
    quant_lambda = float(quant_cfg.get("quant_lambda", pruning_cfg.get("quant_lambda", 0.5)))
    quant_pruner = QuantJointGAEPruner(quant_lambda=quant_lambda)

    for row_idx in selected_indices:
        raw_sample = raw_samples[row_idx]
        sample = _prepare_sample_paths(raw_sample, image_root)
        inputs = adapter.prepare_inputs(processor, sample)
        inputs = _move_inputs_to_model_device(model, inputs)
        meta = adapter.get_visual_token_meta(model, inputs)
        image_grid = None if meta.image_grid_thw is None else meta.image_grid_thw.detach().cpu().tolist()
        seq_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else None
        print(
            "[visualize] "
            f"source={sample_source_name} row={row_idx} id={sample.get('id', row_idx)} "
            f"seq_len={seq_len} visual_tokens={int(meta.visual_indices.numel())} "
            f"image_grid_thw={image_grid} "
            f"processor_min_pixels={_resolve_processor_pixels(config, 'min')} "
            f"processor_max_pixels={_resolve_processor_pixels(config, 'max')}"
        )
        answer = _sample_answer(
            model=model,
            processor=processor,
            inputs=inputs,
            sample=sample,
            answer_source=str(scoring_cfg.get("answer_source", "sample")),
            max_new_tokens=max_new_tokens,
        )
        gae_scores = _score_gae_oracle(
            model=model,
            processor=processor,
            adapter=adapter,
            pruner=gae_pruner,
            sample=sample,
            answer=answer,
            per_token=_bool_value(scoring_cfg.get("per_token", True)),
        )
        gae_scores = _normalize_gae_scores(
            _as_numpy(gae_scores, name="gae_scores"),
            str(scoring_cfg.get("gae_normalizer", "none")),
        )
        quant_components = _score_gae_quant_joint(
            model=model,
            processor=processor,
            adapter=adapter,
            pruner=quant_pruner,
            sample=sample,
            answer=answer,
            per_token=_bool_value(scoring_cfg.get("per_token", True)),
            quant_method=str(quant_cfg.get("quant_method", pruning_cfg.get("quant_method", "rtn"))),
            rtn_bits=int(quant_cfg.get("rtn_bits", pruning_cfg.get("rtn_bits", 4))),
            rtn_group_size=int(quant_cfg.get("rtn_group_size", pruning_cfg.get("rtn_group_size", 0))),
            return_components=True,
        )
        quant_scores = quant_components["joint"]
        predictions = {}
        if show_predictions:
            predictions = _build_prediction_variants(
                model=model,
                processor=processor,
                adapter=adapter,
                inputs=inputs,
                gae_scores=gae_scores,
                quant_scores=quant_scores,
                retention_ratio=retention_ratio,
                min_keep=min_keep,
                max_new_tokens=max_new_tokens,
            )
        artifact = _build_sample_artifact(
            model=model,
            processor=processor,
            adapter=adapter,
            sample=sample,
            inputs=inputs,
            meta=meta,
            gae_scores=gae_scores,
            quant_joint_scores=quant_scores,
            quant_components=quant_components,
            answer=answer,
            predictions=predictions,
        )
        yield artifact, {
            "retention_ratio": retention_ratio,
            "min_keep": min_keep,
            "output_dir": _resolve_path(vis_cfg.get("output_dir"), base_dir) or Path(__file__).resolve().parent / "outputs",
            "output_name": vis_cfg.get("output_name"),
            "save_sample_artifacts": _bool_value(vis_cfg.get("save_sample_artifacts", False)),
            "image_overlay": _bool_value(vis_cfg.get("image_overlay", True)),
            "green_highlight": str(vis_cfg.get("green_highlight", "proxy")),
            "score_bars": _bool_value(vis_cfg.get("score_bars", True)),
            "sample_artifact_dir": _resolve_path(vis_cfg.get("sample_artifact_dir"), base_dir)
            or Path(__file__).resolve().parent / "samples",
            "row_idx": row_idx,
        }


def _squeeze_embeddings(embeds: np.ndarray) -> np.ndarray:
    if embeds.ndim == 3:
        if embeds.shape[0] != 1:
            raise ValueError(f"Only one sample per plot is supported, got embeddings shape {embeds.shape}.")
        embeds = embeds[0]
    if embeds.ndim != 2:
        raise ValueError(f"inputs_embeds must have shape [S, D] or [1, S, D], got {embeds.shape}.")
    return embeds.astype(np.float32, copy=False)


def _infer_visual_indices(sample: dict[str, Any], seq_len: int, visual_count: int | None) -> np.ndarray:
    if "visual_indices" in sample:
        visual_indices = _as_numpy(sample["visual_indices"], name="visual_indices").astype(np.int64).reshape(-1)
    elif visual_count is not None:
        visual_indices = np.arange(int(visual_count), dtype=np.int64)
    elif "vision_text_boundary" in sample:
        visual_indices = np.arange(int(np.asarray(sample["vision_text_boundary"]).item()), dtype=np.int64)
    else:
        raise ValueError(
            "Provide visual_indices, vision_text_boundary, or --visual-count so the script can identify visual tokens."
        )
    if visual_indices.size == 0:
        raise ValueError("visual_indices is empty.")
    if visual_indices.min() < 0 or visual_indices.max() >= seq_len:
        raise ValueError("visual_indices contains positions outside the embedding sequence.")
    return np.sort(visual_indices)


def _infer_boundary(sample: dict[str, Any], visual_indices: np.ndarray, boundary: int | None) -> int:
    if boundary is not None:
        return int(boundary)
    if "vision_text_boundary" in sample:
        return int(np.asarray(sample["vision_text_boundary"]).item())
    return int(visual_indices.max()) + 1


def _select_removed(
    scores: np.ndarray,
    visual_indices: np.ndarray,
    retention_ratio: float,
    min_keep: int,
    *,
    score_mode: str,
) -> np.ndarray:
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got {scores.shape}.")
    if scores.size != visual_indices.size:
        raise ValueError(f"score count {scores.size} does not match visual token count {visual_indices.size}.")
    if not (0.0 < retention_ratio <= 1.0):
        raise ValueError("--retention-ratio must be in (0, 1].")
    if min_keep < 0:
        raise ValueError("--min-keep must be non-negative.")

    keep_count = max(min_keep, math.ceil(visual_indices.size * retention_ratio))
    keep_count = min(keep_count, visual_indices.size)
    drop_count = visual_indices.size - keep_count
    if drop_count <= 0:
        return np.empty((0,), dtype=np.int64)

    if score_mode == "keep":
        order = np.argsort(scores, kind="stable")
        removed_local = order[:drop_count]
    elif score_mode == "drop":
        order = np.argsort(-scores, kind="stable")
        removed_local = order[:drop_count]
    else:
        raise ValueError("score_mode must be 'keep' or 'drop'.")
    return np.sort(visual_indices[removed_local])


def _select_top_percent(values: np.ndarray, positions: np.ndarray, *, fraction: float) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got {values.shape}.")
    if values.size != positions.size:
        raise ValueError(f"value count {values.size} does not match position count {positions.size}.")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")
    if values.size == 0:
        return np.empty((0,), dtype=np.int64)

    selected_count = max(1, math.ceil(values.size * fraction))
    order = np.argsort(-values, kind="stable")
    return positions[order[:selected_count]].astype(np.int64, copy=False)


def _rank_normalize_np(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"values must be 1D for rank normalization, got {values.shape}.")
    if values.size == 0:
        raise ValueError("values must be non-empty for rank normalization.")
    if values.size == 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order].astype(np.float32, copy=False)
    sorted_ranks = np.empty(sorted_values.shape, dtype=np.float32)
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = (start + end - 1) / 2.0
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks / float(values.size - 1)


def _normalize_gae_scores(scores: np.ndarray, normalizer: str) -> np.ndarray:
    mode = str(normalizer).strip().lower()
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if mode in {"none", "raw"}:
        return scores.copy()
    if mode == "sum":
        denom = max(float(scores.sum()), np.finfo(np.float32).eps)
        return scores / denom
    if mode in {"rn", "rank", "rank_norm", "rank-normalize"}:
        return _rank_normalize_np(scores)
    raise ValueError("GAE normalizer must be one of: sum, RN, none.")


def _green_highlight_spec(
    mode: str,
    *,
    embeds: np.ndarray,
    sample: dict[str, Any],
    positions: np.ndarray,
    fraction: float = 0.2,
) -> GreenHighlightSpec:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"none", "null", "off"}:
        return GreenHighlightSpec(
            mode="none",
            label="None",
            values=None,
            tokens=np.empty((0,), dtype=np.int64),
        )
    if normalized_mode == "proxy":
        values = _visual_outlier_proxy(embeds)
        return GreenHighlightSpec(
            mode="proxy",
            label="Abs-max proxy",
            values=values,
            tokens=_select_top_percent(values, positions, fraction=fraction),
        )
    if normalized_mode in {"c_quant", "quant", "top_c_quant", "top-c-quant"}:
        values = _score_array(sample, "c_quant", positions.size, required=True)
        return GreenHighlightSpec(
            mode="c_quant",
            label="Top C_i^quant",
            values=values,
            tokens=_select_top_percent(values, positions, fraction=fraction),
        )
    raise ValueError("green highlight mode must be one of: none, proxy, c_quant.")


def _visual_outlier_proxy(embeds: np.ndarray) -> np.ndarray:
    if embeds.ndim != 2:
        raise ValueError(f"embeds must be 2D, got {embeds.shape}.")
    return np.abs(embeds).max(axis=1)


def _remaining_embeddings(embeds: np.ndarray, removed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep_mask = np.ones(embeds.shape[0], dtype=bool)
    keep_mask[removed] = False
    kept_positions = np.flatnonzero(keep_mask)
    return embeds[keep_mask], kept_positions


def _first_image_grid(sample: dict[str, Any]) -> tuple[int, int, int] | None:
    value = sample.get("image_grid_thw")
    if value is None:
        return None
    grid = _as_numpy(value, name="image_grid_thw").astype(np.int64).reshape(-1, 3)
    if grid.shape[0] != 1:
        print(f"[visualize] image overlay currently uses the first image only; image_grid_thw={grid.tolist()}")
    if grid.shape[0] == 0:
        return None
    t, h, w = grid[0].tolist()
    return int(t), int(h), int(w)


def _mask_for_image_tokens(
    tokens: np.ndarray,
    *,
    image_grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
) -> np.ndarray:
    t, h, w = image_grid_thw
    merge = max(1, int(spatial_merge_size))
    grid_h = max(1, h // merge)
    grid_w = max(1, w // merge)
    per_frame = grid_h * grid_w
    first_image_tokens = max(1, int(t) * per_frame)
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for token_idx in tokens.astype(np.int64).tolist():
        if token_idx < 0 or token_idx >= first_image_tokens:
            continue
        within_frame = token_idx % per_frame
        y = within_frame // grid_w
        x = within_frame % grid_w
        mask[y, x] = True
    return mask


def _removed_mask_for_image(
    removed: np.ndarray,
    *,
    image_grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
) -> np.ndarray:
    return _mask_for_image_tokens(
        removed,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size,
    )


def _draw_image_overlay_panel(
    image: Any,
    removed_mask: np.ndarray,
    *,
    outlier_mask: np.ndarray | None = None,
    title: str,
    alpha: int = 118,
) -> Any:
    from PIL import Image, ImageDraw

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    if outlier_mask is None:
        outlier_mask = np.zeros_like(removed_mask, dtype=bool)
    if outlier_mask.shape != removed_mask.shape:
        raise ValueError("outlier_mask shape must match removed_mask shape.")
    h, w = removed_mask.shape
    cell_w = base.width / float(w)
    cell_h = base.height / float(h)
    for y in range(h):
        for x in range(w):
            is_removed = bool(removed_mask[y, x])
            is_outlier = bool(outlier_mask[y, x])
            if not is_removed and not is_outlier:
                continue
            box = (
                int(round(x * cell_w)),
                int(round(y * cell_h)),
                int(round((x + 1) * cell_w)),
                int(round((y + 1) * cell_h)),
            )
            if is_outlier:
                draw.rectangle(box, fill=(24, 168, 84, 104), outline=(0, 115, 55, 190), width=1)
            if is_removed and not is_outlier:
                draw.rectangle(box, fill=(220, 22, 22, alpha), outline=(150, 0, 0, 180), width=1)
            if is_removed and is_outlier:
                draw.rectangle(box, outline=(170, 0, 0, 230), width=3)
                draw.line((box[0], box[1], box[2], box[3]), fill=(170, 0, 0, 220), width=2)
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return composed, title


def _load_overlay_font(size: int) -> Any:
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _metadata_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            value = value.tolist()
    if hasattr(value, "detach"):
        value = _as_numpy(value, name="metadata")
        if value.shape == ():
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = " ".join(_metadata_string(item) for item in value)
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _text_width(draw: Any, text: str, font: Any) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _line_height(font: Any) -> int:
    try:
        bbox = font.getbbox("Ag")
        return int(bbox[3] - bbox[1]) + 6
    except AttributeError:
        return 18


def _split_to_width(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = f"{current}{char}"
        if current and _text_width(draw, candidate, font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            wrapped.append("")
            continue
        if _text_width(draw, paragraph, font) <= max_width:
            wrapped.append(paragraph)
            continue
        words = paragraph.split(" ")
        if len(words) == 1:
            wrapped.extend(_split_to_width(draw, paragraph, font, max_width))
            continue
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if _text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue
            if current:
                wrapped.append(current)
            if _text_width(draw, word, font) <= max_width:
                current = word
            else:
                pieces = _split_to_width(draw, word, font, max_width)
                wrapped.extend(pieces[:-1])
                current = pieces[-1]
        if current:
            wrapped.append(current)
    return wrapped


def _overlay_text_lines(sample: dict[str, Any], draw: Any, font: Any, max_width: int, max_lines: int = 14) -> list[str]:
    question = (
        _metadata_string(sample.get("question"))
        or _metadata_string(sample.get("prompt"))
        or _metadata_string(sample.get("text"))
    )
    answer = _metadata_string(sample.get("answer")) or _metadata_string(sample.get("reference_answer"))
    prediction_entries = [
        ("Original prediction", _prediction_entry(sample, "original")),
        ("GAE prediction", _prediction_entry(sample, "gae_pruned")),
        ("Quant-joint prediction", _prediction_entry(sample, "quant_joint_pruned")),
    ]
    entries = []
    if question:
        entries.append(f"Question: {question}")
    if answer:
        entries.append(f"Answer: {answer}")
    for label, prediction in prediction_entries:
        if prediction:
            entries.append(f"{label}: {prediction}")
    if not entries:
        return []

    lines: list[str] = []
    for entry in entries:
        if lines:
            lines.append("")
        lines.extend(_wrap_text(draw, entry, font, max_width))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        suffix = " ..."
        while lines[-1] and _text_width(draw, f"{lines[-1]}{suffix}", font) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] = f"{lines[-1]}{suffix}".strip()
    return lines


def _save_image_overlay(
    *,
    sample: dict[str, Any],
    sample_id: str,
    output_path: Path,
    gae_removed: np.ndarray,
    quant_removed: np.ndarray,
    green_spec: GreenHighlightSpec,
) -> Path | None:
    from PIL import Image, ImageDraw

    image_value = sample.get("image")
    image_path = sample.get("image_path")
    if isinstance(image_value, Image.Image):
        image = image_value.convert("RGB")
    else:
        if not image_path:
            print("[visualize] skipped image overlay: sample has no image or image_path.")
            return None
        image_path = Path(str(image_path)).expanduser()
        if not image_path.exists():
            print(f"[visualize] skipped image overlay: image_path does not exist: {image_path}")
            return None
        image = Image.open(image_path).convert("RGB")
    grid = _first_image_grid(sample)
    if grid is None:
        print("[visualize] skipped image overlay: missing image_grid_thw.")
        return None
    spatial_merge_size_value = sample.get("spatial_merge_size", 2)
    if hasattr(spatial_merge_size_value, "detach"):
        spatial_merge_size = int(_as_numpy(spatial_merge_size_value, name="spatial_merge_size").item())
    else:
        spatial_merge_size = int(np.asarray(spatial_merge_size_value).item())
    gae_mask = _removed_mask_for_image(
        gae_removed,
        image_grid_thw=grid,
        spatial_merge_size=spatial_merge_size,
    )
    quant_mask = _removed_mask_for_image(
        quant_removed,
        image_grid_thw=grid,
        spatial_merge_size=spatial_merge_size,
    )
    green_mask = _mask_for_image_tokens(
        green_spec.tokens,
        image_grid_thw=grid,
        spatial_merge_size=spatial_merge_size,
    )
    green_count = int(green_mask.sum())
    green_suffix = "" if green_spec.mode == "none" else f" + {green_spec.label} ({green_count})"
    panels = [
        (image, "Original image"),
        _draw_image_overlay_panel(
            image,
            gae_mask,
            outlier_mask=green_mask,
            title=f"GAE removed ({int(gae_mask.sum())}){green_suffix}",
        ),
        _draw_image_overlay_panel(
            image,
            quant_mask,
            outlier_mask=green_mask,
            title=f"Quant-joint removed ({int(quant_mask.sum())}){green_suffix}",
        ),
    ]
    max_panel_w = 560
    resized = []
    for panel, title in panels:
        scale = min(1.0, max_panel_w / float(panel.width))
        new_size = (max(1, int(panel.width * scale)), max(1, int(panel.height * scale)))
        resized.append((panel.resize(new_size), title))
    title_h = 34
    margin = 24
    gutter = 20
    canvas_w = sum(panel.width for panel, _ in resized) + gutter * (len(resized) - 1) + margin * 2
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    font = _load_overlay_font(14)
    header_font = _load_overlay_font(16)
    text_font = _load_overlay_font(15)
    overlay_lines = _overlay_text_lines(sample, measure, text_font, canvas_w - 2 * margin)
    text_line_h = _line_height(text_font)
    text_h = 0 if not overlay_lines else 20 + len(overlay_lines) * text_line_h
    legend_h = 26
    canvas_h = max(panel.height for panel, _ in resized) + title_h + margin * 2 + 28 + legend_h + text_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        f"Pruned visual tokens projected to image: {sample_id}",
        fill=(10, 20, 30),
        font=header_font,
    )
    x = margin
    y = margin + title_h
    for panel, title in resized:
        draw.text((x, y - 18), title, fill=(35, 45, 55), font=font)
        canvas.paste(panel, (x, y))
        x += panel.width + gutter
    legend_y = y + max(panel.height for panel, _ in resized) + 18
    draw.rectangle((margin, legend_y - 7, margin + 18, legend_y + 7), fill=(220, 22, 22), outline=(150, 0, 0))
    draw.text((margin + 26, legend_y - 11), "removed token", fill=(35, 45, 55), font=font)
    if green_spec.mode != "none":
        draw.rectangle((margin + 170, legend_y - 7, margin + 188, legend_y + 7), fill=(24, 168, 84), outline=(0, 115, 55))
        draw.text((margin + 196, legend_y - 11), green_spec.label, fill=(35, 45, 55), font=font)
        draw.rectangle((margin + 442, legend_y - 7, margin + 460, legend_y + 7), fill=(24, 168, 84), outline=(170, 0, 0), width=3)
        draw.line((margin + 442, legend_y - 7, margin + 460, legend_y + 7), fill=(170, 0, 0), width=2)
        draw.text((margin + 468, legend_y - 11), "both", fill=(35, 45, 55), font=font)
    if overlay_lines:
        text_y = legend_y + 22
        for line in overlay_lines:
            draw.text((margin, text_y), line, fill=(20, 30, 42), font=text_font)
            text_y += text_line_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Wrote {output_path}")
    return output_path


def _plot_token_ranges(
    ax: Any,
    embeds: np.ndarray,
    positions: np.ndarray,
    boundary_position: float | None,
    removed: np.ndarray,
    *,
    title: str,
    show_removed: bool,
) -> None:
    mins = embeds.min(axis=1)
    maxs = embeds.max(axis=1)
    removed_set = set(int(item) for item in removed.tolist())
    colors = np.array(["#C81E1E" if int(pos) in removed_set else "#356A8A" for pos in positions], dtype=object)
    if not show_removed:
        colors[:] = "#356A8A"
    linewidths = np.where(colors == "#C81E1E", 1.8, 0.85)

    ax.vlines(positions, mins, maxs, colors=colors, linewidth=linewidths, alpha=0.9)
    if boundary_position is not None:
        ax.axvline(boundary_position, color="black", linewidth=3.0, alpha=0.95)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Visual token idx")
    ax.set_ylabel("proxy value")
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _scale_values(values: np.ndarray, vmin: float, vmax: float, top: int, bottom: int) -> np.ndarray:
    if vmax <= vmin:
        return np.full_like(values, fill_value=(top + bottom) / 2.0, dtype=np.float32)
    return bottom - ((values - vmin) / (vmax - vmin)) * (bottom - top)


def _draw_pillow_panel(
    draw: Any,
    box: tuple[int, int, int, int],
    embeds: np.ndarray,
    positions: np.ndarray,
    boundary_position: float | None,
    removed: np.ndarray,
    *,
    title: str,
    show_removed: bool,
    value_min: float,
    value_max: float,
) -> None:
    from PIL import ImageFont

    left, top, right, bottom = box
    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    plot_left = left + 58
    plot_top = top + 36
    plot_right = right - 18
    plot_bottom = bottom - 42
    width = max(1, plot_right - plot_left)

    draw.text((left + 8, top + 8), title, fill=(25, 34, 44), font=title_font)
    for frac in np.linspace(0.0, 1.0, 5):
        y = int(plot_top + frac * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill=(224, 229, 235), width=1)
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(170, 178, 188), width=1)
    draw.text((plot_left + width // 2 - 44, bottom - 22), "Visual token idx", fill=(70, 78, 88), font=label_font)
    draw.text((left + 8, plot_top + 8), "proxy", fill=(70, 78, 88), font=label_font)

    max_pos = max(1, int(positions.max()) if positions.size else 1)
    mins = embeds.min(axis=1)
    maxs = embeds.max(axis=1)
    y0 = _scale_values(mins, value_min, value_max, plot_top, plot_bottom)
    y1 = _scale_values(maxs, value_min, value_max, plot_top, plot_bottom)
    removed_set = set(int(item) for item in removed.tolist())

    if boundary_position is not None:
        bx = int(plot_left + (float(boundary_position) / max_pos) * width)
        draw.line((bx, plot_top, bx, plot_bottom), fill=(0, 0, 0), width=4)

    for pos, low_y, high_y in zip(positions, y0, y1):
        x = int(plot_left + (int(pos) / max_pos) * width)
        is_removed = int(pos) in removed_set and show_removed
        color = (200, 30, 30) if is_removed else (53, 106, 138)
        line_width = 2 if is_removed else 1
        draw.line((x, int(high_y), x, int(low_y)), fill=color, width=line_width)


def _save_with_pillow(
    output_path: Path,
    *,
    sample_id: str,
    retention_ratio: float,
    embeds: np.ndarray,
    boundary: int | None,
    gae_removed: np.ndarray,
    quant_removed: np.ndarray,
    gae_after: np.ndarray,
    gae_after_positions: np.ndarray,
    quant_after: np.ndarray,
    quant_after_positions: np.ndarray,
    sample: dict[str, Any] | None = None,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 1080
    margin = 46
    gutter = 34
    header = 70
    legend_h = 54
    text_font = _load_overlay_font(15)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    text_lines = _overlay_text_lines(sample or {}, measure, text_font, width - 2 * margin, max_lines=10)
    text_line_h = _line_height(text_font)
    text_h = 0 if not text_lines else 18 + len(text_lines) * text_line_h
    panel_w = (width - 2 * margin - gutter) // 2
    panel_h = max(120, (height - header - legend_h - text_h - 2 * margin - gutter) // 2)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    draw.text(
        (margin, 24),
        f"Visual-token pruning visualization: {sample_id} | retention={retention_ratio:g} | visual_tokens={embeds.shape[0]}",
        fill=(10, 20, 30),
        font=title_font,
    )

    value_min = float(embeds.min())
    value_max = float(embeds.max())
    boxes = [
        (margin, header, margin + panel_w, header + panel_h),
        (margin + panel_w + gutter, header, margin + 2 * panel_w + gutter, header + panel_h),
        (margin, header + panel_h + gutter, margin + panel_w, header + 2 * panel_h + gutter),
        (
            margin + panel_w + gutter,
            header + panel_h + gutter,
            margin + 2 * panel_w + gutter,
            header + 2 * panel_h + gutter,
        ),
    ]
    original_positions = np.arange(embeds.shape[0], dtype=np.int64)
    _draw_pillow_panel(
        draw,
        boxes[0],
        embeds,
        original_positions,
        None if boundary is None else boundary - 0.5,
        gae_removed,
        title=f"Original visual tokens | GAE removed: {gae_removed.size}",
        show_removed=True,
        value_min=value_min,
        value_max=value_max,
    )
    _draw_pillow_panel(
        draw,
        boxes[1],
        embeds,
        original_positions,
        None if boundary is None else boundary - 0.5,
        quant_removed,
        title=f"Original visual tokens | Quant-joint removed: {quant_removed.size}",
        show_removed=True,
        value_min=value_min,
        value_max=value_max,
    )
    _draw_pillow_panel(
        draw,
        boxes[2],
        gae_after,
        gae_after_positions,
        None if boundary is None else boundary - 0.5,
        np.empty((0,), dtype=np.int64),
        title="After GAE pruning",
        show_removed=False,
        value_min=value_min,
        value_max=value_max,
    )
    _draw_pillow_panel(
        draw,
        boxes[3],
        quant_after,
        quant_after_positions,
        None if boundary is None else boundary - 0.5,
        np.empty((0,), dtype=np.int64),
        title="After quant-joint pruning",
        show_removed=False,
        value_min=value_min,
        value_max=value_max,
    )

    legend_y = height - 42
    draw.line((margin, legend_y, margin + 32, legend_y), fill=(53, 106, 138), width=3)
    draw.text((margin + 40, legend_y - 7), "kept/original token range", fill=(45, 52, 61), font=title_font)
    draw.line((margin + 310, legend_y, margin + 342, legend_y), fill=(200, 30, 30), width=4)
    draw.text((margin + 350, legend_y - 7), "removed visual token", fill=(45, 52, 61), font=title_font)
    if text_lines:
        text_y = legend_y + 22
        for line in text_lines:
            draw.text((margin, text_y), line, fill=(20, 30, 42), font=text_font)
            text_y += text_line_h
    image.save(output_path)


def _score_array(sample: dict[str, Any], key: str, expected_size: int, *, required: bool = True) -> np.ndarray | None:
    value = sample.get(key)
    if value is None:
        if required:
            raise ValueError(f"Missing required field: {key}")
        return None
    scores = _as_numpy(value, name=key).astype(np.float32).reshape(-1)
    if scores.size != expected_size:
        raise ValueError(f"{key} count {scores.size} does not match visual token count {expected_size}.")
    return scores


def _nice_score_limit(values: list[np.ndarray]) -> tuple[float, float]:
    finite_values = np.concatenate([value[np.isfinite(value)] for value in values if value.size])
    if finite_values.size == 0:
        return 0.0, 1.0
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if vmin == vmax:
        pad = max(0.05, abs(vmax) * 0.1)
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.08
    return vmin - pad, vmax + pad


def _save_score_bars_matplotlib(
    output_path: Path,
    *,
    sample_id: str,
    positions: np.ndarray,
    gae_scores: np.ndarray,
    c_quant: np.ndarray,
    joint_scores: np.ndarray,
    outlier_proxy: np.ndarray,
    gae_removed: np.ndarray,
    quant_removed: np.ndarray,
    green_spec: GreenHighlightSpec,
) -> None:
    plt, _ = _load_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is not available.")
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True, constrained_layout=True)
    series = [
        ("Abs-max proxy | max(abs(channel))", outlier_proxy, "#6B7280", green_spec.tokens),
        ("GAE score | removed: lowest top-k", gae_scores, "#356A8A", gae_removed),
        (r"$C_i^{quant}$ | quant-joint removed tokens", c_quant, "#7A5C00", quant_removed),
        (r"$D_i = \lambda C_i^{quant} - C_i^{drop}$ | removed: highest top-k", joint_scores, "#8E3B46", quant_removed),
    ]
    x = np.arange(positions.size, dtype=np.int64)
    removed_color = "#C81E1E"
    outlier_color = "#18A854"
    outlier_set = set(int(item) for item in green_spec.tokens.tolist())
    for row_idx, (ax, (label, values, color, highlighted)) in enumerate(zip(axes, series)):
        highlighted_set = set(int(item) for item in highlighted.tolist())
        if row_idx == 0:
            colors = [outlier_color if green_spec.mode != "none" and int(pos) in highlighted_set else color for pos in positions]
            edgecolors = ["#007337" if green_spec.mode != "none" and int(pos) in highlighted_set else color for pos in positions]
            linewidths = [0.85 if green_spec.mode != "none" and int(pos) in highlighted_set else 0.35 for pos in positions]
        else:
            colors = [removed_color if int(pos) in highlighted_set else color for pos in positions]
            edgecolors = [
                "#00A651" if green_spec.mode != "none" and int(pos) in outlier_set else "#7F0000" if int(pos) in highlighted_set else color
                for pos in positions
            ]
            linewidths = [
                1.15 if green_spec.mode != "none" and int(pos) in outlier_set else 0.55 if int(pos) in highlighted_set else 0.35
                for pos in positions
            ]
        ax.bar(x, values, color=colors, edgecolor=edgecolors, linewidth=linewidths, width=0.86, alpha=0.9)
        ax.axhline(0.0, color="#24292F", linewidth=0.8, alpha=0.75)
        value_min, value_max = _nice_score_limit([values])
        ax.set_ylim(value_min, value_max)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#D0D7DE", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    tick_count = min(18, max(2, positions.size))
    tick_idx = np.linspace(0, positions.size - 1, num=tick_count, dtype=np.int64)
    axes[-1].set_xticks(tick_idx)
    axes[-1].set_xticklabels([str(int(positions[i])) for i in tick_idx], rotation=0)
    axes[-1].set_xlabel("Visual token idx")
    fig.suptitle(f"Per-token pruning score statistics: {sample_id}", fontsize=15, fontweight="bold")
    handles = [
        Patch(facecolor="#356A8A", label="kept / not removed by that row's pruning rule"),
        Patch(facecolor=removed_color, edgecolor="#7F0000", label="removed token"),
    ]
    if green_spec.mode != "none":
        handles.append(Patch(facecolor=outlier_color, edgecolor="#007337", label=green_spec.label))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_score_bars_pillow(
    output_path: Path,
    *,
    sample_id: str,
    positions: np.ndarray,
    gae_scores: np.ndarray,
    c_quant: np.ndarray,
    joint_scores: np.ndarray,
    outlier_proxy: np.ndarray,
    gae_removed: np.ndarray,
    quant_removed: np.ndarray,
    green_spec: GreenHighlightSpec,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1800, 1080
    margin = 52
    header = 70
    gutter = 24
    footer = 76
    panel_h = (height - header - margin - gutter * 3 - footer) // 4
    panel_w = width - 2 * margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_overlay_font(16)
    label_font = _load_overlay_font(14)
    draw.text((margin, 24), f"Per-token pruning score statistics: {sample_id}", fill=(10, 20, 30), font=title_font)

    series = [
        ("Abs-max proxy | max(abs(channel))", outlier_proxy, (107, 114, 128), green_spec.tokens),
        ("GAE score | removed: lowest top-k", gae_scores, (53, 106, 138), gae_removed),
        ("C_i^quant | quant-joint removed tokens", c_quant, (122, 92, 0), quant_removed),
        ("D_i = lambda*C_i^quant - C_i^drop | removed: highest top-k", joint_scores, (142, 59, 70), quant_removed),
    ]
    x_count = max(1, positions.size)
    removed_color = (200, 30, 30)
    removed_outline = (127, 0, 0)
    outlier_color = (24, 168, 84)
    outlier_outline = (0, 115, 55)
    outlier_set = set(int(item) for item in green_spec.tokens.tolist())
    for panel_idx, (label, values, color, highlighted) in enumerate(series):
        highlighted_set = set(int(item) for item in highlighted.tolist())
        left = margin
        top = header + panel_idx * (panel_h + gutter)
        right = left + panel_w
        bottom = top + panel_h
        plot_left = left + 72
        plot_top = top + 28
        plot_right = right - 24
        plot_bottom = bottom - 34
        value_min, value_max = _nice_score_limit([values])
        draw.text((left + 8, top + 4), label, fill=(25, 34, 44), font=label_font)
        for frac in np.linspace(0.0, 1.0, 5):
            y = int(plot_top + frac * (plot_bottom - plot_top))
            draw.line((plot_left, y, plot_right, y), fill=(224, 229, 235), width=1)
        zero_y = int(_scale_values(np.asarray([0.0], dtype=np.float32), value_min, value_max, plot_top, plot_bottom)[0])
        draw.line((plot_left, zero_y, plot_right, zero_y), fill=(45, 52, 61), width=2)
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(170, 178, 188), width=1)
        plot_w = max(1, plot_right - plot_left)
        bar_w = max(1, int(plot_w / x_count * 0.78))
        scaled = _scale_values(values, value_min, value_max, plot_top, plot_bottom)
        for idx, y_value in enumerate(scaled):
            center_x = int(plot_left + (idx + 0.5) / x_count * plot_w)
            x0 = center_x - bar_w // 2
            x1 = center_x + max(1, bar_w // 2)
            y0 = int(min(zero_y, y_value))
            y1 = int(max(zero_y, y_value))
            is_highlighted = int(positions[idx]) in highlighted_set
            is_outlier = int(positions[idx]) in outlier_set
            fill = (
                outlier_color
                if green_spec.mode != "none" and panel_idx == 0 and is_highlighted
                else removed_color
                if panel_idx > 0 and is_highlighted
                else color
            )
            outline = (
                outlier_outline
                if green_spec.mode != "none" and is_outlier
                else removed_outline
                if panel_idx > 0 and is_highlighted
                else None
            )
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=fill,
                outline=outline,
            )
        draw.text((plot_left, bottom - 22), f"min={float(values.min()):.4g}", fill=(70, 78, 88), font=label_font)
        draw.text((plot_left + 150, bottom - 22), f"max={float(values.max()):.4g}", fill=(70, 78, 88), font=label_font)
    legend_y = height - 46
    draw.rectangle((margin, legend_y - 8, margin + 24, legend_y + 8), fill=(53, 106, 138))
    draw.text((margin + 32, legend_y - 10), "kept / not removed by that row's pruning rule", fill=(70, 78, 88), font=label_font)
    draw.rectangle((margin + 380, legend_y - 8, margin + 404, legend_y + 8), fill=removed_color, outline=removed_outline)
    draw.text((margin + 412, legend_y - 10), "removed token", fill=(70, 78, 88), font=label_font)
    if green_spec.mode != "none":
        draw.rectangle((margin + 550, legend_y - 8, margin + 574, legend_y + 8), fill=outlier_color, outline=outlier_outline)
        draw.text((margin + 582, legend_y - 10), green_spec.label, fill=(70, 78, 88), font=label_font)
    draw.text((width // 2 - 54, height - 24), "Visual token idx", fill=(70, 78, 88), font=label_font)
    image.save(output_path)


def _save_score_bars(
    *,
    sample: dict[str, Any],
    sample_id: str,
    output_path: Path,
    positions: np.ndarray,
    gae_scores: np.ndarray,
    quant_scores: np.ndarray,
    outlier_proxy: np.ndarray,
    gae_removed: np.ndarray,
    quant_removed: np.ndarray,
    green_spec: GreenHighlightSpec,
) -> Path | None:
    c_quant = _score_array(sample, "c_quant", positions.size, required=False)
    if c_quant is None:
        print("[visualize] skipped score bars: sample has no c_quant field.")
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _save_score_bars_matplotlib(
            output_path,
            sample_id=sample_id,
            positions=positions,
            gae_scores=gae_scores,
            c_quant=c_quant,
            joint_scores=quant_scores,
            outlier_proxy=outlier_proxy,
            gae_removed=gae_removed,
            quant_removed=quant_removed,
            green_spec=green_spec,
        )
    except RuntimeError:
        _save_score_bars_pillow(
            output_path,
            sample_id=sample_id,
            positions=positions,
            gae_scores=gae_scores,
            c_quant=c_quant,
            joint_scores=quant_scores,
            outlier_proxy=outlier_proxy,
            gae_removed=gae_removed,
            quant_removed=quant_removed,
            green_spec=green_spec,
        )
    print(f"Wrote {output_path}")
    return output_path


def _scalar_id(sample: dict[str, Any], fallback: str) -> str:
    value = sample.get("id", fallback)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            value = value.tolist()
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw a 2x2 token pruning visualization.")
    parser.add_argument("--sample", type=Path, help="Path to a .pt/.pth/.npz sample artifact.")
    parser.add_argument("--config", type=Path, help="YAML config that points to calibration data and pruning params.")
    parser.add_argument("--demo", action="store_true", help="Generate a synthetic sample to preview the plot style.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--output-name", help="Optional output PNG filename.")
    parser.add_argument("--limit", type=int, help="Override visualization.limit when using --config.")
    parser.add_argument("--image-overlay", action="store_true", help="Also project removed visual tokens back to the source image when available.")
    parser.add_argument(
        "--green-highlight",
        choices=["none", "proxy", "c_quant"],
        default="proxy",
        help="Green overlay/highlight source: none, proxy, or top C_i^quant.",
    )
    parser.add_argument(
        "--score-bars",
        dest="score_bars",
        action="store_true",
        default=True,
        help="Also draw per-token GAE, C_i^quant, and D_i score bar charts.",
    )
    parser.add_argument("--no-score-bars", dest="score_bars", action="store_false", help="Disable score bar charts.")
    parser.add_argument("--retention-ratio", type=float, default=0.5)
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--visual-count", type=int, help="Use first N tokens as visual tokens if the sample has no indices.")
    parser.add_argument("--boundary", type=int, help="Token index where text tokens begin. Defaults to max(visual_indices)+1.")
    parser.add_argument("--gae-key", default="gae_scores")
    parser.add_argument("--quant-key", default="quant_joint_scores")
    parser.add_argument("--embeds-key", default="inputs_embeds")
    parser.add_argument("--demo-visual-tokens", type=int, default=96)
    parser.add_argument("--demo-text-tokens", type=int, default=32)
    parser.add_argument("--demo-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _render_sample(
    *,
    sample: dict[str, Any],
    retention_ratio: float,
    min_keep: int,
    output_dir: Path,
    output_name: str | None,
    embeds_key: str,
    gae_key: str,
    quant_key: str,
    image_overlay: bool = False,
    score_bars: bool = True,
    green_highlight: str = "proxy",
    visual_count: int | None = None,
    boundary_override: int | None = None,
) -> Path:
    full_embeds = _squeeze_embeddings(_as_numpy(sample.get(embeds_key), name=embeds_key))
    visual_indices = _infer_visual_indices(sample, full_embeds.shape[0], visual_count)
    del boundary_override
    embeds = full_embeds[visual_indices]
    visual_positions = np.arange(visual_indices.size, dtype=np.int64)
    gae_scores = _as_numpy(sample.get(gae_key), name=gae_key).astype(np.float32).reshape(-1)
    quant_scores = _as_numpy(sample.get(quant_key), name=quant_key).astype(np.float32).reshape(-1)
    outlier_proxy = _visual_outlier_proxy(embeds)
    green_spec = _green_highlight_spec(
        green_highlight,
        embeds=embeds,
        sample=sample,
        positions=visual_positions,
    )

    gae_removed = _select_removed(
        gae_scores,
        visual_positions,
        retention_ratio,
        min_keep,
        score_mode="keep",
    )
    quant_removed = _select_removed(
        quant_scores,
        visual_positions,
        retention_ratio,
        min_keep,
        score_mode="drop",
    )
    gae_after, gae_after_positions = _remaining_embeddings(embeds, gae_removed)
    quant_after, quant_after_positions = _remaining_embeddings(embeds, quant_removed)

    sample_id = _scalar_id(sample, "sample")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_name = output_name or f"{sample_id}_token_pruning.png"
    output_path = output_dir / resolved_output_name
    overlay_path = output_path.with_name(f"{output_path.stem}_image_overlay.png")
    score_bars_path = output_path.with_name(f"{output_path.stem}_score_bars.png")

    plt, Line2D = _load_matplotlib()
    if plt is None:
        _save_with_pillow(
            output_path,
            sample_id=sample_id,
            retention_ratio=retention_ratio,
            embeds=embeds,
            boundary=None,
            gae_removed=gae_removed,
            quant_removed=quant_removed,
            gae_after=gae_after,
            gae_after_positions=gae_after_positions,
            quant_after=quant_after,
            quant_after_positions=quant_after_positions,
            sample=sample,
        )
        print(f"Wrote {output_path} (Pillow fallback; install matplotlib for publication-style axes)")
        if image_overlay:
            _save_image_overlay(
                sample=sample,
                sample_id=sample_id,
                output_path=overlay_path,
                gae_removed=gae_removed,
                quant_removed=quant_removed,
                green_spec=green_spec,
            )
        if score_bars:
            _save_score_bars(
                sample=sample,
                sample_id=sample_id,
                output_path=score_bars_path,
                positions=visual_positions,
                gae_scores=gae_scores,
                quant_scores=quant_scores,
                outlier_proxy=outlier_proxy,
                gae_removed=gae_removed,
                quant_removed=quant_removed,
                green_spec=green_spec,
            )
        return output_path

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        f"Visual-token pruning visualization: {sample_id} | retention={retention_ratio:g} | visual_tokens={embeds.shape[0]}",
        fontsize=15,
        fontweight="bold",
    )

    original_positions = np.arange(embeds.shape[0], dtype=np.int64)
    _plot_token_ranges(
        axes[0, 0],
        embeds,
        original_positions,
        None,
        gae_removed,
        title=f"Original visual tokens | GAE removed: {gae_removed.size}",
        show_removed=True,
    )
    _plot_token_ranges(
        axes[0, 1],
        embeds,
        original_positions,
        None,
        quant_removed,
        title=f"Original visual tokens | Quant-joint removed: {quant_removed.size}",
        show_removed=True,
    )
    _plot_token_ranges(
        axes[1, 0],
        gae_after,
        gae_after_positions,
        None,
        np.empty((0,), dtype=np.int64),
        title="After GAE pruning",
        show_removed=False,
    )
    _plot_token_ranges(
        axes[1, 1],
        quant_after,
        quant_after_positions,
        None,
        np.empty((0,), dtype=np.int64),
        title="After quant-joint pruning",
        show_removed=False,
    )

    legend_handles = [
        Line2D([0], [0], color="#356A8A", lw=2.0, label="kept/original token range"),
        Line2D([0], [0], color="#C81E1E", lw=2.4, label="removed visual token"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
    from PIL import Image, ImageDraw

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    text_lines = _overlay_text_lines(sample, measure, _load_overlay_font(13), 1600, max_lines=8)
    if text_lines:
        fig.text(
            0.01,
            0.01,
            "\n".join(text_lines),
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#182230",
        )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Wrote {output_path}")
    if image_overlay:
        _save_image_overlay(
            sample=sample,
            sample_id=sample_id,
            output_path=overlay_path,
            gae_removed=gae_removed,
            quant_removed=quant_removed,
            green_spec=green_spec,
        )
    if score_bars:
        _save_score_bars(
            sample=sample,
            sample_id=sample_id,
            output_path=score_bars_path,
            positions=visual_positions,
            gae_scores=gae_scores,
            quant_scores=quant_scores,
            outlier_proxy=outlier_proxy,
            gae_removed=gae_removed,
            quant_removed=quant_removed,
            green_spec=green_spec,
        )
    return output_path


def _save_sample_artifact(sample: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample, path)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.config:
        for artifact, render_cfg in _iter_config_artifacts(args.config.expanduser().resolve(), cli_limit=args.limit):
            output_name = render_cfg["output_name"]
            if output_name and (args.limit or int(render_cfg["row_idx"]) > 0):
                stem = Path(output_name).stem
                suffix = Path(output_name).suffix or ".png"
                output_name = f"{stem}_{int(render_cfg['row_idx']):04d}{suffix}"
            _render_sample(
                sample=artifact,
                retention_ratio=float(render_cfg["retention_ratio"]),
                min_keep=int(render_cfg["min_keep"]),
                output_dir=Path(render_cfg["output_dir"]),
                output_name=output_name,
                embeds_key="inputs_embeds",
                gae_key="gae_scores",
                quant_key="quant_joint_scores",
                image_overlay=bool(render_cfg["image_overlay"]),
                score_bars=bool(render_cfg["score_bars"]),
                green_highlight=str(render_cfg["green_highlight"]),
            )
            if render_cfg["save_sample_artifacts"]:
                sample_id = _scalar_id(artifact, f"sample_{int(render_cfg['row_idx']):04d}")
                _save_sample_artifact(artifact, Path(render_cfg["sample_artifact_dir"]) / f"{sample_id}.pt")
        return

    if args.demo:
        sample = _make_demo_sample(args.demo_visual_tokens, args.demo_text_tokens, args.demo_dim, args.seed)
    elif args.sample:
        sample = _load_sample(args.sample.expanduser().resolve())
    else:
        raise SystemExit("Use --config YAML, --sample PATH, or --demo.")

    _render_sample(
        sample=sample,
        retention_ratio=args.retention_ratio,
        min_keep=args.min_keep,
        output_dir=args.output_dir,
        output_name=args.output_name,
        embeds_key=args.embeds_key,
        gae_key=args.gae_key,
        quant_key=args.quant_key,
        image_overlay=args.image_overlay,
        score_bars=args.score_bars,
        green_highlight=args.green_highlight,
        visual_count=args.visual_count,
        boundary_override=args.boundary,
    )


if __name__ == "__main__":
    main()
