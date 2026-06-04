#!/usr/bin/env python3
"""Visualize token ranges removed by GAE and quantization-aware pruning.

The script expects one sample per run. A sample contains token embeddings
(`inputs_embeds`) and two visual-token score vectors:

* GAE scores are keep scores, so low-scoring visual tokens are removed.
* Quant-joint scores are drop scores, so high-scoring visual tokens are removed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported sample file suffix {suffix!r}; use .pt, .pth, or .npz.")


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
    quant_joint_scores = (
        0.6 * np.exp(-((token_axis - 0.18) ** 2) / 0.018)
        + 0.8 * np.exp(-((token_axis - 0.62) ** 2) / 0.02)
        + 0.05 * rng.random(num_visual)
    )
    return {
        "id": np.asarray("demo_sample"),
        "inputs_embeds": inputs_embeds,
        "visual_indices": np.arange(num_visual, dtype=np.int64),
        "text_indices": np.arange(num_visual, num_visual + num_text, dtype=np.int64),
        "gae_scores": gae_scores.astype(np.float32),
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
        "quant_joint": {
            "quant_lambda": 1.0,
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
            "max_new_tokens": 16,
        },
        "visualization": {
            "limit": 1,
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
) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        inputs_embeds = adapter.build_inputs_embeds(model, inputs)
    boundary = int(meta.visual_indices.max().item()) + 1
    return {
        "id": sample.get("id", "sample"),
        "inputs_embeds": inputs_embeds.detach().float().cpu(),
        "visual_indices": meta.visual_indices.detach().cpu(),
        "text_indices": None if meta.text_indices is None else meta.text_indices.detach().cpu(),
        "vision_text_boundary": boundary,
        "gae_scores": gae_scores.detach().float().cpu(),
        "quant_joint_scores": quant_joint_scores.detach().float().cpu(),
    }


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
    data_cfg = config.get("data", {})
    scoring_cfg = config["scoring"]
    quant_cfg = config["quant_joint"]
    pruning_cfg = config["pruning"]
    vis_cfg = config["visualization"]

    calib_path = _resolve_path(
        calibration_cfg.get("path")
        or calibration_cfg.get("calib_jsonl")
        or calibration_cfg.get("input_jsonl")
        or data_cfg.get("calib_jsonl")
        or data_cfg.get("input_jsonl"),
        base_dir,
    )
    if calib_path is None:
        raise ValueError("Missing required config field: calibration.path")
    image_root = _resolve_path(calibration_cfg.get("image_root") or data_cfg.get("image_root"), base_dir)
    limit = cli_limit if cli_limit is not None else int(vis_cfg.get("limit", 1))
    sample_offset = int(vis_cfg.get("sample_offset", 0))

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
    quant_lambda = float(quant_cfg.get("quant_lambda", pruning_cfg.get("quant_lambda", 1.0)))
    quant_pruner = QuantJointGAEPruner(quant_lambda=quant_lambda)

    produced = 0
    for row_idx, raw_sample in enumerate(_read_jsonl(calib_path)):
        if row_idx < sample_offset:
            continue
        if limit is not None and produced >= limit:
            break
        sample = _prepare_sample_paths(raw_sample, image_root)
        inputs = adapter.prepare_inputs(processor, sample)
        inputs = _move_inputs_to_model_device(model, inputs)
        meta = adapter.get_visual_token_meta(model, inputs)
        answer = _sample_answer(
            model=model,
            processor=processor,
            inputs=inputs,
            sample=sample,
            answer_source=str(scoring_cfg.get("answer_source", "sample")),
            max_new_tokens=int(scoring_cfg.get("max_new_tokens", 16)),
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
        quant_scores = _score_gae_quant_joint(
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
        )
        yield artifact, {
            "retention_ratio": float(pruning_cfg.get("retention_ratio", 0.5)),
            "min_keep": int(pruning_cfg.get("min_keep", 1)),
            "output_dir": _resolve_path(vis_cfg.get("output_dir"), base_dir) or Path(__file__).resolve().parent / "outputs",
            "output_name": vis_cfg.get("output_name"),
            "save_sample_artifacts": _bool_value(vis_cfg.get("save_sample_artifacts", False)),
            "sample_artifact_dir": _resolve_path(vis_cfg.get("sample_artifact_dir"), base_dir)
            or Path(__file__).resolve().parent / "samples",
            "row_idx": row_idx,
        }
        produced += 1


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


def _remaining_embeddings(embeds: np.ndarray, removed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep_mask = np.ones(embeds.shape[0], dtype=bool)
    keep_mask[removed] = False
    kept_positions = np.flatnonzero(keep_mask)
    return embeds[keep_mask], kept_positions


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
    ax.set_xlabel("Token idx")
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
    draw.text((plot_left + width // 2 - 28, bottom - 22), "Token idx", fill=(70, 78, 88), font=label_font)
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
    boundary: int,
    gae_removed: np.ndarray,
    quant_removed: np.ndarray,
    gae_after: np.ndarray,
    gae_after_positions: np.ndarray,
    quant_after: np.ndarray,
    quant_after_positions: np.ndarray,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 1080
    margin = 46
    gutter = 34
    header = 70
    legend_h = 54
    panel_w = (width - 2 * margin - gutter) // 2
    panel_h = (height - header - legend_h - 2 * margin - gutter) // 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    draw.text(
        (margin, 24),
        f"Token pruning visualization: {sample_id} | retention={retention_ratio:g}",
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
        boundary - 0.5,
        gae_removed,
        title=f"Original tokens | GAE removed: {gae_removed.size}",
        show_removed=True,
        value_min=value_min,
        value_max=value_max,
    )
    _draw_pillow_panel(
        draw,
        boxes[1],
        embeds,
        original_positions,
        boundary - 0.5,
        quant_removed,
        title=f"Original tokens | Quant-joint removed: {quant_removed.size}",
        show_removed=True,
        value_min=value_min,
        value_max=value_max,
    )
    _draw_pillow_panel(
        draw,
        boxes[2],
        gae_after,
        gae_after_positions,
        boundary - 0.5,
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
        boundary - 0.5,
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
    draw.line((margin + 570, legend_y, margin + 602, legend_y), fill=(0, 0, 0), width=4)
    draw.text((margin + 610, legend_y - 7), "vision/text boundary", fill=(45, 52, 61), font=title_font)
    image.save(output_path)


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
    visual_count: int | None = None,
    boundary_override: int | None = None,
) -> Path:
    embeds = _squeeze_embeddings(_as_numpy(sample.get(embeds_key), name=embeds_key))
    visual_indices = _infer_visual_indices(sample, embeds.shape[0], visual_count)
    boundary = _infer_boundary(sample, visual_indices, boundary_override)
    gae_scores = _as_numpy(sample.get(gae_key), name=gae_key).astype(np.float32).reshape(-1)
    quant_scores = _as_numpy(sample.get(quant_key), name=quant_key).astype(np.float32).reshape(-1)

    gae_removed = _select_removed(
        gae_scores,
        visual_indices,
        retention_ratio,
        min_keep,
        score_mode="keep",
    )
    quant_removed = _select_removed(
        quant_scores,
        visual_indices,
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

    plt, Line2D = _load_matplotlib()
    if plt is None:
        _save_with_pillow(
            output_path,
            sample_id=sample_id,
            retention_ratio=retention_ratio,
            embeds=embeds,
            boundary=boundary,
            gae_removed=gae_removed,
            quant_removed=quant_removed,
            gae_after=gae_after,
            gae_after_positions=gae_after_positions,
            quant_after=quant_after,
            quant_after_positions=quant_after_positions,
        )
        print(f"Wrote {output_path} (Pillow fallback; install matplotlib for publication-style axes)")
        return output_path

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        f"Token pruning visualization: {sample_id} | retention={retention_ratio:g}",
        fontsize=15,
        fontweight="bold",
    )

    original_positions = np.arange(embeds.shape[0], dtype=np.int64)
    _plot_token_ranges(
        axes[0, 0],
        embeds,
        original_positions,
        boundary - 0.5,
        gae_removed,
        title=f"Original tokens | GAE removed: {gae_removed.size}",
        show_removed=True,
    )
    _plot_token_ranges(
        axes[0, 1],
        embeds,
        original_positions,
        boundary - 0.5,
        quant_removed,
        title=f"Original tokens | Quant-joint removed: {quant_removed.size}",
        show_removed=True,
    )
    _plot_token_ranges(
        axes[1, 0],
        gae_after,
        gae_after_positions,
        boundary - 0.5,
        np.empty((0,), dtype=np.int64),
        title="After GAE pruning",
        show_removed=False,
    )
    _plot_token_ranges(
        axes[1, 1],
        quant_after,
        quant_after_positions,
        boundary - 0.5,
        np.empty((0,), dtype=np.int64),
        title="After quant-joint pruning",
        show_removed=False,
    )

    legend_handles = [
        Line2D([0], [0], color="#356A8A", lw=2.0, label="kept/original token range"),
        Line2D([0], [0], color="#C81E1E", lw=2.4, label="removed visual token"),
        Line2D([0], [0], color="black", lw=3.0, label="vision/text boundary"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Wrote {output_path}")
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
        visual_count=args.visual_count,
        boundary_override=args.boundary,
    )


if __name__ == "__main__":
    main()
