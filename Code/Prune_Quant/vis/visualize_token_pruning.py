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
from pathlib import Path
from typing import Any

import numpy as np


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
        value = value.detach().cpu().numpy()
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
    parser.add_argument("--demo", action="store_true", help="Generate a synthetic sample to preview the plot style.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--output-name", help="Optional output PNG filename.")
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


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.demo:
        sample = _make_demo_sample(args.demo_visual_tokens, args.demo_text_tokens, args.demo_dim, args.seed)
    elif args.sample:
        sample = _load_sample(args.sample.expanduser().resolve())
    else:
        raise SystemExit("Use either --sample PATH or --demo.")

    embeds = _squeeze_embeddings(_as_numpy(sample.get(args.embeds_key), name=args.embeds_key))
    visual_indices = _infer_visual_indices(sample, embeds.shape[0], args.visual_count)
    boundary = _infer_boundary(sample, visual_indices, args.boundary)
    gae_scores = _as_numpy(sample.get(args.gae_key), name=args.gae_key).astype(np.float32).reshape(-1)
    quant_scores = _as_numpy(sample.get(args.quant_key), name=args.quant_key).astype(np.float32).reshape(-1)

    gae_removed = _select_removed(
        gae_scores,
        visual_indices,
        args.retention_ratio,
        args.min_keep,
        score_mode="keep",
    )
    quant_removed = _select_removed(
        quant_scores,
        visual_indices,
        args.retention_ratio,
        args.min_keep,
        score_mode="drop",
    )
    gae_after, gae_after_positions = _remaining_embeddings(embeds, gae_removed)
    quant_after, quant_after_positions = _remaining_embeddings(embeds, quant_removed)

    sample_id = _scalar_id(sample, "sample")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{sample_id}_token_pruning.png"
    output_path = output_dir / output_name

    plt, Line2D = _load_matplotlib()
    if plt is None:
        _save_with_pillow(
            output_path,
            sample_id=sample_id,
            retention_ratio=args.retention_ratio,
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
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        f"Token pruning visualization: {sample_id} | retention={args.retention_ratio:g}",
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


if __name__ == "__main__":
    main()
