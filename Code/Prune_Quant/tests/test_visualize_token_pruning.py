from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "vis" / "visualize_token_pruning.py"
SPEC = spec_from_file_location("visualize_token_pruning", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
visualize_token_pruning = module_from_spec(SPEC)
SPEC.loader.exec_module(visualize_token_pruning)


def test_select_top_percent_uses_ceil_and_stable_descending_order() -> None:
    values = np.asarray([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], dtype=np.float32)

    selected = visualize_token_pruning._select_top_percent(values, np.arange(values.size), fraction=0.2)

    assert selected.tolist() == [1, 3]


def test_absmax_proxy_prefers_largest_absolute_channel() -> None:
    embeds = np.asarray(
        [
            [-3.0, 3.0],
            [-7.0, 1.0],
            [-2.0, 2.0],
        ],
        dtype=np.float32,
    )

    proxy = visualize_token_pruning._visual_outlier_proxy(embeds)
    selected = visualize_token_pruning._select_top_percent(proxy, np.arange(proxy.size), fraction=1 / 3)

    assert proxy.tolist() == [3.0, 7.0, 2.0]
    assert selected.tolist() == [1]


def test_green_highlight_none_selects_no_tokens() -> None:
    embeds = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    sample = {"c_quant": np.asarray([0.8, 0.1], dtype=np.float32)}
    positions = np.arange(embeds.shape[0])

    spec = visualize_token_pruning._green_highlight_spec(
        "none",
        embeds=embeds,
        sample=sample,
        positions=positions,
    )

    assert spec.values is None
    assert spec.tokens.size == 0
    assert spec.label == "None"


def test_green_highlight_proxy_uses_absmax_proxy() -> None:
    embeds = np.asarray(
        [
            [1.0, 0.0],
            [0.0, -4.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    sample = {"c_quant": np.asarray([0.9, 0.1, 0.8], dtype=np.float32)}
    positions = np.arange(embeds.shape[0])

    spec = visualize_token_pruning._green_highlight_spec(
        "proxy",
        embeds=embeds,
        sample=sample,
        positions=positions,
        fraction=1 / 3,
    )

    assert spec.values.tolist() == [1.0, 4.0, 3.0]
    assert spec.tokens.tolist() == [1]
    assert spec.label == "Abs-max proxy"


def test_green_highlight_c_quant_selects_top_quant_difficulty() -> None:
    embeds = np.asarray(
        [
            [10.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    sample = {"c_quant": np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32)}
    positions = np.arange(embeds.shape[0])

    spec = visualize_token_pruning._green_highlight_spec(
        "c_quant",
        embeds=embeds,
        sample=sample,
        positions=positions,
    )

    assert np.allclose(spec.values, np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32))
    assert spec.tokens.tolist() == [1]
    assert "C_i^quant" in spec.label


def test_normalize_gae_scores_supports_sum_rank_and_none() -> None:
    scores = np.asarray([2.0, 6.0, 2.0], dtype=np.float32)

    sum_scores = visualize_token_pruning._normalize_gae_scores(scores, "sum")
    rank_scores = visualize_token_pruning._normalize_gae_scores(scores, "rn")
    raw_scores = visualize_token_pruning._normalize_gae_scores(scores, "none")

    assert np.allclose(sum_scores, np.asarray([0.2, 0.6, 0.2], dtype=np.float32))
    assert np.allclose(rank_scores, np.asarray([0.25, 1.0, 0.25], dtype=np.float32))
    assert np.allclose(raw_scores, scores)


def test_config_defaults_use_half_lambda_and_no_gae_normalizer(tmp_path) -> None:
    cfg_path = tmp_path / "visualization.yaml"
    cfg_path.write_text("visualization:\n  limit: 1\n", encoding="utf-8")

    cfg = visualize_token_pruning._load_visualization_config(cfg_path)

    assert cfg["quant_joint"]["quant_lambda"] == 0.5
    assert cfg["scoring"]["gae_normalizer"] == "none"
    assert cfg["visualization"]["show_predictions"] is True


def test_build_sample_artifact_stores_prediction_variants() -> None:
    torch = pytest.importorskip("torch")

    class Adapter:
        def build_inputs_embeds(self, model, inputs):
            del model, inputs
            return torch.zeros((1, 3, 2), dtype=torch.float32)

    class Meta:
        visual_indices = torch.as_tensor([0, 1], dtype=torch.long)
        text_indices = torch.as_tensor([2], dtype=torch.long)
        image_grid_thw = None

    artifact = visualize_token_pruning._build_sample_artifact(
        model=None,
        processor=None,
        adapter=Adapter(),
        sample={"id": "sample-1", "answer": "yes"},
        inputs={"input_ids": torch.as_tensor([[1, 2, 3]], dtype=torch.long)},
        meta=Meta(),
        gae_scores=np.asarray([0.1, 0.2], dtype=np.float32),
        quant_joint_scores=np.asarray([0.3, 0.4], dtype=np.float32),
        predictions={
            "original": "vanilla answer",
            "gae_pruned": "gae answer",
            "quant_joint_pruned": "joint answer",
        },
    )

    assert artifact["predictions"] == {
        "original": "vanilla answer",
        "gae_pruned": "gae answer",
        "quant_joint_pruned": "joint answer",
    }


def test_overlay_text_lines_include_prediction_variants() -> None:
    from PIL import Image, ImageDraw, ImageFont

    sample = {
        "question": "What is shown?",
        "answer": "cat",
        "predictions": {
            "original": "a cat",
            "gae_pruned": "a dog",
            "quant_joint_pruned": "a cat on a mat",
        },
    }
    draw = ImageDraw.Draw(Image.new("RGB", (640, 480), "white"))
    lines = visualize_token_pruning._overlay_text_lines(
        sample,
        draw,
        ImageFont.load_default(),
        max_width=600,
        max_lines=12,
    )

    text = "\n".join(lines)
    assert "Original prediction: a cat" in text
    assert "GAE prediction: a dog" in text
    assert "Quant-joint prediction: a cat on a mat" in text


def test_mask_for_image_tokens_projects_local_indices_to_first_frame() -> None:
    selected = np.asarray([0, 3, 4], dtype=np.int64)

    mask = visualize_token_pruning._mask_for_image_tokens(
        selected,
        image_grid_thw=(1, 4, 4),
        spatial_merge_size=2,
    )

    assert mask.tolist() == [[True, False], [False, True]]
