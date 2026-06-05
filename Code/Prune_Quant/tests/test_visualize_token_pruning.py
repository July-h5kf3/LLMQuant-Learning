from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "vis" / "visualize_token_pruning.py"
SPEC = spec_from_file_location("visualize_token_pruning", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
visualize_token_pruning = module_from_spec(SPEC)
SPEC.loader.exec_module(visualize_token_pruning)


def test_select_top_percent_uses_ceil_and_stable_descending_order() -> None:
    values = np.asarray([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], dtype=np.float32)

    selected = visualize_token_pruning._select_top_percent(values, np.arange(values.size), fraction=0.2)

    assert selected.tolist() == [1, 3]


def test_mask_for_image_tokens_projects_local_indices_to_first_frame() -> None:
    selected = np.asarray([0, 3, 4], dtype=np.int64)

    mask = visualize_token_pruning._mask_for_image_tokens(
        selected,
        image_grid_thw=(1, 4, 4),
        spatial_merge_size=2,
    )

    assert mask.tolist() == [[True, False], [False, True]]
