import pytest
import torch

from prune_quant_baseline.pruners.token_gather import (
    build_keep_indices,
    gather_sequence_tensors,
    select_topk_visual_tokens,
)
from prune_quant_baseline.scripts.run_infer_pruned import _build_pruned_generation_inputs


def test_topk_returns_original_sequence_order() -> None:
    visual_indices = torch.tensor([2, 4, 6, 8])
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    kept = select_topk_visual_tokens(visual_indices, scores, retention_ratio=0.5)
    assert kept.tolist() == [4, 8]


def test_retention_ratio_bounds() -> None:
    visual_indices = torch.tensor([1, 3])
    scores = torch.tensor([0.1, 0.2])
    with pytest.raises(ValueError, match="retention_ratio"):
        select_topk_visual_tokens(visual_indices, scores, retention_ratio=0.0)
    kept = select_topk_visual_tokens(visual_indices, scores, retention_ratio=1.0)
    assert kept.tolist() == [1, 3]


def test_build_keep_indices_keeps_non_visual_and_selected_visual() -> None:
    keep = build_keep_indices(
        seq_len=7,
        visual_indices=torch.tensor([1, 2, 5]),
        kept_visual_indices=torch.tensor([2]),
    )
    assert keep.tolist() == [0, 2, 3, 4, 6]


def test_gather_sequence_tensors_shapes() -> None:
    inputs_embeds = torch.arange(1 * 5 * 2).view(1, 5, 2)
    attention_mask = torch.ones(1, 5)
    position_ids = torch.arange(5).view(1, 5)
    keep_indices = torch.tensor([0, 2, 4])

    embeds, mask, pos = gather_sequence_tensors(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        keep_indices=keep_indices,
    )

    assert embeds.shape == (1, 3, 2)
    assert mask.shape == (1, 3)
    assert pos.tolist() == [[0, 2, 4]]


def test_gather_position_ids_b_3_s() -> None:
    inputs_embeds = torch.zeros(1, 5, 2)
    position_ids = torch.arange(15).view(1, 3, 5)
    _, _, pos = gather_sequence_tensors(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        keep_indices=torch.tensor([1, 3]),
    )
    assert pos.shape == (1, 3, 2)
    assert pos.tolist() == [[[1, 3], [6, 8], [11, 13]]]


class _DummyAdapter:
    def get_visual_token_meta(self, model, inputs):
        from prune_quant_baseline.core.datatypes import VisualTokenMeta

        return VisualTokenMeta(visual_indices=torch.tensor([1, 3]))

    def build_inputs_embeds(self, model, inputs):  # pragma: no cover - should not be called.
        raise AssertionError("100% retention should bypass pruning tensor rebuild.")

    def build_position_ids(self, model, inputs):  # pragma: no cover - should not be called.
        raise AssertionError("100% retention should bypass pruning tensor rebuild.")


def test_build_pruned_generation_inputs_bypasses_full_retention() -> None:
    inputs = {
        "input_ids": torch.tensor([[10, 20, 30, 40]]),
        "attention_mask": torch.ones(1, 4),
    }

    gen_inputs, before, after = _build_pruned_generation_inputs(
        model=object(),
        adapter=_DummyAdapter(),
        inputs=inputs,
        scores=torch.tensor([0.1, 0.2]),
        retention_ratio=1.0,
        min_keep=1,
    )

    assert gen_inputs is inputs
    assert before == 2
    assert after == 2
