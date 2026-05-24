import pytest
import torch

from prune_quant_baseline.pruners.token_gather import (
    build_keep_indices,
    gather_sequence_tensors,
    select_topk_visual_tokens,
)


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
