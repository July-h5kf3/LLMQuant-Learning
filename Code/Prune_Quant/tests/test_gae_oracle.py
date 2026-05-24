import torch

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner


def _attention_with_grad(data: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    attn = data.clone().requires_grad_(True)
    attn.retain_grad()
    attn.grad = grad
    return attn


def test_gae_oracle_uses_rollout_across_layers() -> None:
    layer0 = torch.zeros(1, 1, 3, 3)
    layer0[0, 0, 1, 0] = 0.5
    layer1 = torch.zeros(1, 1, 3, 3)
    layer1[0, 0, 2, 1] = 0.5
    grad = torch.ones_like(layer0)
    attentions = [
        _attention_with_grad(layer0, grad),
        _attention_with_grad(layer1, grad),
    ]
    meta = VisualTokenMeta(visual_indices=torch.tensor([0, 1]), text_indices=torch.tensor([2]))

    scores = GAEOraclePruner(normalize=False).score(
        attentions=attentions,
        meta=meta,
        query_indices=torch.tensor([2]),
    )

    assert torch.allclose(scores, torch.tensor([0.25, 0.5]))


def test_gae_oracle_normalizes_scores() -> None:
    attn_data = torch.zeros(1, 1, 3, 3)
    attn_data[0, 0, 2, 0] = 1.0
    attn_data[0, 0, 2, 1] = 3.0
    attn = _attention_with_grad(attn_data, torch.ones_like(attn_data))
    meta = VisualTokenMeta(visual_indices=torch.tensor([0, 1]), text_indices=torch.tensor([2]))

    scores = GAEOraclePruner().score(
        attentions=[attn],
        meta=meta,
        query_indices=torch.tensor([2]),
    )

    assert torch.allclose(scores, torch.tensor([0.25, 0.75]))
