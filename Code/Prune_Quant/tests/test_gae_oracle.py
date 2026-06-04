import torch

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner, QuantJointGAEPruner


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


def test_quant_joint_gae_uses_attention_delta_and_drop_score() -> None:
    attn_data = torch.zeros(1, 1, 3, 3)
    attn_data[0, 0, 2, 0] = 1.0
    attn_data[0, 0, 2, 1] = 3.0
    quant_attn = attn_data.clone()
    quant_attn[0, 0, 2, 0] = 0.5
    quant_attn[0, 0, 2, 1] = 2.0
    attn = _attention_with_grad(attn_data, torch.ones_like(attn_data))
    meta = VisualTokenMeta(visual_indices=torch.tensor([0, 1]), text_indices=torch.tensor([2]))

    components = QuantJointGAEPruner(quant_lambda=2.0, normalize=False).score_components(
        attentions=[attn],
        quantized_attentions=[quant_attn],
        meta=meta,
        query_indices=torch.tensor([2]),
    )

    assert torch.allclose(components["c_drop"], torch.tensor([1.0, 3.0]))
    assert torch.allclose(components["c_quant"], torch.tensor([0.5, 1.0]))
    assert torch.allclose(components["joint"], torch.tensor([0.0, -1.0]))


def test_quant_joint_gae_score_matches_joint_component() -> None:
    attn_data = torch.zeros(1, 1, 3, 3)
    attn_data[0, 0, 2, 0] = 1.0
    attn_data[0, 0, 2, 1] = 3.0
    quant_attn = attn_data.clone()
    quant_attn[0, 0, 2, 0] = 0.5
    quant_attn[0, 0, 2, 1] = 2.0
    meta = VisualTokenMeta(visual_indices=torch.tensor([0, 1]), text_indices=torch.tensor([2]))
    pruner = QuantJointGAEPruner(quant_lambda=2.0, normalize=False)

    components = pruner.score_components(
        attentions=[_attention_with_grad(attn_data, torch.ones_like(attn_data))],
        quantized_attentions=[quant_attn],
        meta=meta,
        query_indices=torch.tensor([2]),
    )
    scores = pruner.score(
        attentions=[_attention_with_grad(attn_data, torch.ones_like(attn_data))],
        quantized_attentions=[quant_attn],
        meta=meta,
        query_indices=torch.tensor([2]),
    )

    assert set(components) == {"c_drop", "c_quant", "joint"}
    assert torch.allclose(scores, components["joint"])
