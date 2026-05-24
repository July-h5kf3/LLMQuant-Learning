import pytest
import torch

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.pruners.attention_proxy import AttentionProxyPruner


def test_attention_proxy_scores_text_to_visual_mean() -> None:
    attn = torch.zeros(1, 2, 5, 5)
    attn[0, 0, 0, 2] = 0.2
    attn[0, 0, 0, 4] = 0.6
    attn[0, 1, 0, 2] = 0.4
    attn[0, 1, 0, 4] = 0.8
    attn[0, 0, 1, 2] = 0.4
    attn[0, 0, 1, 4] = 1.0
    attn[0, 1, 1, 2] = 0.6
    attn[0, 1, 1, 4] = 0.2
    meta = VisualTokenMeta(visual_indices=torch.tensor([2, 4]), text_indices=torch.tensor([0, 1]))

    scores = AttentionProxyPruner().score(attentions=[attn], meta=meta)

    assert torch.allclose(scores, torch.tensor([0.4, 0.65]))


def test_attention_proxy_empty_visual_indices_errors() -> None:
    attn = torch.zeros(1, 2, 5, 5)
    meta = VisualTokenMeta(visual_indices=torch.tensor([], dtype=torch.long), text_indices=torch.tensor([0]))
    with pytest.raises(ValueError, match="visual_indices is empty"):
        AttentionProxyPruner().score(attentions=[attn], meta=meta)


def test_attention_proxy_batch_gt_one_errors() -> None:
    attn = torch.zeros(2, 2, 5, 5)
    meta = VisualTokenMeta(visual_indices=torch.tensor([2]), text_indices=torch.tensor([0]))
    with pytest.raises(ValueError, match="B=1"):
        AttentionProxyPruner().score(attentions=[attn], meta=meta)
