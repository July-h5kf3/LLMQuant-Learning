from typing import Any, Optional

import torch

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.pruners.base import VisualTokenPruner


class AttentionProxyPruner(VisualTokenPruner):
    """Score visual tokens with first-layer text-to-visual attention."""

    def score(
        self,
        *,
        attentions: Optional[Any] = None,
        hidden_states: Optional[Any] = None,
        meta: VisualTokenMeta,
        **kwargs: Any,
    ) -> torch.Tensor:
        if attentions is None:
            raise ValueError("AttentionProxyPruner requires attentions, got None.")
        if len(attentions) == 0:
            raise ValueError("AttentionProxyPruner requires at least one attention layer.")
        if meta.visual_indices.numel() == 0:
            raise ValueError("visual_indices is empty; cannot score visual tokens.")
        if meta.text_indices is None or meta.text_indices.numel() == 0:
            raise ValueError("text_indices must be provided and non-empty for attention-proxy scoring.")

        attn = attentions[0]
        if not isinstance(attn, torch.Tensor):
            raise TypeError(f"Expected first attention layer to be torch.Tensor, got {type(attn)!r}.")
        if attn.dim() != 4:
            raise ValueError(f"Expected attention shape [B, H, S, S], got {tuple(attn.shape)}.")
        if attn.shape[0] != 1:
            raise ValueError(f"AttentionProxyPruner currently supports B=1, got B={attn.shape[0]}.")

        query_idx = meta.text_indices.to(device=attn.device, dtype=torch.long)
        visual_idx = meta.visual_indices.to(device=attn.device, dtype=torch.long)
        seq_len = attn.shape[-1]
        if query_idx.min() < 0 or query_idx.max() >= seq_len:
            raise ValueError("text_indices contain positions outside the attention sequence length.")
        if visual_idx.min() < 0 or visual_idx.max() >= seq_len:
            raise ValueError("visual_indices contain positions outside the attention sequence length.")

        sub = attn[0].index_select(dim=1, index=query_idx).index_select(dim=2, index=visual_idx)
        return sub.mean(dim=(0, 1))
