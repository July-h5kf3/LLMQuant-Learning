from typing import Any, Optional

import torch
import torch.nn.functional as F

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.pruners.base import VisualTokenPruner


class GAEOraclePruner(VisualTokenPruner):
    """Skeleton for gradient-attention explanation oracle pruning."""

    def __init__(self, *, normalize: bool = True) -> None:
        self.normalize = normalize

    def score(
        self,
        *,
        attentions: Optional[Any] = None,
        hidden_states: Optional[Any] = None,
        meta: VisualTokenMeta,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute oracle relevance scores for visual tokens."""

        query_indices = kwargs.get("query_indices")
        if attentions is None:
            raise ValueError("GAEOraclePruner requires differentiable attentions.")
        if len(attentions) == 0:
            raise ValueError("GAEOraclePruner requires at least one attention layer.")
        if query_indices is None:
            raise ValueError("GAEOraclePruner requires query_indices for answer-token positions.")
        if meta.visual_indices.numel() == 0:
            raise ValueError("visual_indices is empty; cannot compute GAE scores.")
        query_indices = query_indices.to(dtype=torch.long)
        scores: torch.Tensor | None = None
        for layer_idx, attn in enumerate(attentions):
            if attn is None:
                continue
            if attn.grad is None:
                raise ValueError(
                    f"Attention layer {layer_idx} has no gradient. Call retain_grad() before backward()."
                )
            if attn.dim() != 4 or attn.shape[0] != 1:
                raise ValueError(f"Expected attention shape [1, H, S, S], got {tuple(attn.shape)}.")
            visual_idx = meta.visual_indices.to(attn.device, dtype=torch.long)
            query_idx = query_indices.to(attn.device, dtype=torch.long)
            relevance = F.relu(attn * attn.grad)
            sub = relevance[0].index_select(dim=1, index=query_idx).index_select(dim=2, index=visual_idx)
            layer_scores = sub.mean(dim=(0, 1))
            scores = layer_scores if scores is None else scores + layer_scores
        if scores is None:
            raise ValueError("No usable attention tensors were provided.")
        if self.normalize:
            denom = scores.sum().clamp_min(torch.finfo(scores.dtype).eps)
            scores = scores / denom
        return scores.detach().to(meta.visual_indices.device)


def compute_answer_logprob_target(
    *,
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    answer_start: int,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """Return summed answer log-prob target and attention query positions."""

    if logits.shape[0] != 1 or input_ids.shape[0] != 1:
        raise ValueError("GAE oracle currently supports B=1.")
    seq_len = input_ids.shape[1]
    if answer_start <= 0 or answer_start >= seq_len:
        raise ValueError(f"answer_start must be in [1, {seq_len - 1}], got {answer_start}.")
    target_ids = input_ids[:, answer_start:]
    pred_logits = logits[:, answer_start - 1 : seq_len - 1, :]
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    selected = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    query_indices = torch.arange(answer_start - 1, seq_len - 1, device=input_ids.device, dtype=torch.long)
    return selected.sum(), query_indices


def generate_gae_labels(*args: Any, **kwargs: Any) -> None:
    """Placeholder for future GAE label generation."""

    raise NotImplementedError("GAE label generation is not implemented in the first-stage baseline.")
