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
        """Compute GAE rollout relevance scores for visual tokens."""

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
        rollout: torch.Tensor | None = None
        for layer_idx, attn in enumerate(attentions):
            if attn is None:
                continue
            if attn.grad is None:
                raise ValueError(
                    f"Attention layer {layer_idx} has no gradient. Call retain_grad() before backward()."
                )
            if attn.dim() != 4 or attn.shape[0] != 1:
                raise ValueError(f"Expected attention shape [1, H, S, S], got {tuple(attn.shape)}.")
            relevance = F.relu(attn * attn.grad).mean(dim=1)[0]
            if rollout is None:
                seq_len = relevance.shape[-1]
                rollout = torch.eye(seq_len, device=attn.device, dtype=relevance.dtype)
            rollout = rollout + relevance @ rollout
        if rollout is None:
            raise ValueError("No usable attention tensors were provided.")
        visual_idx = meta.visual_indices.to(rollout.device, dtype=torch.long)
        query_idx = query_indices.to(rollout.device, dtype=torch.long)
        if query_idx.min() < 0 or query_idx.max() >= rollout.shape[0]:
            raise ValueError("query_indices contain positions outside the attention sequence length.")
        if visual_idx.min() < 0 or visual_idx.max() >= rollout.shape[1]:
            raise ValueError("visual_indices contain positions outside the attention sequence length.")
        scores = rollout.index_select(dim=0, index=query_idx).index_select(dim=1, index=visual_idx).mean(dim=0)
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
