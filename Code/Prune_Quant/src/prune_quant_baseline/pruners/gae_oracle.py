from typing import Any, Optional

import torch
import torch.nn.functional as F

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.pruners.base import VisualTokenPruner


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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
        normalize = bool(kwargs.get("normalize", self.normalize))
        source_attentions = kwargs.get("source_attentions")
        return self._rollout_score(
            attentions=attentions,
            source_attentions=source_attentions,
            meta=meta,
            query_indices=query_indices,
            normalize=normalize,
        )

    def _rollout_score(
        self,
        *,
        attentions: Optional[Any],
        source_attentions: Optional[Any],
        meta: VisualTokenMeta,
        query_indices: Any,
        normalize: bool,
    ) -> torch.Tensor:
        if attentions is None:
            raise ValueError("GAEOraclePruner requires differentiable attentions.")
        if len(attentions) == 0:
            raise ValueError("GAEOraclePruner requires at least one attention layer.")
        if query_indices is None:
            raise ValueError("GAEOraclePruner requires query_indices for answer-token positions.")
        if source_attentions is not None and len(source_attentions) != len(attentions):
            raise ValueError("source_attentions must have the same number of layers as attentions.")
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
            source = attn if source_attentions is None else source_attentions[layer_idx]
            if source is None:
                continue
            if source.shape != attn.shape:
                raise ValueError(
                    f"Attention source layer {layer_idx} shape {tuple(source.shape)} "
                    f"does not match attention shape {tuple(attn.shape)}."
                )
            relevance = (attn.grad.detach() * source.detach().to(attn.device, dtype=attn.dtype)).relu().mean(dim=1)[0]
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
        if normalize:
            denom = scores.sum().clamp_min(torch.finfo(scores.dtype).eps)
            scores = scores / denom
        return scores.detach().to(meta.visual_indices.device)


class QuantJointGAEPruner(GAEOraclePruner):
    """GAE variant that drops tokens with high quantization-sensitive drop scores."""

    def __init__(self, *, quant_lambda: float = 1.0, normalize: bool = True) -> None:
        super().__init__(normalize=normalize)
        self.quant_lambda = float(quant_lambda)

    def score_components(
        self,
        *,
        attentions: Optional[Any] = None,
        quantized_attentions: Optional[Any] = None,
        visual_activations: Optional[torch.Tensor] = None,
        meta: VisualTokenMeta,
        query_indices: Any,
        normalize: bool | None = None,
        quant_bits: int = 8,
        quant_symmetric: bool = True,
        eps: float = 1e-12,
    ) -> dict[str, torch.Tensor]:
        """Return raw C_drop/C_quant and normalized D = lambda * C_quant - C_drop."""

        if quantized_attentions is None:
            raise ValueError("QuantJointGAEPruner requires quantized_attentions from an RTN scoring forward.")
        if visual_activations is None:
            raise ValueError("QuantJointGAEPruner requires visual_activations for quant difficulty scoring.")
        normalize_joint = self.normalize if normalize is None else bool(normalize)
        if attentions is None:
            raise ValueError("QuantJointGAEPruner requires differentiable attentions.")
        if len(attentions) != len(quantized_attentions):
            raise ValueError("quantized_attentions must have the same number of layers as attentions.")

        c_drop = self._rollout_score(
            attentions=attentions,
            source_attentions=quantized_attentions,
            meta=meta,
            query_indices=query_indices,
            normalize=False,
        )
        c_quant = self.compute_quant_difficulty(
            visual_activations,
            quant_bits=quant_bits,
            quant_symmetric=quant_symmetric,
            eps=eps,
        )
        if c_quant.numel() != c_drop.numel():
            raise ValueError(
                f"Quant difficulty count {c_quant.numel()} does not match visual score count {c_drop.numel()}."
            )
        c_drop = normalize_relevance_scores(c_drop)
        c_quant = normalize_relevance_scores(c_quant)
        joint = normalize_joint_scores(self.quant_lambda * c_quant - c_drop) if normalize_joint else (
            self.quant_lambda * c_quant - c_drop
        )
        return {
            "c_drop": c_drop,
            "c_quant": c_quant,
            "joint": joint.detach().to(meta.visual_indices.device),
        }

    @staticmethod
    def compute_quant_difficulty(
        visual_activations: torch.Tensor,
        *,
        quant_bits: int = 8,
        quant_symmetric: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if quant_bits < 2:
            raise ValueError("quant_bits must be >= 2.")
        if visual_activations.dim() != 2:
            raise ValueError(
                f"visual_activations must have shape [num_visual_tokens, hidden_dim], "
                f"got {tuple(visual_activations.shape)}."
            )
        compute = visual_activations.detach().float()
        hidden_dim = compute.shape[-1]
        if quant_symmetric:
            qmax = (1 << (quant_bits - 1)) - 1
            delta = compute.abs().amax(dim=-1) / qmax
        else:
            qlevels = (1 << quant_bits) - 1
            delta = (compute.amax(dim=-1) - compute.amin(dim=-1)) / qlevels
        norm_sq = compute.pow(2).sum(dim=-1)
        return (hidden_dim * delta.pow(2) / (12.0 * (norm_sq + eps))).to(device=visual_activations.device)

    def score(
        self,
        *,
        attentions: Optional[Any] = None,
        hidden_states: Optional[Any] = None,
        meta: VisualTokenMeta,
        **kwargs: Any,
    ) -> torch.Tensor:
        del hidden_states
        quant_lambda = float(kwargs.get("quant_lambda", self.quant_lambda))
        old_lambda = self.quant_lambda
        self.quant_lambda = quant_lambda
        try:
            components = self.score_components(
                attentions=attentions,
                quantized_attentions=kwargs.get("quantized_attentions"),
                visual_activations=kwargs.get("visual_activations"),
                meta=meta,
                query_indices=kwargs.get("query_indices"),
                normalize=kwargs.get("normalize", self.normalize),
                quant_bits=int(kwargs.get("quant_bits", 8)),
                quant_symmetric=_coerce_bool(kwargs.get("quant_symmetric", True)),
                eps=float(kwargs.get("eps", 1e-12)),
            )
        finally:
            self.quant_lambda = old_lambda
        return components["joint"]


def normalize_relevance_scores(scores: torch.Tensor) -> torch.Tensor:
    denom = scores.sum().clamp_min(torch.finfo(scores.dtype).eps)
    return scores / denom


def normalize_joint_scores(scores: torch.Tensor) -> torch.Tensor:
    denom = scores.abs().sum().clamp_min(torch.finfo(scores.dtype).eps)
    return scores / denom


def compute_answer_token_logprobs(
    *,
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    answer_start: int,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """Return per-answer-token log-probs and their prediction query positions."""

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
    return selected.squeeze(0), query_indices


def compute_answer_logprob_target(
    *,
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    answer_start: int,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """Return summed answer log-prob target and attention query positions."""

    selected, query_indices = compute_answer_token_logprobs(
        logits=logits,
        input_ids=input_ids,
        answer_start=answer_start,
    )
    return selected.sum(), query_indices


def generate_gae_labels(*args: Any, **kwargs: Any) -> None:
    """Placeholder for future GAE label generation."""

    raise NotImplementedError("GAE label generation is not implemented in the first-stage baseline.")
