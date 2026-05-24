import math

import torch


def select_topk_visual_tokens(
    visual_indices: torch.LongTensor,
    scores: torch.Tensor,
    retention_ratio: float,
    min_keep: int = 1,
) -> torch.LongTensor:
    """Select top-K visual token global indices by score, returned in sequence order."""

    if not (0 < retention_ratio <= 1):
        raise ValueError("retention_ratio must be in the range (0, 1].")
    if min_keep < 0:
        raise ValueError("min_keep must be non-negative.")
    if visual_indices.dim() != 1:
        raise ValueError(f"visual_indices must be 1D, got shape {tuple(visual_indices.shape)}.")
    if scores.dim() != 1:
        raise ValueError(f"scores must be 1D, got shape {tuple(scores.shape)}.")
    if visual_indices.numel() == 0:
        raise ValueError("visual_indices is empty; cannot select visual tokens.")
    if scores.numel() != visual_indices.numel():
        raise ValueError(
            f"scores length ({scores.numel()}) must match visual_indices length ({visual_indices.numel()})."
        )

    num_visual = visual_indices.numel()
    k = max(min_keep, math.ceil(num_visual * retention_ratio))
    k = min(k, num_visual)
    if k == 0:
        return visual_indices.new_empty((0,), dtype=torch.long)

    topk_local = torch.topk(scores, k=k, largest=True, sorted=False).indices
    kept = visual_indices.to(device=scores.device).index_select(dim=0, index=topk_local)
    return torch.sort(kept).values.to(device=visual_indices.device, dtype=torch.long)


def build_keep_indices(
    seq_len: int,
    visual_indices: torch.LongTensor,
    kept_visual_indices: torch.LongTensor,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return sequence indices keeping all non-visual tokens and selected visual tokens."""

    if seq_len < 0:
        raise ValueError("seq_len must be non-negative.")
    out_device = device or visual_indices.device
    all_indices = torch.arange(seq_len, device=out_device, dtype=torch.long)
    keep_mask = torch.ones(seq_len, device=out_device, dtype=torch.bool)

    visual_indices = visual_indices.to(device=out_device, dtype=torch.long)
    kept_visual_indices = kept_visual_indices.to(device=out_device, dtype=torch.long)
    if visual_indices.numel() > 0:
        if visual_indices.min() < 0 or visual_indices.max() >= seq_len:
            raise ValueError("visual_indices contain positions outside seq_len.")
        keep_mask[visual_indices] = False
    if kept_visual_indices.numel() > 0:
        if kept_visual_indices.min() < 0 or kept_visual_indices.max() >= seq_len:
            raise ValueError("kept_visual_indices contain positions outside seq_len.")
        keep_mask[kept_visual_indices] = True
    return all_indices[keep_mask]


def gather_sequence_tensors(
    *,
    inputs_embeds: torch.Tensor,
    keep_indices: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Physically gather sequence tensors according to keep_indices."""

    if inputs_embeds.dim() != 3:
        raise ValueError(f"inputs_embeds must have shape [B, S, D], got {tuple(inputs_embeds.shape)}.")
    seq_len = inputs_embeds.shape[1]
    keep_indices = keep_indices.to(device=inputs_embeds.device, dtype=torch.long)
    if keep_indices.dim() != 1:
        raise ValueError(f"keep_indices must be 1D, got shape {tuple(keep_indices.shape)}.")
    if keep_indices.numel() > 0 and (keep_indices.min() < 0 or keep_indices.max() >= seq_len):
        raise ValueError("keep_indices contain positions outside inputs_embeds sequence length.")

    gathered_embeds = inputs_embeds.index_select(dim=1, index=keep_indices)

    gathered_mask = None
    if attention_mask is not None:
        if attention_mask.dim() < 2:
            raise ValueError(f"attention_mask must include a sequence dimension, got {tuple(attention_mask.shape)}.")
        if attention_mask.shape[-1] != seq_len:
            raise ValueError("attention_mask last dimension must match inputs_embeds sequence length.")
        gathered_mask = attention_mask.index_select(dim=-1, index=keep_indices.to(attention_mask.device))

    gathered_pos = None
    if position_ids is not None:
        pos_indices = keep_indices.to(position_ids.device)
        if position_ids.dim() == 2 and position_ids.shape[1] == seq_len:
            gathered_pos = position_ids.index_select(dim=1, index=pos_indices)
        elif position_ids.dim() == 3 and position_ids.shape[2] == seq_len:
            gathered_pos = position_ids.index_select(dim=2, index=pos_indices)
        else:
            raise ValueError(
                "Unsupported position_ids shape. Expected [B, S], [B, 3, S], or [3, B, S] "
                f"with sequence length {seq_len}, got {tuple(position_ids.shape)}."
            )

    return gathered_embeds, gathered_mask, gathered_pos
