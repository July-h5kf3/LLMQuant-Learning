from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VisualTokenMeta:
    """Sequence-level metadata for visual token pruning."""

    visual_indices: torch.LongTensor
    text_indices: Optional[torch.LongTensor] = None
    keep_indices: Optional[torch.LongTensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    rope_deltas: Optional[torch.Tensor] = None


@dataclass
class PruneResult:
    """Output of sequence pruning."""

    inputs_embeds: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    keep_indices: torch.LongTensor
    kept_visual_indices: torch.LongTensor
    visual_scores: torch.Tensor
