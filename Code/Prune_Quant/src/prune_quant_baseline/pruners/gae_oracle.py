from typing import Any, Optional

import torch

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.pruners.base import VisualTokenPruner


class GAEOraclePruner(VisualTokenPruner):
    """Skeleton for gradient-attention explanation oracle pruning."""

    def score(
        self,
        *,
        attentions: Optional[Any] = None,
        hidden_states: Optional[Any] = None,
        meta: VisualTokenMeta,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute oracle relevance scores for visual tokens."""

        raise NotImplementedError(
            "GAEOraclePruner is a stage-2 skeleton. It must run on remote with "
            "teacher-forced answer replay, output_attentions=True, use_cache=False, "
            "and eager attention before attention-gradient rollout is implemented."
        )


def generate_gae_labels(*args: Any, **kwargs: Any) -> None:
    """Placeholder for future GAE label generation."""

    raise NotImplementedError("GAE label generation is not implemented in the first-stage baseline.")
