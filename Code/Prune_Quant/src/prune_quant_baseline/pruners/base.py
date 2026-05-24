from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from prune_quant_baseline.core.datatypes import VisualTokenMeta


class VisualTokenPruner(ABC):
    """Base interface for visual token scoring."""

    @abstractmethod
    def score(
        self,
        *,
        attentions: Optional[Any] = None,
        hidden_states: Optional[Any] = None,
        meta: VisualTokenMeta,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return one scalar score per visual token."""

        raise NotImplementedError
