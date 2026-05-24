from abc import ABC, abstractmethod
from typing import Any

from prune_quant_baseline.core.datatypes import VisualTokenMeta


class MLLMAdapter(ABC):
    """Adapter for model-specific multimodal preprocessing and token metadata."""

    @abstractmethod
    def prepare_inputs(self, processor: Any, sample: dict, device: str | None = None) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_visual_token_meta(self, model: Any, inputs: dict) -> VisualTokenMeta:
        raise NotImplementedError

    @abstractmethod
    def build_inputs_embeds(self, model: Any, inputs: dict):
        raise NotImplementedError

    def build_position_ids(self, model: Any, inputs: dict):
        """Optionally build model-specific position ids before pruning."""

        return inputs.get("position_ids")
