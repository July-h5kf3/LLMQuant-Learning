from typing import Any

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.models.base_adapter import MLLMAdapter


class Qwen2VLHFAdapter(MLLMAdapter):
    """Hugging Face Qwen2-VL adapter skeleton."""

    def prepare_inputs(self, processor: Any, sample: dict, device: str | None = None) -> dict:
        if "video" in sample:
            raise NotImplementedError("Qwen2-VL video samples are not implemented in stage 1.")
        if "image" not in sample or "prompt" not in sample:
            raise ValueError("Qwen2-VL image samples must contain 'image' and 'prompt'.")
        inputs = processor(images=sample["image"], text=sample["prompt"], return_tensors="pt")
        if device is not None:
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        return dict(inputs)

    def get_visual_token_meta(self, model: Any, inputs: dict) -> VisualTokenMeta:
        raise NotImplementedError(
            "Reliable Qwen2-VL visual-token position extraction must be implemented against the remote HF model. "
            "Preserve image_grid_thw, video_grid_thw, rope_deltas, and gather existing position_ids after pruning."
        )

    def build_inputs_embeds(self, model: Any, inputs: dict):
        if "inputs_embeds" in inputs:
            return inputs["inputs_embeds"]
        if "input_ids" not in inputs:
            raise ValueError("inputs must include input_ids or inputs_embeds.")
        return model.get_input_embeddings()(inputs["input_ids"])
