from typing import Any

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.models.base_adapter import MLLMAdapter


class LlavaOneVisionHFAdapter(MLLMAdapter):
    """Hugging Face LLaVA-OneVision adapter skeleton."""

    def prepare_inputs(self, processor: Any, sample: dict, device: str | None = None) -> dict:
        if "video" in sample:
            raise NotImplementedError("LLaVA-OneVision video samples are not implemented in stage 1.")
        if "image" not in sample or "prompt" not in sample:
            raise ValueError("LLaVA-OneVision image samples must contain 'image' and 'prompt'.")
        inputs = processor(images=sample["image"], text=sample["prompt"], return_tensors="pt")
        if device is not None:
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        return dict(inputs)

    def get_visual_token_meta(self, model: Any, inputs: dict) -> VisualTokenMeta:
        raise NotImplementedError(
            "Reliable LLaVA-OneVision visual-token position extraction is model/version specific. "
            "Implement this using image special tokens and expanded image feature positions on the remote model."
        )

    def build_inputs_embeds(self, model: Any, inputs: dict):
        if "inputs_embeds" in inputs:
            return inputs["inputs_embeds"]
        if "input_ids" not in inputs:
            raise ValueError("inputs must include input_ids or inputs_embeds.")
        return model.get_input_embeddings()(inputs["input_ids"])
