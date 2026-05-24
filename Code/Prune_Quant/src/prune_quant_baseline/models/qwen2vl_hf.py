from typing import Any

from PIL import Image

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.models.base_adapter import MLLMAdapter


def _sample_prompt(sample: dict) -> str:
    prompt = sample.get("prompt") or sample.get("question") or sample.get("text")
    if not prompt:
        raise ValueError("Sample must contain one of: prompt, question, text.")
    return str(prompt)


def _sample_image(sample: dict) -> Image.Image:
    image = sample.get("image")
    if image is None and sample.get("images"):
        image = sample["images"][0]
    if image is None:
        raise ValueError("Image sample must contain 'image' or non-empty 'images'.")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


class Qwen2VLHFAdapter(MLLMAdapter):
    """Hugging Face Qwen2-VL adapter skeleton."""

    def prepare_inputs(self, processor: Any, sample: dict, device: str | None = None) -> dict:
        if "video" in sample:
            raise NotImplementedError("Qwen2-VL video samples are not implemented in stage 1.")
        prompt = _sample_prompt(sample)
        image = _sample_image(sample)
        if hasattr(processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = processor(images=image, text=text, return_tensors="pt")
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
