from typing import Any

import torch
from PIL import Image

from prune_quant_baseline.core.datatypes import VisualTokenMeta
from prune_quant_baseline.models.base_adapter import MLLMAdapter


def _as_feature_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, (list, tuple)):
        return torch.cat(features, dim=0)
    return features


def _sample_prompt(sample: dict) -> str:
    prompt = sample.get("prompt") or sample.get("question") or sample.get("text")
    if not prompt:
        raise ValueError("Sample must contain one of: prompt, question, text.")
    return str(prompt)


def _sample_image(sample: dict) -> Image.Image:
    image = sample.get("image") or sample.get("image_path")
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

    def prepare_teacher_forcing_inputs(
        self,
        processor: Any,
        sample: dict,
        answer: str,
        device: str | torch.device | None = None,
    ) -> tuple[dict, int]:
        """Prepare prompt+answer inputs and return the answer start offset."""

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
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = prompt
        prompt_inputs = processor(images=image, text=prompt_text, return_tensors="pt")
        answer_start = int(prompt_inputs["input_ids"].shape[-1])
        combined_inputs = processor(images=image, text=prompt_text + answer, return_tensors="pt")
        if device is not None:
            combined_inputs = {
                key: value.to(device) if hasattr(value, "to") else value for key, value in combined_inputs.items()
            }
        return dict(combined_inputs), answer_start

    def get_visual_token_meta(self, model: Any, inputs: dict) -> VisualTokenMeta:
        if "input_ids" not in inputs:
            raise ValueError("Qwen2-VL visual-token metadata requires input_ids.")
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError(f"Qwen2VLHFAdapter currently supports B=1, got B={input_ids.shape[0]}.")
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("model.config.image_token_id is missing; cannot locate Qwen2-VL visual tokens.")
        ids = input_ids[0]
        visual_indices = torch.nonzero(ids == image_token_id, as_tuple=False).flatten().to(dtype=torch.long)
        if visual_indices.numel() == 0:
            raise ValueError("No Qwen2-VL image tokens found in input_ids.")
        special_ids = {
            getattr(model.config, "image_token_id", None),
            getattr(model.config, "video_token_id", None),
            getattr(model.config, "vision_start_token_id", None),
            getattr(model.config, "vision_end_token_id", None),
        }
        special_ids = {int(token_id) for token_id in special_ids if token_id is not None}
        text_mask = torch.ones_like(ids, dtype=torch.bool)
        text_mask[visual_indices] = False
        for token_id in special_ids:
            text_mask &= ids != token_id
        text_indices = torch.nonzero(text_mask, as_tuple=False).flatten().to(dtype=torch.long)
        return VisualTokenMeta(
            visual_indices=visual_indices,
            text_indices=text_indices,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            rope_deltas=inputs.get("rope_deltas"),
        )

    def build_inputs_embeds(self, model: Any, inputs: dict):
        if "inputs_embeds" in inputs:
            return inputs["inputs_embeds"]
        if "input_ids" not in inputs:
            raise ValueError("inputs must include input_ids or inputs_embeds.")
        model_core = getattr(model, "model", model)
        inputs_embeds = model_core.get_input_embeddings()(inputs["input_ids"])
        if inputs.get("pixel_values") is not None:
            image_features = model_core.get_image_features(inputs["pixel_values"], inputs.get("image_grid_thw"))
            image_features = _as_feature_tensor(image_features).to(inputs_embeds.device, inputs_embeds.dtype)
            if hasattr(model_core, "get_placeholder_mask"):
                image_mask, _ = model_core.get_placeholder_mask(
                    inputs["input_ids"], inputs_embeds=inputs_embeds, image_features=image_features
                )
            else:
                image_token_id = getattr(model.config, "image_token_id", None)
                if image_token_id is None:
                    raise ValueError("model.config.image_token_id is missing; cannot scatter image features.")
                image_mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        if inputs.get("pixel_values_videos") is not None:
            video_features = model_core.get_video_features(inputs["pixel_values_videos"], inputs.get("video_grid_thw"))
            video_features = _as_feature_tensor(video_features).to(inputs_embeds.device, inputs_embeds.dtype)
            if hasattr(model_core, "get_placeholder_mask"):
                _, video_mask = model_core.get_placeholder_mask(
                    inputs["input_ids"], inputs_embeds=inputs_embeds, video_features=video_features
                )
            else:
                video_token_id = getattr(model.config, "video_token_id", None)
                if video_token_id is None:
                    raise ValueError("model.config.video_token_id is missing; cannot scatter video features.")
                video_mask = (inputs["input_ids"] == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)
        return inputs_embeds

    def build_position_ids(self, model: Any, inputs: dict):
        if inputs.get("position_ids") is not None:
            return inputs["position_ids"]
        model_core = getattr(model, "model", model)
        try:
            position_ids, rope_deltas = model_core.get_rope_index(
                input_ids=inputs.get("input_ids"),
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=inputs.get("attention_mask"),
            )
        except TypeError:
            position_ids, rope_deltas = model_core.get_rope_index(
                inputs.get("input_ids"),
                inputs.get("image_grid_thw"),
                inputs.get("video_grid_thw"),
                inputs.get("attention_mask"),
            )
        inputs["rope_deltas"] = rope_deltas
        return position_ids
