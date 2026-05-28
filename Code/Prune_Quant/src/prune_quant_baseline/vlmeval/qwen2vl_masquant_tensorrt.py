from __future__ import annotations

from typing import Any

from vlmeval.vlm.base import BaseModel
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin

from prune_quant_baseline.quant.tensorrt import (
    load_masquant_tensorrt_artifact,
    load_tensorrt_runtime,
)
from .qwen2vl_pruned_gae import (
    _bool_value,
    _is_omni_model,
    _resolve_pixels,
)


class Qwen2VLMASQuantTensorRT(Qwen2VLPromptMixin, BaseModel):
    """VLMEvalKit Qwen2-VL wrapper for saved MASQuant TensorRT artifacts."""

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        artifact_dir: str,
        model_path: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        min_visual_tokens: int | None = None,
        max_visual_tokens: int | None = None,
        max_new_tokens: int = 16,
        runtime_class: str | None = None,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        **runtime_kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor
        from vlmeval.vlm.qwen2_vl.model import ensure_image_url

        super().__init__()
        self.artifact = load_masquant_tensorrt_artifact(artifact_dir)
        self.model_path = model_path or str(self.artifact.manifest["model_path"])
        if _is_omni_model(self.model_path):
            raise NotImplementedError("Qwen2VLMASQuantTensorRT supports Qwen2-VL/Qwen2.5-VL image models only.")
        self.ensure_image_url = ensure_image_url
        self.min_pixels = _resolve_pixels(pixels=min_pixels, visual_tokens=min_visual_tokens, name="min")
        self.max_pixels = _resolve_pixels(pixels=max_pixels, visual_tokens=max_visual_tokens, name="max")
        self.max_new_tokens = int(max_new_tokens)
        self.use_custom_prompt_flag = _bool_value(use_custom_prompt)
        self.system_prompt = system_prompt

        processor_path = self.artifact.processor_dir or self.model_path
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)
        self.runtime = load_tensorrt_runtime(
            self.artifact,
            runtime_class=runtime_class,
            runtime_kwargs=runtime_kwargs,
        )

    def use_custom_prompt(self, dataset: str) -> bool:
        return self.use_custom_prompt_flag

    def build_prompt(self, line: Any, dataset: str | None = None):
        return super().build_prompt(line, dataset)

    def generate(self, message: Any, dataset: str | None = None) -> str:
        return super().generate(message, dataset)

    def _prepare_content(self, inputs: list[dict[str, str]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for item in inputs:
            if item["type"] == "image":
                image_item: dict[str, Any] = {"type": "image", "image": self.ensure_image_url(item["value"])}
                if self.min_pixels is not None:
                    image_item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    image_item["max_pixels"] = self.max_pixels
                content.append(image_item)
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            else:
                raise NotImplementedError(f"Unsupported VLMEvalKit message item for TensorRT inference: {item['type']}")
        return content

    def _build_inputs(self, message: list[dict[str, str]]) -> dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:
            import logging

            logging.critical("qwen_vl_utils is required for Qwen2VLMASQuantTensorRT.")
            raise exc

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(message)})
        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        if videos:
            raise NotImplementedError("Qwen2VLMASQuantTensorRT currently supports image-only inputs.")
        return dict(self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt"))

    def generate_inner(self, message: list[dict[str, str]], dataset: str | None = None) -> str:
        del dataset
        inputs = self._build_inputs(message)
        return self.runtime.generate(
            inputs=inputs,
            processor=self.processor,
            max_new_tokens=self.max_new_tokens,
        )
