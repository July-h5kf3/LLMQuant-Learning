from __future__ import annotations

import logging
from typing import Any

import torch
from vlmeval.vlm.base import BaseModel
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin

from prune_quant_baseline.models.qwen2vl_hf import Qwen2VLHFAdapter
from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner
from prune_quant_baseline.scripts.run_infer_pruned import (
    _build_pruned_generation_inputs,
    _generate_from_pruned_inputs,
    _generate_vanilla,
    _move_inputs_to_model_device,
    _score_gae_oracle,
)


def _is_omni_model(model_path: str) -> bool:
    return "omni" in model_path.lower()


class Qwen2VLPrunedGAE(Qwen2VLPromptMixin, BaseModel):
    """VLMEvalKit Qwen2-VL wrapper with GAE-guided visual token pruning."""

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int = 16,
        retention_ratio: float = 0.5,
        min_keep: int = 1,
        gae_answer_source: str = "generated",
        gae_per_token: bool = True,
        attn_implementation: str = "eager",
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        **_: Any,
    ) -> None:
        if _is_omni_model(model_path):
            raise NotImplementedError("Qwen2VLPrunedGAE supports Qwen2-VL image models only.")
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        from vlmeval.vlm.qwen2_vl.model import ensure_image_url

        super().__init__()
        self.ensure_image_url = ensure_image_url
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.retention_ratio = float(retention_ratio)
        self.min_keep = int(min_keep)
        self.gae_answer_source = gae_answer_source
        self.gae_per_token = gae_per_token
        self.use_custom_prompt_flag = use_custom_prompt
        self.system_prompt = system_prompt
        self.verbose = verbose

        self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self._pq_adapter = Qwen2VLHFAdapter()
        self._pq_pruner = GAEOraclePruner()

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
                raise NotImplementedError(f"Unsupported VLMEvalKit message item for pruning: {item['type']}")
        return content

    def _build_inputs(self, message: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:
            logging.critical("qwen_vl_utils is required for Qwen2VLPrunedGAE.")
            raise exc

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(message)})
        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        if videos:
            raise NotImplementedError("Qwen2VLPrunedGAE currently supports image-only inputs.")
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        plain_prompt = "\n".join(item["value"] for item in message if item["type"] == "text")
        sample = {
            "image": images[0] if images else None,
            "prompt": plain_prompt,
        }
        return sample, dict(inputs)

    def generate_inner(self, message: list[dict[str, str]], dataset: str | None = None) -> str:
        del dataset
        sample, inputs = self._build_inputs(message)
        inputs = _move_inputs_to_model_device(self.model, inputs)

        with torch.no_grad():
            answer = _generate_vanilla(self.model, self.processor, inputs, self.max_new_tokens)
        if self.gae_answer_source == "empty":
            answer = ""
        sample["answer"] = answer or "Yes"

        with torch.enable_grad():
            scores = _score_gae_oracle(
                model=self.model,
                processor=self.processor,
                adapter=self._pq_adapter,
                pruner=self._pq_pruner,
                sample=sample,
                answer=sample["answer"],
                per_token=self.gae_per_token,
            )
        pruned_inputs, _, _ = _build_pruned_generation_inputs(
            model=self.model,
            adapter=self._pq_adapter,
            inputs=inputs,
            scores=scores,
            retention_ratio=self.retention_ratio,
            min_keep=self.min_keep,
        )
        response = _generate_from_pruned_inputs(
            model=self.model,
            processor=self.processor,
            pruned_inputs=pruned_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        torch.cuda.empty_cache()
        return response
