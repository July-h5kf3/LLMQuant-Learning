from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from vlmeval.vlm.base import BaseModel
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin

from prune_quant_baseline.models.qwen2vl_hf import Qwen2VLHFAdapter
from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner
from prune_quant_baseline.quant.loaders import load_model_and_processor
from prune_quant_baseline.scripts.run_infer_pruned import (
    _build_pruned_generation_inputs,
    _generate_from_pruned_inputs,
    _generate_vanilla,
    _move_inputs_to_model_device,
    _score_gae_oracle,
    _visual_tokens_to_pixels,
)


def _is_omni_model(model_path: str) -> bool:
    return "omni" in model_path.lower()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _resolve_pixels(*, pixels: Any, visual_tokens: Any, name: str) -> int | None:
    pixels = _optional_int(pixels)
    visual_tokens = _optional_int(visual_tokens)
    if pixels is not None and visual_tokens is not None:
        raise ValueError(f"Use either {name}_pixels or {name}_visual_tokens, not both.")
    if pixels is not None:
        return pixels
    return _visual_tokens_to_pixels(visual_tokens)


@contextmanager
def _disable_masquant_fake_quant_for_scoring(model: Any):
    patched: list[tuple[Any, Any]] = []

    def plain_linear_forward(self: Any, input: torch.Tensor, multi_modal_mask: Any = None) -> torch.Tensor:
        del multi_modal_mask
        bias = getattr(self, "bias", None)
        if not torch.is_tensor(bias):
            bias = None
        return F.linear(input, self.weight.to(dtype=input.dtype), bias.to(dtype=input.dtype) if bias is not None else None)

    try:
        for module in model.modules():
            if not hasattr(module, "weight") or not hasattr(module, "forward_mas_infer"):
                continue
            patched.append((module, module.forward))
            module.forward = MethodType(plain_linear_forward, module)
        yield
    finally:
        for module, forward in patched:
            module.forward = forward


class Qwen2VLPrunedGAE(Qwen2VLPromptMixin, BaseModel):
    """VLMEvalKit Qwen2-VL wrapper with GAE-guided visual token pruning."""

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        model_type: str = "qwen2vl",
        quant_method: str = "none",
        dtype: str = "auto",
        device_map: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        min_visual_tokens: int | None = None,
        max_visual_tokens: int | None = None,
        max_new_tokens: int = 16,
        retention_ratio: float = 0.5,
        min_keep: int = 1,
        gae_answer_source: str = "generated",
        gae_per_token: bool = False,
        attn_implementation: str = "eager",
        processor_use_fast: bool | None = None,
        masquant_root: str | None = None,
        masquant_resume: str | None = None,
        masquant_act_scales: str | None = None,
        masquant_wbits: int = 4,
        masquant_abits: int = 8,
        masquant_group_size: int = 0,
        masquant_inference_mode: str = "split_scales",
        masquant_symmetric: bool = True,
        masquant_batch_size: int = 1,
        masquant_cmc_low_rank_adapters: str | None = None,
        masquant_cmc_white_matrix: str | None = None,
        masquant_cmc_rank: float = 0.2,
        masquant_cmc_quant_cmc: int = 0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        gae_score_disable_masquant_fake_quant: bool = False,
        allow_vanilla_fallback: bool = False,
        verbose: bool = False,
        **_: Any,
    ) -> None:
        if _is_omni_model(model_path):
            raise NotImplementedError("Qwen2VLPrunedGAE supports Qwen2-VL/Qwen2.5-VL image models only.")
        from vlmeval.vlm.qwen2_vl.model import ensure_image_url

        super().__init__()
        self.ensure_image_url = ensure_image_url
        self.model_path = model_path
        self.model_type = model_type
        self.quant_method = quant_method
        self.min_pixels = _resolve_pixels(pixels=min_pixels, visual_tokens=min_visual_tokens, name="min")
        self.max_pixels = _resolve_pixels(pixels=max_pixels, visual_tokens=max_visual_tokens, name="max")
        self.max_new_tokens = int(max_new_tokens)
        self.retention_ratio = float(retention_ratio)
        self.min_keep = int(min_keep)
        self.gae_answer_source = gae_answer_source
        self.gae_per_token = _bool_value(gae_per_token)
        self.use_custom_prompt_flag = _bool_value(use_custom_prompt)
        self.system_prompt = system_prompt
        self.gae_score_disable_masquant_fake_quant = _bool_value(gae_score_disable_masquant_fake_quant)
        self.allow_vanilla_fallback = _bool_value(allow_vanilla_fallback)
        self.verbose = _bool_value(verbose)

        load_kwargs: dict[str, Any] = {
            "model_id_or_path": model_path,
            "model_type": model_type,
            "quant_method": quant_method,
            "dtype": dtype,
            "device_map": device_map,
            "local_files_only": _bool_value(local_files_only),
            "trust_remote_code": _bool_value(trust_remote_code),
            "attn_implementation": None if attn_implementation == "none" else attn_implementation,
            "processor_use_fast": None if processor_use_fast is None else _bool_value(processor_use_fast),
            "processor_min_pixels": self.min_pixels,
            "processor_max_pixels": self.max_pixels,
        }
        if quant_method == "masquant":
            load_kwargs.update(
                {
                    "masquant_root": masquant_root,
                    "masquant_resume": masquant_resume,
                    "masquant_act_scales": masquant_act_scales,
                    "masquant_wbits": int(masquant_wbits),
                    "masquant_abits": int(masquant_abits),
                    "masquant_group_size": int(masquant_group_size),
                    "masquant_inference_mode": masquant_inference_mode,
                    "masquant_symmetric": _bool_value(masquant_symmetric),
                    "masquant_batch_size": int(masquant_batch_size),
                    "masquant_cmc_low_rank_adapters": masquant_cmc_low_rank_adapters,
                    "masquant_cmc_white_matrix": masquant_cmc_white_matrix,
                    "masquant_cmc_rank": float(masquant_cmc_rank),
                    "masquant_cmc_quant_cmc": int(masquant_cmc_quant_cmc),
                }
            )
        self.model, self.processor = load_model_and_processor(
            **load_kwargs,
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

        if self.retention_ratio >= 1.0:
            return _generate_vanilla(self.model, self.processor, inputs, self.max_new_tokens)

        with torch.no_grad():
            answer = _generate_vanilla(self.model, self.processor, inputs, self.max_new_tokens)
        if self.gae_answer_source == "empty":
            answer = ""
        sample["answer"] = answer or "Yes"

        score_context = (
            _disable_masquant_fake_quant_for_scoring(self.model)
            if self.gae_score_disable_masquant_fake_quant
            else nullcontext()
        )
        try:
            with torch.enable_grad():
                with score_context:
                    scores = _score_gae_oracle(
                        model=self.model,
                        processor=self.processor,
                        adapter=self._pq_adapter,
                        pruner=self._pq_pruner,
                        sample=sample,
                        answer=sample["answer"],
                        per_token=self.gae_per_token,
                    )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if self.allow_vanilla_fallback:
                logging.warning("GAE scoring OOM; falling back to unpruned generation for this sample.")
                return _generate_vanilla(self.model, self.processor, inputs, self.max_new_tokens)
            raise
        scores = scores.detach()
        torch.cuda.empty_cache()
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
