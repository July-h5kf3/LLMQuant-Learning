from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from types import MethodType
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from prune_quant_baseline.models.qwen2vl_hf import Qwen2VLHFAdapter
from prune_quant_baseline.pruners.gae_oracle import GAEOraclePruner, QuantJointGAEPruner
from prune_quant_baseline.quant.loaders import load_model_and_processor
from prune_quant_baseline.scripts.run_infer_pruned import (
    _build_pruned_generation_inputs,
    _generate_from_pruned_inputs,
    _generate_vanilla,
    _move_inputs_to_model_device,
    _score_gae_oracle,
    _score_gae_quant_joint,
    _visual_tokens_to_pixels,
)

LOGGER = logging.getLogger(__name__)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_pixels(*, pixels: Any, visual_tokens: Any, name: str) -> int | None:
    pixels = _optional_int(pixels)
    visual_tokens = _optional_int(visual_tokens)
    if pixels is not None and visual_tokens is not None:
        raise ValueError(f"Use either {name}_pixels or {name}_visual_tokens, not both.")
    if pixels is not None:
        return pixels
    return _visual_tokens_to_pixels(visual_tokens)


def _first_visual(visuals: Any) -> Any:
    if visuals is None:
        return None
    if isinstance(visuals, (list, tuple)):
        for item in visuals:
            if isinstance(item, (list, tuple)):
                nested = _first_visual(item)
                if nested is not None:
                    return nested
            elif item is not None:
                return item
        return None
    return visuals


# Qwen2-VL/2.5-VL patch factor = patch_size(14) * merge_size(2). smart_resize
# rejects any image with a side smaller than this, which OCRBench (e.g. 27px
# crops) trips. Upscale tiny images so both sides clear the factor.
_QWEN_VL_MIN_SIDE = 28


def _ensure_min_side(image: Image.Image, min_side: int = _QWEN_VL_MIN_SIDE) -> Image.Image:
    width, height = image.size
    if width >= min_side and height >= min_side:
        return image
    scale = max(min_side / max(width, 1), min_side / max(height, 1))
    new_width = max(min_side, int(round(width * scale)))
    new_height = max(min_side, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.BICUBIC)


def _as_rgb_image(visual: Any) -> Image.Image | None:
    if visual is None:
        return None
    if isinstance(visual, Image.Image):
        return _ensure_min_side(visual.convert("RGB"))
    if isinstance(visual, str):
        if visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            raise NotImplementedError("prune_quant_qwen2vl currently supports image-only lmms-eval tasks.")
        return _ensure_min_side(Image.open(visual).convert("RGB"))
    raise TypeError(f"Unsupported visual type for prune_quant_qwen2vl: {type(visual)!r}")


def _first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return ""
    return str(value).strip()


def _sample_answer_from_doc(doc: dict[str, Any]) -> str:
    if "answer" in doc and doc.get("choices") is not None and isinstance(doc["answer"], int):
        answer_idx = int(doc["answer"])
        if 0 <= answer_idx < len(doc["choices"]):
            return chr(ord("A") + answer_idx)
    for key in ("answer", "answers", "label", "target"):
        text = _first_text(doc.get(key))
        if text:
            return text
    return ""


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


@register_model("prune_quant_qwen2vl", "qwen2vl_pruned_gae")
class Qwen2VLPrunedGAE(lmms):
    """lmms-eval wrapper for Qwen2-VL/Qwen2.5-VL with optional GAE visual-token pruning."""

    def __init__(
        self,
        pretrained: str = "",
        model_path: str = "",
        model_type: str = "qwen2vl",
        quant_method: str = "none",
        dtype: str = "auto",
        device: str | None = None,
        device_map: str | None = "auto",
        batch_size: int | str = 1,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        min_visual_tokens: int | None = None,
        max_visual_tokens: int | None = None,
        max_new_tokens: int = 16,
        retention_ratio: float = 0.5,
        min_keep: int = 1,
        pruner: str = "gae_oracle",
        gae_answer_source: str = "generated",
        gae_per_token: bool = False,
        gae_quant_lambda: float = 0.5,
        gae_quant_method: str = "rtn",
        rtn_bits: int = 4,
        rtn_group_size: int = 0,
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
        system_prompt: str | None = None,
        gae_score_disable_masquant_fake_quant: bool = False,
        allow_vanilla_fallback: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            LOGGER.debug("Ignoring unused lmms-eval model args: %s", sorted(kwargs))
        model_path = model_path or pretrained
        if not model_path:
            raise ValueError("Set pretrained=... or model_path=... for prune_quant_qwen2vl.")
        if "omni" in model_path.lower():
            raise NotImplementedError("prune_quant_qwen2vl supports Qwen2-VL/Qwen2.5-VL image models only.")

        self.model_path = model_path
        self.model_type = model_type
        self.quant_method = quant_method
        self.min_pixels = _resolve_pixels(pixels=min_pixels, visual_tokens=min_visual_tokens, name="min")
        self.max_pixels = _resolve_pixels(pixels=max_pixels, visual_tokens=max_visual_tokens, name="max")
        self.max_new_tokens = int(max_new_tokens)
        self.retention_ratio = float(retention_ratio)
        self.min_keep = int(min_keep)
        self.pruner_name = pruner
        self.gae_answer_source = gae_answer_source
        self.gae_per_token = _bool_value(gae_per_token)
        self.gae_quant_lambda = float(gae_quant_lambda)
        self.gae_quant_method = gae_quant_method
        self.rtn_bits = int(rtn_bits)
        self.rtn_group_size = int(rtn_group_size)
        self.quant_bits = int(masquant_abits)
        self.quant_symmetric = _bool_value(masquant_symmetric)
        self.system_prompt = system_prompt
        self.gae_score_disable_masquant_fake_quant = _bool_value(gae_score_disable_masquant_fake_quant)
        self.allow_vanilla_fallback = _bool_value(allow_vanilla_fallback)
        self.verbose = _bool_value(verbose)
        self.batch_size_per_gpu = 1 if str(batch_size).startswith("auto") else int(batch_size)
        self.device_map = device_map or device or "auto"
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        load_kwargs: dict[str, Any] = {
            "model_id_or_path": model_path,
            "model_type": model_type,
            "quant_method": quant_method,
            "dtype": dtype,
            "device_map": self.device_map,
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
        self.model, self.processor = load_model_and_processor(**load_kwargs)
        self.model.eval()
        self._pq_adapter = Qwen2VLHFAdapter()
        if self.pruner_name == "gae_oracle":
            self._pq_pruner = GAEOraclePruner()
        elif self.pruner_name == "gae_quant_joint":
            self._pq_pruner = QuantJointGAEPruner(quant_lambda=self.gae_quant_lambda)
        else:
            raise ValueError("pruner must be one of: gae_oracle, gae_quant_joint.")

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self._tokenizer = tokenizer
        self._config = self.model.config
        self._rank = 0
        self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eot_token_id(self):
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def max_length(self):
        return 2048

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("prune_quant_qwen2vl implements generate_until image tasks only.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        return self.generate_until(requests)

    def _build_inputs(self, context: str, visuals: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        context = context.replace("<image>", "").strip()
        image = _as_rgb_image(_first_visual(visuals))
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": context})

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is None:
            inputs = self.processor(text=text, padding=True, return_tensors="pt")
        else:
            inputs = self.processor(text=text, images=image, padding=True, return_tensors="pt")
        sample = {
            "image": image,
            "prompt": context,
        }
        return sample, dict(inputs)

    def _generate_one(self, context: str, gen_kwargs: dict[str, Any], visuals: Any, doc: dict[str, Any]) -> str:
        sample, inputs = self._build_inputs(context, visuals)
        inputs = _move_inputs_to_model_device(self.model, inputs)
        max_new_tokens = int(gen_kwargs.get("max_new_tokens") or self.max_new_tokens)

        if self.retention_ratio >= 1.0:
            return _generate_vanilla(self.model, self.processor, inputs, max_new_tokens)
        if sample["image"] is None:
            return _generate_vanilla(self.model, self.processor, inputs, max_new_tokens)

        if self.gae_answer_source == "sample":
            answer = _sample_answer_from_doc(doc)
        elif self.gae_answer_source == "empty":
            answer = ""
        else:
            with torch.no_grad():
                answer = _generate_vanilla(self.model, self.processor, inputs, max_new_tokens)
        sample["answer"] = answer or "Yes"

        score_context = (
            _disable_masquant_fake_quant_for_scoring(self.model)
            if self.gae_score_disable_masquant_fake_quant
            else nullcontext()
        )
        try:
            with torch.enable_grad():
                with score_context:
                    if isinstance(self._pq_pruner, QuantJointGAEPruner):
                        scores = _score_gae_quant_joint(
                            model=self.model,
                            processor=self.processor,
                            adapter=self._pq_adapter,
                            pruner=self._pq_pruner,
                            sample=sample,
                            answer=sample["answer"],
                            per_token=self.gae_per_token,
                            quant_method=self.gae_quant_method,
                            rtn_bits=self.rtn_bits,
                            rtn_group_size=self.rtn_group_size,
                            quant_bits=self.quant_bits,
                            quant_symmetric=self.quant_symmetric,
                        )
                    else:
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
                LOGGER.warning("GAE scoring OOM; falling back to unpruned generation for this sample.")
                return _generate_vanilla(self.model, self.processor, inputs, max_new_tokens)
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
            score_mode="drop" if isinstance(self._pq_pruner, QuantJointGAEPruner) else "keep",
        )
        if "inputs_embeds" in pruned_inputs:
            response = _generate_from_pruned_inputs(
                model=self.model,
                processor=self.processor,
                pruned_inputs=pruned_inputs,
                max_new_tokens=max_new_tokens,
            )
        else:
            response = _generate_vanilla(self.model, self.processor, pruned_inputs, max_new_tokens)
        torch.cuda.empty_cache()
        return response

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses: list[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)
            response = self._generate_one(str(context), dict(gen_kwargs), visuals, doc)
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            for term in until:
                if term:
                    response = response.split(term)[0]
            responses.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            pbar.update(1)
        pbar.close()
        return responses
