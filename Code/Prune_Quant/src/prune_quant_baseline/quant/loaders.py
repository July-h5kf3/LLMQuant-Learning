from typing import Any


def _resolve_torch_dtype(dtype: str) -> Any:
    import torch

    if dtype == "auto":
        return "auto"

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of {sorted(mapping)}.")
    return mapping[dtype]


def _model_class_for_type(model_type: str) -> Any:
    from transformers import AutoModelForCausalLM

    if model_type == "qwen2_5_vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except ImportError:
            return AutoModelForCausalLM
    if model_type == "qwen2vl":
        try:
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        except ImportError:
            return AutoModelForCausalLM
    if model_type == "llava_onevision":
        try:
            from transformers import LlavaOnevisionForConditionalGeneration

            return LlavaOnevisionForConditionalGeneration
        except ImportError:
            return AutoModelForCausalLM
    raise ValueError("model_type must be one of: llava_onevision, qwen2vl, qwen2_5_vl.")


def load_model_and_processor(
    *,
    model_id_or_path: str,
    model_type: str,
    quant_method: str = "none",
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    attn_implementation: str | None = None,
    processor_use_fast: bool | None = None,
    processor_min_pixels: int | None = None,
    processor_max_pixels: int | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Load HF model and processor.

    This function is intended for the remote machine where model files already exist.
    local_files_only defaults to True to avoid accidental local downloads.
    """

    from transformers import AutoProcessor

    torch_dtype = _resolve_torch_dtype(dtype)
    model_cls = _model_class_for_type(model_type)
    quantization_config = None

    if quant_method == "masquant":
        from prune_quant_baseline.quant.masquant import load_masquant_model_and_processor

        return load_masquant_model_and_processor(
            masquant_root=kwargs.pop("masquant_root"),
            model_id_or_path=model_id_or_path,
            resume=kwargs.pop("masquant_resume", None),
            act_scales=kwargs.pop("masquant_act_scales", None),
            wbits=int(kwargs.pop("masquant_wbits", 4)),
            abits=int(kwargs.pop("masquant_abits", 8)),
            group_size=kwargs.pop("masquant_group_size", 0),
            symmetric=bool(kwargs.pop("masquant_symmetric", True)),
            inference_mode=kwargs.pop("masquant_inference_mode", "split_scales"),
            model_type=model_type,
            cmc_low_rank_adapters=kwargs.pop("masquant_cmc_low_rank_adapters", None),
            cmc_white_matrix=kwargs.pop("masquant_cmc_white_matrix", None),
            cmc_rank=float(kwargs.pop("masquant_cmc_rank", 0.2)),
            cmc_quant_cmc=int(kwargs.pop("masquant_cmc_quant_cmc", 0)),
            attn_implementation=attn_implementation or "eager",
            processor_use_fast=processor_use_fast,
            processor_min_pixels=processor_min_pixels,
            processor_max_pixels=processor_max_pixels,
            local_files_only=local_files_only,
            batch_size=int(kwargs.pop("masquant_batch_size", 1)),
        )
    if quant_method in ("bnb4", "bnb8"):
        from prune_quant_baseline.quant.bnb import build_bnb_config

        quantization_config = build_bnb_config(quant_method, torch_dtype)
    elif quant_method in ("gptq", "awq"):
        raise NotImplementedError(f"{quant_method} loading is a skeleton in the first-stage baseline.")
    elif quant_method != "none":
        raise ValueError("quant_method must be one of: none, bnb4, bnb8, gptq, awq, masquant.")

    common_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    processor_kwargs = dict(common_kwargs)
    if processor_use_fast is not None:
        processor_kwargs["use_fast"] = processor_use_fast
    if processor_min_pixels is not None:
        processor_kwargs["min_pixels"] = int(processor_min_pixels)
    if processor_max_pixels is not None:
        processor_kwargs["max_pixels"] = int(processor_max_pixels)
    try:
        processor = AutoProcessor.from_pretrained(model_id_or_path, **processor_kwargs)
        model_kwargs: dict[str, Any] = {
            **common_kwargs,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            **kwargs,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        model = model_cls.from_pretrained(model_id_or_path, **model_kwargs)
    except OSError as exc:
        raise OSError(
            "Failed to load model/processor. This baseline defaults to local_files_only=True; "
            "run this on the remote machine with existing model files or pass a valid local model path."
        ) from exc
    return model, processor
