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
    raise ValueError("model_type must be one of: llava_onevision, qwen2vl.")


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

    if quant_method in ("bnb4", "bnb8"):
        from prune_quant_baseline.quant.bnb import build_bnb_config

        quantization_config = build_bnb_config(quant_method, torch_dtype)
    elif quant_method in ("gptq", "awq"):
        raise NotImplementedError(f"{quant_method} loading is a skeleton in the first-stage baseline.")
    elif quant_method != "none":
        raise ValueError("quant_method must be one of: none, bnb4, bnb8, gptq, awq.")

    common_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    processor_kwargs = dict(common_kwargs)
    if processor_use_fast is not None:
        processor_kwargs["use_fast"] = processor_use_fast
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
