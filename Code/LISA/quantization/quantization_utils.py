import os

import yaml
from transformers import AwqConfig, BitsAndBytesConfig


def validate_quantization_config(quant_method, quant_kwargs=None):
    if quant_kwargs is None:
        quant_kwargs = {}

    supported_methods = {"none", "bnb_8bit", "bnb_4bit", "awq"}
    if quant_method not in supported_methods:
        raise ValueError(
            f"Unsupported quant_method: {quant_method}. "
            f"Supported values: {sorted(supported_methods)}"
        )

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs must be a mapping.")


def load_quant_config(config_path, base_dir=None):
    if not config_path:
        return {}

    resolved_path = config_path
    if not os.path.isabs(resolved_path) and base_dir is not None:
        resolved_path = os.path.join(base_dir, resolved_path)

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("Quantization config YAML must be a mapping.")
    return config


def build_bnb_8bit_kwargs():
    return {
        "load_in_8bit": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
        ),
    }


def build_bnb_4bit_kwargs(torch_dtype, quant_kwargs=None):
    if quant_kwargs is None:
        quant_kwargs = {}

    quant_type = quant_kwargs.get("quant_type", "nf4")
    use_double_quant = quant_kwargs.get("use_double_quant", True)

    return {
        "load_in_4bit": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_type=quant_type,
        ),
    }


def build_awq_kwargs(quant_kwargs=None):
    if quant_kwargs is None:
        quant_kwargs = {}

    bits = quant_kwargs.get("bits", 4)
    group_size = quant_kwargs.get("group_size", 128)
    zero_point = quant_kwargs.get("zero_point", True)
    backend = quant_kwargs.get("backend", "auto")
    modules_to_not_convert = quant_kwargs.get("modules_to_not_convert")

    awq_config = AwqConfig(
        bits=bits,
        group_size=group_size,
        zero_point=zero_point,
        backend=backend,
        modules_to_not_convert=modules_to_not_convert,
    )

    extra_kwargs = {
        k: v
        for k, v in quant_kwargs.items()
        if k
        not in {
            "bits",
            "group_size",
            "zero_point",
            "backend",
            "modules_to_not_convert",
        }
    }

    return {
        "quantization_config": awq_config,
        **extra_kwargs,
    }


def build_quantization_kwargs(quant_method, torch_dtype, quant_kwargs=None):
    validate_quantization_config(quant_method, quant_kwargs)

    if quant_method == "none":
        return {}
    if quant_method == "bnb_8bit":
        return build_bnb_8bit_kwargs()
    if quant_method == "bnb_4bit":
        return build_bnb_4bit_kwargs(torch_dtype, quant_kwargs)
    if quant_method == "awq":
        return build_awq_kwargs(quant_kwargs)

    raise ValueError(f"Unsupported quant_method: {quant_method}")


def is_quantized_model(model):
    return bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "quantization_method", None) == "awq"
        or getattr(getattr(model, "config", None), "quantization_config", None)
        is not None
    )

