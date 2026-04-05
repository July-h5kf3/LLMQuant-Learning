from transformers import BitsAndBytesConfig


def validate_quantization_config(quant_method, quant_kwargs=None):
    if quant_kwargs is None:
        quant_kwargs = {}

    supported_methods = {"none", "bnb_8bit", "bnb_4bit"}
    if quant_method not in supported_methods:
        raise ValueError(
            f"Unsupported quant_method: {quant_method}. "
            f"Supported values: {sorted(supported_methods)}"
        )

    if not isinstance(quant_kwargs, dict):
        raise ValueError("quant_kwargs must be a mapping.")


def build_bnb_8bit_kwargs():
    return {"load_in_8bit": True}


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


def build_quantization_kwargs(quant_method, torch_dtype, quant_kwargs=None):
    validate_quantization_config(quant_method, quant_kwargs)

    if quant_method == "none":
        return {}
    if quant_method == "bnb_8bit":
        return build_bnb_8bit_kwargs()
    if quant_method == "bnb_4bit":
        return build_bnb_4bit_kwargs(torch_dtype, quant_kwargs)

    raise ValueError(f"Unsupported quant_method: {quant_method}")
