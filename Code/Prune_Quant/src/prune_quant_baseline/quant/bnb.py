from typing import Any


def build_bnb_config(quant_method: str, dtype: Any) -> Any:
    """Build a bitsandbytes quantization config lazily."""

    from transformers import BitsAndBytesConfig

    if quant_method == "bnb4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if quant_method == "bnb8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported bitsandbytes quant_method: {quant_method}")
