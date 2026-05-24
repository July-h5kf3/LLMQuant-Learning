import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any, *, strict: bool) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                if strict:
                    raise KeyError(f"Environment variable {name!r} is required by config.")
                return match.group(0)
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item, strict=strict) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item, strict=strict) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True,
    },
    "quant": {
        "method": "none",
        "ignore_modules": ["visual", "vision_tower", "multi_modal_projector"],
    },
    "pruning": {
        "method": "attention_proxy",
        "retention_ratio": 0.5,
        "min_keep": 1,
        "physical_delete": True,
    },
    "inference": {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "do_sample": False,
        "output_attentions": True,
        "use_cache": True,
    },
    "data": {},
}


def load_config(path: str | Path, *, strict_env: bool = False) -> dict[str, Any]:
    """Load YAML config, expand ${ENV_VAR}, apply defaults, and validate required fields."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    expanded = _expand_env(raw, strict=strict_env)
    config = _deep_merge(DEFAULT_CONFIG, expanded)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required first-stage config fields."""

    model = config.get("model") or {}
    missing = [key for key in ("model_type", "model_path") if not model.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required model config field(s): {joined}.")

    ratio = config.get("pruning", {}).get("retention_ratio")
    if not isinstance(ratio, (int, float)) or not (0 < float(ratio) <= 1):
        raise ValueError("pruning.retention_ratio must be in the range (0, 1].")
